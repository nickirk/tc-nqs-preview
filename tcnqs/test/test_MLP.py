import jax.numpy as jnp
from jax import random
import pyscf
from pyscf import fci
import pickle

from tcnqs.utils import generate_ci_data
import tcnqs.mlp as mlp
from tcnqs.test.test_hamiltoian import test_hamiltonian_jit as test_hamiltonian



def test_mlp_supervised(mol,random_key):
    myhf = mol.RHF().run()
    cisolver = fci.FCI(myhf)
    rng = random.PRNGKey(random_key)
    
    num_orbitals = myhf.mo_coeff.shape[1]
    num_alpha_electrons, num_beta_electrons = mol.nelec
    cie,ci_vector=cisolver.kernel()
    
    x_train, y_train = generate_ci_data(num_orbitals,num_alpha_electrons,num_beta_electrons,ci_vector)
    input_size = 2*num_orbitals # Example input size
    num_samples = len(y_train) # Number of training samples
    
    model, variables = mlp.create_model(rng, input_size, 
                                        hidden_layer_sizes=[4], activation='relu')
    state = mlp.create_train_state(rng, model, variables)
    
    # with open('params_mlp.pkl', 'rb') as f:
    #     parameters = pickle.load(f)
    # parameters = {'params': parameters}
    
    # Training loop
    num_epochs = 50
    batch_size = num_samples
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        rng, subrng = random.split(rng)
        perm = random.permutation(rng, num_samples)
        x_train = x_train[perm]
        y_train = y_train[perm]
        for i in range(0, num_samples, batch_size):
            batch = (x_train[i:i+batch_size], y_train[i:i+batch_size])
            state, loss = mlp.train_step_mse(state, batch)
            epoch_loss += loss
        
        average_epoch_loss = epoch_loss / (num_samples // batch_size)
        train_losses.append(average_epoch_loss)
        rng = subrng
        print(f"Epoch {epoch+1}, Loss: {average_epoch_loss}")
    
    #with open('params_mlp.pkl', 'wb') as f:
    #    pickle.dump(state.params, f)
    assert train_losses[-1] < 1e-3
    print("Success")
    
    return train_losses
    
def test_mlp_unsupervised(mol,random_key):
    rng = random.PRNGKey(random_key)
    
    myhf = mol.RHF().run()
    cisolver = pyscf.fci.FCI(myhf)
    num_orbitals = myhf.mo_coeff.shape[1]
    num_alpha_electrons, num_beta_electrons = mol.nelec
    FCI_e_pyscf, ci_vector=cisolver.kernel()
    cisolver = pyscf.fci.FCI(myhf)
    
    num_alpha_electrons, num_beta_electrons = mol.nelec
    
    x_train, y_train = generate_ci_data(num_orbitals,num_alpha_electrons,num_beta_electrons,ci_vector)
    
    hamiltonian , ecore = test_hamiltonian(mol)

    
    num_orbitals = 2*myhf.mo_coeff.shape[1]
    num_samples = len(y_train) 

    model_mlp, variables_mlp = mlp.create_model(rng, input_shape = num_orbitals ,hidden_layer_sizes=[4,28],activation='tanh')
    state_mlp = mlp.create_train_state(rng, model_mlp, variables_mlp)
    
    num_epochs = 4000
    batch_size = num_samples
    train_losses_mlp = []

    for epoch in range(num_epochs):
        epoch_loss_mlp = 0.0
        for i in range(0, num_samples, batch_size):
            batch = (x_train[i:i+batch_size], y_train[i:i+batch_size])
            
            state_mlp, loss_mlp = mlp.train_step_hamiltonian(state_mlp, batch, hamiltonian)
            epoch_loss_mlp += loss_mlp
        
        average_epoch_loss_mlp = epoch_loss_mlp / (num_samples // batch_size)
        train_losses_mlp.append(average_epoch_loss_mlp + ecore)
        
        print(f"Epoch {epoch+1} , Loss_mlp: {average_epoch_loss_mlp+ ecore}")
    
    # print(FCI_e_pyscf, jnp.average(jnp.array(train_losses_mlp[-50:],dtype=jnp.float32)))
    assert jnp.absolute(train_losses_mlp[-1]-FCI_e_pyscf) < 1e-3
    print("Success: Model trained successfully")
    
    return train_losses_mlp
    
if __name__ == '__main__':
    mol = pyscf.M(
    atom = 'H 0 0 0; H 0 0 1.0 ;H 0 0 2; H 0 0 3.0',  
    basis = 'sto-3g',
    spin = 0
    )

    test_mlp_unsupervised(mol, random_key=17)
    # test_mlp_supervised(mol, random_key=0)