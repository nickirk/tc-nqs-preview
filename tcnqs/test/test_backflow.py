import jax.numpy as jnp
from jax import random
import pyscf
import pickle

from tcnqs.utils import generate_ci_data
import tcnqs.backflow as bf
from tcnqs.test.test_hamiltoian import test_hamiltonian_jit as test_hamiltonian



def test_backflow_supervised(mol, random_key):
    myhf = mol.RHF().run()
    cisolver = pyscf.fci.FCI(myhf)
    rng = random.PRNGKey(random_key)
    
    num_orbitals = myhf.mo_coeff.shape[1]
    num_alpha_electrons, num_beta_electrons = mol.nelec
    cie, ci_vector=cisolver.kernel()
    
    x_train, y_train = generate_ci_data(num_orbitals, num_alpha_electrons, num_beta_electrons, ci_vector)
    input_size = 2*num_orbitals 
    num_samples = len(y_train) 
    
    model, variables = bf.create_model(rng, input_size,
                                       num_electrons=num_alpha_electrons+num_beta_electrons,
                                       hidden_layer_sizes=[4,4], activation='tanh')
    state = bf.create_train_state(rng, model, variables)
    #print(variables)
    # Training loop
    num_epochs = 50
    batch_size = num_samples
    train_losses = []


    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # randomize the order of the training data
        rng, subrng = random.split(rng)
        perm = random.permutation(rng, num_samples)
        x_train = x_train[perm]
        y_train = y_train[perm]

        for i in range(0, num_samples, batch_size):
            batch = (x_train[i:i+batch_size], y_train[i:i+batch_size])
            state, loss = bf.train_step_log(state, batch)
            epoch_loss += loss
        
        average_epoch_loss = epoch_loss / (num_samples // batch_size)
        train_losses.append(average_epoch_loss)
        rng = subrng
        print(f"Epoch {epoch+1}, Loss: {average_epoch_loss}")
    
    
    # with open('params.pkl', 'wb') as f:
    #     pickle.dump(state.params, f)

    # print("Parameters saved")
    
    assert train_losses[-1] < 1e-3
    print("Success")
    
    return train_losses

def test_backflow_unsupervised(mol,random_key):
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

    model_bf, variables_bf = bf.create_model(rng, input_shape = num_orbitals,
                                num_electrons=num_alpha_electrons+num_beta_electrons
                                ,hidden_layer_sizes=[4],activation='tanh')
    state_bf = bf.create_train_state(rng, model_bf, variables_bf)
    
    num_epochs = 2400
    batch_size = num_samples
    train_losses_bf = []

    for epoch in range(num_epochs):
        epoch_loss_bf = 0.0
        for i in range(0, num_samples, batch_size):
            batch = (x_train[i:i+batch_size], y_train[i:i+batch_size])
            
            state_bf, loss_bf = bf.train_step_hamiltonian(state_bf, batch, hamiltonian)
            epoch_loss_bf += loss_bf
        
        average_epoch_loss_bf = epoch_loss_bf / (num_samples // batch_size)
        train_losses_bf.append(average_epoch_loss_bf + ecore)
        
        print(f"Epoch {epoch+1} , Loss_bf: {average_epoch_loss_bf+ ecore}")
    
    # print(FCI_e_pyscf, jnp.average(jnp.array(train_losses_bf[-50:],dtype=jnp.float32)))
    assert jnp.absolute(train_losses_bf[-1]-FCI_e_pyscf) < 1e-3
    print("Success: Model trained successfully")
    
    return train_losses_bf

if __name__ == '__main__':
    mol = pyscf.M(
    atom = 'H 0 0 0; H 0 0 1.0 ;H 0 0 2; H 0 0 3.0',  
    basis = 'sto-3g',
    spin = 0
    )

    # test_backflow_supervised(mol, 0)
    test_backflow_unsupervised(mol, 17)