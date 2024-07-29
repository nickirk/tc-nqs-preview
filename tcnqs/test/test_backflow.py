import jax.numpy as jnp
from jax import random
import pyscf
import pickle
from pyscf.tools import fcidump
import jax

from tcnqs.fcidump import read_2_spin_orbital_seprated as read2
from tcnqs.utils import generate_ci_data
import tcnqs.backflow as bf
from tcnqs.test.test_hamiltoian import test_hamiltonian_jit as test_hamiltonian
from tcnqs.hamiltonian_jit import HAMILTONIAN
from tcnqs.sampler.connected_dets import generate_connected_space


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

def test_backflow_unsupervised(mol, random_key, num_epochs=2400, test = False):
    if test:
        jax.config.update("jax_enable_x64", True)
    
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
    
    # num_epochs = 400
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
    if test:
        assert jnp.absolute(train_losses_bf[-1]-FCI_e_pyscf) < 1e-3
        print("Success: Model trained successfully")
    
    return train_losses_bf

def test_backflow_connected(mol, random_key , num_epochs=2400, test=False):
    if test:
        jax.config.update("jax_enable_x64", True)
    
    rng = random.PRNGKey(random_key)
    
    myhf = mol.RHF().run()
    cisolver = pyscf.fci.FCI(myhf)
    num_orbitals = myhf.mo_coeff.shape[1]
    num_alpha_electrons, num_beta_electrons = mol.nelec
    FCI_e_pyscf, ci_vector=cisolver.kernel()
    cisolver = pyscf.fci.FCI(myhf)
    
    num_alpha_electrons, num_beta_electrons = mol.nelec
    
    fcidump_file = 'tcnqs/test/dataset_fcidump/fcidump'
    fcidump.from_scf(myhf, fcidump_file)
    n_sites, n_elec, ecore, h1e_s, g2e_s = read2(fcidump_file)
    h1e_s = jnp.asarray(h1e_s)
    g2e_s = jnp.asarray(g2e_s)
    
    # Create FCI Hamiltonian
    hamiltonian = HAMILTONIAN(n_elec, 2*n_sites, h1e_s, g2e_s)
    
    x_train, y_train = generate_ci_data(num_orbitals,num_alpha_electrons,num_beta_electrons,ci_vector)
    x_train = jnp.asarray(x_train,dtype=jnp.uint8)[:30]
    # hamiltonian , ecore = test_hamiltonian(mol)
    num_orbitals = 2*myhf.mo_coeff.shape[1]
    
    n_core = 20
    # hfs = jnp.concatenate((jnp.ones(num_alpha_electrons), jnp.zeros(int(num_orbitals/2) - num_alpha_electrons)))  
    # hf = jnp.concatenate((hfs, hfs),dtype=jnp.uint8)
    # x_train =  generate_connected_space(hf, num_orbitals, num_alpha_electrons+num_beta_electrons)[:n_core]
    
    # num_samples = len(y_train) 

    model_bf, variables_bf = bf.create_model(rng, input_shape = num_orbitals, 
                                            num_electrons= num_alpha_electrons + num_beta_electrons
                                ,hidden_layer_sizes=[],activation='tanh')
    state_bf = bf.create_train_state(rng, model_bf, variables_bf)
    
    # num_epochs = 400
    #batch_size = num_samples
    train_losses_bf = []
    #nwf = jax.jit(bf.train_step_connections)
    
    for epoch in range(num_epochs):
        epoch_loss_bf = 0.0
            
        state_bf, loss_bf = bf.train_step_connections(state_bf, x_train, hamiltonian)
        epoch_loss_bf += loss_bf
        
        average_epoch_loss_bf = epoch_loss_bf #/ num_samples // num_samples
        train_losses_bf.append(average_epoch_loss_bf + ecore)
        
        print(f"Epoch {epoch+1} , Loss_bf: {average_epoch_loss_bf+ ecore}")
    
    # print(FCI_e_pyscf, jnp.average(jnp.array(train_losses_bf[-50:],dtype=jnp.float32)))
    if test:
        assert jnp.absolute(train_losses_bf[-1]-FCI_e_pyscf) < 1e-3
        print("Success: Model trained successfully")
    
    return train_losses_bf, FCI_e_pyscf

if __name__ == '__main__':
    mol = pyscf.M(
    atom = 'H 0 0 0; H 0 0 1.0; H 0 0 3.0; H 0 0 4.0 ' , # H 0 0 3.0; H 0 0 4.0  
    basis = 'sto-3g',
    spin = 0
    )

    # test_backflow_supervised(mol, 0)
    #test_backflow_unsupervised(mol,17, )#test=True)
    test_backflow_connected(mol, 17, num_epochs=6000 )#test= True)