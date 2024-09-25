import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
# os.environ['XLA_FLAGS'] = '--xla_gpu_enable_tracing'
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pyscf
import jax
from scipy.special import comb
import time

from tcnqs.fcidump import read_2_spin_orbital_seprated as read2
from tcnqs.utils import generate_ci_data, build_ham_from_pyscf
import tcnqs.backflow as bf
import tcnqs.trainer as trainer
from tcnqs.sampler.fssc import FSSC
import tcnqs.test.test_parameters as t

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
    state = trainer.create_train_state(rng, model, variables)
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
            state, loss = trainer.train_step_log(state, batch)
            epoch_loss += loss
        
        average_epoch_loss = epoch_loss / (num_samples // batch_size)
        train_losses.append(average_epoch_loss)
        rng = subrng
        print(f"Epoch {epoch+1}, Loss: {average_epoch_loss}" )
    
    
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
    fci_e_pyscf, ci_vector=cisolver.kernel()
    cisolver = pyscf.fci.FCI(myhf)
    
    num_alpha_electrons, num_beta_electrons = mol.nelec
    
    x_train, y_train = generate_ci_data(num_orbitals,num_alpha_electrons,num_beta_electrons,ci_vector)
    
    hamiltonian = build_ham_from_pyscf(mol, myhf)

    # build H matrix 
    H = np.zeros((len(x_train), len(x_train)))
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            H[i,j] = hamiltonian(x_train[i], x_train[j])
    # diagolize H matrix
    fci_e_diagonal = np.sort(np.linalg.eig(H)[0])[0] + hamiltonian.e_core
    print(fci_e_diagonal, fci_e_pyscf)


    
    num_orbitals = 2*myhf.mo_coeff.shape[1]
    num_samples = len(y_train) 

    model_bf, variables_bf = bf.create_model(rng, input_shape = num_orbitals,
                                num_electrons=num_alpha_electrons+num_beta_electrons
                                ,hidden_layer_sizes=[4],activation='tanh')
    state_bf = trainer.create_train_state(rng, model_bf, variables_bf)
    
    # num_epochs = 400
    batch_size = num_samples
    train_losses_bf = []

    for epoch in range(num_epochs):
        epoch_loss_bf = 0.0
        for i in range(0, num_samples, batch_size):
            batch = (x_train[i:i+batch_size], y_train[i:i+batch_size])
            
            state_bf, loss_bf = trainer.train_step_hamiltonian(state_bf, batch, H)
            epoch_loss_bf += loss_bf
        
        average_epoch_loss_bf = epoch_loss_bf / (num_samples // batch_size)
        train_losses_bf.append(average_epoch_loss_bf + hamiltonian.e_core)
        
        print(f"Epoch {epoch+1} , Loss_bf: {average_epoch_loss_bf+ hamiltonian.e_core}")
    
    # print(FCI_e_pyscf, jnp.average(jnp.array(train_losses_bf[-50:],dtype=jnp.float32)))
    if test:
        assert jnp.absolute(train_losses_bf[-1]-fci_e_pyscf) < 1e-3
        print("Success: Model trained successfully")
    
    return train_losses_bf

def test_backflow_connected(mol, random_key , num_epochs=2400, test=False):
    if test:
        jax.config.update("jax_enable_x64", True)
    
    rng = random.PRNGKey(random_key)
    
    myhf = mol.RHF().run()
    cisolver = pyscf.fci.FCI(myhf)
    fci_e_pyscf, ci_vector=cisolver.kernel()
    cisolver = pyscf.fci.FCI(myhf)
    print("E FCI = ", fci_e_pyscf)
    
    
    #fcidump_file = 'tcnqs/test/dataset_fcidump/fcidump'
    # Create FCI Hamiltonian
    hamiltonian = build_ham_from_pyscf(mol, myhf)
    
    x_train, y_train = generate_ci_data(hamiltonian.n_orb//2, hamiltonian.n_elec_a, 
                                        hamiltonian.n_elec_b, ci_vector)
    x_train = jnp.asarray(x_train,dtype=jnp.uint8)
    # hamiltonian , ecore = test_hamiltonian(mol)

    
    num_orbitals = hamiltonian.n_orb
    num_samples = len(y_train) 

    model_bf, variables_bf = bf.create_model(rng, input_shape = num_orbitals, 
                                            num_electrons= hamiltonian.n_elec
                                ,hidden_layer_sizes=[], activation='tanh')
    state_bf = trainer.create_train_state(rng, model_bf, variables_bf)
    
    # num_epochs = 400
    #batch_size = num_samples
    train_losses_bf = []
    #nwf = jax.jit(trainer.train_step_connections)
    
    for epoch in range(num_epochs):
        epoch_loss_bf = 0.0
        #len_random  = np.random.randint(10, len(x_train))
        #x_train_tmp = x_train[:len_random]
            
        state_bf, loss_bf = trainer.train_step_connections(state_bf, x_train, hamiltonian)
        epoch_loss_bf += loss_bf
        
        average_epoch_loss_bf = epoch_loss_bf # / (num_samples // batch_size)
        train_losses_bf.append(average_epoch_loss_bf + hamiltonian.e_core)
        
        print(f"Epoch {epoch+1} , Loss_bf: {average_epoch_loss_bf + hamiltonian.e_core}")
    
    # print(FCI_e_pyscf, jnp.average(jnp.array(train_losses_bf[-50:],dtype=jnp.float32)))
    if test:
        assert jnp.absolute(train_losses_bf[-1]-fci_e_pyscf) < 1e-3
        print("Success: Model trained successfully")
    
    return train_losses_bf, fci_e_pyscf

def test_backflow_fssc(mol,n_core,num_epochs=2400, test=False ,random_key=17 ):
    if test:
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_debug_nans", True)
    rng = random.PRNGKey(random_key)
    myhf = mol.RHF().run()
    cisolver = pyscf.fci.FCI(myhf)
    fci_e_pyscf, ci_vector=cisolver.kernel()
    cisolver = pyscf.fci.FCI(myhf)
    print("E FCI = ", fci_e_pyscf)
 
    hamiltonian = build_ham_from_pyscf(mol, myhf)
    
    
    num_orbitals = hamiltonian.n_orb


    model_bf, variables_bf = bf.create_model(rng, input_shape = num_orbitals, 
                                            num_electrons= hamiltonian.n_elec,
                                            hidden_layer_sizes=t.hidden_layer_sizes, activation='tanh',n_bf_dets=t.n_bf_dets)
    state_bf =trainer.create_train_state(rng, model_bf, variables_bf)
    
    train_losses_bf = []

    n_s_orb = (hamiltonian.n_orb//2)
    n_total_dets = comb(n_s_orb, hamiltonian.n_elec_a,exact=True)*comb(n_s_orb, hamiltonian.n_elec_b ,exact=True)
    
    if n_core > n_total_dets:
        n_core = n_total_dets
        print(f"Warning: n_core specified is greater than total determinants in hilbert space. Falling back to n_core ={n_total_dets}")

    max_n_full= n_core*(1 + comb(hamiltonian.n_elec_a,2, exact=True)*
                        comb(n_s_orb-hamiltonian.n_elec_a,2,exact=True)+comb(hamiltonian.n_elec_b,2, 
                        exact=True)*comb(n_s_orb-hamiltonian.n_elec_b,2,exact=True)
                        + hamiltonian.n_elec_a*hamiltonian.n_elec_b*(n_s_orb-hamiltonian.n_elec_a)
                        *(n_s_orb-hamiltonian.n_elec_b) + n_s_orb*(hamiltonian.n_elec_a+hamiltonian.n_elec_b)
                        - hamiltonian.n_elec_a**2 - hamiltonian.n_elec_b**2) 
    
    n_connected =  jnp.minimum(n_total_dets, max_n_full) - n_core
    
    sampler = FSSC(n_core, n_connected ,hamiltonian.n_elec_a, hamiltonian.n_elec_b, num_orbitals,state_bf.apply_fn)
    sample = sampler.initialize()
    stored = sampler.next_sample_stored(sample,hamiltonian)
    flag = True
    #stored = trainer.sample_ham_wrap(sample,hamiltonian,sampler)
    #sample = sample[0][:sampler.n_core]
    for epoch in range(num_epochs):
        
        epoch_loss_bf = 0.0
        #a=time.time()
        state_bf, loss_bf, sample, flag, stored = trainer.train_step_fssc(state_bf, sample, flag, stored, hamiltonian,sampler)
        # relevant_indices = jnp.where(jnp.logical_not(jnp.all(sample[0]==jnp.zeros(num_orbitals),axis=1)))[0]
        # sample =(sample[0][relevant_indices],sample[1][relevant_indices]) 
        #b=time.time()
        #epoch_loss_bf += loss_bf
        #average_epoch_loss_bf = epoch_loss_bf # / (num_samples // batch_size)
        train_losses_bf.append(loss_bf + hamiltonian.e_core)
        
        print(f"Epoch {epoch+1} , Loss_bf: {loss_bf + hamiltonian.e_core},{flag}")
        #jax.profiler.stop_trace()
   
    if test:
        assert jnp.absolute(train_losses_bf[-1]-fci_e_pyscf) < 5e-3
        print("Success: Model trained successfully")
    
    return train_losses_bf, fci_e_pyscf

if __name__ == '__main__':
    mol = t.mol
    print(jax.devices())
    #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    # test_backflow_supervised(mol, 0)
    #test_backflow_unsupervised(mol,17, test=True)
    #test_backflow_connected(mol, 17, )#test= True)
    start = time.time()
    #jax.profiler.start_trace("tmp/jax-trace",create_perfetto_link=True)
    test_backflow_fssc(mol,n_core=t.n_core, test= True, random_key=17, num_epochs=t.num_epochs)
    #jax.profiler.stop_trace()
    end = time.time()
    print("Time taken: ", end-start)