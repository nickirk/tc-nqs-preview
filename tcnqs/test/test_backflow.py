import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#os.environ['XLA_FLAGS'] = '--xla_gpu_enable_tracing'
#os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pyscf
import jax
from scipy.special import comb
import time


from tcnqs.utils import generate_ci_data, build_ham_from_pyscf
import tcnqs.backflow as bf
import tcnqs.trainer as trainer
from tcnqs.sampler.fssc import FSSC
import tcnqs.test.test_parameters as t
import matplotlib.pyplot as plt

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
        train_losses_bf.append(average_epoch_loss_bf )
        
        print(f"Epoch {epoch+1} , Loss_bf: {average_epoch_loss_bf}")
    
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
        train_losses_bf.append(average_epoch_loss_bf )
        
        print(f"Epoch {epoch+1} , Loss_bf: {average_epoch_loss_bf }")
    
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
    
    sampler = FSSC(n_core, int(n_connected) ,hamiltonian.n_elec_a, hamiltonian.n_elec_b, num_orbitals,n_batch=n_core)
    sampler = sampler.initialize()
    stored = sampler.next_sample_stored(hamiltonian)
    flag = True
    #stored = trainer.sample_ham_wrap(sample,hamiltonian,sampler)
    #sample = sample[0][:sampler.n_core]
    for epoch in range(num_epochs):
        
        #epoch_loss_bf = 0.0
        #a=time.time()
        state_bf, loss_bf, sampler, flag, stored = trainer.train_step_fssc(state_bf, hamiltonian, sampler, flag, stored)
        # relevant_indices = jnp.where(jnp.logical_not(jnp.all(sample[0]==jnp.zeros(num_orbitals),axis=1)))[0]
        # sample =(sample[0][relevant_indices],sample[1][relevant_indices]) 
        #b=time.time()
        #epoch_loss_bf += loss_bf
        #average_epoch_loss_bf = epoch_loss_bf # / (num_samples // batch_size)
        train_losses_bf.append(loss_bf )
        
        print(f"Epoch {epoch+1} , Loss_bf: {loss_bf },{flag}")
        #jax.profiler.stop_trace()
   
    # if test:
    #     assert jnp.absolute(train_losses_bf[-1]-fci_e_pyscf) < 5e-3
    #     print("Success: Model trained successfully")
    
    return train_losses_bf, fci_e_pyscf

def test_backflow_batched(mol,n_core,num_epochs=2400, test=False ,random_key=17 ):
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
    variables_bf = jax.tree.map(lambda x: x.astype(jnp.float64), variables_bf)
    state_bf = trainer.create_train_state(rng, model_bf, variables_bf)
    
    train_losses_bf = []

    n_s_orb = (hamiltonian.n_orb//2)
    n_total_dets = comb(n_s_orb, hamiltonian.n_elec_a,exact=True)*comb(n_s_orb, hamiltonian.n_elec_b ,exact=True)
    batch_size = t.batch_size
    
    if n_core > n_total_dets:
        n_core = n_total_dets
        print(f"Warning: n_core specified is greater than total determinants in hilbert space. Falling back to n_core ={n_total_dets}")
    if n_core < batch_size:
        batch_size =n_core
        print(f"Warning: n_core specified is less than batch_size. Falling back to batch_size ={batch_size}")
    if n_core % batch_size != 0:
        n_core = n_core - n_core % t.batch_size
        print(f"Warning: n_core specified is not a multiple of batch_size. Falling back to n_core ={n_core}")

    n_connections= (1 + comb(hamiltonian.n_elec_a,2, exact=True)*
                        comb(n_s_orb-hamiltonian.n_elec_a,2,exact=True)+comb(hamiltonian.n_elec_b,2, 
                        exact=True)*comb(n_s_orb-hamiltonian.n_elec_b,2,exact=True)
                        + hamiltonian.n_elec_a*hamiltonian.n_elec_b*(n_s_orb-hamiltonian.n_elec_a)
                        *(n_s_orb-hamiltonian.n_elec_b) + n_s_orb*(hamiltonian.n_elec_a+hamiltonian.n_elec_b)
                        - hamiltonian.n_elec_a**2 - hamiltonian.n_elec_b**2) 
    
    max_n_full= n_core*n_connections
    n_connected = 0 #30000 #jnp.minimum(n_total_dets, max_n_full) - n_core
    n_full = n_total_dets
    #unique fn min of (n_total_dets, n_batch*n_connections) 
    sampler = FSSC(n_core, int(n_connected) ,hamiltonian.n_elec_a, hamiltonian.n_elec_b, num_orbitals, n_batch=batch_size)
    sampler = sampler.initialize()
   
    #stored = trainer.sample_ham_wrap(sample,hamiltonian,sampler)
    #sample = sample[0][:sampler.n_core]
    for epoch in range(num_epochs):
        
        #epoch_loss_bf = 0.0
        #a=time.time()
        state_bf, loss_bf, sampler = trainer.train_step_batched(state_bf, hamiltonian, sampler)
        # relevant_indices = jnp.where(jnp.logical_not(jnp.all(sample[0]==jnp.zeros(num_orbitals),axis=1)))[0]
        # sample =(sample[0][relevant_indices],sample[1][relevant_indices]) 
        #b=time.time()
        #epoch_loss_bf += loss_bf
        #average_epoch_loss_bf = epoch_loss_bf # / (num_samples // batch_size)
        train_losses_bf.append(loss_bf)
        
        print(f"Epoch {epoch+1} , Loss_bf: {loss_bf }")
        #jax.profiler.stop_trace()
   
    # if test:
    #     assert jnp.absolute(train_losses_bf[-1]-fci_e_pyscf) < 5e-3
    #     print("Success: Model trained successfully")
    
    return train_losses_bf, fci_e_pyscf

def test_electron_backflow(mol,n_core,num_epochs=2400, test=False ,random_key=17 ):
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


    model_bf, variables_bf = bf.create_model_electron_bf(rng, input_shape = num_orbitals, 
                                            num_alpha_electron= hamiltonian.n_elec_a,
                                            num_beta_electron= hamiltonian.n_elec_b,
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
    
    sampler = FSSC(n_core, int(n_connected) ,hamiltonian.n_elec_a, hamiltonian.n_elec_b, num_orbitals,n_batch=n_core)
    sampler = sampler.initialize()
    stored = sampler.next_sample_stored(hamiltonian)
    flag = True
    #stored = trainer.sample_ham_wrap(sample,hamiltonian,sampler)
    #sample = sample[0][:sampler.n_core]
    for epoch in range(num_epochs):
        
        #epoch_loss_bf = 0.0
        #a=time.time()
        state_bf, loss_bf, sampler, flag, stored = trainer.train_step_fssc(state_bf, hamiltonian, sampler, flag, stored)
        # relevant_indices = jnp.where(jnp.logical_not(jnp.all(sample[0]==jnp.zeros(num_orbitals),axis=1)))[0]
        # sample =(sample[0][relevant_indices],sample[1][relevant_indices]) 
        #b=time.time()
        #epoch_loss_bf += loss_bf
        #average_epoch_loss_bf = epoch_loss_bf # / (num_samples // batch_size)
        train_losses_bf.append(loss_bf )
        
        print(f"Epoch {epoch+1} , Loss_bf: {loss_bf },{flag}")
        #jax.profiler.stop_trace()
   
    if test:
        assert jnp.absolute(train_losses_bf[-1]-fci_e_pyscf) < 5e-3
        print("Success: Model trained successfully")
    
    return train_losses_bf, fci_e_pyscf

def test_backflow_vite(mol,n_core,num_epochs=2400, test=False ,random_key=17 ):
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
    variables_bf = jax.tree.map(lambda x: x.astype(jnp.float64), variables_bf)
    state_bf = trainer.create_train_state_VITE(rng, model_bf, variables_bf)
    
    train_losses_bf = []

    n_s_orb = (hamiltonian.n_orb//2)
    n_total_dets = comb(n_s_orb, hamiltonian.n_elec_a,exact=True)*comb(n_s_orb, hamiltonian.n_elec_b ,exact=True)
    batch_size = t.batch_size
    
    if n_core > n_total_dets:
        n_core = n_total_dets
        print(f"Warning: n_core specified is greater than total determinants in hilbert space. Falling back to n_core ={n_total_dets}")
    if n_core < batch_size:
        batch_size =n_core
        print(f"Warning: n_core specified is less than batch_size. Falling back to batch_size ={batch_size}")
    if n_core % batch_size != 0:
        n_core = n_core - n_core % t.batch_size
        print(f"Warning: n_core specified is not a multiple of batch_size. Falling back to n_core ={n_core}")

    n_connections= (1 + comb(hamiltonian.n_elec_a,2, exact=True)*
                        comb(n_s_orb-hamiltonian.n_elec_a,2,exact=True)+comb(hamiltonian.n_elec_b,2, 
                        exact=True)*comb(n_s_orb-hamiltonian.n_elec_b,2,exact=True)
                        + hamiltonian.n_elec_a*hamiltonian.n_elec_b*(n_s_orb-hamiltonian.n_elec_a)
                        *(n_s_orb-hamiltonian.n_elec_b) + n_s_orb*(hamiltonian.n_elec_a+hamiltonian.n_elec_b)
                        - hamiltonian.n_elec_a**2 - hamiltonian.n_elec_b**2) 
    
    max_n_full= n_core*n_connections
    n_connected =  jnp.minimum(n_total_dets, max_n_full) - n_core
    n_full = n_total_dets
    #unique fn min of (n_total_dets, n_batch*n_connections) 
    sampler = FSSC(n_core, int(n_connected) ,hamiltonian.n_elec_a, hamiltonian.n_elec_b, num_orbitals, n_batch=batch_size)
    sampler = sampler.initialize()
 
    for epoch in range(num_epochs):
        
      
        state_bf, loss_bf, sampler = trainer.train_step_VITE(state_bf, hamiltonian, sampler)
        
        train_losses_bf.append(loss_bf)
        
        print(f"Epoch {epoch+1} , Loss_bf: {loss_bf }")
   
   
    # if test:
    #     assert jnp.absolute(train_losses_bf[-1]-fci_e_pyscf) < 5e-3
    #     print("Success: Model trained successfully")
    
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
    #test_backflow_unsupervised(mol, random_key=15,test= True, num_epochs=t.num_epochs)
    #test_backflow_fssc(mol,n_core=t.n_core, test= True, random_key=15, num_epochs=t.num_epochs)
    train_losses_vite, fci_e_pyscf=test_backflow_vite(mol,n_core=t.n_core, test= True, random_key=15, num_epochs=t.num_epochs)
    # jnp.save("tcnqs/LiH_vite.npy",train_losses_vite)
    # jnp.save("tcnqs/LiH_fci.npy",fci_e_pyscf)


    # train_losses_fssc, fci_e_pyscf=test_backflow_fssc(mol,n_core=t.n_core, test= True, random_key=15, num_epochs=t.num_epochs)
    
    # plt.figure(figsize=(8, 6))
    
    # # Plot final energies from the simulation for current bond length
    # plt.plot(jnp.arange(len(train_losses_fssc)), train_losses_fssc - fci_e_pyscf, label='Reighleigh Ritz')
    
    # plt.plot(jnp.arange(len(train_losses_vite)), train_losses_vite - fci_e_pyscf, label='VITE')
    
    # # Plot HF and CCSD energies as horizontal lines
    # #plt.axhline(y=jnp.asarray(hf_energies[i]) - fci_energies[i], color='g', label='HF')
    # #plt.axhline(y=jnp.asarray(ccsd_energies[i]) - fci_energies[i], color='r', label='CCSD')
    
    # # Add labels and legend
    # plt.xlabel('Epochs')
    # plt.ylabel('Energy difference (E-FCI)')
    # plt.title(f'Energy Convergence for LiH Molecule')
    # plt.legend()
    # plt.yscale('log')
    
    # # Add grid
    # plt.grid(True)
    # #plt.ylim(-0.001, 0.01)
    # # Save plot with unique filename for each bond length
    # plt.savefig(f"tcnqs/test.png")
    


    #test_electron_backflow(mol,n_core=t.n_core, test= True, random_key=15, num_epochs=t.num_epochs)
    #jax.profiler.stop_trace()
    end = time.time()
    print("Time taken: ", end-start)
