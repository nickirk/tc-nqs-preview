import os
os.environ["JAX_PLATFORM_NAME"] = "cuda"
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import logging
import pickle
#os.environ['XLA_FLAGS'] = '--xla_gpu_enable_tracing'
#os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
from jax import random
import pyscf
import jax
from scipy.special import comb
import time

from tcnqs.utils import build_ham_from_pyscf
import tcnqs.backflow as bf
# import tcnqs.trainer as trainer
import tcnqs.trainer_vite as trainer
from tcnqs.sampler.fssc import FSSC
import tcnqs.test.test_parameters as t



def test_backflow_vite(mol,n_core,num_epochs=2400, test=False ,random_key=17 ):
   
    if test:
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_debug_nans", True)
    rng = random.PRNGKey(random_key)
    myhf = mol.RHF().run()
    fci_e_pyscf = 0
    cisolver = pyscf.fci.FCI(myhf)
    fci_e_pyscf, ci_vector=cisolver.kernel()
    cisolver = pyscf.fci.FCI(myhf)
    print("E FCI = ", fci_e_pyscf)
 
    hamiltonian = build_ham_from_pyscf(mol, myhf, is_tc=t.is_tc)
    
    
    num_orbitals = hamiltonian.n_orb


    model_bf, variables_bf = bf.create_model(rng, input_shape = num_orbitals, 
                                            num_electrons= hamiltonian.n_elec,
                                            hidden_layer_sizes=t.hidden_layer_sizes, activation='tanh',n_bf_dets=t.n_bf_dets)
    variables_bf = jax.tree.map(lambda x: x.astype(jnp.float64), variables_bf)
    state_bf = trainer.create_train_state_VITE(rng, model_bf, variables_bf)
    
    train_losses_bf = []

    n_s_orb = (hamiltonian.n_orb//2)
    n_total_dets = comb(n_s_orb, hamiltonian.n_elec_a,exact=True)*comb(n_s_orb, hamiltonian.n_elec_b ,exact=True)
    batch_size = t.n_batch
    
    if n_core > n_total_dets:
        n_core = n_total_dets
        print(f"Warning: n_core specified is greater than total determinants in hilbert space. Falling back to n_core ={n_total_dets}")
    if n_core < batch_size:
        batch_size =n_core
        print(f"Warning: n_core specified is less than batch_size. Falling back to batch_size ={batch_size}")
    if n_core % batch_size != 0:
        n_core = n_core - n_core % t.n_batch
        print(f"Warning: n_core specified is not a multiple of batch_size. Falling back to n_core ={n_core}")

    n_connections= (1 + comb(hamiltonian.n_elec_a,2, exact=True)*
                        comb(n_s_orb-hamiltonian.n_elec_a,2,exact=True)+comb(hamiltonian.n_elec_b,2, 
                        exact=True)*comb(n_s_orb-hamiltonian.n_elec_b,2,exact=True)
                        + hamiltonian.n_elec_a*hamiltonian.n_elec_b*(n_s_orb-hamiltonian.n_elec_a)
                        *(n_s_orb-hamiltonian.n_elec_b) + n_s_orb*(hamiltonian.n_elec_a+hamiltonian.n_elec_b)
                        - hamiltonian.n_elec_a**2 - hamiltonian.n_elec_b**2) 
    
    max_n_full= batch_size*n_connections
    n_connected =  jnp.minimum(n_total_dets, max_n_full) - n_core
    n_full = n_total_dets
    #unique fn min of (n_total_dets, n_batch*n_connections) 
    sampler = FSSC(n_core, int(n_connected) ,hamiltonian.n_elec_a, hamiltonian.n_elec_b, num_orbitals, n_batch=batch_size)
    sampler = sampler.initialize()
    # svd_save = []
    # jax.profiler.start_trace("tmp/tensorboard")
    
    # ci_data = jax.vmap(hamiltonian, in_axes=(0,0))(sampler.core_space, sampler.core_space)

    ci_data = jnp.concatenate([jnp.expand_dims(hamiltonian(sampler.core_space[0], sampler.core_space[0]),axis=0),jnp.zeros(sampler.n_core-1)], axis=0)

    for epoch in range(num_epochs//20):
        state_bf, energy, pre_loss = trainer.pretrainer(state_bf, hamiltonian, sampler, ci_data)
        train_losses_bf.append(energy)
        # print(f"Pretraining Epoch: {epoch+1}, Loss: {energy}, Pre_loss: {pre_loss} ")
        logging.info(f"Pretraining Epoch: {epoch+1}, Loss: {energy}, Pre_loss: {pre_loss} ")
    for epoch in range(num_epochs):
        a = time.time()
        state_bf, loss_bf, sampler, grad_norm  = trainer.trainer_tc_stationary(state_bf, hamiltonian, sampler)
        
        #trainer.trainer_vite(state_bf, hamiltonian, sampler , solver='SR')
        # loss_bf.block_until_ready()
        train_losses_bf.append(loss_bf)
        b = time.time()
        # print(f"Epoch: {epoch+1}, Loss_bf: {loss_bf}, Time taken: {b-a} ") #, Time taken: {b-a}
        logging.info(f"Epoch: {epoch+1}, Loss_bf: {loss_bf}, grad_norm = {grad_norm} ")
    # train_losses_bf.block_until_ready()
    # jax.device_get(train_losses_bf)
    # jax.profiler.stop_trace()
    if t.save:
        save_path = f"tcnqs/simulations/SR/{mol.atom_symbol(0)}/{mol.atom}/ncore={t.n_core}/lr={t.learning_rate}/"
        os.makedirs(save_path, exist_ok=True) 
        with open(f"{save_path}/state_params.pkl", 'wb') as fp:
            pickle.dump(state_bf.params, fp)
    return train_losses_bf, fci_e_pyscf

if __name__ == '__main__':
    logging.basicConfig(
    filename=f"/scratch/u/Unik.Wadhwani/MasterArbeit/tc-nqs/tcnqs/minSR_{t.mol.atom_symbol(0)}_lr={t.learning_rate}_ncore={t.n_core}.log", 
    filemode='a', # Log file name
    level=logging.INFO,  # Set the logging level (INFO to capture losses)
    format='%(asctime)s - %(message)s',  # Format: timestamp followed by the loss
    )

    mol = t.mol
    print(jax.devices())
    start = time.time()
    print(mol.atom)

    losses , fci_e_pyscf = test_backflow_vite(mol , random_key=15,n_core=t.n_core,num_epochs=t.num_epochs, test= True)
    if t.save:
        print("Saving to file")
        save_path = f"tcnqs/simulations/SR/{mol.atom_symbol(0)}/{mol.atom}/ncore={t.n_core}/lr={t.learning_rate}/"
        os.makedirs(save_path, exist_ok=True)  # Create directories if they don't exist

        file_path = f"{save_path}/losses.npy"
        jnp.save(file_path, jnp.array(losses))
        #jnp.save(f"tcnqs/simulations/SR/{mol.atom}/ncore={t.n_core}/lr={t.learning_rate}.npy",jnp.array(losses))
        # jnp.save(f"tcnqs/simulations/fci_{mol.atom_symbol(0)}.npy",jnp.array(fci_e_pyscf))
    end = time.time()

    print("Finished execution!")
    print("Time taken: ", end-start)