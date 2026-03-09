import os
# os.environ["JAX_PLATFORMS"] = "cuda"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'


# os.environ['XLA_FLAGS'] = '--xla_gpu_enable_tracing'
# os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
from jax import random
import pyscf
from scipy.special import comb
import time
# import numpy as np



from tcnqs.utils import build_ham_from_pyscf, wandb_init, wandb_log_energy
import tcnqs.backflow as bf
import tcnqs.trainer_vite as trainer
from tcnqs.sampler.fssc import FSSC
import tcnqs.test.test_parameters as t_params

 

def test_backflow_vite(mol,n_core,num_epochs=2400, test=False ,random_key=17 ):
    if test:
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_debug_nans", True)
        # jax.config.update("jax_cuda_visible_devices", 1)
    rng = random.PRNGKey(random_key)
    rng, init_rng = jax.random.split(rng)
    
    myhf = mol.RHF().run()
    cisolver = pyscf.fci.FCI(myhf)
    # fci_e_pyscf, ci_vector=cisolver.kernel()
    # cisolver = pyscf.fci.FCI(myhf)
    fci_e_pyscf = 0
    print("E FCI = ", fci_e_pyscf)
   
    
    hamiltonian = build_ham_from_pyscf(mol, myhf, is_tc=t_params.is_tc)
    
    num_orbitals = hamiltonian.n_orb

    model_bf, variables_bf = bf.create_model(init_rng, input_shape = num_orbitals, 
                                            num_electrons = hamiltonian.n_elec,
                                            hidden_layer_sizes = t_params.hidden_layer_sizes, 
                                            activation='tanh', 
                                            n_bf_dets=t_params.n_bf_dets)
    
    variables_bf = jax.tree.map(lambda x: x.astype(jnp.float64), variables_bf)
    state_bf = trainer.create_train_state_VITE(init_rng, model_bf, variables_bf)
    
    train_losses_bf = []

    n_s_orb = (hamiltonian.n_orb//2)
    n_total_dets = comb(n_s_orb, hamiltonian.n_elec_a,exact=True)*comb(n_s_orb, hamiltonian.n_elec_b ,exact=True)
    batch_size = t_params.n_batch
    
    if n_core > n_total_dets:
        n_core = n_total_dets
        print(f"Warning: n_core specified is greater than total determinants in hilbert space. Falling back to n_core ={n_total_dets}")
    if n_core < batch_size:
        batch_size =n_core
        print(f"Warning: n_core specified is less than batch_size. Falling back to batch_size ={batch_size}")
    if n_core % batch_size != 0:
        n_core = n_core - n_core % t_params.n_batch
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
    # svd_save = []
    if t_params.save:
        wandb_run = wandb_init(mol,t_params)
    else:
        wandb_run = None
        
    for epoch in range(num_epochs):
        state_bf, loss_bf, sampler  = trainer.trainer_vite(state_bf, hamiltonian, sampler)
        wandb_log_energy(wandb_run, loss_bf, epoch, fci_e_pyscf)
        # _wandb_log_params(wandb_run, state_bf, epoch, t_params, mol)
        train_losses_bf.append(loss_bf)
        print(f"Epoch: {epoch+1}, Loss_bf: {loss_bf}")
   
    # jnp.save(f"tcnqs/simulations/svd_Aij_{mol.atom_symbol(0)+mol.atom_symbol(1)}_lr=
    # {t_params.learning_rate}_ncore={t_params.n_core}.npy",jnp.array(svd_save))
    # if test:
    #     assert jnp.absolute(train_losses_bf[-1]-fci_e_pyscf) < 5e-3
    #     print("Success: Model trained successfully")
    
    if 'wandb_run' in locals() and wandb_run is not None:
        wandb_run.finish()
 
    return train_losses_bf, fci_e_pyscf

if __name__ == '__main__':
    mol = t_params.mol
    print(jax.devices())
    start = time.time()
    losses , fci_e_pyscf = test_backflow_vite(mol , random_key=15,n_core=t_params.n_core,
                                              num_epochs=t_params.num_epochs, test= True)

    if t_params.save:
        save_path = f"tcnqs/simulations/SR/{mol.atom_symbol(0)}/{mol.atom}/{mol.basis}/ncore={t_params.n_core}/lr={t_params.learning_rate}/"
        print("Saving to file:", save_path)
        os.makedirs(save_path, exist_ok=True)  # Create directories if they don't exist
        file_path = f"{save_path}/losses_{t_params.hidden_layer_sizes}_{t_params.n_bf_dets}.npy"
        jnp.save(file_path, jnp.array(losses))
        # jnp.save(f"tcnqs/simulations/SR/{mol.atom}/ncore={t_params.n_core}/lr={t_params.learning_rate}.npy",jnp.array(losses))
        jnp.save(f"tcnqs/simulations/fci_{mol.atom_symbol(0)}.npy",jnp.array(fci_e_pyscf))
    end = time.time()
    print("Time taken: ", end-start)