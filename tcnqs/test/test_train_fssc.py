import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
import tcnqs.trainer as trainer
from tcnqs.sampler.fssc import FSSC
import tcnqs.test.test_parameters as t_params
from tcnqs.utils import build_ham_from_pyscf, wandb_init, wandb_log_energy

def test_backflow_fssc(mol, n_core, num_epochs=2400, test=False ,random_key=17):
    if t_params.save:
        wandb_run = wandb_init(mol,t_params)
    else:
        wandb_run = None

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
                                             hidden_layer_sizes=t_params.hidden_layer_sizes, 
                                             activation='tanh',n_bf_dets=t_params.n_bf_dets)
    
    state_bf =trainer.create_train_state(rng, model_bf, variables_bf)
    
    train_losses_bf = []

    n_s_orb = (hamiltonian.n_orb//2)
    n_total_dets = comb(n_s_orb, hamiltonian.n_elec_a, exact=True)
    n_total_dets *= comb(n_s_orb, hamiltonian.n_elec_b ,exact=True)
    
    if n_core > n_total_dets:
        n_core = n_total_dets
        print(f"Warning: n_core specified is greater than total determinants in hilbert space. Falling back to n_core ={n_total_dets}")

    # TODO: make this line more readable
    max_n_full= n_core*(1 + comb(hamiltonian.n_elec_a,2, exact=True)*
                        comb(n_s_orb-hamiltonian.n_elec_a,2,exact=True)+comb(hamiltonian.n_elec_b,2, 
                        exact=True)*comb(n_s_orb-hamiltonian.n_elec_b,2,exact=True)
                        + hamiltonian.n_elec_a*hamiltonian.n_elec_b*(n_s_orb-hamiltonian.n_elec_a)
                        *(n_s_orb-hamiltonian.n_elec_b) + n_s_orb*(hamiltonian.n_elec_a+hamiltonian.n_elec_b)
                        - hamiltonian.n_elec_a**2 - hamiltonian.n_elec_b**2) 
    
    n_connected =  jnp.minimum(n_total_dets, max_n_full) - n_core
    
    sampler = FSSC(n_core, int(n_connected), hamiltonian.n_elec_a, 
                   hamiltonian.n_elec_b, num_orbitals,n_batch=n_core)
    sampler = sampler.initialize()
    # stored = sampler.next_sample_stored(hamiltonian)
    # flag = True
    

    for epoch in range(num_epochs):
        # old_sample = sampler.core_space
        state_bf, loss_bf, sampler = trainer.train_step_fssc(
            state_bf, hamiltonian, sampler) # , flag, stored_tuple
        train_losses_bf.append(loss_bf)
        wandb_log_energy(wandb_run, loss_bf, epoch, fci_e_pyscf)
        # new_sample = sampler.core_space
        # print(jnp.all(jnp.unique(old_sample, axis=0, size=sampler.n_core) == jnp.unique(new_sample, axis=0, size=sampler.n_core)))
        print(f"Epoch {epoch+1} , Loss_bf: {loss_bf }")
    
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
    losses, fci_e_pyscf = test_backflow_fssc(mol, n_core=t_params.n_core, test=True, 
                       random_key=15, num_epochs=t_params.num_epochs)
    if t_params.save:
        save_path = f"tcnqs/simulations/fssc/{mol.atom_symbol(0)}/{mol.atom}/{mol.basis}/ncore={t_params.n_core}/lr={t_params.learning_rate}/"
        print("Saving to file:", save_path)
        os.makedirs(save_path, exist_ok=True)  # Create directories if they don't exist
        file_path = f"{save_path}/losses.npy"
        jnp.save(file_path, jnp.array(losses))
        # jnp.save(f"tcnqs/simulations/SR/{mol.atom}/ncore={t.n_core}/lr={t.learning_rate}.npy",jnp.array(losses))
        #jnp.save(f"tcnqs/simulations/fci_{mol.atom_symbol(0)}.npy",jnp.array(fci_e_pyscf))
    end = time.time()
    print("Time taken: ", end-start)
