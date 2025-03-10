import os
os.environ["JAX_PLATFORM_NAME"] = "cuda"
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'
os.environ['CUDA_VISIBLE_DEVICES'] = '8'

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
    
    max_n_full= n_core*n_connections
    n_connected =  jnp.minimum(n_total_dets, max_n_full) - n_core
    n_full = n_total_dets
    #unique fn min of (n_total_dets, n_batch*n_connections) 
    sampler = FSSC(n_core, int(n_connected) ,hamiltonian.n_elec_a, hamiltonian.n_elec_b, num_orbitals, n_batch=batch_size)
    sampler = sampler.initialize()
    # svd_save = []
    for epoch in range(num_epochs):
        
        state_bf, loss_bf, sampler  = trainer.trainer_vite(state_bf, hamiltonian, sampler)
        #,A_ij
        # if epoch % 1000 == 0:
        #     svd = jax.jit(lambda x : jnp.linalg.svd(x , compute_uv=False,hermitian=True))(A_ij)
        #     svd = jnp.abs(svd)
        #     svd_index = jnp.argsort(svd)
        #     svd_save.append(svd[svd_index]/ jnp.max(svd))
            
        train_losses_bf.append(loss_bf)
        
        print(f"Epoch: {epoch+1}, Loss_bf: {loss_bf}")
   
    # jnp.save(f"tcnqs/simulations/svd_Aij_{mol.atom_symbol(0)+mol.atom_symbol(1)}_lr={t.learning_rate}_ncore={t.n_core}.npy",jnp.array(svd_save))
    # if test:
    #     assert jnp.absolute(train_losses_bf[-1]-fci_e_pyscf) < 5e-3
    #     print("Success: Model trained successfully")
    
    return train_losses_bf, fci_e_pyscf

if __name__ == '__main__':
    mol = t.mol
    print(jax.devices())
    start = time.time()
    losses , fci_e_pyscf = test_backflow_vite(mol , random_key=15,n_core=t.n_core,num_epochs=t.num_epochs, test= True)
    
    if t.save:
        print("Saving to file")
        jnp.save(f"tcnqs/simulations/vite_{mol.atom_symbol(0)+mol.atom_symbol(1)}_lr={t.learning_rate}_ncore={t.n_core}.npy",jnp.array(losses))
        jnp.save(f"tcnqs/simulations/fci_{mol.atom_symbol(0)+mol.atom_symbol(1)}.npy",jnp.array(fci_e_pyscf))
    end = time.time()
    print("Time taken: ", end-start)