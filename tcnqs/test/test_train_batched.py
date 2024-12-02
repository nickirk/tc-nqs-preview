import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.02'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
import tcnqs.test.test_parameters as t

def test_backflow_batched(mol, n_core, num_epochs=2400, test=False, random_key=17):
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
                                             hidden_layer_sizes=t.hidden_layer_sizes, 
                                             activation='tanh',n_bf_dets=t.n_bf_dets)
    variables_bf = jax.tree.map(lambda x: x.astype(jnp.float64), variables_bf)
    state_bf = trainer.create_train_state(rng, model_bf, variables_bf)
    
    train_losses_bf = []

    n_s_orb = (hamiltonian.n_orb//2)
    n_total_dets = comb(n_s_orb, hamiltonian.n_elec_a, exact=True)
    n_total_dets *= comb(n_s_orb, hamiltonian.n_elec_b, exact=True)
    batch_size = t.batch_size
    
    if n_core > n_total_dets:
        n_core = n_total_dets
        print(f"Warning: n_core specified is greater than total determinants in hilbert space." 
              f"Falling back to n_core ={n_total_dets}")
    if n_core < batch_size:
        batch_size =n_core
        print(f"Warning: n_core specified is less than batch_size."
              f"Falling back to batch_size ={batch_size}")
    if n_core % batch_size != 0:
        n_core = n_core - n_core % t.batch_size
        print(f"Warning: n_core specified is not a multiple of batch_size."
              f"Falling back to n_core ={n_core}")

    # TODO: make this more readable
    n_connections= (1 + comb(hamiltonian.n_elec_a,2, exact=True)*
                        comb(n_s_orb-hamiltonian.n_elec_a,2,exact=True)+comb(hamiltonian.n_elec_b,2, 
                        exact=True)*comb(n_s_orb-hamiltonian.n_elec_b,2,exact=True)
                        + hamiltonian.n_elec_a*hamiltonian.n_elec_b*(n_s_orb-hamiltonian.n_elec_a)
                        *(n_s_orb-hamiltonian.n_elec_b) + n_s_orb*(hamiltonian.n_elec_a+hamiltonian.n_elec_b)
                        - hamiltonian.n_elec_a**2 - hamiltonian.n_elec_b**2) 
    
    n_connected =  30000 #jnp.minimum(n_total_dets, max_n_full) - n_core
    #unique fn min of (n_total_dets, n_batch*n_connections) 
    sampler = FSSC(n_core, int(n_connected), hamiltonian.n_elec_a, hamiltonian.n_elec_b, 
                   num_orbitals, n_batch=batch_size)
    sampler = sampler.initialize()
   
    #stored = trainer.sample_ham_wrap(sample,hamiltonian,sampler)
    #sample = sample[0][:sampler.n_core]
    for epoch in range(num_epochs):
        
        state_bf, loss_bf, sampler = trainer.train_step_batched(state_bf, hamiltonian, sampler)
        train_losses_bf.append(loss_bf)
        
        print(f"Epoch {epoch+1} , Loss_bf: {loss_bf }")
   
    if test:
        assert jnp.absolute(train_losses_bf[-1]-fci_e_pyscf) < 5e-3
        print("Success: Model trained successfully")
    
    return train_losses_bf, fci_e_pyscf

if __name__ == '__main__':
    mol = t.mol
    print(jax.devices())
    start = time.time()
    test_backflow_batched(mol, n_core=t.n_core, test=True)
    end = time.time()
    print("Time taken: ", end-start)
