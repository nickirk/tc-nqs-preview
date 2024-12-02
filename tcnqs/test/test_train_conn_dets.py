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
import time


from tcnqs.utils import generate_ci_data, build_ham_from_pyscf
import tcnqs.backflow as bf
import tcnqs.trainer as trainer
import tcnqs.test.test_parameters as t


def test_backflow_connected(mol, random_key , num_epochs=2400, test=False):
    if test:
        jax.config.update("jax_enable_x64", True)
    
    rng = random.PRNGKey(random_key)
    
    myhf = mol.RHF().run()
    cisolver = pyscf.fci.FCI(myhf)
    fci_e_pyscf, ci_vector=cisolver.kernel()
    cisolver = pyscf.fci.FCI(myhf)
    print("E FCI = ", fci_e_pyscf)
    
    
    # Create FCI Hamiltonian
    hamiltonian = build_ham_from_pyscf(mol, myhf)
    
    x_train, y_train = generate_ci_data(hamiltonian.n_orb//2, hamiltonian.n_elec_a, 
                                        hamiltonian.n_elec_b, ci_vector)
    x_train = jnp.asarray(x_train,dtype=jnp.uint8)
    # hamiltonian , ecore = test_hamiltonian(mol)

    
    num_orbitals = hamiltonian.n_orb

    model_bf, variables_bf = bf.create_model(rng, input_shape = num_orbitals, 
                                            num_electrons= hamiltonian.n_elec
                                ,hidden_layer_sizes=[], activation='tanh')
    state_bf = trainer.create_train_state(rng, model_bf, variables_bf)
    
    train_losses_bf = []
    
    for epoch in range(num_epochs):
        epoch_loss_bf = 0.0
            
        state_bf, loss_bf = trainer.train_step_connections(state_bf, x_train, hamiltonian)
        epoch_loss_bf += loss_bf
        
        average_epoch_loss_bf = epoch_loss_bf # / (num_samples // batch_size)
        train_losses_bf.append(average_epoch_loss_bf )
        
        print(f"Epoch {epoch+1} , Loss_bf: {average_epoch_loss_bf }")
    
    if test:
        assert jnp.absolute(train_losses_bf[-1]-fci_e_pyscf) < 1e-3
        print("Success: Model trained successfully")
    
    return train_losses_bf, fci_e_pyscf

if __name__ == '__main__':
    mol = t.mol
    print(jax.devices())
    start = time.time()
    test_backflow_connected(mol, 17, test= True)
    end = time.time()
    print("Time taken: ", end-start)
