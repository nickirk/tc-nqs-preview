import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.02'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#os.environ['XLA_FLAGS'] = '--xla_gpu_enable_tracing'
#os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pyscf
import jax
import time


from tcnqs.utils import generate_ci_data, build_ham_from_pyscf
import tcnqs.backflow as bf
import tcnqs.trainer as trainer
import tcnqs.test.test_parameters as t

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
                                ,hidden_layer_sizes=[4],activation='tanh', n_bf_dets=1)
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


if __name__ == '__main__':
    mol = t.mol
    print(jax.devices())
    start = time.time()
    test_backflow_unsupervised(mol, 17, test=True)
    end = time.time()
    print("Time taken: ", end-start)
