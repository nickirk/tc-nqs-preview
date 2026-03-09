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
from scipy.special import comb
import time


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
                                       hidden_layer_sizes=[4,4], activation='tanh', n_bf_dets=1)
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


if __name__ == '__main__':
    mol = t.mol
    print(jax.devices())
    test_backflow_supervised(mol, 0)
    start = time.time()
    end = time.time()
    print("Time taken: ", end-start)
