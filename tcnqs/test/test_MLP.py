from jax import numpy as jnp
from jax import random
import pyscf
from pyscf import fci
import numpy as np
from pyscf.fci import cistring
import matplotlib.pyplot as plt

from tcnqs import mlp
from tcnqs.utils import generate_ci_data

mol = pyscf.M(
    atom = 'O 0 0 0; O 0 0 1.0',  
    basis = 'sto-3g',
    spin = 0
)
myhf = mol.RHF().run()
cisolver = fci.FCI(myhf)



def test_mlp():
    rng = random.PRNGKey(7)
    num_orbitals = myhf.mo_coeff.shape[1]
    num_alpha_electrons, num_beta_electrons = mol.nelec
    cie,ci_vector=cisolver.kernel()
    
    x_train, y_train = generate_ci_data(num_orbitals,num_alpha_electrons,num_beta_electrons,ci_vector)
    input_size = 2*num_orbitals # Example input size
    num_samples = len(y_train) # Number of training samples
    
    model, variables = mlp.create_model(rng, (input_size,))
    state = mlp.create_train_state(rng, model, variables)
    
    # Training loop
    num_epochs = 50
    batch_size = 1
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            batch = (x_train[i:i+batch_size], y_train[i:i+batch_size])
            state, loss = mlp.train_step(state, batch)
            epoch_loss += loss
        
        average_epoch_loss = epoch_loss / (num_samples // batch_size)
        train_losses.append(average_epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {average_epoch_loss}")
    

if __name__ == '__main__':
    test_mlp()