from jax import random
import pyscf
import pickle

from tcnqs.utils import generate_ci_data
from tcnqs import backflow as bf

mol = pyscf.M(
    atom = 'O 0 0 0; O 0 0 1.0',  
    basis = 'sto3g',
    spin = 0
)
myhf = mol.RHF().run()
cisolver = pyscf.fci.FCI(myhf)


def test_backflow_supervised():
    rng = random.PRNGKey(7)
    num_orbitals = myhf.mo_coeff.shape[1]
    num_alpha_electrons, num_beta_electrons = mol.nelec
    cie, ci_vector=cisolver.kernel()
    
    x_train, y_train = generate_ci_data(num_orbitals, num_alpha_electrons, num_beta_electrons, ci_vector)
    input_size = 2*num_orbitals 
    num_samples = len(y_train) 
    
    model, variables = bf.create_model(rng, input_size,
                                       num_electrons=num_alpha_electrons+num_beta_electrons, hidden_layer_sizes=[4], activation='relu')
    state = bf.create_train_state(rng, model, variables)
    #print(variables)
    # Training loop
    num_epochs = 10000
    batch_size = 196
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
            state, loss = bf.train_step(state, batch)
            epoch_loss += loss
        
        average_epoch_loss = epoch_loss / (num_samples // batch_size)
        train_losses.append(average_epoch_loss)
        rng = subrng
        print(f"Epoch {epoch+1}, Loss: {average_epoch_loss}")
    
    with open('params.pkl', 'wb') as f:
        pickle.dump(state.params, f)

    print("Parameters saved")
    
    assert train_losses[-1] < 1e-3
    print("Success")

if __name__ == '__main__':
    test_backflow_supervised()