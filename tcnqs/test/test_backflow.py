from tcnqs import backflow as bf
from jax import random
import pyscf
from tcnqs.utils import generate_ci_data

mol = pyscf.M(
    atom = 'O 0 0 0; O 0 0 1.0',  
    basis = 'sto3g',
    symmetry = False,
    spin = 0,
)
myhf = mol.RHF().run()
#myhf.mo_coeff = myhf.mo_coeff[:, :3]
cisolver = pyscf.fci.FCI(myhf)


def test_backflow():
    rng = random.PRNGKey(7)
    num_orbitals = myhf.mo_coeff.shape[1]
    num_alpha_electrons, num_beta_electrons = mol.nelec
    cie,ci_vector=cisolver.kernel()
    
    x_train, y_train = generate_ci_data(num_orbitals,num_alpha_electrons,num_beta_electrons,ci_vector)
    input_size = 2*num_orbitals 
    num_samples = len(y_train) 
    
    model, variables = bf.create_model(rng, input_size,
                                    num_electrons=num_alpha_electrons+num_beta_electrons)
    state = bf.create_train_state(rng, model, variables)
    #print(variables)
    # Training loop
    num_epochs = 50
    batch_size = 1
    train_losses = []


    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            batch = (x_train[i:i+batch_size], y_train[i:i+batch_size])
            state, loss = bf.train_step(state, batch)
            epoch_loss += loss
        
        average_epoch_loss = epoch_loss / (num_samples // batch_size)
        train_losses.append(average_epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {average_epoch_loss}")
    
    # Plotting the training curve

if __name__ == '__main__':
    test_backflow()