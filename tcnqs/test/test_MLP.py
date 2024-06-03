from tcnqs.MLP import NN1,create_model,mse_loss,train_step,create_train_state

from jax import numpy as jnp
from jax import random
import pyscf
from pyscf import fci,scf
import numpy as np
from pyscf.fci import cistring
import matplotlib.pyplot as plt

mol = pyscf.M(
    atom = 'O 0 0 0; O 0 0 1.1',  
    basis = 'sto-3g',
    symmetry = True,
    spin = 2,
    
)
myhf = mol.RHF().run()
cisolver = fci.FCI(myhf)


def convert_binary_to_array(strfil, num_orbitals):
    binary_str=str(bin(strfil))
    binary_str = binary_str[2:]
    binary_array = [int(bit) for bit in binary_str]
    leading_zeros = num_orbitals - len(binary_array)
    result_array = [0] * leading_zeros + binary_array
    
    return result_array

def generate_ci_data(num_orbitals,num_alpha_electrons,num_beta_electrons,ci):
    
    x=[]
    y=[]
    # print(ci.shape[0])
    for i in range(ci.shape[0]):
        for j in range(ci.shape[1]):
                
            y.append(ci[i,j])
            #print(num_orbitals,num_alpha_electrons,i)
            orba=cistring.addr2str(num_orbitals,num_alpha_electrons,i)
            orbb=cistring.addr2str(num_orbitals,num_beta_electrons,j)
            
            orba=convert_binary_to_array(orba,num_orbitals)
            orbb=convert_binary_to_array(orbb,num_orbitals)
            x.append(np.concatenate((orba,orbb),axis=0))
    x=jnp.array(x)
    y=jnp.array(y)
    return x,y

def test_backflow():
    rng = random.PRNGKey(7)
    num_orbitals = myhf.mo_coeff.shape[1]
    num_alpha_electrons, num_beta_electrons = mol.nelec
    cie,ci_vector=cisolver.kernel()
    
    x_train, y_train = generate_ci_data(num_orbitals,num_alpha_electrons,num_beta_electrons,ci_vector)
    input_size = 2*num_orbitals # Example input size
    num_samples = len(y_train) # Number of training samples
    
    model, variables = create_model(rng, (input_size,))
    state = create_train_state(rng, model, variables)
    
    # Training loop
    num_epochs = 50
    batch_size = 1
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            batch = (x_train[i:i+batch_size], y_train[i:i+batch_size])
            state, loss = train_step(state, batch)
            epoch_loss += loss
        
        average_epoch_loss = epoch_loss / (num_samples // batch_size)
        train_losses.append(average_epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {average_epoch_loss}")
    
    # Plotting the training curve
    plt.plot(range(1, num_epochs + 1), train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training Curve')
    plt.show()
    #plt.plot(x_tra,y_train)
    print("Training complete")


if __name__ == '__main__':
    test_backflow()