import jax.numpy as jnp
from tcnqs.fcidump import read_2_spin_orbital_seprated as read2
from tcnqs.hamiltonian import HAMILTONIAN
import pyscf
from tcnqs.utils import generate_ci_data
import numpy as np
from pyscf.tools import fcidump

# Define the molecule and Pyscf calculations
mol = pyscf.M(
atom = 'H 0 0 0; H 0 0 1.0;H 0 0 2.0; H 0 0 3.0',  
basis = 'sto3g',
spin = 0,
charge = 0,
symmetry = False
)

myhf = mol.RHF().run()
cisolver = pyscf.fci.FCI(myhf)
num_orbitals = myhf.mo_coeff.shape[1]
num_alpha_electrons, num_beta_electrons = mol.nelec
FCI_e_pyscf, ci_vector=cisolver.kernel()
cisolver = pyscf.fci.FCI(myhf)
#cie,civ=cisolver.kernel()
# print(civ )
num_orbitals = myhf.mo_coeff.shape[1]
# print(num_orbitals)
num_alpha_electrons, num_beta_electrons = mol.nelec
# print(num_alpha_electrons,num_beta_electrons)
# print(civ.shape)
x_train, y_train = generate_ci_data(num_orbitals,num_alpha_electrons,num_beta_electrons,ci_vector)
# #
# Read the FCIDUMP file
fcidump_file = 'tcnqs/test/dataset_fcidump/fcidump'
fcidump.from_scf(myhf, fcidump_file)

n_sites, n_elec, ecore, h1e_s, g2e_s = read2(fcidump_file)

# Create FCI Hamiltonian
hamiltonian = HAMILTONIAN(n_elec, n_sites, h1e_s, g2e_s)

H= np.zeros((len(x_train), len(x_train)))
for i in range(len(x_train)):
    for j in range(len(x_train)):
        H[i,j] = hamiltonian(x_train[i], x_train[j])
        
        
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from jax import random
from typing import List
from dataclasses import field

from jax.nn.initializers import normal
import tcnqs.backflow as bf

import tcnqs.mlp as mlp





def loss_fon(params,apply_fn,x,H):
    # this will calculate the batch loss directly for all x,y
    preds = apply_fn({'params': params}, x)
    Norm_preds = jnp.sum(jnp.power(preds,2))
    #Norm_y= jnp.sum(jnp.power(y,2))
    
    overlap_coeff = jnp.sum(jnp.conj(preds)[:, None] * preds[None, :] * H)# Define jax vector product of preds and y  
    
    return  overlap_coeff/Norm_preds # return the log loss error

@jax.jit
def train_step_loss(state,batch,H):
    def loss_fn(params):
        x, y = batch
        loss = loss_fon(params, state.apply_fn, x, H)
        return loss
    # loss_fn = log_loss()
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_fn(state.params)




rng = random.PRNGKey(27)


input_size = 2*num_orbitals 
num_samples = len(y_train) 

model_bf, variables_bf = bf.create_model(rng, input_size,
                                num_electrons=num_alpha_electrons+num_beta_electrons
                                ,hidden_layer_sizes=[4],activation='tanh')
state_bf = bf.create_train_state(rng, model_bf, variables_bf)

model_mlp, variables_mlp = mlp.create_model(rng, input_size ,hidden_layer_sizes=[4,784],activation='tanh')
state_mlp = mlp.create_train_state(rng, model_mlp, variables_mlp)
#print(variables)
# Training loop
num_epochs = 3000
batch_size = num_samples
train_losses_bf = []
train_losses_mlp = []

for epoch in range(num_epochs):
    epoch_loss_bf = 0.0
    epoch_loss_mlp = 0.0
    for i in range(0, num_samples, batch_size):
        batch = (x_train[i:i+batch_size], y_train[i:i+batch_size])
        
        state_bf, loss_bf = train_step_loss(state_bf, batch, H)
        epoch_loss_bf += loss_bf
        
        
        state_mlp, loss_mlp = train_step_loss(state_mlp, batch, H)
        epoch_loss_mlp += loss_mlp
        
    
    average_epoch_loss_bf = epoch_loss_bf / (num_samples // batch_size)
    train_losses_bf.append(average_epoch_loss_bf + ecore)
    
    average_epoch_loss_mlp = epoch_loss_mlp / (num_samples // batch_size)
    train_losses_mlp.append(average_epoch_loss_mlp + ecore)
    
    print(f"Epoch {epoch+1}, Loss_bf: {average_epoch_loss_bf + ecore} , Loss_mlp: {average_epoch_loss_mlp+ ecore}")
    
print(FCI_e_pyscf, jnp.average(jnp.array(train_losses_bf[-50:],dtype=jnp.float32)), jnp.average(jnp.array(train_losses_mlp[-50:],dtype=jnp.float32)))

import matplotlib.pyplot as plt

plt.plot(range(1, num_epochs + 1), train_losses_bf, label="Backflow")
plt.plot(range(1, num_epochs + 1), train_losses_mlp, label="MLP")

plt.plot(range(1, num_epochs + 1),FCI_e_pyscf*np.ones(num_epochs), color='r', label="true FCI energy" , linewidth=1)
plt.legend(loc="upper right")
plt.xlabel("Epoch")
plt.ylabel("Energy of H2")
plt.ylim(1.05*FCI_e_pyscf,0.8*FCI_e_pyscf)
plt.savefig('result.png')

#plt.show()

