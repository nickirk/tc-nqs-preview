import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from jax import random
from typing import List
from dataclasses import field
from functools import partial
from jax.nn.initializers import normal
from functools import partial

from tcnqs.sampler.connected_dets import generate_connected_space

class BACKFLOW(nn.Module):

    num_orbital: int
    num_electron: int
    hidden_layer_sizes: List[int] = field(default_factory=lambda: [4, 4])
    activation: str = 'relu'

    def setup(self) -> None:
        # Map the string to the actual function
        self.activation_fn = {
            'relu': nn.relu,
            'sigmoid': nn.sigmoid,
            'tanh': nn.tanh,
            # Add more if needed
        }[self.activation]

        self.hidden_dense = [nn.Dense(
            features=size, kernel_init=positive_random_init
            ) for size in self.hidden_layer_sizes]
        self.dense_general = nn.DenseGeneral(
            features=(self.num_orbital, self.num_electron)
            )
        

    @partial(jax.vmap, in_axes=(None,0))
    def __call__(self, x):
        
        selected_config = jnp.nonzero(x, size=self.num_electron)[0]
        
        for dense_layer in self.hidden_dense:
            x = dense_layer(x)
            x = self.activation_fn(x)

        x = self.dense_general(x)
        x = x[selected_config , :]
        x = jnp.linalg.det(x)#*jnp.sqrt(1/(self.num_electron))
        return x
    
def positive_random_init(key, shape, dtype=jnp.float32):
    return random.uniform(key, shape, dtype, minval=0, maxval=0.2)

def create_model(rng, input_shape, num_electrons, hidden_layer_sizes=[4],    
                 activation='relu'): 
    model = BACKFLOW(num_orbital=input_shape, num_electron=num_electrons, 
                     hidden_layer_sizes=hidden_layer_sizes,
                     activation=activation)
    initial=jnp.concatenate((jnp.ones(num_electrons),
                            jnp.zeros(input_shape-num_electrons)),
                            axis=0)
    initial=jnp.reshape(initial,(1,input_shape))
    variables = model.init(rng,initial)#initializer=normal 
    return model, variables

# Define the mean squared error (MSE) loss function

@jax.jit
def train_step(state, batch):
    def mse_loss(params, apply_fn, x, y):    
        preds = apply_fn({'params': params}, x)
        #preds_value = jax.device_get(preds).item()
        #y_value = jax.device_get(y).item()
        #print(preds_value, y_value)
        # preds = preds/ jnp.linalg.norm(preds)
        return jnp.mean((preds - y) ** 2)
    
    def loss_fn(params):
        x, y = batch
        loss = mse_loss(params, state.apply_fn, x, y)
        return loss
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_fn(state.params)

#@jax.jit
def train_step_log(state,batch):
    
    def log_loss(params,apply_fn,x,y):
        # this will calculate the batch loss directly for all x,y
        preds = apply_fn({'params': params}, x)
        Norm_preds = jnp.sum(jnp.power(preds,2))
        Norm_y= jnp.sum(jnp.power(y,2))
        
        overlap_coeff = jnp.dot(preds,y)**2# Define jax vector product of preds and y  
        
        return -jnp.log(overlap_coeff/(Norm_y*Norm_preds)) # return the log loss error
    
    def loss_fn(params):
        x, y = batch
        loss = log_loss(params, state.apply_fn, x, y)
        return loss
    # loss_fn = log_loss()
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_fn(state.params)

@jax.jit
def train_step_hamiltonian(state,batch,H):
    def hamiltonian_loss(params,apply_fn,x,H):
    
        preds = apply_fn({'params': params}, x)
        Norm_preds = jnp.linalg.norm(preds)**2
        #print(Norm_preds)
        overlap_coeff = jnp.dot(preds,jnp.dot(H,preds))
        #jnp.sum((preds)[:, None] * preds[None, :] * H)
        
        return  overlap_coeff/Norm_preds 

    def loss_fn(params):
        x, y = batch
        loss = hamiltonian_loss(params, state.apply_fn, x, H)
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_fn(state.params)

@partial(jax.jit, static_argnums=(2))
def train_step_connections(state, batch, Hamiltonain):
    def hamiltonian_loss(params, apply_fn, x, Hamiltonain):
        C_i = apply_fn({'params': params}, x)
        #a= apply_fn({'params': params}, jnp.expand_dims(x[5],axis=0))
        Norm = jnp.linalg.norm(C_i)
        
        def find_Ci(det):
            c=jnp.all(x==det,axis=1)
            ## Lax condition if teh array doesnt have any True values
            ## Debug and test this line
            # define lambda function 
            # return jax.lax.cond(jnp.any(c), lambda x: jnp.asarray(C_i[jnp.where(c,size = 1)[0][0]]),
            # lambda : apply_fn({'params': params}, det)  )
            
            # return jax.lax.cond(jnp.any(c), lambda x: jnp.asarray(C_i[jnp.where(c,size = 1)[0][0]]),lambda x: apply_fn, ({'params': params}, det))                            
            
            return jnp.asarray(C_i[jnp.where(c,size = 1)[0][0]])
        #return apply_fn({'params': params}, det)
       
        def overlap(slater_determinant, C_i):
            connected_space = generate_connected_space(slater_determinant, Hamiltonain.n_orb, Hamiltonain.n_elec)
            psi_H_xi = jax.vmap(Hamiltonain,in_axes=(None,0))(slater_determinant,connected_space)
            xi_psi = jax.vmap(find_Ci,in_axes=0)(connected_space)
            # xi_psi = 
            return C_i*jnp.dot(psi_H_xi,xi_psi)
            
        overlap_coeff = jnp.sum(jax.vmap(overlap, in_axes =(0,0))(x,C_i))
        
        return overlap_coeff/Norm**2
    
    def loss_fn(params):
        x, y = batch
        loss = hamiltonian_loss(params, state.apply_fn, x, Hamiltonain)
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_fn(state.params)

def create_train_state(rng, model, variables):
    tx = optax.adam(learning_rate=0.001)
    return train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
