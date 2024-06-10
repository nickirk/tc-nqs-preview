import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from jax import random
import matplotlib.pyplot as plt

from jax.nn.initializers import normal
class BACKFLOW(nn.Module):

    num_orbital:int
    num_electron:int

    @nn.compact
    def __call__(self, x):
        
        #y = jnp.copy(x)
        mask = jnp.where(x == 1, 1, 0)
        selected_config = jnp.nonzero(mask, size=self.num_electron)[1]
        
        x = nn.Dense(features=4)(x)  
        x = nn.tanh(x)
        x = nn.Dense(features=4)(x)  

        x = nn.tanh(x)
        
        #Backflow
        x = nn.DenseGeneral(features=(self.num_orbital,self.num_electron))(x)
        x = nn.tanh(x)
        
        x = x[:,selected_config , :]
        #print(jnp.linalg.det(x))
        return jnp.linalg.det(x)
    
def row_select(y):
    selected_configs=[]
    for j, yj in enumerate(y):
            # print(yj)
            if yj==1:
                selected_configs.append(j)
    return selected_configs

def create_model(rng, input_shape,num_electrons): #input shape = #orbitals total
    model = BACKFLOW(num_orbital=input_shape,num_electron=num_electrons)
    initial=jnp.concatenate((jnp.ones((num_electrons,)),
                                                 jnp.zeros((input_shape-num_electrons,))),axis=0)
    initial=jnp.reshape(initial,(1,input_shape))
    variables = model.init(rng,initial)#initializer=normal 
    return model, variables

# Define the mean squared error (MSE) loss function
def mse_loss(params, apply_fn, x, y):    
    preds = apply_fn({'params': params}, x)
    return jnp.mean((preds - y) ** 2)


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        x, y = batch
        loss = mse_loss(params, state.apply_fn, x, y)
        #print(loss)
        return loss
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    #print(loss_fn(state.params),grads)
    return state, loss_fn(state.params)


def create_train_state(rng, model, variables):
    tx = optax.adam(learning_rate=0.01)
    return train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
