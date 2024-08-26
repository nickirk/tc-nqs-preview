import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
from typing import List
from dataclasses import field
from functools import partial
from jax.nn.initializers import normal
from functools import partial

from tcnqs.sampler.connected_dets import generate_connected_space
## using float 64 for the determinant to solve the bug temporarily
jax.config.update("jax_enable_x64", True)
class Backflow(nn.Module):

    num_orbital: int
    num_electron: int
    hidden_layer_sizes: List[int] = field(default_factory=lambda: [4, 4])
    activation: str = 'relu'
    n_bf_dets: int = 1
    
    def setup(self) -> None:
        # Map the string to the actual function
        self.activation_fn = {
            'relu': nn.relu,
            'sigmoid': nn.sigmoid,
            'tanh': nn.tanh,
            # Add more if needed
        }[self.activation]

        self.hidden_dense = [nn.Dense(
            features=size, kernel_init=positive_random_init, dtype=jnp.float64
            ) for size in self.hidden_layer_sizes]
        self.dense_general = nn.DenseGeneral(
            features=(self.num_orbital, self.num_electron), dtype=jnp.float64
            )
        
        

    @partial(jax.vmap, in_axes=(None,0))
    def __call__(self, x):
        
        selected_config = jnp.nonzero(x, size=self.num_electron)[0]
        
        for dense_layer in self.hidden_dense:
            x = dense_layer(x)
            x = self.activation_fn(x)
        # jax.debug.breakpoint()
        x = self.dense_general(x)
        x = x[selected_config , :]
        ##x = jnp.sum(x)
        # x = x + x.T
        ### Slogdet is not a good option as the sgn it provides is discontinous
        ### This creats a problem in the gradient calculation 
        sgn, val = jnp.linalg.slogdet(x)
        x = sgn * jnp.exp(val)

        #x = jax.jit(jnp.linalg.det,device=jax.devices('cpu')[0])(x)
        
        #x = jnp.average(x) 

        # trim away inf values in val and replace them with 0
        # val = jnp.where(jnp.isinf(val), 0, val)
        #logmax = jnp.max(val)
        
        #print(x)
        # jax.debug.breakpoint()
        #x = self.dense_general(x)
        #x = x[selected_config , :]
        #x = jnp.sum(x)# sgn, val = jnp.linalg.slogdet(x)
        # x = sgn * jnp.exp(val)
        # x = jnp.linalg.det(x)
        
        ### Slogdet is not a good option as the sgn it provides is discontinous
        ### This creats a problem in the gradient calculation 
        
        #sgn, val = jnp.linalg.slogdet(x)
        # x = sgn * jnp.exp(val)
        # x = jax.lax.select(val>-5,jnp.float64(sgn * jnp.exp(val)),0.0)

        return jax.lax.cond(jnp.sum(selected_config)==0,lambda :0.0,lambda :jnp.float64(x))

def positive_random_init(key, shape, dtype=jnp.float32):
    return random.uniform(key, shape, dtype, minval=0, maxval=0.2)

def create_model(rng, input_shape, num_electrons, hidden_layer_sizes=[4],    
                 activation='relu'): 
    model = Backflow(num_orbital=input_shape, num_electron=num_electrons, 
                     hidden_layer_sizes=hidden_layer_sizes,
                     activation=activation)
    initial=jnp.concatenate((jnp.ones(num_electrons),
                            jnp.zeros(input_shape-num_electrons)),
                            axis=0)
    initial=jnp.reshape(initial,(1,input_shape))
    variables = model.init(rng,initial)#initializer=normal 
    return model, variables

# @jax.jit(device=jax.devices('cpu')[0])
# def det(x):
#     # sgn, val = jnp.linalg.slogdet(x)
#     # x = sgn * jnp.exp(val)
#     x = jnp.linalg.det(x)
#     return x
    

