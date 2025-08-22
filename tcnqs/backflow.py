import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
from typing import List
from dataclasses import field
from functools import partial

## using float 64 for the determinant to solve the bug temporarily
jax.config.update("jax_enable_x64", True)
class Backflow(nn.Module):

    num_orbital: int
    num_electron: int
    n_bf_dets: int = 1
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
            features=size, kernel_init=positive_random_init, dtype=jnp.float64
            ) for size in self.hidden_layer_sizes]
        self.dense_general = nn.DenseGeneral(
            features=(self.n_bf_dets,self.num_orbital, self.num_electron), dtype=jnp.float64
            )
        self.bf_det_params = self.param('bf_det_params', nn.initializers.lecun_normal(dtype=jnp.float64), (self.n_bf_dets,1))
        
        

    @partial(jax.vmap, in_axes=(None,0))
    def __call__(self, x):
        
        selected_config = jnp.nonzero(x, size=self.num_electron)[0]
        for dense_layer in self.hidden_dense:
            x = dense_layer(x)
            x = self.activation_fn(x)
        
        x = self.dense_general(x)
        x = x[:,selected_config,:]
        #x = jnp.linalg.vecdot( self.bf_det_params ,x, axis=0)
        # sgn, val = jnp.linalg.slogdet(x)
        # x = sgn * jnp.exp(val)
        x = jnp.linalg.det(x)
        x = jnp.dot(x,self.bf_det_params)[0]
        return jax.lax.select(jnp.sum(selected_config)==0, 0.0,jnp.float64(x))

def positive_random_init(key, shape, dtype=jnp.float64):
    return random.uniform(key, shape, dtype, minval=0.8, maxval=1.2)

def create_model(rng, input_shape, num_electrons, hidden_layer_sizes,    
                 activation, n_bf_dets): 
    model = Backflow(num_orbital=input_shape, num_electron=num_electrons,  n_bf_dets= n_bf_dets,
                     hidden_layer_sizes=hidden_layer_sizes,
                     activation=activation)
    initial=jnp.concatenate((jnp.ones(num_electrons,dtype=jnp.int8),
                            jnp.zeros(input_shape-num_electrons,dtype=jnp.int8)),
                            axis=0)
    initial=jnp.reshape(initial,(1,input_shape))
    variables = model.init(rng,initial)#initializer=normal 
    return model, variables


class Electron_Backflow(nn.Module):

    num_orbital: int
    num_electron: int
    num_alpha_electron: int
    num_beta_electron: int
    n_bf_dets: int 
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
            features=size, kernel_init=positive_random_init, dtype=jnp.float64
            ) for size in self.hidden_layer_sizes]
        self.dense_general = nn.DenseGeneral(
            features=(self.n_bf_dets,self.num_electron, self.num_electron), dtype=jnp.float64
            )
        self.bf_det_params = self.param('bf_det_params', nn.initializers.lecun_normal(), (self.n_bf_dets,1))
        
        

    @partial(jax.vmap, in_axes=(None,0))
    def __call__(self, x):
        # Input x is a 1D array of size num_orbital
        # Need to select the electron positions from the input
        # and then pass it through the backflow transformation

        
        selected_config_alpha = jnp.nonzero(x[:self.num_orbital//2], size=self.num_alpha_electron)[0]
        selected_config_beta = jnp.nonzero(x[self.num_orbital//2:], size=self.num_beta_electron)[0]
        selected_config = jnp.concatenate((selected_config_alpha,selected_config_beta),axis=0)

        x = selected_config
        for dense_layer in self.hidden_dense:
            x = dense_layer(x)
            x = self.activation_fn(x)
        
        x = self.dense_general(x)
        #x = x[:,selected_config,:]
        #x = jnp.linalg.vecdot( self.bf_det_params ,x, axis=0)
        # sgn, val = jnp.linalg.slogdet(x)
        # x = sgn * jnp.exp(val)
        x = jnp.linalg.det(x[:, :self.num_alpha_electron, :self.num_alpha_electron]) * jnp.linalg.det(x[:, self.num_alpha_electron:, self.num_alpha_electron:])
        x = jnp.dot(x,self.bf_det_params)[0]
        return jax.lax.select(jnp.sum(selected_config)==0, 0.0,jnp.float64(x))

def create_model_electron_bf(rng, input_shape, num_alpha_electron, num_beta_electron, hidden_layer_sizes=[4],    
                 activation='relu', n_bf_dets = 1): 
    num_electrons = num_alpha_electron + num_beta_electron
    model = Electron_Backflow(num_orbital=input_shape, num_electron=num_electrons, 
                     num_alpha_electron = num_alpha_electron,
                     num_beta_electron = num_beta_electron,
                     n_bf_dets= n_bf_dets,
                     hidden_layer_sizes=hidden_layer_sizes,
                     activation=activation)
    initial=jnp.concatenate((jnp.ones(num_alpha_electron),
                            jnp.zeros(input_shape//2-num_alpha_electron),
                            jnp.ones(num_beta_electron),
                            jnp.zeros(input_shape//2-num_beta_electron)),
                            axis=0)
    initial=jnp.reshape(initial,(1,input_shape))
    variables = model.init(rng,initial)#initializer=normal 
    return model, variables
