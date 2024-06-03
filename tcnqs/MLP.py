import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from jax import random
import matplotlib.pyplot as plt

class NN1(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)  
        x = nn.relu(x)
        x = nn.Dense(features=1)(x) 
        x = nn.tanh(x)    
        print(x.shape)
        return x             


def create_model(rng, input_shape):
    model = NN1()
    variables = model.init(rng, jnp.ones(input_shape))
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
        return loss
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_fn(state.params)


def create_train_state(rng, model, variables):
    tx = optax.adam(learning_rate=0.001)
    return train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)


