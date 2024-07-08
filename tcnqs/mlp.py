import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import List
from dataclasses import field
class MLP(nn.Module):
    
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
            features=size
            ) for size in self.hidden_layer_sizes]
        self.output_layer = nn.Dense(features=1)

    def __call__(self, x):
        for dense_layer in self.hidden_dense:
            x = dense_layer(x)
            x = self.activation_fn(x)
        
        x = self.output_layer(x) 
        x = x.reshape(-1) 
        return x             


def create_model(rng, input_shape, hidden_layer_sizes=[4, 4], activation='relu'):
    model = MLP(hidden_layer_sizes=hidden_layer_sizes, activation=activation)
    variables = model.init(rng, jnp.ones((input_shape,)))
    return model, variables

# Define the mean squared error (MSE) loss function
def mse_loss(params, apply_fn, x, y):
    preds = apply_fn({'params': params}, x)
    #preds_value = jax.device_get(preds).item()
    #y_value = jax.device_get(y).item()
    #print(preds_value, y_value)
    # preds = preds/ jnp.linalg.norm(preds)
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


