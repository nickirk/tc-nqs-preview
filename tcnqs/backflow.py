import jax.numpy as jnp
import jax
from flax import linen as nn 
import numpy as np


class BACKFLOW(nn.Module):
    features: Sequence[int] 

    def setup(self):
        self.layers = [nn.Dense(n) for n in self.features]

    @nn.compact
    def __call__(self, x):
        for i, lyr in enumerate(self.layers):
          x = lyr(x)
          if i != len(self.layers) - 1:
            x = nn.relu(x) 
        # TODO: select corresponding rows and columns of the matrix
        x = jnp.linalg.det(x)
        return x