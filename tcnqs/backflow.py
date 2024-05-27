import jax.numpy as jnp
import jax
from flax import linen as nn 
import numpy as np


class BACKFLOW(nn.Module):
    # number of electrons
    n_e: int
    # number of total orbitals 
    n_orb: int
    # number of slater determinants
    n_sd: int = 1

    @nn.compact
    def __call__(self, x, params):
        if n_e > n_orb:
            raise ValueError("Number of electrons should be less than or equal to number of orbitals")
        x = nn.Dense(features=2*n_orb)(x)
        x = nn.Dense(features=10)(x)
        x = nn.relu(x)
        x = nn.Dense(features=(n_orb, n_orb))(x)
        return x