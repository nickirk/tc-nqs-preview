import jax.numpy as jnp
import numpy as np
# abstract class for sampler
# with abstract member functions for sampling and calculating energy
class SAMPLER:
    def __init__(self, wfn) -> None:
        self.wfn = wfn
        # n_u: unique number of configurations
        self.n_u = None
        # n_tot: total number of configurations
        self.n_tot = None
    
    def sample(self, n_u: int) -> jnp.ndarray:
        pass