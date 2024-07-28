import jax.numpy as jnp
import jax
from . import SAMPLER
from tcnqs.hamiltonian import Hamiltonian
from tcnqs.slater_det import SD

class FSSC(SAMPLER):
    def __init__(self, wfn, ham: Hamiltonian) -> None:
        super().__init__(wfn)
        self.ham = ham
        self.n_core = None
        self.n_connected = None
    
    def sample(self, n_u: int, n_core: int, n_connected: int, init_core: jnp.ndarray) -> jnp.ndarray:
        self.n_u = n_u
        self.n_core = n_core
        self.n_connected = n_connected
        self.init_core = init_core

        # sample connected configurations
        connected = self._sample_connected(init_core, n_connected)
        return jnp.concatenate((init_core, connected), axis=0)
    

    def _sample_connected(self, core: jnp.ndarray, n_conn: int) -> jnp.ndarray:
        def sample_operation(carry, x):
            connected_samples =  jnp.tile(x, (n_conn, 1))
            return carry, connected_samples
        # use jax loop to sample connected configurations
        carry = None
        _, connected = jax.lax.scan(sample_operation, carry, core)

        connected = connected.reshape(-1, self.n_core)
        return connected
