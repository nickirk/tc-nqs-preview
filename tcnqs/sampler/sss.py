import jax.numpy as jnp
import jax
from jax.tree_util import register_pytree_node_class
from functools import partial
from .fssc import FSSC
from tcnqs.hamiltonian import Hamiltonian
from tcnqs.sampler.connected_dets import generate_connected_space

@register_pytree_node_class
class SSS(FSSC):
    def __init__(self, n_core: int, n_connected: int, n_elec_a: int, n_elec_b: int, n_spac_orb: int, n_batch: int, fraction_fixed_samples: float) -> None:
        super().__init__(n_core, n_connected, n_elec_a, n_elec_b, n_spac_orb, n_batch)
        self.fraction_fixed_samples = fraction_fixed_samples

    @jax.jit
    def initialize(self):
        return super().initialize()

    def next_sample_stored(self, hamiltonian: Hamiltonian) -> jnp.ndarray:
        
        pass

    def update_core_space(self, new_sample_core):
        return super().update_core_space(new_sample_core)

    def tree_flatten(self):
        # Return the dynamic fields and static fields separately
        dynamic = (self.core_space, self.fraction_fixed_samples)  # Include any fields that should be transformed with jax (for example, mutable arrays)
        static = (self.n_core, self.n_connected, self.n_elec_a, self.n_elec_b, self.n_spac_orb, self.n_batch, self.n_full)
        return dynamic, static

    @classmethod
    def tree_unflatten(cls, static, dynamic):
        instance = cls(*static, dynamic[1])
        instance.core_space = dynamic[0]
        return instance