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

    def eloc(self, hamiltonian: Hamiltonian, apply_fn) -> jnp.ndarray:
        eloc = jax.vmap(hamiltonian.hamiltonian_and_connections, in_axes =(0,None))(self.core_space, apply_fn)
        return eloc
    
    def sampling_random(self,ci_core,rng):
        n_samples = self.core_space.shape[0]
        n_fixed = int(self.fraction_fixed_samples * n_samples)

        # Sort the b-values in ci_core in descending order and select indices to keep fixed.
        sorted_indices = jnp.argsort(jnp.abs(ci_core),descending=True)
        fixed_indices = sorted_indices[:n_fixed]
        random_indices = sorted_indices[n_fixed:]

        # Keep the fixed determinants and generate new random excitations for the others.
        # fixed_dets = self.core_space[fixed_indices]

        # Vmapped # Make sure to conserve particles
        new_random = self.generate_random_excitations(self.core_space[random_indices], rng)
        

        # Replace the non‐fixed determinants with the newly generated ones.
        updated_core = self.core_space.at[random_indices].set(new_random)
        return updated_core

    def energy_fn(self, hamiltonian: Hamiltonian, state):
        return 0
    
    @partial(jax.vmap, in_axes=(0, None))
    def generate_random_excitations(self, det, rng):
        return 0
    
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