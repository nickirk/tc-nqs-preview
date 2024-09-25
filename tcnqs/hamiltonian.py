import jax.numpy as jnp
from jax import jit, lax
import jax
from jax.tree_util import register_pytree_node_class
from functools import partial

## WARNING: Using @partial for self to be static_argnums:- Once the hamiltonian is created, it is fixed and does not change
## refer Strategy 2 in https://jax.readthedocs.io/en/latest/faq.html 

@register_pytree_node_class
class Hamiltonian:
    def __init__(self, n_elec_a: int, n_elec_b: int, 
                 n_orb: int, e_core: float, h1g: jnp.array, g2e: jnp.array) -> None:
        # n_elec is total number of alpha and beta electrons
        self.n_elec = n_elec_a + n_elec_b
        self.n_elec_a = n_elec_a
        self.n_elec_b = n_elec_b
        # n_orb is the number of spin orbitals, first half are alpha and second half are beta
        self.n_orb = n_orb 
        self.h1g = h1g
        self.g2e = g2e
        self.e_core = e_core
        
        self.sorted_g2e, self.sorted_inds = self.setup_hci()

    def tree_flatten(self):
    # Return the dynamic fields and static fields separately
        dynamic = ()  # Include any fields that should be transformed (for example, if any mutable arrays)
        static = (self.n_elec_a, self.n_elec_b, self.n_orb, self.e_core,self.h1g, self.g2e,self.n_elec,self.sorted_g2e,self.sorted_inds)
        return dynamic, static

    @classmethod
    def tree_unflatten(cls, static, dynamic):
        # Reconstruct the class with static fields, dynamic can be ignored if empty
        instance = cls(*static[:6])
        instance.n_elec = static[6]
        instance.sorted_g2e , instance.sorted_inds = static[7:]

        return instance
    # By convention det1 is the bra and det2 is the ket always
    # @partial(jit, static_argnums=(0))
    @jax.jit
    def __call__(self, det1, det2):
        det1 = jnp.asarray(det1, dtype=jnp.int8)
        det2 = jnp.asarray(det2, dtype=jnp.int8)
        return self._get_1body(det1, det2)  + self._get_2body(det1, det2)
    
    # Potential Issue: Only for even number of electrons in the alpha and beta seprated orbitals both
    def phase(self,det,j):
        sum_result = jnp.sum(jnp.where(jnp.arange(self.n_orb) < j, det, 0))
        return 1 - 2 * (sum_result % 2)
    
    #@partial(jit, static_argnums=(0))
    def _get_1body(self, det1, det2):
        diff = jnp.bitwise_xor(det1, det2)
        num_diff = jnp.sum(diff)

        def diff_0():
            sum_indices = jnp.where(det1 == 1, size=self.n_elec)[0]
            return jnp.sum(self.h1g[sum_indices, sum_indices])

        def diff_2():
            diff_index = jnp.nonzero(diff, size=2)[0]
            i = diff_index[jnp.where(det1[diff_index]==1, size=1)][0]
            j = diff_index[jnp.where(det2[diff_index]==1, size=1)][0]
            phase_global=self.phase(det1,i) * self.phase(det2,j)
            return (phase_global * self.h1g[i,j])
        

        return jax.lax.cond(num_diff == 2, 
                                diff_2,
                        lambda : lax.cond(
                        num_diff == 0, diff_0, lambda : 0.0))

    #@partial(jit, static_argnums=(0))
    def _get_2body(self, det1, det2):
        diff = jnp.bitwise_xor(det1, det2)
        num_diff = jnp.sum(diff)

        def diff_0():
            sum_indices = jnp.where(det1 == 1, size=self.n_elec)[0]
            i, j = jnp.meshgrid(sum_indices, sum_indices, indexing='ij')
            sum_result = jnp.sum(self.g2e[i, j, j, i] - self.g2e[i, j, i, j])     
            return sum_result/2

        def diff_2():
            diff_index = jnp.nonzero(diff, size=2)[0]
            k = diff_index[jnp.where(det1[diff_index]==1,size=1)][0]
            j = diff_index[jnp.where(det2[diff_index]==1, size=1)][0]
            
            sum_indices = jnp.where(jnp.logical_and(det1, det2),size=self.n_elec-1)[0]
            
            phase_global=self.phase(det1,k) * self.phase(det2,j)
            
            return phase_global*jnp.sum(self.g2e[k, sum_indices, sum_indices, j] - self.g2e[k, sum_indices, j, sum_indices])

        def diff_4():
            diff_index = jnp.nonzero(diff,size=4)[0]
            det1_indices = diff_index[jnp.where(det1[diff_index]==1, size=2)]
            det2_indices = diff_index[jnp.where(det2[diff_index]==1, size=2)]
            i = det1_indices[0]
            k = det1_indices[1]
            j = det2_indices[0]
            l = det2_indices[1]
            
            phase_global = self.phase(det1,i)*self.phase(det1,k)*self.phase(det2,j)*self.phase(det2,l)
            
            #print(phase_global==1)
            
            return phase_global*(self.g2e[i, k, l, j] - self.g2e[i, k, j, l])

        return jax.lax.cond(num_diff == 4, 
                        diff_4,
                        lambda : jax.lax.cond(num_diff == 2, diff_2, 
                                           lambda : jax.lax.cond(num_diff == 0, diff_0, lambda : 0.0)))
    
    def setup_hci(self) -> jnp.ndarray:
        # Generate all the pairs of indices
        # can use jnp.meshgrid
        # pairs = jnp.meshgrid(jnp.arange(self.n_orb), jnp.arange(self.n_orb), indexing='ij')
        # pairs = jnp.array(pairs).reshape(2,-1).T
        pairs = jnp.array([(i, j) for i in range(self.n_orb) for j in range(self.n_orb)])

        def sort_elements(carry, pair):
            i, j = pair
            # dynamically slice the 2-body integrals
            # sort the elements by their absolute values in descending order
            block = self.g2e[i, :, :, j].flatten()
            sorted_inds = jnp.argsort(-jnp.abs(block))
            elements = jnp.take(block, sorted_inds)
            return carry, (elements, sorted_inds)

        # Initialize carry (not used in this case)
        # if carry is not used make use of vmap
        carry = None

        # Use jax.lax.scan to iterate over pairs and collect nonzero elements and indices
        _, results = jax.lax.scan(sort_elements, carry, pairs)

        sorted_elements, sorted_indices = results


        return sorted_elements, sorted_indices