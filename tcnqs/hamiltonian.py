import jax.numpy as jnp
from jax import jit, lax
import jax
from jax.tree_util import register_pytree_node_class
from functools import partial

# For documentation to Jax PyTrees:
# https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees

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
        
        self.max_g2e, self.sorted_g2e, self.sorted_inds = self.setup_hci()

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
    # @jax.jit
    def __call__(self, det1, det2):
        det1 = jnp.asarray(det1, dtype=jnp.int8)
        det2 = jnp.asarray(det2, dtype=jnp.int8)
        return self._get_1body(det1, det2)  + self._get_2body(det1, det2) 
    
    # @jax.jit
    def phase(self,det,j):
        sum_result = jnp.cumsum(det)[j-1]
        # sum_result = jnp.sum(det[:j-1])
        # return sum_result
        return 1 - 2 * (sum_result % 2)
    
    @jax.jit
    def _get_1body(self, det1, det2):
        diff = jnp.bitwise_xor(det1, det2)
        num_diff = jnp.sum(diff,dtype=jnp.int8)

        def diff_0():
            # sum_indices = jnp.where(det1 == jnp.ones_like(det1), size=self.n_elec)[0]
            sum_indices = jnp.nonzero(det1 , size=self.n_elec)[0]
            return jnp.sum(self.h1g[sum_indices, sum_indices]) + self.e_core

        def diff_2():
            # diff_index = jnp.nonzero(diff, size=2)[0]
            # diff_index = jnp.where(diff == jnp.ones_like(diff), size=2)[0]
            diff_index = jnp.nonzero(diff, size=2)[0]
            i = diff_index[jnp.nonzero(det1[diff_index], size=1)][0]
            j = diff_index[jnp.nonzero(det2[diff_index], size=1)][0]
            phase_global=self.phase(det1,i) * self.phase(det2,j)
            return (phase_global * self.h1g[i,j])
        

        return jax.lax.cond(num_diff == 2, 
                                diff_2,
                        lambda : lax.cond(
                        num_diff == 0, diff_0, lambda : 0.0))

    @jax.jit
    def _get_2body(self, det1, det2):
        diff = jnp.bitwise_xor(det1, det2)
        num_diff = jnp.sum(diff, dtype=jnp.int8)

        def diff_0():
            # sum_indices = jnp.where(det1 == jnp.ones_like(det1), size=self.n_elec)[0]
            sum_indices = jnp.nonzero(det1 , size=self.n_elec)[0]
            # i, j = jnp.meshgrid(sum_indices, sum_indices, indexing='ij')
            # i = sum_indices[:, None]  # Column vector (n_elec, 1)
            # j = sum_indices[None, :]
            # sum_result = jnp.sum(self.g2e[i, j, j, i] - self.g2e[i, j, i, j])    
            sum_result = jnp.sum(self.g2e[sum_indices[:, None], sum_indices, sum_indices, sum_indices[:, None]] - 
                     self.g2e[sum_indices[:, None], sum_indices, sum_indices[:, None], sum_indices]) 
            return sum_result/2

        def diff_2():
            #diff_index = jnp.nonzero(diff, size=2)[0]
            # diff_index = jnp.where(diff == jnp.ones_like(diff), size=2)[0]
            diff_index = jnp.nonzero(diff, size=2)[0]
            k = diff_index[jnp.where(det1[diff_index]==jnp.ones_like(det1[diff_index]),size=1)][0]
            j = diff_index[jnp.where(det2[diff_index]==jnp.ones_like(det2[diff_index]), size=1)][0]
            
            sum_indices = jnp.where(jnp.logical_and(det1, det2),size=self.n_elec-1)[0]
            
            phase_global=self.phase(det1,k) * self.phase(det2,j)
            
            return phase_global*jnp.sum(self.g2e[k, sum_indices, sum_indices, j] - self.g2e[k, sum_indices, j, sum_indices])

        def diff_4():
            # diff_index = jnp.nonzero(diff,size=4)[0]
            # diff_index = jnp.where(diff == jnp.ones_like(diff), size=4)[0]
            diff_index = jnp.nonzero(diff, size=4)[0]
            det1_indices = diff_index[jnp.where(det1[diff_index]==jnp.ones_like(det2[diff_index]), size=2)]
            det2_indices = diff_index[jnp.where(det2[diff_index]==jnp.ones_like(det2[diff_index]), size=2)]
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
        # get the maximum stored elements using the first column of sorted_elements
        max_element = jnp.max(jnp.abs(sorted_elements[:, 0]))


        return max_element, sorted_elements, sorted_indices