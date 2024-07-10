import jax.numpy as jnp
from jax import jit, lax

# Work in progress 

class HAMILTONIAN:

    def __init__(self, n_elec, n_orb, h1g, g2e):
        self.n_elec = n_elec
        self.n_orb = n_orb 
        self.h1g = h1g
        self.g2e = g2e
    
    def phase(self, det , j):
        sliced_det = lax.dynamic_slice(det, (0,), (j,))
        return 1 - 2 * (jnp.sum(sliced_det) % 2)
    
    def __call__(self, det1, det2):
        return self._get_1body(det1, det2) + self._get_2body(det1, det2)
    
    
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
            return self.phase(det1,i) * self.phase(det2,j)*self.h1g[i,j]

        return lax.cond(num_diff == 2, 
                        diff_2,
                        lambda _: lax.cond(num_diff == 0, diff_0, lambda _: 0.0))

    
    def _get_2body(self, det1, det2):
        diff = jnp.bitwise_xor(det1, det2)
        num_diff = jnp.sum(diff)

        def diff_0():
            sum_indices = jnp.where(det1 == 1, size=self.n_elec)[0]
            return jnp.sum((self.g2e[sum_indices[:, None], sum_indices, sum_indices, sum_indices] - 
                            self.g2e[sum_indices[:, None], sum_indices, sum_indices, sum_indices]))/2

        def diff_2():
            diff_index = jnp.nonzero(diff)[0]
            k = diff_index[jnp.where(det1[diff_index]==1)][0]
            j = diff_index[jnp.where(det2[diff_index]==1)][0]
            
            sum_indices = jnp.where(jnp.logical_and(det1, det2),size=self.n_elec-1)[0]
            return jnp.sum(self.g2e[k, sum_indices, sum_indices, j] - self.g2e[k, sum_indices, j, sum_indices])

        def diff_4():
            diff_index = jnp.nonzero(diff,size=4)[0]
            det1_indices = diff_index[jnp.where(det1[diff_index]==1, size=2)]
            det2_indices = diff_index[jnp.where(det2[diff_index]==1, size=2)]
            i = det1_indices[0]
            k = det1_indices[1]
            j = det2_indices[0]
            l = det2_indices[1]
            
            phase_global = self.phase(det1,i)*self.phase(det1,k)*self.phase(det2,j)*self.phase(det2,l)
            
            
            return phase_global*(self.g2e[i, k, l, j] - self.g2e[i, k, j, l])

        return lax.cond(num_diff == 4, 
                        diff_4,
                        lambda _: lax.cond(num_diff == 2, diff_2, 
                                           lambda _: lax.cond(num_diff == 0, diff_0, lambda _: 0.0)))
