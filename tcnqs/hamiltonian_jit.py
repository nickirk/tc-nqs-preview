import jax.numpy as jnp
from jax import jit, lax
import jax

# Potential Issue: not compitable with different number of electrons in alpha and beta seprated orbitals

class HAMILTONIAN:
    # n_orb is the number of orbitals in the alpha and beta seprated orbitals
    def __init__(self, n_elec, n_orb, h1g, g2e):
        self.n_elec = n_elec
        self.n_orb = n_orb 
        self.h1g = h1g
        self.g2e = g2e
    
    # By convention det1 is the bra and det2 is the ket always
    def __call__(self, det1, det2):
        return self._get_1body(det1, det2)  + self._get_2body(det1, det2)
    
    # Potential Issue: Only for even number of electrons in the alpha and beta seprated orbitals both
    def phase(self,det,j):
        sum_result = jnp.sum(jnp.where(jnp.arange(self.n_orb) < j, det, 0))
        return 1 - 2 * (sum_result % 2)
    
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

