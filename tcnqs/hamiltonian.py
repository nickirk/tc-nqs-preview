import jax.numpy as jnp
import numpy as np

# Potential Issue: if we want to use jit avoid if-else statements below 

# By convention det1 is the bra and det2 is the ket always
class HAMILTONIAN:

    def __init__(self, n_elec, n_orb, h1g, g2e):
        self.n_elec = n_elec
        self.n_orb = n_orb 
        self.h1g = h1g
        self.g2e = g2e
    
    def __call__(self, det1, det2):
        a=self._get_1body(det1, det2) + self._get_2body(det1, det2)
        # print(type(a))
        # if type(a) == np.ndarray :
        #     b=1
        return a
    
    
    def _get_1body(self, det1, det2):
        # compare the two binary arrays() and find out the index where they differ
        
        diff = jnp.bitwise_xor(det1, det2)
        if jnp.sum(diff) > 2:
            return 0.0
        
        if jnp.sum(diff) == 2:
            diff_index = jnp.nonzero(diff)[0]
            i = diff_index[jnp.where(det1[diff_index]==1)][0]
            j = diff_index[jnp.where(det2[diff_index]==1)][0]
            
            return self.h1g[i,j]
        
        if jnp.sum(diff) == 0:
            sum = 0.0
            sum_indices=jnp.where(det1==1)[0]
            for i in sum_indices:
        
                # n_orb is already multiplied by 2 
                # make sure while implememting the code 

                sum += self.h1g[i,i] 
            
            return sum
    
    def _get_2body(self, det1, det2):
        # bitwise_xor operator to find the difference between the two determinants
        
        diff = jnp.bitwise_xor(det1, det2)
        if jnp.sum(diff) > 4:
            return 0.0
        if jnp.sum(diff) == 4:
            diff_index = jnp.nonzero(diff)[0]
            i = diff_index[jnp.where(det1[diff_index]==1)][0]
            k = diff_index[jnp.where(det1[diff_index]==1)][1]
            j = diff_index[jnp.where(det2[diff_index]==1)][0]
            l = diff_index[jnp.where(det2[diff_index]==1)][1]
            
            return self.g2e[i, k, l, j] - self.g2e[i, k, j, l]
        if jnp.sum(diff) == 2:
            i,j= jnp.nonzero(diff)[0]
            
            diff_index = jnp.nonzero(diff)[0]
            k = diff_index[jnp.where(det1[diff_index]==1)][0]
            j = diff_index[jnp.where(det2[diff_index]==1)][0]
            
            sum = 0.0
            common_occupancy=jnp.logical_and(det1,det2).astype(int)
            
            sum_indices=jnp.where(common_occupancy==1)[0]
            for i in sum_indices:
                
                # n_orb is already multiplied by 2 
                # make sure while implememting the code 
                
                sum += self.g2e[k,i, i, j] - self.g2e[k,i, j, i]
            
            return sum
        
        if jnp.sum(diff) == 0:
            sum = 0.0
            sum_indices=jnp.where(det1==1)[0]
            for i in sum_indices:
                for j in sum_indices:
                        # n_orb is already multiplied by 2 
                        # make sure while implememting the code 

                        sum += self.g2e[i,j,j,i] - self.g2e[i,j,i,j]
                        
            return sum/2
            
    
    

