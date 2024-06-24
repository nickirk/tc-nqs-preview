from itertools import combinations
import jax.numpy as jnp

# Potential Issue: if we want to use jit avoid if-else statements below 

# By convention det1 is the bra and det2 is the ket always
class HAMILTONIAN:

    def __init__(self, n_elec, n_orb, h1g, g2e):
        self.n_elec = n_elec
        self.n_orb = n_orb 
        self.h1g = h1g
        self.g2e = g2e
    
    def __call__(self, det1, det2):
        return self._get_1body(det1, det2) + self._get_2body(det1, det2)
    
    
    def _get_1body(self, det1, det2):
        # compare the two binary arrays() and find out the index where they differ
        
        diff = jnp.bitwise_xor(det1, det2)
        if jnp.sum(diff)%2 > 1:
            return 0.0
        
        if jnp.sum(diff)%2 == 1:
            diff_index = jnp.nonzero(diff)[0]
            i, j = jnp.where(det1[diff_index]==1)[0],jnp.where(det2[diff_index]==1)[0]
            
            return self.h1g[i,j]
        
        if jnp.sum(diff)%2 == 0:
            sum = 0
            for i in range(self.n_orb):
        
                # n_orb is already multiplied by 2 
                # make sure while implememting the code 

                sum += self.h1g[i,i] 
            
            return sum
    
    def _get_2body(self, det1, det2):
        # bitwise_xor operator to find the difference between the two determinants
        
        diff = jnp.bitwise_xor(det1, det2)
        if jnp.sum(diff)%2 > 2:
            return 0.0
        if jnp.sum(diff)%2 == 2:
            diff_index = jnp.nonzero(diff)[0]
            i, k, j, l = jnp.where(det1[diff_index]==1)[0],jnp.where(det2[diff_index]==1)[0]
            
            # Check once again- Getting confused with the indices
            return self.g2e[i, k, j, l] - self.g2e[i, k, l, j]
        if jnp.sum(diff)%2 == 1:
            i,j= jnp.nonzero(diff)[0]
            
            diff_index = jnp.nonzero(diff)[0]
            k, j = jnp.where(det1[diff_index]==1)[0],jnp.where(det2[diff_index]==1)[0]
            
            sum = 0
            for i in range(self.n_orb):
                
                # n_orb is already multiplied by 2 
                # make sure while implememting the code 
                
                sum += self.g2e[k,i, j, i] - self.g2e[k,i, i, j]
            
            return sum
        
        if jnp.sum(diff)%2 == 0:
            sum = 0
            for i in range(self.n_orb):
                for j in range(self.n_orb):
                        # n_orb is already multiplied by 2 
                        # make sure while implememting the code 

                        sum += self.g2e[i,j,i,j] - self.g2e[i,j,j,i]
                        
            return sum/2
            
    
    

