import jax.numpy as jnp
import numpy as np

## Some code that we dont need for Now

class SD:
    """
    Class to create a slater determinant from alpha and beta determinants.
    params:
    det_alpha & det_beta :Takes input as alpha and beta seprated spin orbital values  
    n_orb: includes spin seprated orbital number count
    
    Returns:
    sd:[arr_a_1, arr_a_2,....,arr_b_1, arr_b_2,....] 
    arr_a_1: 32 bit integer array of 1st 32 alpha spin orbitals 
    arr_a_2: 32 bit integer array of next 32 alpha spin orbitals
    arr_b_1: 32 bit integer array of 1st 32 beta spin orbitals
    arr_b_2: 32 bit integer array of next 32 beta spin orbitals
    """
    
    def __init__(self, n_orb, det_alpha, det_beta): 
        dtype = jnp.uint32
        data_digit= 32
        def required_params(n_orb):
            # Division by 64(=2*32) to get the spin individual number of orbitals. 
            n_orb = n_orb/(data_digit*2)
            return jnp.ceil(n_orb).astype(int)
        
        length = required_params(n_orb)
        # print(length)
        def create_sd_rep(length,det):
            sd = np.zeros(length, dtype=dtype)
            for i in range(length):
                sd[i]= det >> data_digit*i & 0xFFFFFFFF
            return jnp.array(sd)
        
        self.det = jnp.concatenate((create_sd_rep(length,det_alpha), create_sd_rep(length,det_beta)))
    
    
    ## Think , test and write : Can create some problems in the future
    # Function that can create slater determinant that can be used in the form inside hamiltonian
    def unpacked_data(self,det,n_orb):
        as_list = [bin(n)[2:] for n in self.det]
        orbitalrep = jnp.array([np.array(list(n.zfill(32))).astype(int)[::-1] for n in as_list])
        orbitalrep=orbitalrep.reshape(-1)
        b=32
        c=jnp.ceil(n_orb/(b*2)).astype(int)
        n_s_orbital = int(n_orb/2)
        #print(c)
        orbitalrep = jnp.concatenate((orbitalrep[:n_s_orbital], orbitalrep[c*b:c*b+n_s_orbital]))
        return orbitalrep.astype(jnp.uint8)
    
    def diff(self, sd1, sd2 , n_orb):
        diff = jnp.bitwise_xor(sd1, sd2)
        return self.unpacked_data( diff , n_orb)
    
    def common(self, sd1, sd2 , n_orb):
        common = jnp.bitwise_andwise_and(sd1, sd2)
        return self.unpacked_data( common , n_orb)
    
    def alpha(self, sd, n_orb):
        return self.unpacked_data(sd[:len(sd)//2] , int(n_orb/2))
    
    def beta(self, sd, n_orb):
        return self.unpacked_data(sd[len(sd)//2:] , int(n_orb/2))
# Size for @jax.jit
def find_nonzero_index(sd, size):
    # return indices where SD or a similar object is 1
    unpacked=jnp.array(list(bin(n)[2:])[::-1] for n in sd).astype(jnp.int8)
    return jnp.nonzero(unpacked,size=size)[0]



    

# def detwise_xor(sd1, sd2):
#     """
#     Function to calculate the XOR of two slater determinants.
#     params:
#     sd1 & sd2 : Takes input as two slater determinants.
    
#     Returns:
#     sd: XOR of two slater determinants.
#     """
#     return jnp.bitwise_xor(sd1, sd2)

# def detwise_and(sd1, sd2):
#     """
#     Function to calculate the AND of two slater determinants.
#     params:
#     sd1 & sd2 : Takes input as two slater determinants.
    
#     Returns:
#     sd: AND of two slater determinants.
#     """
#     return jnp.bitwise_and(sd1, sd2)

