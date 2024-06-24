import jax.numpy as jnp
import numpy as np
from pyscf.fci import cistring

def generate_ci_data(num_orbitals,num_alpha_electrons,num_beta_electrons,ci):
    x=[]
    y=[]
    for i in range(ci.shape[0]):
        for j in range(ci.shape[1]):
            if np.abs(ci[i,j]) < 1e-5:
                ci[i,j]=0
                #continue
            y.append(ci[i,j])
            orba=cistring.addr2str(num_orbitals,num_alpha_electrons,i)
            orbb=cistring.addr2str(num_orbitals,num_beta_electrons,j)
            
            orba=convert_binary_to_array(orba,num_orbitals)
            orbb=convert_binary_to_array(orbb,num_orbitals)
            x.append(np.concatenate((orba,orbb),axis=0))
    x=jnp.array(x)
    y=jnp.array(y)
    return x,y

def convert_binary_to_array(str_int, num_orbitals):
    binary_str=str(bin(str_int))
    binary_str = binary_str[2:]
    binary_array = [int(bit) for bit in binary_str]
    leading_zeros = num_orbitals - len(binary_array)
    result_array = [0] * leading_zeros + binary_array
    
    return result_array[::-1]