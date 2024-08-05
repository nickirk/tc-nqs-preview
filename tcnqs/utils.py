import jax.numpy as jnp
import numpy as np
from pyscf.fci import cistring
from pyscf.tools import fcidump
from tcnqs.hamiltonian import Hamiltonian
from tcnqs.fcidump import read_2_spin_orbital_seprated as read2

def generate_ci_data(num_orbitals, num_alpha_electrons, num_beta_electrons, ci):
    """
    Generate data for configuration interaction (CI) calculations.

    Parameters
    ----------
    num_orbitals : int
        Number of spatial orbitals.

    num_alpha_electrons : int
        Number of alpha electrons.

    num_beta_electrons : int
        Number of beta electrons.

    ci : numpy.ndarray
        Array containing the CI coefficients.

    Returns
    -------
    x : jax.numpy.ndarray
        Array of input features for the CI data.

    y : jax.numpy.ndarray
        Array of target values for the CI data.
    """
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
    """
    Convert a binary string representation of an integer to an array of bits.

    Parameters:
    str_int (int): The binary string representation of the integer.
    num_orbitals (int): The number of orbitals.

    Returns:
    list: The array of bits, with leading zeros if necessary.
    """
    
    binary_str=str(bin(str_int))
    binary_str = binary_str[2:]
    binary_array = [int(bit) for bit in binary_str]
    leading_zeros = num_orbitals - len(binary_array)
    result_array = [0] * leading_zeros + binary_array
    
    return result_array[::-1]


def build_ham_from_pyscf(mol, myhf): 
    # Read the FCIDUMP file
    #fcidump_file = 'tcnqs/test/dataset_fcidump/fcidump'
    fcidump_file = './fcidump'
    fcidump.from_scf(myhf, fcidump_file)
    n_sites, n_elec, ecore, h1e_s, g2e_s = read2(fcidump_file)
    n_elec_a, n_elec_b = mol.nelec
    h1e_s = jnp.asarray(h1e_s)
    g2e_s = jnp.asarray(g2e_s)
    
    # Create FCI Hamiltonian
    hamiltonian = Hamiltonian(n_elec_a, n_elec_b, 2*n_sites, ecore, h1e_s, g2e_s)
    return hamiltonian