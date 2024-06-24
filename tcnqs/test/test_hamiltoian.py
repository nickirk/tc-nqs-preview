
import jax.numpy as jnp
from tcnqs.fcidump import read_2_spin_orbital_seprated as read2
from tcnqs.hamiltonian import HAMILTONIAN
import pyscf
from tcnqs.utils import generate_ci_data
import numpy as np
from pyscf.tools import fcidump

# Incomlete test: need to solve for ground state energy using obtained hamiltonian 
# and compare with pyscf

def test_hamiltonian():
    
    
    
    
    # Define the molecule
    mol = pyscf.M(
    atom = 'H 0 0 0; H 0 0 1.0; H 0 0 2; H 0 0 3.0',  
    basis = 'sto3g',
    spin = 0
    )
    
    # use assert to check the energies
    # hf= jnp.array([1,0,1,0]) 
    # # ket= jnp.array([1,0,1,0])
    # # bra= jnp.array([0,1,0,1])
    
    # # a=ham(ket,bra)
    # print("hf energy=",ham(hf,hf)+ecore)
    myhf = mol.RHF().run()
    cisolver = pyscf.fci.FCI(myhf)
    
    fcidump_file = 'tcnqs/test/dataset_fcidump/fcidump_H2'
    fcidump.from_scf(myhf, fcidump_file)

    fcidump_file = 'tcnqs/test/dataset_fcidump/fcidump_H2'
    n_sites, n_elec, ecore, h1e_s, g2e_s = read2(fcidump_file)
    
    ham = HAMILTONIAN(n_elec, n_sites, h1e_s, g2e_s)
    
    num_orbitals = myhf.mo_coeff.shape[1]
    num_alpha_electrons, num_beta_electrons = mol.nelec
    cie, ci_vector=cisolver.kernel()
    
    x_train, y_train = generate_ci_data(num_orbitals, num_alpha_electrons, num_beta_electrons, ci_vector)
    print("FCI Energy",cie)
    H= np.zeros((len(x_train), len(x_train)))
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            H[i,j] = ham(x_train[i], x_train[j])
    print("H=",H) 
    #H[0,3]=H[3,0]
    eig=np.sort(np.linalg.eig(H)[0])[0] +ecore
    print("check ref ", myhf.e_tot,H[0,0]+ecore)
    print("check",eig,cie)
    
if __name__ == '__main__':
    test_hamiltonian()