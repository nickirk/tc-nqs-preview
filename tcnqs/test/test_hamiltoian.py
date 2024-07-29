import jax
import jax.numpy as jnp
import numpy as np
import pyscf
from pyscf.tools import fcidump

from tcnqs.fcidump import read_2_spin_orbital_seprated as read2
from tcnqs.hamiltonian import Hamiltonian
from tcnqs.utils import generate_ci_data, build_ham_from_pyscf


def run_fci(mol, myhf):
    cisolver = pyscf.fci.FCI(myhf)
    fci_e_pyscf, ci_vector=cisolver.kernel()
    return fci_e_pyscf, ci_vector

def test_hamiltonian(mol, test=False):
    
    myhf = mol.RHF().run()

    ham = build_ham_from_pyscf(mol, myhf) 

    fci_e_pyscf, ci_vector = run_fci(mol, myhf)
    x_train, y_train = generate_ci_data(ham.n_orb//2, ham.n_elec_a, ham.n_elec_b, ci_vector)
    x_train = jnp.asarray(x_train)
    
    H = np.zeros((len(x_train), len(x_train)), dtype=jnp.float32)
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            H[i,j]=ham(x_train[i], x_train[j])
    if test:            
        fci_e_diagonal = np.sort(np.linalg.eig(H)[0])[0] + ham.e_core
        
        e_hf=H[0,0]+ham.e_core
        print(fci_e_diagonal, fci_e_pyscf) 
        assert jnp.absolute(myhf.e_tot- e_hf) < 1e-6
        print("Success: HF energies match!")
        assert jnp.absolute(fci_e_diagonal-fci_e_pyscf) < 2e-6 
        print("Success: FCI energies match!")
        
        

def test_setup_hci(mol):
    myhf = mol.RHF().run()
    ham = build_ham_from_pyscf(mol, myhf)

    ham.setup_hci()

   
if __name__ == '__main__':
    
    mol = pyscf.M(
    atom = 'H 0 0 0; H 0 0 1.0 ;H 0 0 2;',  
    basis = '321g',
    spin = 1,
    charge = 0,
    symmetry = False
    )
    
    #test_hamiltonian(mol, test=True)
    test_hamiltonian(mol, test=True)

    #mol = pyscf.M(
    #atom = 'H 0 0 0; H 0 0 1.0 ; H 0 0 3; H 0 0 4',  
    #basis = '321g',
    #spin = 0,
    #charge = 0,
    #symmetry = False
    #)
    
    ##test_hamiltonian(mol, test=True)
    #test_hamiltonian_jit(mol, test=True)