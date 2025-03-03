import jax
import jax.numpy as jnp
# import numpy as np
import pyscf
# from pyscf.tools import fcidump

from tcnqs.fcidump import read_2_spin_orbital_seprated as read2
# from tcnqs.hamiltonian import Hamiltonian
from tcnqs.utils import generate_ci_data, build_ham_from_pyscf


def run_fci(mol, myhf):
    cisolver = pyscf.fci.FCI(myhf)
    fci_e_pyscf, ci_vector=cisolver.kernel()
    return fci_e_pyscf, ci_vector

def test_hamiltonian(mol, test=False):
    if test:
        jax.config.update("jax_enable_x64", True)
        
    myhf = mol.RHF().run()

    ham = build_ham_from_pyscf(mol, myhf) 

    fci_e_pyscf, ci_vector = run_fci(mol, myhf)
    x_train, y_train = generate_ci_data(ham.n_orb//2, ham.n_elec_a, ham.n_elec_b, ci_vector)
    x_train = jnp.asarray(x_train)
    
    jax_ham = jax.vmap(jax.vmap(ham._hamiltonian_element, in_axes=(0, None)), in_axes=(None, 0))
    H = jax_ham(x_train, x_train)

    if test:            
        fci_e_diagonal = jnp.min(jnp.linalg.eigh(H)[0]) #+ ham.e_core
        
        e_hf=H[0,0]#+ham.e_core
        print(f"Hamiltonian:{fci_e_diagonal}, Pyscf:{fci_e_pyscf} ") 
        assert jnp.absolute(myhf.e_tot- e_hf) < 1e-7
        print("Success: HF energies match!")
        assert jnp.absolute(fci_e_diagonal-fci_e_pyscf) < 1e-7 
        print("Success: FCI energies match!")
        
        

def test_setup_hci(mol):
    myhf = mol.RHF().run()
    ham = build_ham_from_pyscf(mol, myhf)

    ham.setup_hci()

   
if __name__ == '__main__':
    
    mol = pyscf.M(
    atom = 'H 0 0 0; H 0 0 1.0 ;H 0 0 2;',  
    basis = 'sto-3g',
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