import jax
import jax.numpy as jnp
import numpy as np
import pyscf
from pyscf.tools import fcidump

from tcnqs.fcidump import read_2_spin_orbital_seprated as read2
from tcnqs.hamiltonian import HAMILTONIAN
from tcnqs.hamiltonian_jit import HAMILTONIAN as HAMILTONIAN_JIT
from tcnqs.utils import generate_ci_data


def test_hamiltonian(mol, test=False):
    
    myhf = mol.RHF().run()
    cisolver = pyscf.fci.FCI(myhf)
    num_orbitals = myhf.mo_coeff.shape[1]
    num_alpha_electrons, num_beta_electrons = mol.nelec

    FCI_e_pyscf, ci_vector=cisolver.kernel()
    
    # Read the FCIDUMP file
    fcidump_file = 'tcnqs/test/dataset_fcidump/fcidump'
    fcidump.from_scf(myhf, fcidump_file)
    n_sites, n_elec, ecore, h1e_s, g2e_s = read2(fcidump_file)
    
    # Create FCI Hamiltonian
    hamiltonian = HAMILTONIAN(n_elec, 2*n_sites, h1e_s, g2e_s)
    
    x_train, y_train = generate_ci_data(num_orbitals, num_alpha_electrons, num_beta_electrons, ci_vector)
    
    HHH = hamiltonian(x_train[0], x_train[2])
    
    H= np.zeros((len(x_train), len(x_train)))
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            H[i,j] = hamiltonian(x_train[i], x_train[j])
    
    if test:
        FCI_e_diagonal=np.sort(np.linalg.eig(H)[0])[0] + ecore
        
        e_hf=H[0,0]+ecore
        
        assert jnp.absolute(FCI_e_diagonal-FCI_e_pyscf) < 1e-8 and jnp.absolute(myhf.e_tot- e_hf) < 1e-8
        print("Success: FCI & HF energy matches ")
        
    return H, ecore
    
def test_hamiltonian_jit(mol, test=False):
    
    myhf = mol.RHF().run()
    cisolver = pyscf.fci.FCI(myhf)
    num_orbitals = myhf.mo_coeff.shape[1]
    num_alpha_electrons, num_beta_electrons = mol.nelec
    FCI_e_pyscf, ci_vector=cisolver.kernel()
    
    # Read the FCIDUMP file
    fcidump_file = 'tcnqs/test/dataset_fcidump/fcidump'
    fcidump.from_scf(myhf, fcidump_file)
    n_sites, n_elec, ecore, h1e_s, g2e_s = read2(fcidump_file)
    h1e_s = jnp.asarray(h1e_s)
    g2e_s = jnp.asarray(g2e_s)
    
    # Create FCI Hamiltonian
    hamiltonian = HAMILTONIAN_JIT(n_elec, 2*n_sites, h1e_s, g2e_s)
    #hamiltonian = jax.jit(hamiltonian)
    x_train, y_train = generate_ci_data(num_orbitals, num_alpha_electrons, num_beta_electrons, ci_vector)
    x_train = jnp.asarray(x_train)
    
    
    H = np.zeros((len(x_train), len(x_train)), dtype=jnp.float32)
    for i in range(len(x_train)):
        for j in range(len(x_train)):
            H[i,j]=hamiltonian(x_train[i], x_train[j])
    if test:            
        FCI_e_diagonal=np.sort(np.linalg.eig(H)[0])[0] + ecore
        
        e_hf=H[0,0]+ecore
        
        assert jnp.absolute(FCI_e_diagonal-FCI_e_pyscf) < 2e-6 and jnp.absolute(myhf.e_tot- e_hf) < 1e-6
        print("Success: FCI & HF energy matches ")
        
    return H, ecore

def test_setup_hci(mol):
    myhf = mol.RHF().run()
    num_orbitals = myhf.mo_coeff.shape[1]
    num_alpha_electrons, num_beta_electrons = mol.nelec
    
    # Read the FCIDUMP file
    fcidump_file = 'fcidump'
    # write the fcidump file
    fcidump.from_scf(myhf, fcidump_file)
    n_sites, n_elec, ecore, h1e_s, g2e_s = read2(fcidump_file)
    h1e_s = jnp.asarray(h1e_s)
    g2e_s = jnp.asarray(g2e_s)
    
    # Create FCI Hamiltonian
    hamiltonian = HAMILTONIAN_JIT(n_elec, 2*n_sites, h1e_s, g2e_s)
    hamiltonian.setup_hci()

   
if __name__ == '__main__':
    
    mol = pyscf.M(
    atom = 'H 0 0 0; H 0 0 1.0 ;H 0 0 3; H 0 0 4.0   ',  
    basis = 'sto3g',
    spin = 0,
    charge = 0,
    symmetry = False
    )
    
    #test_hamiltonian(mol, test=True)
    #test_hamiltonian_jit(mol, test=True)
    test_setup_hci(mol)