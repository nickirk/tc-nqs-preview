
import jax.numpy as jnp
from tcnqs.fcidump import read_2_spin_orbital_seprated as read2
from hamiltonian import HAMILTONIAN
import pyscf
from tcnqs.utils import generate_ci_data

def test_hamiltonian():
    fcidump_file = 'tcnqs/test/dataset_fcidump/fcidump_H2'
    n_sites, n_elec, ecore, h1e_s, g2e_s = read2(fcidump_file)
    
    ham = HAMILTONIAN(n_elec, n_sites, h1e_s, g2e_s)
    
    
    # Define the molecule
    mol = pyscf.M(
    atom = 'H 0 0 0; H 0 0 1.0',  
    basis = 'sto3g',
    spin = 0
    )
    
    myhf = mol.RHF().run()
    cisolver = pyscf.fci.FCI(myhf)
    
    num_orbitals = 2*myhf.mo_coeff.shape[1]
    num_alpha_electrons, num_beta_electrons = mol.nelec
    cie, ci_vector=cisolver.kernel()
    
    x_train, y_train = generate_ci_data(num_orbitals, num_alpha_electrons, num_beta_electrons, ci_vector)
    print(ham(x_train[0],x_train[0]))
    
if __name__ == '__main__':
    test_hamiltonian()