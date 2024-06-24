import numpy as np
import pyscf
from pyscf import gto, scf, fci
from pyscf.tools import fcidump
from tcnqs import fcidump as fd

# Incomplete - check with standard values and then assert 

def generate_fcidump():
    # Define the molecule
    mol = pyscf.M(
    atom = 'H 0 0 0; H 0 0 1.0',  
    basis = 'sto3g',
    spin = 0
    )

    # Perform Hartree-Fock calculation
    myhf = scf.RHF(mol)
    myhf.kernel()

    #
    # Example 1: Convert an SCF object to FCIDUMP
    #
    fcidump_file = 'tcnqs/test/dataset_fcidump/fcidump_H2'
    fcidump.from_scf(myhf, fcidump_file)

    print(f'FCI dump file saved to {fcidump_file}')
    
def test_fcidump():
    generate_fcidump()
    fcidump_file = 'tcnqs/test/dataset_fcidump/fcidump_H2'
    n_sites, n_elec, ecore, h1e, g2e = fd.read(fcidump_file)
    print(n_sites, n_elec, ecore, h1e, g2e)
    
    #Incomplete - check with standard values and then assert 
    
if __name__ == '__main__':
    test_fcidump()