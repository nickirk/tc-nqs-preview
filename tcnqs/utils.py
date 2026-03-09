import os

import time
try:
    import wandb
except ImportError:
    wandb = None

import jax.numpy as jnp
import numpy as np
from pyscf.fci import cistring
import pyscf
from pyscf.tools import fcidump
from tcnqs.hamiltonian import Hamiltonian
from tcnqs.fcidump import read_2_spin_orbital_seprated as read2

from pytc.xtc import XTC
from pytc.jastrow import REXP
from pytc.utils import fcidump as fcidump_pytc
from pytc.optimize import optimize_jastrow
import pytc.xtc as xtc


def generate_ci_data(num_orbitals:int, num_alpha_electrons:int, num_beta_electrons:int, ci_vector):
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
    for i in range(ci_vector.shape[0]):
        for j in range(ci_vector.shape[1]):
            if np.abs(ci_vector[i,j]) < 1e-5:
                ci_vector[i,j]=0
                #continue
            y.append(ci_vector[i,j])
            orba=cistring.addr2str(num_orbitals,num_alpha_electrons,i)
            orbb=cistring.addr2str(num_orbitals,num_beta_electrons,j)
            
            orba=convert_binary_to_array(orba,num_orbitals)
            orbb=convert_binary_to_array(orbb,num_orbitals)
            x.append(np.concatenate((orba,orbb),axis=0))
    x=jnp.array(x)
    y=jnp.array(y)
    return x,y

def convert_binary_to_array(str_int:int, num_orbitals:int):
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

def generate_fci_dump(myhf, filename:str, is_tc:bool):
    """
    Generate an FCIDUMP file for a molecule.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule for which to generate the FCIDUMP file.

    filename : str
        Name of the FCIDUMP file.

    is_tc : bool
        Whether to generate the FCIDUMP file for a two-component calculation.
    """

    if not is_tc:
        fcidump.from_scf(myhf, filename)
    else:
        mol = myhf.mol
        my_jastrow = REXP()
        #my_jastrow = REXP(np.array([1.]))
        my_xtc = XTC.from_pyscf(myhf, my_jastrow, grid_lvl=2)
        
        h1e_xtc = my_xtc.get_1b(jastrow_params=my_jastrow.init_params())
        h2e_xtc = my_xtc.get_2b(jastrow_params=my_jastrow.init_params())
        ecore_xtc = my_xtc.get_const(jastrow_params=my_jastrow.init_params())
        n_orb_xtc = h1e_xtc.shape[0]
        n_elec_xtc = mol.nelectron
        
        fcidump_pytc.write(filename, h1e_xtc, h2e_xtc, ecore_xtc, n_orb_xtc, n_elec_xtc)
def optimize_rexp(mol, mf):
    mol.basis = 'ccpvTz'
    print(mol.basis)
    mf = mol.RHF().run()

    init_params = jnp.array([0.5], dtype=jnp.float64)
    my_jastrow = REXP()  # Remove params from constructor
    
    # Run optimization with smaller learning rate
    myxtc = xtc.XTC.from_pyscf(mf, my_jastrow, grid_lvl=2)
   
    optimized_params = optimize_jastrow(myxtc, mf, init_params,
                                          optimizer_name='rmsprop',
                                          learning_rate=1e-2,
                                          n_steps=20)
    print(f"optimized parameters:", optimized_params)
    return optimized_params
    
def generate_fci_dump_temp(myhf, filename:str, is_tc:bool, require_params: bool =True):
    """
    Generate an FCIDUMP file for a molecule.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule for which to generate the FCIDUMP file.

    filename : str
        Name of the FCIDUMP file.

    is_tc : bool
        Whether to generate the FCIDUMP file for a two-component calculation.
    """

    if not is_tc:
        fcidump.from_scf(myhf, filename)
    else:
        mol = myhf.mol
        
        if require_params:
            optimized_params = optimize_rexp(mol,myhf)
            # my_jastrow = REXP(optimized_params)
        my_jastrow = REXP()
        my_xtc = xtc.XTC.from_pyscf(myhf, my_jastrow, grid_lvl=2)
        
        h1e_xtc = my_xtc.get_1b(jastrow_params=optimized_params)
        h2e_xtc = my_xtc.get_2b(jastrow_params=optimized_params)
        ecore_xtc = my_xtc.get_const(jastrow_params=optimized_params)
        n_orb_xtc = h1e_xtc.shape[0]
        n_elec_xtc = mol.nelectron
        
        fcidump_pytc.write(filename, h1e_xtc, h2e_xtc, ecore_xtc, n_orb_xtc, n_elec_xtc)

def build_ham_from_pyscf(mol, myhf, is_tc= False): 
    # Read the FCIDUMP file
    #fcidump_file = 'tcnqs/test/dataset_fcidump/fcidump'
    # fcidump_file = './fcidump'
    if is_tc:
        save_path = f"tcnqs/simulations/fcidump_tc/{mol.atom_symbol(0)}/{mol.atom}/{mol.basis}/"
        os.makedirs(save_path, exist_ok=True)
        fcidump_file = save_path + 'fcidump'
    else:
        save_path = f"tcnqs/simulations/fcidump/{mol.atom_symbol(0)}/{mol.atom}/{mol.basis}/"
        os.makedirs(save_path, exist_ok=True)
        fcidump_file = save_path + 'fcidump'

    if not os.path.exists(fcidump_file):
        generate_fci_dump(myhf, fcidump_file, is_tc)
    
    n_sites, n_elec, ecore, h1e_s, g2e_s = read2(fcidump_file,is_tc=is_tc)

    n_elec_a, n_elec_b = mol.nelec
    h1e_s = jnp.asarray(h1e_s, dtype=jnp.float64)
    g2e_s = jnp.asarray(g2e_s, dtype=jnp.float64)
    # Create FCI Hamiltonian
    hamiltonian = Hamiltonian(n_elec_a, n_elec_b, 2*n_sites, ecore, h1e_s, g2e_s , is_tc=is_tc)
    return hamiltonian

def wandb_init(mol, t_params):
    """Initialize a wandb run if library present. Returns run or None.

    Run name encodes system + key training hyperparameters for sweep grouping.
    """
    if wandb is None:
        print("WandB library not found. Skipping initialization.")
        return None

    config = {
        'learning_rate': t_params.learning_rate,
        'num_epochs': t_params.num_epochs,
        'n_core': t_params.n_core,
        'n_batch': getattr(t_params, 'n_batch', t_params.n_core),
        'hidden_layer_sizes': t_params.hidden_layer_sizes,
        'n_bf_dets': t_params.n_bf_dets,
        'n_eig_projections': t_params.n_eig_projections,
        'is_tc': t_params.is_tc,
        'save': t_params.save,
        'basis': getattr(mol, 'basis', None),
        'atom_spec': getattr(mol, 'atom', None),
    }

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    atom_compact = str(config['atom_spec']).replace(' ', '').replace(';', '_')
    run_name = (
        f"{atom_compact}_basis={config['basis']}_lr={t_params.learning_rate}_"
        f"ncore={t_params.n_core}_{timestamp}"
    )

    tags = [
        f"basis:{config['basis']}",
        f"ncore:{t_params.n_core}",
        f"lr:{t_params.learning_rate}",
        f"hl:{'x'.join(map(str, t_params.hidden_layer_sizes))}",
        f"is_tc:{int(t_params.is_tc)}",
        f"n_bf_dets:{t_params.n_bf_dets}"
    ]

    run = wandb.init(
        project='final-runs-tcnqs',
        name=run_name,
        tags=tags,
        config=config,
        reinit=True,
        mode=os.environ.get('WANDB_MODE', 'online'),
    )
    return run


def wandb_log_energy(run, energy, epoch, e_fci_pyscf=0.0):
    if run is None:
        return

    energy_diff = float(energy) - e_fci_pyscf
    run.log({'epoch': epoch + 1, 'energy': float(energy), 'energy_diff': energy_diff})
