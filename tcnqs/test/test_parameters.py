from numpy import save
import pyscf 

learning_rate =0.01
num_epochs = 20
n_core = 4096
batch_size = n_core
hidden_layer_sizes =[16, 16]
is_tc = False
mol = pyscf.M(
    atom = 'N 0 0 0 ; N 0 0 1 ', #  H 0 0 3.0;  H 0 0 4.0 , # H 0 0 3.0; H 0 0 4.0  ,
    basis = 'sto-3g',
    
    spin = 0,
    charge = 0,
    symmetry = False,
#    unit = 'Ang'
    )
n_bf_dets = 1
n_eig_projections = 50
save = False
