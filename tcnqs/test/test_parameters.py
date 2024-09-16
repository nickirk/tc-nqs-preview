import pyscf 

learning_rate = 0.005
num_epochs = 2000
n_core = 4096
hidden_layer_sizes =[64,64]
mol = pyscf.M(
    atom = 'H 0 0 0;H 0 0 1.0 ;', #  H 0 0 3.0;  H 0 0 4.0 , # H 0 0 3.0; H 0 0 4.0  ,
    basis = 'sto-3g',
    spin = 0,
    charge = 0,
    symmetry = False,
#    unit = 'Ang'
    )
n_bf_dets = 1
