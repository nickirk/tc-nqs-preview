import pyscf 

learning_rate =0.01
num_epochs = 1500
n_core = 128
batch_size = n_core
hidden_layer_sizes =[2]
is_tc = True
mol = pyscf.M(
    atom = 'He 0 0 0 ; ', #  H 0 0 3.0;  H 0 0 4.0 , # H 0 0 3.0; H 0 0 4.0  ,
    basis = 'ccpvdz',
    spin = 0,
    charge = 0,
    symmetry = False,
#    unit = 'Ang'
    )
n_bf_dets = 1
n_eig_projections = 50
