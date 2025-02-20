import pyscf 

learning_rate =0.05
num_epochs = 25000
n_core = 2048
batch_size = n_core
hidden_layer_sizes =[4,8]
is_tc = True
mol = pyscf.M(
    atom = 'Be 0 0 0 ; ', #  H 0 0 3.0;  H 0 0 4.0 , # H 0 0 3.0; H 0 0 4.0  ,
    basis = 'cc-pVTZ',
    
    spin = 0,
    charge = 0,
    symmetry = False,
#    unit = 'Ang'
    )
n_bf_dets = 1
n_eig_projections = 256
