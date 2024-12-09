import pyscf 

learning_rate =0.01
num_epochs = 15000
n_core = 225
batch_size = 225
hidden_layer_sizes =[2]
mol = pyscf.M(
    atom = 'Li 0 0 0;H 0 0 1 ; ', #  H 0 0 3.0;  H 0 0 4.0 , # H 0 0 3.0; H 0 0 4.0  ,
    basis = 'sto-3g',
    spin = 0,
    charge = 0,
    symmetry = False,
#    unit = 'Ang'
    )
n_bf_dets = 1
