import pyscf 

learning_rate = 0.001
num_epochs = 5000
n_core = 4096
batch_size = 512
hidden_layer_sizes =[8,8]
mol = pyscf.M(
    atom = 'Li 0 0 0;Cl 0 0 1 ; ', #  H 0 0 3.0;  H 0 0 4.0 , # H 0 0 3.0; H 0 0 4.0  ,
    basis = 'sto-3g',
    spin = 0,
    charge = 0,
    symmetry = False,
#    unit = 'Ang'
    )
n_bf_dets = 1
