import pyscf 

learning_rate = 0.1
num_epochs = 20
n_core = 200
hidden_layer_sizes =[4,4]
mol = pyscf.M(
    atom = 'Li 0 0 0; F 0 0 1.0  ', #  H 0 0 3.0;  H 0 0 4.0 , # H 0 0 3.0; H 0 0 4.0  ,
    basis = 'sto-3g',
    spin = 0,
    charge = 0,
    symmetry = False
    )
n_bf_dets = 1
