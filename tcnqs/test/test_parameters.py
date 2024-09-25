import pyscf 

learning_rate = 0.001
num_epochs = 10000
n_core = 50
hidden_layer_sizes =[4,4]
mol = pyscf.M(
    atom = 'Li 0 0 0;H 0 0 1 ;  ', #  H 0 0 3.0;  H 0 0 4.0 , # H 0 0 3.0; H 0 0 4.0  ,
    basis = 'sto-3g',
    spin = 0,
    charge = 0,
    symmetry = False,
#    unit = 'Ang'
    )
n_bf_dets = 1
