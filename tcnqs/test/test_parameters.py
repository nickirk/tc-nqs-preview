import pyscf 

learning_rate =0.05
num_epochs = 2000
n_core = 200
n_batch = n_core
hidden_layer_sizes =[4,4]

mol = pyscf.M(
    atom = 'Li 0 0 0;H 0 0 1 ' , #  H 0 0 3.0;  H 0 0 4.0 , # H 0 0 3.0; H 0 0 4.0  ,
    basis = 'sto-3g',
    
    spin = 0,
    charge = 0,
    symmetry = False,
#    unit = 'Ang'
    )
n_bf_dets = 1
n_eig_projections = 50
save = False #True  
is_tc = False #True
