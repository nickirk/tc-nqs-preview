import pyscf 

learning_rate = (0.01,0.005,800) #(0.3,0.01,500)
num_epochs = 12000
n_core = 196#4096*4
n_batch = n_core
hidden_layer_sizes =[4,4]

mol = pyscf.M(
    atom = 'He 0 0 0; ' , #  H 0 0 3.0;  H 0 0 4.0 , # H 0 0 3.0; H 0 0 4.0  ,
    basis = 'ccpvdz',
    
    spin = 0,
    charge = 0,
    symmetry = False,
#    unit = 'Ang'
    )
n_bf_dets = 1
n_eig_projections = 50
save = 0 #True  
is_tc = 0 #True

# learning_rate = (0.3,0.01,500)
# num_epochs = 20
# n_core = 200
# n_batch = n_core
# hidden_layer_sizes =[2,2]

# mol = pyscf.M(
#     atom = 'Li 0 0 0 '  H 0 0 3.0, #  H 0 0 3.0;  H 0 0 4.0 , # H 0 0 3.0; H 0 0 4.0  ,
#     basis = 'sto3g',
    
#     spin = 0,
#     charge = 0,
#     symmetry = False,
# #    unit = 'Ang'
#     )
# n_bf_dets = 1
# n_eig_projections = 50
# save = False#True  
# is_tc = False#True
