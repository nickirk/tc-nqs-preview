import jax
import jax.numpy as jnp
import optax
#from jax import vmap
from flax.training import train_state

from functools import partial
from tcnqs.sampler.connected_dets import generate_connected_space
from tcnqs.sampler.fssc import FSSC
from tcnqs.hamiltonian import Hamiltonian
from tcnqs.test.test_parameters import learning_rate 

@jax.jit
def train_step(state, batch):
    def mse_loss(params, apply_fn, x, y):    
        preds = apply_fn({'params': params}, x)
        #preds_value = jax.device_get(preds).item()
        #y_value = jax.device_get(y).item()
        #print(preds_value, y_value)
        # preds = preds/ jnp.linalg.norm(preds)
        return jnp.mean((preds - y) ** 2)
    
    def loss_fn(params):
        x, y = batch
        loss = mse_loss(params, state.apply_fn, x, y)
        return loss
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_fn(state.params)

#@jax.jit
def train_step_log(state,batch):
    
    def log_loss(params,apply_fn,x,y):
        # this will calculate the batch loss directly for all x,y
        preds = apply_fn({'params': params}, x)
        Norm_preds = jnp.sum(jnp.power(preds,2))
        Norm_y= jnp.sum(jnp.power(y,2))
        
        overlap_coeff = jnp.dot(preds,y)**2# Define jax vector product of preds and y  
        
        return -jnp.log(overlap_coeff/(Norm_y*Norm_preds)) # return the log loss error
    
    def loss_fn(params):
        x, y = batch
        loss = log_loss(params, state.apply_fn, x, y)
        return loss
    # loss_fn = log_loss()
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_fn(state.params)

@jax.jit
def train_step_hamiltonian(state, batch, H: jnp.ndarray): # H is the Hamiltonian matrix
    def hamiltonian_loss(params,apply_fn,x,H):
    
        preds = apply_fn({'params': params}, x)
        Norm_preds = jnp.linalg.norm(preds)**2
        #print(Norm_preds)
        overlap_coeff = jnp.dot(preds,jnp.dot(H,preds))
        #jnp.sum((preds)[:, None] * preds[None, :] * H)
        
        return  overlap_coeff/Norm_preds 

    def loss_fn(params):
        x, y = batch
        loss = hamiltonian_loss(params, state.apply_fn, x, H)
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_fn(state.params)

@partial(jax.jit, static_argnums=(2))
def train_step_connections(state, batch, Hamiltonain): # H is the Hamiltonian class
    """
    Parameters
    ----------
    state : flax.training.train_state.TrainState
        The current state of the training. 
    batch : jnp.ndarray, the full Hilbert space determinants. 
    Hamiltonain : Hamiltonian, the Hamiltonian object.
    """
    def hamiltonian_loss(params, apply_fn, x, Hamiltonain):
        C_i = apply_fn({'params': params}, x)
        #a= apply_fn({'params': params}, jnp.expand_dims(x[5],axis=0))
        Norm = jnp.linalg.norm(C_i)
        # x_connected,C_i_connected=[],[]

        def find_Ci(det):
            condition=jnp.all(x==det,axis=1)
            return jnp.asarray(C_i[jnp.where(condition, size = 1)[0][0]])
            
        def overlap(slater_determinant, C_i):
            connected_space = generate_connected_space(slater_determinant, Hamiltonain.n_elec_a, Hamiltonain.n_elec_b)
            psi_H_xi = jax.vmap(Hamiltonain,in_axes=(None,0))(slater_determinant,connected_space)
            xi_psi = jax.vmap(find_Ci,in_axes=0)(connected_space)
            return C_i*jnp.dot(psi_H_xi,xi_psi)
            
        overlap_coeff = jnp.sum(jax.vmap(overlap, in_axes =(0,0))(x,C_i))
        
        # jax.debug.breakpoint()
        # jax.debug.print("{det}",det=det)
        return overlap_coeff/Norm**2
    
    def loss_fn(params):
        loss = hamiltonian_loss(params, state.apply_fn, batch, Hamiltonain)
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss_fn(state.params)


#Deprecated Function 
#@partial(jax.jit, static_argnums=(2,3))
def _train_step_fssc_old(state, last_sample, Hamiltonain, sampler):
    """
    Parameters
    ----------
    state : flax.training.train_state.TrainState
        The current state of the training.
    last_sample : tuple (jnp.ndarray,jnp.ndarry), the last sample of the Hilbert 
                    space determinants and the corresponding C_i coefficients. 
    Hamiltonain : Hamiltonian, the Hamiltonian object.
    Sampler : FSSC, the FSSC object. 
              ## or a general sampler depending on future implementations  
    """
    
    def hamiltonian_loss(params, last_sample, Hamiltonain,sampler):
        ## Sample is a tuple of x and C_i (x , C_i)
        # C_i = sample[1] 
        #a= apply_fn({'params': params}, jnp.expand_dims(x[5],axis=0))s
        #sampler = FSSC(n_core, hamiltonian.n_elec_a, hamiltonian.n_elec_b, num_orbitals)
        sample = sampler.next_sample(last_sample, params)
        
        Norm = jnp.linalg.norm(sample[1]) #[:sampler.n_core]
        # x_connected,C_i_connected=[],[]
        
        def find_Ci(det):
            condition=jnp.all(sample[0]==det,axis=1)
            return jnp.asarray(sample[1][jnp.where(condition, size = 1)[0][0]])
            ## return jax.lax.cond(jnp.any , jnp.array , 0.0 )
            # Edge case if condition is false everywhere the ci value returned is wrong*
            # Use jax.lax.cond to solve it 
            # Will be used when we select only particular samples

        def overlap(slater_determinant, C_i):
            connected_space = generate_connected_space(slater_determinant,Hamiltonain.n_elec_a, Hamiltonain.n_elec_b)
            psi_H_xi = jax.vmap(Hamiltonain,in_axes=(None,0))(slater_determinant,connected_space)
            xi_psi = jax.vmap(find_Ci,in_axes=0)(connected_space)
            return C_i*jnp.dot(psi_H_xi,xi_psi)
            
        overlap_coeff = jnp.sum(jax.vmap(overlap, in_axes =(0,0))(sample[0][:sampler.n_core],sample[1][:sampler.n_core]))
        
        # jax.debug.breakpoint()
        # jax.debug.print("{det}",det=det)
        return overlap_coeff/Norm**2, sample
    
    def loss_fn(params):
        loss , sample = hamiltonian_loss(params, last_sample, Hamiltonain, sampler)
        return loss
    # loss_fn = lambda params: hamiltonian_loss(params, last_sample, Hamiltonain, sampler)[0]
    grads = jax.grad(loss_fn)(state.params)
    # put nan to 0
    #grads = jax.tree.map(lambda x: jnp.where(jnp.isnan(x), 0., x), grads)
    loss, new_sample = hamiltonian_loss(state.params, last_sample, Hamiltonain, sampler)
    
    state = state.apply_gradients(grads=grads)
    return state, loss, new_sample
    
# Handle Energy calculation
#@partial(jax.jit)
def energy(params,sample,ham_stored,state,sampler):
    # calculate ci coefficients
    # calculate norm
    # claculate sum Psi_H_Psi = sum_ij C_i* C_j* <xi|H|xj> 
    # return sum(Psi_H_Psi)/Norm**2
    
    unique_full,idx = sample
    Ci = state.apply_fn({'params': params},unique_full)
    #Norm = jnp.linalg.norm(Ci)
    next_sample_idx = jnp.argsort(jnp.abs(Ci),descending =True)[:sampler.n_core]
    Ci = Ci[idx]
    ci_core, ci_connected = Ci[:sampler.n_core], Ci[sampler.n_core:].reshape(sampler.n_core,-1)
    Norm = jnp.linalg.norm(ci_core)
    e = jnp.dot(ci_core,jnp.einsum('ij,ij->i', ham_stored,ci_connected))/Norm**2
    
    new_sample_core = unique_full[next_sample_idx]

    #return e,new_sample_core
    ## Try: numerator and denominator seprately 
    return e, new_sample_core
#@partial(jax.jit)
def new_state(state, sample ,ham_stored,sampler):
    #jax.grad = jax.grad(energy) only in first input
    energy_sample ,grads = jax.value_and_grad(energy, argnums=0, has_aux = True)(state.params, sample, ham_stored, state, sampler)
    new_state = state.apply_gradients(grads=grads)
    sampler = sampler.update_core_space(energy_sample[1])
    return new_state ,energy_sample[0],sampler ## energy , New sample 

# @partial(jax.jit,static_argnums=(4))
@jax.jit
def train_step_fssc(state , hamiltonain, sampler, flag, stored_tuple):
    sample_core = sampler.core_space
    def sample_ham_stored():
        return sampler.next_sample_stored(hamiltonain)

    sample , ham_stored = jax.lax.cond(flag,lambda :stored_tuple,sample_ham_stored)
 
    state , loss, sampler = new_state(state,sample,ham_stored,sampler)
    
    flag = jnp.all(jnp.unique(sample_core,axis =0,size = sampler.n_core)==jnp.unique(sampler.core_space,axis = 0,size = sampler.n_core))
    return state, loss, sampler, flag, (sample,ham_stored)

# OLD FUNCTION
# def energy_batch(params,hamiltonian,state,sampler,batch_size):
#     batches = sampler.core_space.reshape(-1,batch_size,sampler.n_spac_orb)
#     def energy_batch(carry, batch):
#         sample_new,numerator,denominator = carry 

#         batch_full, ham_stored= sampler.next_sample_stored_batch(batch,hamiltonian)
 
#         unique_full,idx = batch_full
#         ci = state.apply_fn({'params': params},unique_full)
#         #next_sample_idx = jnp.argsort(jnp.abs(Ci),descending =True)[:sampler.n_core]
        
#         Ci = ci[idx]
#         #reshape into batchsize multiples
#         ci_core, ci_connected = Ci[:batch.shape[0]], Ci[batch.shape[0]:].reshape(batch.shape[0],-1)
#         denominator += jnp.linalg.norm(ci_core)**2
#         numerator += jnp.dot(ci_core,jnp.einsum('ij,ij->i', ham_stored,ci_connected))
        

#         ## ERROR: The next_sample_idx is not unique for each batch
#         next_sample_idx = jnp.argsort(jnp.concatenate([sample_new[1], ci]), descending=True)[:sampler.n_core]
#         sample_new = (jnp.concatenate([sample_new[0], unique_full])[next_sample_idx], 
#                   jnp.concatenate([sample_new[1], ci])[next_sample_idx])
#         carry = (sample_new,numerator,denominator)
#         ## 
        
#         return (carry, None)
#     carry_init = ((jnp.zeros((sampler.n_core,sampler.n_spac_orb),dtype=jnp.uint8),jnp.zeros(sampler.n_core,dtype=jnp.float64)), 0.0 , 0.0)
#     final_carry,_ = jax.lax.scan(energy_batch,carry_init, batches)
#     energy = final_carry[1]/final_carry[2] #(final_carry[1],final_carry[2])
#     sample_new = final_carry[0]
#     return energy, sample_new


def batched_energy(params, carry ,state, stored, batch, sampler):
    
    ## needs to be updated
    sample = carry[0]
    #numerator, denominator = value

    batch_full, ham_stored = stored
    unique_full,idx = batch_full
    ci_unique = state.apply_fn({'params': params},unique_full)
    #next_sample_idx = jnp.argsort(jnp.abs(Ci),descending =True)[:sampler.n_core]
    
    ci_reconstructed = ci_unique[idx]
    #reshape into batchsize multiples
    ci_core, ci_connected = ci_reconstructed[:batch.shape[0]], ci_reconstructed[batch.shape[0]:].reshape(batch.shape[0],-1)
    denominator = jnp.linalg.norm(ci_core)**2
    numerator = jnp.dot(ci_core,jnp.einsum('ij,ij->i', ham_stored,ci_connected))
    
    max_indices = jnp.argsort(jnp.abs(ci_unique),descending =True)[:sampler.n_core]
    
    combined_sample_space =  jnp.concatenate([sample[0], unique_full[max_indices]]) 
    ci_combined = jnp.concatenate([sample[1], ci_unique[max_indices]])
    next_sample_idx = jnp.argsort(jnp.abs(ci_combined), descending=True)[:sampler.n_core]
    sample_new = (combined_sample_space[next_sample_idx], ci_combined[next_sample_idx])
   
    return (numerator,denominator),sample_new



@jax.jit
def train_step_batched(state, hamiltonian: Hamiltonian, sampler : FSSC):
    #jax.grad = jax.grad(energy) only in first input
    
    def batched_value_and_grad(carry, batch):
        stored = sampler.next_sample_stored_batch(batch,hamiltonian)
        # Jax grad only in the first input of f2
        # f2_grad = jax.grad(f2, argnums=0) 
        
        # take gradient wrt individual inputs seprately
        f2_params = lambda params: batched_energy(params, carry, state, stored, batch, sampler)
        eval , grad_fn, sample = jax.vjp(f2_params, state.params ,has_aux = True)
        # grads, aux_sample = jax.jacrev(f2,has_aux=True, argnums=0)(state.params, carry ,state, stored,batch,sampler)
        # aux = f2(state.params, carry ,state, stored, batch, sampler)
        grads = (grad_fn((1.,0.))[0],grad_fn((0.,1.))[0])
        grads = jax.tree.map(lambda x,y: x+y, grads, carry[1])
        #jax.debug.breakpoint()
        # jax.debug.print("grads ={grads}",grads=grads)
        carry = (sample, grads)
        # transform x to the carry form
        return carry, eval

    # carry_init = ((jnp.zeros((sampler.n_core,sampler.n_spac_orb),dtype=jnp.uint8),jnp.zeros(sampler.n_core,dtype=jnp.float64)), 0.0 , 0.0)
    # initiailize carry with tree structure
    init_grads = (jax.tree.map(lambda x: jnp.zeros_like(x), state.params),jax.tree.map(lambda x: jnp.zeros_like(x), state.params))
    carry_init = (jnp.zeros_like(sampler.core_space),jnp.zeros(sampler.n_core,dtype=jnp.float64)), init_grads
    batches = sampler.core_space.reshape(-1,sampler.n_batch,sampler.n_spac_orb)
    final_carry, eval= jax.lax.scan(batched_value_and_grad,carry_init,batches)
    sample_new, grads = final_carry
     
    value = jax.tree.map(lambda x: jnp.sum(x), eval)
    # get grads from final carry
    # aux ,grads = jax.value_and_grad(energy_batched, argnums=0, has_aux = True)(state.params, hamiltonian ,state, sampler,batch_size)
    overall_grads = jax.tree.map(lambda grads_0,grads_1: (value[1]*grads_0 - value[0]*grads_1)/value[1]**2, grads[0],grads[1])
    new_state = state.apply_gradients(grads=overall_grads)
    sampler = sampler.update_core_space(sample_new[0])
    return new_state , value[0]/value[1] , sampler        ## energy , new_sample 

@jax.jit
def train_step_VITE(state, hamiltonian: Hamiltonian, sampler : FSSC):
    """
    Perform a single training step for the Variational Imaginary Time Evolution (VITE) algorithm.
    Args:
        state: The current state of the model, including parameters and apply function.
        hamiltonian (Hamiltonian): The Hamiltonian of the system being simulated.
        sampler (FSSC): The sampler used to generate samples from the quantum state.
    Returns:
        tuple: A tuple containing the updated state, the energy and the updated sampler.
    """
    

    ## Calculate inverse A_ij
    apply_fn_params = lambda params: state.apply_fn({'params': params}, sampler.core_space)
    Jacobian = jax.jacrev(apply_fn_params,argnums=0)(state.params)
    #jax.debug.breakpoint()

    # Jacobian, _ = jax.flatten_util.ravel_pytree(Jacobian)
    # Jacobian = Jacobian.reshape((-1, sampler.n_core)) #(-1,sampler.n_core)
    Jacobian = jax.tree_map(lambda x: jnp.reshape(x,(sampler.n_core,-1)),Jacobian)
    Jacobian = jax.tree.flatten(Jacobian)[0]
    #jax.debug.breakpoint()
    Jacobian = jnp.concatenate(Jacobian ,axis= 1)

    #jax.debug.breakpoint()

    Aij =  100*Jacobian.T @ Jacobian  + 1e-4*jnp.diag(jnp.ones(Jacobian.shape[1]))  #Jacobian.T @ Jacobian # 
    #inverse_Aij = jnp.linalg.inv(Aij, hermitian=True)
    
    ## Calculate B_i
    # B_i = dC_j/dtheta_i * H_jk * C_k
    # j - core space , k - connecected space
    sample , H_ij = sampler.next_sample_stored(hamiltonian)
    unique_full , idx = sample
    Ci = state.apply_fn({'params': state.params},unique_full)
    ci_core, ci_connected = Ci[idx][:sampler.n_core], Ci[idx][sampler.n_core:].reshape(sampler.n_core,-1)
    H_Psi = jnp.einsum('ij,ij->i', H_ij,ci_connected)
    B_i = jnp.dot(Jacobian.T, H_Psi)
    
    ## Calculate energy and update core space
    next_sample_idx = jnp.argsort(jnp.abs(Ci),descending =True)[:sampler.n_core]
    Norm = jnp.linalg.norm(ci_core)
    energy = jnp.dot(ci_core,H_Psi)/Norm**2
    new_sample_core = unique_full[next_sample_idx]
    sampler = sampler.update_core_space(new_sample_core)


    ## Update parameters- calculate grads and update
    _, unravel_fn = jax.flatten_util.ravel_pytree(state.params)
    grads = jax.scipy.sparse.linalg.cg(Aij,100*B_i)[0] #jnp.dot(inverse_Aij,B_i)
    grads = unravel_fn(grads)
    # take care of negative sign in the update
    # try with and without adam
    #grads = jax.tree_map(lambda x: -x, grads)
    state = state.apply_gradients(grads=grads)


    return state, energy, sampler   


# Without ADAM
def create_train_state_VITE(rng, model, variables):
    tx = optax.sgd(learning_rate=learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)

def create_train_state(rng, model, variables):
    tx = optax.adam(learning_rate = learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)