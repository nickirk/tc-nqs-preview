import jax
import jax.numpy as jnp
import optax
#from jax import vmap
from flax.training import train_state

from functools import partial
from tcnqs.sampler.connected_dets import generate_connected_space
from tcnqs.sampler.fssc import FSSC
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
def train_step_hamiltonian(state, batch, H):
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
def train_step_connections(state, batch, Hamiltonain):
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

    grads = jax.grad(loss_fn)(state.params)
    # put nan to 0
    #grads = jax.tree_map(lambda x: jnp.where(jnp.isnan(x), 0., x), grads)
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
    Norm = jnp.linalg.norm(Ci)
    next_sample_idx = jnp.argsort(jnp.abs(Ci),descending =True)[:sampler.n_core]
    Ci = Ci[idx]
    ci_core, ci_connected = Ci[:sampler.n_core], Ci[sampler.n_core:].reshape(sampler.n_core,-1)

    e = jnp.dot(ci_core,jnp.einsum('ij,ij->i', ham_stored,ci_connected))/Norm**2
    
    new_sample_core = unique_full[next_sample_idx]

    return e,new_sample_core

@partial(jax.jit)
def new_state(state, sample ,ham_stored,sampler):
    #jax.grad = jax.grad(energy) only in first input
    energy_sample ,grads = jax.value_and_grad(energy, argnums=0, has_aux = True)(state.params, sample, ham_stored, state, sampler)
    new_state = state.apply_gradients(grads=grads)
    return new_state ,energy_sample[0],energy_sample[1] ## energy , New sample 

# @partial(jax.jit,static_argnums=(4))
@jax.jit
def train_step_fssc(state, sample_core , flag, stored_tuple, hamiltonain, sampler):

    def sample_ham_stored(sample_core):
        return sampler.next_sample_stored(sample_core, hamiltonain)

    sample , ham_stored = jax.lax.cond(flag,lambda x:stored_tuple,sample_ham_stored,sample_core)
 
    state , loss, new_sample_core = new_state(state,sample,ham_stored,sampler)
    
    flag = jnp.all(jnp.unique(sample_core,axis =0,size = sampler.n_core)==jnp.unique(new_sample_core,axis = 0,size = sampler.n_core))
    return state, loss, new_sample_core, flag, (sample,ham_stored)


def create_train_state(rng, model, variables):
    tx = optax.adam(learning_rate = learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)