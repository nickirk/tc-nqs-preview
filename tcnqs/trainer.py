import jax
import jax.numpy as jnp
import optax
#from jax import vmap
from flax.training import train_state

from functools import partial
from tcnqs.sampler.connected_dets import generate_connected_space
from tcnqs.sampler.fssc import FSSC

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

#@partial(jax.jit, static_argnums=(2,3))
def train_step_fssc(state, last_sample, Hamiltonain, sampler):
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
    
    def hamiltonian_loss(params, apply_fn, last_sample, Hamiltonain,sampler):
        ## Sample is a tuple of x and C_i (x , C_i)
        # C_i = sample[1] 
        #a= apply_fn({'params': params}, jnp.expand_dims(x[5],axis=0))s
        #sampler = FSSC(n_core, hamiltonian.n_elec_a, hamiltonian.n_elec_b, num_orbitals)
        sample = sampler.next_sample(last_sample, params, apply_fn)
        
        Norm = jnp.linalg.norm(sample[1]) #[:sampler.n_core]
        # x_connected,C_i_connected=[],[]
        
        def find_Ci(det):
            condition=jnp.all(sample[0]==det,axis=1)
            return jnp.asarray(sample[1][jnp.where(condition, size = 1)[0][0]])
            

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
        loss , sample = hamiltonian_loss(params, state.apply_fn, last_sample, Hamiltonain, sampler)
        return loss

    grads = jax.grad(loss_fn)(state.params)
    # put nan to 0
    grads = jax.tree_map(lambda x: jnp.where(jnp.isnan(x), 0., x), grads)
    loss, new_sample = hamiltonian_loss(state.params, state.apply_fn, last_sample, Hamiltonain, sampler)
    
    state = state.apply_gradients(grads=grads)
    return state, loss, new_sample

def create_train_state(rng, model, variables):
    tx = optax.adam(learning_rate=0.01)
    return train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)