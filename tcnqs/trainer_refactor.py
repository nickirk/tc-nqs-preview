import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from functools import partial
from tcnqs.sampler.connected_dets import generate_connected_space
from tcnqs.sampler.fssc import FSSC
from tcnqs.test.test_parameters import learning_rate 

# Handle sample function partial(jax.jit, static_argnums=(1,2))

## Temperory function with truncated functionality
@partial(jax.jit, static_argnums=(1))
def sample_new(sample_core, sampler):
    connected_space = jax.vmap(generate_connected_space,in_axes=(0,None,None))(sample_core,sampler.n_elec_a, sampler.n_elec_b)
    full_space = jnp.concatenate((sample_core,connected_space.reshape(-1,sampler.n_spac_orb)),axis=0)
    unique_full, idx = jnp.unique(full_space,axis=0, size = sampler.n_full, return_inverse=True
                                        ,fill_value=jnp.zeros(sampler.n_spac_orb))
                                        
    return unique_full,idx


# Handle Hamiltonian Function
#@jax.profiler.annotate_function
@partial(jax.jit, static_argnums=(1,2))
def ham(sample_core, Hamiltonain, sampler):
    #sample_core = sample[0][sample[1][:sampler.n_core]].reshape(-1,sampler.n_spac_orb)
    connected_space = jax.vmap(generate_connected_space,in_axes=(0,None,None))(sample_core,Hamiltonain.n_elec_a, Hamiltonain.n_elec_b)
    #connected_space = sample[0][sample[1][sampler.n_core:]].reshape(sampler.n_core,-1,sampler.n_spac_orb)
    @partial(jax.vmap,in_axes=(0,0))
    def psi_H_xi(slater_determinant,connections):
        return jax.vmap(Hamiltonain,in_axes=(None,0))(slater_determinant,connections)

    return psi_H_xi(sample_core,connected_space)

    
# Handle Energy calculation
#@jax.profiler.annotate_function
@partial(jax.jit,static_argnums=(4))
def energy(params,sample,ham_stored,state,sampler):
    # calculate ci coefficients
    # calculate norm
    # claculate sum Psi_H_Psi = sum_ij C_i* C_j* <xi|H|xj> 
    # return sum(Psi_H_Psi)/Norm**2
    
    # full_space = jnp.concatenate((ham_out[0],ham_out[1].reshape(-1,sampler.n_spac_orb)),axis=0)
    # unique_full, idx = jnp.unique(full_space,axis=0, size = sampler.n_full, return_inverse=True
    #                                     ,fill_value=jnp.zeros(sampler.n_spac_orb))
    unique_full,idx = sample
    Ci = state.apply_fn({'params': params},unique_full)
    Norm = jnp.linalg.norm(Ci)
    next_sample_idx = jnp.argsort(jnp.abs(Ci),descending =True)[:sampler.n_core]
    Ci = Ci[idx]
    ci_core, ci_connected = Ci[:sampler.n_core], Ci[sampler.n_core:].reshape(sampler.n_core,-1)

    e = jnp.dot(ci_core,jnp.einsum('ij,ij->i', ham_stored,ci_connected))/Norm**2
    
    new_sample_core = unique_full[next_sample_idx]

    return e,new_sample_core

    #ci_connected = jax.vmap(find_Ci, in_axes=(0))(ham_out[1])
    # def findciconnected(carry,connected_space):
    #     return carry,find_Ci(connected_space)
    # carry,ci_connected = jax.lax.scan(findciconnected,0,ham_out[1])
    # Ci = state.apply_fn({'params': params}, sample)
    # @partial(jax.vmap, in_axes=(0))
    # def find_Ci(det):
    #     condition=jnp.all(sample==det,axis=1)
    #     return jnp.asarray(Ci[jnp.where(condition, size = 1)[0][0]])
    #with jax.profiler.TraceAnnotation("find ci"):
    #jnp.sum(Ci),(sample,Ci)#
    
# Handle Gradients and Optimizers  
#@jax.profiler.annotate_function
@partial(jax.jit,static_argnums=(3))
def new_state(state, sample ,ham_stored,sampler):
    #jax.grad = jax.grad(energy) only in first input
    energy_sample ,grads = jax.value_and_grad(energy, argnums=0, has_aux = True)(state.params, sample, ham_stored, state, sampler)
    new_state = state.apply_gradients(grads=grads)
    return new_state ,energy_sample[0],energy_sample[1] ## energy , New sample 
    
@partial(jax.jit,static_argnums=(1,2))
def sample_ham_wrap(sample_core,Hamiltonain,sampler):
    #sample = sample_new(sample_core,sampler)
    return sample_new(sample_core,sampler), ham(sample_core,Hamiltonain,sampler)
# Wrapper function - seprate out all the steps for better memory management in jit 
#@jax.profiler.annotate_function
#@partial(jax.jit,static_argnums=(2,3))
# import time

@partial(jax.jit,static_argnums=(2,3))
def train_step_fssc(state, sample_core, Hamiltonain, sampler,flag,stored_tuple):
   
    @jax.jit
    def s_h_w(sample_core):
        return sample_ham_wrap(sample_core,Hamiltonain,sampler)

    sample , ham_stored = jax.lax.cond(flag,lambda x:stored_tuple,s_h_w,sample_core)
 
    state , loss, new_sample_core = new_state(state,sample,ham_stored,sampler)
    
    flag = jnp.all(sample_core==new_sample_core)
    return state, loss, new_sample_core, flag, (sample,ham_stored)


#     jax.profiler.start_trace("tmp/jax-trace",create_perfetto_link=True)
#     with jax.profiler.TraceAnnotation("ham_sample"):
#     if sample_core[1].shape[0]!= sampler.n_core:
#     a = time.time()
# else:
#        sample , ham_stored = s
#     b= time.time()
#     loss, _ = energy(state.params,sample,ham_out,state)
#     with jax.profiler.TraceAnnotation("newstate"):
#  jax.profiler.stop_trace()
#     if jnp.all(sample_core==new_sample_core):
#         new_sample_core = sample , ham_stored
#     c = time.time()
#     print(c-b,b-a)
def create_train_state(rng, model, variables):
    tx = optax.adam(learning_rate = learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)