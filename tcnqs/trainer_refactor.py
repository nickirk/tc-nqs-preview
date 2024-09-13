import jax
import jax.numpy as jnp
import optax
#from jax import vmap
from flax.training import train_state

from functools import partial
from tcnqs.sampler.connected_dets import generate_connected_space
from tcnqs.sampler.fssc import FSSC
from tcnqs.test.test_parameters import learning_rate 

# Handle sample function partial(jax.jit, static_argnums=(1,2))

## Temperory function with truncated functionality
def sample_core_new(params, last_sample, sampler):
    return sampler.next_sample(last_sample, params)[0][:sampler.n_core]


# Handle Hamiltonian Function
#@jax.profiler.annotate_function
@partial(jax.jit, static_argnums=(1,2))
def ham(sample_core, Hamiltonain,sampler):
    connected_space = jax.vmap(generate_connected_space,in_axes=(0,None,None))(sample_core,Hamiltonain.n_elec_a, Hamiltonain.n_elec_b)
    @partial(jax.vmap,in_axes=(0,0))
    def psi_H_xi(slater_determinant,connections):
        return jax.vmap(Hamiltonain,in_axes=(None,0))(slater_determinant,connections)

    return sample_core,connected_space,psi_H_xi(sample_core,connected_space)

    
# Handle Energy calculation
#@jax.profiler.annotate_function
@partial(jax.jit,static_argnums=(1))
def energy(params,sampler,ham_out,state):
    # calculate ci coefficients
    # calculate norm
    # claculate sum Psi_H_Psi = sum_ij C_i* C_j* <xi|H|xj> 
    # return sum(Psi_H_Psi)/Norm**2
    
    # Sampling part  will be shifted later to sampler
    # unique_connected, idx = jnp.unique(ham_out[1].reshape(-1,sampler.n_orb),axis=1,
    #                                     size = sampler.n_full, return_inverse=True)

    # ci_core , ci_connected = state.apply_fn({'params': params}, jnp.concatenate(ham_out[0],unique_connected))
    # ci_connected = ci_connected[idx].reshape()

    full_space = jnp.concatenate((ham_out[0],ham_out[1].reshape(-1,sampler.n_spac_orb)),axis=0)
    unique_full, idx = jnp.unique(full_space,axis=0, size = sampler.n_full, return_inverse=True
                                        ,fill_value=jnp.zeros(sampler.n_spac_orb))
    Ci = state.apply_fn({'params': params},unique_full)
    Norm = jnp.linalg.norm(Ci)
    next_sample_idx = jnp.argsort(jnp.abs(Ci),descending =True)[:sampler.n_core]
    Ci = Ci[idx]
    ci_core, ci_connected = Ci[:sampler.n_core], Ci[sampler.n_core:].reshape(sampler.n_core,-1)

    e = jnp.dot(ci_core,jnp.einsum('ij,ij->i', ham_out[2],ci_connected))/Norm**2
    
    new_sample = unique_full[next_sample_idx]

    return e,new_sample

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
@partial(jax.jit,static_argnums=(1))
def new_state(state, sampler, ham_out):
    #jax.grad = jax.grad(energy) only in first input
    energy_sample ,grads = jax.value_and_grad(energy, argnums=0, has_aux = True)(state.params,sampler, ham_out,state)
    new_state = state.apply_gradients(grads=grads)
    return new_state ,energy_sample[0],energy_sample[1] ## energy , New sample 
    


# Wrapper function - seprate out all the steps for better memory management in jit 
#@jax.profiler.annotate_function
def train_step_fssc(state, sample_core, Hamiltonain, sampler):
    #jax.profiler.start_trace("tmp/jax-trace",create_perfetto_link=True)
    #with jax.profiler.TraceAnnotation("ham_sample"):
    # sample_core = sample_core_new(state.params,last_sample,sampler)
    ham_out= ham(sample_core,Hamiltonain,sampler)
    #loss, _ = energy(state.params,sample,ham_out,state)
    #with jax.profiler.TraceAnnotation("newstate"):
    state , loss, new_sample_core = new_state(state,sampler,ham_out)

    #jax.profiler.stop_trace()
    return state, loss, new_sample_core


def create_train_state(rng, model, variables):
    tx = optax.adam(learning_rate = learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)