import jax
import jax.numpy as jnp
import optax
#from jax import vmap
from flax.training.train_state import TrainState

from functools import partial
# from tcnqs.sampler.connected_dets import generate_connected_space
from tcnqs.sampler.fssc import FSSC
from tcnqs.hamiltonian import Hamiltonian
import tcnqs.test.test_parameters as t
# from tcnqs.trainer import energy 

@partial(jax.jit, static_argnames=["is_tc", "solver"])
def trainer_vite(state:TrainState, hamiltonian: Hamiltonian, sampler : FSSC , is_tc=t.is_tc, proj_matrix = None,solver = 'SR'):
    """
    Performs one training iteration using VITE.

    Args:
        state (TrainState): Training state with parameters and optimizer.
        hamiltonian (Hamiltonian): Hamiltonian defining the system energy.
        sampler (FSSC): Sampler managing the core space for sampling.

    Returns:
        TrainState: Updated training state.
        float: Local energy.
        FSSC: Updated sampler.
    """    
    # Get energy and new core space
    energy, new_sample_core, ci_core, E_loc, Norm  = energy_fn(state, hamiltonian, sampler)

    # Compute normalized Jacobian
    jacobian, tc_adjust = jacobian_normalized(state, sampler.core_space, ci_core, Norm, energy)
    # jacobian = generate_jacobian(state, sampler.core_space)
    
    # overlap = jnp.sum(jnp.dot(ci_core,ci_connected))
    # # jax.debug.print("{overlap}" , overlap = overlap)
    # overlap_jacobian = jnp.sum(jnp.dot(jacobian.T, ci_connected), axis = 1)


    # Solve for gradients
    # B_i = jnp.dot(jacobian.T, E_loc)#jnp.sum(jacobian.T,axis=1)*jnp.sum(E_loc)#jnp.dot(jacobian.T, E_loc) 
    # # if is_tc:
    # B_i = 2*(jacobian.T @ (H_ij-energy))@(E_loc - energy*ci_core*Norm)#tc_adjust
    grads =  vite_solver(jacobian, E_loc, tc_adjust, is_tc ,proj_matrix=proj_matrix ,method = solver)
    # norm_g = jnp.linalg.norm(grads)
    # jax.debug.print("{grad}", grad=norm_g)
    #   jnp.dot(jacobian.T, E_loc) #
    #   Update sampler core space
    sampler = sampler.update_core_space(new_sample_core)

    # Apply gradients
    _, unravel_fn = jax.flatten_util.ravel_pytree(state.params)
    grads = unravel_fn(grads)
    state = state.apply_gradients(grads=grads)

    if solver=='projectedSR': # Incomplete
        return state, energy, sampler, jacobian.T @ jacobian
    else:
        return state, energy, sampler


@jax.jit
def pretrainer(state, hamiltonian: Hamiltonian, sampler: FSSC, ci_data):
    def loss_fn(params):
        preds = state.apply_fn({'params': params}, sampler.core_space)
        return jnp.mean((preds - ci_data) ** 2)
    loss ,grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, energy_fn(state, hamiltonian, sampler)[0], loss

@jax.jit
def energy_fn(state, hamiltonian: Hamiltonian, sampler: FSSC):
    """
    Computes the local energy and prepares new sampling points.

    Args:
        state (TrainState): Holds parameters for the wavefunction model.
        hamiltonian (Hamiltonian): Defines the system Hamiltonian.
        sampler (FSSC): Manages sampling of core and connected spaces.

    Returns:
        float: Computed energy.
        jnp.ndarray: Newly selected sample core.
        jnp.ndarray: Normalized core coefficients.
        jnp.ndarray: Local energy contributions.
        float: Norm of the core coefficients.
    """

    if sampler.n_core==sampler.n_batch:
        sample , Hij_Hji = sampler.next_sample_stored(hamiltonian)
        unique_full , idx = sample
        Ci = state.apply_fn({'params': state.params},unique_full)
        ci_core, ci_connected = Ci[idx][:sampler.n_core], Ci[idx][sampler.n_core:].reshape(sampler.n_core,-1)
        norm = jnp.linalg.norm(ci_core)
        ci_core, ci_connected = ci_core/norm, ci_connected/norm
        next_sample_idx = jnp.argsort(jnp.abs(Ci),descending =True)[:sampler.n_core]
        E_loc = jax.tree.map(lambda H: jnp.einsum('ij,ij->i', H,ci_connected), Hij_Hji )## introduce tree.map here
        energy = jnp.dot(ci_core,E_loc[0]) ## eloc [0] for pytree here left eigenvector as corespace
        #jnp.sum(ci_core)*jnp.sum(E_loc[0])
        new_sample_core = unique_full[next_sample_idx]
        
        E_loc = E_loc[0]
        #jnp.average(jnp.column_stack(E_loc),axis=1) # check


    else:
        batched_energy = lambda carry, batch_core: batched_energy_fn(carry, batch_core, state, hamiltonian, sampler)

        (energy, (new_sample_core, next_sample_ci)), (ci_core, E_loc) = jax.lax.scan(batched_energy, (0.0, (jnp.zeros_like(sampler.core_space), jnp.zeros(sampler.n_core,dtype=jnp.float64)))
                                                                                ,sampler.core_space.reshape(-1,sampler.n_batch,sampler.n_spac_orb) )
        norm = jnp.linalg.norm(ci_core.reshape(sampler.n_core,))
        ci_core = ci_core.reshape(sampler.n_core,) / norm
        E_loc = E_loc.reshape(sampler.n_core,) / norm 
        energy = energy/norm**2

    return energy, new_sample_core, ci_core, E_loc, norm#, ci_connected

def batched_energy_fn(carry, batch_core, state, hamiltonian, sampler):
    sample , H_ij = sampler.next_sample_stored_batch(hamiltonian, batch_core)
    unique_full , idx = sample
    Ci = state.apply_fn({'params': state.params},unique_full)
    ci_core, ci_connected = Ci[idx][:sampler.n_batch], Ci[idx][sampler.n_batch:].reshape(sampler.n_batch,-1)
    # norm = jnp.linalg.norm(ci_core)
    # ci_core, ci_connected = ci_core/norm, ci_connected/norm
    max_indices = jnp.argsort(jnp.abs(Ci),descending =True)[:sampler.n_core]
    E_loc = jax.tree.map(lambda H: jnp.einsum('ij,ij->i', H,ci_connected), H_ij )## introduce tree.map here
    energy = jnp.dot(ci_core,E_loc[0])
    E_loc = E_loc[0]
    #jnp.average(jnp.column_stack(E_loc),axis=1) # check
    # new_sample_core = unique_full[next_sample_idx]
    


    # Function unique among last sample Ci and new sample Ci 
    # size to be sampler.n_core
    combined_sample_space =  jnp.concatenate((jnp.zeros((1,sampler.n_spac_orb),dtype=jnp.int8),carry[1][0], unique_full[max_indices]),axis=0) 

    combined_sample_space, unique_idx = jnp.unique(combined_sample_space, axis=0, size = 2*sampler.n_core+1,
                                                fill_value=jnp.zeros(sampler.n_spac_orb,dtype=jnp.int8), return_index=True)
    ci_combined = jnp.concatenate((jnp.zeros(1),carry[1][1], Ci[max_indices]),axis=0)[unique_idx]

    next_sample_idx = jnp.argsort(jnp.abs(ci_combined), descending=True)[:sampler.n_core]
    # = (combined_sample_space[next_sample_idx], ci_combined[next_sample_idx])

    carry_new_sample = (combined_sample_space[next_sample_idx],ci_combined[next_sample_idx])

    return (carry[0] + energy, carry_new_sample), (ci_core, E_loc)


@partial(jax.vmap, in_axes=(None, 0))
def generate_jacobian(state, slater_det):
    """
    Computes the gradient of the wavefunction output with respect to model parameters.

    Args:
        state (TrainState): Holds model parameters and the apply function.
        slater_det (jnp.ndarray): Input configuration for which the Jacobian is computed.

    Returns:
        jnp.ndarray: Flattened gradient of the wavefunction with respect to parameters.
    """
    apply_fn = lambda params, sd: state.apply_fn({'params': params}, jnp.expand_dims(sd, axis=0))[0]
    jacobian_1d = jax.grad(apply_fn, argnums=0)(state.params, slater_det)
    jacobian_1d = jax.tree_map(lambda x: jnp.reshape(x, (1, -1)), jacobian_1d)
    jacobian_1d = jax.tree.flatten(jacobian_1d)[0]
    jacobian_1d = jnp.concatenate(jacobian_1d, axis=1)
    return jacobian_1d[0]

@jax.jit
def jacobian_normalized(state, core_space, ci_core, norm, energy):
    """
    Computes the normalized Jacobian by removing the projection on ci_core and dividing by norm.

    Args:
        state (TrainState): Training state with model parameters and apply function.
        core_space (jnp.ndarray): Core space configurations used for Jacobian computation.
        ci_core (jnp.ndarray): Wavefunction coefficients of the core space.
        norm (float): Norm of ci_core for normalization.

    Returns:
        jnp.ndarray: Normalized Jacobian.
    """
    Jacobian = generate_jacobian(state, core_space) 
    # Jacobian = (Jacobian - jnp.outer(ci_core,  jnp.dot(Jacobian.T, ci_core))) / norm
    return Jacobian , 0#energy * jnp.dot(Jacobian.T, ci_core)

    
def Stochastic_Reconfiguration(jacobian, E_loc, tc_adjust , is_tc):
    """
    Performs the Stochastic Reconfiguration method for gradient computation.

    Args:
        jacobian (jnp.ndarray): Jacobian matrix of the wavefunction model.
        E_loc (jnp.ndarray): Local energies corresponding to each sample.

    Returns:
        jnp.ndarray: Computed gradients using CG on the SR matrix.
    """
    Aij = jacobian.T @ jacobian
    Aij = Aij + 1e-6 * jnp.eye(Aij.shape[0])
    B_i = jnp.dot(jacobian.T, E_loc) 
    if is_tc:
        B_i = B_i - tc_adjust
    grads = jax.scipy.sparse.linalg.cg(Aij, B_i)[0]
     #jnp.linalg.pinv(Aij,rtol=1e-10)@B_i#
    # maxiter = 100
    return grads

def MinSR(jacobian, E_loc):
    """
    Minimizes the SR metric by inverting the projected Jacobian product.

    Args:
        jacobian (jnp.ndarray): Jacobian matrix of the wavefunction model.
        E_loc (jnp.ndarray): Local energies corresponding to each sample.

    Returns:
        jnp.ndarray: Computed gradients using pseudo-inverse.
    """
    Tij = jacobian @ jacobian.T+1e-7 * jnp.eye(jacobian.shape[0])
    B_i = E_loc
    grads = jacobian.T @ jax.scipy.sparse.linalg.cg(Tij, B_i ,maxiter=10)[0] #@ B_i
    return grads

def projected_SR(jacobian, E_loc, proj_matrix):
    """
    Applies a projection matrix before performing SR.

    Args:
        jacobian (jnp.ndarray): Jacobian matrix of the wavefunction model.
        E_loc (jnp.ndarray): Local energies corresponding to each sample.
        proj_matrix (jnp.ndarray): Matrix used for dimensionality reduction.

    Returns:
        jnp.ndarray: Computed gradients in the projected space.
    """
    Aij_save = jacobian.T @ jacobian
    Aij = proj_matrix.T @ Aij_save @ proj_matrix
    Aij = Aij + 1e-7 * jnp.eye(Aij.shape[0])
    B_i = proj_matrix.T @ jnp.dot(jacobian.T, E_loc)
    grads = jax.scipy.sparse.linalg.cg(Aij, B_i)[0]
    grads = proj_matrix @ grads
    return grads

@partial(jax.jit, static_argnames=["method", "is_tc"])
def vite_solver(jacobian: jnp.ndarray, E_loc: jnp.ndarray, tc_adjust:jnp.ndarray, is_tc:bool,proj_matrix: jnp.ndarray = None, method: str = 'SR'):
    """
    Selects and applies a solver method for gradient computation.

    Args:
        jacobian (jnp.ndarray): Jacobian matrix of the wavefunction model.
        E_loc (jnp.ndarray): Local energies for each sample.
        tc_adjust (jnp.ndarray): adjustment for B_i in SR.
        proj_matrix (jnp.ndarray, optional): Projection matrix for the 'Projections' method.
        method (str, optional): Solver method ('SR', 'MinSR', or 'Projections').

    Returns:
        jnp.ndarray: Computed gradients based on the selected solver.
    """
    if method == 'SR':
        return Stochastic_Reconfiguration(jacobian, E_loc, tc_adjust ,is_tc)
    elif method == 'minSR':
        return MinSR(jacobian, E_loc)
    elif method == 'projectedSR':
        return projected_SR(jacobian, E_loc, proj_matrix)
    else:
        raise ValueError('Method not implemented')


# Without ADAM
def create_train_state_VITE(rng, model, variables, optimizer=optax.sgd):
    scheduler = optax.linear_schedule(init_value=t.learning_rate[0], end_value=t.learning_rate[1], transition_steps=t.learning_rate[2])
    tx = optax.adam(learning_rate=scheduler)
    return TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)

def trainer_tc_stationary(state , hamiltonian, sampler):
    grads, (r_square, energy, new_sample_core,norm) = stationery_grads1(state, hamiltonian, sampler)
    
    state = state.apply_gradients(grads=grads)
    sampler = sampler.update_core_space(new_sample_core)
    grads = jax.flatten_util.ravel_pytree(grads)[0]

    return state, r_square, energy, norm, sampler, jnp.linalg.norm(grads)



@jax.jit
def stationery_grads1(state, hamiltonian , sampler):
 
    sample , Hij1 = sampler.next_sample_stored(hamiltonian)
    unique_full , idx = sample
    Ci = state.apply_fn({'params': state.params},unique_full)
    ci_core, ci_connected = Ci[idx][:sampler.n_core], Ci[idx][sampler.n_core:].reshape(sampler.n_core,-1)
    norm = jnp.linalg.norm(ci_core)
    # ci_core, ci_connected = ci_core/norm, ci_connected/norm
    Hij =Hij1[0]
    next_sample_idx = jnp.argsort(jnp.abs(Ci),descending =True)[:sampler.n_core]
    E_loc = jnp.einsum('ij,ij->i', Hij,ci_connected) # jax.tree.map(lambda H: jnp.einsum('ij,ij->i', H,ci_connected), Hij_Hji )## introduce tree.map here
    energy = jnp.dot(ci_core,E_loc)/norm**2 ## eloc [0] for pytree here left eigenvector as corespace
    #jnp.sum(ci_core)*jnp.sum(E_loc[0])
    new_sample_core = unique_full[next_sample_idx]
    def R_square(params, energy, Hij):
        Ci = state.apply_fn({'params': params},unique_full)
        ci_core, ci_connected = Ci[idx][:sampler.n_core], Ci[idx][sampler.n_core:].reshape(sampler.n_core,-1)
        E_loc = jnp.einsum('ij,ij->i', Hij,ci_connected)
        norm = jnp.linalg.norm(ci_core)
        # e = jnp.dot(ci_core, E_loc)/norm
    
        R = (E_loc- energy*ci_core)/norm
        return jnp.dot(R,R)

    
    r_square, grads = jax.value_and_grad(R_square)(state.params, energy, Hij)

    return grads , (r_square, energy , new_sample_core, norm)
