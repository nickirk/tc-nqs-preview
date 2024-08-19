import jax.numpy as jnp
import jax
from . import Sampler
from tcnqs.hamiltonian import Hamiltonian
from tcnqs.slater_det import SD
from functools import partial
from tcnqs.sampler.connected_dets import generate_connected_space
import time

class FSSC(Sampler):
    def __init__(self, n_core, n_connected ,n_elec_a, n_elec_b,n_spac_orb) -> None:
        # super().__init__(wfn)
        # self.ham = ham
        self.n_core = n_core
        self.n_connected = n_connected
        self.n_elec_a = n_elec_a
        self.n_elec_b = n_elec_b
        self.n_spac_orb = n_spac_orb
        #self.n_connected = n_connected
    
    
    #@partial(jax.jit, static_argnums=(0))    
    def initialize(self, state) -> tuple[jnp.ndarray, jnp.ndarray]:
        alpha = jnp.concatenate((jnp.ones(self.n_elec_a),jnp.zeros(int(self.n_spac_orb/2)-self.n_elec_a)), dtype=jnp.uint8)
        beta = jnp.concatenate((jnp.ones(self.n_elec_b),jnp.zeros(int(self.n_spac_orb/2)-self.n_elec_b)), dtype=jnp.uint8)
        hartree_fock = jnp.concatenate((alpha,beta))
        cisd_space = generate_connected_space(hartree_fock, self.n_elec_a, self.n_elec_b)
        
        ## 3 is arbitary here
        number = 3*int(jnp.ceil(self.n_core/len(cisd_space)))
        core_space = jnp.empty((0, self.n_spac_orb), dtype=jnp.uint8)
        for i in range(1, number+1, 1):
            single_sd_connected_space = generate_connected_space(cisd_space[i], self.n_elec_a, self.n_elec_b)#[1:] # remove the core determinant
            determinants_to_append = self.find_sd_in_space(single_sd_connected_space, core_space, cisd_space)
            
            core_space = jnp.concatenate((core_space, determinants_to_append)) 
            
        core_space = jnp.concatenate((cisd_space,core_space))
        sorted_indices = jnp.argsort(jnp.sum(core_space, axis=1))[::-1]
        sorted_core_space = core_space[sorted_indices][:self.n_core]
        connected = self._sample_connected(sorted_core_space)
        
        #connected= self.remove_core_elements(connected,sorted_core_space)
        
        full_space = jnp.concatenate((sorted_core_space, connected))
        #full_space = jnp.concatenate((sorted_core_space, connected))
        # relevant_indices = jnp.where(jnp.logical_not(jnp.all(full_space==jnp.zeros(self.n_spac_orb),axis=1)))[0]
        # full_space = full_space[relevant_indices]
        #full_space = jnp.unique(full_space, axis=0)
                
        return (full_space, state.apply_fn({'params': state.params}, full_space))
            
        
    @partial(jax.jit, static_argnums=(0,3))    
    def next_sample(self, last_sample, params, apply_fn) -> jnp.ndarray:   #input: state instead of params,last_sample
        last_slater_determinants = last_sample[0]
        preds = apply_fn({'params': params}, last_slater_determinants)
        #jax.debug.breakpoint()
        # Reorder Ci and slater_determinants in decreasing order of mod of Ci

        connected_space = jnp.empty((self.n_connected,self.n_spac_orb))

        sorted_indices = jnp.argsort(jnp.abs(preds),descending =True)
        
        preds = preds[sorted_indices]
        last_slater_determinants = last_slater_determinants[sorted_indices]
        
        core_space = last_slater_determinants[:self.n_core]
        # full_connected_space = jax.vmap(generate_connected_space, in_axes=(0, None, None))(core_space, self.n_elec_a, self.n_elec_b)
        connected_space = self._sample_connected(core_space)
        #full_space = jnp.empty((self.n_core+self.n_connected,self.n_spac_orb))
        full_space = jnp.concatenate((core_space, connected_space)) 
        # relevant_indices = jnp.where(jnp.logical_not(jnp.all(full_space==jnp.zeros(self.n_spac_orb),axis=1)))[0]
        # full_space = full_space[relevant_indices]
        
        @partial(jax.vmap, in_axes=(0))
        def find_Ci(det):
            condition=jnp.all(last_slater_determinants==det,axis=1)
            return jax.lax.cond(jnp.any(condition), Ci_in_preds, 
                                Ci_not_in_preds,(det,condition))
            
        def Ci_in_preds(input_tuple):
            det,cond = input_tuple
            return jnp.asarray(preds[jnp.where(cond,size = 1)[0][0]])
        
        def Ci_not_in_preds(input_tuple):
            det,cond = input_tuple
            ci = apply_fn({'params': params}, jnp.expand_dims(det,axis=0))
            return ci[0]
        ##jnp.asarray(full_space,dtype=jnp.uint8), jnp.asarray(find_Ci(full_space),jnp.uint8)
        return (jnp.asarray(full_space,dtype=jnp.uint8), jnp.asarray(find_Ci(full_space),jnp.float64))
        
    @partial(jax.vmap, in_axes=(None, 0, None, None))    
    def find_sd_in_space(self,det,core_space,connected_space):
        condition_core=jnp.all(core_space==det,axis=1)
        condition_connected=jnp.all(connected_space==det,axis=1)
        
        condition_or = jnp.logical_not(jnp.logical_or(jnp.any(condition_core) , jnp.any(condition_connected)))

        return jax.lax.cond(condition_or, lambda : det, lambda : jnp.zeros(self.n_spac_orb, dtype=jnp.uint8))
        #return jnp.any(condition)
    
    @partial(jax.vmap, in_axes=(None, 0, None))    
    def remove_core_elements(self,det,core_space):
        condition_core=jnp.all(core_space==det,axis=1)
        #condition_connected=jnp.all(connected_space==det,axis=1)
        
        condition = jnp.logical_not(jnp.any(condition_core))

        return jax.lax.cond(condition, lambda : det, lambda : jnp.zeros(self.n_spac_orb, dtype=jnp.uint8))
        #return jnp.any(condition)
       
    
    ## Can be more efficient    
    def _sample_connected(self, core: jnp.ndarray) -> jnp.ndarray:
        connected = jnp.empty((0,self.n_spac_orb), dtype=jnp.uint8)
        
        ### For lax.scan function
        # def generate_unique_space(carry,det):
        #     single_sd_connected_space = generate_connected_space(det, self.n_elec_a, self.n_elec_b)#[1:] # remove the core determinant
        #     indices = jnp.where(not self.find_sd_in_space(single_sd_connected_space, core, connected), size= single_sd_connected_space.shape[0])
        #     _ = 
        #     connected = jnp.append(connected, _)
            
        #     return carry,connected
        
        # carry, connected = jax.lax.scan(generate_unique_space, 0, core)
        
        
        # for det in core:
        #     single_sd_connected_space = generate_connected_space(det, self.n_elec_a, self.n_elec_b)#[1:] # remove the core determinant
        #     determinants_to_append = self.find_sd_in_space(single_sd_connected_space, core, connected)
        #     connected = jnp.concatenate((connected, determinants_to_append))
        
        #start = jnp.array(time.time())
        connected = jax.vmap(generate_connected_space,in_axes=(0,None,None))(core, self.n_elec_a, self.n_elec_b)[1:]
        connected = jnp.reshape(connected,(-1, self.n_spac_orb))
        connected = jnp.unique(connected, size = self.n_connected ,axis=0,fill_value=jnp.zeros(self.n_spac_orb))
        connected = self.remove_core_elements(connected,core)
        
        #end = jnp.array(time.time())

        #jax.debug.print("Elapsed time: {time} seconds", time=end-start)

        return connected
