import jax.numpy as jnp
import jax
from jax.tree_util import register_pytree_node_class

from . import Sampler
from tcnqs.hamiltonian import Hamiltonian
from tcnqs.slater_det import SD
from functools import partial
from scipy.special import comb
from tcnqs.sampler.connected_dets import generate_connected_space
import time

@register_pytree_node_class
class FSSC(Sampler):
    def __init__(self, n_core: int, n_connected:int ,n_elec_a :int, n_elec_b:int, n_spac_orb:int, n_batch:int) -> None:
    # TODO: n_connected = number of connnections to be made
    # change the meaning for n_connected everywhere in code
    # change n_full to mean the max possible slater determinants
    
    # TODO: jax unique less than batchsize*connections

        self.n_core = n_core
        self.n_connected = n_connected
        self.n_elec_a = n_elec_a
        self.n_elec_b = n_elec_b
        self.n_spac_orb = n_spac_orb
        
        self.n_batch = n_batch
        self.n_full = self.n_core + self.n_connected

        # self.hamiltonain = hamiltonain
        self.core_space = jnp.zeros((self.n_core,self.n_spac_orb),dtype=jnp.uint8)
        # self.ham_stored = self.next_sample_stored(self.sample_core,hamiltonain)
    

   
    #@jax.jit   
    def initialize(self): # -> tuple[jnp.ndarray, jnp.ndarray]:
        alpha = jnp.concatenate((jnp.ones(self.n_elec_a),jnp.zeros(int(self.n_spac_orb/2)-self.n_elec_a)), dtype=jnp.uint8)
        beta = jnp.concatenate((jnp.ones(self.n_elec_b),jnp.zeros(int(self.n_spac_orb/2)-self.n_elec_b)), dtype=jnp.uint8)
        hartree_fock = jnp.concatenate((alpha,beta))
        cisd_space = generate_connected_space(hartree_fock, self.n_elec_a, self.n_elec_b)
        
        
        ## ADDED: Add functionality to keep going untill maximum has reached
        ## USE: jax.lax.while
        
        def generate_core_space(core_space):
            # Assumption: Jax will return padding as the 0th element when taking jnp.unique 
            # padding == jnp.zeros(self.n_spac_orb)
            connections =  self._vmap_generate_connected_space(core_space)
            connections = jnp.reshape(connections,(-1, self.n_spac_orb))
            #core_space = jnp.concatenate((core_space,connections))
            core_space = jnp.unique(connections,axis = 0,size=self.n_core+1,fill_value=jnp.zeros(self.n_spac_orb))
            
            # 1st element is padding because of the lexicographic sort(jnp.unique)
            return core_space[1:]
        
        init_value = jnp.unique(cisd_space,axis = 0,size=self.n_core,fill_value=jnp.zeros(self.n_spac_orb))
        core_space = jax.lax.while_loop(lambda core_space:jnp.all(core_space[-1]==jnp.zeros_like(core_space[-1])),generate_core_space,init_value)
        # core_space = self._vmap_generate_connected_space(cisd_space[1:])
        # core_space = jnp.reshape(core_space,(-1, self.n_spac_orb))
        # core_space = jnp.unique(core_space,axis=0,size=self.n_core ,fill_value=jnp.zeros(self.n_spac_orb))
        self.core_space = core_space
        # return core_space
        return self

     
    @jax.jit 
    def next_sample_stored(self, hamiltonian :Hamiltonian) -> jnp.ndarray:   #input: state instead of params,last_sample
        # sorted_indices = jnp.argsort(jnp.abs(last_sample[1]),descending =True)
        # core_space = last_sample[0][sorted_indices][:self.n_core]
        
        # connected_space = jnp.empty((self.n_connected,self.n_spac_orb))   
        # connected_space = self._sample_connected(core_space)
        # full_space = jnp.concatenate((core_space, connected_space)) 
        # return (jnp.asarray(full_space,dtype=jnp.uint8), jnp.asarray(self.wfn_apply({'params': params},(full_space)),jnp.float64))
        return self._full_space_(self.core_space),self.ham_stored(self.core_space, hamiltonian)
    
    @jax.jit 
    def next_sample_stored_batch(self,batch_core, hamiltonian :Hamiltonian) -> jnp.ndarray:   #input: state instead of params,last_sample
        # sorted_indices = jnp.argsort(jnp.abs(last_sample[1]),descending =True)
        # core_space = last_sample[0][sorted_indices][:self.n_core]
        
        # connected_space = jnp.empty((self.n_connected,self.n_spac_orb))   
        # connected_space = self._sample_connected(core_space)
        # full_space = jnp.concatenate((core_space, connected_space)) 
        # return (jnp.asarray(full_space,dtype=jnp.uint8), jnp.asarray(self.wfn_apply({'params': params},(full_space)),jnp.float64))
        return self._full_space_(batch_core),self.ham_stored(batch_core, hamiltonian)
    
      
    def _full_space_(self, sample_core: jnp.ndarray) -> jnp.ndarray:
        # connected_space = jnp.empty((self.n_connected,self.n_spac_orb), dtype=jnp.uint8)
        # connected = self._vmap_generate_connected_space(core)
        # connected = jnp.reshape(connected,(-1, self.n_spac_orb))
        
        # connected = jnp.unique(connected, size = self.n_full ,axis=0,fill_value=jnp.zeros(self.n_spac_orb))
        # connected = self._remove_core_elements(connected,core)
        # connected_space = self._remove_excess_padding(connected,size=self.n_connected)
        
        connected_space = self._vmap_generate_connected_space(sample_core)
        full_space = jnp.concatenate((sample_core,connected_space.reshape(-1,self.n_spac_orb)),axis=0)

        unique_full, idx = jnp.unique(full_space,axis=0, size = self.n_full, return_inverse=True
                                        ,fill_value=jnp.zeros(self.n_spac_orb))
        return unique_full,idx


    def ham_stored(self, sample_core, hamiltonain):
        # sample_core = sample[0][sample[1][:sampler.n_core]].reshape(-1,sampler.n_spac_orb)
        connected_space = self._vmap_generate_connected_space(sample_core)
        # connected_space = sample[0][sample[1][sampler.n_core:]].reshape(sampler.n_core,-1,sampler.n_spac_orb)
        @partial(jax.vmap,in_axes=(0,0))
        def psi_H_xi(slater_determinant,connections):
            return jax.vmap(hamiltonain,in_axes=(None,0))(slater_determinant,connections)

        return psi_H_xi(sample_core,connected_space)

    def update_core_space(self, new_sample_core):
        self.core_space = new_sample_core
        return self
    
    def _vmap_generate_connected_space(self, sd):
        # Apply vmap to the function, passing self attributes explicitly
        return jax.vmap(generate_connected_space, in_axes=(0, None, None))(sd, self.n_elec_a, self.n_elec_b)

    # Deprecated
    def _remove_excess_padding(self, sd_space, size=None):
        padding = jnp.zeros(self.n_spac_orb)
        relevant_indices = jnp.where(jnp.logical_not(jnp.all(sd_space==padding,axis=1)),size=size)[0]
        sd_space = sd_space[relevant_indices]
        return sd_space
    
    # Deprecated
    @partial(jax.vmap, in_axes=(None, 0, None))    
    def _remove_core_elements(self,det,core_space):
        condition_core=jnp.all(core_space==det,axis=1)
        #condition_connected=jnp.all(connected_space==det,axis=1)
        
        condition = jnp.logical_not(jnp.any(condition_core))

        return jax.lax.cond(condition, lambda : det, lambda : jnp.zeros(self.n_spac_orb, dtype=jnp.uint8))
        #return jnp.any(condition)

    def tree_flatten(self):
        # Return the dynamic fields and static fields separately
        dynamic = (self.core_space,)  # Include any fields that should be transformed with jax (for example, mutable arrays)
        static = (self.n_core, self.n_connected, self.n_elec_a, self.n_elec_b, self.n_spac_orb,self.n_batch,self.n_full)
        return dynamic, static

    @classmethod
    def tree_unflatten(cls, static, dynamic):
        # Reconstruct the class with static fields, dynamic can be ignored if empty
        instance = cls(*static[:-1])
        instance.n_full = static[-1]
        instance.core_space = dynamic[0]
        #instance.ham_stored = dynamic[1]
        return instance


## Deprecated Methods: Can be useful in the future
 # @partial(jax.vmap, in_axes=(0))
        # def find_Ci(det):
        #     condition=jnp.all(last_slater_determinants==det,axis=1)
        #     return jax.lax.cond(jnp.any(condition), Ci_in_preds, 
        #                         Ci_not_in_preds,(det,condition))
            
        # def Ci_in_preds(input_tuple):
        #     det,cond = input_tuple
        #     return jnp.asarray(preds[jnp.where(cond,size = 1)[0][0]])
        
        # def Ci_not_in_preds(input_tuple):
        #     det,cond = input_tuple
        #     ci = apply_fn({'params': params}, jnp.expand_dims(det,axis=0))
        #     return ci[0]

    # @partial(jax.vmap, in_axes=(None, 0, None, None))    
    # def find_sd_in_space(self,det,core_space,connected_space):
    #     condition_core=jnp.all(core_space==det,axis=1)
    #     condition_connected=jnp.all(connected_space==det,axis=1)
        
    #     condition_or = jnp.logical_not(jnp.logical_or(jnp.any(condition_core) , jnp.any(condition_connected)))

    #     return jax.lax.cond(condition_or, lambda : det, lambda : jnp.zeros(self.n_spac_orb, dtype=jnp.uint8))
    #     #return jnp.any(condition)

    # return self._full_space_(core_space)
    # n_s_orb = int(self.n_spac_orb/2)
    # totals_dets = comb(num_s_orb, self.n_elec_a,exact=True)*comb(num_s_orb, self.n_elec_b ,exact=True)

    # num_cisd = (1 + comb(self.n_elec_a,2, exact=True)*comb(n_s_orb-self.n_elec_a,2,exact=True)+comb(self.n_elec_b,2, exact=True)*comb(n_s_orb-self.n_elec_b,2,exact=True)
    #             + self.n_elec_a*self.n_elec_b*(n_s_orb-self.n_elec_a)*(n_s_orb-self.n_elec_b) + n_s_orb*(self.n_elec_a+self.n_elec_b)- self.n_elec_a**2 - self.n_elec_b**2)
    # ## 3 is arbitary here ## Approx number of samples to take
    # approx_number = int(jnp.ceil(3*self.n_core/num_cisd))

    # padding = jnp.zeros(self.n_spac_orb)
    # core_space = jnp.empty((self.n_core, self.n_spac_orb), dtype=jnp.uint8)
    # 2 lines generates new elemets for core space from the cisd space 
    # and then remove the elements that are  already present in cisd space and extra padding
    # core_space = self._remove_core_elements(core_space,cisd_space)
    # core_space = self._remove_excess_padding(core_space)
    # core_space = jnp.concatenate((cisd_space,core_space))[:self.n_core]
    # core_space = jnp.unique(core_space, size = self.n_core ,axis=0,fill_value=jnp.zeros(self.n_spac_orb))
    # connected_space = self._sample_connected(core_space)
    # full_space = jnp.concatenate((core_space, connected_space))


    # relevant_indices = jnp.where(jnp.logical_not(jnp.all(full_space==padding,axis=1)))[0]
    # full_space =full_space[relevant_indices]

    # return (full_space, self.wfn_apply({'params': params}, full_space))
        