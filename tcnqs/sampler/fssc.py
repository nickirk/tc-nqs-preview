import jax.numpy as jnp
import jax
from . import Sampler
from tcnqs.hamiltonian import Hamiltonian
from tcnqs.slater_det import SD
from functools import partial
from scipy.special import comb
from tcnqs.sampler.connected_dets import generate_connected_space
import time

class FSSC(Sampler):
    def __init__(self, n_core, n_connected ,n_elec_a, n_elec_b, n_spac_orb, model_apply) -> None:
        # super().__init__(wfn)
        # self.wfn = apply_fn
        self.wfn_apply = model_apply
        self.n_core = n_core
        self.n_connected = n_connected
        self.n_full = n_core + n_connected
        self.n_elec_a = n_elec_a
        self.n_elec_b = n_elec_b
        self.n_spac_orb = n_spac_orb
        #self.n_connected = n_connected
    
    # Dont jit: dynamic methods involved
    # Jit doesnt give any real advantage as the function is called only once!
    #@partial(jax.jit, static_argnums=(0))    
    def initialize(self, params) -> tuple[jnp.ndarray, jnp.ndarray]:
        alpha = jnp.concatenate((jnp.ones(self.n_elec_a),jnp.zeros(int(self.n_spac_orb/2)-self.n_elec_a)), dtype=jnp.uint8)
        beta = jnp.concatenate((jnp.ones(self.n_elec_b),jnp.zeros(int(self.n_spac_orb/2)-self.n_elec_b)), dtype=jnp.uint8)
        hartree_fock = jnp.concatenate((alpha,beta))
        cisd_space = generate_connected_space(hartree_fock, self.n_elec_a, self.n_elec_b)
        
        n_s_orb = int(self.n_spac_orb/2)
        # totals_dets = comb(num_s_orb, self.n_elec_a,exact=True)*comb(num_s_orb, self.n_elec_b ,exact=True)

        num_cisd = (1 + comb(self.n_elec_a,2, exact=True)*comb(n_s_orb-self.n_elec_a,2,exact=True)+comb(self.n_elec_b,2, exact=True)*comb(n_s_orb-self.n_elec_b,2,exact=True)
                    + self.n_elec_a*self.n_elec_b*(n_s_orb-self.n_elec_a)*(n_s_orb-self.n_elec_b) + n_s_orb*(self.n_elec_a+self.n_elec_b)- self.n_elec_a**2 - self.n_elec_b**2)
        ## 3 is arbitary here ## Approx number of samples to take
        approx_number = int(jnp.ceil(3*self.n_core/num_cisd))

        # padding = jnp.zeros(self.n_spac_orb)
        # core_space = jnp.empty((self.n_core, self.n_spac_orb), dtype=jnp.uint8)
        # 2 lines generates new elemets for core space from the cisd space 
        # and then remove the elements that are  already present in cisd space and extra padding
        core_space = self._vmap_generate_connected_space(cisd_space[1:approx_number])
        core_space = jnp.reshape(core_space,(-1, self.n_spac_orb))
        core_space = self._remove_core_elements(core_space,cisd_space)
        core_space = self._remove_excess_padding(core_space)
        core_space = jnp.unique(core_space,axis=0)
        core_space = jnp.concatenate((cisd_space,core_space))[:self.n_core]
        # core_space = jnp.unique(core_space, size = self.n_core ,axis=0,fill_value=jnp.zeros(self.n_spac_orb))
        

        connected_space = self._sample_connected(core_space)
        full_space = jnp.concatenate((core_space, connected_space))

        
        # relevant_indices = jnp.where(jnp.logical_not(jnp.all(full_space==padding,axis=1)))[0]
        # full_space =full_space[relevant_indices]

        return (full_space, self.wfn_apply({'params': params}, full_space))
            
     
    @partial(jax.jit, static_argnums=(0))    
    def next_sample(self, last_sample, params) -> jnp.ndarray:   #input: state instead of params,last_sample
        sorted_indices = jnp.argsort(jnp.abs(last_sample[1]),descending =True)
        core_space = last_sample[0][sorted_indices][:self.n_core]
        
        connected_space = jnp.empty((self.n_connected,self.n_spac_orb))   
        connected_space = self._sample_connected(core_space)
        full_space = jnp.concatenate((core_space, connected_space)) 
        return (jnp.asarray(full_space,dtype=jnp.uint8), jnp.asarray(self.wfn_apply({'params': params},(full_space)),jnp.float64))
       
    
    @partial(jax.vmap, in_axes=(None, 0, None))    
    def _remove_core_elements(self,det,core_space):
        condition_core=jnp.all(core_space==det,axis=1)
        #condition_connected=jnp.all(connected_space==det,axis=1)
        
        condition = jnp.logical_not(jnp.any(condition_core))

        return jax.lax.cond(condition, lambda : det, lambda : jnp.zeros(self.n_spac_orb, dtype=jnp.uint8))
        #return jnp.any(condition)
       
      
    def _sample_connected(self, core: jnp.ndarray) -> jnp.ndarray:
        connected_space = jnp.empty((self.n_connected,self.n_spac_orb), dtype=jnp.uint8)
        #start = jnp.array(time.time())
        connected = self._vmap_generate_connected_space(core)
        connected = jnp.reshape(connected,(-1, self.n_spac_orb))
        
        connected = jnp.unique(connected, size = self.n_full ,axis=0,fill_value=jnp.zeros(self.n_spac_orb))
        connected = self._remove_core_elements(connected,core)
        connected_space = self._remove_excess_padding(connected,size=self.n_connected)
        
        #end = jnp.array(time.time())

        #jax.debug.print("Elapsed time: {time} seconds", time=end-start)

        return connected_space

    @partial(jax.vmap,in_axes=(None,0))
    def _vmap_generate_connected_space(self,sd):
        return generate_connected_space(sd, self.n_elec_a, self.n_elec_b)[1:]
 
    def _remove_excess_padding(self, sd_space, size=None):
        padding = jnp.zeros(self.n_spac_orb)
        relevant_indices = jnp.where(jnp.logical_not(jnp.all(sd_space==padding,axis=1)),size=size)[0]
        sd_space = sd_space[relevant_indices]
        return sd_space

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