from  itertools import combinations
import jax.numpy as jnp
import numpy as np
from functools import partial
import jax
from scipy.special import comb



@partial(jax.jit, static_argnums=(1,2))
def generate_connected_space(determinant: jnp.array, 
                             n_elec_a: int, n_elec_b: int) -> jnp.array:
    n_spa_orb = len(determinant)//2
    def _generate_connected_space():
        
        n_holes_a = n_spa_orb - n_elec_a
        n_holes_b = n_spa_orb - n_elec_b

        det_alpha_0 = determinant[:n_spa_orb]
        det_beta_0 =  determinant[n_spa_orb:]
        
        
        particles_pos_a = jnp.where(det_alpha_0 == 1, size = n_elec_a)[0]
        holes_pos_a = jnp.where(det_alpha_0 == 0, size = n_holes_a)[0]
        
        particles_pos_b = jnp.where(det_beta_0== 1, size = n_elec_b)[0]
        holes_pos_b = jnp.where(det_beta_0 == 0, size = n_holes_b)[0]
        
        det_alpha_0 = jnp.expand_dims(det_alpha_0, axis=0)
        det_beta_0 = jnp.expand_dims(det_beta_0, axis=0)

        dets_alpha = det_alpha_0.copy()
        dets_beta = det_beta_0.copy()
        
        if n_elec_a > 0 and n_holes_a > 0: 
            det_alpha_1 = single_excitations(determinant[:n_spa_orb], particles_pos_a, holes_pos_a)
            dets_alpha = jnp.concatenate((dets_alpha, det_alpha_1),axis=0)

        if n_elec_b > 0 and n_holes_b > 0: 
            det_beta_1 = single_excitations(determinant[n_spa_orb:], particles_pos_b, holes_pos_b)
            dets_beta = jnp.concatenate((dets_beta, det_beta_1),axis=0)

        all_excitations = possible_excitations(dets_alpha, dets_beta, 2*n_spa_orb)
                                            
        if n_elec_a > 1  and n_holes_a > 1:
            det_alpha_2 = double_excitations(determinant[:n_spa_orb], particles_pos_a, holes_pos_a)
            all_excitations = jnp.concatenate((all_excitations,
                                            possible_excitations(det_alpha_2, det_beta_0, 2*n_spa_orb)),axis=0)
        if n_elec_b > 1 and n_holes_b > 1:
            det_beta_2 = double_excitations(determinant[n_spa_orb:], particles_pos_b, holes_pos_b)
            all_excitations = jnp.concatenate((all_excitations,
                                            possible_excitations(det_alpha_0, det_beta_2, 2*n_spa_orb)),axis=0)
        
        return all_excitations
    
    
    # If the determinant is a a padded element, return a zero array
    # num_connections is required to make size same as output of _generate_connected_space
    def _generate_padding():
        # n_spa_orb = len(determinant)//2
        num_connections = (1 + comb(n_elec_a,2, exact=True)*comb(n_spa_orb-n_elec_a,2,exact=True)+comb(n_elec_b,2, exact=True)*comb(n_spa_orb-n_elec_b,2,exact=True)
                    + n_elec_a*n_elec_b*(n_spa_orb-n_elec_a)*(n_spa_orb-n_elec_b) + n_spa_orb*(n_elec_a+n_elec_b)- n_elec_a**2 - n_elec_b**2)
        return jnp.zeros((num_connections, 2*n_spa_orb),dtype =jnp.uint8)
    return jax.lax.cond(jnp.sum(determinant)==n_elec_a+n_elec_b,_generate_connected_space ,_generate_padding)
    
    
# why not using the vmap here?
# - Cannot use vmap as it doesnt takes all the possible combinations of alpha and beta
#@partial(jax.vmap, in_axes=(0,1))
def possible_excitations(alpha, beta, n_orb):
    i,j =jnp.meshgrid(jnp.arange(len(alpha)),jnp.arange(len(beta)), indexing='ij')
    return jnp.concatenate((alpha[i],beta[j]), axis=2).reshape(-1, n_orb)

#@jax.jit
def single_excitations(determinant, particle_pos , hole_pos):
    i,j=jnp.meshgrid(particle_pos, hole_pos, indexing='ij')
    particle_hole_pairs=jnp.array([i,j]).reshape(2,-1).T
    
    return excite_single(particle_hole_pairs, determinant)

@partial(jax.vmap, in_axes=(0,None))
def excite_single(pair,determinant):
    excite = jnp.zeros((determinant.shape),dtype=jnp.uint8)
    excite = excite.at[pair].set(1)
    return jnp.bitwise_xor(determinant,excite)
    #return determinant.at[pair[0]].set(0).at[pair[1]].set(1)  

#@jax.jit
def double_excitations(determinant, particle_pos, hole_pos):

    particles_select = jnp.asarray(list(combinations(particle_pos, 2)))
    holes_select = jnp.asarray(list(combinations(hole_pos,2)))
    i,j =jnp.meshgrid(jnp.arange(len(particles_select)),
                      jnp.arange(len(holes_select)), indexing='ij')

    particle_hole_pairs=jnp.array([i,j]).reshape(2,-1).T
    
    return excite_double(particle_hole_pairs, particles_select, holes_select, determinant)
    
@partial(jax.vmap, in_axes=(0,None,None,None))
def excite_double(pair, particles_select, holes_select , determinant):
    a = jnp.concatenate((holes_select[pair[1]],particles_select[pair[0]]))
    excite = jnp.zeros((determinant.shape),dtype=jnp.uint8)
    excite = excite.at[a].set(1)
    return jnp.bitwise_xor(determinant,excite)
    #return determinant.at[a_i].set(0).at[a_d_i].set(1)
    
if __name__ == '__main__':
    det= jnp.array([1,0,0,1,0,1,0,1,0,0], dtype=jnp.uint8)
    a = generate_connected_space(det,2,2)
    assert jnp.unique(a,axis=0).shape == (55,10) and a.shape == (55,10)

    det= jnp.array([1,0,0,0,0,1,0,0,0,0], dtype=jnp.uint8)
    a = generate_connected_space(det,1,1)
    assert jnp.unique(a,axis=0).shape == (25,10) and a.shape == (25,10)

    det= jnp.array([1,1,0,0,0,1,0,0,0,0], dtype=jnp.uint8)
    a = generate_connected_space(det,2,1)
    assert jnp.unique(a,axis=0).shape == (38,10) and a.shape == (38,10)

    det= jnp.array([1,1,0,1,0,0], dtype=jnp.uint8)
    a = generate_connected_space(det,2,1)
    assert jnp.unique(a,axis=0).shape == (9,6) and a.shape == (9,6)

    det= jnp.array([0,0,0,0], dtype=jnp.uint8)
    a = generate_connected_space(det,1,1)
    assert jnp.unique(a,axis=0).shape == (1,4) and a.shape == (4,4)