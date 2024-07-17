from  itertools import combinations
import jax.numpy as jnp
from functools import partial
import jax

@partial(jax.jit, static_argnums=(1,2))
def generate_connected_space(determinant, n_orb, n_elec):
    particles = jnp.where(determinant == 1, size = n_elec)[0]
    holes = jnp.where(determinant == 0, size = n_orb - n_elec)[0]
    return jnp.concatenate((single_excitations(determinant,particles,holes),double_excitations(determinant,particles,holes)),axis=0)

@jax.jit
def single_excitations(determinant, particle_pos , hole_pos):
    # connected_space_single = []
    # for i in particle_pos:
    #     for j in hole_pos:
    #         single_excitation = determinant
    #         single_excitation = single_excitation.at[i].set(0)
    #         single_excitation = single_excitation.at[j].set(1)
    #         connected_space_single.append(single_excitation) 
    # return jnp.asarray(connected_space_single, dtype=jnp.uint8)
    
    i,j=jnp.meshgrid(particle_pos,hole_pos, indexing='ij')
    particle_hole_pairs=jnp.array([i,j]).reshape(2,-1).T
    def excite_single(pair):
        return determinant.at[pair[0]].set(0).at[pair[1]].set(1)
        

    return jax.vmap(excite_single)(particle_hole_pairs)
        
        
@jax.jit
def double_excitations(determinant, particle_pos, hole_pos):
    particles_select = jnp.asarray(list(combinations(particle_pos,2)))
    holes_select = jnp.asarray(list(combinations(hole_pos,2)))
    i,j =jnp.meshgrid(jnp.arange(len(particles_select)),jnp.arange(len(holes_select)), indexing='ij')

    particle_hole_pairs=jnp.array([i,j]).reshape(2,-1).T

    def excite_double(pair):
        a_i = particles_select[pair[0]]
        a_d_i = holes_select[pair[1]]
        return determinant.at[a_i].set(0).at[a_d_i].set(1)
    
    return jax.vmap(excite_double)(particle_hole_pairs)
    # connected_space_double = []
    # for a in particles_select:
    #     for b in holes_select:
    #         double_excitation = determinant
    #         double_excitation = double_excitation.at[a[0]].set(0)
    #         double_excitation = double_excitation.at[a[1]].set(0)
    #         double_excitation = double_excitation.at[b[0]].set(1)
    #         double_excitation = double_excitation.at[b[1]].set(1)
    #         connected_space_double.append(double_excitation)
    
    # return jnp.asarray(connected_space_double, dtype=jnp.uint8)
    
    
if __name__ == '__main__':
    det= jnp.array([1,1,0,0,1], dtype=jnp.uint8)
    a = generate_connected_space(det, 5 ,3)
    print(a.shape)