from  itertools import combinations
import jax.numpy as jnp
from functools import partial
import jax


# valid for equal number of electorns in both spin orbitals

# def generate_connected_space(determinant, n_orb, n_elec):
#     n_orb = int(n_orb/2)
#     n_elec = int(n_elec/2)
#     particles = jnp.where(determinant == 1, size = n_elec)[0]
#     holes = jnp.where(determinant == 0, size = n_orb - n_elec)[0]
#     alpha_space = generate_connected_space_spin(determinant[:n_orb], n_orb, n_elec)
#     beta_space = generate_connected_space_spin(determinant[n_orb:], n_orb, n_elec)
    
#     return 0

@partial(jax.jit, static_argnums=(1,2))
def generate_connected_space(determinant, n_orb, n_elec):
    n_orb = int(n_orb/2)
    n_elec = int(n_elec/2)
    
    alpha_0 = determinant[:n_orb]
    beta_0 =  determinant[n_orb:]
    
    particles_alpha = jnp.where(alpha_0== 1, size = n_elec)[0]
    holes_alpha = jnp.where(alpha_0 == 0, size = n_orb - n_elec)[0]
    
    particles_beta = jnp.where(beta_0== 1, size = n_elec)[0]
    holes_beta = jnp.where(beta_0 == 0, size = n_orb - n_elec)[0]
    
    alpha_0 = jnp.expand_dims(alpha_0, axis=0)
    beta_0 = jnp.expand_dims(beta_0, axis=0)
    
    
    alpha_1 = single_excitations(determinant[:n_orb], particles_alpha, holes_alpha)
    beta_1 = single_excitations(determinant[n_orb:], particles_beta, holes_beta)
    all_excitations = possible_excitations(jnp.concatenate((alpha_0,alpha_1),axis=0),jnp.concatenate((beta_0,beta_1),axis=0), 2*n_orb)
    
    alpha_2 = double_excitations(determinant[:n_orb], particles_alpha, holes_alpha)
    beta_2 = double_excitations(determinant[n_orb:], particles_beta, holes_beta)
    all_excitations = jnp.concatenate((all_excitations,possible_excitations(alpha_2, beta_0, 2*n_orb)),axis=0)
    all_excitations = jnp.concatenate((all_excitations,possible_excitations(alpha_0, beta_2, 2*n_orb)),axis=0)
    
    return all_excitations
    
    
    # particles = jnp.where(determinant == 1, size = n_elec)[0]
    # holes = jnp.where(determinant == 0, size = n_orb - n_elec)[0]
    # return jnp.concatenate((single_excitations(determinant,particles,holes),
    #                         double_excitations(determinant,particles,holes),
    #                         determinant.reshape(1, len(determinant))),axis=0)

#@partial(jax.vmap, in_axes=(0,0))
def possible_excitations(alpha,beta,n_orb):
    i,j =jnp.meshgrid(jnp.arange(len(alpha)),jnp.arange(len(beta)), indexing='ij')
    return jnp.concatenate((alpha[i],beta[j]),axis=2).reshape(-1,n_orb)

#@jax.jit
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
    
    return excite_single(particle_hole_pairs, determinant)

@partial(jax.vmap, in_axes=(0,None))
def excite_single(pair,determinant):
    return determinant.at[pair[0]].set(0).at[pair[1]].set(1)  

#@jax.jit
def double_excitations(determinant, particle_pos, hole_pos):
    particles_select = jnp.asarray(list(combinations(particle_pos,2)))
    holes_select = jnp.asarray(list(combinations(hole_pos,2)))
    i,j =jnp.meshgrid(jnp.arange(len(particles_select)),jnp.arange(len(holes_select)), indexing='ij')

    particle_hole_pairs=jnp.array([i,j]).reshape(2,-1).T
    
    return excite_double(particle_hole_pairs, particles_select, holes_select, determinant)
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
    
@partial(jax.vmap, in_axes=(0,None,None,None))
def excite_double(pair, particles_select, holes_select , determinant):
    a_i = particles_select[pair[0]]
    a_d_i = holes_select[pair[1]]
    return determinant.at[a_i].set(0).at[a_d_i].set(1)
    
if __name__ == '__main__':
    det= jnp.array([1,0,0,1,0,1,0,1,0,0], dtype=jnp.uint8)
    a = generate_connected_space(det, 10 ,4)
    print(a.shape)