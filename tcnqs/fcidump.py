import numpy as np
import jax.numpy as jnp

def read(fcidump_file, is_tc):

    with open(fcidump_file, 'r') as f:
        lines = [x.lower().strip() for x in f.readlines()]
        lbrk = [il for il, l in enumerate(lines) if "&end" in l or "/" in l][0]
        keys = {'norb': None, 'nelec': None, 'ms2': None}
        for k in keys:
            keys[k] = [int(l.split(k)[1].split('=')[1].split(',')[0]) for l in lines if k in l][0]
        print(keys)
        n_sites = keys['norb']
        n_elec = keys['nelec']
        spin = keys['ms2']
        h1e = np.zeros((n_sites, n_sites),dtype=float)
        g2e = np.zeros((n_sites, n_sites, n_sites, n_sites),dtype=float)
        ecore = 0
        for l in lines[lbrk + 1:]:
            if len(l.split()) == 0:
                continue
            a, i, j, k, l = l.split()
            i, j, k, l = [int(x) - 1 for x in [i, j, k, l]]
            
            #print(a)
           
            if i + j + k + l == -4:
                ecore += float(a)
            
            elif k + l ==  -2:
                h1e[i, j] = float(a)
                h1e[j, i] = float(a)
               # print(h1e[i,j],h1e[j,i])
            else:
                if not is_tc:
                    g2e[i, k, l, j] = float(a)
                    g2e[j, k, l, i] = float(a)
                    g2e[j, l, k, i] = float(a)
                    g2e[i, l, k, j] = float(a)
                    g2e[k, i, j, l] = float(a)
                    g2e[k, j, i, l] = float(a)
                    g2e[l, j, i, k] = float(a)
                    g2e[l, i, j, k] = float(a)
                else:
                    g2e[i, k, l, j] = float(a)
                    g2e[k, i, j, l] = float(a)


    print('h1e norm = ', np.linalg.norm(h1e))
    print('g2e norm = ', np.linalg.norm(g2e))

    return n_sites, n_elec, ecore, h1e, g2e

def read_2_spin(fcidump_file):
    n_sites, n_elec, ecore, h1e, g2e = read(fcidump_file)

    h1e_s = np.kron(h1e, np.eye(2))
    g2e_s = np.zeros(np.asarray(g2e.shape)*2,dtype=float)
    for i in range(n_sites):
        for j in range(n_sites):
            for k in range(n_sites):
                for l in range(n_sites):
                    g2e_s[2*i, 2*k, 2*l, 2*j] = g2e[i, k, l, j]
                    g2e_s[2*i+1, 2*k+1, 2*l+1, 2*j+1] = g2e[i, k, l, j]
                    g2e_s[2*i, 2*k+1, 2*l+1, 2*j] = g2e[i, k, l, j]
                    g2e_s[2*i+1, 2*k, 2*l, 2*j+1] = g2e[i, k, l, j]
    return n_sites, n_elec, ecore, h1e_s, g2e_s

def read_2_spin_orbital_seprated(fcidump_file, is_tc):
    n_sites, n_elec, ecore, h1e, g2e = read(fcidump_file,is_tc)
    
    h1e_s = np.kron(np.eye(2),h1e)
    g2e_s = np.zeros(np.asarray(g2e.shape)*2)
    for i in range(n_sites):
        for j in range(n_sites):
            for k in range(n_sites):
                for l in range(n_sites):
                    # New function for my string order of orbitals 
                    # \alpha =(1,1,0,0) \beta=(1,1,0,0) --> \alpha \beta = (1,1,0,0,1,1,0,0)
                    g2e_s[i, k, l, j] = g2e[i, k, l, j]
                    g2e_s[i+n_sites, k+n_sites, l+n_sites, j+n_sites] = g2e[i, k, l, j]
                    g2e_s[i, k+n_sites, l+n_sites, j] = g2e[i, k, l, j]
                    g2e_s[i+n_sites, k, l, j+n_sites] = g2e[i, k, l, j]
    return n_sites, n_elec, ecore, h1e_s, g2e_s

# # Can make a new function by using jax functions instead of these classical functions

# import jax
# import jax.numpy as jnp
# from jax import jit, vmap
# from functools import partial

# def read(fcidump_file, is_tc=False):
#     with open(fcidump_file, 'r') as f:
#         lines = [x.lower().strip() for x in f.readlines()]
#         lbrk = next(il for il, l in enumerate(lines) if "&end" in l or "/" in l)

#     keys = {'norb': None, 'nelec': None, 'ms2': None}
#     for k in keys:
#         keys[k] = int([l.split(k)[1].split('=')[1].split(',')[0] for l in lines if k in l][0])

#     n_sites = keys['norb']
#     n_elec = keys['nelec']
#     spin = keys['ms2']

#     h1e = jnp.zeros((n_sites, n_sites), dtype=float)
#     g2e = jnp.zeros((n_sites, n_sites, n_sites, n_sites), dtype=float)
#     ecore = 0.0

#     def parse_line(line):
#         a, i, j, k, l = line.split()
#         i, j, k, l = map(lambda x: int(x) - 1, (i, j, k, l))
#         a = float(a)
#         return a, i, j, k, l

#     def process_entries(a, i, j, k, l):
#         nonlocal ecore, h1e, g2e
#         if i + j + k + l == -4:
#             ecore += a
#         elif k + l == -2:
#             h1e = h1e.at[i, j].set(a).at[j, i].set(a)
#         else:
#             if not is_tc:
#                 g2e = g2e.at[i, k, l, j].set(a).at[j, k, l, i].set(a)
#                 g2e = g2e.at[j, l, k, i].set(a).at[i, l, k, j].set(a)
#                 g2e = g2e.at[k, i, j, l].set(a).at[k, j, i, l].set(a)
#                 g2e = g2e.at[l, j, i, k].set(a).at[l, i, j, k].set(a)
#             else:
#                 g2e = g2e.at[i, k, l, j].set(a).at[k, i, j, l].set(a)

#     for line in lines[lbrk + 1:]:
#         if line:
#             process_entries(*parse_line(line))

#     print('h1e norm = ', jnp.linalg.norm(h1e))
#     print('g2e norm = ', jnp.linalg.norm(g2e))

#     return n_sites, n_elec, ecore, h1e, g2e

# #@jit
# def read_2_spin(fcidump_file):
#     n_sites, n_elec, ecore, h1e, g2e = read(fcidump_file)
#     h1e_s = jnp.kron(h1e, jnp.eye(2))
#     shape = (n_sites * 2,) * 4
#     g2e_s = jnp.zeros(shape, dtype=float)

#     def map_g2e(i, j, k, l):
#         g2e_s = jnp.zeros(shape, dtype=float)
#         g2e_s = g2e_s.at[2*i, 2*k, 2*l, 2*j].set(g2e[i, k, l, j])
#         g2e_s = g2e_s.at[2*i+1, 2*k+1, 2*l+1, 2*j+1].set(g2e[i, k, l, j])
#         g2e_s = g2e_s.at[2*i, 2*k+1, 2*l+1, 2*j].set(g2e[i, k, l, j])
#         g2e_s = g2e_s.at[2*i+1, 2*k, 2*l, 2*j+1].set(g2e[i, k, l, j])
#         return g2e_s

#     map_func = vmap(vmap(vmap(vmap(map_g2e, in_axes=(None, None, None, 0)))))
#     g2e_s = map_func(jnp.arange(n_sites), jnp.arange(n_sites), jnp.arange(n_sites), jnp.arange(n_sites))
    
#     return n_sites, n_elec, ecore, h1e_s, g2e_s

# #@jit
# def read_2_spin_orbital_seprated(fcidump_file):
#     # Call the read function with JIT to load data
#     n_sites, n_elec, ecore, h1e, g2e = read(fcidump_file)

#     # Create `h1e_s` using Kronecker product with JAX
#     h1e_s = jnp.kron(jnp.eye(2), h1e)
    
#     # Initialize `g2e_s` with shape doubled in each dimension
#     shape = (n_sites * 2,) * 4
#     g2e_s = jnp.zeros(shape, dtype=float)

#     # Define a function to handle the indexing assignments
#     def map_g2e(i, j, k, l):
#         g2e_entry = g2e[i, k, l, j]
#         g2e_s_entry = jnp.zeros(shape, dtype=float)

#         # Set entries based on spin-orbital separated indices
#         g2e_s_entry = g2e_s_entry.at[i, k, l, j].set(g2e_entry)
#         g2e_s_entry = g2e_s_entry.at[i + n_sites, k + n_sites, l + n_sites, j + n_sites].set(g2e_entry)
#         g2e_s_entry = g2e_s_entry.at[i, k + n_sites, l + n_sites, j].set(g2e_entry)
#         g2e_s_entry = g2e_s_entry.at[i + n_sites, k, l, j + n_sites].set(g2e_entry)

#         return g2e_s_entry

#     # Use `vmap` to apply `map_g2e` across all indices for parallelization
#     vmap_func = vmap(vmap(vmap(vmap(map_g2e, in_axes=(None, None, None, 0)))))
#     g2e_s = vmap_func(jnp.arange(n_sites), jnp.arange(n_sites), jnp.arange(n_sites), jnp.arange(n_sites))

#     return n_sites, n_elec, ecore, h1e_s, g2e_s