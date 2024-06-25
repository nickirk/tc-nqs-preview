import numpy as np


def read(fcidump_file, is_tc=False):
    # read fcidump
    with open(fcidump_file, 'r') as f:
        lines = [x.lower().strip() for x in f.readlines()]
        lbrk = [il for il, l in enumerate(lines) if "&end" in l or "/" in l][0]
        k = 'orbsym'
        #orb_sym = [[int(x) for x in (l.split(k)[1].split('=')[1].split(','))] for l in lines if k in l][0]


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

def read_2_spin_orbital_seprated(fcidump_file):
    n_sites, n_elec, ecore, h1e, g2e = read(fcidump_file)
    # Check Logic again: We have 2 spin seprated orbitals collated together
    # h1e_s = np.kron(h1e, np.eye(2))
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