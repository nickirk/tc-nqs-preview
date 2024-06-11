

class HAMILTONIAN:

    def __init__(self, n_elec, n_orb, h1g, g2e):
        self.n_elec = n_elec
        self.n_orb = n_orb 
        self.h1g = h1g
        self.g2e = g2e
    
    def __call__(self, det1, det2):
        return self._get_1body(det1, det2) + self._get_2body(det1, det2)
    
    def _get_1body(self, det1, det2):
        # compare the two binary strings and find out the index where they differ
        diff = [i for i in range(len(det1)) if det1[i] != det2[i]]
        if len(diff) != 2:
            return 0.0
        return self.h1g[diff[0], diff[1]]
    
    def _get_2body(self, det1, det2):
        diff = [i for i in range(len(det1)) if det1[i] != det2[i]]
        if len(diff) != 4:
            return 0.0

        return self.g2e[i, j]