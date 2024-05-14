import numpy as np


def init_hamiltonian(n, degen_n=0, delta=1.0, m_off=0.1, nnz_ratio=0.1):
    """
    Initialize the toy Hamiltonian matrix.
    :param n: the size of the Hamiltonian matrix
    :param degen_n: the number of degenerate states
    :param delta: the diagonal gap of the Hamiltonian matrix
    :param m_off: the magnitude of the off-diagonal element of the Hamiltonian matrix
    :param nnz_ratio: the ratio of the non-zero elements in the off-diagonal part
    :return: the Hamiltonian matrix
    """
    H = np.zeros((n, n))
    for i in range(n):
        if i >= degen_n:
            H[i, i] = delta * i
        for j in range(i):
            if np.random.rand() < nnz_ratio:
                H[i, j] = (0.5-np.random.rand()) * m_off
                H[j, i] = H[i, j]
    return H