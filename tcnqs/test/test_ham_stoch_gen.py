import jax.numpy as jnp
from tcnqs.hamiltonian import Hamiltonian

if __name__ == '__main__':
    ham = Hamiltonian(2,2,10,0.0,jnp.zeros((10,10)),jnp.zeros((10,10,10,10)))
    det= jnp.array([1,0,0,1,0,1,0,1,0,0], dtype=jnp.uint8)
    a = ham.hamiltonian_and_connections(det)[1]
    assert jnp.unique(a,axis=0).shape == (55,10) and a.shape == (55,10)

    ham = Hamiltonian(1,1,10,0.0,jnp.zeros((4,4)),jnp.zeros((4,4,4,4)))
    det= jnp.array([1,0,0,0,0,1,0,0,0,0], dtype=jnp.uint8)
    a = ham.hamiltonian_and_connections(det)[1]
    assert jnp.unique(a,axis=0).shape == (25,10) and a.shape == (25,10)

    ham = Hamiltonian(2,1,10,0.0,jnp.zeros((10,10)),jnp.zeros((10,10,10,10)))
    det= jnp.array([1,1,0,0,0,1,0,0,0,0], dtype=jnp.uint8)
    a = ham.hamiltonian_and_connections(det)[1]
    assert jnp.unique(a,axis=0).shape == (38,10) and a.shape == (38,10)

    ham = Hamiltonian(2,1,6,0.0,jnp.zeros((6,6)),jnp.zeros((6,6,6,6)))
    det= jnp.array([1,1,0,1,0,0], dtype=jnp.uint8)
    a = ham.hamiltonian_and_connections(det)[1]
    assert jnp.unique(a,axis=0).shape == (9,6) and a.shape == (9,6)

    ham = Hamiltonian(1,1,4,0.0,jnp.zeros((4,4)),jnp.zeros((4,4,4,4)))
    det= jnp.array([0,0,0,0], dtype=jnp.uint8)
    a = ham.hamiltonian_and_connections(det)[1]
    assert jnp.unique(a,axis=0).shape == (1,4) and a.shape == (4,4)
    print("All tests passed!")