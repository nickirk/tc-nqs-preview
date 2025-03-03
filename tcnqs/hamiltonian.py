import jax.numpy as jnp
import jax
from jax.tree_util import register_pytree_node_class
from functools import partial


# For documentation to Jax PyTrees:
# https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees

@register_pytree_node_class
class Hamiltonian:
    """
    Hamiltonian represents an electronic Hamiltonian for a quantum system, providing methods to evaluate
    matrix elements between determinants taking into account one- and two-body interactions. It is
    designed to work with JAX for just-in-time compilation and automatic differentiation.
    Parameters:
        n_elec_a (int): Number of alpha electrons.
        n_elec_b (int): Number of beta electrons.
        n_orb (int): Total number of spin orbitals (first half correspond to alpha spins, second half to beta spins).
        e_core (float): Core energy contribution.
        h1g (jnp.array): One-body integrals.
        g2e (jnp.array): Two-body integrals.
    Attributes:
        n_elec (int): Total number of electrons (n_elec_a + n_elec_b).
        n_elec_a (int): Number of alpha electrons.
        n_elec_b (int): Number of beta electrons.
        n_orb (int): Total number of spin orbitals.
        h1g (jnp.array): One-body integrals.
        g2e (jnp.array): Two-body integrals.
        e_core (float): Core energy.
        max_g2e (jnp.array): Maximum absolute value among sorted two-body integrals, used for screening.
        sorted_g2e (jnp.array): Sorted two-body integrals for efficient evaluation.
        sorted_inds (jnp.array): Sorted indices corresponding to the sorted two-body integrals.
    Methods:
        __call__(det1, det2):
            Evaluates the Hamiltonian matrix element between two determinants by determining the type of excitation
            (none, single, or double) and invoking the corresponding internal method.
        tree_flatten():
            Returns a tuple (dynamic, static) that separates mutable (dynamic) and immutable (static) fields of the instance
            for use with JAX pytree transformations.
        tree_unflatten(static, dynamic):
            Class method that reconstructs a Hamiltonian instance from flattened static and dynamic fields.
        phase(det, j):
            Computes the fermionic sign (phase) for a given determinant "det" up to index j, based on the parity of
            occupied orbitals.
        phase_2_pos(det, i, j):
            Computes the combined phase factor for two positions (i and j) in a determinant, using the individual phases.
        _unexcited_element(elec_pos_det1, pos_excite, pos_holes, det1, det2):
            Computes the Hamiltonian matrix element when there is no excitation between the determinants (i.e., diagonal element).
            It sums up the one-body and two-body contributions for the occupied orbitals and includes the core energy.
        _single_excitation_element(elec_pos_det1, pos_excite, pos_holes, det1, det2):
            Computes the Hamiltonian matrix element for a single excitation between determinants. It calculates the phase
            factors and sums the corresponding one-body and two-body contributions.
        _double_excitation_element(elec_pos_det1, pos_excite, pos_holes, det1, det2):
            Computes the Hamiltonian matrix element for a double excitation, where two electrons are excited simultaneously.
            It evaluates the two-body integral contribution corrected by the appropriate phase factors.
        _hamiltonian_element(det1, det2):
            Determines the type of excitation (0, 1, or 2 electron differences) between two determinants and returns
            the corresponding Hamiltonian matrix element by dispatching the call to the appropriate helper function.
        _get_1body(det1, det2):
            Evaluates the one-body part of the Hamiltonian matrix element based on the difference between determinants.
            Handles both the case of no excitation and a single excitation, returning the corresponding integral contribution.
        _get_2body(det1, det2):
            Evaluates the two-body part of the Hamiltonian matrix element by considering no excitation, single excitation,
            or double excitation scenarios, and returning the corresponding integral contribution.
        setup_hci():
            Prepares auxiliary data for Hamiltonian construction by sorting all pairs of two-body integrals. For each pair,
            the integrals are sorted in descending order by absolute value. It returns the maximum value among these sorted
            integrals, the sorted integrals, and their corresponding sorted indices.
    """

    def __init__(self, n_elec_a: int, n_elec_b: int, 
                 n_orb: int, e_core: float, h1g: jnp.array, g2e: jnp.array) -> None:
        # n_elec is total number of alpha and beta electrons
        self.n_elec = n_elec_a + n_elec_b
        self.n_elec_a = n_elec_a
        self.n_elec_b = n_elec_b
        # n_orb is the number of spin orbitals, first half are alpha and second half are beta
        self.n_orb = n_orb 
        self.h1g = h1g
        self.g2e = g2e
        self.e_core = e_core
        
        self.max_g2e, self.sorted_g2e, self.sorted_inds = self.setup_hci()

    def tree_flatten(self):
    # Return the dynamic fields and static fields separately
        dynamic = ()  # Include any fields that should be transformed (for example, if any mutable arrays)
        static = (self.n_elec_a, self.n_elec_b, self.n_orb, self.e_core,self.h1g, self.g2e,self.n_elec,self.sorted_g2e,self.sorted_inds)
        return dynamic, static

    @classmethod
    def tree_unflatten(cls, static, dynamic):
        # Reconstruct the class with static fields, dynamic can be ignored if empty
        instance = cls(*static[:6])
        instance.n_elec = static[6]
        instance.sorted_g2e , instance.sorted_inds = static[7:]

        return instance
        
    # By convention det1 is the bra and det2 is the ket always
    # @jax.jit
    def __call__(self, det1, det2):
        # det1 = jnp.asarray(det1, dtype=jnp.int8)
        # det2 = jnp.asarray(det2, dtype=jnp.int8)
        return self._hamiltonian_element(det1,det2) #self._get_1body(det1, det2)  + self._get_2body(det1, det2) 
    

    def phase(self,det,j):
        return 1-2*jnp.remainder(jnp.cumsum(det)[j-1],2)
    
    def phase_2_pos(self,det,i,j):
        # Replace by more efficient function later
        # return self.phase(det,i)*self.phase(det,j)
        cumsum = jnp.cumsum(det)
        return 1-2*jnp.remainder(cumsum[j-1]-cumsum[i-1],2)
    
    # @jax.jit
    def _unexcited_element(self, elec_pos_det1,pos_excite,pos_holes, det1, det2):
        # elec_pos_det1 = elec_pos_det1 
        get_1body = jnp.sum(self.h1g[elec_pos_det1, elec_pos_det1])
        get_2body =0.5*jnp.sum(self.g2e[elec_pos_det1[:, None], elec_pos_det1, elec_pos_det1, elec_pos_det1[:, None]] - 
                     self.g2e[elec_pos_det1[:, None], elec_pos_det1, elec_pos_det1[:, None], elec_pos_det1])
        
        return  self.e_core  + get_1body + get_2body
    
    # @jax.jit
    def _single_excitation_element(self, elec_pos_det1,pos_excite,pos_holes, det1, det2):
        i = pos_excite[0]
        j = pos_holes[0]
        # phase for 1 excitation will always be 1
        get_1body = self.h1g[i,j]
        # elec_pos_det1 = elec_pos_det1
        get_2body = jnp.sum(self.g2e[i, elec_pos_det1, elec_pos_det1, j] - self.g2e[i, elec_pos_det1, j, elec_pos_det1])
        return get_1body + get_2body
    
    # @jax.jit  
    def _double_excitation_element(self, elec_pos_det1,pos_excite,pos_holes, det1, det2):        
        i,k = pos_excite
        j,l = pos_holes
        phase_global =self.phase_2_pos(det1,i,k)*self.phase_2_pos(det2,j,l) #self.phase(det1,i)*self.phase(det1,k)*self.phase(det2,j)*self.phase(det2,l)
        #self.phase_2_pos(det1,i,k)*self.phase_2_pos(det2,j,l)
        
        # jax.debug.print("{tuple}",tuple=(det1,det2,phase_global))
        return phase_global*(self.g2e[i, k, l, j] - self.g2e[i, k, j, l])
       
    @jax.jit
    def _hamiltonian_element(self, det1, det2):
        """
        Calculate the Hamiltonian matrix element between two determinants.
        This function computes the Hamiltonian element between two many-body quantum states, represented as determinants. It
        first identifies the differing occupations between the two determinants using bitwise operations and then determines
        the excitation level based on the number of differences (0, 2, or 4). Depending on the excitation level, it either
        computes the unexcited, single excitation, or double excitation contribution to the Hamiltonian matrix element.
        If the number of differences does not match one of the expected cases, the function returns 0.0.
        Parameters:
            det1 (jax.numpy.DeviceArray): A 1D array representing the first determinant, where each bit indicates whether an
                                            orbital is occupied.
            det2 (jax.numpy.DeviceArray): A 1D array representing the second determinant, similarly defined.
        Returns:
            float: The computed Hamiltonian matrix element between the two determinants. If the determinants differ by an
                   unsupported number of excitations, the element is 0.0.
        """
        diff = jnp.bitwise_xor(det1, det2)
        num_diff = jnp.sum(diff,dtype=jnp.int8)
        elec_pos_det1 = jnp.nonzero(det1, size=self.n_elec)[0]
        # elec_pos_det1 = jnp.ones_like(det1[:7])
        pos_excite = jnp.nonzero(jnp.logical_and(det1,diff),size=2,fill_value =-1)[0]
        pos_holes = jnp.nonzero(jnp.logical_and(det2,diff),size=2,fill_value =-1)[0]
        input_tuple = (elec_pos_det1,pos_excite,pos_holes, det1, det2)
       
        cond_0 = lambda op: jax.lax.cond(
            num_diff == 0,
            lambda op: self._unexcited_element(*op),
            lambda op: 0.0,
            op)
        cond_2 = lambda op: jax.lax.cond(
            num_diff == 2,
            lambda op: self._single_excitation_element(*op),
            cond_0,
            op)
        cond_4 = lambda op: jax.lax.cond(
            num_diff == 4,
            lambda op: self._double_excitation_element(*op),
            cond_2,
            op)
        
        return cond_4(input_tuple)
    
    @jax.jit
    def _get_1body(self, det1, det2):
        diff = jnp.bitwise_xor(det1, det2)
        num_diff = jnp.sum(diff,dtype=jnp.int8)

        def diff_0():
            sum_indices = jnp.nonzero(det1 , size=self.n_elec)[0]
            return jnp.sum(self.h1g[sum_indices, sum_indices]) + self.e_core

        def diff_2():
           
            diff_index = jnp.nonzero(diff, size=2)[0]
            i = diff_index[jnp.nonzero(det1[diff_index], size=1)][0]
            j = diff_index[jnp.nonzero(det2[diff_index], size=1)][0]
            phase_global=self.phase(det1,i) * self.phase(det2,j)
            return (phase_global * self.h1g[i,j])
        
        return jax.lax.cond(num_diff == 2, 
                                diff_2,
                        lambda : jax.lax.cond(
                        num_diff == 0, diff_0, lambda : 0.0))

    @jax.jit
    def _get_2body(self, det1, det2):
        diff = jnp.bitwise_xor(det1, det2)
        num_diff = jnp.sum(diff, dtype=jnp.int8)

        def diff_0():
            sum_indices = jnp.nonzero(det1 , size=self.n_elec)[0]
            sum_result = jnp.sum(self.g2e[sum_indices[:, None], sum_indices, sum_indices, sum_indices[:, None]] - 
                     self.g2e[sum_indices[:, None], sum_indices, sum_indices[:, None], sum_indices]) 
            return sum_result/2

        def diff_2():

            diff_index = jnp.nonzero(diff, size=2)[0]
           
            k = diff_index[jnp.nonzero(det1[diff_index], size=1)][0]
            j = diff_index[jnp.nonzero(det2[diff_index], size=1)][0]
            sum_indices = jnp.nonzero(jnp.logical_and(det1, det2),size=self.n_elec-1)[0]
            
            phase_global=self.phase(det1,k) * self.phase(det2,j)
            
            return phase_global*jnp.sum(self.g2e[k, sum_indices, sum_indices, j] - self.g2e[k, sum_indices, j, sum_indices])

        def diff_4():

            diff_index = jnp.nonzero(diff, size=4)[0]
           
            det1_indices = diff_index[jnp.nonzero(det1[diff_index], size=2)]
            det2_indices = diff_index[jnp.nonzero(det2[diff_index], size=2)]
            i = det1_indices[0]
            k = det1_indices[1]
            j = det2_indices[0]
            l = det2_indices[1]
            
            phase_global = self.phase(det1,i)*self.phase(det1,k)*self.phase(det2,j)*self.phase(det2,l)
            
    
            
            return phase_global*(self.g2e[i, k, l, j] - self.g2e[i, k, j, l])

        return jax.lax.cond(num_diff == 4, 
                        diff_4,
                        lambda : jax.lax.cond(num_diff == 2, diff_2, 
                                           lambda : jax.lax.cond(num_diff == 0, diff_0, lambda : 0.0)))
    
    def setup_hci(self) -> jnp.ndarray:
        # Generate all the pairs of indices
        # can use jnp.meshgrid
        # pairs = jnp.meshgrid(jnp.arange(self.n_orb), jnp.arange(self.n_orb), indexing='ij')
        # pairs = jnp.array(pairs).reshape(2,-1).T
        pairs = jnp.array([(i, j) for i in range(self.n_orb) for j in range(self.n_orb)])

        def sort_elements(carry, pair):
            i, j = pair
            # dynamically slice the 2-body integrals
            # sort the elements by their absolute values in descending order
            block = self.g2e[i, :, :, j].flatten()
            sorted_inds = jnp.argsort(-jnp.abs(block))
            elements = jnp.take(block, sorted_inds)
            return carry, (elements, sorted_inds)

        # Initialize carry (not used in this case)
        # if carry is not used make use of vmap
        carry = None

        # Use jax.lax.scan to iterate over pairs and collect nonzero elements and indices
        _, results = jax.lax.scan(sort_elements, carry, pairs)

        sorted_elements, sorted_indices = results
        # get the maximum stored elements using the first column of sorted_elements
        max_element = jnp.max(jnp.abs(sorted_elements[:, 0]))


        return max_element, sorted_elements, sorted_indices