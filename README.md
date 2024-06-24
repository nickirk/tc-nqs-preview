# TC-NQS
Transcorrelated Second-Quantized Neural Network Quantum State


This project explores novel neural network quantum states for quantum chemistry in second quantization.

The main goal is to address the dynamic correlation (basis set convergence) problem in NQS with transcorrelation theory.


## 03.05.2024 Meeting
**TODO** 
- [x] Read about Cauchy interlacing theorem and understand why with a larger basis set, the ground state energy will be lower. See: https://en.wikipedia.org/wiki/Min-max_theorem#Cauchy_interlacing_theorem
- [x] Read about Hartree-Fock theory and understand the iterative solution/theory from the book by Szabo.
- [x] Play with PySCF and reproduce HF calculations on Szabo. 
- [x] Calculate HF energies with increasingly large basis sets: STO-3G, STO-6G, cc-pVDZ, cc-pVTZ, cc-pVQZ, cc-pV5Z, etc, and plot the energy versus basis set plot.


Add your plots in the `study.ipynb` file.

## 13.05.2024

- [x] Read about configuration interaction theory
- [x] Learn the basics about JAX: https://jax.readthedocs.io/en/latest/tutorials.html
    and Flax: https://flax.readthedocs.io/en/latest/quick_start.html 
- [x] configure your laptop and upload any notes that have been updated.

## 27.05.2024
- [x] Figure out in PySCF the indexing of slater determinants.
    - add test for this
- [x] Train the simple MLP network with the FCI wavefunction.
    - add test
- [x] Implement the backflow (finish backflow.py)

## 03.06.2024
- [x] Finish debugging backflow
    - produce the same test as in MLP, ensuring the number of parameters are the same in the two networks
    - plot the two trainning process and add them to the Nextcloud latex 

## 10.06.2024
- [ ] Start looking at how to implement Hamiltonian.
    - How to find the different indices between two strings of dets. 
    - Ke will add a FCIDUMP reader 
    - Slater-Condon rules: https://en.wikipedia.org/wiki/Slater–Condon_rules 
    - Finish the functions in hamiltonian.py


## 24.06.2024
- [ ] **Debug 4 electron system H4**
    - Write simple test inputting two SD directly and compute the known value.
- [ ] Fix the warning "assigning array to scalar".
- [ ] Think about how to improve the performance of Hamiltonian part.
    - **jit it, removing the if statements. **
    - *binary rep of SD, see if np array takes more memory?* 
        
