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
- [ ] Figure out in PySCF the indexing of slater determinants.
    - add test for this
- [ ] Train the simple MLP network with the FCI wavefunction.
    - add test
- [ ] Implement the backflow (finish backflow.py)

  
