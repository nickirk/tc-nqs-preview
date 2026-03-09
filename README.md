# TC-NQS: Transcorrelated Second-Quantized Neural Network Quantum State

This project explores novel **neural network quantum states** (NQS) for quantum chemistry in **second quantization**. The primary focus is addressing the **dynamic correlation** (basis set convergence) problem in NQS using **transcorrelation (TC) theory**.

This project was developed in support of the master thesis by Unik Anil Wadhwani titled "Transcorelated Neural Network Quantum States".

## Project Overview
Standard NQS approaches can struggle with basis set convergence. TC-NQS implements transcorrelated Hamiltonians which, by construction, handle much of the short-range cusp condition and correlation, allowing simpler neural networks to describe the remaining wave function parts more efficiently. 

A key challenge introduced by transcorrelation is the **non-hermiticity** of the resulting Hamiltonian, which naively prevents standard variational optimization based on the Rayleigh-Ritz principle. To address this, TC-NQS utilizes specialized sampling and efficient second-order solvers:

- **Fixed-Size Selected Configuration (FSSC)**: A specialized sampling technique designed to handle the non-hermitian nature of the problem.
- **Variational Imaginary Time Evolution (VITE)**: A second-order method that evolves parameters in imaginary time by solving $A \dot{\theta} = -B$. It captures the curvature of the loss landscape, enabling faster convergence in the complex parameter spaces of NQS.
- **Minimum Stochastic Reconfiguration (MinSR)**: An efficient reformulation of VITE that utilizes the "tangent kernel trick" to invert a smaller $N_{\text{core}} \times N_{\text{core}}$ matrix. This significantly reduces memory and computational requirements for large-scale simulations.
- **Projected Stochastic Reconfiguration (ProjectedSR)**: A generalization of MinSR that projects the optimization onto the dominant eigenvectors of the Quantum Fisher Information Matrix (QFIM). It exploits the intrinsic low-rank structure of the QFIM to balance computational cost and solution accuracy.



## Core Technology Stack
- **Framework**: [JAX](https://github.com/google/jax) for high-performance autodiff and JIT compilation.
- **Deep Learning**: [Flax](https://github.com/google/flax) for neural network architectures (MLP, Backflow, VITE).
- **Quantum Chemistry**: [PySCF](https://pyscf.org/) for integrals and baseline molecular calculations.
- **Optimization**: [Optax](https://github.com/google-deepmind/optax) for parameter optimization.

## Installation

### Dependencies
The project requires Python >= 3.10 and the following core libraries:
- `jax`, `jaxlib`
- `pyscf`
- `scipy`
- `numpy`
- `optax`
- `flax`
- `h5py`
- `folx`
- `pytc` (Optional: required for on-the-fly transcorrelated integrals. *Coming soon to open source!*)
- `wandb` (Optional: for experiment tracking and logging)

### Setup with Conda
```bash
conda create -n tc-nqs python=3.10 -y
conda activate tc-nqs
pip install -e .
```
To include optional features (transcorrelation or experiment tracking):
```bash
# For transcorrelation (once pytc is available)
pip install -e ".[tc]"

# For experiment tracking
pip install -e ".[wandb]"

# For all extras
pip install -e ".[tc,wandb]"
```

### Fallback for Transcorrelation
While `pytc` is being prepared for open-source release, users can still perform transcorrelated calculations by reading pre-computed integrals from **FCIDUMP** files. Sample TC Hamiltonians and FCIDUMP formats can be found in the [TC Hamiltonians Resource](https://github.com/dobrautz/tc-varqite-hamiltonians) listed below.

### CUDA Requirements
The following nvidia dependencies are recommended for CUDA version 12.4+ environments (e.g., standard physics clusters):
- `nvidia-cublas-cu12`, `nvidia-cuda-cupti-cu12`, `nvidia-cuda-nvcc-cu12`, `nvidia-cuda-nvrtc-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, `nvidia-cufft-cu12`, `nvidia-curand-cu12`, `nvidia-cusolver-cu12`, `nvidia-cusparse-cu12`, `nvidia-nccl-cu12`, `nvidia-nvjitlink-cu12`

## Naming Conventions
1. **Classes**: CamelCase (e.g., `Hamiltonian`). Acronyms stay capitalized (e.g., `NQS`).
2. **Functions**: snake_case (e.g., `get_energy`).
3. **Variables**: snake_case (e.g., `n_elec_a`).

## Resources
- [TC Hamiltonians Resource](https://github.com/dobrautz/tc-varqite-hamiltonians)
