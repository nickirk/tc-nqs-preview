# TC-NQS
Transcorrelated Second-Quantized Neural Network Quantum State


This project explores novel neural network quantum states for quantum chemistry in second quantization.

The main goal is to address the dynamic correlation (basis set convergence) problem in NQS with transcorrelation theory.

Conventions in naming:
1. Class names should have the first letter of the word capitalized, e.g. Hamiltonian. If using acronyms, capitalize all letters, e.g. NQS.
2. Function names should be all lowercase, with words separated by underscores, e.g. get_energy.
3. Variable names should be all lowercase, with words separated by underscores, e.g. n_elec_a.

The following nvidia dependencies are required for CUDA version 12.4 installed on most devices on physics cluster.

nvidia-cublas-cu12           12.3.4.1\
nvidia-cuda-cupti-cu12       12.3.101\
nvidia-cuda-nvcc-cu12        12.3.107\
nvidia-cuda-nvrtc-cu12       12.3.107\
nvidia-cuda-runtime-cu12     12.3.101\
nvidia-cudnn-cu12            9.3.0.75\
nvidia-cufft-cu12            11.0.12.1\
nvidia-curand-cu12           10.3.4.107\
nvidia-cusolver-cu12         11.4.5.107\
nvidia-cusparse-cu12         12.2.0.103\
nvidia-nccl-cu12             2.19.3\
nvidia-nvjitlink-cu12        12.3.101\
