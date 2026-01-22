# Active-Matter-Adjustable-Networks

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
<!-- [![CUDA](https://img.shields.io/badge/CUDA-supported-red.svg)](https://developer.nvidia.com/cuda-toolkit) --> 
<!-- [![Linux](https://img.shields.io/badge/OS-Linux-lightgrey.svg)](https://www.kernel.org/) --> 
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx) -->

A CUDA-C framework for simulating **active matter on deformable and dynamically adjustable networks**, optimized for large-scale GPU acceleration.

---

## Summary
This repository provides a high-performance CUDA-C framework for simulating **active Brownian particles (ABPs)** moving on lattice-based networks that can undergo **stochastic bond breaking and regeneration**. The framework allows the study of **transport properties, particle mobility, diffusion, and dynamical heterogeneity** in dynamically remodeling networks. Its GPU-accelerated design enables simulations of very large systems, suitable for exploring complex active-matter phenomena.

---

## Repository structure
The repository is organized into **article-specific directories**. Each directory is self-contained and includes its own simulation code, CUDA kernels, and job submission scripts.

```text
.
├── README.md
├── LICENSE
├── Transport_Properties/
│   ├── lattice_coordination/
│   ├── msd_and_dynamical_heterogeneity/
│   ├── non_steric_interactions/
│   └── stopping_class/
```
Each study directory follows the same internal structure:

```text
├── main.cu        # Main CUDA entry point
├── submit.sh      # GPU cluster submission script
├── src/           # CUDA source files and kernels
└── include/       # Headers and configuration files
```
Detailed instructions and model-specific descriptions are provided in the corresponding local ```README.md``` files.

---

## Requirements

### Hardware
- NVIDIA GPU with CUDA support  
  - Compute capability ≥ 6.0 (Ampere GPUs recommended for large systems)
- Sufficient GPU memory for large-scale simulations (≥ 8 GB recommended)
- Multi-core CPU host system

### Software
- Linux-based operating system
- NVIDIA CUDA Toolkit (tested with CUDA ≥ 11.x)
- Compatible NVIDIA GPU driver
- Bash-compatible shell environment
- Standard CUDA/C toolchain:
  - `nvcc`
  - `gcc`
  - `bc` (required for floating-point arithmetic in job scripts)

### Compilation model
- Simulations are **compiled on-the-fly** to allow parameter-dependent optimization using `nvcc` within the submission script.
- CUDA architecture flags (`-arch=sm_xx`) may need to be adapted to the target GPU.

### Execution environment (recommended)
- Access to a GPU workstation or HPC cluster
- Optional job scheduler (e.g. SLURM, PBS, or equivalent)
- SSH access and filesystem permissions for batch execution

### Notes
- The code is written in **CUDA-C** and does not rely on external C++ libraries.
- Job submission scripts (`submit.sh`) are provided as templates and may require minor adaptation depending on the local system configuration.

---

## Scientific scope
The current implementation focuses on the **transport properties** of ABPs in dynamically rearranging networks. In particular, the code allows one to investigate how stochastic network remodeling affects:

- Particle mobility  
- Diffusion properties  
- Dynamical heterogeneity  
- Large-scale transport behavior
- Network-conectivity 

Planned extensions include the incorporation of **elastic and mechanical responses of the network**, as well as couplings between particle dynamics and network elasticity.

---

## Authors and Contributions
This repository is developed and maintained by the **Liarte-Group**, an open and growing research group focused on computational and theoretical studies of active and soft matter systems.

### Code Author
- **William G. C. Oropesa**

Main developer and author of the code. Responsible for the design, implementation, optimization, and validation of all CUDA-based simulations.

### Scientific Contributions
- **Pablo de Castro**
- **Danilo B. Liarte** *(owner of group)*

Provided key scientific ideas and conceptual guidance that strongly influenced the development of the models and numerical strategies implemented in this framework.

### Computational Support
- **André P. Vieira**

Provided continuous access to GPU cluster resources essential for the development, testing, and large-scale execution of the simulations.

---

## Acknowledgements

This work was supported by Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP) under grants #2024/23876-3, #2021/10139-2, and #2022/13872-5.  
We also acknowledge support from ICTP‑SAIFR, which is supported by FAPESP grant #2021/14335-0.


---

## License
This project is distributed under the terms specified in the ```LICENSE``` file.
