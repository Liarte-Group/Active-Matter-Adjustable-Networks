# Active-Matter-Adjustable-Networks
This repository contains a CUDA-C framework for simulating **active matter on deformable and dynamically adjustable networks**, with an emphasis on large-scale GPU acceleration.

The code is designed to efficiently simulate systems of **active Brownian particles (ABPs)** moving on lattice-based networks that can undergo **stochastic bond breaking and regeneration**. All simulations are implemented using massively parallel GPU computing, enabling the study of networks with very large system sizes.

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
- Simulations are **compiled on-the-fly** using `nvcc` within the submission script.
- CUDA architecture flags (`-arch=sm_xx`) may need to be adapted to the target GPU.

### Execution environment (recommended)
- Access to a GPU workstation or HPC cluster
- Optional job scheduler (e.g. SLURM, PBS, or equivalent)
- SSH access and filesystem permissions for batch execution

### Notes
- The code is written in **CUDA-C** and does not rely on external C++ libraries.
- Job submission scripts (`submit.sh`) are provided as templates and may require minor adaptation depending on the local system configuration.


## Authors

- **William G. C. Oropesa**:  *(Liarte-Group — ICTP-SAIFR / IFT-UNESP)*


## Acknowledgements

The author gratefully acknowledges **Prof. Danilo B. Liarte** for valuable scientific discussions and conceptual contributions that significantly influenced the design of this code.

The author also thanks **Prof. André P. Vieira** for providing continuous access to GPU cluster resources essential for the development and testing of this software.
