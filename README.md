# Active-Matter-Adjustable-Networks
This repository contains a CUDA-C framework for simulating **active matter on deformable and dynamically adjustable networks**, with an emphasis on large-scale GPU acceleration.

The code is designed to efficiently simulate systems of **active Brownian particles (ABPs)** moving on lattice-based networks that can undergo **stochastic bond breaking and regeneration**. All simulations are implemented using massively parallel GPU computing, enabling the study of networks with very large system sizes.

## Repository structure
The repository is organized around **article-specific directories**, each containing a self-contained implementation of the corresponding study.
```text
.
├── README.md                     # General overview of the project
├── LICENSE                       # License information
├── Transport_Properties/
│   ├── README.md                 # Article-specific description and instructions
│   ├── main.cu                   # Main CUDA entry point for simulations
│   ├── submit.sh                 # Job submission script (HPC / GPU clusters)
│   ├── src/
│   │   ├── network.cu            # Network construction and update logic
│   │   └── network_kernels.cu    # CUDA kernels for network and particle dynamics
│   └── include/
│       ├── config.h              # Global parameters and compile-time configuration
│       └── network.h             # Network data structures and function declarations

```
