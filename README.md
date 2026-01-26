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
├── ACKNOWLEDGEMENTS.md
├── CONTRIBUTING.md
├── USER_GUIDE.md
├── LICENSE
├── Transport_Properties/
│   ├── lattice_coordination/
│   ├── msd_and_dynamical_heterogeneity/
│   ├── non_steric_interactions/
│   └── stopping_class/
```
Each study directory in `Transport_Properties/<project>/` follows the same internal structure:

```text
├── main.cu        # Main CUDA entry point
├── submit.sh      # GPU cluster submission script
├── src/           # CUDA source files and kernels
└── include/       # Headers and configuration files

```
Detailed instructions are provided in the ```USER_GUIDE.md``` file.


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
- **Hartmut Löwen**
- **Danilo B. Liarte**

Provided key scientific ideas and conceptual guidance that strongly influenced the development of the models and numerical strategies implemented in this framework.

*Note:* This project's documentation (Markdown guides, user manuals, and compilation guides) was created with assistance from **Claude AI (Anthropic)**. All content has been reviewed and validated by the development team. The core simulation code, scientific methodology, and research approach are authored by **Liarte-Group**.

---

## Citation
If you use this code in your research, please cite:

 - William G. C. Oropesa *et. al.*, "Active Matter on Adjustable Networks", Zenodo, DOI: [to be assigned]

<!-- Actualizar DOI cuando Zenodo lo genere. -->
---

## License
This project is distributed under the terms of the **GNU General Public License v3.0 (GPL-3.0)**.
You are free to use, modify, and distribute the software under the conditions of GPL-3.0 (see ```LICENSE``` file).
