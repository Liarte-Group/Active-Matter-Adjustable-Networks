# Compilation Guide

This document describes how the program is **compiled using CUDA (`nvcc`)**, including required dependencies, compile-time parameters, and GPU architecture settings. 

⚠️ **Important:**  
This project intentionally recompiles the code for each parameter set. Physical and numerical parameters are passed at **compile time** via preprocessor macros to ensure strict reproducibility and a one-to-one correspondence between simulation parameters and the generated binary.

---

## 1. System Requirements

To compile the program, the following requirements must be satisfied:

- Linux operating system
- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested with CUDA ≥ 11)
- `nvcc` available in the system `PATH`
- `bc` (used for floating-point arithmetic in the compilation script)

Check CUDA installation:
```bash
nvcc --version
```

--
