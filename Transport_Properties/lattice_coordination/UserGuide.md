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

---

## 2. CUDA Environment Setup

The compilation script automatically configures the CUDA environment assuming a standard installation:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

```
No manual setup is required unless CUDA is installed in a non-standard location.

---

## 3. Compilation Strategy

Compilation is handled internally by the execution script (`submith.sh`).
For each simulation, the program is compiled using `nvcc` with simulation parameters injected via `-D` flags.

This design guarantees:
 - Full reproducibility
 - No ambiguity between runtime input and compiled physics
 - One binary per parameter set

 ---