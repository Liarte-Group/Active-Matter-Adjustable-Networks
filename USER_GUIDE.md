# Compilation Guide

This document describes how the program is **compiled using CUDA (`nvcc`)**, including required dependencies, compile-time parameters, and GPU architecture settings. 

**Project**: `./Transport_Properties/<name>/` 
**Language**: CUDA-C
**Compiler**: NVCC (NVIDIA CUDA Compiler)  
**Architecture**: Ampere (sm_86)  
**Last Updated**: January 2026

⚠️ **Important:**  
This project intentionally recompiles the code for each parameter set. Physical and numerical parameters are passed at **compile time** via preprocessor macros to ensure strict reproducibility and a one-to-one correspondence between simulation parameters and the generated binary.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Automatic Compilation](#automatic-compilation)
4. [Manual Compilation](#manual-compilation)
5. [Compilation Flags](#compilation-flags)
6. [Practical Examples](#practical-examples)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### 1. CUDA Toolkit Installed
```bash
nvcc --version
```

**Expected Output:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Fri_Dec_13_20:28:41_PST_2024
Cuda compilation tools, release 12.4, V12.4.127
```

**If not installed:**
- Download from: https://developer.nvidia.com/cuda-downloads
- Follow official installation instructions

### 2. CUDA Environment Variables
Ensure CUDA is in your PATH:

```bash
# Verify PATH
echo $PATH | grep -i cuda

# If not present, add to ~/.bashrc or ~/.zshrc:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Apply changes
source ~/.bashrc  # or source ~/.zshrc
```

### 3. C Compiler
NVCC requires a compatible C compiler:

```bash
# Check GCC
gcc --version
```

**Recommended Versions:**
- GCC >= 9.x
---

## 📁 Project Structure

```
Transport_Properties/
├── submit.sh                    ← Main script
├── main.cu                      ← Main CUDA kernel
├── src/
│   ├── network.cu               ← Network implementation
│   └── network_kernels.cu       ← Additional CUDA kernels
├── include/
│   ├── network.h                ← Network headers
│   └── config.h
├── a.out                        ← Compiled executable (generated)
└── Realization_X/               ← Output directory (generated)
```

---

## ✅ Automatic Compilation

The `submit.sh` script handles all compilation automatically:

### Syntax
```bash
./submit.sh [LATTICE_TYPE] [LX] [PT] [HT] [PACKING_FRACTION]
```

### What the script does internally:
```bash
nvcc -arch=sm_86 \
  -Iinclude \
  -DLATTICE_TYPE=TRIANGULAR \
  -DLX=512 \
  -DPACKING_FRACTION=0.128 \
  -DP_PERST=0.8 \
  -DP_REGEN=0.015625 \
  -DNUMBER_OF_THREADS_PER_BLOCK=1024 \
  -Xcompiler -Wall,-Wextra \
  main.cu src/network.cu src/network_kernels.cu \
  -lcurand
```

### Automatic Example
```bash
./submit.sh TRIANGULAR 512 5.0 64 0.128
```

✅ **Advantage**: No need to worry about flags or configuration

---

## 🔨 Manual Compilation

If you need to compile manually (for debugging or testing):

### Basic Compilation
```bash
nvcc -arch=sm_86 -Iinclude main.cu src/network.cu src/network_kernels.cu -lcurand
```

### With Warnings
```bash
nvcc -arch=sm_86 \
  -Iinclude \
  -Xcompiler -Wall,-Wextra \
  main.cu src/network.cu src/network_kernels.cu \
  -lcurand
```

### With Parameter Defines
```bash
nvcc -arch=sm_86 \
  -Iinclude \
  -DLATTICE_TYPE=TRIANGULAR \
  -DLX=256 \
  -DPACKING_FRACTION=0.064 \
  -DP_PERST=0.8 \
  -DP_REGEN=0.015625 \
  -DNUMBER_OF_THREADS_PER_BLOCK=1024 \
  -Xcompiler -Wall,-Wextra \
  main.cu src/network.cu src/network_kernels.cu \
  -lcurand
```

### With Debugging
```bash
nvcc -g -G -arch=sm_86 \
  -Iinclude \
  -DLATTICE_TYPE=TRIANGULAR \
  -DLX=128 \
  -Xcompiler -Wall,-Wextra \
  main.cu src/network.cu src/network_kernels.cu \
  -lcurand -o debug_sim
```

Additional flags:
- `-g` - Debugging info for host code
- `-G` - Debugging info for device code
- `-o debug_sim` - Custom executable name

### With Optimizations
```bash
nvcc -O3 -arch=sm_86 \
  -Iinclude \
  -DLATTICE_TYPE=TRIANGULAR \
  -DLX=1024 \
  -Xcompiler -Wall,-Wextra \
  main.cu src/network.cu src/network_kernels.cu \
  -lcurand
```

Optimization flags:
- `-O0` - No optimization (default)
- `-O1` - Light optimization
- `-O2` - Moderate optimization
- `-O3` - Maximum optimization (may be slower in some cases)

---

## 🚩 Compilation Flags Explained

### Main Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `-arch=sm_XX` | GPU architecture | `-arch=sm_86` (Ampere) |
| `-I<dir>` | Include directory | `-Iinclude` |
| `-D<n>=<VAL>` | Preprocessor define | `-DLATTICE_TYPE=TRIANGULAR` |
| `-Xcompiler` | Flags for C++ compiler | `-Xcompiler -Wall,-Wextra` |
| `-l<lib>` | Link library | `-lcurand` |
| `-o <file>` | Output executable | `-o a.out` |

### Common CUDA Architectures

| Flag | GPU | CUDA Version |
|------|-----|--------------|
| `-arch=sm_61` | Pascal (GTX 1080) | 6.1 |
| `-arch=sm_70` | Volta (V100) | 7.0 |
| `-arch=sm_75` | Turing (T4, RTX 2080) | 7.5 |
| `-arch=sm_80` | Ampere (A40, A100) | 8.0 |
| `-arch=sm_86` | Ampere (RTX 30 series, A6000) | 8.6 |
| `-arch=sm_89` | Ada (RTX 40 series) | 8.9 |

### Dynamic Defines

Used to pass parameters to code without editing files:

```bash
# Common defines in this project
-DLATTICE_TYPE=TRIANGULAR          # Network type
-DLX=512                           # Linear size
-DPACKING_FRACTION=0.128           # Packing fraction
-DP_PERST=0.8                      # Persistence probability
-DP_REGEN=0.015625                 # Regeneration probability
-DNUMBER_OF_THREADS_PER_BLOCK=1024 # Threads per block
```

---

## 💡 Practical Examples

### Example 1: Quick Compilation (Testing)
```bash
nvcc -arch=sm_86 \
  -Iinclude \
  -DLATTICE_TYPE=TRIANGULAR \
  -DLX=128 \
  main.cu src/network.cu src/network_kernels.cu \
  -lcurand
```

**When to use**: Quick testing, development

---

### Example 2: Compilation with Full Warnings
```bash
nvcc -arch=sm_86 \
  -Iinclude \
  -DLATTICE_TYPE=SQUARE_MOORE \
  -DLX=256 \
  -DPACKING_FRACTION=0.064 \
  -DP_PERST=0.6 \
  -DP_REGEN=0.0625 \
  -DNUMBER_OF_THREADS_PER_BLOCK=512 \
  -Xcompiler -Wall,-Wextra,-Wpedantic \
  main.cu src/network.cu src/network_kernels.cu \
  -lcurand
```

**When to use**: Identify potential code issues

---

### Example 3: Optimized Compilation for Production
```bash
nvcc -O3 -arch=sm_86 \
  -Iinclude \
  -DLATTICE_TYPE=TRIANGULAR \
  -DLX=1024 \
  -DPACKING_FRACTION=0.128 \
  -DP_PERST=0.8 \
  -DP_REGEN=0.015625 \
  -DNUMBER_OF_THREADS_PER_BLOCK=1024 \
  -Xcompiler -Wall,-Wextra,-O3 \
  main.cu src/network.cu src/network_kernels.cu \
  -lcurand -o simulation_prod
```

**When to use**: Final simulations, maximum performance

---

### Example 4: Compilation with Debugging
```bash
nvcc -g -G -arch=sm_86 \
  -Iinclude \
  -DLATTICE_TYPE=TRIANGULAR \
  -DLX=256 \
  -Xcompiler -Wall,-Wextra,-g \
  main.cu src/network.cu src/network_kernels.cu \
  -lcurand -o simulation_debug
```

**When to use**: Debugging with cuda-gdb

Run with debugger:
```bash
cuda-gdb ./simulation_debug
(cuda-gdb) run
(cuda-gdb) break main.cu:100
(cuda-gdb) continue
```

---

### Example 5: Modular Compilation (Separate)
```bash
# Compile object files
nvcc -c -arch=sm_86 -Iinclude main.cu -o main.o
nvcc -c -arch=sm_86 -Iinclude src/network.cu -o network.o
nvcc -c -arch=sm_86 -Iinclude src/network_kernels.cu -o network_kernels.o

# Link
nvcc -arch=sm_86 main.o network.o network_kernels.o -lcurand -o a.out
```

**When to use**: Large projects, incremental compilation

---

### Example 6: Compilation with Environment Variables
```bash
#!/bin/bash

# Define parameters as variables
LATTICE="TRIANGULAR"
LX=512
PACKING=0.128
PT=5.0
HT=64

# Calculate probabilities
P_PERST=$(echo "scale=12; 1 - (1 / $PT)" | bc)
P_REGEN=$(echo "scale=12; 1 / $HT" | bc)

# Compile with variables
nvcc -O3 -arch=sm_86 \
  -Iinclude \
  -DLATTICE_TYPE=${LATTICE} \
  -DLX=${LX} \
  -DPACKING_FRACTION=${PACKING} \
  -DP_PERST=${P_PERST} \
  -DP_REGEN=${P_REGEN} \
  -DNUMBER_OF_THREADS_PER_BLOCK=1024 \
  -Xcompiler -Wall,-Wextra \
  main.cu src/network.cu src/network_kernels.cu \
  -lcurand
```

**When to use**: Automated scripts, batch processing

---

## 🔴 Troubleshooting

### ❌ Error: "nvcc: command not found"
```
bash: nvcc: command not found
```

**Solution:**
```bash
# Verify CUDA installation
ls -la /usr/local/cuda/bin/nvcc

# Add to PATH
export PATH=/usr/local/cuda/bin:$PATH

# Make permanent
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

---

### ❌ Error: "cannot find -lcurand"
```
nvcc fatal : Unsupported gpu architecture 'compute_20'
```

**Solution:**
```bash
# Verify libcurand exists
find /usr/local/cuda -name "*curand*"

# Add library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Compile specifying path
nvcc -L/usr/local/cuda/lib64 -lcurand main.cu ...
```

---

### ❌ Error: "error: identifier 'LATTICE_TYPE' is undefined"
```
error: identifier "LATTICE_TYPE" is undefined in function "main"
```

**Solution:**
You forgot the define. Add `-D`:
```bash
# ✗ Wrong
nvcc main.cu src/network.cu

# ✓ Correct
nvcc -DLATTICE_TYPE=TRIANGULAR main.cu src/network.cu
```

---

### ❌ Error: "fatal error: network.h: No such file or directory"
```
fatal error: network.h: No such file or directory
```

**Solution:**
Specify include directory:
```bash
# ✗ Wrong
nvcc main.cu src/network.cu

# ✓ Correct
nvcc -Iinclude main.cu src/network.cu
```

---

### ❌ Error: "Unsupported gpu architecture"
```
error : Unsupported gpu architecture 'compute_20'
```

**Solution:**
Verify your GPU architecture and use the correct flag:
```bash
# Check available GPU
nvidia-smi

# Compile with correct architecture
nvcc -arch=sm_86 main.cu ...  # For RTX 30 series
nvcc -arch=sm_80 main.cu ...  # For A40/A100
```

---

### ❌ Error: "compilation terminated"
```
compilation terminated.
```

**Solution:**
Usually due to syntax errors. Check:
```bash
# Compile with more detail
nvcc -arch=sm_86 main.cu src/network.cu src/network_kernels.cu 2>&1 | head -20

# Find exact error line
nvcc -arch=sm_86 main.cu 2>&1 | grep "error:"
```

---

### ⚠️ Warning: "long long"
```
warning: variable of type "long long" requires 8-byte alignment
```

**Solution:**
Generally not critical. To suppress:
```bash
nvcc -Xcompiler -Wall,-Wno-long-long main.cu ...
```

---

## 🔍 Post-Compilation Verification

### Verify executable was created
```bash
ls -lh a.out

# Expected output
-rwxr-xr-x 1 user group 5.2M Jan 26 14:30 a.out
```

### Verify it's a CUDA executable
```bash
file a.out

# Expected output
a.out: ELF 64-bit LSB executable, x86-64, version 1
```

### Test basic execution
```bash
./a.out

# Should run without symbol errors
```

---

## 📊 Compilation Flow Summary

```
                                           ┌──────────────────────────────────┐
                                           │    ./submit.sh [PARAMETERS]      │
                                           └────────────────┬─────────────────┘
                                                            │
                                                            ▼
                                           ┌──────────────────────────────────┐
                                           │      Parameter validation        │
                                           └────────────────┬─────────────────┘
                                                            │
                                                            ▼
                                           ┌──────────────────────────────────┐
                                           │       CFLAGS construction        │
                                           │    (Includes, Defines, Flags)    │
                                           └────────────────┬─────────────────┘
                                                            │
                                                            ▼
                                           ┌──────────────────────────────────┐
                                           │         CUDA compilation         │
                                           │  nvcc ${FLAGS} ${SRC} ${LDFLAGS} │
                                           └────────────────┬─────────────────┘
                                                            │
                                                            ▼
                                           ┌──────────────────────────────────┐
                                           │  ✓ Executable generated (a.out)  │
                                           └────────────────┬─────────────────┘
                                                            │
                                                            ▼
                                           ┌──────────────────────────────────┐
                                           │       Simulation execution       │
                                           │       ./a.out > output.dat       │
                                           └────────────────┬─────────────────┘
                                                            │
                                                            ▼
                                           ┌──────────────────────────────────┐
                                           │          ✓ Data saved            │
                                           └──────────────────────────────────┘
```

---

## 📚 Additional Resources

- **CUDA Toolkit Documentation**: https://docs.nvidia.com/cuda/
- **NVIDIA NVCC Manual**: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
- **Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **Debugging Guide**: https://docs.nvidia.com/cuda/cuda-gdb/

---

**Last Updated**: January 2026  
**Version**: 1.0