#ifndef __CONFIG_H__
#define __CONFIG_H__

/**
 * ============================================================================
 * Configuration Header: Active Matter on Adjustable Networks
 * ============================================================================
 *
 * Global compile-time configuration for Active Brownian Particle (ABP)
 * simulations on dynamic lattice networks.
 *
 * This file centralizes all tunable parameters:
 *   - System size and lattice geometry
 *   - Active particle dynamics (persistence, regeneration)
 *   - Monte Carlo simulation parameters
 *   - CUDA execution configuration
 *   - Visualization (OpenGL) settings
 *   - Color palette definitions
 *
 * All parameters can be overridden at compile time using -D flags.
 *
 * Author: William G. C. Oropesa
 * Institution: ICTP South American Institute for Fundamental Research
 * GitHub Repository: https://github.com/williamGOC/
 * Date: November 2025
 * ============================================================================
 */

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <curand_kernel.h>
#include <cuda_runtime.h>

// ========================================================================
// SYSTEM SIZE AND LATTICE PARAMETERS
// ========================================================================

// Linear size of the system (L x L lattice)
// Can be overridden with -DLX=...
#ifndef LX
#define LX 256
#endif

// Probability that a bond is regenerated (restored)
// Range: 0.0 (no regeneration) to 1.0 (all bonds regenerate)
// Can be overridden with -DP_REGEN=...
#ifndef P_REGEN
#define P_REGEN 0.00001
#endif

// Probability that a particle persists in its current direction
// Range: 0.0 (always change) to 1.0 (never change)
// Can be overridden with -DP_PERST=...
#ifndef P_PERST
#define P_PERST 0.8
#endif

// Packing fraction of active particles
// Range: 0.0 (empty) to 1.0 (full)
// Can be overridden with -DPACKING_FRACTION=...
#ifndef PACKING_FRACTION
#define PACKING_FRACTION 0.9999
#endif

// Distance between lattice points (lattice spacing)
// Can be overridden with -DDIST=...
#ifndef DIST
#define DIST 1
#endif

// ========================================================================
// MONTE CARLO SIMULATION PARAMETERS
// ========================================================================

// Number of MC steps before data collection begins (equilibration)
// Can be overridden with -DTHERMAL=...
#ifndef THERMAL
#define THERMAL 100000
#endif

// Total number of MC iterations for measurement
// Can be overridden with -DITERATION=...
#ifndef ITERATION
#define ITERATION 100000
#endif

// Data collection interval (sampling frequency)
// Measure every PASS_TIME Monte Carlo steps
// Can be overridden with -DPASS_TIME=...
#ifndef PASS_TIME
#define PASS_TIME 100
#endif

// ========================================================================
// CUDA EXECUTION CONFIGURATION
// ========================================================================

// Number of threads per CUDA block
// Typical values: 256, 512, 1024
// Can be overridden with -DNUMBER_OF_THREADS_PER_BLOCK=...
#ifndef NUMBER_OF_THREADS_PER_BLOCK
#define NUMBER_OF_THREADS_PER_BLOCK 1024
#endif

// Lattice type selection (default: TRIANGULAR)
// Options: SQUARE_MOORE (8 neighbors), TRIANGULAR (6 neighbors)
// Can be overridden with -DLATTICE_TYPE=...
#ifndef LATTICE_TYPE
#define LATTICE_TYPE TRIANGULAR
#endif

// ========================================================================
// DERIVED PARAMETERS
// ========================================================================

// Total number of lattice sites
#define N (LX * LX)

// ========================================================================
// CUDA ERROR HANDLING
// ========================================================================

// Macro to check CUDA errors
// Usage: HANDLE_ERROR(cudaMalloc(...));
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// CUDA error handler function
// Prints error message and exits if CUDA call fails
inline void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n",
                file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // No action needed if there is no error
}

// ========================================================================
// LATTICE GEOMETRY - SHIFT VECTORS
// ========================================================================

// Square lattice with Moore neighbors (z = 8)
// Displacement vectors: horizontal and vertical shifts per direction
static const double shift_0_sm[8] = {DIST, DIST, 0, -DIST, -DIST, -DIST, 0, DIST};
static const double shift_1_sm[8] = {0, DIST, DIST, DIST, 0, -DIST, -DIST, -DIST};

// Triangular lattice (z = 6)
// Hexagonal close-packed displacement vectors
static const double shift_0_tr[6] = {DIST, DIST / 2.0, -DIST / 2.0, -DIST, -DIST / 2.0, DIST / 2.0};
static const double shift_1_tr[6] = {0, sqrt(3) * DIST / 2.0, sqrt(3) * DIST / 2.0, 0, -sqrt(3) * DIST / 2.0, -sqrt(3) * DIST / 2.0};

#endif