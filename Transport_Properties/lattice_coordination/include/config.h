#ifndef __CONFIG_H__
#define __CONFIG_H__

/**
 * ============================================================================
 * Configuration Header: Active Matter on Adjustable Networks
 * ============================================================================
 *
 * Global compile-time configuration for Active Brownian Particle (ABP)
 * simulations on dynamic networks.
 *
 * This file centralizes all tunable parameters:
 *   - System size and network geometry
 *   - Active particle dynamics (persistence, regeneration)
 *   - Monte Carlo simulation parameters
 *   - CUDA execution configuration
 *
 * All parameters can be overridden at compile time using -D flags.
 *
 * Reference:
 *   // TODO
 *
 * Author: William G. C. Oropesa
 * Institution: ICTP South American Institute for Fundamental Research
 * GitHub Repository: https://github.com/Liarte-Group/Active-Matter-Adjustable-Networks
 * Date: January 2026
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
// Linear size of the system (LX x LX lattice)
// Can be overridden at compile time with -DLX=...
// ========================================================================
#ifndef LX
#define LX 256
#endif

// ========================================================================
// Probability that a bond is regenerated (restored)
// Value between 0 (no bonds regenerated) and 1 (all bonds regenerated)
// Can be overridden with -DP_REGEN=...
//
// Physically, this parameter controls the healing dynamics of the network.
// ========================================================================
#ifndef P_REGEN
#define P_REGEN 0.001
#endif

// ========================================================================
// Probability that a particle persists in its current direction
// Value between 0 (always change) and 1 (never change)
// Can be overridden with -DP_PERST=...
//
// This parameter controls the persistence time of active motion.
// ========================================================================
#ifndef P_PERST
#define P_PERST 0.8
#endif

// ========================================================================
// Packing-fraction of active particles
// Density of occupied sites in the lattice (from 0 to 1)
// Can be overridden with -DPACKING_FRACTION=...
// ========================================================================
#ifndef PACKING_FRACTION
#define PACKING_FRACTION 0.9999
#endif

// ========================================================================
// Distance between lattice points (lattice spacing)
// Can be overridden with -DDIST=...
//
// Sets the physical length scale of the lattice.
// ========================================================================
#ifndef DIST
#define DIST 1
#endif

// ========================================================================
// Number of Monte Carlo steps before data collection begins
// Allows the system to reach thermal equilibrium
// Can be overridden with -DTHERMAL=...
// ========================================================================
#ifndef THERMAL
#define THERMAL 100000
#endif

// ========================================================================
// Total number of Monte Carlo iterations
// Number of steps after thermalization during which data is collected
// Can be overridden with -DITERATION=...
// ========================================================================
#ifndef ITERATION
#define ITERATION 100000
#endif

// ========================================================================
// Data collection interval (sampling frequency)
// Collect measurements every PASS_TIME Monte Carlo steps
// Reduces correlation between successive measurements
// Can be overridden with -DPASS_TIME=...
// ========================================================================
#ifndef PASS_TIME
#define PASS_TIME 100
#endif

// ========================================================================
// CUDA configuration: threads per block
// Typical values: 256, 512, 1024
// Can be overridden with -DNUMBER_OF_THREADS_PER_BLOCK=...
//
// This parameter affects GPU occupancy and performance.
// ========================================================================
#ifndef NUMBER_OF_THREADS_PER_BLOCK
#define NUMBER_OF_THREADS_PER_BLOCK 1024
#endif

// ========================================================================
// Lattice type selection (default: TRIANGULAR)
// 0 = SQUARE_MOORE (8 neighbors)
// 1 = TRIANGULAR (6 neighbors)
// Can be overridden with -DLATTICE_TYPE=...
//
// Determines the connectivity and neighbor geometry.
// ========================================================================
#ifndef LATTICE_TYPE
#define LATTICE_TYPE TRIANGULAR
#endif

// ========================================================================
// Total number of lattice sites
// ========================================================================
#define N (LX * LX)

// ========================================================================
// Macro to check CUDA errors
// Usage: HANDLE_ERROR(cudaMalloc(...));
// ========================================================================
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// ========================================================================
// CUDA error handler function
// Prints error message and exits if CUDA call fails
// Ensures safe CUDA execution and early failure detection
// ========================================================================
inline void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n",
                file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // No action needed if there is no error
}

// ========================================================================
// Host projectors for each lattice type
//
// These arrays define the displacement vectors to neighboring sites
// for each lattice geometry.
// ========================================================================

// Square lattice with Moore neighbors (z = 8)
static const double shift_0_sm[8] = {DIST, DIST, 0, -DIST, -DIST, -DIST, 0, DIST};
static const double shift_1_sm[8] = {0, DIST, DIST, DIST, 0, -DIST, -DIST, -DIST};

// Triangular regular (z = 6)
static const double shift_0_tr[6] = {DIST, DIST / 2.0, -DIST / 2.0, -DIST, -DIST / 2.0, DIST / 2.0};
static const double shift_1_tr[6] = {0, sqrt(3) * DIST / 2.0, sqrt(3) * DIST / 2.0, 0, -sqrt(3) * DIST / 2.0, -sqrt(3) * DIST / 2.0};

#endif