#ifndef __CONFIG_H__
#define __CONFIG_H__

/*
 * Active Matter on Adjustable Networks
 *
 * Global configuration header.
 *
 * This file defines all compile-time parameters controlling:
 *   - System size and lattice geometry
 *   - Active particle dynamics (persistence)
 *   - Bond regeneration dynamics
 *   - Monte Carlo simulation parameters
 *   - CUDA execution configuration
 *   - Visualization (OpenGL) settings
 *
 * All parameters can be overridden at compile time using -D flags.
 *
 * Reference:
 *   Active Matter on Adjustable Networks I: Transport Properties
 */

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <curand_kernel.h>
#include <cuda_runtime.h>

// --------------------------------------------
// Linear size of the system (L x L lattice)
// Can be overridden at compile time with -DLX=...
// --------------------------------------------
#ifndef LX
#define LX 256
#endif

// --------------------------------------------
// Probability that a bond is regenerated (restored)
// Value between 0 (no bonds regenerated) and 1 (all bonds regenerated)
// Can be overridden with -DP_REGEN=...
//
// Physically, this parameter controls the healing dynamics of the network.
// --------------------------------------------
#ifndef P_REGEN
#define P_REGEN 0.00001
#endif

// ---------------------------------------------
// Probability that a particle persists in its current direction (does not change direction)
// Value between 0 (particle always changes direction) and 1 (particle never changes direction)
// Can be overridden with -DP_PERST=...
//
// This parameter controls the persistence time of active motion.
// ---------------------------------------------
#ifndef P_PERST
#define P_PERST 0.8
#endif

// ---------------------------------------------
// Packing fraction of active particles
// Density of occupied sites in the lattice (0 to 1)
// Can be overridden with -DPACKING_FRACTION=...
// ---------------------------------------------
#ifndef PACKING_FRACTION
#define PACKING_FRACTION 0.9999
#endif

// --------------------------------------------
// Distance between lattice points (lattice spacing)
// Can be overridden with -DDIST=...
//
// Sets the physical length scale of the lattice.
// --------------------------------------------
#ifndef DIST
#define DIST 1
#endif

// --------------------------------------------
// Number of Monte Carlo steps before data collection begins
// Allows the system to reach thermal equilibrium
// Can be overridden with -DTHERMAL=...
// --------------------------------------------
#ifndef THERMAL
#define THERMAL 100000
#endif

//---------------------------------------------
// Total number of Monte Carlo iterations
// Number of steps after thermalization during which data is collected
// Can be overridden with -DITERATION=...
// --------------------------------------------
#ifndef ITERATION
#define ITERATION 100000
#endif

//---------------------------------------------
// Data collection interval (sampling frequency)
// Collect measurements every PASS_TIME Monte Carlo steps
// Reduces correlation between successive measurements
// Can be overridden with -DPASS_TIME=...
// --------------------------------------------
#ifndef PASS_TIME
#define PASS_TIME 100
#endif

// --------------------------------------------
// CUDA configuration: threads per block
// Typical values: 256, 512, 1024
// Can be overridden with -DNUMBER_OF_THREADS_PER_BLOCK=...
//
// This parameter affects GPU occupancy and performance.
// --------------------------------------------
#ifndef NUMBER_OF_THREADS_PER_BLOCK
#define NUMBER_OF_THREADS_PER_BLOCK 1024
#endif

// --------------------------------------------
// Lattice type selection (default: SQUARE_MOORE)
// 0 = SQUARE_MOORE
// 1 = TRIANGULAR
// Can be overridden with -DLATTICE_TYPE=...
//
// Determines the connectivity and neighbor geometry.
// --------------------------------------------
#ifndef LATTICE_TYPE
#define LATTICE_TYPE TRIANGULAR
#endif



// Total number of lattice sites
#define N (LX * LX)

// --------------------------------------------
// Macro to check CUDA errors
// Usage: HANDLE_ERROR(cudaMalloc(...));
// --------------------------------------------
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))


// Inline function to print error message and exit if CUDA call fails
// Ensures safe CUDA execution and early failure detection
inline void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n",
                file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // No action needed if there is no error
}

// --------------------------------------------
// Device projectors for each lattice type
//
// These arrays define the displacement vectors to neighboring sites
// for each lattice geometry.
// --------------------------------------------


// Square lattice with Moore neighbors (z = 8)
static const double shift_0_sm[8] = {DIST, DIST, 0, -DIST, -DIST, -DIST, 0, DIST};
static const double shift_1_sm[8] = {0, DIST, DIST, DIST, 0, -DIST, -DIST, -DIST};

// Triangular regular (z = 6)
static const double shift_0_tr[6] = {DIST, DIST / 2.0, -DIST / 2.0, -DIST, -DIST / 2.0, DIST / 2.0};
static const double shift_1_tr[6] = {0, sqrt(3) * DIST / 2.0, sqrt(3) * DIST / 2.0, 0, -sqrt(3) * DIST / 2.0, -sqrt(3) * DIST / 2.0};


// --------------------------------------------
// Visualization parameters (OpenGL)
// --------------------------------------------

// Number of line segments used to approximate circles
#define NUM_SEGMENTS 15

// Line width for rendering bonds or lattice edges
#define LINE_WIDTH 1.5f

// Relative radius of particle representation
#define CIRCLE_RADIUS_FACTOR 0.25

// --------------------------------------------
// Color definitions (RGB)
// Used for particle and lattice visualization
// --------------------------------------------

// Colores existentes
#define BLACK   0.0, 0.0, 0.0
#define RED     1.0, 0.0, 0.0
#define GREEN   0.0, 1.0, 0.0
#define BLUE    0.0, 0.0, 1.0
#define YELLOW  1.0, 1.0, 0.0
#define CYAN    0.0, 1.0, 1.0
#define MAGENTA 1.0, 0.0, 1.0
#define ORANGE  1.0, 0.5, 0.0
#define PURPLE  0.5, 0.0, 0.5
#define PINK    1.0, 0.4, 0.7
#define BROWN   0.6, 0.3, 0.1
#define GRAY    0.5, 0.5, 0.5
#define LIGHTGRAY 0.8, 0.8, 0.8
#define DARKGRAY  0.3, 0.3, 0.3
#define LIME    0.5, 1.0, 0.0
#define SKYBLUE 0.0, 0.5, 1.0
#define GOLD    1.0, 0.84, 0.0
#define SILVER  0.75, 0.75, 0.75
#define MAROON  0.5, 0.0, 0.0
#define NAVY    0.0, 0.0, 0.5
#define OLIVE   0.5, 0.5, 0.0
#define TEAL    0.0, 0.5, 0.5
#define MINT        0.6, 1.0, 0.6
#define CORAL       1.0, 0.5, 0.4
#define SALMON      1.0, 0.6, 0.5
#define TURQUOISE   0.25, 0.88, 0.82
#define INDIGO      0.29, 0.0, 0.51
#define LAVENDER    0.9, 0.9, 0.98
#define BEIGE       0.96, 0.96, 0.86
#define PEACH       1.0, 0.9, 0.71
#define PLUM        0.87, 0.63, 0.87
#define KHAKI       0.76, 0.69, 0.57
#define AQUA        0.0, 1.0, 1.0
#define CHOCOLATE   0.82, 0.41, 0.12
#define CRIMSON     0.86, 0.08, 0.24
#define FUCHSIA     1.0, 0.0, 1.0
#define IVORY       1.0, 1.0, 0.94
#define TAN         0.82, 0.71, 0.55
#define SEAGREEN    0.18, 0.55, 0.34
#define SLATEBLUE   0.42, 0.35, 0.80
#define TOMATO      1.0, 0.39, 0.28
#define WHEAT       0.96, 0.87, 0.7

// --------------------------------------------
// External color array
// Defined elsewhere and used for rendering
// --------------------------------------------
extern float colors[][3];

// Number total of available colors
#define NUM_COLORS (sizeof(colors) / sizeof(colors[0]))


#endif
