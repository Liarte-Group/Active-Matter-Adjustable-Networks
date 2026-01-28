/**
 * ============================================================================
 * CUDA Kernels for Active Brownian Particles (ABP) Simulation on Lattice
 * ============================================================================
 * 
 * This file contains optimized CUDA kernels for simulating active Brownian
 * particles on various 2D lattice types with dynamic bond formation/breaking.
 * 
 * Lattice types supported:
 *   - SQUARE_MOORE (z = 8):         8-neighbor Moore neighborhood
 *   - TRIANGULAR   (z = 6):         6-neighbor standard triangular
 * 
 * Key features:
 *   - Periodic boundary conditions (toroidal topology)
 *   - Atomic operations for thread-safe updates
 *   - Parallel reduction for global statistics
 *   - Thread-safe random number generation with cuRAND
 * 
 * Author: William G. C. Oropesa (Liarte-Group)
 * Institution: ICTP South American Institute for Fundamental Research
 * GitHub Repository: https://github.com/Liarte-Group/Active-Matter-Adjustable-Networks
 * Date: January 2026
 * ============================================================================
 */

#include "config.h"
#include "network.h"


__global__ void setupCurandState(curandState *states, const int size, unsigned long seed) {
    
    // Compute global thread index across entire grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ========================================================================
    // Initialize RNG state only for valid indices
    // ========================================================================
    // Threads with idx >= size exit early (warp divergence is minimal)
    // This is safe because grid is typically sized conservatively
    if (idx < size) {
        // curand_init(seed, sequence, offset, state)
        // 
        // Parameters:
        //   seed     - base seed for reproducibility
        //   sequence - unique identifier for this thread (idx)
        //              ensures independent random sequences between threads
        //   offset   - additional offset within the sequence (0 for standard use)
        //   state    - pointer to this thread's curandState to initialize
        //
        // Each thread gets a completely independent RNG stream thanks to
        // the unique sequence number, allowing safe concurrent random generation
        curand_init(seed, idx, 0, &states[idx]);
    }
}



__global__ void kerGetNeighborList(int *neighbor, const int z, LatticeType type) {
    
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ========================================================================
    // Boundary check: Exit if thread index exceeds total number of sites
    // ========================================================================
    if (idx >= N) return;

    // ========================================================================
    // Convert linear index to 2D lattice coordinates
    // ========================================================================
    // 2D position (i, j) from linear index using row-major ordering
    // - i: column/x-coordinate (horizontal, 0 to LX-1, left to right)
    // - j: row/y-coordinate (vertical, 0 to LX-1, top to bottom)
    int i = idx % LX;    // Column (horizontal position)
    int j = idx / LX;    // Row (vertical position)

    // ========================================================================
    // Compute neighbors based on lattice type and periodic boundary conditions
    // ========================================================================
    // All calculations use modulo (%) to wrap around at edges (toroidal topology)
    // neighbor[z * idx + dir] stores the index of the neighbor in direction dir
    
    switch(type) {
        // ====================================================================
        // SQUARE_MOORE: 8 neighbors (Moore neighborhood)
        // ====================================================================
        // Cardinal directions + diagonal directions (4+4 = 8 neighbors)
        // Coordination number z = 8
        // Ordering: right, up-right, up, up-left, left, down-left, down, down-right
        case SQUARE_MOORE:
            neighbor[z * idx + 0] = ((i + 1) % LX) + j * LX;                         // Right:      (i+1, j)
            neighbor[z * idx + 1] = ((i + 1) % LX) + ((j + 1) % LX) * LX;            // Up-right:   (i+1, j+1)
            neighbor[z * idx + 2] = i + ((j + 1) % LX) * LX;                         // Up:         (i, j+1)
            neighbor[z * idx + 3] = ((i - 1 + LX) % LX) + ((j + 1) % LX) * LX;       // Up-left:    (i-1, j+1)
            neighbor[z * idx + 4] = ((i - 1 + LX) % LX) + j * LX;                    // Left:       (i-1, j)
            neighbor[z * idx + 5] = ((i - 1 + LX) % LX) + ((j - 1 + LX) % LX) * LX;  // Down-left:  (i-1, j-1)
            neighbor[z * idx + 6] = i + ((j - 1 + LX) % LX) * LX;                    // Down:       (i, j-1)
            neighbor[z * idx + 7] = ((i + 1) % LX) + ((j - 1 + LX) % LX) * LX;       // Down-right: (i+1, j-1)
            break;

        // ====================================================================
        // TRIANGULAR: 6 neighbors (standard triangular lattice)
        // ====================================================================
        // Triangular lattice with different neighbor pattern than RIGHT_TRIANGULAR
        // Coordination number z = 6
        case TRIANGULAR:
            neighbor[z * idx + 0] = ((i + 1) % LX) + j * LX;                         // Right:      (i+1, j)
            neighbor[z * idx + 1] = i + ((j + 1) % LX) * LX;                         // Up-right:   (i, j+1)
            neighbor[z * idx + 2] = ((i - 1 + LX) % LX) + ((j + 1) % LX) * LX;       // Up-left:    (i-1, j+1)
            neighbor[z * idx + 3] = ((i - 1 + LX) % LX) + j * LX;                    // Left:       (i-1, j)
            neighbor[z * idx + 4] = i + ((j - 1 + LX) % LX) * LX;                    // Down-left:  (i, j-1)
            neighbor[z * idx + 5] = ((i + 1) % LX) + ((j - 1 + LX) % LX) * LX;       // Down-right: (i+1, j-1)
            break;
    }
}


__global__ void kerSetBonds(int *bond, const int z, const int value) {
    
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ========================================================================
    // Boundary check: Exit if thread index exceeds total number of bonds
    // ========================================================================
    // Total number of bonds = N * z (z bonds per each of N sites)
    if (idx >= z * N) return;

    // ========================================================================
    // Set bond value at this index
    // ========================================================================
    // Simple assignment: each thread independently writes to one element
    // No dependencies between threads -> fully parallel execution
    // 
    // Common values:
    //   value = 1: Bond is active/connected (allow particle movement)
    //   value = 0: Bond is broken/removed (block particle movement)
    bond[idx] = value;
}


__global__ void updateParticles(int *site, int *neighbor, int *bond, int *index, int *direction,
 int nParticles, int dim, int z, curandState *states) {

    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ========================================================================
    // Boundary check: Exit if thread does not correspond to a valid particle
    // ========================================================================
    if (idx >= nParticles) return;

    // Get current lattice index of the particle
    int oldIdx = index[idx];

    // Load RNG state for this thread/particle
    curandState localState = states[idx];

    // ========================================================================
    // PERSISTENCE: With probability (1 - P_PERST), change direction
    // ========================================================================
    // P_PERSIST is the probability that a particle maintains its direction
    // (1 - P_PERST) is the probability it changes direction
    // Example: P_PERST = 1.0 -> always changes; P_PERST = 0.0 -> never changes
    if (curand_uniform(&localState) < (1.0f - P_PERST)) {
        direction[idx] = curand(&localState) % z;  // Assign new random direction
    }

    // Read particle's current direction
    int dir = direction[idx];

    // ========================================================================
    // MOVEMENT: Try to move to neighboring site
    // ========================================================================
    
    // Compute candidate new position using neighbor list
    // neighbor array layout: neighbor[z * siteIdx + direction]
    int newIdx = neighbor[z * oldIdx + dir];

    // Check if bond exists (bond is connected)
    int validBond = bond[z * oldIdx + dir];

    if (validBond) {
        // Try to occupy the new site atomically
        // atomicCAS: Compare-And-Swap
        //   if site[newIdx] == expected (0), set it to 1 and return 0
        //   otherwise, return current value (collision!)
        int expected = 0;
        if (atomicCAS(&site[newIdx], expected, 1) == 0) {
            // Movement successful: free old site and update particle's index
            atomicExch(&site[oldIdx], 0);
            index[idx] = newIdx;

            // ================================================================
            // BOND DEACTIVATION: Remove bidirectional bond after use
            // ================================================================
            // Deactivate used bond in both directions to prevent simultaneous use
            // Opposite direction: (dir + z/2) % z (pointing back from newIdx to oldIdx)
            // This maintains consistency: forward/backward bond pair are always equal
            atomicExch(&bond[z * oldIdx + dir], 0);
            atomicExch(&bond[z * newIdx + (dir + z / 2) % z], 0);
        }
        // If atomicCAS fails, another particle beat us to the site -> nothing happens
    }

    // Save updated RNG state
    states[idx] = localState;
}



/*
The bond regeneration step is implemented using an early-exit strategy to minimize unnecessary random number generation 
and atomic operations, ensuring optimal GPU performance without altering the underlying stochastic dynamics
*/
__global__ void updateBonds(int *bond, int *neighbor, int z, curandState *bondStates) {

    // ========================================================================
    // Compute global bond index: one thread per (site, direction)
    // ========================================================================
    int bondIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // ========================================================================
    // Boundary check: Exit if this thread corresponds to an invalid bond
    // ========================================================================
    // Total directional bonds = N * z
    if (bondIdx >= N * z) return;

    // ========================================================================
    // Extract site index and direction from linear bond index
    // ========================================================================
    // bondIdx = z * siteIdx + dir
    int siteIdx = bondIdx / z;     // site index ∈ [0, N-1]
    int dir     = bondIdx % z;     // direction  ∈ [0, z-1]

    // ========================================================================
    // Neighbor site in this direction
    // ========================================================================
    int neighborIdx = neighbor[z * siteIdx + dir];

    // ========================================================================
    // AVOID DOUBLE-COUNTING:
    // Process each undirected bond only once
    // ========================================================================
    // Only threads with siteIdx < neighborIdx are responsible for updates
    if (siteIdx >= neighborIdx) return;

    // ========================================================================
    // Skip if bond is already active
    // ========================================================================
    // bond == 0 → broken bond
    // bond == 1 → active bond
    if (bond[bondIdx] != 0) return;

    // ========================================================================
    // Load RNG state *only when needed*
    // ========================================================================
    curandState localState = bondStates[bondIdx];

    // ========================================================================
    // REGENERATION:
    // With probability P_REGEN, activate a broken bond
    // ========================================================================
    if (curand_uniform(&localState) < P_REGEN) {

        // ====================================================================
        // Atomically activate both directions of the bond
        // ====================================================================
        // Forward  direction: (siteIdx, dir)
        atomicExch(&bond[bondIdx], 1);

        // Backward direction: (neighborIdx, opposite_dir)
        // opposite_dir = (dir + z/2) mod z
        atomicExch(&bond[z * neighborIdx + (dir + z / 2) % z], 1);
    }

    // ========================================================================
    // Save updated RNG state
    // ========================================================================
    bondStates[bondIdx] = localState;
}



// CUDA kernel to compute the coordination number (number of occupied neighbors) 
// for each particle and build a histogram of coordination frequencies.
__global__ void getZDistb(const int *bond, int z, int *coordination) {

    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process if thread corresponds to a valid particle
    if (idx >= N) return; 

    int zIdx = 0;
    
    // Loop over all z neighbors of the current particle
    for (int dir = 0; dir < z; dir++) {
        zIdx += bond[z * idx + dir];
    }

    atomicAdd(coordination, zIdx);
}