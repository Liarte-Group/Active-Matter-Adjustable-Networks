/**
 * ============================================================================
 * Main Program: Active Brownian Particles - Stop Reason Classification
 * ============================================================================
 *
 * This program simulates ABP dynamics on an adjustable network and measures
 * the frequency of different particle stop reasons over time.
 *
 * Workflow:
 *   1. Create and initialize network structure
 *   2. Setup random number generators
 *   3. Equilibrate system (THERMAL steps)
 *   4. Measure stop reason statistics periodically (ITERATION steps)
 *   5. Output stop reason counts vs time
 *   6. Clean up and exit
 *
 * Output format:
 *   Column 1: Time (in MC steps)
 *   Column 2: Stop reason count 0
 *   Column 3: Stop reason count 1
 *   Column 4: Stop reason count 2
 *   Column 5: Stop reason count 3
 *
 * Author: William G. C. Oropesa (Liarte-Group)
 * Institution: ICTP South American Institute for Fundamental Research
 * GitHub Repository: https://github.com/Liarte-Group/Active-Matter-Adjustable-Networks
 * Date: January 2026
 * ============================================================================
 */

#include "config.h"
#include "network.h"

int main() {
    
    // ========================================================================
    // STEP 1: Create and initialize network
    // ========================================================================
    // Set spatial dimension (2D system)
    int dim = 2;

    // ========================================================================
    // Select lattice type from compile-time configuration
    // LATTICE_TYPE is defined in config.h
    // Options: SQUARE_MOORE (z=8), TRIANGULAR (z=6)
    // ========================================================================
    LatticeType type = LATTICE_TYPE;

    // ========================================================================
    // Allocate and initialize network structure on GPU/CPU
    // Parameters (from config.h):
    //   type              - Lattice geometry (SQUARE_MOORE or TRIANGULAR)
    //   dim               - Spatial dimension (2 for 2D)
    //   PACKING_FRACTION  - Fraction of sites occupied by particles
    //   P_REGEN           - Bond regeneration probability
    //   P_PERST           - Particle persistence probability
    //
    // This function:
    //   - Allocates GPU/CPU memory for all arrays
    //   - Initializes lattice topology (neighbor list)
    //   - Places particles randomly on lattice (Fisher-Yates shuffle)
    //   - Sets all bonds to active state
    //   - Initializes stop reason tracking array
    //   - Returns pointer to initialized network structure
    // ========================================================================
    network *pN = makeNetwork(
        type,               // Lattice type
        dim,                // Dimension (2D)
        PACKING_FRACTION,   // Packing fraction
        P_REGEN,            // Bond regeneration probability
        P_PERST             // Particle persistence probability
    );

    // ========================================================================
    // Error checking: verify network was successfully created
    // ========================================================================
    if (!pN) {
        fprintf(stderr, "Failed to create network!\n");
        return EXIT_FAILURE;
    }

    // ========================================================================
    // Store number of particles for kernel configuration
    // ========================================================================
    // Caching particle count for efficient kernel launch calculations
    // nParticles used to configure thread blocks for reduction kernels
    int nParticles = pN -> nParticles;

    // ========================================================================
    // Calculate GPU memory size for reduction accumulator
    // ========================================================================
    // Allocate space for 4 stop reason counters (one integer each)
    // Each counter tracks particles with a specific stop reason
    // Stop reasons: 0, 1, 2, 3 (physical interpretation depends on model)
    // Total memory = 4 * sizeof(int) bytes
    size_t memoryCounts = 4 * sizeof(int);

    // ========================================================================
    // STEP 2: Initialize random number generators
    // ========================================================================
    // Setup independent cuRAND states for:
    //   - Each bond (for bond regeneration stochasticity)
    //   - Each particle (for movement and direction change)
    //
    // This ensures thread-safe, reproducible random number generation
    // on the GPU with independent sequences per thread
    initCurand(pN);

    // ========================================================================
    // STEP 3: Equilibration phase (thermalization)
    // ========================================================================
    // Run THERMAL Monte Carlo steps to reach steady state
    // - Removes memory of initial configuration
    // - Allows network to reorganize
    // - Allows particles to redistribute
    // - Allows stop reason statistics to stabilize
    //
    // THERMAL parameter defined in config.h (default: 100000 steps)
    // During equilibration, no measurements are collected
    // No measurements are output during this phase
    while (pN -> iter < THERMAL) {
        // Execute one MC step:
        // - updateParticles: Move and reorient particles, track stop reasons
        // - updateBonds: Regenerate broken bonds stochastically
        mcStep(pN);
    }

    // ========================================================================
    // Reset iteration counter for measurement phase
    // ========================================================================
    // Set counter to 0 to count measurement steps from t=0
    pN -> iter = 0;

    // ========================================================================
    // STEP 4: Allocate GPU memory for stop reason counters
    // ========================================================================
    // Temporary GPU buffer to accumulate stop reason statistics
    // Size: 4 integers (one accumulator per stop reason: 0, 1, 2, 3)
    // This will hold the output from reduceStopCauses kernel
    // Memory layout: [count_reason0, count_reason1, count_reason2, count_reason3]
    int* devPtrCounts;
    HANDLE_ERROR(cudaMalloc(&devPtrCounts, 4 * sizeof(int)));

    // ========================================================================
    // STEP 5: Configure grid for kernel execution
    // ========================================================================
    // Determine GPU grid dimensions for reduceStopCauses kernel
    // - Each thread processes one particle (nParticles threads total)
    // - Each block has NUMBER_OF_THREADS_PER_BLOCK threads
    // - Number of blocks = ceil(nParticles / NUMBER_OF_THREADS_PER_BLOCK)
    // Grid configuration ensures all particles are processed concurrently
    unsigned threadsPerBlock = NUMBER_OF_THREADS_PER_BLOCK;
    unsigned blocksForStates = (nParticles + threadsPerBlock - 1) / threadsPerBlock;

    // ========================================================================
    // STEP 6: Allocate host-side buffer for stop reason counts
    // ========================================================================
    // Host array to store results copied from GPU
    // Array indices correspond to: [reason0, reason1, reason2, reason3]
    // Contains total count of particles with each stop reason
    int counts[4];

    // ========================================================================
    // STEP 7: Main measurement loop
    // ========================================================================
    // Run ITERATION Monte Carlo steps while collecting stop reason statistics
    // - Each step performs particle movement and bond dynamics
    // - Stop reasons are tracked during particle updates
    // - Statistics accumulated and measured every step
    // - Outputs time and reason counts to stdout
    //
    // ITERATION parameter defined in config.h (default: 100000 steps)
    // Output frequency: every MC step (no sampling interval)
    while (pN -> iter < ITERATION) {
        
        // ====================================================================
        // Initialize GPU accumulators to zero
        // ====================================================================
        // Reset all four stop reason counters before kernel execution
        // Ensures clean slate for counting each measurement interval
        HANDLE_ERROR(cudaMemset(devPtrCounts, 0, memoryCounts));

        // ====================================================================
        // Launch kernel to count stop reasons
        // ====================================================================
        // reduceStopCauses computes distribution of particle stop reasons
        // Parameters:
        //   pN->devPtrStopReason: stop reason array [0,1,2,3] for each particle
        //   devPtrCounts: output accumulator array (4 stop reason counters)
        //   nParticles: number of particles to process
        //
        // Kernel logic:
        //   - Each thread i reads stop reason of particle i
        //   - Increments corresponding counter: devPtrCounts[reason[i]]
        //   - atomicAdd ensures thread-safe concurrent increments
        //   - Result: counts[j] = number of particles with stop reason j
        reduceStopCauses<<<blocksForStates, threadsPerBlock>>>(
            pN -> devPtrStopReason,   // Stop reason array
            devPtrCounts,             // Output: 4 reason counters
            nParticles                // Number of particles
        );

        // ====================================================================
        // Copy results from GPU to CPU
        // ====================================================================
        // Transfer stop reason counts from device to host memory
        // Blocking operation: waits for kernel to complete
        HANDLE_ERROR(cudaMemcpy(counts, devPtrCounts, memoryCounts, cudaMemcpyDeviceToHost));

        // ====================================================================
        // Output measurement to stdout
        // ====================================================================
        // Format: time <tab> count0 <tab> count1 <tab> count2 <tab> count3
        // Time = pN->iter (current MC step number)
        // Counts = number of particles with each stop reason
        //
        // Physical interpretation varies by model:
        //   - Different reasons may indicate: bond blocked, persistence change,
        //     random move attempt, or other dynamical constraints
        //
        // Example output:
        //   0  1523  456  234  120
        //   1  1521  458  235  119
        //   2  1525  452  236  121
        //   ...
        // Can be piped to file: ./a.out > output.dat
        printf("%d\t%d\t%d\t%d\t%d\n", pN -> iter, counts[0], counts[1], counts[2], counts[3]);

        // ====================================================================
        // Execute one Monte Carlo step
        // ====================================================================
        // Performs:
        // 1. updateParticles: Particle movement and direction changes
        //    - Particles may change direction (probability: 1 - P_PERST)
        //    - Particles attempt movement if bond exists
        //    - Stop reason recorded for each particle attempt
        //    - Uses atomic operations for collision-free concurrent updates
        // 2. updateBonds: Bond regeneration
        //    - Broken bonds regenerate with probability P_REGEN
        //    - Updates bidirectional bond pairs for consistency
        //
        // After mcStep completes:
        //   - pN->iter is incremented by 1
        //   - pN->devPtrStopReason is updated with new reasons
        //   - GPU state is updated (particle positions, bond states)
        mcStep(pN);
    }

    // ========================================================================
    // STEP 8: Clean up and exit
    // ========================================================================
    // Free all allocated GPU memory
    // devPtrCounts: temporary accumulator buffer for stop reason counts
    HANDLE_ERROR(cudaFree(devPtrCounts));

    // Free entire network structure (GPU and CPU memory)
    // Includes: site, index, direction, bond arrays, stop reason array,
    //           RNG states, and all associated data structures
    destroyNetwork(pN);

    return 0;
}