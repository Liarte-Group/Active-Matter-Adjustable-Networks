/**
 * ============================================================================
 * Main Program: Active Brownian Particles - Coordination Number Measurement
 * ============================================================================
 *
 * This program simulates ABP dynamics on an adjustable network and measures
 * the mean coordination number over time.
 *
 * Workflow:
 *   1. Create and initialize network structure
 *   2. Setup random number generators
 *   3. Equilibrate system (THERMAL steps)
 *   4. Measure coordination number periodically (ITERATION steps)
 *   5. Output mean coordination number vs time
 *   6. Clean up and exit
 *
 * Output format:
 *   Column 1: Time (in units of PASS_TIME steps)
 *   Column 2: Mean coordination number <z>
 *
 * Author: William G. C. Oropesa
 * Institution: ICTP South American Institute for Fundamental Research
 * GitHub Repository: ...
 * Date: ...
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
    // - Establishes equilibrium connectivity
    //
    // THERMAL parameter defined in config.h (default: 100000 steps)
    // During equilibration, no measurements are collected
    while (pN -> iter < THERMAL) {
        // Execute one MC step:
        // - updateParticles: Move and reorient particles
        // - updateBonds: Regenerate broken bonds stochastically
        mcStep(pN);
    }

    // ========================================================================
    // Reset iteration counter for measurement phase
    // ========================================================================
    // Set counter to 0 to count measurement steps from t=0
    pN -> iter = 0;

    // ========================================================================
    // STEP 4: Allocate GPU memory for coordination number accumulator
    // ========================================================================
    // Temporary GPU buffer to store total active bonds
    // Size: 1 integer (accumulator for sum of all active bonds)
    // This will hold the output from getZDistb kernel
    int *devPtrCoordination;
    HANDLE_ERROR(cudaMalloc((void **)&devPtrCoordination, sizeof(int)));

    // ========================================================================
    // STEP 5: Configure grid for kernel execution
    // ========================================================================
    // Determine GPU grid dimensions for getZDistb kernel
    // - Each thread processes one site (N threads total)
    // - Each block has NUMBER_OF_THREADS_PER_BLOCK threads
    // - Number of blocks = ceil(N / NUMBER_OF_THREADS_PER_BLOCK)
    unsigned threadsPerBlock = NUMBER_OF_THREADS_PER_BLOCK;
    unsigned blocksForStates = (N + threadsPerBlock - 1) / threadsPerBlock;

    // ========================================================================
    // STEP 6: Main measurement loop
    // ========================================================================
    // Run ITERATION Monte Carlo steps while collecting statistics
    // - Each step performs particle movement and bond dynamics
    // - Coordination number measured every PASS_TIME steps
    // - Outputs time vs <z> to stdout
    //
    // ITERATION parameter defined in config.h (default: 100000 steps)
    // PASS_TIME parameter defined in config.h (default: 100 steps)
    while (pN -> iter < ITERATION) {
        
        // ====================================================================
        // Execute one Monte Carlo step
        // ====================================================================
        // Performs:
        // 1. updateParticles: Particle movement and direction changes
        //    - Particles may change direction (probability: 1 - P_PERST)
        //    - Particles attempt movement if bond exists
        //    - Uses atomic operations for collision-free concurrent updates
        // 2. updateBonds: Bond regeneration
        //    - Broken bonds regenerate with probability P_REGEN
        //    - Updates bidirectional bond pairs for consistency
        //
        // After mcStep completes:
        //   - pN->iter is incremented by 1
        //   - GPU state is updated (particle positions, bond states)
        mcStep(pN);

        // ====================================================================
        // Measure coordination number at specified intervals
        // ====================================================================
        // Only compute every PASS_TIME steps to reduce data overhead
        // PASS_TIME = sampling interval (reduces correlation between measurements)
        // Example: PASS_TIME=100 means measure every 100 MC steps
        if (pN -> iter % PASS_TIME == 0) {
            
            // ================================================================
            // Initialize GPU accumulator to zero
            // ================================================================
            // Reset coordination counter before kernel execution
            // Ensures clean slate for counting active bonds
            HANDLE_ERROR(cudaMemset(devPtrCoordination, 0, sizeof(int)));

            // ================================================================
            // Launch kernel to count total active bonds
            // ================================================================
            // getZDistb computes sum of all active bonds across network
            // Parameters:
            //   devPtrBond: bond state array [0,1] for each bond
            //   pN->z: coordination number (6 or 8)
            //   devPtrCoordination: output accumulator (total active bonds)
            //
            // Kernel logic:
            //   - Each thread i counts bonds at site i
            //   - sum_dir(bond[z*i + dir]) adds up bonds
            //   - atomicAdd increments devPtrCoordination
            //   - Result: total number of active bonds in entire network
            getZDistb<<<blocksForStates, threadsPerBlock>>>(
                pN -> devPtrBond,         // Bond state array
                pN -> z,                  // Coordination number
                devPtrCoordination        // Output: total active bonds
            );

            // ================================================================
            // Copy result from GPU to CPU
            // ================================================================
            // Transfer coordination number from device to host
            int coordination = 0;
            HANDLE_ERROR(cudaMemcpy(&coordination, devPtrCoordination, sizeof(int), cudaMemcpyDeviceToHost));

            // ================================================================
            // Compute mean coordination number
            // ================================================================
            // Coordination = total active bonds = sum_i (z_i)
            // Mean coordination = (total bonds) / (number of sites)
            // zMean = <z> = coordination / N
            //
            // Physical interpretation:
            //   - zMean = 6: All bonds active (fully connected triangular)
            //   - zMean = 4: Half the bonds broken on average
            //   - zMean = 0: All bonds broken (isolated system)
            double zMean = (double)coordination / (double)N;

            // ================================================================
            // Output measurement to stdout
            // ================================================================
            // Format: time <tab> mean_coordination_number
            // Time = pN->iter / 100 converts MC steps to measurement index
            //   (dividing by 100 normalizes if PASS_TIME=100)
            // Example output:
            //   0  6.000000
            //   1  5.923456
            //   2  5.847123
            //   ...
            // Can be piped to file: ./a.out > output.dat
            printf("%d\t%.6f\n", pN -> iter / PASS_TIME, zMean);
        }
    }

    // ========================================================================
    // STEP 7: Clean up and exit
    // ========================================================================
    // Free all allocated GPU memory
    // devPtrCoordination: temporary accumulator buffer
    HANDLE_ERROR(cudaFree(devPtrCoordination));

    // Free entire network structure (GPU and CPU memory)
    // Includes: site, index, direction, bond arrays, RNG states, etc.
    destroyNetwork(pN);

    return 0;
}