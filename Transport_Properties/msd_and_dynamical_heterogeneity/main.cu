/**
 * ============================================================================
 * Main Program: Active Brownian Particles on Adjustable Networks
 * ============================================================================
 *
 * This is the main entry point for ABP lattice simulations.
 * 
 * Workflow:
 *   1. Create and initialize network structure
 *   2. Initialize random number generators
 *   3. Equilibrate system (THERMAL steps)
 *   4. Collect measurements (ITERATION steps)
 *   5. Output statistics (MSD, C4, alpha2)
 *   6. Clean up and exit
 * 
 * Author: William G. C. Oropesa
 * Institution: ICTP South American Institute for Fundamental Research
 * GitHub Repository: ....
 * Date: ....
 * ============================================================================
 */

#include "config.h"
#include "network.h"

/**
 * Main function: Execute ABP simulation and collect statistics
 * 
 * Program execution sequence:
 *   1. Allocate and initialize network
 *   2. Setup random number generation
 *   3. Run equilibration phase
 *   4. Reset measurements
 *   5. Run measurement phase with data collection
 *   6. Output results
 *   7. Free all memory
 * 
 * Returns:
 *   EXIT_SUCCESS (0) on successful completion
 *   EXIT_FAILURE (1) if network creation fails
 */
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
    // STEP 3: Equilibration phase
    // ========================================================================
    // Run THERMAL Monte Carlo steps to reach steady state
    // - Removes memory of initial configuration
    // - Allows network to reorganize
    // - Allows particles to redistribute
    // - Establishes equilibrium connectivity
    //
    // THERMAL parameter defined in config.h (default: 100000 steps)
    // During equilibration, no data is collected
    while (pN -> iter < THERMAL) {
        // Execute one MC step:
        // - updateParticles: Move and reorient particles
        // - updateBonds: Regenerate broken bonds stochastically
        mcStep(pN);
    }

    // ========================================================================
    // STEP 4: Reset iteration counter for measurement phase
    // ========================================================================
    // Set counter to 0 to measure time from t=0 after equilibration
    pN -> iter = 0;

    // ========================================================================
    // Store current particle coordinates as initial reference
    // ========================================================================
    // Copy current positions (x array) to initial position array (x0)
    // This sets the displacement reference point for MSD calculations
    // All future MSD values will be measured relative to this point
    storeCoordinates(pN);

    // ========================================================================
    // STEP 5: Measurement phase
    // ========================================================================
    // Run ITERATION Monte Carlo steps while collecting statistics
    // - Each step performs particle movement and bond dynamics
    // - MSD (mean squared displacement) computed at each step
    // - C4 (fourth cumulant) computed at each step
    // - alpha2 (non-Gaussian parameter) computed at each step
    //
    // ITERATION parameter defined in config.h (default: 100000 steps)
    while (pN -> iter < ITERATION) {
        
        // ====================================================================
        // Compute mean squared displacement and non-Gaussianity metrics
        // ====================================================================
        // updateMSD computes:
        //   pN -> msd    : Mean squared displacement <(r(t) - r(0))^2>
        //   pN -> c4     : Fourth cumulant (dynamic heterogeneity)
        //   pN -> alpha2 : Non-Gaussian parameter (indication of non-Gaussian dynamics)
        //
        // These are computed on GPU using parallel reduction, then
        // copied to host for output
        updateMSD(pN);
        
        // ====================================================================
        // Output current statistics
        // ====================================================================
        // Format: iteration <tab> msd <tab> c4 <tab> alpha2
        // 
        // Fields:
        //   pN -> iter   : Current iteration number (0 to ITERATION)
        //   pN -> msd    : Mean squared displacement
        //   pN -> c4     : Fourth cumulant (C4 = <(Δr^2)^2> - <Δr^2>^2)
        //   pN -> alpha2 : Non-Gaussian parameter (alpha2 = (d/(d+2)) * (<Δr^4> / <Δr^2>^2) - 1)
        printf("%d\t%lf\t%lf\t%lf\n", pN -> iter, pN -> msd, pN -> c4, pN -> alpha2);
        
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
        //   - pN -> iter is incremented by 1
        //   - GPU state is updated (particle positions, bond states)
        mcStep(pN);
    }

    // ========================================================================
    // STEP 6: Clean up and exit
    // ========================================================================
    // Free all allocated GPU and CPU memory
    // Must be called to prevent memory leaks on both devices
    destroyNetwork(pN);

    return EXIT_SUCCESS;
}