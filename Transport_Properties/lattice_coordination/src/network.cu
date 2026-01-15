/**
 * ============================================================================
 * Host Functions for Active Brownian Particles (ABP) Lattice Simulation
 * ============================================================================
 * 
 * This file contains all CPU-side (host) functions for managing the network
 * structure, including initialization, memory management, and parameter setup.
 * 
 * Key responsibilities:
 *   - Memory allocation/deallocation (host and device)
 *   - Lattice initialization and parameter configuration
 *   - Kernel launch wrapper functions with error checking
 *   - Random number generator setup
 *   - Particle placement on lattice
 * 
 * Design patterns:
 *   - Error checking with HANDLE_ERROR macro for all CUDA calls
 *   - Kernel wrappers abstract GPU complexity from user code
 *   - Separation of concerns: host logic separate from kernel logic
 *   - Proper resource cleanup in destroyNetwork function
 * 
 * Author: William G. C. Oropesa
 * Institution: ICTP South American Institute for Fundamental Research
 * GitHub Repository: https://github.com/williamGOC/
 * Date: November 2025
 * ============================================================================
 */

#include "config.h"
#include "network.h"


__host__ network *makeNetwork(LatticeType type, const int dim, const double pack, const double pRegen, const double pPerst) {

    // ========================================================================
    // STEP 1: Allocate and initialize network structure on heap
    // ========================================================================
    network *pN = (network *)malloc(sizeof(network));
    assert(pN != NULL);

    // ========================================================================
    // STEP 2: Determine coordination number based on lattice type
    // ========================================================================
    // Different lattice geometries have different numbers of neighbors per site
    int z;
    switch (type) {
        case SQUARE_MOORE:     z = 8;  break;  // 8 cardinal + diagonal
        case TRIANGULAR:       z = 6;  break;  // 6 triangular neighbors
        default:
            fprintf(stderr, "ERROR: Unknown lattice type %d!\n", type);
            exit(EXIT_FAILURE);
    }

    // ========================================================================
    // STEP 3: Initialize simulation parameters in network structure
    // ========================================================================
    pN -> z       = z;                    // Number of neighbors per site
    pN -> dim     = dim;                  // Dimension for position vectors
    pN -> iter    = 0;                    // Iteration counter (MC steps)
    pN -> seed    = time(NULL);           // Random seed from system time
    pN -> pack    = pack;                 // Packing fraction (particles/sites)
    pN -> pRegen  = pRegen;               // Bond regeneration probability
    pN -> pPerst  = pPerst;               // Particle persistence probability
    pN -> type    = type;                 // Lattice type

    // ========================================================================
    // STEP 4: Calculate number of particles and memory requirements
    // ========================================================================
    // Use lround for banker's rounding (round half to even)
    // This provides consistent results across multiple runs
    int nParticles = lround(pack * N); pN -> nParticles = nParticles;

    // Calculate memory sizes (in bytes) for all arrays
    // These will be used for both allocation and memcpy operations
    pN -> memorySite                = N * sizeof(int);
    pN -> memoryIndex               = nParticles * sizeof(int);
    pN -> memoryDirection           = nParticles * sizeof(int);
    pN -> memoryNeighbor            = z * N * sizeof(int);
    pN -> memoryBond                = z * N * sizeof(int);
    pN -> memoryCurandStatesBond    = z * N * sizeof(curandState);
    pN -> memoryCurandStatesSite    = N * sizeof(curandState);

    ////////////////////////////////////////////////////////////////////////
    //                  HOST MEMORY ALLOCATION                            //
    ////////////////////////////////////////////////////////////////////////
    // CPU-side buffers for frequently accessed data
    // These are used for initialization and can be used for analysis

    // Allocate host memory for particle state vectors and bond state
    pN -> site      = (int *)malloc(pN -> memorySite);
    pN -> index     = (int *)malloc(pN -> memoryIndex);
    pN -> direction = (int *)malloc(pN -> memoryDirection);
    pN -> bond      = (int *)malloc(pN -> memoryBond);

    // Verify successful allocation before proceeding
    assert(pN -> site != NULL && pN -> index != NULL && pN -> direction != NULL && pN -> bond != NULL);


    ////////////////////////////////////////////////////////////////////////
    //                DEVICE MEMORY ALLOCATION                            //
    ////////////////////////////////////////////////////////////////////////
    // GPU-side buffers for simulation and computation
    // Most simulation work happens with these device pointers

    // Allocate device memory for lattice structure and dynamics
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrSite),        pN -> memorySite));
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrIndex),       pN -> memoryIndex));
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrDirection),   pN -> memoryDirection));
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrNeighbor),    pN -> memoryNeighbor));
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrBond),        pN -> memoryBond));
    
    // Allocate device memory for random number generation
    // Separate RNG states for bonds and sites to avoid correlation
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrCurandStatesBond), pN -> memoryCurandStatesBond));
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrCurandStatesSite), pN -> memoryCurandStatesSite));
    

    ////////////////////////////////////////////////////////////////////////
    //                  LATTICE INITIALIZATION                            //
    ////////////////////////////////////////////////////////////////////////
    // Build static lattice structures that don't change during simulation

    // Generate neighbor list on GPU
    // Pre-computes which sites are neighbors for each site
    getNeighborList(pN);

    // Place particles on lattice using Fisher-Yates shuffle
    // Done on CPU for simplicity and random number availability
    putABPOnNetwork(pN);

    // Copy particle initial configuration to device
    HANDLE_ERROR(cudaMemcpy(pN -> devPtrSite,      pN -> site,      pN -> memorySite,      cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pN -> devPtrIndex,     pN -> index,     pN -> memoryIndex,     cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pN -> devPtrDirection, pN -> direction, pN -> memoryDirection, cudaMemcpyHostToDevice));

    // Initialize all bonds as active/connected (fully connected network)
    // Bonds may break or regenerate during simulation
    setBonds(pN, 1);

    return pN;
}


__host__ void getNeighborList(network *pN) {

    int z = pN -> z;
    int threads = NUMBER_OF_THREADS_PER_BLOCK;
    int blocks  = (z * N + threads - 1) / threads;

    // Launch device kernel
    // Grid configuration: threads cover all N * z neighbor entries
    kerGetNeighborList<<<blocks, threads>>>(pN -> devPtrNeighbor, z, pN -> type);

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
}


__host__ void setBonds(network *pN, const int value) {

    int z = pN -> z;
    int threads = NUMBER_OF_THREADS_PER_BLOCK;
    int blocks  = (z * N + threads - 1) / threads;

    // Launch device kernel
    // Grid configuration: threads cover all N*z bond entries
    kerSetBonds<<<blocks, threads>>>(pN -> devPtrBond, z, value);

    HANDLE_ERROR(cudaGetLastError());
    // Note: No cudaDeviceSynchronize() here - kernel launches asynchronously
    // Add if synchronization is needed before next GPU operation
}


__host__ void putABPOnNetwork(network *pN) {

    // ========================================================================
    // STEP 1: Clear site array (initialize all sites as empty)
    // ========================================================================
    memset(pN -> site, 0, pN -> memorySite);

    // ========================================================================
    // STEP 2: Create list of all possible site indices [0, N-1]
    // ========================================================================
    // This will be shuffled to select random positions for particles
    int *indices = (int *)malloc(N * sizeof(int));
    assert(indices);

    // Initialize indices to identity permutation
    for (int i = 0; i < N; i++) {
        indices[i] = i;
    }

    // ========================================================================
    // STEP 3: Fisher-Yates partial shuffle to select nParticles unique sites
    // ========================================================================
    // Only shuffle first nParticles positions for efficiency
    for (int i = 0; i < pN -> nParticles; i++) {
        // Pick a random index between i and N-1 (inclusive)
        // This ensures we select from unshuffled portion
        int j = i + rand() % (N - i);

        // Swap indices[i] and indices[j]
        // This brings an unvisited index into position i
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;

        // ====================================================================
        // STEP 4: Assign particle i to site indices[i]
        // ====================================================================
        
        // Use the chosen index as particle position
        int idx = indices[i];

        // Mark the site as occupied
        pN -> site[idx] = 1;

        // Store particle position in index array
        // index[i] = which lattice site particle i occupies
        pN -> index[i] = idx;

        // Assign random direction to particle
        // direction[i] = which neighbor direction particle i faces
        // Range: [0, z-1] where z is coordination number
        pN -> direction[i] = rand() % pN -> z;
    }

    // ========================================================================
    // STEP 5: Free temporary array
    // ========================================================================
    // indices was only needed for shuffling
    free(indices);
}


__host__ void initCurand(network *pN) {
    
    int z = pN -> z;

    // ========================================================================
    // STEP 1: Calculate total number of RNG states needed
    // ========================================================================
    // Bond RNG: One per bond directional pair (N sites × z directions)
    int totalBondStates = N * z;

    // Particle/Site RNG: One per site (particles placed on sites)
    // Note: All N sites get RNG states for simplicity
    int totalSiteStates = N;

    // ========================================================================
    // STEP 2: Calculate grid configuration for kernel launches
    // ========================================================================
    unsigned threads = NUMBER_OF_THREADS_PER_BLOCK;
    unsigned blocksBond = (totalBondStates + threads - 1) / threads;
    unsigned blocksSite = (totalSiteStates + threads - 1) / threads;

    // ========================================================================
    // STEP 3: Initialize RNG states for bonds
    // ========================================================================
    // Each thread initializes one bond RNG state
    // Sequence number = thread index for independence
    setupCurandState<<<blocksBond, threads>>>(pN -> devPtrCurandStatesBond, totalBondStates, pN -> seed);

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    // ========================================================================
    // STEP 4: Initialize RNG states for sites/particles
    // ========================================================================
    // Each thread initializes one site/particle RNG state
    // Different seed to avoid correlation with bond RNG
    // Offset of 1234 ensures independent random sequences
    setupCurandState<<<blocksSite, threads>>>(pN -> devPtrCurandStatesSite, totalSiteStates, pN -> seed + 1234UL);

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
}



__host__ void destroyNetwork(network *pN) {
    if (pN != NULL) {

        ////////////////////////////////////////////////////////////////////
        //                   FREE HOST MEMORY                             //
        ////////////////////////////////////////////////////////////////////
        
        // Free host memory for particle state vectors and bond state
        free(pN -> site);
        free(pN -> index);
        free(pN -> direction);
        free(pN -> bond);

        ////////////////////////////////////////////////////////////////////
        //                  FREE DEVICE MEMORY                            //
        ////////////////////////////////////////////////////////////////////
        
        // Free device memory for lattice structure
        HANDLE_ERROR(cudaFree(pN -> devPtrSite));
        HANDLE_ERROR(cudaFree(pN -> devPtrIndex));
        HANDLE_ERROR(cudaFree(pN -> devPtrDirection));
        HANDLE_ERROR(cudaFree(pN -> devPtrNeighbor));
        HANDLE_ERROR(cudaFree(pN -> devPtrBond));
        
        // Free device memory for random number generation
        HANDLE_ERROR(cudaFree(pN -> devPtrCurandStatesBond));
        HANDLE_ERROR(cudaFree(pN -> devPtrCurandStatesSite));
    

        ////////////////////////////////////////////////////////////////////
        //                 FREE NETWORK STRUCTURE                         //
        ////////////////////////////////////////////////////////////////////
        
        // Free the network structure itself
        free(pN);
    }
}


__host__ void mcStep(network *pN) {

    int nParticles = pN -> nParticles;
    int z = pN -> z;
    int dim = pN -> dim;

    // ========================================================================
    // STEP 1: Update particle positions and directions
    // ========================================================================
    // Particles attempt movement, change direction based on persistence
    
    int threadsParticles = NUMBER_OF_THREADS_PER_BLOCK;
    int blocksParticles = (nParticles + threadsParticles - 1) / threadsParticles;

    // Launch particle update kernel
    // One thread per particle
    updateParticles<<<blocksParticles, threadsParticles>>>(
        pN -> devPtrSite,
        pN -> devPtrNeighbor,
        pN -> devPtrBond,
        pN -> devPtrIndex,
        pN -> devPtrDirection,
        nParticles,
        dim,
        z,
        pN -> devPtrCurandStatesSite
    );

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    // ========================================================================
    // STEP 2: Update bond states (regeneration)
    // ========================================================================
    // Broken bonds regenerate with probability P_REGEN
    
    int threadsBonds = NUMBER_OF_THREADS_PER_BLOCK;
    int blocksBonds = (N * z + threadsBonds - 1) / threadsBonds;

    // Launch bond update kernel
    // One thread per bond directional pair
    updateBonds<<<blocksBonds, threadsBonds>>>(
        pN -> devPtrBond,
        pN -> devPtrNeighbor,
        z,
        pN -> devPtrCurandStatesBond
    );

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    // ========================================================================
    // Increment iteration counter
    // ========================================================================
    pN -> iter++;
}



__host__ void mcSteps(network *pN, int nSteps) {

    for (int step = 0; step < nSteps; step++) {
        
        // Perform one MC step
        mcStep(pN);

        //TODO

    }
}



__host__ void syncAndCopyToCPU(network *pN) {

    // Ensure all GPU operations are complete
    HANDLE_ERROR(cudaDeviceSynchronize());

    // ========================================================================
    // Copy particle configuration from device to host
    // ========================================================================

    // Copy site occupancy (which sites are occupied)
    HANDLE_ERROR(cudaMemcpy(pN -> site, pN -> devPtrSite, pN -> memorySite, cudaMemcpyDeviceToHost));

    // Copy particle positions (which lattice site each particle is at)
    HANDLE_ERROR(cudaMemcpy(pN -> index, pN -> devPtrIndex, pN -> memoryIndex, cudaMemcpyDeviceToHost));

    // Copy particle directions (which direction each particle faces)
    HANDLE_ERROR(cudaMemcpy(pN -> direction, pN -> devPtrDirection, pN -> memoryDirection, cudaMemcpyDeviceToHost));

    // Copy bond on the lattice
    HANDLE_ERROR(cudaMemcpy(pN -> bond, pN -> devPtrBond, pN -> memoryBond, cudaMemcpyDeviceToHost));
}
