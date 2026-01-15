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
    int nBlocks = (int)(nParticles + NUMBER_OF_THREADS_PER_BLOCK - 1) / NUMBER_OF_THREADS_PER_BLOCK;

    // Calculate memory sizes (in bytes) for all arrays
    // These will be used for both allocation and memcpy operations
    pN -> memorySite                = N * sizeof(int);
    pN -> memoryIndex               = nParticles * sizeof(int);
    pN -> memoryDirection           = nParticles * sizeof(int);
    pN -> memoryNeighbor            = z * N * sizeof(int);
    pN -> memoryBond                = z * N * sizeof(int);
    pN -> memoryPartX               = dim * nParticles * sizeof(double);
    pN -> memoryShift               = z * sizeof(double);
    pN -> memoryCurandStatesBond    = z * N * sizeof(curandState);
    pN -> memoryCurandStatesSite    = N * sizeof(curandState);
    pN -> memoryMSD                 = nBlocks * sizeof(double);

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

    pN -> x  = (double *)malloc(pN -> memoryPartX);

    // Verify successful allocation before proceeding
    assert(pN -> x != NULL);

    pN -> shiftDir0 = (double *)malloc(pN -> memoryShift);
    pN -> shiftDir1 = (double *)malloc(pN -> memoryShift);

    // Verify successful allocation before proceeding
    assert(pN -> shiftDir0 != NULL && pN -> shiftDir1 != NULL);

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
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrX),           pN -> memoryPartX));
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrX0),          pN -> memoryPartX));
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrPartialMSD),  pN -> memoryMSD));
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrPartialMSD2), pN -> memoryMSD));
    
    // Allocate device memory for random number generation
    // Separate RNG states for bonds and sites to avoid correlation
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrCurandStatesBond), pN -> memoryCurandStatesBond));
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrCurandStatesSite), pN -> memoryCurandStatesSite));
    
    // Allocate device memory for elastic/mechanical vectors
    // Used for continuum elasticity calculations or other analysis
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrX), pN -> memoryPartX));

    // Allocate device memory for projection matrices
    // These are lattice-specific and used for elastic tensor calculations
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrShiftDir0), pN -> memoryShift));
    HANDLE_ERROR(cudaMalloc((void **)&(pN -> devPtrShiftDir1), pN -> memoryShift));

    ////////////////////////////////////////////////////////////////////////
    //                  LATTICE INITIALIZATION                            //
    ////////////////////////////////////////////////////////////////////////
    // Build static lattice structures that don't change during simulation

    // Generate shift on host based on lattice type
    // These are small (z doubles) and faster to compute on CPU
    getShift(pN);

    // Copy shift to device for GPU-based elastic calculations
    HANDLE_ERROR(cudaMemcpy(pN -> devPtrShiftDir0, pN -> shiftDir1, pN -> memoryShift, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pN -> devPtrShiftDir1, pN -> shiftDir0, pN -> memoryShift, cudaMemcpyHostToDevice));

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

    // Compute physical coordinates for each lattice site
    // Uses DIST constant and lattice type to generate positions
    getParticlesCoordinate(pN);

    HANDLE_ERROR(cudaMemcpy(pN -> devPtrX,  pN -> x,  pN -> memoryPartX, cudaMemcpyHostToDevice));

    // Initialize all bonds as active/connected (fully connected network)
    // Bonds may break or regenerate during simulation
    setBonds(pN, 1);

    return pN;
}


__host__ void getShift(network *pN) {

    LatticeType type = pN -> type;
    size_t memoryShift = pN -> memoryShift;
    
    // ========================================================================
    // Copy projectors based on lattice type
    // ========================================================================
    switch(type){
        case SQUARE_MOORE:
            // 8-neighbor Moore lattice projectors
            memcpy(pN -> shiftDir0, shift_0_sm, memoryShift);
            memcpy(pN -> shiftDir1, shift_1_sm, memoryShift);
            break;
        
        case TRIANGULAR:
            // 6-neighbor standard triangular lattice Shifts
            memcpy(pN -> shiftDir0, shift_0_tr, memoryShift);
            memcpy(pN -> shiftDir1, shift_1_tr, memoryShift);
            break;
    }
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



__host__ void getParticlesCoordinate(network *pN) {

    LatticeType type = pN -> type;
    int dim = pN -> dim;
    int nParticles = pN -> nParticles;

    // Displacement on eje Y for the points in each row
    double y_offset = DIST * sqrt(3) / 2;   // Height of an equilateral triangle

    switch(type) {

        case SQUARE_MOORE:

            for (int idx = 0; idx < nParticles; idx++) {
                int i = pN -> index[idx] % LX;
                int j = pN -> index[idx] / LX;

                pN -> x[dim * idx + 0] = DIST * i;
                pN -> x[dim * idx + 1] = DIST * j;
            }
        break;

        case TRIANGULAR:

            for (int idx = 0; idx < nParticles; idx++) {
                int i = pN -> index[idx] % LX;
                int j = pN -> index[idx] / LX;

                pN -> x[dim * idx + 0] = DIST * (i + 0.5 * j);
                pN -> x[dim * idx + 1] = j * y_offset;
            }
        break;
    }
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
        free(pN -> x);
        free(pN -> x0);

        // Free host memory for shift matrices
        free(pN -> shiftDir0);
        free(pN -> shiftDir1);
    

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
        
        // Free device memory for elastic vectors
        HANDLE_ERROR(cudaFree(pN -> devPtrX));
        HANDLE_ERROR(cudaFree(pN -> devPtrX0));
        
        // Free device memory for projection matrices
        HANDLE_ERROR(cudaFree(pN -> devPtrShiftDir0));
        HANDLE_ERROR(cudaFree(pN -> devPtrShiftDir1));

        HANDLE_ERROR(cudaFree(pN -> devPtrPartialMSD));
        HANDLE_ERROR(cudaFree(pN -> devPtrPartialMSD2));

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
        pN -> devPtrX,
        pN -> devPtrShiftDir0,
        pN -> devPtrShiftDir1,
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


__host__ void updateMSD(network *pN) {

    int nParticles = pN -> nParticles;
    int dim = pN -> dim;

    // Número de bloques para lanzar el kernel
    unsigned threadsPerBlock = NUMBER_OF_THREADS_PER_BLOCK;
    unsigned nBlocks = (nParticles + threadsPerBlock - 1) / threadsPerBlock;

    computeMSDAndAlpha2<<<nBlocks, threadsPerBlock>>>(
        pN -> devPtrX, 
        pN -> devPtrX0,
        pN -> devPtrPartialMSD,
        pN -> devPtrPartialMSD2,
        nParticles, dim
    );

    // Check for kernel launch errors
    HANDLE_ERROR(cudaGetLastError());

    // Synchronize to ensure kernel completion
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Reservar memoria para resultados finales
    double *devPtrMSDResult, *devPtrC4Result, *devPtrAlpha2Result;
    HANDLE_ERROR(cudaMalloc(&devPtrMSDResult,    sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&devPtrC4Result,     sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&devPtrAlpha2Result, sizeof(double)));

    // Ejecutar kernel finalizador que suma parciales y calcula alpha2
    finalizeMSDAndAlpha2<<<1,1>>>(
        pN -> devPtrPartialMSD,
        pN -> devPtrPartialMSD2,
        nBlocks,
        nParticles,
        devPtrMSDResult,
        devPtrC4Result,
        devPtrAlpha2Result
    );
    
    // Check for kernel launch errors
    HANDLE_ERROR(cudaGetLastError());

    // Synchronize to ensure kernel completion
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Copy msd from device to host
    HANDLE_ERROR(cudaMemcpy(&pN -> msd,    devPtrMSDResult,    sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&pN -> c4,     devPtrC4Result,     sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&pN -> alpha2, devPtrAlpha2Result, sizeof(double), cudaMemcpyDeviceToHost));


    // Liberar memoria device
    HANDLE_ERROR(cudaFree(devPtrMSDResult));
    HANDLE_ERROR(cudaFree(devPtrC4Result));
    HANDLE_ERROR(cudaFree(devPtrAlpha2Result));
}



__host__ void storeCoordinates(network *pN) {

    // Dynamically compute number of blocks for particle "updateParticles" kernel
    // and bonds "updateLinks" kernel
    unsigned threadsPerBlock = NUMBER_OF_THREADS_PER_BLOCK;
    unsigned blocksForParticles = (pN -> nParticles + threadsPerBlock - 1) / threadsPerBlock;

    getInitialCoordinates<<<blocksForParticles, threadsPerBlock>>>(
        pN -> devPtrX,
        pN -> devPtrX0,
        pN -> nParticles, 
        pN -> dim
    );

    // Check for kernel launch errors
    HANDLE_ERROR(cudaGetLastError());

    // Synchronize to ensure kernel completion
    HANDLE_ERROR(cudaDeviceSynchronize());
}