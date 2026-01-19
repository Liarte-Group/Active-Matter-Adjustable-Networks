#ifndef __NETWORK_H__
#define __NETWORK_H__
/**
 * ============================================================================
 * Header File: Network Data Structure and Lattice Management
 * ============================================================================
 * 
 * This header defines the core network data structure and associated functions
 * for managing lattices with active Brownian particles (ABPs). It provides:
 *   - Network topology representation (neighbors, bonds, sites)
 *   - Particle management (positions, directions, occupancy)
 *   - Random number generation (cuRAND integration)
 *   - Lattice geometry (coordinates, boundaries)
 * 
 * Key features:
 *   - Support for multiple lattice types (square Moore, triangular)
 *   - Dual host/device memory management
 *   - Integrated particle and bond dynamics
 *   - Thread-safe random number generation
 * 
 * Workflow:
 *   1. Create network: network *net = makeNetwork(type, dim, pack, pPerst, pRegen)
 *   2. Initialize geometry: getNeighborList(), getNetworkCoordinate(), etc.
 *   3. Place particles: putABPOnNetwork()
 *   4. Set RNG: initCurand()
 *   5. Simulate: mcSteps(net, nSteps) in a loop
 *   6. Clean up: destroyNetwork()
 * 
 * Author: William G. C. Oropesa
 * Institution: ICTP South American Institute for Fundamental Research
 * GitHub Repository: TODO
 * Date: TODO
 * ============================================================================
 */


#include "config.h"

/**
 * ============================================================================
 * Lattice Type Enumeration
 * ============================================================================
 * 
 * LatticeType specifies the geometry and connectivity of the lattice.
 * Different lattice types have different coordination numbers and geometric
 * arrangements, which affect particle dynamics and elastic properties.
 * 
 * Supported lattice types:
 * 
 *   SQUARE_MOORE (z=8):
 *     ──────────────────────────────────────────────────────────────────
 *     Coordination number: z = 8 (Moore neighborhood)
 *     Neighbors: Cardinal (4) + Diagonal (4) directions
 *     
 *     Neighbor arrangement (for site i at position (x,y)):
 *       
 *          NW   N   NE
 *            \  |  /
 *             \ | /
 *         w --- i --- E
 *             / | \
 *            /  |  \
 *          SW   S   SE
 *     
 *     Direction indices: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
 *     
 *     Physical properties:
 *       - Square lattice with maximum connectivity
 *       - Isotropy: uniform in all 8 directions
 *       - Common in numerical simulations (well-studied)
 *       - More particle tracks possible (higher coordination)
 *     
 *     Use cases:
 *       - Simple numerical model
 *       - Study high-density systems
 *       - Compare with theoretical predictions
 *   
 *   TRIANGULAR (z=6):
 *     ──────────────────────────────────────────────────────────────────
 *     Coordination number: z = 6 (regular triangular lattice)
 *     Neighbors: Hexagonal close-packing arrangement
 *     
 *     Neighbor arrangement (hexagonal pattern):
 *       
 *             NW     NE
 *              \     /
 *               \   /
 *           W --- i --- E
 *               /   \
 *              /     \
 *            SW      SE
 *     
 *     Direction indices: 0=E, 1=NE, 2=NW, 3=W, 4=SW, 5=SE
 *     
 *     Staggered rows: Even and odd rows offset for hexagonal packing
 *     Vertical spacing: DIST * sqrt(3) / 2 (optimal packing)
 *     Horizontal offset: Every other row shifted by DIST / 2
 *     
 *     Physical properties:
 *       - Highest packing density (hexagonal close packing)
 *       - Natural optimal packing for circles
 *       - Common in colloids and granular materials
 *       - Better reflects real material geometry
 *       - Lower coordination than Moore (but optimal connections)
 *     
 *     Use cases:
 *       - Realistic model for granular systems
 *       - Study optimal packing effects
 *       - Compare with experimental results
 *       - Colloidal particle simulations
 * 
 * Selection guidelines:
 *   Use SQUARE_MOORE for:
 *     - Theoretical studies (simpler geometry)
 *     - Comparison with literature (standard choice)
 *     - Initial parameter exploration
 *   
 *   Use TRIANGULAR for:
 *     - Realistic material simulations
 *     - High-density systems
 *     - Experimental comparison
 *     - Packing fraction studies
 * 
 * See also:
 *   - kerGetNeighborList: Device kernel building connectivity
 *   - kerGetNetworkCoordinate: Device kernel computing coordinates
 */
typedef enum {
    SQUARE_MOORE,              // 8 neighbors (Moore neighborhood, square lattice)
    TRIANGULAR                 // 6 neighbors (regular triangular lattice)
} LatticeType;


/**
 * ============================================================================
 * Network Data Structure
 * ============================================================================
 * 
 * The network structure is the central data container for simulation state.
 * It holds pointers to all GPU memory (device), corresponding CPU copies
 * (host), lattice properties, and simulation parameters.
 * 
 * Design pattern:
 *   - Device pointers (devPtr*): GPU memory (fast, must use kernels to access)
 *   - Host arrays (*): CPU memory (slow, direct CPU access possible)
 *   - Synchronization: Use syncAndCopyToCPU() to update host from device
 * 
 * Memory organization:
 * 
 *   ═══════════════════════════════════════════════════════════════════
 *   MEMORY SIZE TRACKING (in bytes)
 *   ═══════════════════════════════════════════════════════════════════
 *   
 *   size_t memorySite
 *     Size: N * sizeof(int) bytes
 *     Purpose: Track allocation size for site arrays
 *     Use: Memory management, validation
 *   
 *   size_t memoryBoundary
 *     Size: N * sizeof(int) bytes
 *     Purpose: Track allocation for boundary markers
 *     Use: Memory cleanup, checks
 *   
 *   size_t memoryIndex
 *     Size: nParticles * sizeof(int) bytes
 *     Purpose: Track allocation for particle indices
 *     Use: Memory management
 *   
 *   size_t memoryDirection
 *     Size: nParticles * sizeof(int) bytes
 *     Purpose: Track allocation for particle directions
 *     Use: Memory management
 *   
 *   size_t memoryNeighbor
 *     Size: N * z * sizeof(int) bytes
 *     Purpose: Track neighbor list allocation
 *     Use: Memory management, validation
 *   
 *   size_t memoryBond
 *     Size: N * z * sizeof(int) bytes
 *     Purpose: Track bond state allocation
 *     Use: Memory management
 *   
 *   size_t memoryCurandStatesBond
 *     Size: N * z * sizeof(curandState) bytes
 *     Purpose: Track RNG state allocation for bonds
 *     Use: Memory cleanup
 *   
 *   size_t memoryCurandStatesSite
 *     Size: N * sizeof(curandState) bytes
 *     Purpose: Track RNG state allocation for sites
 *     Use: Memory cleanup
 * 
 *   ═══════════════════════════════════════════════════════════════════
 *   HOST MEMORY ARRAYS (CPU side, slower but directly accessible)
 *   ═══════════════════════════════════════════════════════════════════
 *   
 *   int *site
 *     Size: N integers
 *     Purpose: Occupancy state (1=occupied by particle, 0=empty)
 *     Access: CPU direct read/write
 *     Update: Via syncAndCopyToCPU() from devPtrSite
 *     Use: Data analysis, debugging, particle enumeration
 *   
 *   int *index
 *     Size: nParticles integers
 *     Purpose: Maps particle ID → current lattice site index
 *     Access: CPU direct read/write
 *     Update: Via syncAndCopyToCPU() from devPtrIndex
 *     Use: Particle tracking, position queries
 *   
 *   int *direction
 *     Size: nParticles integers
 *     Purpose: Current direction [0, z-1] for each particle
 *     Access: CPU direct read/write
 *     Update: Via syncAndCopyToCPU() from devPtrDirection
 *     Use: Direction analysis, particle state inspection
 *   
 *   int *bond
 *     Size: N*z integers
 *     Purpose: Bond state array (1=active, 0=broken/removed)
 *     Access: CPU direct read/write
 *     Update: Via syncAndCopyToCPU() from devPtrBond
 *     Use: Bond statistics, network connectivity analysis
 *   
 * 
 *   ═══════════════════════════════════════════════════════════════════
 *   DEVICE MEMORY ARRAYS (GPU side, fast, kernel-accessible only)
 *   ═══════════════════════════════════════════════════════════════════
 *   
 *   PARTICLE AND OCCUPANCY ARRAYS:
 *   
 *   int *devPtrSite
 *     Size: N integers
 *     Purpose: Occupancy state on GPU (1=occupied, 0=empty)
 *     Access: GPU kernels (atomicCAS, atomicExch, etc.)
 *     Update: updateParticles kernel modifies
 *     Use: Fast particle movement, concurrent access
 *   
 *   int *devPtrIndex
 *     Size: nParticles integers
 *     Purpose: Particle ID → site index mapping on GPU
 *     Access: GPU kernels read/write
 *     Update: updateParticles kernel modifies
 *     Use: Particle position tracking during MC steps
 *   
 *   int *devPtrDirection
 *     Size: nParticles integers
 *     Purpose: Particle direction [0, z-1] on GPU
 *     Access: GPU kernels read/write
 *     Update: updateParticles kernel modifies
 *     Use: Direction updates, persistence physics
 *   
 *   NETWORK TOPOLOGY:
 *   
 *   int *devPtrBoundary
 *     Size: N integers (1=boundary site, 0=interior)
 *     Purpose: Marks which sites are at lattice edges
 *     Access: GPU kernels read (not modified after init)
 *     Update: getBoundary kernel sets, then constant
 *     Use: Apply boundary conditions, exclude surface sites
 *   
 *   int *devPtrNeighbor
 *     Size: N*z integers
 *     Purpose: Neighbor connectivity list
 *     Layout: neighbor[z*i + dir] = index of neighbor at direction dir
 *     Access: GPU kernels read (not modified after init)
 *     Update: getNeighborList kernel sets, then constant
 *     Use: Particle movement, bond interactions
 *   
 *   int *devPtrBond
 *     Size: N*z integers (1=active, 0=broken)
 *     Purpose: Bond state determines particle passage
 *     Access: GPU kernels read/write (atomic operations)
 *     Update: updateParticles removes bonds; updateBonds regenerates
 *     Use: Stochastic bond dynamics, network connectivity
 *   
 *   ═══════════════════════════════════════════════════════════════════
 *   LATTICE AND SIMULATION PARAMETERS
 *   ═══════════════════════════════════════════════════════════════════
 *   
 *   int z
 *     Purpose: Coordination number (neighbors per site)
 *     Values: 8 (SQUARE_MOORE), 6 (TRIANGULAR)
 *     Set by: makeNetwork() based on lattice type
 *     Use: Loop bounds in kernels, memory allocation sizes
 *   
 *   int dim
 *     Purpose: Spatial dimensionality
 *     Values: Typically 2 (2D systems)
 *     Set by: Configuration or makeNetwork() parameter
 *     Use: Vector sizing, coordinate computations
 *   
 *   int iter
 *     Purpose: Global iteration counter
 *     Values: Incremented by mcSteps()
 *     Use: Simulation progress tracking, data collection timing
 *   
 *   unsigned long seed
 *     Purpose: Random number generator seed for reproducibility
 *     Set by: makeNetwork() or externally before initCurand()
 *     Use: setupCurandState kernel initialization
 *   
 *   double pack
 *     Purpose: Packing fraction (fraction of sites occupied by particles)
 *     Range: 0.0 (empty) to 1.0 (full)
 *     Example: pack = 0.5 means 50% of sites have particles
 *     Use: Initialize particle density, system characterization
 *   
 *   double pPerst
 *     Purpose: Persistence probability (probability particle maintains direction)
 *     Range: 0.0 to 1.0
 *     Example: pPerst = 0.9 means 90% chance particle keeps direction
 *     Use: updateParticles kernel, controls particle rotational freedom
 *   
 *   double pRegen
 *     Purpose: Bond regeneration probability
 *     Range: 0.0 to 1.0
 *     Example: pRegen = 0.5 means 50% chance broken bond repairs
 *     Use: updateBonds kernel, controls bond dynamics
 *   
 *   int nParticles
 *     Purpose: Total number of active Brownian particles
 *     Values: nParticles = pack * N
 *     Computed by: putABPOnNetwork()
 *     Use: Particle loop bounds, array sizing
 *   
 *   LatticeType type
 *     Purpose: Lattice geometry type
 *     Values: SQUARE_MOORE (z=8), TRIANGULAR (z=6)
 *     Set by: makeNetwork() parameter
 *     Use: Kernel selection, geometry computation
 * 
 *   ═══════════════════════════════════════════════════════════════════
 *   RANDOM NUMBER GENERATION STATE
 *   ═══════════════════════════════════════════════════════════════════
 *   
 *   curandState *devPtrCurandStatesBond
 *     Size: N*z states (one per bond, one per direction)
 *     Purpose: Independent RNG state for each bond
 *     Use: Bond regeneration stochasticity (updateBonds kernel)
 *     Init: setupCurandState kernel sets unique sequence per state
 *   
 *   curandState *devPtrCurandStatesSite
 *     Size: N states (one per lattice site)
 *     Purpose: Independent RNG state for each site/particle
 *     Use: Particle movement and direction stochasticity (updateParticles kernel)
 *     Init: setupCurandState kernel sets unique sequence per state
 * 
 * Memory footprint estimation (for N sites, dim=2, z=6):
 *   
 *   Device memory:
 *     Particle/occupancy: 4*N (site) + 4*nParticles (index, direction) ≈ 8*N bytes
 *     Topology: 4*N (boundary) + 4*6*N (neighbor) + 4*6*N (bond) = 52*N bytes
 *     Elasticity: 8*2*N (u, du, b) + 8*2*N (x: coordinates) = 32*N bytes
 *     RNG: sizeof(curandState)*7*N ≈ 56*N bytes (large!)
 *     Total: ~148*N bytes (dominated by RNG)
 *   
 *   For N = 100,000:
 *     Total ≈ TODO MB (fits easily on modern GPUs with GBs)
 * 
 * Thread safety:
 *   - Host arrays: Single-threaded access only
 *   - Device arrays: Thread-safe within kernel blocks
 *   - No OpenMP parallelism (GPU handles parallelism)
 *   - Atomic operations prevent race conditions on device
 */
typedef struct {
    
    // ========================================================================
    // MEMORY SIZE TRACKING (in bytes)
    // ========================================================================
    // Used for allocation/deallocation of GPU memory
    
    size_t memorySite;                 // Size of site array
    size_t memoryPartX;                // size of particles position
    size_t memoryIndex;                // Size of particle index array
    size_t memoryShift;                // Size of particles steps on lattice
    size_t memoryDirection;            // Size of particle direction array
    size_t memoryNeighbor;             // Size of neighbor list
    size_t memoryBond;                 // Size of bond array
    size_t memoryCurandStatesBond;     // Size of bond RNG states
    size_t memoryCurandStatesSite;     // Size of site RNG states
    size_t memoryMSD;                  // Size of auxiliar vector for msd calculations

    // ========================================================================
    // HOST MEMORY ARRAYS (CPU side)
    // ========================================================================
    // These are CPU copies of device data, updated via syncAndCopyToCPU()
    
    int *site;                         // Host copy: occupancy state
    int *index;                        // Host copy: particle indices
    int *direction;                    // Host copy: particle directions
    int *bond;                         // Host copy: bond states

    double *x;                         // Host copy: particle positions
    double *x0;                        // Host copy: initial particle positions

    double *shiftDir0;                 // Host copy: horizontal lattice shifts
    double *shiftDir1;                 // Host copy: vertical lattice shifts

    // ========================================================================
    // DEVICE MEMORY ARRAYS (GPU side)
    // ========================================================================
    // Fast GPU memory, accessible only via kernels
    
    int *devPtrSite;                   // Occupancy: 1=occupied, 0=empty
    int *devPtrIndex;                  // Maps particle ID → site index
    int *devPtrDirection;              // Particle direction (0 to z-1)
    int *devPtrNeighbor;               // Neighbor list: neighbor[z*i + dir]
    int *devPtrBond;                   // Bond state: 1=active, 0=broken

    double *devPtrX;                   // Particle positions
    double *devPtrX0;                  // Initial particle positions

    double *devPtrShiftDir0;           // Horizontal lattice shifts
    double *devPtrShiftDir1;           // Vertical lattice shifts

    // ========================================================================
    // LATTICE AND SIMULATION PROPERTIES
    // ========================================================================
    // Structural and dynamical parameters
    
    int z;                             // Coordination number (neighbors per site: 8 or 6)
    int dim;                           // Spatial dimension (typically 2 for 2D)
    int iter;                          // Iteration counter for simulation progress
    unsigned long seed;                // RNG seed value for reproducible simulations
    double pack;                       // Packing fraction of ABPs (0.0 to 1.0)
    double pPerst;                     // Persistence probability
    double pRegen;                     // Bond regeneration probability
    int nParticles;                    // Total number of ABPs on network
    LatticeType type;                  // Type of lattice (enum: SQUARE_MOORE or TRIANGULAR)

    double msd;                        // Mean square displacement
    double c4;                         // Dinamical heterogeneity
    double alpha2;                     // non-Gaussianity parameter

    double *devPtrPartialMSD;          // Auxiliar vector to compute msd
    double *devPtrPartialMSD2;         // Auxiliar vector to compute c4

    // ========================================================================
    // DEVICE RNG STATES (CURAND library)
    // ========================================================================
    // Thread-safe independent random number generation
    
    curandState *devPtrCurandStatesBond;  // One RNG state per bond (for regeneration process)
    curandState *devPtrCurandStatesSite;  // One RNG state per site (for particle dynamics)

} network;


/**
 * ============================================================================
 * HOST FUNCTION DECLARATIONS - Core Network Operations
 * ============================================================================
 */

/**
 * Create and initialize network structure
 * 
 * Allocates GPU and CPU memory for all arrays, initializes lattice geometry,
 * and sets up initial conditions for particle-bond dynamics simulation.
 * 
 * Parameters:
 *   type - Lattice geometry type (SQUARE_MOORE or TRIANGULAR)
 *   dim - Spatial dimension (typically 2)
 *   pack - Packing fraction (0.0 to 1.0, fraction of sites with particles)
 *   pPerst - Persistence probability (0.0 to 1.0, probability of maintaining direction)
 *   pRegen - Bond regeneration probability (0.0 to 1.0, probability of bond repair)
 * 
 * Returns:
 *   Pointer to allocated and initialized network structure
 *   Must be freed with destroyNetwork() when done
 * 
 * Operations performed:
 *   1. Allocate network structure on CPU
 *   2. Allocate all GPU memory arrays
 *   3. Call geometry initialization kernels:
 *      - getNeighborList(): Build connectivity
 *      - getNetworkCoordinate(): Calculate coordinates
 *      - getBoundary(): Identify boundary sites
 *      - getProjector(): Compute elastic projectors
 *   4. Place particles: putABPOnNetwork()
 *   5. Initialize bonds: setBonds()
 *   6. Initialize RNG: initCurand()
 * 
 * Memory allocation (total for N sites, z=coordination):
 *   GPU: ~148*N bytes (see network structure documentation)
 *   CPU: Temporary during initialization, then ~40*N bytes for host copies
 * 
 * Example usage:
 *   
 *   // Create triangular lattice with 50% packing
 *   network *net = makeNetwork(TRIANGULAR, 2, 0.5, 0.8, 0.5);
 *   
 *   // Use network for simulation
 *   mcSteps(net, 1000);  // Run 1000 MC steps
 *   
 *   // Clean up
 *   destroyNetwork(net);
 * 
 * See also:
 *   - destroyNetwork(): Free memory
 *   - getNeighborList(), getNetworkCoordinate(), etc.: Initialization
 */
__host__ network *makeNetwork(LatticeType, const int, const double, const double, const double);

/**
 * Destroy network and free all memory
 * 
 * Deallocates GPU and CPU memory for all network arrays.
 * Must be called to prevent memory leaks.
 * 
 * Parameters:
 *   pN - pointer to network (safe to pass NULL)
 * 
 * Freed memory:
 *   - All GPU arrays (device pointers)
 *   - All CPU arrays (host pointers)
 *   - Network structure itself
 * 
 * Performance:
 *   - O(1) operation (immediate deallocation)
 *   - Typically < 1ms
 * 
 * See also:
 *   - makeNetwork(): Create network
 */
__host__ void destroyNetwork(network *);



/**
 * ============================================================================
 * HOST FUNCTION DECLARATIONS - Lattice Geometry Initialization
 * ============================================================================
 */


/**
 * Copy lattice-specific shift vectors from host to memory
 * 
 * Initializes shift vectors (horizontal and vertical displacement per direction)
 * based on lattice type. These define how particles move on the lattice.
 * 
 * Operations:
 *   - Copies lattice-specific constants to host memory
 *   - Shift vectors define connectivity geometry
 *   - Different lattices have different shift patterns
 * 
 * Parameters:
 *   pN - pointer to network
 * 
 * Output:
 *   pN -> shiftDir0: horizontal shift per direction
 *   pN -> shiftDir1: vertical shift per direction
 * 
 * See also:
 *   - getParticlesCoordinate(): Uses shifts for position calculation
 *   - makeNetwork(): Calls during initialization
 */
__host__ void getShift(network *);


/**
 * Build neighbor connectivity list
 * 
 * Computes and stores the neighbor indices for each lattice site.
 * Called during makeNetwork() initialization.
 * 
 * Calls kerGetNeighborList CUDA kernel
 * 
 * Output:
 *   pN -> devPtrNeighbor[z*i + dir] = index of neighbor at site i, direction dir
 * 
 * Properties:
 *   - Periodic boundary conditions (toroidal topology)
 *   - Sites wrap around at edges (no true boundary)
 *   - Independent of particle positions (topology only)
 * 
 * See also:
 *   - kerGetNeighborList: Device kernel
 */
__host__ void getNeighborList(network *);


/**
 * Calculate physical coordinates of particles on lattice sites
 * 
 * Computes (x, y) Cartesian coordinates for each site based on lattice type.
 * Different lattices have different geometric arrangements.
 * Called during makeNetwork() initialization.
 * 
 * Calls kerGetNetworkCoordinate CUDA kernel
 * 
 * Output:
 *   pN->devPtrX[dim*i + comp] = coordinate component for site i
 * 
 * Coordinate systems:
 *   SQUARE_MOORE: x[i] = DIST * col, y[i] = DIST * row
 *   TRIANGULAR: x[i] = DIST * (col + 0.5*row), y[i] = row * DIST * sqrt(3)/2
 * 
 * Physical interpretation:
 *   - Used in elasticity calculations
 *   - Defines metric (distances between sites)
 *   - Determines bond direction for projection matrices
 * 
 * See also:
 *   - kerGetNetworkCoordinate: Device kernel
 *   - getProjector(): Uses coordinates to compute projectors
 */
__host__ void getParticlesCoordinate(network *);



/**
 * Place active Brownian particles randomly on lattice sites
 * 
 * Uses Fisher-Yates shuffle algorithm to select random unique sites
 * and assigns random initial directions to each particle.
 * Called during makeNetwork() initialization.
 * 
 * Operations:
 *   1. Initialize all sites as empty (site[i] = 0)
 *   2. Create array of all site indices [0, N-1]
 *   3. Partial Fisher-Yates shuffle to select nParticles unique sites
 *   4. For each selected site:
 *      - Mark site as occupied (site[i] = 1)
 *      - Store particle position (index[p] = site_i)
 *      - Assign random direction (direction[p] = rand() % z)
 *   5. Free temporary indices array
 * 
 * Parameters:
 *   pN - pointer to network
 * 
 * Output:
 *   pN -> site[i]: 1 if site occupied, 0 if empty
 *   pN -> index[p]: which lattice site particle p occupies
 *   pN -> direction[p]: initial direction [0, z-1] for particle p
 * 
 * Properties:
 *   - No two particles on same site (guaranteed unique positions)
 *   - Uniform random placement (each site equally likely)
 *   - Random initial directions (isotropic distribution)
 *   - Executed on CPU (RNG convenience, small memory overhead)
 * 
 * Complexity:
 *   - Time: O(nParticles) with Fisher-Yates shuffle
 *   - Space: O(N) temporary array
 * 
 * See also:
 *   - makeNetwork(): Calls during initialization
 */
__host__ void putABPOnNetwork(network *);

/**
 * Initialize all bonds to given state
 * 
 * Sets all bonds to uniform state (typically 1=active).
 * Called during makeNetwork() initialization to set initial connectivity.
 * 
 * Calls kerSetBonds CUDA kernel
 * 
 * Parameters:
 *   pN - pointer to network
 *   value - state value to assign (typically 1 for active, 0 for broken)
 * 
 * Output:
 *   pN -> devPtrBond[z*i + dir] = value of the bond originating at site i
 *                                 and pointing in direction dir
 * 
 * Common initialization:
 *   setBonds(net, 1)  // All bonds active initially
 * 
 * See also:
 *   - kerSetBonds: Device kernel
 *   - updateBonds: Modifies bonds during simulation
 */
__host__ void setBonds(network *, const int);

/**
 * Initialize cuRAND random number generator states
 * 
 * Sets up independent RNG states for each bond and particle.
 * Enables thread-safe random number generation throughout simulation.
 * Called during makeNetwork() initialization.
 * 
 * Calls setupCurandState CUDA kernel (for both bond and site states)
 * 
 * Operations:
 *   - Initialize devPtrCurandStatesBond with unique sequences
 *   - Initialize devPtrCurandStatesSite with unique sequences
 *   - Each thread gets independent random number stream
 * 
 * Properties:
 *   - Reproducible: same seed produces same sequence
 *   - Independent: different sequences never overlap
 *   - Thread-safe: no race conditions on RNG state
 * 
 * Set seed before calling:
 *   pN->seed = 12345;  // Or from environment
 *   initCurand(pN);
 * 
 * See also:
 *   - setupCurandState: Device kernel
 *   - updateParticles, updateBonds: Use RNG for stochasticity
 */
__host__ void initCurand(network *);


/**
 * ============================================================================
 * HOST FUNCTION DECLARATIONS - Simulation Execution
 * ============================================================================
 */

/**
 * Execute single Monte Carlo step
 * 
 * Performs one complete update cycle:
 *   1. updateParticles: Move and reorient particles
 *   2. updateBonds: Regenerate broken bonds stochastically
 * 
 * Parameters:
 *   pN - pointer to network
 * 
 * Side effects:
 *   - Increments pN->iter
 *   - Modifies devPtrSite, devPtrIndex, devPtrDirection
 *   - Modifies devPtrBond based on regeneration probability
 * 
 * See also:
 *   - mcSteps(): Execute multiple steps
 *   - updateParticles: Device kernel
 *   - updateBonds: Device kernel
 */
__host__ void mcStep(network *);


/**
 * ============================================================================
 * HOST FUNCTION DECLARATIONS - Data Synchronization
 * ============================================================================
 */

/**
 * Synchronize and copy GPU data to CPU memory
 * 
 * Transfers current state of GPU arrays to host (CPU) memory.
 * Necessary before CPU analysis or printing device data.
 * 
 * GPU → CPU transfers:
 *   - devPtrSite → site (occupancy)
 *   - devPtrIndex → index (particle positions)
 *   - devPtrDirection → direction (particle orientations)
 *   - devPtrBond → bond (bond states)
 * 
 * Parameters:
 *   pN - pointer to network
 * 
 * Performance:
 *   - D2H transfer (PCIe bandwidth: ~10-20 GB/s)
 *   - ~100-500 μs for typical network size
 *   - Should NOT be called every iteration (use sparingly)
 * 
 * Example usage:
 *   
 *   // Simulate
 *   mcSteps(net, 1000);
 *   
 *   // Analyze on CPU
 *   syncAndCopyToCPU(net);
 *   
 *   // Now can directly access host arrays
 *   for (int i = 0; i < N; i++) {
 *       if (net->site[i] == 1) {
 *           // Site occupied, analyze particle
 *       }
 *   }
 * 
 * See also:
 *   - printDU(): Debug output using CPU data
 */
__host__ void syncAndCopyToCPU(network *);


/**
 * ============================================================================
 * DEVICE KERNEL DECLARATIONS - Geometry
 * ============================================================================
 */

/**
 * Device kernel: Build neighbor connectivity list
 * Host wrapper: getNeighborList()
 * See getNeighborList() for documentation
 */
__global__ void kerGetNeighborList(int *, int, LatticeType);

/**
 * Device kernel: Calculate site coordinates
 * Host wrapper: getNetworkCoordinate()
 * See getNetworkCoordinate() for documentation
 */
__global__ void kerGetNetworkCoordinate(double *, int, LatticeType);


/**
 * ============================================================================
 * DEVICE KERNEL DECLARATIONS - Initialization
 * ============================================================================
 */

/**
 * Device kernel: Set all bonds to given value
 * Host wrapper: setBonds()
 * See setBonds() for documentation
 */
__global__ void kerSetBonds(int *, int, int);

/**
 * Device kernel: Initialize cuRAND RNG states
 * Host wrapper: initCurand()
 * See initCurand() for documentation
 */
__global__ void setupCurandState(curandState *, const int, unsigned long);


/**
 * ============================================================================
 * DEVICE KERNEL DECLARATIONS - Dynamics
 * ============================================================================
 */

/**
 * Device kernel: Update particle positions and directions
 * 
 * Each MC step:
 *   1. With probability (1-pPerst), change direction
 *   2. Attempt move to neighbor site if bond exists
 *   3. Use atomicCAS for collision-free movement
 *   4. Deactivate used bond
 * 
 * Parameters:
 *   site - occupancy array (modified by atomicCAS/atomicExch)
 *   neighbor - neighbor list (read-only)
 *   bond - bond states (1=active, 0=broken)
 *   index - particle indices (modified)
 *   direction - particle directions (modified)
 *   nParticles - total number of particles
 *   z - coordination number
 *   states - RNG states (one per particle)
 * 
 * Thread safety:
 *   One thread per particle
 *   Atomic operations prevent race conditions
 *   Independent RNG per particle (cuRAND states)
 * 
 * Calls in: mcStep()
 */

__global__ void updateParticles(int *, int *, int *, int *, int *, int, int, int, curandState *);

/**
 * Device kernel: Update bond regeneration
 * 
 * Each MC step:
 *   1. With probability pRegen, regenerate broken bonds
 *   2. Only update once per bond (lower index site only)
 *   3. Update bidirectionally for consistency
 * 
 * Parameters:
 *   bond - bond states (modified)
 *   neighbor - neighbor list (for finding opposite site)
 *   z - coordination number
 *   bondStates - RNG states (one per bond)
 * 
 * Thread safety:
 *   One thread per bond
 *   Atomic operations for consistency
 *   Independent RNG per bond
 * 
 * Calls in: mcStep()
 */
__global__ void updateBonds(int *, int *, int, curandState *);


/**
 * ============================================================================
 * Device Kernel: Get coordination number distribution
 * ============================================================================
 *
 * Computes the histogram of coordination numbers across all sites.
 * Provides statistical information about network connectivity patterns.
 *
 * Purpose:
 *   - Analyze how coordination number varies throughout the network
 *   - Identify defects or regions with different connectivity
 *   - Characterize network disorder/heterogeneity
 *   - Monitor network evolution during simulation
 *
 * Parameters:
 *   bond - bond state array (read-only)
 *     Size: N*z integers
 *     Layout: bond[z*i + dir] = state of bond at site i, direction dir
 *     Values: 1=active (counts toward coordination), 0=broken (doesn't count)
 *   
 *   z - coordination number (neighbors per site)
 *     Values: 6 (TRIANGULAR), 8 (SQUARE_MOORE)
 *   
 *   distrib - output histogram array (modified)
 *     Size: (z+1) integers
 *     Purpose: distrib[k] = number of sites with exactly k active bonds
 *     Range: k ∈ [0, z] (minimum 0 bonds, maximum z bonds)
 *     Layout: distrib[0] = isolated sites
 *             distrib[1] = sites with 1 active bond
 *             ...
 *             distrib[z] = sites with all z bonds active (fully connected)
 *
 * Algorithm:
 *   1. Each thread processes one site
 *   2. Counts active bonds at that site (sum of bond[z*i + dir])
 *   3. Atomically increments distrib[count]
 *   4. Result: histogram of coordination number distribution
 *
 * Output interpretation:
 *   - If distrib[6] = 100: 100 sites have all 6 bonds active (triangular)
 *   - If distrib[3] = 50:  50 sites have exactly 3 active bonds (half-connected)
 *   - If distrib[0] = 5:   5 isolated sites (connectivity = 0)
 *   - Sum(distrib[k]) = N (total sites)
 *
 * Physical significance:
 *   - Higher distrib[z] → more connected, stiffer network
 *   - More spread distribution → heterogeneous connectivity
 *   - Indicates bond breaking/healing patterns
 *
 * Parallelization:
 *   - One thread per site (N threads total)
 *   - Atomic operations ensure safe histogram updates
 *   - No synchronization needed (atomic operations are independent)
 *
 * Performance:
 *   - Time: O(N) to scan all sites
 *   - Space: O(z) for output histogram (small)
 *
 * Usage example:
 *   
 *   int *devDistrib;
 *   cudaMalloc(&devDistrib, (z+1) * sizeof(int));
 *   
 *   // Initialize histogram to zero
 *   cudaMemset(devDistrib, 0, (z+1) * sizeof(int));
 *   
 *   // Compute distribution
 *   getZDistb<<<nBlocks, nThreads>>>(devPtrBond, z, devDistrib);
 *   
 *   // Copy to host
 *   int *hostDistrib = (int*)malloc((z+1) * sizeof(int));
 *   cudaMemcpy(hostDistrib, devDistrib, (z+1) * sizeof(int), 
 *              cudaMemcpyDeviceToHost);
 *   
 *   // Analyze
 *   printf("Coordination number distribution:\n");
 *   for (int k = 0; k <= z; k++) {
 *       printf("  z=%d: %d sites (%.1f%%)\n", k, hostDistrib[k],
 *              100.0 * hostDistrib[k] / N);
 *   }
 *
 * Related functions:
 *   - getMeanCoordinationNumber(): Computes average z (scalar)
 *   - getZDistb(): This function (histogram distribution)
 *   - Difference: Mean vs full distribution characterization
 *
 * Notes:
 *   - Output must be initialized to zero before kernel call
 *   - Atomic operations ensure correctness with concurrent threads
 *   - Works with both SQUARE_MOORE (z=8) and TRIANGULAR (z=6) lattices
 *   - Distribution size always (z+1) to accommodate [0, z] range
 */
__global__ void getZDistb(const int *, int, int *);

#endif  // __NETWORK_H__
