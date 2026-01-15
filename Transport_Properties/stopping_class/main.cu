#include "config.h"
#include "network.h"


// ExampNe main
int main() {

    // ------------------------------
    // 1. Create network
    // ------------------------------
    int dim = 2;                // spatial dimension
    
    LatticeType type = LATTICE_TYPE;

    network *pN = makeNetwork(
        type,               // Lattice type
        dim,                // Dimension (2D)
        PACKING_FRACTION,   // Packing fraction
        P_REGEN,            // Bond regeneration probability
        P_PERST             // Particle persistence probability
    );

    if (!pN) {
        fprintf(stderr, "Failed to create network!\n");
        return EXIT_FAILURE;
    }

    int nParticles = pN -> nParticles;
    size_t memoryCounts = 4 * sizeof(int);

    // ------------------------------
    // 2. Initialize curand states
    // ------------------------------
    initCurand(pN);


    while (pN -> iter < THERMAL) {
        mcStep(pN);
        //printf("%d\n", pN -> iter);
    }

    pN -> iter = 0;

    int* devPtrCounts;
    HANDLE_ERROR(cudaMalloc(&devPtrCounts, 4 * sizeof(int)));

    unsigned threadsPerBlock = NUMBER_OF_THREADS_PER_BLOCK;
    unsigned blocksForStates = (nParticles + threadsPerBlock - 1) / threadsPerBlock;

    int counts[4];


    while (pN -> iter < ITERATION) {

        HANDLE_ERROR(cudaMemset(devPtrCounts, 0, memoryCounts));

        reduceStopCauses<<<blocksForStates, threadsPerBlock>>>(pN -> devPtrStopReason, devPtrCounts, nParticles);

        HANDLE_ERROR(cudaMemcpy(counts, devPtrCounts, memoryCounts, cudaMemcpyDeviceToHost));

        printf("%d\t%d\t%d\t%d\t%d\n", pN -> iter, counts[0], counts[1], counts[2], counts[3]);

        mcStep(pN);
        
    }


    destroyNetwork(pN);
    HANDLE_ERROR(cudaFree(devPtrCounts));

    return 0;
}
