#include "config.h"
#include "network.h"


int main() {

    // ------------------------------
    // 1. Create network
    // ------------------------------
    int dim = 2;
    LatticeType type = LATTICE_TYPE;

    network *pN = makeNetwork(
        type,
        dim,
        PACKING_FRACTION,
        P_REGEN,
        P_PERST
    );

    if (!pN) {
        fprintf(stderr, "Failed to create network!\n");
        return EXIT_FAILURE;
    }

    // ------------------------------
    // 2. Initialize curand states
    // ------------------------------
    initCurand(pN);

    // Thermalization
    while (pN->iter < THERMAL) {
        mcStep(pN);
    }
    pN->iter = 0;

    // ------------------------------
    // 3. Allocate accumulator
    // ------------------------------
    int *devPtrCoordination;
    HANDLE_ERROR(cudaMalloc((void **)&devPtrCoordination, sizeof(int)));

    unsigned threadsPerBlock = NUMBER_OF_THREADS_PER_BLOCK;
    unsigned blocksForStates = (N + threadsPerBlock - 1) / threadsPerBlock;

    // ------------------------------
    // 4. Main loop
    // ------------------------------
    while (pN->iter < ITERATION) {

        mcStep(pN);

        if (pN -> iter % PASS_TIME == 0) {

            // Measure every step (o cambia a %100 si quieres)
            HANDLE_ERROR(cudaMemset(devPtrCoordination, 0, sizeof(int)));

            getZDistb<<<blocksForStates, threadsPerBlock>>>(
                pN->devPtrBond,
                pN->z,
                devPtrCoordination
            );

            int coordination = 0;
            HANDLE_ERROR(cudaMemcpy(&coordination, devPtrCoordination, sizeof(int), cudaMemcpyDeviceToHost));

            double zMean = (double)coordination / (double)N;

            printf("%d\t%.6f\n", pN->iter / 100, zMean);
        }
    }

    // ------------------------------
    // 5. Cleanup
    // ------------------------------
    HANDLE_ERROR(cudaFree(devPtrCoordination));
    destroyNetwork(pN);

    return 0;
}
