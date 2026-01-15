#include "config.h"
#include "network.h"


// Example main
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

    // ------------------------------
    // 2. Initialize curand states
    // ------------------------------
    initCurand(pN);


    while (pN -> iter < THERMAL) {
        mcStep(pN);
    }

    pN -> iter = 0;
    storeCoordinates(pN);

    while (pN -> iter < ITERATION) {

        updateMSD(pN);
        
        printf("%d\t%lf\t%lf\t%lf\n", pN -> iter, pN -> msd, pN -> c4, pN -> alpha2);
        
        mcStep(pN);
    }


    return 0;
}
