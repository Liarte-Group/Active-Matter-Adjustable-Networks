#!/bin/bash

# Set up environment variables for CUDA
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Check if required commands are installed
command -v nvcc >/dev/null 2>&1 || { echo >&2 "nvcc not found. Please install CUDA."; exit 1; }
command -v bc >/dev/null 2>&1 || { echo >&2 "bc not found. Please install bc."; exit 1; }


LATTICE=${1:-"TRIANGULAR"}
LX=${2:-64}
HT=${3:-20.0}

NVCC=nvcc
ARCH_FLAGS="-arch=sm_86"
LDFLAGS="-lcurand"
SRC="main.cu src/network.cu src/network_kernels.cu"

# Function to run the simulation
run_simulation() {
    local PACKING_FRACTION=$1
    local PT=$2
    local realization=$3
    local NUMBER_OF_THREADS_PER_BLOCK=1024
    
    # Calculate P_PERST and P_REGEN from PT and HT (HT is global)
    # P_PERST = 1 - (1/PT)
    local P_PERST=$(echo "scale=12; 1 - (1 / $PT)" | bc)

    # P_REGEN = 1/HT
    local P_REGEN=$(echo "scale=12; 1 / $HT" | bc)
    
    # Format parameters with 6 decimals for filename
    local pack_fmt=$(printf "%.6f" "$PACKING_FRACTION")
    local pt_fmt=$(printf "%.6f" "$PT")
    local ht_fmt=$(printf "%.6f" "$HT")
    
    local CFLAGS="-Iinclude -DLATTICE_TYPE=${LATTICE} -DLX=${LX} -DPACKING_FRACTION=${PACKING_FRACTION} -DP_PERST=${P_PERST} -DP_REGEN=${P_REGEN} -DNUMBER_OF_THREADS_PER_BLOCK=${NUMBER_OF_THREADS_PER_BLOCK} -Xcompiler -Wall,-Wextra"
    
    # Compile and run the simulation
    ${NVCC} ${ARCH_FLAGS} ${CFLAGS} ${SRC} ${LDFLAGS} || { echo "Compilation failed"; exit 1; }
    ./a.out > "msd_${timestamp}_L_${LX}_PACK_${pack_fmt}_PT_${pt_fmt}_HT_${ht_fmt}_R_${realization}.dat" || { echo "Simulation failed"; exit 1; }
}

# Number of realizations (replicates) of the experiment
NUM_REALIZATIONS=2

# Define arrays for parameters
PT_VALUES=(1 2 3 4 6 8 12 16 24 32 48 64 96 128 192 256 384 512 768 1024)
PACKING_FRACTION_VALUES=(0.016 0.032 0.064 0.128 0.256)

# Calculate total number of iterations per realization
NUM_PT_VALUES=${#PT_VALUES[@]}
NUM_PACK_VALUES=${#PACKING_FRACTION_VALUES[@]}
TOTAL_ITERATIONS=$((NUM_PT_VALUES * NUM_PACK_VALUES))

echo "Configuration:"
echo "  LX: $LX"
echo "  HT: $HT"
echo "  PT values: ${#PT_VALUES[@]}"
echo "  PACKING_FRACTION values: ${#PACKING_FRACTION_VALUES[@]}"
echo "  Total iterations per realization: $TOTAL_ITERATIONS"
echo "  Total realizations: $NUM_REALIZATIONS"
echo ""

# Realization loop
for realization in $(seq 1 $NUM_REALIZATIONS); do
    echo "Starting realization $realization..."
    
    # Create a main directory for this realization
    REALIZATION_DIR="Realization_${realization}"
    mkdir -p "$REALIZATION_DIR" || { echo "Failed to create realization directory"; exit 1; }
    
    # Counter for completed iterations
    completed_iterations=0
    
    # Main loop
    for pt in "${PT_VALUES[@]}"; do
        for pack in "${PACKING_FRACTION_VALUES[@]}"; do
            run_simulation $pack $pt $realization
            
            # Increment the completed iterations counter
            completed_iterations=$((completed_iterations + 1))
            
            # Calculate progress percentage
            progress=$(echo "scale=4; $completed_iterations / $TOTAL_ITERATIONS * 100" | bc)
            progress=$(printf "%.2f" "$progress")
            
            # Print progress
            echo "Realization $realization: $progress% completed ($completed_iterations/$TOTAL_ITERATIONS)"
        done
    done
    
    # Create a subdirectory for the results
    ht_fmt=$(printf "%.6f" "$HT")
    mkdir -p "$REALIZATION_DIR/MSD_HT_${ht_fmt}" || { echo "Failed to create directory"; exit 1; }
    mv *.dat "$REALIZATION_DIR/MSD_HT_${ht_fmt}/" || { echo "Failed to move files"; exit 1; }
    
    echo "Realization $realization completed."
    echo ""
done

echo "All realizations of the experiment have been completed successfully."
