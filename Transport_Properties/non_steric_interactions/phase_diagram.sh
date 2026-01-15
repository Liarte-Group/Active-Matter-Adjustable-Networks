#!/bin/bash
# Set up environment variables for CUDA
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# Check if required commands are installed
command -v nvcc >/dev/null 2>&1 || { echo >&2 "nvcc not found. Please install CUDA."; exit 1; }
command -v bc >/dev/null 2>&1 || { echo >&2 "bc not found. Please install bc."; exit 1; }

LATTICE=${1:-"TRIANGULAR"}
LX=${2:-64}

NVCC=nvcc
ARCH_FLAGS="-arch=sm_86"
LDFLAGS="-lcurand"
SRC="main.cu src/network.cu src/network_kernels.cu"

# Generate HT values logarítmicamente espaciados entre 1 y 320
generate_ht_values() {
    local min=1
    local max=320
    local n_values=15
    local ht_values=()
    
    # log(max/min) / (n_values - 1)
    local log_ratio=$(echo "scale=10; l($max / $min) / ($n_values - 1)" | bc -l)
    
    for i in $(seq 0 $((n_values - 1))); do
        # HT = min * exp(i * log_ratio)
        local ht=$(echo "scale=6; $min * e($i * $log_ratio)" | bc -l)
        ht_values+=("$ht")
    done
    
    echo "${ht_values[@]}"
}

# Function to run the simulation
run_simulation() {
    local PACKING_FRACTION=$1
    local PT=$2
    local HT=$3
    local realization=$4
    local NUMBER_OF_THREADS_PER_BLOCK=1024
    
    # Calculate P_PERST and P_REGEN from PT and HT
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
PACKING_FRACTION_VALUES=(0.016000 0.019248 0.023156 0.027858 0.033513 0.040317 0.048503 0.058350 0.070197 0.084449 0.101594 0.122220 0.147033 0.176885 0.212797 0.256000)

# Generate HT values
HT_VALUES=($(generate_ht_values))

# Calculate total number of iterations per realization
NUM_PT_VALUES=${#PT_VALUES[@]}
NUM_PACK_VALUES=${#PACKING_FRACTION_VALUES[@]}
NUM_HT_VALUES=${#HT_VALUES[@]}
TOTAL_ITERATIONS=$((NUM_PT_VALUES * NUM_PACK_VALUES * NUM_HT_VALUES))

echo "Configuration:"
echo "  LX: $LX"
echo "  LATTICE: $LATTICE"
echo "  PT values: ${#PT_VALUES[@]}"
echo "  PACKING_FRACTION values: ${#PACKING_FRACTION_VALUES[@]}"
echo "  HT values: ${#HT_VALUES[@]}"
echo "    (logarítmicamente espaciados entre 1 y 320)"
echo "  Total iterations per realization: $TOTAL_ITERATIONS"
echo "  Total realizations: $NUM_REALIZATIONS"
echo ""
echo "HT values:"
for i in "${!HT_VALUES[@]}"; do
    printf "  %2d: %.6f\n" $((i+1)) "${HT_VALUES[$i]}"
done
echo ""

# Realization loop
for realization in $(seq 1 $NUM_REALIZATIONS); do
    echo "Starting realization $realization..."
    
    # Create a main directory for this realization
    REALIZATION_DIR="Realization_${realization}"
    mkdir -p "$REALIZATION_DIR" || { echo "Failed to create realization directory"; exit 1; }
    
    # Counter for completed iterations
    completed_iterations=0
    
    # Main loop over HT, PT, and PACKING_FRACTION
    for ht in "${HT_VALUES[@]}"; do
        for pt in "${PT_VALUES[@]}"; do
            for pack in "${PACKING_FRACTION_VALUES[@]}"; do
                run_simulation $pack $pt $ht $realization
                
                # Increment the completed iterations counter
                completed_iterations=$((completed_iterations + 1))
                
                # Calculate progress percentage
                progress=$(echo "scale=4; $completed_iterations / $TOTAL_ITERATIONS * 100" | bc)
                progress=$(printf "%.2f" "$progress")
                
                # Print progress
                echo "Realization $realization: $progress% completed ($completed_iterations/$TOTAL_ITERATIONS)"
            done
        done
    done
    
    # Create subdirectories for the results organized by HT
    for ht in "${HT_VALUES[@]}"; do
        ht_fmt=$(printf "%.6f" "$ht")
        mkdir -p "$REALIZATION_DIR/MSD_HT_${ht_fmt}" || { echo "Failed to create directory"; exit 1; }
        mv "msd_"*"HT_${ht_fmt}"* "$REALIZATION_DIR/MSD_HT_${ht_fmt}/" 2>/dev/null || true
    done
    
    echo "Realization $realization completed."
    echo ""
done

echo "All realizations of the experiment have been completed successfully."