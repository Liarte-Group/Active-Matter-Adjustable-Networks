#!/bin/bash

################################################################################
# Batch Simulation Script: Stopping Class Analysis
################################################################################
#
# This script orchestrates multiple simulations to measure particle stopping
# class distribution across varying network parameters (PT, packing fraction).
#
# Workflow:
#   1. Setup CUDA environment
#   2. Validate command-line arguments
#   3. Parse parameters (LATTICE_TYPE, LX, HT)
#   4. Define parameter sweeps (PT values and packing fractions)
#   5. For each realization:
#      - Create output directory
#      - For each parameter combination:
#        - Calculate derived parameters (P_PERST, P_REGEN)
#        - Compile and execute simulation
#        - Save stopping class data to file
#      - Organize output files
#   6. Report completion
#
# Author: William G. C. Oropesa (Liarte-Group)
# Institution: ICTP South American Institute for Fundamental Research
# GitHub Repository: https://github.com/Liarte-Group/Active-Matter-Adjustable-Networks
# Date: January 2026
################################################################################

# ============================================================================
# COLOR DEFINITIONS FOR OUTPUT
# ============================================================================
# ANSI color codes for colorized terminal output
RED='\033[0;31m'           # Red color for errors
GREEN='\033[0;32m'         # Green color for success
YELLOW='\033[1;33m'        # Yellow color for warnings
BLUE='\033[0;34m'          # Blue color for info
CYAN='\033[0;36m'          # Cyan color for data/progress
MAGENTA='\033[0;35m'       # Magenta color for section headers
WHITE='\033[0;37m'         # White color for normal text
BOLD='\033[1m'             # Bold text
NC='\033[0m'               # No Color (reset to default)

# ============================================================================
# STEP 1: Setup CUDA environment variables
# ============================================================================
# Configure paths for CUDA compiler and libraries
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

echo -e "${MAGENTA}${BOLD}============================================================================${NC}"
echo -e "${MAGENTA}${BOLD}Stopping Class Analysis - Batch Simulation${NC}"
echo -e "${MAGENTA}${BOLD}============================================================================${NC}\n"

# ============================================================================
# STEP 2: Verify required commands are available
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Checking required commands..."

command -v nvcc >/dev/null 2>&1 || { 
    echo -e "${RED}[ERROR]${NC} nvcc not found. Please install CUDA."
    exit 1
}
echo -e "${GREEN}[OK]${NC} NVIDIA CUDA compiler (nvcc) found"

command -v bc >/dev/null 2>&1 || { 
    echo -e "${RED}[ERROR]${NC} bc not found. Please install bc for floating-point arithmetic."
    exit 1
}
echo -e "${GREEN}[OK]${NC} Basic calculator (bc) found\n"

# ============================================================================
# STEP 3: Generate timestamp for unique file identification
# ============================================================================
# Creates unique filename prefix to avoid overwriting previous results
# Format: YYYYMMDD_HHMMSS (human-readable timestamp)
timestamp=$(date +%Y%m%d_%H%M%S)
echo -e "${CYAN}[TIMESTAMP]${NC} Simulation batch ID: ${BOLD}${timestamp}${NC}\n"

# ============================================================================
# STEP 4: Parse and validate command-line arguments
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Parsing command-line arguments..."

# ========================================================================
# Parameter 1: Lattice type (default: TRIANGULAR)
# ========================================================================
LATTICE=${1:-"TRIANGULAR"}
if [[ "$LATTICE" != "SQUARE_MOORE" && "$LATTICE" != "TRIANGULAR" ]]; then
    echo -e "${RED}[ERROR]${NC} Invalid lattice type: $LATTICE"
    echo -e "${YELLOW}[USAGE]${NC} ./stopping_class.sh [LATTICE_TYPE] [LX] [HT]"
    echo -e "${YELLOW}        LATTICE_TYPE: SQUARE_MOORE or TRIANGULAR (default: TRIANGULAR)${NC}"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Lattice type: ${BOLD}${LATTICE}${NC}"

# ========================================================================
# Parameter 2: Linear lattice size (default: 32)
# ========================================================================
LX=${2:-32}
if ! [[ "$LX" =~ ^[0-9]+$ ]] || [ "$LX" -lt 16 ]; then
    echo -e "${RED}[ERROR]${NC} Invalid LX value: $LX (must be integer >= 16)"
    echo -e "${YELLOW}[USAGE]${NC} ./stopping_class.sh [LATTICE_TYPE] [LX] [HT]"
    exit 1
fi
N=$((LX * LX))
echo -e "${GREEN}[OK]${NC} Lattice size: ${BOLD}${LX} × ${LX} = ${N} sites${NC}"

# ========================================================================
# Parameter 3: Healing time (default: 40.0)
# ========================================================================
HT=${3:-40.0}
if ! [[ "$HT" =~ ^[0-9]+\.?[0-9]*$ ]] || (( $(echo "$HT <= 0" | bc -l) )); then
    echo -e "${RED}[ERROR]${NC} Invalid HT value: $HT (must be positive number)"
    echo -e "${YELLOW}[USAGE]${NC} ./stopping_class.sh [LATTICE_TYPE] [LX] [HT]"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Healing time: ${BOLD}${HT}${NC}\n"

# ============================================================================
# STEP 5: Configure CUDA compilation
# ============================================================================
NVCC=nvcc
ARCH_FLAGS="-arch=sm_86"
LDFLAGS="-lcurand"
SRC="main.cu src/network.cu src/network_kernels.cu"

echo -e "${BLUE}[INFO]${NC} CUDA compilation configuration:"
echo -e "  ${CYAN}Compiler:${NC} ${BOLD}${NVCC}${NC}"
echo -e "  ${CYAN}Architecture:${NC} ${BOLD}sm_86${NC} (Ampere: RTX 30 series, RTX A6000)"
echo -e "  ${CYAN}Source files:${NC} ${BOLD}${SRC}${NC}\n"

# ============================================================================
# STEP 6: Define function to run individual simulation
# ============================================================================
# run_simulation: Execute one simulation with specified parameters
#
# Parameters:
#   $1 (PACKING_FRACTION) - Fraction of lattice sites occupied by particles
#   $2 (PT) - Persistence time (controls particle direction changes)
#   $3 (realization) - Replication number (for ensemble averaging)
#
# Returns:
#   0 on success, 1 on compilation or execution failure
run_simulation() {
    local PACKING_FRACTION=$1
    local PT=$2
    local realization=$3
    local NUMBER_OF_THREADS_PER_BLOCK=1024
    
    # ========================================================================
    # Calculate P_PERST (persistence probability) from PT
    # ========================================================================
    # PT = expected steps before direction change
    # P_PERST = 1 - 1/PT (ensures geometric distribution with mean PT)
    local P_PERST=$(echo "scale=12; 1 - (1 / $PT)" | bc)

    # ========================================================================
    # Calculate P_REGEN (bond regeneration probability) from HT
    # ========================================================================
    # HT = expected steps before bond regeneration
    # P_REGEN = 1/HT (ensures geometric distribution with mean HT)
    # Note: HT is a global parameter passed to this script
    local P_REGEN=$(echo "scale=12; 1 / $HT" | bc)
    
    # ========================================================================
    # Format parameters for filename (6 decimal places)
    # ========================================================================
    local pack_fmt=$(printf "%.6f" "$PACKING_FRACTION")
    local pt_fmt=$(printf "%.6f" "$PT")
    local ht_fmt=$(printf "%.6f" "$HT")
    
    # ========================================================================
    # Build compiler flags with all parameters
    # ========================================================================
    # Compile-time constants passed to CUDA kernel
    # These enable compile-time optimizations and prevent runtime overhead
    local CFLAGS="-Iinclude -DLATTICE_TYPE=${LATTICE} -DLX=${LX} -DPACKING_FRACTION=${PACKING_FRACTION} -DP_PERST=${P_PERST} -DP_REGEN=${P_REGEN} -DNUMBER_OF_THREADS_PER_BLOCK=${NUMBER_OF_THREADS_PER_BLOCK} -Xcompiler -Wall,-Wextra"
    
    # ========================================================================
    # Compile simulation with error checking
    # ========================================================================
    ${NVCC} ${ARCH_FLAGS} ${CFLAGS} ${SRC} ${LDFLAGS} 2>&1 >/dev/null
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${NC} Compilation failed for PACK=${pack_fmt}, PT=${pt_fmt}"
        return 1
    fi
    
    # ========================================================================
    # Execute simulation and save output
    # ========================================================================
    # Output filename format:
    #   stoppingClass_TIMESTAMP_L_SIZE_PACK_FRACTION_PT_TIME_HT_TIME_R_REALIZATION.dat
    #
    # Fields:
    #   TIMESTAMP - Unique batch identifier
    #   SIZE - Lattice dimension (LX)
    #   FRACTION - Packing fraction
    #   PT_TIME - Persistence time
    #   HT_TIME - Healing time
    #   REALIZATION - Replication number
    local output_file="stoppingClass_${timestamp}_L_${LX}_PACK_${pack_fmt}_PT_${pt_fmt}_HT_${ht_fmt}_R_${realization}.dat"
    
    ./a.out > "$output_file" 2>&1
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${NC} Simulation failed for PACK=${pack_fmt}, PT=${pt_fmt}"
        rm -f "$output_file"
        return 1
    fi
    
    # ========================================================================
    # Store file in temporary location
    # ========================================================================
    # Files will be moved to organized directory structure after realization completes
    # This allows for batch operations at the end of each realization
    echo "$output_file" >> "${REALIZATION_DIR}/.file_list"
    
    return 0
}

# ============================================================================
# STEP 7: Define simulation parameter sweeps
# ============================================================================
# Number of independent realizations (replicas) to run
NUM_REALIZATIONS=2

# Persistence time values: PT = 1 to 1024 (logarithmic spacing)
# Physical interpretation: expected steps before particle changes direction
PT_VALUES=(1 2 3 4 6 8 12 16 24 32 48 64 96 128 192 256 384 512 768 1024)

# Packing fraction values: 0.016, 0.064, 0.256 (logarithmic spacing)
# Physical interpretation: fraction of lattice sites occupied by particles
PACKING_FRACTION_VALUES=(0.016 0.064 0.256)

# ============================================================================
# STEP 8: Calculate and display total job statistics
# ============================================================================
NUM_PT_VALUES=${#PT_VALUES[@]}
NUM_PACK_VALUES=${#PACKING_FRACTION_VALUES[@]}
TOTAL_ITERATIONS=$((NUM_PT_VALUES * NUM_PACK_VALUES))
TOTAL_SIMULATIONS=$((TOTAL_ITERATIONS * NUM_REALIZATIONS))

echo -e "${MAGENTA}${BOLD}============================================================================${NC}"
echo -e "${MAGENTA}${BOLD}Simulation Configuration Summary${NC}"
echo -e "${MAGENTA}${BOLD}============================================================================${NC}"
echo -e "  ${CYAN}Lattice Type:${NC}           ${BOLD}${LATTICE}${NC}"
echo -e "  ${CYAN}Lattice Size:${NC}           ${BOLD}${LX} × ${LX} = ${N} sites${NC}"
echo -e "  ${CYAN}Healing Time (HT):${NC}       ${BOLD}${HT}${NC} (P_REGEN = 1/HT)"
echo -e "  ${CYAN}PT values:${NC}              ${BOLD}${NUM_PT_VALUES}${NC} values: ${PT_VALUES[@]}"
echo -e "  ${CYAN}Packing fractions:${NC}      ${BOLD}${NUM_PACK_VALUES}${NC} values: ${PACKING_FRACTION_VALUES[@]}"
echo -e "  ${CYAN}Iterations per realization:${NC} ${BOLD}${TOTAL_ITERATIONS}${NC} (PT × Packing)"
echo -e "  ${CYAN}Number of realizations:${NC}  ${BOLD}${NUM_REALIZATIONS}${NC}"
echo -e "  ${CYAN}Total simulations:${NC}       ${BOLD}${TOTAL_SIMULATIONS}${NC}"
echo -e "${MAGENTA}${BOLD}============================================================================${NC}\n"

# ============================================================================
# STEP 9: Main realization loop
# ============================================================================
start_time=$(date +%s)
total_completed=0

for realization in $(seq 1 $NUM_REALIZATIONS); do
    echo -e "${YELLOW}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}${BOLD}REALIZATION ${realization}/${NUM_REALIZATIONS}${NC}"
    echo -e "${YELLOW}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    
    # ========================================================================
    # Create directory structure for this realization
    # ========================================================================
    # Main realization directory
    REALIZATION_DIR="Realization_${realization}"
    mkdir -p "$REALIZATION_DIR" || { 
        echo -e "${RED}[ERROR]${NC} Failed to create realization directory: $REALIZATION_DIR"
        exit 1
    }
    
    # Create temporary file to track generated data files
    > "${REALIZATION_DIR}/.file_list"
    
    # ========================================================================
    # Format HT for directory naming
    # ========================================================================
    ht_fmt=$(printf "%.6f" "$HT")
    
    # Create subdirectory for stopping class data organized by HT
    mkdir -p "$REALIZATION_DIR/STOPPING_CLASS_HT_${ht_fmt}" || { 
        echo -e "${RED}[ERROR]${NC} Failed to create stopping class directory"
        exit 1
    }
    
    # ========================================================================
    # Initialize progress counter
    # ========================================================================
    completed_iterations=0
    realization_start=$(date +%s)
    
    # ========================================================================
    # STEP 10: Parameter sweep loops
    # ========================================================================
    # Nested loops iterate over all parameter combinations:
    # - PT values (persistence time)
    # - Packing fractions
    for pt in "${PT_VALUES[@]}"; do
        for pack in "${PACKING_FRACTION_VALUES[@]}"; do
            
            # ==============================================================
            # Execute simulation
            # ==============================================================
            run_simulation $pack $pt $realization
            
            if [ $? -eq 0 ]; then
                completed_iterations=$((completed_iterations + 1))
                total_completed=$((total_completed + 1))
            else
                echo -e "${RED}[SKIP]${NC} PT=${pt}, PACK=${pack}"
                continue
            fi
            
            # ==============================================================
            # Calculate and display progress
            # ==============================================================
            # Calculate progress as integer percentage
            progress_int=$((completed_iterations * 100 / TOTAL_ITERATIONS))
            overall_progress_int=$((total_completed * 100 / TOTAL_SIMULATIONS))
            
            # Progress bar (visual indicator)
            bar_length=30
            filled=$((progress_int * bar_length / 100))
            empty=$((bar_length - filled))
            bar=$(printf '%*s' $filled | tr ' ' '=')$(printf '%*s' $empty | tr ' ' '-')
            
            # Elapsed time for this realization
            elapsed=$(($(date +%s) - realization_start))
            
            echo -ne "${CYAN}[R${realization}]${NC} ${bar} ${BOLD}${progress_int}%${NC} (${completed_iterations}/${TOTAL_ITERATIONS}) | Overall: ${BOLD}${overall_progress_int}%${NC} | Time: ${elapsed}s\r"
        done
    done
    
    # ========================================================================
    # STEP 11: Organize output files
    # ========================================================================
    # Move all generated data files to organized directory structure
    if [ -f "${REALIZATION_DIR}/.file_list" ] && [ -s "${REALIZATION_DIR}/.file_list" ]; then
        while IFS= read -r datafile; do
            mv "$datafile" "$REALIZATION_DIR/STOPPING_CLASS_HT_${ht_fmt}/" 2>/dev/null
        done < "${REALIZATION_DIR}/.file_list"
        rm -f "${REALIZATION_DIR}/.file_list"
    fi
    
    # ========================================================================
    # STEP 12: Print realization completion summary
    # ========================================================================
    realization_end=$(($(date +%s) - realization_start))
    echo -e "\n"
    echo -e "${GREEN}[SUCCESS]${NC} Realization ${BOLD}${realization}/${NUM_REALIZATIONS}${NC} completed in ${BOLD}${realization_end}s${NC}"
    echo -e "  ${CYAN}Data directory:${NC} ${BOLD}${REALIZATION_DIR}/STOPPING_CLASS_HT_${ht_fmt}/${NC}"
    echo -e "  ${CYAN}Files generated:${NC} ${BOLD}${completed_iterations}${NC} data files\n"
done

# ============================================================================
# STEP 13: Final summary and statistics
# ============================================================================
end_time=$(($(date +%s) - start_time))
minutes=$((end_time / 60))
seconds=$((end_time % 60))

echo -e "${MAGENTA}${BOLD}============================================================================${NC}"
echo -e "${GREEN}${BOLD}✓ ALL SIMULATIONS COMPLETED SUCCESSFULLY${NC}"
echo -e "${MAGENTA}${BOLD}============================================================================${NC}\n"

echo -e "  ${CYAN}Total runtime:${NC}        ${BOLD}${minutes}m ${seconds}s${NC}"
echo -e "  ${CYAN}Total simulations:${NC}    ${BOLD}${total_completed}/${TOTAL_SIMULATIONS}${NC}"
echo -e "  ${CYAN}Success rate:${NC}        ${BOLD}$((total_completed * 100 / TOTAL_SIMULATIONS))%${NC}"
echo -e "  ${CYAN}Output directories:${NC}   $(seq 1 $NUM_REALIZATIONS | tr '\n' ' ' | sed 's/.$//')\n"

echo -e "${BLUE}[INFO]${NC} Results saved in:"
for i in $(seq 1 $NUM_REALIZATIONS); do
    echo -e "  ${BOLD}Realization_${i}/STOPPING_CLASS_HT_${ht_fmt}/${NC}"
done

echo -e "\n${MAGENTA}${BOLD}============================================================================${NC}"
echo -e "${GREEN}${BOLD}Batch simulation finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${MAGENTA}${BOLD}============================================================================${NC}\n"