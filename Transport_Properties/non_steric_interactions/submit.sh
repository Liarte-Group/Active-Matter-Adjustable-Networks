#!/bin/bash

################################################################################
# Batch Simulation Script: Active Brownian Particles on Adjustable Networks
################################################################################
#
# This script orchestrates multiple simulations with varying parameters.
# 
# Workflow:
#   1. Setup CUDA environment
#   2. Validate command-line arguments
#   3. Parse parameters (LATTICE_TYPE, LX, HT, PT, PACKING_FRACTION)
#   4. For each realization:
#      - Create output directory
#      - Calculate derived parameters (P_PERST, P_REGEN)
#      - Compile and execute simulation
#      - Save results to file
#   5. Report completion
#
# Author: William G. C. Oropesa
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
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

echo -e "${MAGENTA}${BOLD}============================================================================${NC}"
echo -e "${MAGENTA}${BOLD}Active Brownian Particles on Adjustable Networks - Single Simulation${NC}"
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
# Format: Unix timestamp (seconds since 1970-01-01)
timestamp=$(date +%s)
echo -e "${CYAN}[TIMESTAMP]${NC} Simulation batch ID: ${BOLD}${timestamp}${NC}\n"

# ============================================================================
# STEP 4: Parse and validate command-line arguments
# ============================================================================
echo -e "${BLUE}[INFO]${NC} Parsing command-line arguments...\n"

# ========================================================================
# Parameter 1: Lattice type (default: TRIANGULAR)
# ========================================================================
LATTICE=${1:-"TRIANGULAR"}
if [[ "$LATTICE" != "SQUARE_MOORE" && "$LATTICE" != "TRIANGULAR" ]]; then
    echo -e "${RED}[ERROR]${NC} Invalid lattice type: $LATTICE"
    echo -e "${YELLOW}[USAGE]${NC} ./submit.sh [LATTICE_TYPE] [LX] [HT] [PT] [PACKING_FRACTION]"
    echo -e "${YELLOW}[EXAMPLE]${NC} ./submit.sh TRIANGULAR 64 20.0 256 0.128"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Lattice type: ${BOLD}${LATTICE}${NC}"

# ========================================================================
# Parameter 2: Linear lattice size (default: 64)
# ========================================================================
LX=${2:-64}
if ! [[ "$LX" =~ ^[0-9]+$ ]] || [ "$LX" -lt 16 ]; then
    echo -e "${RED}[ERROR]${NC} Invalid LX value: $LX (must be integer >= 16)"
    echo -e "${YELLOW}[USAGE]${NC} ./submit.sh [LATTICE_TYPE] [LX] [HT] [PT] [PACKING_FRACTION]"
    exit 1
fi
N=$((LX * LX))
echo -e "${GREEN}[OK]${NC} Lattice size: ${BOLD}${LX} × ${LX} = ${N} sites${NC}"

# ========================================================================
# Parameter 3: Healing time (default: 20.0)
# ========================================================================
HT=${3:-20.0}
if ! [[ "$HT" =~ ^[0-9]+\.?[0-9]*$ ]] || (( $(echo "$HT <= 0" | bc -l) )); then
    echo -e "${RED}[ERROR]${NC} Invalid HT value: $HT (must be positive number)"
    echo -e "${YELLOW}[USAGE]${NC} ./submit.sh [LATTICE_TYPE] [LX] [HT] [PT] [PACKING_FRACTION]"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Healing time: ${BOLD}${HT}${NC}"

# ========================================================================
# Parameter 4: Persistence time (default: 256)
# ========================================================================
PT=${4:-256}
if ! [[ "$PT" =~ ^[0-9]+\.?[0-9]*$ ]] || (( $(echo "$PT <= 0" | bc -l) )); then
    echo -e "${RED}[ERROR]${NC} Invalid PT value: $PT (must be positive number)"
    echo -e "${YELLOW}[USAGE]${NC} ./submit.sh [LATTICE_TYPE] [LX] [HT] [PT] [PACKING_FRACTION]"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Persistence time: ${BOLD}${PT}${NC}"

# ========================================================================
# Parameter 5: Packing fraction (default: 0.128)
# ========================================================================
PACKING_FRACTION=${5:-0.128}
if ! [[ "$PACKING_FRACTION" =~ ^[0-9]+\.?[0-9]*$ ]] || (( $(echo "$PACKING_FRACTION <= 0" | bc -l) )) || (( $(echo "$PACKING_FRACTION > 1" | bc -l) )); then
    echo -e "${RED}[ERROR]${NC} Invalid PACKING_FRACTION value: $PACKING_FRACTION (must be between 0 and 1)"
    echo -e "${YELLOW}[USAGE]${NC} ./submit.sh [LATTICE_TYPE] [LX] [HT] [PT] [PACKING_FRACTION]"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Packing fraction: ${BOLD}${PACKING_FRACTION}${NC}\n"

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
#   $2 (PT) - Persistence time (controls particle direction persistence)
#   $3 (HT) - Healing time (controls bond regeneration)
#   $4 (realization) - Replication number (for ensemble averaging)
#
# Returns:
#   0 on success, 1 on compilation or execution failure
run_simulation() {
    local PACKING_FRACTION=$1
    local PT=$2
    local HT=$3
    local realization=$4
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
    #   msd_TIMESTAMP_L_SIZE_PACK_FRACTION_PT_TIME_HT_TIME_R_REALIZATION.dat
    local output_file="msd_${timestamp}_L_${LX}_PACK_${pack_fmt}_PT_${pt_fmt}_HT_${ht_fmt}_R_${realization}.dat"
    
    ./a.out > "$output_file" 2>&1
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${NC} Simulation failed for PACK=${pack_fmt}, PT=${pt_fmt}"
        rm -f "$output_file"
        return 1
    fi
    
    # Move data file to realization directory immediately
    # This ensures files don't accumulate in the root directory
    mv "$output_file" "$REALIZATION_DIR/MSD_HT_${ht_fmt}/" 2>/dev/null
    
    return 0
}

# ============================================================================
# STEP 7: Define simulation parameters (single values)
# ============================================================================
NUM_REALIZATIONS=2

# ============================================================================
# STEP 8: Calculate and display job summary
# ============================================================================
echo -e "${MAGENTA}${BOLD}============================================================================${NC}"
echo -e "${MAGENTA}${BOLD}Simulation Configuration Summary${NC}"
echo -e "${MAGENTA}${BOLD}============================================================================${NC}"
echo -e "  ${CYAN}Lattice Type:${NC}           ${BOLD}${LATTICE}${NC}"
echo -e "  ${CYAN}Lattice Size:${NC}           ${BOLD}${LX} × ${LX} = ${N} sites${NC}"
echo -e "  ${CYAN}Healing Time (HT):${NC}       ${BOLD}${HT}${NC} (P_REGEN = 1/${HT})"
echo -e "  ${CYAN}Persistence Time (PT):${NC}   ${BOLD}${PT}${NC} (P_PERST = 1 - 1/${PT})"
echo -e "  ${CYAN}Packing Fraction:${NC}       ${BOLD}${PACKING_FRACTION}${NC}"
echo -e "  ${CYAN}Number of Realizations:${NC}  ${BOLD}${NUM_REALIZATIONS}${NC}"
echo -e "  ${CYAN}Total Simulations:${NC}       ${BOLD}${NUM_REALIZATIONS}${NC}"
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
    REALIZATION_DIR="Realization_${realization}"
    mkdir -p "$REALIZATION_DIR" || { 
        echo -e "${RED}[ERROR]${NC} Failed to create realization directory: $REALIZATION_DIR"
        exit 1
    }
    
    ht_fmt=$(printf "%.6f" "$HT")
    mkdir -p "$REALIZATION_DIR/MSD_HT_${ht_fmt}" || { 
        echo -e "${RED}[ERROR]${NC} Failed to create data directory"
        exit 1
    }
    
    # ========================================================================
    # Initialize progress counter
    # ========================================================================
    realization_start=$(date +%s)
    
    # ========================================================================
    # Execute simulation
    # ========================================================================
    run_simulation $PACKING_FRACTION $PT $HT $realization
    
    if [ $? -eq 0 ]; then
        total_completed=$((total_completed + 1))
    else
        echo -e "${RED}[SKIP]${NC} HT=${HT}, PT=${PT}, PACK=${PACKING_FRACTION}"
        continue
    fi
    
    # ========================================================================
    # Print realization completion summary
    # ========================================================================
    realization_end=$(($(date +%s) - realization_start))
    echo -e "\n"
    echo -e "${GREEN}[SUCCESS]${NC} Realization ${BOLD}${realization}/${NUM_REALIZATIONS}${NC} completed in ${BOLD}${realization_end}s${NC}"
    pack_fmt=$(printf "%.6f" "$PACKING_FRACTION")
    pt_fmt=$(printf "%.6f" "$PT")
    echo -e "  ${CYAN}Data directory:${NC} ${BOLD}${REALIZATION_DIR}/MSD_HT_${ht_fmt}/${NC}"
    echo -e "  ${CYAN}Output file:${NC} ${BOLD}msd_${timestamp}_L_${LX}_PACK_${pack_fmt}_PT_${pt_fmt}_HT_${ht_fmt}_R_${realization}.dat${NC}\n"
done

# ============================================================================
# STEP 10: Final summary and statistics
# ============================================================================
end_time=$(($(date +%s) - start_time))
minutes=$((end_time / 60))
seconds=$((end_time % 60))

echo -e "${MAGENTA}${BOLD}============================================================================${NC}"
echo -e "${GREEN}${BOLD}✓ ALL SIMULATIONS COMPLETED SUCCESSFULLY${NC}"
echo -e "${MAGENTA}${BOLD}============================================================================${NC}\n"

echo -e "  ${CYAN}Total runtime:${NC}        ${BOLD}${minutes}m ${seconds}s${NC}"
echo -e "  ${CYAN}Total simulations:${NC}    ${BOLD}${total_completed}/${NUM_REALIZATIONS}${NC}"
echo -e "  ${CYAN}Success rate:${NC}        ${BOLD}$((total_completed * 100 / NUM_REALIZATIONS))%${NC}\n"

echo -e "${BLUE}[INFO]${NC} Results saved in:"
for i in $(seq 1 $NUM_REALIZATIONS); do
    echo -e "  ${BOLD}Realization_${i}/MSD_HT_${ht_fmt}/${NC}"
done

echo -e "\n${MAGENTA}${BOLD}============================================================================${NC}"
echo -e "${GREEN}${BOLD}Batch simulation finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${MAGENTA}${BOLD}============================================================================${NC}\n"