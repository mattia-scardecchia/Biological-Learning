#!/bin/bash
# Usage: bash submit_grid_search.sh --seeds 1,2,3,4,5

# Parse command-line argument for seeds
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seeds) seeds_arg="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

if [ -z "$seeds_arg" ]; then
    echo "Usage: $0 --seeds seed1,seed2,..."
    exit 1
fi

# Convert the comma-separated seed list into an array
IFS=',' read -ra seeds_array <<< "$seeds_arg"

# Define the list of cnodes to cycle through
cnodes=(cnode01 cnode02 cnode03 cnode04)
num_nodes=${#cnodes[@]}

# Loop over each seed and submit a job with the corresponding cnode
for i in "${!seeds_array[@]}"; do
    seed="${seeds_array[$i]}"
    # Cycle through cnodes using modulo arithmetic
    cnode=${cnodes[$(( i % num_nodes ))]}
    
    echo "Sleeping, then submitting job with seed=${seed} on node=${cnode}"
    sleep 5
    sbatch --export=ALL,seed=$seed --nodelist=$cnode scripts/slurm/run.sh
done
