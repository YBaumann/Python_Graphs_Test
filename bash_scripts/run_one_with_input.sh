#!/bin/bash
#SBATCH --job-name=${1}                          # Use dataset name as the job name
#SBATCH --output=slurm_output_%x_%j.log          # Output log file (job name and job ID)
#SBATCH --error=slurm_error_%x_%j.log            # Error log file (job name and job ID)
#SBATCH --time=96:00:00                          # Time limit hrs:min:sec
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --cpus-per-task=1                        # Number of CPU cores per task
#SBATCH --mem-per-cpu=8GB                       # Memory per node

# Load necessary modules (if any, like Python)
# module load python/3.x

# Take dataset name and additional parameters as arguments
DATASET_NAME=$1
NR_DIVIDERS=$2
NR_REPEATS=$3
NR_SPARSIFIER=$4
NR_EPOCHS=$5
TREE_FUNCTION_NAMES=$6  # Comma-separated string of tree function names
SAMPLER_NAMES=$7        # Comma-separated string of sampler names
ONE_OR_K=$8             # Comma-separated string of "one or k" values

# Define a timestamp-based directory under runs/
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="runs/run_$TIMESTAMP"
mkdir -p "$RUN_DIR/output_csv"

# Run the main program, passing in all arguments
yes y | python3 main.py "$DATASET_NAME" "$NR_DIVIDERS" "$NR_REPEATS" "$NR_SPARSIFIER" "$NR_EPOCHS" "$TREE_FUNCTION_NAMES" "$SAMPLER_NAMES" "$ONE_OR_K" > "$RUN_DIR/output.log" 2> "$RUN_DIR/error.log"
