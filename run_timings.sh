#!/bin/bash
#SBATCH --job-name=timings                 # Job name
#SBATCH --output=slurm_output_%x_%j.log  # Output log file (job name and job ID)
#SBATCH --error=slurm_error_%x_%j.log    # Error log file (job name and job ID)
#SBATCH --time=20:00:00                  # Time limit hrs:min:sec
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem-per-cpu=64GB                # Memory per node

# Load necessary modules (e.g., Python)
# module load python/3.x  # Uncomment if needed, depending on your environment

# Run the Python script
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="timings/run_$TIMESTAMP"
mkdir -p "$RUN_DIR/timing_csv"

# Run the Python script, passing the directory as an argument
yes y | python3 run_timings.py "$RUN_DIR" > "$RUN_DIR/output.log" 2> "$RUN_DIR/error.log"