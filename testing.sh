#!/bin/bash
#SBATCH --job-name=graph_stats                   # Job name
#SBATCH --output=slurm_output_%x_%j.log  # Output log file (job name and job ID)
#SBATCH --error=slurm_error_%x_%j.log    # Error log file (job name and job ID)
#SBATCH --time=04:00:00                  # Time limit hrs:min:sec
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem-per-cpu=4GB                # Memory per node

# Load necessary modules (if any, like Python)
# module load python/3.x



# Run your program with automatic 'y' responses, storing outputs in the unique run directory
yes y | python3 testing.py
