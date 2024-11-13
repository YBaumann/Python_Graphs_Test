#!/bin/bash
#SBATCH --job-name=graph_stats_job   # Job name
#SBATCH --output=graph_stats_output.log  # Output log file
#SBATCH --error=graph_stats_error.log   # Error log file
#SBATCH --ntasks=1                      # Number of tasks (1 task for this job)
#SBATCH --cpus-per-task=1               # Number of CPUs per task (adjust as needed)
#SBATCH --mem-per-cpu=4G                # Memory per CPU (8GB)
#SBATCH --time=48:00:00                 # Time limit (adjust as needed)

# Load any necessary modules, e.g., if you need Python
module load python/3.8  # Adjust Python version as needed

# Run the Python script
python3 graph_stats.py
