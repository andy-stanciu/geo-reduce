#!/bin/bash
#SBATCH --job-name=train-clip             # Job name

#SBATCH --account=andys22                 # Replace with your netid
#SBATCH --partition=gpu                   # Partition name (default is gpu)
#SBATCH --nodes=1                         # One node (tomago/tempura)
#SBATCH --ntasks-per-node=1               # Number of CPU cores (vary it as needed)
#SBATCH --mem=20G                         # Memory per node
#SBATCH --gpus=8                          # Number of GPUs (for HW1, let it be 1)
#SBATCH --time=02:00:00                   # Max runtime = 2 hours (cluster limit)

#SBATCH --chdir=/homes/iws/andys22/599o/geo-reduce # Working directory
#SBATCH --export=ALL                      # Export environment variables
#SBATCH --output=andys22-slurm-%j.out             # STDOUT log file (%j = JobID)
#SBATCH --error=andys22-slurm-%j.err              # STDERR log file (%j = JobID)

# --- Example Command to run ---
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
srun --gpus-per-node=8 python src/train/train.py