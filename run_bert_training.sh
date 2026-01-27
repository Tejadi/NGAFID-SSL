#!/bin/bash

#SBATCH -J bert_full_flights              # Job name
#SBATCH -n 1                              # Number of tasks
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --mem=32G                         # Request 32GB memory
#SBATCH -t 48:00:00                       # 24 hour time limit
#SBATCH -o logs/bert_training_%j.out      # Output file (%j = job ID)
#SBATCH -e logs/bert_training_%j.err      # Error file
#SBATCH --partition=gpu                   # GPU partition

# Create logs directory if it doesn't exist
mkdir -p logs


# Print job information
echo "================================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "================================================================"

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo "================================================================"

# Load required modules (adjust based on your environment)
echo "Loading modules..."
module load python/3.9.0
module load cuda/11.8
module load gcc/8.3

# Activate conda environment if you have one
# Uncomment and modify the line below if you use conda
# source activate your_environment_name

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

# Print Python and CUDA versions
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version)"
echo "================================================================"

# Change to project directory
cd /oscar/home/cduong5/NGAFID-SSL

# Run the training script
echo "Starting BERT training..."
python train_full_flights.py

# Print completion information
echo "================================================================"
echo "Job completed at: $(date)"
echo "================================================================"