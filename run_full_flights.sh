#!/bin/bash
#SBATCH --job-name=bert-full-flights
#SBATCH --output=logs/bert_full_flights_%j.out
#SBATCH --error=logs/bert_full_flights_%j.err
#SBATCH --time=30:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Optional: Email notifications (uncomment and add your email)
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your.email@brown.edu

echo "=========================================="
echo "BERT Full Flights Training Job"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# Print GPU information
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
    echo "=========================================="
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Load any required modules (adjust for your cluster)
# Example module loads (uncomment and modify as needed):
# module load python/3.10
# module load cuda/11.8
# module load anaconda3

# Activate conda environment (adjust path as needed)
# Option 1: If conda is in your PATH
source ~/.bashrc
conda activate bert-flight

# Option 2: If using specific conda path (uncomment if needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate bert-flight

# Verify environment
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Current directory: $(pwd)"

# Install wandb if not already installed
pip install wandb --quiet

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Starting BERT Full Flights training..."
echo "Data directory: /oscar/data/sbach/shared/ngafid"
echo "Sequence length: 10,000"
echo "Expected runtime: 20-24 hours"
echo "=========================================="

# Run the training script
python train_full_flights.py

# Capture exit code
EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "Job finished at: $(date)"

# Print some final statistics
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"

    # Show output directory contents
    if [ -d "./results" ]; then
        echo "Output files created:"
        ls -la ./results/*/

        # Show model file sizes
        echo "Model file sizes:"
        find ./results -name "*.pt" -exec du -h {} \; 2>/dev/null || echo "No model files found"
    fi
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "Check the error log for details"
fi

echo "=========================================="

exit $EXIT_CODE
