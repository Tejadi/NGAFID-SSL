#!/bin/bash
#SBATCH --job-name=bert-flight-training      # Job name
#SBATCH --output=logs/bert_training_%j.out   # Output file (%j = job ID)
#SBATCH --error=logs/bert_training_%j.err    # Error file
#SBATCH --time=24:00:00                      # Time limit (24 hours)
#SBATCH --partition=gpu                      # Partition (adjust for your cluster)
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=8                    # Number of CPU cores
#SBATCH --mem=64G                            # Memory per node
#SBATCH --gres=gpu:1                         # Number of GPUs (adjust as needed)

# Optional: Email notifications (uncomment and add your email)
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your.email@university.edu

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# Print GPU information if available
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

# Activate conda environment
# Option 1: If conda is in your PATH
source ~/.bashrc  # Ensure conda is initialized
conda activate bert-flight

# Install wandb if not already installed
pip install wandb --quiet

# Option 2: If using specific conda path (uncomment if needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate bert-flight

# Option 3: Direct python path (uncomment if conda doesn't work)
# export PATH="~/miniconda3/envs/bert-flight/bin:$PATH"

# Verify environment
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Current directory: $(pwd)"

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Training configuration
JOB_NAME="full_bert_training_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="./NGAFID-LOCI-GATS-Data"
OUTPUT_DIR="./bert_results"

echo "Starting BERT training..."
echo "Job name: $JOB_NAME"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Run the training with full dataset and longer sequences
python train_bert_masked_regressor.py \
    --local_data_dir "$DATA_DIR" \
    --job_name "$JOB_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --seq_len 1024 \
    --batch_size 16 \
    --epochs 18 \
    --learning_rate 8e-5 \
    --hidden_size 1536 \
    --encoder_layers 12 \
    --decoder_layers 8 \
    --num_heads 16 \
    --eval_interval 500 \
    --save_interval 2000 \
    --num_workers 4 \
    --device auto \
    --max_files_train 1000 \
    --max_files_val 200 \
    --warmup_steps 2000 \
    --weight_decay 1e-5 \
    --wandb_project "bert-flight-ssl" \
    --wandb_run_name "$JOB_NAME"

# Capture exit code
EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "Job finished at: $(date)"

# Print some final statistics
if [ $EXIT_CODE -eq 0 ]; then
    echo " Training completed successfully!"

    # Show output directory contents
    echo "Output files created:"
    ls -la "$OUTPUT_DIR/$JOB_NAME/"

    # Show model file sizes
    echo "Model file sizes:"
    du -h "$OUTPUT_DIR/$JOB_NAME/"*.pt 2>/dev/null || echo "No model files found"
else
    echo " Training failed with exit code: $EXIT_CODE"
    echo "Check the error log for details"
fi

echo "=========================================="

# Optional: Clean up temporary files or send notifications
# You can add cleanup commands here if needed

exit $EXIT_CODE