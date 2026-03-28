#!/bin/bash
#SBATCH --job-name=patchtst-flight-training      # Job name
#SBATCH --output=logs/patchtst_training_%j.out   # Output file (%j = job ID)
#SBATCH --error=logs/patchtst_training_%j.err    # Error file
#SBATCH --time=24:00:00                          # Time limit (24 hours)
#SBATCH --partition=gpu                          # Partition (adjust for your cluster)
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --cpus-per-task=8                        # Number of CPU cores
#SBATCH --mem=64G                                # Memory per node
#SBATCH --gres=gpu:nvidia_rtx_a5000:1            # Request 1 A5000 GPU (24GB VRAM)

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

# Print GPU information
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
    echo "=========================================="
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Load any required modules (adjust for your cluster)
# module load python/3.10
# module load cuda/11.8
# module load anaconda3

# Activate conda environment
source ~/.bashrc
conda activate bert-flight

pip install wandb --quiet

# Verify environment
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Current directory: $(pwd)"

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training configuration
JOB_NAME="patchtst_training_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="./NGAFID-LOCI-GATS-Data"
OUTPUT_DIR="./patchtst_results"

echo "Starting PatchTST training..."
echo "Job name: $JOB_NAME"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

python train_patchtst_masked_regressor.py \
    --local_data_dir "$DATA_DIR" \
    --job_name "$JOB_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --seq_len 1024 \
    --patch_len 16 \
    --stride 8 \
    --batch_size 32 \
    --epochs 18 \
    --learning_rate 1e-4 \
    --d_model 512 \
    --n_heads 8 \
    --d_ff 2048 \
    --encoder_layers 6 \
    --decoder_layers 3 \
    --dropout 0.1 \
    --warmup_steps 2000 \
    --weight_decay 1e-5 \
    --eval_interval 500 \
    --save_interval 2000 \
    --num_workers 4 \
    --device auto \
    --use_random_masking \
    --masking_ratios 0.2 0.5 0.8 \
    --mean_mask_lengths 5 60 \
    --wandb_project "patchtst-flight-ssl" \
    --wandb_run_name "$JOB_NAME"

EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "Job finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Output files:"
    ls -la "$OUTPUT_DIR/$JOB_NAME/"
    echo "Model file sizes:"
    du -h "$OUTPUT_DIR/$JOB_NAME/"*.pt 2>/dev/null || echo "No model files found"
else
    echo "Training failed with exit code: $EXIT_CODE"
    echo "Check the error log for details"
fi

echo "=========================================="
exit $EXIT_CODE
