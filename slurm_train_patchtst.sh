#!/bin/bash
#SBATCH --job-name=patchtst-flight-training
#SBATCH --output=logs/patchtst_training_%j.out
#SBATCH --error=logs/patchtst_training_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
    echo "=========================================="
fi

mkdir -p logs

source ~/.bashrc
conda activate bert-flight

pip install wandb --quiet

echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Current directory: $(pwd)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

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
    --batch_size 16 \
    --epochs 18 \
    --learning_rate 8e-5 \
    --patch_len 16 \
    --stride 8 \
    --d_model 512 \
    --n_heads 8 \
    --d_ff 2048 \
    --encoder_layers 6 \
    --decoder_layers 3 \
    --eval_interval 500 \
    --save_interval 2000 \
    --num_workers 4 \
    --device auto \
    --max_files_train 1000 \
    --max_files_val 200 \
    --warmup_steps 2000 \
    --weight_decay 1e-5 \
    --wandb_project "patchtst-flight-ssl" \
    --wandb_run_name "$JOB_NAME"

EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "Job finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"

    echo "Output files created:"
    ls -la "$OUTPUT_DIR/$JOB_NAME/"

    echo "Model file sizes:"
    du -h "$OUTPUT_DIR/$JOB_NAME/"*.pt 2>/dev/null || echo "No model files found"
else
    echo "Training failed with exit code: $EXIT_CODE"
    echo "Check the error log for details"
fi

echo "=========================================="

exit $EXIT_CODE
