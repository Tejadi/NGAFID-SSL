#!/bin/bash
#SBATCH --job-name=patchtst-forecaster
#SBATCH --output=logs/patchtst_forecaster_%j.out
#SBATCH --error=logs/patchtst_forecaster_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_rtx_a5000:1

mkdir -p logs

source ~/.bashrc
conda activate torch-gpu-12

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

JOB_NAME="patchtst_forecaster_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="./NGAFID-LOCI-GATS-Data"
OUTPUT_DIR="./patchtst_forecaster_runs"

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Node: $SLURM_NODELIST"
nvidia-smi
echo "=========================================="

python train_patchtst_forecaster.py \
    --data_dir "$DATA_DIR" \
    --job_name "$JOB_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --seq_len 1024 \
    --patch_len 16 \
    --stride 8 \
    --batch_size 16 \
    --epochs 18 \
    --learning_rate 1e-4 \
    --d_model 512 \
    --n_heads 8 \
    --d_ff 2048 \
    --encoder_layers 6 \
    --decoder_layers 3 \
    --dropout 0.1 \
    --warmup_steps 1000 \
    --weight_decay 1e-5 \
    --eval_interval 500 \
    --save_interval 2000 \
    --num_workers 4 \
    --forecast_ratios 0.1 0.2 0.3 \
    --min_forecast_horizon 50 \
    --wandb_project "patchtst-flight-forecaster" \
    --wandb_run_name "$JOB_NAME"

EXIT_CODE=$?
echo "=========================================="
echo "Finished at: $(date) with exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
