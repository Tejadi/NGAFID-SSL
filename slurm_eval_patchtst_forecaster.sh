#!/bin/bash
#SBATCH --job-name=patchtst-forecast-eval
#SBATCH --output=logs/patchtst_forecast_eval_%j.out
#SBATCH --error=logs/patchtst_forecast_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_rtx_a5000:1

mkdir -p logs

source ~/.bashrc
conda activate torch-gpu-12

echo "Started: $(date)"
echo "Node: $SLURM_NODELIST"

python eval_patchtst_forecaster.py \
    --checkpoint patchtst_forecaster_runs/REPLACE_WITH_RUN_NAME/best_model.pt \
    --data_dir ./NGAFID-LOCI-GATS-Data \
    --seq_len 1024 \
    --batch_size 16

echo "Finished: $(date)"
