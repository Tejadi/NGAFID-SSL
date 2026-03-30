#!/bin/bash
#SBATCH --job-name=patchtst-eval
#SBATCH --output=logs/patchtst_eval_%j.out
#SBATCH --error=logs/patchtst_eval_%j.err
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

echo "Python: $(python --version)"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"

python eval_patchtst.py \
    --checkpoint patchtst_results/patchtst_training_20260328_174942/best_model.pt \
    --local_data_dir ./NGAFID-LOCI-GATS-Data \
    --seq_len 1024 \
    --batch_size 16

echo "Finished: $(date)"
