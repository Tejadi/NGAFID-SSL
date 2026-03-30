#!/bin/bash
#SBATCH --job-name=patchtst-maskedrec-physics
#SBATCH --output=logs/patchtst_maskedrec_physics_%j.out
#SBATCH --error=logs/patchtst_maskedrec_physics_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_rtx_a5000:1

mkdir -p logs

source ~/.bashrc
conda activate torch-gpu-12

echo "Started: $(date)"
echo "Node: $SLURM_NODELIST"
nvidia-smi

python eval_forecasting.py \
    --model_type patchtst \
    --checkpoint patchtst_results/patchtst_training_20260328_174942/best_model.pt \
    --data_dir ./NGAFID-LOCI-GATS-Data/preprocessed_data/test \
    --train_data_dir ./NGAFID-LOCI-GATS-Data/preprocessed_data/train \
    --physics_eval \
    --aircraft_preset cessna172s \
    --feature_map_preset ngafid_44col \
    --max_physics_samples 200 \
    --physics_workers 8

echo "Finished: $(date)"
