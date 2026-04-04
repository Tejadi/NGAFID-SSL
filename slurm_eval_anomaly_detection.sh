#!/bin/bash
#SBATCH --job-name=anomaly-eval-patchtst
#SBATCH --output=logs/anomaly_patchtst_%j.out
#SBATCH --error=logs/anomaly_patchtst_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_rtx_a5000:1

mkdir -p logs

source ~/.bashrc
conda activate torch-gpu-12

echo "Python: $(python --version)"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"

DATA_DIR="./NGAFID-LOCI-GATS-Data/preprocessed_data"
TEST_DIR="$DATA_DIR/test"
TRAIN_DIR="$DATA_DIR/train"
EVENTS_FILE="$TEST_DIR/events.csv"
CHECKPOINT="patchtst_results/patchtst_training_20260328_174942/best_model.pt"
OUTPUT_DIR="./anomaly_detection_results"

python eval_anomaly_detection_patchtst.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$TEST_DIR" \
    --events_file "$EVENTS_FILE" \
    --train_data_dir "$TRAIN_DIR" \
    --mask_ratio 0.15 \
    --num_mask_samples 5 \
    --threshold_percentile 95 \
    --topk_percents 1.0 5.0 10.0 \
    --output_dir "$OUTPUT_DIR" \
    --run_name "patchtst_anomaly"

echo "Finished: $(date)"
