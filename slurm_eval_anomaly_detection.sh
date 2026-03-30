#!/bin/bash
#SBATCH --job-name=anomaly-eval
#SBATCH --output=logs/anomaly_eval_%j.out
#SBATCH --error=logs/anomaly_eval_%j.err
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

DATA_DIR="./NGAFID-LOCI-GATS-Data"
TEST_DIR="$DATA_DIR/test"
TRAIN_DIR="$DATA_DIR/train"
EVENTS_FILE="$DATA_DIR/test/events.csv"
OUTPUT_DIR="./anomaly_detection_results"

# --- BERT ---
python eval_anomaly_detection.py \
    --model_type bert \
    --checkpoint bert_results/best_model.pt \
    --data_dir "$TEST_DIR" \
    --events_file "$EVENTS_FILE" \
    --train_data_dir "$TRAIN_DIR" \
    --mask_ratio 0.15 \
    --num_mask_samples 5 \
    --threshold_percentile 95 \
    --topk_percents 1.0 5.0 10.0 \
    --batch_size 16 \
    --output_dir "$OUTPUT_DIR" \
    --run_name "bert_anomaly"

# --- LSTM ---
python eval_anomaly_detection.py \
    --model_type lstm \
    --checkpoint lstm_results/best_model.pt \
    --data_dir "$TEST_DIR" \
    --events_file "$EVENTS_FILE" \
    --train_data_dir "$TRAIN_DIR" \
    --mask_ratio 0.15 \
    --num_mask_samples 5 \
    --threshold_percentile 95 \
    --topk_percents 1.0 5.0 10.0 \
    --batch_size 32 \
    --output_dir "$OUTPUT_DIR" \
    --run_name "lstm_anomaly"

# --- MLP ---
python eval_anomaly_detection.py \
    --model_type mlp \
    --checkpoint mlp_results/best_model.pt \
    --data_dir "$TEST_DIR" \
    --events_file "$EVENTS_FILE" \
    --train_data_dir "$TRAIN_DIR" \
    --mask_ratio 0.15 \
    --num_mask_samples 5 \
    --threshold_percentile 95 \
    --topk_percents 1.0 5.0 10.0 \
    --batch_size 32 \
    --output_dir "$OUTPUT_DIR" \
    --run_name "mlp_anomaly"

echo "Finished: $(date)"
