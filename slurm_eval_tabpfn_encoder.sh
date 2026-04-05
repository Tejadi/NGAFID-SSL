#!/bin/bash
#SBATCH --job-name=tabpfn-encoder
#SBATCH --output=logs/tabpfn_encoder_%j.out
#SBATCH --error=logs/tabpfn_encoder_%j.err
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

pip install tabpfn --upgrade --quiet

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Python: $(python --version)"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TABPFN_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiZGI4ZjJkMDgtNmFjZi00ZjI0LWFkZGItMGQ2N2I2NjBjOGNhIiwiZXhwIjoxODA2ODcyMzE3fQ.YXBqG555IJs0tZcHuGzlXXAO7U02c_CSeaYR6VxE_vI"
export TABPFN_MODEL_CACHE_DIR="$HOME/.cache/tabpfn"

DATA_DIR="./NGAFID-LOCI-GATS-Data"
EVENTS_FILE="$DATA_DIR/events.csv"
OUTPUT_DIR="./tabpfn_results"

# ---- BERT ----
# Update BERT_CKPT to your checkpoint path before running
BERT_CKPT="./bert_results/best_model.pt"
if [ -f "$BERT_CKPT" ]; then
    echo "=========================================="
    echo "Running TabPFN on BERT embeddings: $BERT_CKPT"
    echo "=========================================="
    python eval_tabpfn.py \
        --model_type bert \
        --checkpoint "$BERT_CKPT" \
        --data_dir "$DATA_DIR" \
        --events_file "$EVENTS_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --pca_dim 100 \
        --batch_size 16 \
        --num_workers 4 \
        --no_compile \
        --device auto
    echo "BERT done (exit $?)"
else
    echo "BERT checkpoint not found: $BERT_CKPT — skipping"
fi

# ---- LSTM ----
LSTM_CKPT="./lstm_results/best_model.pt"
if [ -f "$LSTM_CKPT" ]; then
    echo "=========================================="
    echo "Running TabPFN on LSTM embeddings: $LSTM_CKPT"
    echo "=========================================="
    python eval_tabpfn.py \
        --model_type lstm \
        --checkpoint "$LSTM_CKPT" \
        --data_dir "$DATA_DIR" \
        --events_file "$EVENTS_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --pca_dim 100 \
        --batch_size 32 \
        --num_workers 4 \
        --no_compile \
        --device auto
    echo "LSTM done (exit $?)"
else
    echo "LSTM checkpoint not found: $LSTM_CKPT — skipping"
fi

# ---- MLP ----
MLP_CKPT="./mlp_results/best_model.pt"
if [ -f "$MLP_CKPT" ]; then
    echo "=========================================="
    echo "Running TabPFN on MLP embeddings: $MLP_CKPT"
    echo "=========================================="
    python eval_tabpfn.py \
        --model_type mlp \
        --checkpoint "$MLP_CKPT" \
        --data_dir "$DATA_DIR" \
        --events_file "$EVENTS_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --pca_dim 100 \
        --batch_size 32 \
        --num_workers 4 \
        --no_compile \
        --device auto
    echo "MLP done (exit $?)"
else
    echo "MLP checkpoint not found: $MLP_CKPT — skipping"
fi

echo "=========================================="
echo "All done at: $(date)"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null
echo "=========================================="
