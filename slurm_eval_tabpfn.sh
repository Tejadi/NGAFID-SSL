#!/bin/bash
#SBATCH --job-name=tabpfn-eval
#SBATCH --output=logs/tabpfn_eval_%j.out
#SBATCH --error=logs/tabpfn_eval_%j.err
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

pip install tabpfn --quiet

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Python: $(python --version)"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="./NGAFID-LOCI-GATS-Data"
EVENTS_FILE="$DATA_DIR/preprocessed_data/test/events.csv"
OUTPUT_DIR="./tabpfn_results"

# ============================================================
# Option A: TabPFN on frozen encoder embeddings
# Run one per encoder type. Adjust checkpoint paths as needed.
# ============================================================

# --- BERT embeddings ---
BERT_CKPT="bert_results/best_model.pt"  # <-- adjust path
if [ -f "$BERT_CKPT" ]; then
    echo "=========================================="
    echo "Running TabPFN on BERT embeddings..."
    echo "=========================================="
    python eval_tabpfn.py \
        --model_type bert \
        --checkpoint "$BERT_CKPT" \
        --data_dir "$DATA_DIR" \
        --events_file "$EVENTS_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --pca_dim 100 \
        --n_ensemble_configurations 32 \
        --batch_size 16 \
        --num_workers 4 \
        --device auto \
        --no_compile
    echo "BERT done (exit $?)"
fi

# --- LSTM embeddings ---
LSTM_CKPT="lstm_results/best_model.pt"  # <-- adjust path
if [ -f "$LSTM_CKPT" ]; then
    echo "=========================================="
    echo "Running TabPFN on LSTM embeddings..."
    echo "=========================================="
    python eval_tabpfn.py \
        --model_type lstm \
        --checkpoint "$LSTM_CKPT" \
        --data_dir "$DATA_DIR" \
        --events_file "$EVENTS_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --pca_dim 100 \
        --n_ensemble_configurations 32 \
        --batch_size 32 \
        --num_workers 4 \
        --device auto \
        --no_compile
    echo "LSTM done (exit $?)"
fi

# --- MLP embeddings ---
MLP_CKPT="mlp_results/best_model.pt"  # <-- adjust path
if [ -f "$MLP_CKPT" ]; then
    echo "=========================================="
    echo "Running TabPFN on MLP embeddings..."
    echo "=========================================="
    python eval_tabpfn.py \
        --model_type mlp \
        --checkpoint "$MLP_CKPT" \
        --data_dir "$DATA_DIR" \
        --events_file "$EVENTS_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --pca_dim 100 \
        --n_ensemble_configurations 32 \
        --batch_size 32 \
        --num_workers 4 \
        --device auto \
        --no_compile
    echo "MLP done (exit $?)"
fi

# ============================================================
# Option B: TabPFN on raw handcrafted summary statistics
# (no encoder needed — pure tabular baseline)
# ============================================================
echo "=========================================="
echo "Running TabPFN on raw summary-stat features..."
echo "=========================================="
python eval_tabpfn.py \
    --raw_features \
    --data_dir "$DATA_DIR" \
    --events_file "$EVENTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --pca_dim 0 \
    --n_ensemble_configurations 32 \
    --device auto

echo "Raw features done (exit $?)"

echo "=========================================="
echo "All TabPFN evaluations finished at: $(date)"
echo "Results in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null
echo "=========================================="
