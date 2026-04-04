#!/bin/bash
#SBATCH --job-name=tabpfn-raw
#SBATCH --output=logs/tabpfn_raw_%j.out
#SBATCH --error=logs/tabpfn_raw_%j.err
#SBATCH --time=02:00:00
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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="./NGAFID-LOCI-GATS-Data"
EVENTS_FILE="$DATA_DIR/events.csv"
OUTPUT_DIR="./tabpfn_results"

python eval_tabpfn.py \
    --raw_features \
    --data_dir "$DATA_DIR" \
    --events_file "$EVENTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --pca_dim 0 \
    --n_estimators 4 \
    --device auto

echo "Done (exit $?). Results in: $OUTPUT_DIR"
