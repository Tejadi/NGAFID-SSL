#!/bin/bash
#SBATCH --job-name=patchtst-flight-test
#SBATCH --output=logs/patchtst_test_%j.out
#SBATCH --error=logs/patchtst_test_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Current directory: $(pwd)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MODEL_PATH="${1:-./patchtst_results/best_model.pt}"
TEST_DATA_DIR="${2:-./NGAFID-LOCI-GATS-Data/preprocessed_data/test}"

echo "Starting PatchTST testing..."
echo "Model path: $MODEL_PATH"
echo "Test data directory: $TEST_DATA_DIR"
echo "=========================================="

python test_patchtst_masked_regressor.py \
    --data_dir "$TEST_DATA_DIR" \
    --model_path "$MODEL_PATH" \
    --batch_size 16 \
    --masking_ratio 0.6 \
    --mean_mask_length 3 \
    --feature_indices 34 \
    --num_visualization_samples 5

EXIT_CODE=$?

echo "=========================================="
echo "Testing completed with exit code: $EXIT_CODE"
echo "Job finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Testing completed successfully!"

    echo "Results files:"
    ls -la patchtst_test_results.json 2>/dev/null || echo "No results file found"

    echo "Visualization files:"
    ls -la aircraft_comparison_*.png reconstruction_comparison_*.png 2>/dev/null || echo "No visualization files found"
else
    echo "Testing failed with exit code: $EXIT_CODE"
    echo "Check the error log for details"
fi

echo "=========================================="

exit $EXIT_CODE
