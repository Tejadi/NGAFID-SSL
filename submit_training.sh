#!/bin/bash
# Helper script to submit BERT training job to SLURM

echo "🚀 BERT Flight Training Submission Script"
echo "=========================================="

# Check if we're on a SLURM system
if ! command -v sbatch &> /dev/null; then
    echo "❌ Error: sbatch command not found. This script requires a SLURM cluster."
    echo "If you're on a different cluster system, you'll need to adapt the script."
    exit 1
fi

# Create logs directory
mkdir -p logs
echo "✅ Created logs directory"

# Check if data directory exists
if [ ! -d "NGAFID-LOCI-GATS-Data" ]; then
    echo "❌ Error: NGAFID-LOCI-GATS-Data directory not found"
    echo "Please ensure your flight data is in the current directory"
    exit 1
fi

echo "✅ Found data directory: NGAFID-LOCI-GATS-Data"

# Check if training script exists
if [ ! -f "train_bert_masked_regressor.py" ]; then
    echo "❌ Error: train_bert_masked_regressor.py not found"
    exit 1
fi

echo "✅ Found training script"

# Make the SLURM script executable
chmod +x slurm_train_bert.sh
echo "✅ Made SLURM script executable"

# Show current SLURM queue status
echo ""
echo "📊 Current SLURM queue status:"
squeue -u $USER 2>/dev/null || echo "No jobs currently queued"

echo ""
echo "🎯 About to submit training job with these settings:"
echo "   - Sequence length: 1024 (full flight context)"
echo "   - Batch size: 16"
echo "   - Epochs: 18 (better SSL convergence)"
echo "   - Max training files: 1000"
echo "   - Hidden size: 1536 (large model for RTX A5000)"
echo "   - Encoder layers: 12 (BERT-base depth)"
echo "   - Decoder layers: 8"
echo "   - Total parameters: ~250M"
echo "   - Estimated runtime: ~20-24 hours"

echo ""
read -p "Continue with job submission? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Submitting job to SLURM..."

    # Submit the job
    JOB_ID=$(sbatch slurm_train_bert.sh | awk '{print $4}')

    if [ $? -eq 0 ]; then
        echo "✅ Job submitted successfully!"
        echo "📋 Job ID: $JOB_ID"
        echo ""
        echo "📖 To monitor your job:"
        echo "   squeue -j $JOB_ID                    # Check job status"
        echo "   tail -f logs/bert_training_${JOB_ID}.out   # Follow output log"
        echo "   tail -f logs/bert_training_${JOB_ID}.err   # Follow error log"
        echo "   scancel $JOB_ID                      # Cancel job if needed"
        echo ""
        echo "📁 Results will be saved to: ./bert_results/"
        echo ""
        echo "⏱️  Estimated completion: $(date -d '+24 hours' '+%Y-%m-%d %H:%M')"
    else
        echo "❌ Job submission failed"
        exit 1
    fi
else
    echo "❌ Job submission cancelled"
    exit 0
fi