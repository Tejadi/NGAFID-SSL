# BERT Flight Training on SLURM Cluster

## 🎯 Quick Start

### 1. Transfer Files to Cluster
```bash
# On your local machine, compress and upload your project
tar -czf bert-flight-project.tar.gz \
    train_bert_masked_regressor.py \
    models/ \
    ngafid_datasets/ \
    slurm_train_bert.sh \
    submit_training.sh \
    NGAFID-LOCI-GATS-Data/

# Upload to cluster (replace with your cluster details)
scp bert-flight-project.tar.gz username@cluster.university.edu:~/
```

### 2. Set Up on Cluster
```bash
# SSH to your cluster
ssh username@cluster.university.edu

# Extract files
cd ~/
tar -xzf bert-flight-project.tar.gz
cd bert-flight-project/

# Set up conda environment (if not already done)
module load anaconda3  # or whatever your cluster uses
conda create -n bert-flight python=3.10 -y
conda activate bert-flight
pip install torch transformers datasets pandas tqdm tensorboard
```

### 3. Submit Training Job
```bash
# Make scripts executable
chmod +x submit_training.sh
chmod +x slurm_train_bert.sh

# Submit the job (this will ask for confirmation)
./submit_training.sh
```

## 📊 Monitoring Your Job

### Check Job Status
```bash
# See all your jobs
squeue -u $USER

# Check specific job (replace JOBID with actual ID)
squeue -j JOBID

# See detailed job info
scontrol show job JOBID
```

### Monitor Training Progress
```bash
# Follow the output log in real-time
tail -f logs/bert_training_JOBID.out

# Check for errors
tail -f logs/bert_training_JOBID.err

# See what files have been created
ls -la bert_results/
```

### If Something Goes Wrong
```bash
# Cancel the job
scancel JOBID

# Check why job failed
cat logs/bert_training_JOBID.err

# Check cluster node status
sinfo
```

## ⚙️ Configuration Options

### Current Settings (in slurm_train_bert.sh):
- **Sequence Length**: 1024 (4x longer than default)
- **Batch Size**: 16
- **Model Size**: 768 hidden, 12 layers (larger than default)
- **Training Files**: Up to 1000 files
- **Runtime**: 24 hours max
- **Memory**: 64GB
- **GPUs**: 1 GPU

### To Modify Settings:
Edit `slurm_train_bert.sh` and change the training parameters:

```bash
# For even longer sequences (if you have more memory):
--seq_len 2048

# For faster training with smaller model:
--hidden_size 512 --encoder_layers 6

# For full dataset:
# Remove --max_files_train and --max_files_val lines

# For more/less time:
#SBATCH --time=48:00:00  # 48 hours
```

## 🎯 Expected Outputs

After successful training, you'll find:
```
bert_results/
└── full_bert_training_YYYYMMDD_HHMMSS/
    ├── args.json              # Training configuration
    ├── best_model.pt          # Best model checkpoint
    ├── final_model.pt         # Final model
    ├── checkpoint_*.pt        # Intermediate checkpoints
    └── logs/                  # TensorBoard logs
```

## 🚨 Common Issues

### 1. **Out of Memory**
- Reduce `--batch_size` from 16 to 8 or 4
- Reduce `--seq_len` from 1024 to 512

### 2. **Job Gets Killed**
- Increase `#SBATCH --time=`
- Check if you hit memory limits with `seff JOBID`

### 3. **Conda Environment Issues**
- Make sure conda is loaded: `module load anaconda3`
- Check environment exists: `conda env list`

### 4. **GPU Not Found**
- Check GPU availability: `sinfo -p gpu`
- Modify `#SBATCH --gres=gpu:1` for your cluster's GPU naming

## 📞 Getting Help

1. **Check cluster documentation** for specific module names and partitions
2. **Contact your cluster admin** for GPU access and resource limits
3. **Check job efficiency** after completion: `seff JOBID`

## 🏃‍♂️ Quick Test Run

For a quick test before the full training:
```bash
# Edit slurm_train_bert.sh and change:
--epochs 1
--max_files_train 10
--seq_len 256
#SBATCH --time=02:00:00

# Then submit
./submit_training.sh
```

This will run a 2-hour test with 10 files to make sure everything works!