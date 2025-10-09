# NGAFID-SSL: Self-Supervised Learning for General Aviation Flight Data

This repository contains the implementation for self-supervised learning approaches on the NGAFID (National General Aviation Flight Information Database) dataset for flight safety analysis.

## Overview

This codebase implements and benchmarks several self-supervised learning (SSL) models for multivariate time-series flight data, including:

- **BERT-based Masked Regression**: Transformer encoder-decoder architecture for masked feature reconstruction
- **Transformer Autoencoder**: Autoencoder-based approach for missing data reconstruction
- **SimCLR**: Contrastive learning framework adapted for flight time-series
- **ConvMHSA**: Convolutional Multi-Head Self-Attention for aircraft classification

## Repository Structure

```
.
├── models/                          # Model architectures
│   ├── bert_masked_regressor.py    # BERT encoder-decoder for masked regression
│   ├── transformer_autoencoder.py  # Transformer-based autoencoder
│   └── resnet_simclr.py           # ResNet backbone for SimCLR
├── ngafid_datasets/                # Dataset implementations
│   ├── bert_flight_dataset.py     # HuggingFace dataset loader
│   ├── local_flight_dataset.py    # Local CSV data loader
│   ├── masked_flight_dataset.py   # Masking transformations
│   └── transformation_dataset.py  # Data augmentation utilities
├── benchmarks/                     # Benchmark tasks
│   ├── autoencoder/               # Autoencoder baseline
│   ├── conv_mhsa/                 # ConvMHSA classification
│   ├── simclr_classifier/         # SimCLR aircraft classification
│   └── simclr_regression/         # SimCLR masked regression
├── preprocessing/                  # Data preprocessing scripts
├── train_full_flights.py          # Main BERT training script (full sequences)
├── train_bert_masked_regressor.py # Configurable BERT training
└── utils.py                       # Utility functions
```

## Requirements

### Environment Setup

Create a conda environment with the required dependencies:

```bash
conda create -n ngafid-ssl python=3.10 -y
conda activate ngafid-ssl
```

Install PyTorch (adjust CUDA version as needed):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install other requirements:

```bash
pip install transformers datasets pandas numpy tqdm tensorboard wandb scikit-learn matplotlib seaborn
```

For memory-efficient training on large sequences:

```bash
pip install bitsandbytes  # For 8-bit optimizers
```

## Dataset

### Data Format

The dataset consists of CSV files where each file contains a single flight with multiple timesteps. Each row represents one timestep with the following format:

- Columns: Flight parameters (e.g., altitude, airspeed, pitch, roll, engine metrics)
- Rows: Sequential timesteps during the flight
- Typical sequence length: 1,000 - 10,000 timesteps
- Feature dimension: ~44 features (depending on aircraft type)

### Data Directory Structure

Organize your data as follows:

```
/data/ngafid/
├── preprocessed_data/
│   ├── train/          # Training flight CSV files
│   ├── val/            # Validation flight CSV files
│   └── test/           # Test flight CSV files
└── aircraft_types.csv  # Aircraft type metadata (optional)
```

### Using HuggingFace Dataset

Alternatively, load data directly from HuggingFace:

```python
from datasets import load_dataset
dataset = load_dataset("username/NGAFID-LOCI-GATS-Data", split="train")
```

## Training

### 1. BERT Masked Regressor (Recommended)

Train the BERT-based masked regressor on full flight sequences:

```bash
python train_full_flights.py
```

This uses the default configuration optimized for long sequences (10,000 timesteps):
- Model: 249M parameters (1536 hidden size, 12 encoder layers, 8 decoder layers)
- Sequence length: 10,000 timesteps
- Batch size: 8
- Training time: ~20-24 hours on RTX A5000

#### Configurable Training

For custom configurations:

```bash
python train_bert_masked_regressor.py \
    --local_data_dir /data/ngafid \
    --seq_len 1024 \
    --batch_size 16 \
    --epochs 18 \
    --hidden_size 768 \
    --encoder_layers 12 \
    --decoder_layers 8 \
    --learning_rate 5e-5 \
    --output_dir ./results \
    --wandb_project "bert-flight-ssl"
```

Key arguments:
- `--local_data_dir`: Path to local CSV flight data
- `--seq_len`: Maximum sequence length (default: 1024)
- `--batch_size`: Training batch size (default: 16)
- `--hidden_size`: Model hidden dimension (default: 768)
- `--encoder_layers`: Number of BERT encoder layers (default: 12)
- `--decoder_layers`: Number of decoder layers (default: 8)
- `--masking_ratio`: Fraction of features to mask (default: 0.5)
- `--mean_mask_length`: Average length of masked segments (default: 60)

### 2. Transformer Autoencoder

Train the baseline autoencoder for missing data reconstruction:

```bash
python benchmarks/autoencoder/train_autoencoder.py \
    --train_data_dir /data/ngafid/preprocessed_data/train \
    --val_data_dir /data/ngafid/preprocessed_data/val \
    --job_name autoencoder_baseline \
    --hidden_dim 64 \
    --batch_size 32 \
    --n_epochs 100
```

### 3. SimCLR Contrastive Learning

Train SimCLR for representation learning:

```bash
python simclr.py \
    --data_dir /data/ngafid \
    --batch_size 64 \
    --epochs 100 \
    --temperature 0.5
```

Then train a classifier on the learned representations:

```bash
python -m benchmarks.simclr_classifier.classifier \
    -m <path_to_simclr_model> \
    -n "SimCLR Classification" \
    -e 50 \
    -g cuda:0
```

### 4. ConvMHSA Aircraft Classification

Train ConvMHSA for aircraft type classification:

```bash
python -m benchmarks.conv_mhsa.train \
    -e 50 \
    -n "ConvMHSA Classifier" \
    -l 1e-5 \
    -g cuda:0
```

## Evaluation

### BERT Masked Regression

Evaluate trained BERT model on test data:

```bash
python test_bert_masked_regressor.py \
    --data_dir /data/ngafid/preprocessed_data/test \
    --model_path ./results/best_model.pt \
    --batch_size 32 \
    --masking_ratio 0.5 \
    --mean_mask_length 60
```

### Autoencoder Evaluation

Evaluate autoencoder on test data:

```bash
python benchmarks/autoencoder/test_autoencoder.py \
    --data_dir /data/ngafid/preprocessed_data/test \
    --model_path ./models/autoencoder_best.pt \
    --norm_params_path ./models/norm_params.json
```

### SimCLR Masked Regression

Evaluate SimCLR for masked feature prediction:

```bash
python -m benchmarks.simclr_regression.regression \
    -m <model_path> \
    -E \
    -r 0.6 \
    -M 3
```

## Running on SLURM Clusters

For training on GPU clusters with SLURM:

### 1. Edit SLURM Configuration

Modify `slurm_train_bert.sh` or `run_full_flights.sh` to match your cluster setup:

```bash
#SBATCH --partition=gpu           # Your GPU partition name
#SBATCH --gres=gpu:1              # Number of GPUs
#SBATCH --mem=64G                 # Memory allocation
#SBATCH --time=24:00:00           # Time limit
```

### 2. Submit Job

```bash
sbatch slurm_train_bert.sh
```

### 3. Monitor Progress

```bash
# Check job status
squeue -u $USER

# View training logs
tail -f logs/bert_training_*.out

# View errors
tail -f logs/bert_training_*.err
```

See `CLUSTER_TRAINING_GUIDE.md` for detailed cluster setup instructions.

## Results

### Missing Data Reconstruction (Masked Feature Prediction)

| Model | Mean Absolute Error | Mean Squared Error |
|-------|--------------------|--------------------|
| BERT Masked Regressor | **0.42** | **0.58** |
| Transformer Autoencoder | 0.46 | 0.62 |
| SimCLR + Regression Head | 4.44 | 25.50 |

### Aircraft Classification

#### Airframe Model Classification
| Model | Accuracy |
|-------|----------|
| ConvMHSA | **0.99** |
| SimCLR + Classifier | 0.82 |

#### Airframe Class Classification
| Model | Accuracy |
|-------|----------|
| ConvMHSA | **1.00** |
| SimCLR + Classifier | 0.30 |

## Key Features

- **Memory-Efficient Training**: Gradient checkpointing and 8-bit optimizers for long sequences
- **Flexible Masking**: Geometric and random masking strategies for SSL
- **Multiple Benchmarks**: Comprehensive comparison of SSL approaches
- **Distributed Training**: SLURM integration for cluster computing
- **Experiment Tracking**: Weights & Biases and TensorBoard support

## Model Architecture Details

### BERT Masked Regressor

The BERT-based model consists of:

1. **Feature Projection**: Linear projection from flight features to hidden dimension
2. **BERT Encoder**: Transformer encoder with:
   - 12 layers (default)
   - 16 attention heads
   - 1536 hidden size (for large model)
   - Geometric noise masking
3. **Custom Decoder**: 8-layer transformer decoder for reconstruction
4. **Output Projection**: Linear layer projecting back to feature space

Training uses MSE loss on masked positions only, with separate forward and backward masking for encoder-decoder attention.

### Masking Strategy

Geometric masking is used to create realistic missing data patterns:
- Masking ratio: 50% of features (configurable)
- Mean mask length: 60 timesteps (configurable)
- Separate masking per feature or joint masking across features
- Geographically distributed masks mimic sensor failures

## Citation

If you use this code or the NGAFID dataset in your research, please cite:

```
[Citation will be added upon publication]
```

## License

This code is released under the MIT License. See `LICENSE.txt` for details.

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:
1. Reduce `--batch_size` (try 4 or 2)
2. Reduce `--seq_len` (try 512 or 256)
3. Reduce `--hidden_size` (try 768 or 512)
4. Enable gradient checkpointing (enabled by default)
5. Use 8-bit optimizers: `pip install bitsandbytes`

### Data Loading Issues

Ensure your data directory structure matches the expected format. Check:
- CSV files are in the correct directories
- Files contain numeric columns only (preprocessed)
- No NaN values or use proper handling

### CUDA Errors

If you get CUDA errors:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
```

## Additional Documentation

- `RUN_TRAINING.md`: Quick start guide for training
- `CLUSTER_TRAINING_GUIDE.md`: Detailed SLURM cluster setup
- `preprocessing/preprocessing_instructions.md`: Data preprocessing guide
