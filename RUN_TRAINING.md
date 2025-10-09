#  How to Run BERT Flight Training

## Simple Command

Just copy and paste this into your terminal:

```bash
python train_full_flights.py
```

That's it! The script is fully configured and ready to go.

## What Will Happen

1. **Automatic Setup**: Script detects your GPU, data directory, and feature dimensions
2. **Training Starts**: 18 epochs with full 10,000 timestep sequences
3. **Progress Tracking**: Live progress bars and metrics
4. **Results Saved**: Best models and checkpoints saved automatically
5. **Runtime**: About 20-24 hours on RTX A5000

## Training Configuration

- **Model**: 249M parameters (1536 hidden, 12 encoder layers, 8 decoder layers)
- **Data**: `/data/ngafid`
- **Sequence Length**: 10,000 (full flights)
- **Batch Size**: 8 (optimized for long sequences)
- **Learning Rate**: 5e-5 (conservative for stability)
- **Epochs**: 18 (proper SSL convergence)

## Monitoring Progress

The script will show:
- Real-time progress bars with MSE/MAE losses
- Evaluation every 1,000 steps
- Model checkpoints every 3,000 steps
- W&B logging (if available)

## Results Location

Results will be saved to: `./results/bert_full_flights_YYYYMMDD_HHMMSS/`

- `best_model.pt` - Best performing model
- `final_model.pt` - Final model after 18 epochs
- `checkpoint_step_*.pt` - Regular checkpoints
- `tensorboard/` - TensorBoard logs

## If Something Goes Wrong

1. **Out of memory**: The script uses conservative batch size (8), but if you get OOM, restart the script
2. **Data not found**: Check that `/data/ngafid` exists and is accessible
3. **Import errors**: Make sure you're in the project directory and have the right environment

## To Resume Training

If training gets interrupted, you can resume from a checkpoint by modifying the script to load from `checkpoint_step_*.pt`.

---

**Ready to train? Just run:**
```bash
python train_full_flights.py
```