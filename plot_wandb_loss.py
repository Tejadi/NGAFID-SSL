import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Authenticate with W&B
wandb.login()

# Initialize the API
api = wandb.Api()

# Replace with your actual entity, project, and run ID
run = api.run("ngafid-ssl/ngafid-ssl-fall-24/joos2md4")

# Retrieve the history of logged metrics
history = run.history(keys=["train/loss", "val/loss"])

# Save to CSV
history.to_csv("loss_history.csv", index=False)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot training loss
sns.lineplot(data=history, y="train/loss", x=history.index, ax=ax1)
ax1.set_title("Training Loss Over Time")
ax1.set_xlabel("Step")
ax1.set_ylabel("Loss")

# Plot validation loss
sns.lineplot(data=history, y="val/loss", x=history.index, ax=ax2)
ax2.set_title("Validation Loss Over Time")
ax2.set_xlabel("Step")
ax2.set_ylabel("Loss")

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("loss_plots.png")
print("Training and validation loss history saved to 'loss_history.csv'.")
print("Plots saved as 'loss_plots.png'.")
