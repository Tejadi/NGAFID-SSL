import os
import io
import copy
import argparse
import math
import sys
import torch
import wandb
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from models.resnet_simclr import ResNetSimCLR
from benchmarks.conv_mhsa.loader import GADataset, DATA_PATH, RegressionGADataset
from db.interface import DBInterface
from utils import load_flight_data
from tqdm import tqdm
from utils import plot_aircraft_type_comparison

db = DBInterface(connection_string="sqlite:///benchmarks.db")
state_dict = io.BytesIO()

class RegressionHead(nn.Module):
    def __init__(self, simclr_model: ResNetSimCLR, input_dim: int, seq_len: int, feature_dim: int):
        super().__init__()

        head = simclr_model.backbone.fc

        if isinstance(head, nn.Sequential):
            orig_fc = head[-1]
        elif isinstance(head, nn.Linear):
            orig_fc = head
        else:
            raise RuntimeError(f"Unexpected fc type: {type(head)}")

        feat_dim = orig_fc.in_features

        simclr_model.remove_projector()
        self.backbone = simclr_model.backbone

        for p in self.backbone.parameters():
            p.requires_grad = False

        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        self.regressor = nn.Linear(feat_dim, seq_len * feature_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x_reshaped = x.reshape(batch_size, 1, self.seq_len, self.feature_dim)
                
        f = self.backbone(x_reshaped)
        
        output = self.regressor(f)
        
        output = output.reshape(batch_size, self.seq_len, self.feature_dim)
        
        return output

def eval(model_path, device, seq_len, feature_dim, mask_ratio, mask_len):
    base_simclr = ResNetSimCLR("resnet18", out_dim=128).cpu()
    reg_simclr = copy.deepcopy(base_simclr).to(device)

    reg_simclr = RegressionHead(
        reg_simclr, 
        input_dim=44,
        seq_len=seq_len,
        feature_dim=feature_dim
    ).to(device)

    ckpt = torch.load(model_path, map_location="cpu")
    reg_simclr.load_state_dict(ckpt)
    model = reg_simclr

    mse = nn.MSELoss()
    mae = nn.L1Loss()

    test_dataset = RegressionGADataset(db, 'test', masking_ratio=mask_ratio, mean_mask_length=mask_len)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    running_mse, running_mae = 0.0, 0.0
    for flights, labels, fid in tqdm(test_loader, desc='eval', unit="flight"):
        flights, labels = flights.to(device), labels.to(device)
        
        logits = model(flights)              

        loss_mse = mse(flights, logits)
        loss_mae = mae(flights, logits)

        running_mse += loss_mse.item()
        running_mae += loss_mae.item()

    mae = running_mae / len(test_loader)
    mse = running_mse / len(test_loader)

    print(f'Finished, mae={mae:.2f}, mse={mse:.2f}, rmse = {math.sqrt(mse):.2f}')




def create_model(simclr_state_path, device, seq_len=10000, feature_dim=44):
    base_simclr = ResNetSimCLR("resnet18", out_dim=128).cpu()
    ckpt = torch.load(simclr_state_path, map_location="cpu")
    base_simclr.load_state_dict(ckpt['state_dict'])

    reg_simclr = copy.deepcopy(base_simclr).to(device)
    reg_simclr = RegressionHead(
        reg_simclr, 
        input_dim=44,
        seq_len=seq_len,
        feature_dim=feature_dim
    ).to(device)

    return reg_simclr

def train_epoch(model, loader, optimizer, mse_criterion, mae_criterion, device):
    model.train()

    running_mse = 0.0
    running_mae = 0.0
    total = 0

    for flights, labels, _ in tqdm(loader, desc='Epoch', unit="batch"):
        flights, labels  = flights.to(device), labels.to(device)
        
        logits = model(flights)              

        loss   = mse_criterion(logits, labels)
        l1_loss = mae_criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_mse += loss.item() * flights.size(0)
        running_mae += loss.item() * flights.size(0)

        total += labels.size(0)

    mse = running_mse / total
    mae = running_mae / total

    return mse, mae, math.sqrt(mse)

@torch.no_grad()
def eval_epoch(model, loader, mse_criterion, mae_criterion, device):
    model.eval()

    running_mse = 0.0
    running_mae = 0.0
    total = 0

    for flights, labels, _ in tqdm(loader, desc='Eval', unit="batch"):
        flights, labels  = flights.to(device), labels.to(device)
        logits = model(flights)              

        loss   = mse_criterion(logits, labels)
        l1_loss = mae_criterion(logits, labels)

        running_mse += loss.item() * flights.size(0)
        running_mae += loss.item() * flights.size(0)

        total += labels.size(0)

    mse = running_mse / total
    mae = running_mae / total

    return mse, mae, math.sqrt(mse)

def train(model, device, lr, epochs, name, job_id, mask_ratio=0.5, mask_len=5):
    print(f'Masking ratio: {mask_ratio}, mml: {mask_len}.')
    train_dataset = RegressionGADataset(db, 'train', masking_ratio=mask_ratio, mean_mask_length=mask_len)
    val_dataset = RegressionGADataset(db, 'val', masking_ratio=mask_ratio, mean_mask_length=mask_len)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    wandb.init(
        project="ngafid-ssl-fall-24",
        entity="ngafid-ssl",
        name=name,

        config={
            'learning_rate': lr,
            'epochs': epochs,
            'name': name,
        }
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    best_val_mse = -1.0

    for epoch in range(0, epochs):
        mse, mae, rmse = train_epoch(model, train_loader, optimizer, criterion_mse, criterion_mae, device)
        val_mse, val_mae, val_rmse = eval_epoch(model, val_loader, criterion_mse, criterion_mae, device)

        if val_mse > best_val_mse:
            torch.save(model.state_dict(), state_dict)

        epoch_data = {
            'job_id': job_id,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'val_rmse': val_rmse
        }

        print(f'Epoch {epoch}/{epochs} finished!')
        print(epoch_data)

        wandb.log(epoch_data)
        db.insert_row('Training', epoch_data)

    return model


def main(args):
    parser = argparse.ArgumentParser(description="description")

    parser.add_argument("-m", "--model", type=str, default=None, dest="model")
    parser.add_argument("-l", "--lr", type=float, default=1e-4, dest="lr")
    parser.add_argument("-e", "--epochs", type=int, default=50, dest="epochs")
    parser.add_argument("-g", "--gpu", type=str, default='cuda:1', dest="gpu")
    parser.add_argument("-n", "--name", type=str, default="Job", dest="name")
    parser.add_argument("-r", "--mask-ratio", type=float, default=0.2, dest="mask_ratio")
    parser.add_argument("-M", "--mask-length", type=int, default=5, dest="mask_len")
    parser.add_argument("-E", "--eval", action='store_true', dest='eval')

    args = parser.parse_args()

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    device = 'cuda:0'

    if args.eval:
        model_file = args.model
        eval(model_file, device, 10000, 44, args.mask_ratio, args.mask_len)
    else:
        data = {'name': args.name}
        job_id = db.insert_row('Job', data)

        model_file = f'runs/{job_id}_{str(args.mask_ratio)}_{str(args.mask_len)}.pth'
        print(f"Model file is: {str(model_file)}")

        regressor = create_model(args.model, device, seq_len=10000, feature_dim=44)

        model = train(regressor, device, args.lr, args.epochs, args.name, job_id, args.mask_ratio, args.mask_len)

        with open(model_file, 'wb') as outfile:
            outfile.write(state_dict.getbuffer())
            db_data = {'job_id': job_id, 'model': model_file}
            db.insert_row('Model', data=db_data)

if __name__ == "__main__":
    main(sys.argv)
