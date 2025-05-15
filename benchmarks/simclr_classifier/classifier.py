import io
import copy
import argparse
import sys
import torch
import wandb
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from models.resnet_simclr import ResNetSimCLR
from benchmarks.conv_mhsa.loader import GADataset
from db.interface import DBInterface
from tqdm import tqdm

db = DBInterface(connection_string="sqlite:///benchmarks.db")
state_dict = io.BytesIO()

class ClassifierHead(nn.Module):
    def __init__(self, simclr_model: ResNetSimCLR, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
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

        self.classifier = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        x = x.unsqueeze(1)

        f = self.backbone(x)        

        return self.classifier(f)

def create_models(simclr_state_path, device):
    base_simclr = ResNetSimCLR("resnet18", out_dim=128).cpu()
    ckpt = torch.load(simclr_state_path, map_location="cpu")
    base_simclr.load_state_dict(ckpt['state_dict'])

    type_simclr  = copy.deepcopy(base_simclr).to(device)
    class_simclr = copy.deepcopy(base_simclr).to(device)

    type_classifier  = ClassifierHead(type_simclr,  out_dim=3).to(device)
    class_classifier = ClassifierHead(class_simclr, out_dim=2).to(device)

    return type_classifier, class_classifier

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for flights, labels, _ in tqdm(loader, desc='Epoch', unit="batch"):
        flights, labels  = flights.to(device), labels.to(device)
        
        logits = model(flights)              
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * flights.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    train_loss = running_loss / total
    train_acc  = correct / total

    return train_loss, train_acc

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for flights, labels, _ in tqdm(loader, desc="Eval", unit="instance"):
        flights, labels = flights.to(device), labels.to(device)

        logits = model(flights)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * flights.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    val_loss = running_loss / total
    val_acc  = correct / total

    return val_loss, val_acc

def train(model, device, lr, epochs, name, out_dim, job_id):
    trainset = GADataset(db, 'train', predict_engines=True)
    valset = GADataset(db, 'val', predict_engines=True)

    train_loader = DataLoader(trainset, batch_size=16, shuffle=True)
    val_loader = DataLoader(valset, batch_size=1, shuffle=True)

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

    if out_dim >= 3:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_acc = -1.0

    for epoch in range(0, epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            torch.save(model.state_dict(), state_dict)

        epoch_data = {
            'job_id': job_id,
            'loss': train_loss,
            'acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
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
    parser.add_argument("-n", "--name", type=str, required=True, dest="name")

    args = parser.parse_args()

    data = {'name': args.name}
    job_id = db.insert_row('Job', data)

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    type_classifier, class_classifier = create_models(args.model, device)

    model = train(class_classifier, device, args.lr, args.epochs, args.name, type_classifier.out_dim, job_id)

    model_file = f'runs/{job_id}_{model.out_dim}.pth'
    with open(model_file, 'wb') as outfile:
        outfile.write(state_dict.getbuffer())
        db_data = {'job_id': job_id, 'model': model_file}
        db.insert_row('Model', data=db_data)

if __name__ == "__main__":
    main(sys.argv)
