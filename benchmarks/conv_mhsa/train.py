import torch
import argparse
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from db.interface import DBInterface
from benchmarks.conv_mhsa.model import ConvMHSAClassifier
from benchmarks.conv_mhsa.loader import GADataset
from benchmarks.conv_mhsa.flight import AIRCRAFT_CLASS, CLASS_AIRCRAFT, INPUT_COLS
from tqdm import tqdm

def test(db, model, job_id, device):
    dataset = GADataset(db, 'test', predict_engines=True)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model.eval()
    with torch.no_grad():
        for x, y, flight_id in tqdm(test_loader, desc="Testing", unit="instance"):
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            test_data = {
                'job_id': job_id,
                'flight_id': int(flight_id),
                'predicted_aircraft': CLASS_AIRCRAFT[int(preds)]
            }

            db.insert_row('Test', test_data)


def main():
    parser = argparse.ArgumentParser(description="Train NGAFID CONV-MHSA")

    parser.add_argument("-e", "--epochs", type=int, default=50, dest="epochs")
    parser.add_argument("-n", "--name", type=str, required=True, dest="name")
    parser.add_argument("-l", "--lr", type=float, default=1e-4, dest="lr")
    parser.add_argument("-g", "--gpu", type=str, default='cuda', dest="gpu")
    parser.add_argument("-E", "--engines", action='store_true')

    args = parser.parse_args()

    db = DBInterface(connection_string="sqlite:///benchmarks.db")

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    data = {'name': args.name}
    job_id = db.insert_row('Job', data)

    model = ConvMHSAClassifier(input_channels=44, n_classes=3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs

    dataset_train = GADataset(db, 'train', predict_engines=True)
    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)

    dataset_val = GADataset(db, 'val', predict_engines=True)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=True)


    print(f'Job id is {job_id}')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        accuracy, correct, total, running_loss = 0, 0, 0, 0
        for batch_x, batch_y, _ in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
             # Compute accuracy
            _, preds = torch.max(preds, dim=1)  # Get predicted class indices
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

            accuracy = correct / total
            pbar.set_postfix(loss=running_loss / (pbar.n + 1), acc=accuracy)

        val_accuracy, val_correct, val_total, val_running_loss = 0, 0, 0, 0
        val_pbar = tqdm(train_loader, desc=f"Validation epoch {epoch+1}/{num_epochs}", unit="batch")

        with torch.no_grad():
            model.eval()
            for batch_x, batch_y, _ in val_pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_running_loss += loss.item()

                _, preds = torch.max(preds, dim=1)  # Get predicted class indices
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

                val_accuracy = val_correct / val_total
                pbar.set_postfix(loss=val_running_loss / (pbar.n + 1), acc=val_accuracy)


        epoch_data = {
            'job_id': job_id,
            'step': epoch,
            'loss': running_loss,
            'acc': accuracy,
            'val_loss': val_running_loss,
            'val_acc': val_accuracy
        }

        db.insert_row('Training', epoch_data)

    test(db, model, job_id, device)


if __name__ == "__main__":
    main()
