import glob
import torch
from torch.utils.data import DataLoader
from nets.dataset import StressEEGDataset, split_dataset
from nets.model import HEEGNetStress
import torch.nn as nn

def main():
    # ------------------------
    # Device setup
    # ------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[TRAIN] Using device:", device)

    # ------------------------
    # Load dataset
    # ------------------------
    data_dir = "/home/e20286/fyp/FYP/data/processed_32/stress_files/*.npz"
    files = sorted(glob.glob(data_dir))
    print("[TRAIN] Found", len(files), "NPZ files")

    dataset = StressEEGDataset(files, input_key="X_RAW")
    train_set, val_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128)

    # ------------------------
    # Initialize model
    # ------------------------
    model = HEEGNetStress(
        num_classes=2,
        chunk_size=124,
        num_electrodes=32,
        domain_adaptation=True,
        dtype=torch.float32  # <-- important to match CrossEntropyLoss
    ).to(device)

    print("[MODEL] Initialized HEEGNetStress")

    # ------------------------
    # Optimizer and loss
    # ------------------------
    optimizer = model.configure_optimizers(lr=0.01, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()  # logits: [B, 2], labels: [B]

    # ------------------------
    # Training loop
    # ------------------------
    print("\n[TRAIN] Starting training...\n")
    num_epochs = 15

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (features, labels) in enumerate(train_loader):
            # Convert to device and float32
            x = features["inputs"].to(device=device, dtype=torch.float32)
            d = features["domains"].to(device=device, dtype=torch.float32)
            y = labels.to(device=device, dtype=torch.long)  # CrossEntropy requires long labels

            optimizer.zero_grad()
            logits, _ = model(x, d)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[EPOCH {epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f}")

        # ------------------------
        # Validation
        # ------------------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                x = features["inputs"].to(device=device, dtype=torch.float32)
                d = features["domains"].to(device=device, dtype=torch.float32)
                y = labels.to(device=device, dtype=torch.long)

                logits, _ = model(x, d)
                loss = criterion(logits, y)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total
        print(f"[EPOCH {epoch+1}/{num_epochs}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

    print("[TRAIN] Training finished successfully")


if __name__ == "__main__":
    main()

