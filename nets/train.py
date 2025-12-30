import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.stress_dataset import StressEEGDataset, split_dataset
from models.heegnet_stress import HEEGNetStress
from nets.trainer import Trainer
from nets.callbacks import EarlyStopping, ModelCheckpoint

# ───────────────── CONFIG ─────────────────
DATA_DIR = "./data_npz"
INPUT_KEY = "X_RAW"   # CHANGE HERE ONLY
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda"
DTYPE = torch.float64

# ───────────────── LOAD DATA ──────────────
npz_files = sorted(glob.glob(f"{DATA_DIR}/*.npz"))
assert len(npz_files) == 32, "Expected 32 NPZ files"

dataset = StressEEGDataset(npz_files, input_key=INPUT_KEY)
train_set, val_set = split_dataset(dataset)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# ───────────────── MODEL ──────────────────
model = HEEGNetStress(
    chunk_size=dataset.X.shape[-1],
    num_electrodes=dataset.X.shape[1],
    domains=list(range(len(npz_files))),
    domain_adaptation=True,
    device=DEVICE,
    dtype=DTYPE
)

# ───────────────── TRAINER ────────────────
trainer = Trainer(
    max_epochs=EPOCHS,
    min_epochs=20,
    loss=nn.BCEWithLogitsLoss(),
    callbacks=[
        EarlyStopping(monitor="val_score", patience=15),
        ModelCheckpoint(monitor="val_score", mode="max")
    ],
    device=DEVICE,
    dtype=DTYPE,
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

trainer.fit(model, train_loader, val_loader)
