import torch
from torch.utils.data import DataLoader as TorchDataLoader

import sys, os
sys.path.append(os.path.abspath(".."))


from DL_vs_HateSpeech.models.model_v0 import ModelV0
from DL_vs_HateSpeech.loading_data.dataloader import DataLoader
from DL_vs_HateSpeech.training.training import (
    collate_fn,
    get_optimizer_and_criterion,
    train_epoch
)
from DL_vs_HateSpeech.evaluation.evaluate import evaluate
from DL_vs_HateSpeech.plots.plot_loss import plot_losses

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 16
LR = 1e-5
EPOCHS = 2

# Load Data
train_dataset = DataLoader(type="train")
val_dataset = DataLoader(type="val")

train_loader = TorchDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = TorchDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Initialize Model, Optimizer, Loss
model = ModelV0(clip_model_type="32").to(device)
optimizer, criterion = get_optimizer_and_criterion(model, lr=LR)

# Training and evaluation loop
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # Train
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss: {train_loss:.4f}")
    train_losses.append(train_loss)

    # Evaluation loss and accuracy
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy * 100:.2f}%")
    val_losses.append(val_loss)

# Plot at the end
# plot_losses(train_losses, val_losses, save_path="loss_plot.png")