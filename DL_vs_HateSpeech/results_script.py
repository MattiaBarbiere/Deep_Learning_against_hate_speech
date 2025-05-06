import torch
from torch.utils.data import DataLoader as TorchDataLoader

import sys, os
sys.path.append(os.path.abspath(".."))


from DL_vs_HateSpeech.models import *
from DL_vs_HateSpeech.loading_data.dataloader import DataLoader
from DL_vs_HateSpeech.training.training import (
    collate_fn,
    get_optimizer_and_criterion,
    train_epoch
)
from DL_vs_HateSpeech.evaluation.evaluate import evaluate
from DL_vs_HateSpeech.plots.plot_loss import plot_losses
from DL_vs_HateSpeech.utils import check_frozen_params

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 16
LR = 1e-5
EPOCHS = 20
WEIGHT_DECAY = 0.01
AUGMENTATION = True

# Load Data
train_dataset = DataLoader(type="train")
val_dataset = DataLoader(type="val")

train_loader = TorchDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = TorchDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Initialize Model, Optimizer, Loss
model = ModelV1(clip_model_type="32").to(device)
optimizer, criterion = get_optimizer_and_criterion(model, lr=LR, weight_decay=WEIGHT_DECAY)

# Training and evaluation loop
train_losses = []
val_losses = []

# Check how many parameters are frozen
check_frozen_params(model, print_layers=False)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # Train
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, augmentation=AUGMENTATION)
    print(f"Train Loss: {train_loss:.4f}")
    train_losses.append(train_loss)

    # Evaluation loss and accuracy
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy * 100:.2f}%")
    val_losses.append(val_loss)


model_save_path = "./DL_vs_HateSpeech/models/model_checkpoints/model_1_with_augmentation/"
model.save(model_save_path)
torch.save(val_losses, model_save_path + "val_loss.pt")
torch.save(train_losses, model_save_path + "train_loss.pt")


# Plot at the end
# plot_losses(train_losses, val_losses, save_path="loss_plot.png")