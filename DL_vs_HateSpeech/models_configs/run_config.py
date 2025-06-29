#hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader as TorchDataLoader
import sys, os
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

# Append the parent directory to sys.path to import modules
sys.path.append(os.path.abspath(".."))


@hydra.main(version_base=None, config_path="config_files", config_name="model_config.yaml")
def main(cfg: DictConfig):

    # Produce a lot of models
    for i in range(10):

        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        print(f"Using device: {device}")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        print(cfg_dict)

        # Hyperparameters
        BATCH_SIZE = cfg.train.batch_size
        LR = cfg.train.lr
        EPOCHS = cfg.train.epochs
        AUGMENTATION = cfg.train.augmentation
        WEIGHT_DECAY = cfg.train.weight_decay
        DATA_SUBSET = cfg.train.data_subset
        model_class = MODEL_NAMES[cfg.model.type]
        model_kwargs = cfg.model.model_kwargs

        # Load Data
        train_dataset = DataLoader(type="train", subset=DATA_SUBSET)
        val_dataset = DataLoader(type="test", subset=DATA_SUBSET)

        train_loader = TorchDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = TorchDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        # Initialize Model, Optimizer, Loss
        model = model_class(**model_kwargs).to(device)
        optimizer, criterion = get_optimizer_and_criterion(model, lr=LR, weight_decay=WEIGHT_DECAY)

        # Training and evaluation loop
        train_losses = []
        val_losses = []
        f1_scores = []
        accuracies = []

        # Check how many parameters are frozen
        check_frozen_params(model, print_layers=False)

        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")

            # Train
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, augmentation=AUGMENTATION)
            print(f"Train Loss: {train_loss:.4f}")
            train_losses.append(train_loss)

            # Evaluation loss and accuracy
            val_loss, accuracy, f1 = evaluate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {accuracy * 100:.2f}%")
            val_losses.append(val_loss)
            accuracies.append(accuracy)
            f1_scores.append(f1)

            # Save the model every 10 epochs
            # if (epoch + 1) % 10 == 0 or accuracy > 0.6:
            #     model.save(file_name=f"model_epoch_{epoch + 1}_ac_{accuracy}.pth")
            #     torch.save(val_losses, "./val_loss.pt")
            #     torch.save(train_losses, "./train_loss.pt")
            #     torch.save(accuracies, "./accuracies.pt")
            #     torch.save(f1_scores, "./f1_scores.pt")

        model.save(file_name=f"model_{i}.pth")


        # Plot at the end
        # plot_losses(train_losses, val_losses, save_path="loss_plot.png")

if __name__ == "__main__":
    main()