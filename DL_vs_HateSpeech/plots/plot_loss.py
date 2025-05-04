import matplotlib.pyplot as plt
import torch
import os

def plot_losses_from_path(path, save_path=None):
    """
    Loads training and validation losses from specified path and plots them.

    Args:
        path (str): Path to the directory containing loss files.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    train_losses = torch.load(os.path.join(path, "train_loss.pt"))
    val_losses = torch.load(os.path.join(path, "val_loss.pt"))
    
    plot_losses(train_losses, val_losses, save_path)

def plot_losses(train_losses, val_losses, save_path=None):
    """
    Plots training and validation loss curves.

    Args:
        train_losses (list of float): Training loss per epoch.
        val_losses (list of float): Validation loss per epoch.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
