import matplotlib.pyplot as plt
import torch
import os

def plot_losses_from_path(path, save_path=None, title="Training and Validation Loss"):
    """
    Loads training and validation losses from specified path and plots them.

    Args:
        path (str): Path to the directory containing loss files.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    train_losses = torch.load(os.path.join(path, "train_loss.pt"))
    val_losses = torch.load(os.path.join(path, "val_loss.pt"))
    
    plot_losses(train_losses, val_losses, save_path,title=title)

def plot_metrics_from_path(path, save_path=None, title="Training and Validation Metrics"):
    """
    Loads training and validation metrics from specified path and plots them.

    Args:
        path (str): Path to the directory containing metric files.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    accuracy = torch.load(os.path.join(path, "accuracies.pt"))
    f1 = torch.load(os.path.join(path, "f1_scores.pt"))
    
    plot_metrics(accuracy , f1, save_path,title=title)

def plot_metrics(accuracy, f1, save_path=None, title="Training and Validation Metrics"):
    """
    Plots training and validation metrics curves.

    Args:
        train_metrics (list of float): Training metrics per epoch.
        val_metrics (list of float): Validation metrics per epoch.
        save_path (str, optional): If provided, saves the plot to this path.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(accuracy, label='Accuracy', marker='o')
    plt.plot(f1, label='F1', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_losses(train_losses, val_losses, save_path=None, title="Training and Validation Loss"):
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
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
