import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

def collate_fn(batch):
    """
    Custom collate function to handle a batch of (image, text, label)
    where image is a PIL.Image and text is a string.
    """
    # Unzip list of tuples
    images, texts, labels = zip(*batch)

    # Convert images to tensors
    labels = torch.tensor(labels, dtype=torch.float32)
    return list(images), list(texts), labels

def get_optimizer_and_criterion(model, lr=1e-5):
    """
    Returns an AdamW optimizer and a binary cross-entropy loss function.
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    return optimizer, criterion

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, texts, labels in tqdm(dataloader, desc="Training"):
        labels = labels.float().to(device)

        # Forward pass
        optimizer.zero_grad()
        probs = model(texts, images)
        loss = criterion(probs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
