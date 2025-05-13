import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from DL_vs_HateSpeech.models.augmentation import augment_batch

class CustomBCELoss(nn.Module):
    def __init__(self):
        super(CustomBCELoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, inputs, targets):
        # Ensure inputs and targets are of the same shape
        if inputs.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {inputs.shape} vs {targets.shape}")
        # Ensure inputs are in the range [0, 1]
        inputs = torch.sigmoid(inputs)
        return self.criterion(inputs, targets)


def collate_fn(batch):
    """
    Custom collate function to handle a batch of (image, text, label)
    where image is a PIL.Image and text is a string.
    """
    # Unzip list of tuples
    images, texts, labels = zip(*batch)

    # Convert labels to tensors
    # labels = torch.tensor(labels, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)
    return list(images), list(texts), labels


def get_optimizer_and_criterion(model, lr=1e-5, weight_decay=0.01):
    """
    Returns an AdamW optimizer and a cross-entropy loss function.
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CustomBCELoss()
    return optimizer, criterion


def train_epoch(model, dataloader, optimizer, criterion, device, augmentation=True):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for images, texts, labels in tqdm(dataloader, desc="Training"):
        # Move data to the specified device
        labels = labels.to(device)

        # Model should handle preprocessing of texts and images
        optimizer.zero_grad()

        # Code for augmentation
        if augmentation:
            # Augment the batch
            texts, images = augment_batch(texts, images)
            # Duplicate labels to match augmentation
            labels = torch.cat([labels, labels], dim=0)   

        # Model forward pass
        probs = model(texts, images)

        # Compute loss
        loss = criterion(probs.squeeze(1), labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

    return total_loss / len(dataloader)