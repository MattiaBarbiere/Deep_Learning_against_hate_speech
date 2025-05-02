import torch
from tqdm import tqdm

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, texts, labels in tqdm(dataloader, desc="Evaluating"):
            labels = labels.to(device)

            # Forward pass
            probs = model(texts, images)

            # Compute loss
            loss = criterion(probs, labels)
            total_loss += loss.item()

            # Compute predictions (get the class with the highest probability)
            preds = torch.argmax(probs, dim=1)

            # Calculate accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy
