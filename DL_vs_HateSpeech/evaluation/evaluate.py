import torch
from tqdm import tqdm

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, texts, labels in tqdm(dataloader, desc="Evaluating"):
            labels = labels.to(device)

            # Forward pass
            probs = model(texts, images)
            preds = (probs > 0.5).int()

            # Calculate accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total
