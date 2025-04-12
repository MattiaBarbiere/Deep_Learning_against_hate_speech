from DL_vs_HateSpeech.loading_data.dataloader import DataLoader
import matplotlib.pyplot as plt

def test_dataloader():
    train_loader = DataLoader(type="train")
    test_loader = DataLoader(type="test")
    val_loader = DataLoader(type="val")

    assert len(train_loader) > 0, "Train loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."
    assert len(val_loader) > 0, "Validation loader is empty."

    # Show an image from the train loader
    image, text, label = next(iter(train_loader))
    print(f"Text: {text}")
    print(f"Label: {label}")
    
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')  # Hide the axes
    plt.show()
    


if __name__ == "__main__":
    test_dataloader()
    print("All tests passed!")