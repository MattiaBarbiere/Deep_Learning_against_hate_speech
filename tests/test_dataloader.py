from DL_vs_HateSpeech.loading_data.dataloader import DataLoader
import matplotlib.pyplot as plt

def test_dataloader():
    train_loader = DataLoader(type="train", subset="both")
    test_loader = DataLoader(type="test", subset="both")
    val_loader = DataLoader(type="val", subset="both")

    assert len(train_loader) > 0, "Train loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."
    assert len(val_loader) > 0, "Validation loader is empty."

    print(f"Train loader length: {len(train_loader)}")
    print(f"Test loader length: {len(test_loader)}")
    print(f"Validation loader length: {len(val_loader)}")

    # Show an image from the train loader
    image, text, label = next(iter(train_loader))
    print(f"Text: {text}")
    print(f"Label: {label}")
    
    plt.imshow(image)
    plt.axis('off')  # Hide the axes
    plt.show()
    

if __name__ == "__main__":
    test_dataloader()
    print("All tests passed!")