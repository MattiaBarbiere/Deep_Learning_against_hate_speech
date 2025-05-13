import torch
from DL_vs_HateSpeech.models import load_model_from_path
from DL_vs_HateSpeech.utils import read_yaml_file, get_label_str_list
from DL_vs_HateSpeech.loading_data.dataloader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from DL_vs_HateSpeech.training.training import collate_fn
import matplotlib.pyplot as plt
import numpy as np

def plot_attention_rollout(path, device="cpu"):
    # Load model from path
    model = load_model_from_path(path, file_name="model_epoch_20.pth", device=device)

    # Get config dict from path
    config_dict = read_yaml_file(path)
    print(config_dict)

    # Create the dataloaders
    train_dataset = DataLoader(type="train", subset=config_dict["train"]["data_subset"])
    test_dataset = DataLoader(type="test", subset=config_dict["train"]["data_subset"])

    train_loader = TorchDataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = TorchDataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    image, text, labels = next(iter(train_loader))
    print("Image shape:", image[0].size)
    print("Label:", get_label_str_list(labels))

    attention_rollout(model, text, image)


# A function that performs attention rollout
def attention_rollout(model, text, image):
    """
    Perform attention rollout for a given model and inputs.

    Args:
        model: The model to perform attention rollout on. Must have a method called get_model_attention.
        image: The image input for the model.
        text: The text input for the model.

    Returns:
        Tensor: The attention weights from the model of dims (text_shape, image_shape).
    """
    # Assert the model has a method called get_model_attention
    assert hasattr(model, 'get_model_attention'), "Model must have a method called get_model_attention"

    # Move model to eval mode
    model.eval()

    # Forward pass through the model to get the attention weights
    probs = model(text, image)

    # Get the prediction
    pred = torch.argmax(probs, dim=1)
    print("Prediction:", get_label_str_list(pred))
    
    # Get the dict of attention weights from the model
    attn = model.get_model_attention()

    # Unpack the dict
    clip_text_attn = attn['clip_text']
    clip_image_attn = attn['clip_image']
    classifier_attn = attn['classifier']

    print("Shapes of iterables:", clip_text_attn[0].shape, clip_image_attn[0].shape, classifier_attn[0].shape)

    # Get the dimensions of the text and image attention matrices respectively
    dim_text_mat = clip_text_attn[0].shape[2]
    dim_image_mat = clip_image_attn[0].shape[2]

    # Extract the attention weights for the image
    complete_attn_image = clip_image_attn + tuple([c[:,-dim_image_mat:, -dim_image_mat:].unsqueeze(0) for c in classifier_attn])

    # Perform attention rollout
    rollout = torch.eye(dim_image_mat).unsqueeze(0)
    for ten in reversed(complete_attn_image):
        mean_attn = ten.mean(dim=1)
        mean_attn /= mean_attn.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(rollout, mean_attn)
    print("Shape of rollout:", rollout.shape)

    attn = rollout[0, 1:, 0]
    dimension = np.sqrt(rollout.shape[1] - 1)
    attn = 1- attn.reshape(int(dimension), int(dimension))

    # Plot the image and the attention weights side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image[0])
    plt.axis('off')
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.title("Attention Weights")
    plt.axis('off')
    plt.imshow(attn.cpu().detach().numpy(), cmap='viridis')
    plt.colorbar()
    plt.show()









if __name__ == "__main__":
    path = "DL_vs_HateSpeech\models\model_checkpoints\ModelV2_clip_16_hid_dim_64_drop_0.1_aug_False"
    

    plot_attention_rollout(path, device="cpu")
    print("Attention rollout complete.")

