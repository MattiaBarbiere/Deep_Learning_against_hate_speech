import matplotlib.pyplot as plt
import numpy as np
import torch

from DL_vs_HateSpeech.models import load_model_from_path
from DL_vs_HateSpeech.utils import read_yaml_file, get_label_str_list
from DL_vs_HateSpeech.loading_data.dataloader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from DL_vs_HateSpeech.training.training import collate_fn

def plot_attention_rollout(path, self_attn=True, device="cpu"):
    """
    Plot the attention rollout for a given model and inputs.

    Args:
        path (str): Path to the model checkpoint.
        self_attn (bool): Whether to consider self-attention. Default is True.
        device (str): Device to load the model on. Default is "cpu".
    """
    # Load model from path
    model = load_model_from_path(path, file_name="model_epoch_20.pth", device=device)

    # Get config dict from path
    config_dict = read_yaml_file(path)
    print(config_dict)

    # Create the train dataloader
    train_dataset = DataLoader(type="train", subset=config_dict["train"]["data_subset"])
    train_loader = TorchDataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Get the first batch of training data
    image, text, labels = next(iter(train_loader))

    # Print shapes and labels
    print("\n Image shape:", image[0].size, "\n")
    print("Label:", get_label_str_list(labels))

    # Perform attention rollout
    attn_map, _ = attention_rollout(model, text, image, self_attn=self_attn)

    # Plot the image and the attention weights side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image[0])
    plt.axis('off')
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.title("Attention Weights")
    plt.axis('off')
    plt.imshow(attn_map.cpu().detach().numpy(), cmap='viridis')
    plt.colorbar()
    plt.show()

def attention_rollout(model, text, image, self_attn=True):
    """
    Perform attention rollout for a given model and inputs. Considers cross-attention
    between text and image.

    Args:
        model: The model to perform attention rollout on. Must have a method called get_model_attention.
        image: The image input for the model.
        text: The text input for the model.
        self_attn (bool): Whether to consider self-attention. Default is True.

    Returns:
        attn_map: The attention map obtained from the rollout.
        orig_size: The original size of the image.
    """
    # Assert the model has a method called get_model_attention
    assert hasattr(model, 'get_model_attention'), "Model must have a method called get_model_attention"

    # Move model to eval mode
    model.eval()

    # Forward pass through the model to get the attention weights
    probs = model.predict(text, image)

    # Get the prediction
    pred = torch.round(probs).squeeze(0)
    print("Prediction:", get_label_str_list(pred), "\n")

    if self_attn:
        print("Performing self-attention rollout... \n")
    else:
        print("Performing cross-attention rollout... \n")
    
    # Get the dict of attention weights from the model
    attn = model.get_model_attention()

    # Unpack the dict
    clip_text_attn = attn['clip_text']
    clip_image_attn = attn['clip_image']
    classifier_attn = attn['classifier']

    # Get the dimensions of the text and image attention matrices respectively
    dim_text_mat = clip_text_attn[0].shape[2]
    dim_image_mat = clip_image_attn[0].shape[2]

    # Extract the attention weights for the image from the classifier
    if self_attn:
        attn_cumulative_list = [classifier_attn[0][:, :dim_image_mat, :dim_image_mat]]
        for i in range(1, len(classifier_attn)):
            attn_cumulative_list.append(classifier_attn[i][:, :dim_image_mat, :dim_image_mat])
    else:
        attn_cumulative_list = [classifier_attn[0][:, :dim_image_mat, :]]
        for i in range(1, len(classifier_attn)):
            attn_cumulative_list.append(classifier_attn[i])
    
    # Add the attention weights to the clip image attention
    complete_attn_image = clip_image_attn + tuple(attn_cumulative_list)    
    
    # Initialize the rollout matrix as an identity matrix
    rollout = torch.eye(dim_image_mat).unsqueeze(0)

    # Loop through the attention weights and perform the rollout
    for attn in complete_attn_image:
        # Check if the attention weights are 4D (batch_size, num_heads, seq_len, seq_len)
        # If so, take the mean across the heads
        if len(attn.shape) == 4: 
            attn = attn.mean(dim=1)

        # Initialize the identity matrix
        id_mat = torch.eye(attn.shape[1]).to(attn.device)

        # Check if the attention weights are square
        # If so, add the identity matrix to the attention weights
        if attn.shape[1] == attn.shape[2]:
            attn = attn + id_mat.unsqueeze(0)
        else:
            # If not, add the identity matrix to the attention weights
            # and add extra zeros to match the dimensions
            extra_zeros = torch.zeros(attn.shape[1], dim_text_mat).to(attn.device)
            id_rect = torch.cat((id_mat, extra_zeros), dim=-1)
            attn = attn + id_rect.unsqueeze(0)

        # Normalize the attention weights
        attn /= attn.sum(dim=-1, keepdim=True)

        # Forward matrix multiplication
        rollout = torch.matmul(rollout, attn)

    # Print the shape of the final rollout attention weights
    print("Shape of rollout:", rollout.shape, "\n")

    # Extract the attention weights for the image
    attn_flat = rollout[0, 1:, 0]

    # Reshape the attention weights to a square matrix
    dimension = np.sqrt(rollout.shape[1] - 1)
    attn_map = 1 - attn_flat.reshape(int(dimension), int(dimension))

    # Extract original image dimensions
    orig_size = image[0].size[::-1] # (H, W)

    return attn_map, orig_size


if __name__ == "__main__":
    path = "DL_vs_HateSpeech/models/model_checkpoints/ModelV2_single_class"
    
    # Call plot attention rollout function
    plot_attention_rollout(path, self_attn=True, device="cpu")
    print("Attention rollout complete.")
