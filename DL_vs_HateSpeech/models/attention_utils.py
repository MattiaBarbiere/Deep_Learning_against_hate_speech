import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from DL_vs_HateSpeech.models import load_model_from_path
from DL_vs_HateSpeech.utils import read_yaml_file, get_label_str_list
from DL_vs_HateSpeech.loading_data.dataloader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from DL_vs_HateSpeech.training.training import collate_fn

def plot_attention_rollout(path, self_attn=True, blur=True, alpha=0.5, image_index = None, device="cpu"):
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

    # Get the first batch of training data
    if image_index is None:
        # Shuffle the TorchDataLoader to get a random sample
        train_loader = TorchDataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        
        # If the index is not given, get the first batch
        image, text, labels = next(iter(train_loader))
    else:
        # Dataloader with shuffle=False
        train_loader = TorchDataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        # Get the sample from the dataset
        sample = train_dataset[image_index]

        # Collate it to match DataLoader output (expects a list of samples)
        image, text, labels = collate_fn([sample])

    # Print shapes and labels
    print("\n Image shape:", image[0].size, "\n")
    print("Label:", get_label_str_list(labels))

    # Perform attention rollout
    attn_map, orig_size = attention_rollout_image(model, text, image, self_attn=self_attn)

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

    # Overlay attention on image
    overlay_attention_on_image(attn_map, image, orig_size, blur=blur, alpha=alpha)

    # Perform text attention rollout
    attn_weights, tokens = attention_rollout_text(model, text, image)

    # Plot text attention
    plot_text_attention(attn_weights, tokens)

def attention_rollout_image(model, text, image, self_attn=True):
    """
    Perform attention rollout on image for a given model and inputs. Can considers cross-attention
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

def attention_rollout_text(model, text, image):
    """
    Perform attention rollout on text tokens for a given model and inputs.
    This includes both the text encoder and classifier self-attention layers.

    Args:
        model: The model to perform attention rollout on. Must have a method called get_model_attention.
        text: The text input for the model.
        image: The image input for the model.

    Returns:
        final_attn: 1D tensor of attention importance per token after rollout.
        tokens: The list of corresponding text tokens.
    """
    # Move model to eval mode
    model.eval()

    # Forward pass through the model to get the attention weights
    _ = model.predict(text, image)

    # Get the dict of attention weights from the model
    attn = model.get_model_attention()

    # Unpack the attention weights
    clip_text_attn = attn['clip_text']
    classifier_attn = attn['classifier']

    # Get the dimension of the text attention matrix
    dim_text_mat = clip_text_attn[0].shape[2]

    # Extract the attention weights for the text from the classifier
    classifier_text_attn = [classifier_attn[0][:, :dim_text_mat, :dim_text_mat]]
    for i in range(1, len(classifier_attn)):
        classifier_text_attn.append(classifier_attn[i][:, :dim_text_mat, :dim_text_mat])

    # Combine CLIP text attention with classifier attention
    complete_attn_text = clip_text_attn + tuple(classifier_text_attn)

    # Initialize the rollout matrix as an identity matrix
    rollout = torch.eye(dim_text_mat).unsqueeze(0)

    # Loop through the attention weights and perform the rollout
    for attn in complete_attn_text:
        # Check if the attention weights are 4D (batch_size, num_heads, seq_len, seq_len)
        # If so, take the mean across heads
        if len(attn.shape) == 4:
            attn = attn.mean(dim=1)

        # Initialize the identity matrix
        id_mat = torch.eye(attn.shape[1]).to(attn.device)

        # Add the identity matrix to the attention weights (residual connection)
        attn = attn + id_mat.unsqueeze(0)

        # Normalize the attention weights
        attn /= attn.sum(dim=-1, keepdim=True)

        # Forward matrix multiplication
        rollout = torch.matmul(rollout, attn)

    # Extract the attention weights for the text tokens (excluding CLS token at index 0)
    final_attn = rollout[0, 1:, 0]

    # Convert list to string if necessary
    if isinstance(text, list):
        text_input = " ".join(text)
    else:
        text_input = text

    # Convert input text into tokens (must match those used in model input)
    tokens = model.clip.processor.tokenizer.tokenize(text_input)

    return final_attn, tokens

def overlay_attention_on_image(attn_map, image, orig_size, blur=True, alpha=0.5):
    """
    Upsample and overlay attention map onto the original image (PIL-based input).

    Args:
        attn_map (Tensor): Attention map (H_patches, W_patches).
        image (list): List containing a single PIL.Image.
        orig_size (tuple): Original (H, W) of the image.
        blur (bool): Whether to apply Gaussian blur.
        alpha (float): Overlay opacity.

    Returns:
        None. Displays image with overlay.
    """
    # Convert PIL image to numpy array
    pil_img = image[0]
    img_np = np.array(pil_img).astype(np.float32) / 255.0  # (H, W, 3)

    # Convert to grayscale
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)


    # Resize attention to image size
    attn_resized = F.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0), size=orig_size, mode='bilinear', align_corners=False
    ).squeeze()

    # Convert to numpy array and normalize attention map
    attn_np = attn_resized.cpu().detach().numpy()
    attn_norm = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)

    # Optional Gaussian blur
    if blur:
        attn_blur = cv2.GaussianBlur((attn_norm * 255).astype(np.uint8), (11, 11), sigmaX=0)
        attn_overlay = attn_blur.astype(np.float32) / 255.0
    else:
        attn_overlay = attn_norm

    # Show image + heatmap
    plt.figure(figsize=(10, 5))
    plt.imshow(img_np)
    plt.imshow(attn_overlay, cmap='jet', alpha=alpha)
    plt.axis('off')
    # plt.title("Attention Overlay")
    plt.savefig("saved_images/attention_overlay.png", bbox_inches='tight', dpi=300)
    plt.show()

def plot_text_attention(weights, tokens):
    """
    Plot the attention weights for each token.
    
    Args:
        weights (Tensor): Attention weights for each token.
        tokens (list): List of tokens corresponding to the weights.
    """
    plt.figure(figsize=(len(tokens) * 0.5, 2))
    min_len = min(len(tokens), len(weights))
    plt.bar(range(min_len), weights[:min_len].cpu().numpy())
    plt.xticks(range(min_len), tokens[:min_len], rotation=90)
    # plt.bar(range(len(tokens)), weights.cpu().numpy())
    # plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title("Attention per Token")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = "DL_vs_HateSpeech/models/model_checkpoints/ModelV2_single_class_clip_32"
    
    # Call plot attention rollout function
    plot_attention_rollout(path, self_attn=True, blur=False, alpha=0.5, image_index=813, device="cpu")
    print("Attention rollout complete.")
