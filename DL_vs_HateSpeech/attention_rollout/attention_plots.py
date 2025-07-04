import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn.functional as F

from DL_vs_HateSpeech.models import load_model_from_path
from DL_vs_HateSpeech.utils import read_yaml_file, get_label_str_list
from DL_vs_HateSpeech.loading_data.dataloader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from DL_vs_HateSpeech.training.training import collate_fn
from DL_vs_HateSpeech.attention_rollout.attention_utils import attention_rollout_image, attention_rollout_text

def plot_attention_rollout(path, file_name="model_1.pth", self_attn=True, blur=True, 
                           alpha_image=0.5, index = None, 
                           show_fig=False, save_fig=False,
                           device="cpu"):
    """
    Plot the attention rollout for a given model and inputs.
 
    Args:
        path (str): Path to the model checkpoint.
        file_name (str): Name of the model file to load. Default is "model_1.pth".
        self_attn (bool): Whether to consider self-attention. Default is True.
        blur (bool): Whether to apply Gaussian blur to the attention map. Default is True.
        alpha_image (float): Overlay opacity for the attention map. Default is 0.5.
        index (int): Index of the sample to visualize. If None, a random sample is selected.
        show_fig (bool): Whether to display the figure. Default is False.
        save_fig (bool): Whether to save the figure. Default is False.
        device (str): Device to load the model on. Default is "cpu".
    """
    # Load model from path
    model = load_model_from_path(path, file_name=file_name, device=device)
 
    # Model type
    model_type = path[-2:]
    print("Model type:", model_type)

    # Get config dict from path
    config_dict = read_yaml_file(path)
    print(config_dict)

    # Create the train dataloader
    train_dataset = DataLoader(type="train", subset=config_dict["train"]["data_subset"])

    # Get the first batch of training data
    if index is None:
        # Shuffle the TorchDataLoader to get a random sample
        train_loader = TorchDataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        
        # If the index is not given, get the first batch
        image, text, labels = next(iter(train_loader))
    else:
        # Dataloader with shuffle=False
        train_loader = TorchDataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        # Get the sample from the dataset
        sample = train_dataset[index]

        # Collate it to match DataLoader output (expects a list of samples)
        image, text, labels = collate_fn([sample])

    # Print shapes and labels
    print("\n Image shape:", image[0].size, "\n")
    print("Label:", get_label_str_list(labels))

    # Plot the image
    plt.figure(figsize=(10, 5))
    plt.imshow(image[0])
    plt.axis('off')
    plt.title("Original Image")
    if save_fig:
        if index is not None:
            plt.savefig(f"saved_images/original_image_{index}.png", bbox_inches='tight', dpi=300)
        else:
            plt.savefig("saved_images/original_image.png", bbox_inches='tight', dpi=300)
    if show_fig:
        plt.show()

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
    plt.imshow(attn_map.cpu().detach().numpy(), cmap='jet')
    plt.colorbar()
    if save_fig:
        if index is not None:
            plt.savefig(f"saved_images/image_and_weights_{index}_{model_type}.png", bbox_inches='tight', dpi=300)
        else:
            plt.savefig("saved_images/image_and_weights.png", bbox_inches='tight', dpi=300)
    if show_fig:
        plt.show()

    # Overlay attention on image
    overlay_attention_on_image(attn_map, image, orig_size, blur=blur, alpha=alpha_image, 
                               show_fig=show_fig, save_fig=save_fig, index=index, model_type=model_type)

    # Perform text attention rollout
    attn_weights, tokens = attention_rollout_text(model, text, image)

    # Plot text attention
    plot_text_attention(attn_weights, tokens, show_fig=show_fig, save_fig=save_fig, index=index, model_type=model_type)
    overlay_attention_on_text(attn_weights, tokens, show_fig=show_fig, save_fig=save_fig, index=index, model_type=model_type)


def overlay_attention_on_image(attn_map, image, orig_size, model_type, blur=True, alpha=0.5, 
                               show_fig=False, save_fig=False, index=None):
    """
    Upsample and overlay attention map onto the original image (PIL-based input).

    Args:
        attn_map (Tensor): Attention map (H_patches, W_patches).
        image (list): List containing a single PIL.Image.
        model_type (str): Type of model used. 
        orig_size (tuple): Original (H, W) of the image.
        blur (bool): Whether to apply Gaussian blur.
        alpha (float): Overlay opacity.
        show_fig (bool): Whether to display the figure.
        save_fig (bool): Whether to save the figure.
        index (int): Index of the sample to visualize. If None, a random sample is selected.

    Returns:
        None. Displays image with overlay.
    """
    # Convert PIL image to numpy array
    pil_img = image[0]
    img_np = np.array(pil_img).astype(np.float32) / 255.0  # (H, W, 3)

    # Convert to grayscale
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) * 0.5


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
    if save_fig:
        if index is not None:
            plt.savefig(f"saved_images/image_attention_overlay_{index}_{model_type}.png", bbox_inches='tight', dpi=300)
        else:
            plt.savefig("saved_images/image_attention_overlay.png", bbox_inches='tight', dpi=300)
    if show_fig:
        plt.show()

def plot_text_attention(weights, tokens, model_type, save_fig=False, show_fig=False, index=None):
    """
    Plot the attention weights for each token.
    
    Args:
        weights (Tensor): Attention weights for each token.
        tokens (list): List of tokens corresponding to the weights.
        model_type (str): Type of model used.
        save_fig (bool): Whether to save the figure. Default is False.
        show_fig (bool): Whether to display the figure. Default is False.
        index (int): Index of the sample to visualize. If None, a random sample is selected.
    """
    # Retrieve the minimum length of tokens and weights
    min_len = min(len(tokens), len(weights))
    weights_np = weights[:min_len].cpu().numpy()
    tokens_trimmed = tokens[:min_len]

    # Set figure size based on number of tokens
    plt.figure(figsize=(6, 0.4 * min_len))

    # Plot horizontal bar chart
    plt.barh(range(min_len), weights_np, align='center')

    # Set token labels on the y-axis
    plt.yticks(range(min_len), tokens_trimmed)
    plt.gca().invert_yaxis()

    plt.xlabel('Attention Weight')
    plt.tight_layout()

    if save_fig:
        if index is not None:
            plt.savefig(f"saved_images/text_attention_{index}_{model_type}.png", bbox_inches='tight', dpi=300)
        else:
            plt.savefig("saved_images/text_attention.png", bbox_inches='tight', dpi=300)
    if show_fig:
        plt.show()

# A dictionary that returns the vertical spacing of the text given the number of lines
vertical_spacing_dict = {2: 1, 3: 0.65, 4: 0.5, 5: 0.4, 6: 0.3, 7: 0.3, 8: 0.25, 9: 0.2, 10: 0.2}

def overlay_attention_on_text(weights, tokens, model_type, show_fig=False, save_fig=False, index=None):
    """
    Overlay attention weights on text tokens.

    Args:
        weights (Tensor): Attention weights for each token.
        tokens (list): List of tokens corresponding to the weights.
        model_type (str): Type of model used.
        alpha (float): Overlay opacity.
        show_fig (bool): Whether to display the figure. Default is False.
        save_fig (bool): Whether to save the figure. Default is False.
        index (int): Index of the sample to visualize. If None, a random sample is selected.

    Returns:
        None. Displays text with overlay.
    """
    # Create a color map
    cmap = plt.get_cmap('Reds')

    # Create a figure and get the renderer
    number_of_lines = len(tokens) // 10 + 1
    fig, ax = plt.subplots(figsize=(2.5, 0.2 * number_of_lines), )
    renderer = fig.canvas.get_renderer()

    # Variables to keep track of the length of the previous tokens and the line counter
    horrizontal_pos = 0
    vertical_pos = 0

    # Iterate through the tokens and their corresponding attention weights
    for i, token in enumerate(tokens):
        
        # Remove the special tokens from the token
        if token.endswith("</w>"):
            token = token[:-4]
        
        # If there are a lot of tokens, split them into lines
        if i % 10 == 0 and i != 0:
            # Move x to the start
            horrizontal_pos = 0

            # Decrease the y postion
            # vertical_pos -= ax.transData.inverted().transform((1, fig.get_size_inches()[1] * fig.dpi))[1]
            vertical_pos -= vertical_spacing_dict[number_of_lines]
            
        # Print the token with the box of color depending on the attention weight
        color = cmap(weights[i].item())
        text_obj = ax.text(horrizontal_pos, vertical_pos, token, ha='left', va='center', fontsize=12,
                           bbox=dict(facecolor=color, alpha=0.75))
        
        # Get the width of the rendered text in data coordinates
        text_bbox = text_obj.get_window_extent(renderer=renderer)

        # Convert the width from pixels to display coordinates
        width_display = text_bbox.width / fig.dpi

        # Ajust the size of the text box
        width_data = width_display * 0.5

        # Add the previous text box size plus a small margin
        horrizontal_pos += width_data + 0.1

        
    plt.axis('off')
    if save_fig:
        if index is not None:
            plt.savefig(f"saved_images/text_attention_overlay_{index}_{model_type}.png", bbox_inches='tight', dpi=300)
        else:
            plt.savefig("saved_images/text_attention_overlay.png", bbox_inches='tight', dpi=300)

    if show_fig:
        plt.show()