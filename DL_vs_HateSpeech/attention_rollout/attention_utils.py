import numpy as np
import torch
from DL_vs_HateSpeech.utils import get_label_str_list

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

    # Min-max normalization
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

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

    # Min-max normalize the attention weights
    final_attn_norm = (final_attn - final_attn.min()) / (final_attn.max() - final_attn.min())

    return final_attn_norm, tokens
