import torch
from DL_vs_HateSpeech.models import load_model_from_path
from DL_vs_HateSpeech.utils import read_yaml_file
from DL_vs_HateSpeech.loading_data.dataloader import DataLoader
from torchsummary import summary

def plot_attention_rollout(path, device="cpu"):
    # Load model from path
    model = load_model_from_path(path, device=device)
    # summary(model, (3, 224, 224))

    # Get config dict from path
    config_dict = read_yaml_file(path)
    print(config_dict)

    # Create the dataloaders
    # train_dataset = DataLoader(type="train", subset=config_dict["train"]["data_subset"])
    # test_dataset = DataLoader(type="test", subset=config_dict["train"]["data_subset"])
    train_dataset = DataLoader(type="train")
    test_dataset = DataLoader(type="test")

    image, text, labels = next(iter(train_dataset))
    # print("Image shape:", image.shape)
    # print("Text shape:", text.shape)
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
    
    # Get the dict of attention weights from the model
    attn = model.get_model_attention()

    # Unpack the dict
    clip_text_attn = attn['clip_text']
    clip_image_attn = attn['clip_image']
    classifier_attn = attn['classifier'] 

    print("Tuple and list:", type(clip_text_attn), type(clip_image_attn), type(classifier_attn))
    print("Lengths of iterables:", len(clip_text_attn), len(clip_image_attn), len(classifier_attn))

    print("Shapes of iterables:", clip_text_attn[0].shape, clip_image_attn[0].shape, classifier_attn[0].shape)
    print("Shapes of iterables:", clip_text_attn[1].shape, clip_image_attn[1].shape, classifier_attn[1].shape)
    print("Shapes of iterables:", clip_text_attn[2].shape, clip_image_attn[2].shape)
    print("Shapes of iterables:", clip_text_attn[3].shape, clip_image_attn[3].shape)
    print("Shapes of iterables:", clip_text_attn[4].shape, clip_image_attn[4].shape)
    print("Shapes of iterables:", clip_text_attn[5].shape, clip_image_attn[5].shape)
    print("Shapes of iterables:", clip_text_attn[6].shape, clip_image_attn[6].shape)
    print("Shapes of iterables:", clip_text_attn[7].shape, clip_image_attn[7].shape)
    print("Shapes of iterables:", clip_text_attn[8].shape, clip_image_attn[8].shape)
    print("Shapes of iterables:", clip_text_attn[9].shape, clip_image_attn[9].shape)
    print("Shapes of iterables:", clip_text_attn[10].shape, clip_image_attn[10].shape)
    print("Shapes of iterables:", clip_text_attn[11].shape, clip_image_attn[11].shape)

    dim_text_mat = clip_text_attn[0].shape[2]
    dim_image_mat = clip_image_attn[0].shape[2]
    print("Dimensions of text and image matrices:", dim_text_mat, dim_image_mat)
    complete_attn_image = clip_image_attn + tuple([c[:,-dim_image_mat:, -dim_image_mat:].unsqueeze(0) for c in classifier_attn])
    rollout = complete_attn_image[-1].clone()
    for ten in reversed(complete_attn_image[:-1]):
        print("Shape of complete attention image:", ten.shape)
        rollout = torch.matmul(rollout, ten)
        print("Shape of rollout:", rollout.shape)







if __name__ == "__main__":
    path = "DL_vs_HateSpeech\models\model_checkpoints\ModelV2_clip_32_aug_True"

    plot_attention_rollout(path, device="cpu")
    print("Attention rollout complete.")

