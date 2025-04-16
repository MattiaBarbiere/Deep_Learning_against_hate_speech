# Here we can add any trasnformations we want to apply to the data
import torch

class NormalizePixels():
    """
    Normalize the image values between 0 and 1.
        
    Parameters
    image(torch.Tensor): The image to normalize.
    
    Returns
    torhc.Tensor: The normalized image.
    """
    def __init__(self):
        pass
    
    def __call__(self, image):
        # Normalize the image values between 0 and 1
        return image / 255.0