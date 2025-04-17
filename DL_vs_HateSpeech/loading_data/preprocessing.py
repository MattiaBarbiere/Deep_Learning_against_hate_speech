# Here we can add any trasnformations we want to apply to the data
import torch
    
class NormalizePixels():
    def __init__(self, clip_normalize=False):
        self.clip_normalize = clip_normalize

        # Define CLIP normalization parameters
        if clip_normalize:
            self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
            self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    def __call__(self, image):
        # # RGBA (4-channel) to RGB (3-channel)
        # if image.shape[0] == 4:
        #     image = image[:3]  # Discard alpha channel if it exists

        # # Scale to [0, 1]
        # image = image / 255.0

        # Apply CLIP normalization if specified
        if self.clip_normalize:
            image = (image - self.mean[:, None, None]) / self.std[:, None, None]
        return image