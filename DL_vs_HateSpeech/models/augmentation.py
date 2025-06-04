import random
from torchvision import transforms
from PIL import Image
from io import BytesIO

meme_augmentation = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.1,  # smaller to avoid drastic changes
            contrast=0.1,
            saturation=0.1,
            hue=0.02
        )
    ], p=0.9),  # higher probability because it's relatively safe

    transforms.RandomAffine(
        degrees=3,  # smaller rotation to avoid slanted text
        translate=(0.01, 0.01),  # even smaller shifts, just a few pixels
        scale=(0.98, 1.02),  # tighter scaling to avoid text shrinking or growing too much
        shear=1  # very small shear
    ),

    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))  # slight blur to simulate compression
    ], p=0.3),

    transforms.RandomApply([
        transforms.RandomAdjustSharpness(sharpness_factor=2)
    ], p=0.3),

    transforms.RandomApply([
        transforms.RandomPerspective(distortion_scale=0.1, p=1.0)
    ], p=0.2),  # very light perspective distortion

    transforms.RandomGrayscale(p=0.1),
])


# Simulate JPEG artifacts
def simulate_jpeg_artifacts(img, quality_range=(30, 70)):
    quality = random.randint(*quality_range)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

# This is the main function you call
def augment_batch(texts, images):
    """
    Args:
        texts (list of str): batch of texts
        images (list of PIL.Image): batch of images
    Returns:
        new_texts, new_images (both lists of size 2B)
    """
    new_texts = []
    new_images = []

    for text, image in zip(texts, images):
        # Original
        new_texts.append(text)
        new_images.append(image)

        # Augmented version
        aug_image = meme_augmentation(image)
        # if random.random() < 0.3:  # 30% chance of JPEG compression
        #     aug_image = simulate_jpeg_artifacts(aug_image)

        new_texts.append(text)  # Same text
        new_images.append(aug_image)
        # print("Image augmented")
    return new_texts, new_images
