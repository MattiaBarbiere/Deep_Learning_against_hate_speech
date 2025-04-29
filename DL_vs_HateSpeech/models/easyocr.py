from PIL import Image, ImageFilter, ImageOps
import easyocr
import numpy as np

reader = easyocr.Reader(['en'])

def preprocess_image(image):
    # Convert to grayscale
    gray_image = ImageOps.grayscale(image)

    # Apply slight blur to reduce noise
    blurred_image = gray_image.filter(ImageFilter.GaussianBlur(radius=1))

    # Apply thresholding
    threshold_image = blurred_image.point(lambda x: 0 if x < 128 else 255)

    return threshold_image

def perform_ocr(image):
    # Perform OCR using pytesseract
    image_np = np.array(image)
    result = reader.readtext(image_np, detail=0)
    text = " ".join(result)
    return text

def ocr_pipeline(image):
    preprocessed_image = preprocess_image(image)
    extracted_text = perform_ocr(preprocessed_image)
    return extracted_text

if __name__ == "__main__":
    image_path = "your_image.jpg"  # Replace with your image path
    image = Image.open(image_path)
    result_text = ocr_pipeline(image)
    print("Extracted Text:\n", result_text)
