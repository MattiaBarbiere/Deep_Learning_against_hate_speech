from PIL import Image, ImageFilter, ImageOps
import pytesseract

def preprocess_image_tesseract(image):
    # Convert to grayscale
    gray_image = ImageOps.grayscale(image)

    # Apply slight blur to reduce noise
    blurred_image = gray_image.filter(ImageFilter.GaussianBlur(radius=1))

    # Apply thresholding
    threshold_image = blurred_image.point(lambda x: 0 if x < 128 else 255, '1')

    return threshold_image

def perform_tesseract(image):
    # Perform OCR using pytesseract
    text = pytesseract.image_to_string(image)
    text = text.replace('\n', ' ')
    return text

def tesseract_pipeline(image):
    preprocessed_image = preprocess_image_tesseract(image)
    extracted_text = perform_tesseract(preprocessed_image)
    return extracted_text

if __name__ == "__main__":
    image_path = "your_image.jpg"  # Replace with your image path
    image = Image.open(image_path)
    result_text = tesseract_pipeline(image)
    print("Extracted Text:\n", result_text)
