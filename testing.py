import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model("best_cifar10_model.h5")

# Function to preprocess the image
def preprocess_image(image_path):
    """
    Loads an image, crops it to a square, resizes it to 32x32, 
    and converts it into a normalized 32x32x3 RGB array.
    
    Parameters:
        image_path (str): Path to the input image.
    
    Returns:
        np.array: Preprocessed image as a 32x32x3 RGB array.
    """
    with Image.open(image_path) as img:
        # Get dimensions and crop to a square
        width, height = img.size
        new_size = min(width, height)
        left = (width - new_size) // 2
        top = (height - new_size) // 2
        img_cropped = img.crop((left, top, left + new_size, top + new_size))
        
        # Resize to 32x32 and normalize pixel values
        img_resized = img_cropped.resize((32, 32)).convert("RGB")
        img_array = np.array(img_resized).astype(np.float32) / 255.0  # Normalize to [0,1]
        
        # Reshape for model input (1, 32, 32, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

# Load and preprocess the image
image_path = "image.png"  # Replace with your image path
preprocessed_image = preprocess_image(image_path)

# Make a prediction
predictions = model.predict(preprocessed_image)
predicted_class = np.argmax(predictions, axis=1)[0]

# CIFAR-10 classes
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]

# Print the predicted class
print(f"The model predicts this image is a: {class_names[predicted_class]}")