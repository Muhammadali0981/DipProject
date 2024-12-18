import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r'Image'  # Folder containing original images
output_path = r'ProcessedImages'  # Folder to save processed images

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

def load_images(path):
    images = []
    filenames = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

# Flood detection using simple thresholding
def detect_flooded_areas(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to detect water regions
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return thresh

# Function to visualize results with thermal colors and save the output
def visualize_results(image, detected, water_percentage, land_percentage, filename):
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Detected flooded area with thermal colormap
    thermal_image = cv2.applyColorMap(detected, cv2.COLORMAP_JET)
    plt.subplot(1, 2, 2)
    plt.title(f"Detected Flooded Area\nWater: {water_percentage:.2f}% | Land: {land_percentage:.2f}%")
    plt.imshow(thermal_image)
    plt.axis('off')

    plt.tight_layout()
    
    # Save the combined image
    combined_image = np.hstack((image, thermal_image))
    output_filename = os.path.join(output_path, filename)
    cv2.imwrite(output_filename, combined_image)

    # Optionally, display the results
    plt.show()

# Load images
images, image_filenames = load_images(image_path)

# Process and display results for a few images
for i in range(min(15, len(images))):
    image = images[i]
    filename = image_filenames[i]  # Get the original filename
    
    # Detect flooded areas
    detected_flood = detect_flooded_areas(image)
    
    # Calculate water and land percentages
    total_pixels = detected_flood.size
    water_pixels = np.sum(detected_flood == 0)  # White pixels are water
    land_pixels = np.sum(detected_flood == 255)  # Black pixels are land
    
    water_percentage = (water_pixels / total_pixels) * 100
    land_percentage = (land_pixels / total_pixels) * 100
    
    # Visualize results with thermal colormap and save the output
    visualize_results(image, detected_flood, water_percentage, land_percentage, f'processed_{filename}')