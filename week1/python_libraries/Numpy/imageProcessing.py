import numpy as np
import matplotlib.pyplot as plt
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'image1.jpg')

# Load an image from file
image = plt.imread(image_path)
image_data = image / 255.0  # Normalize pixel values to [0, 1]
print(f"Original image shape: {image.shape}")

plt.figure(figsize=(12, 4))

# Display original image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

# Display histogram
plt.subplot(1, 2, 2)
plt.title('Histogram')
plt.hist(image_data.ravel(), bins=256, color='orange', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim([0, 1])
plt.grid()

plt.tight_layout()
plt.show()

# === IMAGE ANALYSIS TECHNIQUES ===

# 1. Basic Image Statistics
print("\n=== IMAGE STATISTICS ===")
print(f"Image dimensions: {image.shape}")
print(f"Data type: {image.dtype}")
print(f"Min pixel value: {np.min(image)}")
print(f"Max pixel value: {np.max(image)}")
print(f"Mean pixel value: {np.mean(image):.2f}")
print(f"Standard deviation: {np.std(image):.2f}")

# 2. Channel Analysis (for RGB images)
if len(image.shape) == 3:
    print(f"\n=== RGB CHANNEL ANALYSIS ===")
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1] 
    blue_channel = image[:, :, 2]
    
    print(f"Red channel - Min: {np.min(red_channel)}, Max: {np.max(red_channel)}, Mean: {np.mean(red_channel):.2f}")
    print(f"Green channel - Min: {np.min(green_channel)}, Max: {np.max(green_channel)}, Mean: {np.mean(green_channel):.2f}")
    print(f"Blue channel - Min: {np.min(blue_channel)}, Max: {np.max(blue_channel)}, Mean: {np.mean(blue_channel):.2f}")

# 3. Convert to grayscale for further analysis
gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale conversion
print(f"\n=== GRAYSCALE ANALYSIS ===")
print(f"Grayscale image shape: {gray_image.shape}")
print(f"Grayscale mean brightness: {np.mean(gray_image):.2f}")

# 4. Visual Analysis - Multiple views
plt.figure(figsize=(15, 10))

# Original image
plt.subplot(3, 3, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

# RGB channels separately
plt.subplot(3, 3, 2)
plt.title('Red Channel')
plt.imshow(image[:, :, 0], cmap='Reds')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.title('Green Channel') 
plt.imshow(image[:, :, 1], cmap='Greens')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.title('Blue Channel')
plt.imshow(image[:, :, 2], cmap='Blues')
plt.axis('off')

# Grayscale
plt.subplot(3, 3, 5)
plt.title('Grayscale')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

# Histogram for each channel
plt.subplot(3, 3, 6)
plt.title('RGB Histograms')
plt.hist(red_channel.ravel(), bins=50, alpha=0.5, color='red', label='Red')
plt.hist(green_channel.ravel(), bins=50, alpha=0.5, color='green', label='Green')
plt.hist(blue_channel.ravel(), bins=50, alpha=0.5, color='blue', label='Blue')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()

# Image brightness analysis
plt.subplot(3, 3, 7)
plt.title('Brightness Distribution')
plt.hist(gray_image.ravel(), bins=50, color='gray', alpha=0.7)
plt.xlabel('Brightness Level')
plt.ylabel('Frequency')

# Edge detection (simple gradient)
grad_x = np.gradient(gray_image, axis=1)
grad_y = np.gradient(gray_image, axis=0)
edges = np.sqrt(grad_x**2 + grad_y**2)

plt.subplot(3, 3, 8)
plt.title('Edge Detection')
plt.imshow(edges, cmap='gray')
plt.axis('off')

# Image regions analysis (darker vs brighter areas)
threshold = np.mean(gray_image)
binary_image = gray_image > threshold

plt.subplot(3, 3, 9)
plt.title(f'Binary (Threshold: {threshold:.1f})')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# 5. Quantitative Analysis
print(f"\n=== QUANTITATIVE ANALYSIS ===")
print(f"Image size in pixels: {image.shape[0] * image.shape[1]:,}")
print(f"Bright pixels (above mean): {np.sum(gray_image > np.mean(gray_image)):,} ({np.sum(gray_image > np.mean(gray_image))/gray_image.size*100:.1f}%)")
print(f"Dark pixels (below mean): {np.sum(gray_image <= np.mean(gray_image)):,} ({np.sum(gray_image <= np.mean(gray_image))/gray_image.size*100:.1f}%)")
print(f"High contrast areas (edge strength > 30): {np.sum(edges > 30):,}")

# 6. Color dominance analysis
dominant_colors = []
for i, color_name in enumerate(['Red', 'Green', 'Blue']):
    channel_mean = np.mean(image[:, :, i])
    dominant_colors.append((color_name, channel_mean))

dominant_colors.sort(key=lambda x: x[1], reverse=True)
print(f"\n=== COLOR DOMINANCE ===")
for i, (color, value) in enumerate(dominant_colors):
    print(f"{i+1}. {color}: {value:.2f}")

print("\n=== ANALYSIS COMPLETE ===")
print("This analysis includes:")
print("• Basic statistics (dimensions, min/max/mean values)")
print("• RGB channel separation and analysis")  
print("• Grayscale conversion")
print("• Visual representations of different aspects")
print("• Edge detection using gradients")
print("• Binary thresholding")
print("• Quantitative measurements")
print("• Color dominance ranking")