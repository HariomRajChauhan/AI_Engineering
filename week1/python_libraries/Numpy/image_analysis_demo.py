import numpy as np
import matplotlib.pyplot as plt
import os

print("=== IMAGE ANALYSIS TECHNIQUES ===\n")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'image1.jpg')

# Load an image from file
image = plt.imread(image_path)
print(f"✓ Loaded image: {image_path}")

# 1. BASIC IMAGE PROPERTIES
print("\n1. BASIC IMAGE PROPERTIES")
print("-" * 40)
print(f"• Dimensions: {image.shape[0]}x{image.shape[1]} pixels")
print(f"• Color channels: {image.shape[2]} (RGB)")
print(f"• Data type: {image.dtype}")
print(f"• Total pixels: {image.shape[0] * image.shape[1]:,}")
print(f"• Memory usage: ~{image.nbytes / (1024*1024):.2f} MB")

# 2. PIXEL VALUE STATISTICS
print("\n2. PIXEL VALUE STATISTICS")
print("-" * 40)
print(f"• Min pixel value: {np.min(image)}")
print(f"• Max pixel value: {np.max(image)}")
print(f"• Mean pixel value: {np.mean(image):.2f}")
print(f"• Median pixel value: {np.median(image):.2f}")
print(f"• Standard deviation: {np.std(image):.2f}")

# 3. RGB CHANNEL ANALYSIS
print("\n3. RGB CHANNEL ANALYSIS")
print("-" * 40)
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

channels = [("Red", red_channel), ("Green", green_channel), ("Blue", blue_channel)]
for name, channel in channels:
    print(f"• {name:5} channel - Min: {np.min(channel):3d}, Max: {np.max(channel):3d}, Mean: {np.mean(channel):6.2f}")

# Determine dominant color
means = [np.mean(red_channel), np.mean(green_channel), np.mean(blue_channel)]
dominant_idx = np.argmax(means)
colors = ['Red', 'Green', 'Blue']
print(f"• Dominant color: {colors[dominant_idx]} (mean: {means[dominant_idx]:.2f})")

# 4. BRIGHTNESS ANALYSIS
print("\n4. BRIGHTNESS ANALYSIS")
print("-" * 40)
# Convert to grayscale using standard weights
gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
print(f"• Overall brightness (grayscale mean): {np.mean(gray_image):.2f}")

# Classify brightness
avg_brightness = np.mean(gray_image)
if avg_brightness < 85:
    brightness_class = "Dark"
elif avg_brightness < 170:
    brightness_class = "Medium"
else:
    brightness_class = "Bright"
print(f"• Brightness classification: {brightness_class}")

# 5. CONTRAST ANALYSIS
print("\n5. CONTRAST ANALYSIS")
print("-" * 40)
contrast = np.std(gray_image)
print(f"• Contrast (grayscale std dev): {contrast:.2f}")

if contrast < 30:
    contrast_class = "Low contrast"
elif contrast < 60:
    contrast_class = "Medium contrast"
else:
    contrast_class = "High contrast"
print(f"• Contrast classification: {contrast_class}")

# 6. HISTOGRAM ANALYSIS
print("\n6. HISTOGRAM ANALYSIS")
print("-" * 40)
hist, bins = np.histogram(gray_image, bins=256, range=(0, 255))
peak_brightness = bins[np.argmax(hist)]
print(f"• Most common brightness level: {peak_brightness:.1f}")
print(f"• Pixels at peak brightness: {np.max(hist):,}")

# 7. IMAGE REGIONS
print("\n7. IMAGE REGIONS")
print("-" * 40)
threshold = np.mean(gray_image)
bright_pixels = np.sum(gray_image > threshold)
dark_pixels = np.sum(gray_image <= threshold)
total_pixels = gray_image.size

print(f"• Bright regions (>{threshold:.1f}): {bright_pixels:,} pixels ({bright_pixels/total_pixels*100:.1f}%)")
print(f"• Dark regions (≤{threshold:.1f}): {dark_pixels:,} pixels ({dark_pixels/total_pixels*100:.1f}%)")

# 8. EDGE DETECTION (SIMPLE)
print("\n8. EDGE ANALYSIS")
print("-" * 40)
# Calculate gradients
grad_x = np.gradient(gray_image, axis=1)
grad_y = np.gradient(gray_image, axis=0)
edges = np.sqrt(grad_x**2 + grad_y**2)

edge_threshold = 30
edge_pixels = np.sum(edges > edge_threshold)
print(f"• Edge pixels (gradient > {edge_threshold}): {edge_pixels:,}")
print(f"• Edge density: {edge_pixels/total_pixels*100:.2f}%")

# 9. COLOR DISTRIBUTION
print("\n9. COLOR DISTRIBUTION")
print("-" * 40)
# Find unique colors (simplified)
unique_pixels = len(np.unique(image.reshape(-1, image.shape[-1]), axis=0))
print(f"• Unique colors in image: {unique_pixels:,}")
print(f"• Color diversity: {unique_pixels/total_pixels*100:.2f}%")

# 10. SUMMARY
print("\n" + "="*50)
print("IMAGE ANALYSIS SUMMARY")
print("="*50)
print(f"Image: {os.path.basename(image_path)}")
print(f"Size: {image.shape[0]}x{image.shape[1]} pixels")
print(f"Brightness: {brightness_class} (avg: {avg_brightness:.1f}/255)")
print(f"Contrast: {contrast_class} (std: {contrast:.1f})")
print(f"Dominant color: {colors[dominant_idx]}")
print(f"Edge density: {edge_pixels/total_pixels*100:.1f}%")
print(f"Color diversity: {unique_pixels:,} unique colors")

print("\n✓ Analysis complete!")