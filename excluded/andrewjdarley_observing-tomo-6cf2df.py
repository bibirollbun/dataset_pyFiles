import os
import re
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

def normalize_image(image):
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Calculate the lower and upper percentiles
    lower_percentile = np.percentile(img_array, 2)
    upper_percentile = np.percentile(img_array, 98)
    
    # Clip the values
    img_array = np.clip(img_array, lower_percentile, upper_percentile)
    
    # Normalize to 0-255
    img_array = ((img_array - lower_percentile) / (upper_percentile - lower_percentile) * 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def draw_circle(image, position, radius, color):
    # Create a transparent overlay for drawing
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Draw a circle on the overlay
    x, y = position
    overlay_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    
    # Composite the overlay with the original image
    combined = Image.alpha_composite(image.convert("RGBA"), overlay)
    
    return combined

def calculate_radius(distance):
    # Scale the radius based on the distance from the motor position
    if distance <= 10:
        return 10  # Small radius for the exact slice
    elif distance <= 25:
        # Linear interpolation for radius between 10 and 50
        return int(10 + (50 - 10) * (distance - 10) / (25 - 10))
    else:
        return 50  # Maximum radius for slices beyond 25

def create_gif_from_slices(directory, output_filename, duration, labels):
    # Get a list of all image files in the directory matching the pattern
    images = []
    pattern = re.compile(r'slice_\d{4}\.jpg$')  # Regex pattern for slice_XXXX.jpg
    
    # Get all matching files and sort them
    matching_files = sorted([f for f in os.listdir(directory) if pattern.match(f)])
    
    # Find the index where slice_0295.jpg would be
    start_index = 0
    for i, filename in enumerate(matching_files):
        slice_num = int(filename.split('_')[1].split('.')[0])
        if slice_num >= 295:
            start_index = i
            break
    
    # Process files starting from slice 295
    for filename in matching_files[start_index:]:
        filepath = os.path.join(directory, filename)
        img = Image.open(filepath)
        normalized_img = normalize_image(img)  # Normalize the image
        
        # Draw circles on the normalized image
        slice_index = int(filename.split('_')[1].split('.')[0])  # Extract slice number from filename
        
        for _, row in labels.iterrows():
            if row['tomo_id'] == 'tomo_6cf2df':
                motor_axis_0 = row['Motor axis 0']
                motor_axis_1 = row['Motor axis 1']
                motor_axis_2 = row['Motor axis 2']
                
                # Calculate the distance from the current slice to the motor position
                distance = abs(slice_index - motor_axis_0)
                
                # Check if the current slice is within 25 slices of the motor position
                if distance <= 25:
                    radius = calculate_radius(distance)  # Calculate the radius based on distance
                    color = (255, 215, 0, 56)  # Gold color with alpha 56
                    
                    # Draw the circle at the specified position
                    normalized_img = draw_circle(normalized_img, (motor_axis_2, motor_axis_1), radius, color)

        images.append(normalized_img)

    if images:
        # Save the images as a GIF with adjustable duration
        images[0].save(output_filename, save_all=True, append_images=images[1:], loop=0, duration=duration)
        print(f"GIF saved as {output_filename}")
    else:
        print("No images found in the directory matching the pattern.")

if __name__ == "__main__":
    directory = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/tomo_6cf2df'  # Use the current working directory
    output_filename = "output/slices.gif"  # Change this to your desired output filename
    frame_duration = 100  # Duration in milliseconds (100 ms = 0.1 seconds)

    # Load the labels from the CSV file
    labels = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    create_gif_from_slices(directory, output_filename, frame_duration, labels)




