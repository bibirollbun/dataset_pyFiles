import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




%%capture
!pip install rasterio matplotlib numpy


import rasterio
import matplotlib.pyplot as plt
import numpy as np

# File path
tiff_file_path = "/kaggle/input/terrabrasilis-brazil-maps/prodes_brasil_2023.tif"

# Open the GeoTIFF file
with rasterio.open(tiff_file_path) as src:
    # Get the affine transform
    transform = src.transform
    pixel_width = transform[0]
    pixel_height = -transform[4]  # Typically negative due to origin at top left

    # Get the overall image size
    width, height = src.width, src.height

    # Print the results
    print(f"Pixel Size (Width x Height): {pixel_width} x {pixel_height} meters")
    print(f"Image Size (Pixels): {width} x {height}")
    print(f"Total Area: {width * pixel_width} x {height * pixel_height} meters")




with rasterio.open(tiff_file_path) as src:
    # Downsample by a factor of 10
    data = src.read(
        1,
        out_shape=(
            int(src.height // 10),
            int(src.width // 10)
        ),
        resampling=rasterio.enums.Resampling.average
    )

plt.figure(figsize=(15, 12))
plt.imshow(data, cmap='terrain')
plt.title("Downsampled PRODES Brazil 2023")
plt.colorbar(label="Deforestation")
plt.show()




# File path
tiff_file_path = "/kaggle/input/terrabrasilis-brazil-maps/prodes_brasil_2023.tif"

# Open the GeoTIFF file
with rasterio.open(tiff_file_path) as src:
    # Define a small window (e.g., 2000x2000 pixels)
    width, height = 80000, 50000
    window = rasterio.windows.Window(0, 0, width, height)
    data = src.read(1, window=window)

    # Get the transform for this window
    transform = src.window_transform(window)

# Plot the sample
plt.figure(figsize=(15, 12))
plt.imshow(data, cmap='terrain')
plt.title("Sample from PRODES Brazil 2023")

plt.colorbar(label="Deforestation")
plt.show()




