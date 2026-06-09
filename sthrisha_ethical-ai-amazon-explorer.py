!pip install rasterio


import os

original_vrt = "/kaggle/input/amazon-earthwork-data-thrisha/proper_img.vrt"
fixed_vrt = "/kaggle/working/fixed_proper_img.vrt"

# Map of band numbers to actual TIF files you've uploaded
band_file_map = {
    "B3": "LC09_L1GT_232062_20250601_20250601_02_T2_B3.TIF",
    "B4": "LC09_L1GT_232062_20250601_20250601_02_T2_B4.TIF"
}

# Read the original VRT lines
with open(original_vrt, 'r') as f:
    lines = f.readlines()

# Fix SourceFilename entries based on band mapping
fixed_lines = []
band_index = 3  # Assuming 3-band VRT: B4, B3, B2 (we have B4 and B3)

for line in lines:
    if "<SourceFilename" in line and ".TIF" in line:
        band_key = f"B{band_index}"
        filename = band_file_map.get(band_key, "MISSING.TIF")
        fixed_lines.append(f'      <SourceFilename relativeToVRT="0">/kaggle/input/amazon-earthwork-data-thrisha/{filename}</SourceFilename>\n')
        band_index -= 1
    else:
        fixed_lines.append(line)

# âœ… Write the fixed VRT to a writable directory
with open(fixed_vrt, 'w') as f:
    f.writelines(fixed_lines)

print("âœ… Custom VRT written to:", fixed_vrt)



import rasterio
with rasterio.open("/kaggle/working/fixed_proper_img.vrt") as src:
    print("âœ… VRT opened. Dimensions:", src.width, "x", src.height)
    print("Bands:", src.count)
    print("CRS:", src.crs)



# Only use 2 real bands: B4 and B3
fixed_vrt = "/kaggle/working/final_proper_img.vrt"

# Build 2-band VRT manually using GDAL
from osgeo import gdal

vrt_options = gdal.BuildVRTOptions(separate=True)
tiff_files = [
    "/kaggle/input/amazon-earthwork-data-thrisha/LC09_L1GT_232062_20250601_20250601_02_T2_B4 - Copy.TIF",  # Band 1 (Red)
    "/kaggle/input/amazon-earthwork-data-thrisha/LC09_L1GT_232062_20250601_20250601_02_T2_B3.TIF"          # Band 2 (Green)
]
gdal.BuildVRT(fixed_vrt, tiff_files, options=vrt_options)

print("âœ… Final 2-band VRT written to:", fixed_vrt)



import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open("/kaggle/working/final_proper_img.vrt") as src:
    red = src.read(1)[2000:2500, 2000:2500].astype(np.float32)
    green = src.read(2)[2000:2500, 2000:2500].astype(np.float32)

fake_nir = green.copy()  # Simulated NIR

# Normalize
red /= red.max()
green /= green.max()
fake_nir /= fake_nir.max()

rgb = np.stack([fake_nir, green, red], axis=-1)

plt.figure(figsize=(6, 6))
plt.imshow(rgb)
plt.title("âœ… Final RGB Crop (Using B4 & B3)")
plt.axis("off")
plt.show()



# Already have red and fake_nir from previous cell

# Calculate NDVI safely
ndvi = (fake_nir - red) / (fake_nir + red + 1e-10)

# Visualize NDVI
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.imshow(ndvi, cmap='RdYlGn')
plt.title("ğŸŒ¿ NDVI (Green as Fake NIR)")
plt.colorbar(label="NDVI")
plt.axis("off")
plt.show()



from scipy.ndimage import gaussian_filter
import numpy as np

# Smooth and detect anomalies
ndvi_smooth = gaussian_filter(ndvi, sigma=2)
anomalies = (ndvi - ndvi_smooth) > 0.015  # tweak threshold as needed

# Visualize anomaly mask
plt.imshow(anomalies, cmap='gray')
plt.title("ğŸ”� NDVI Anomaly Map (Binary Mask)")
plt.axis("off")
plt.show()

# Get coordinates of anomalies
indices = np.argwhere(anomalies)
print(f"Total anomaly pixels: {len(indices)}")



from pyproj import Transformer
import rasterio

# Get geotransform
with rasterio.open("/kaggle/working/final_proper_img.vrt") as src:
    transform = src.transform
    crs = src.crs

# Adjust pixel index back to full image (offset by 2000 crop start)
coords = [transform * (col + 2000, row + 2000) for row, col in indices[::1000][:5]]

# Convert to Lat/Lon
transformer = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
latlon = [transformer.transform(x, y) for x, y in coords]

# Print coordinates
for i, (lon, lat) in enumerate(latlon):
    print(f"ğŸ“� Site {i+1}: Latitude {lat:.5f}, Longitude {lon:.5f}")



from pyproj import Transformer
import rasterio

# Get spatial transform
with rasterio.open("/kaggle/working/final_proper_img.vrt") as src:
    transform = src.transform
    crs = src.crs

# Get coordinates (adjust for 2000 crop offset)
coords = [transform * (col + 2000, row + 2000) for row, col in indices[::100][:5]]

# Convert to WGS84 (Lat/Lon)
transformer = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
latlon = [transformer.transform(x, y) for x, y in coords]

# Print results
print("ğŸŒ� Top 4 Predicted Sites:")
for i, (lon, lat) in enumerate(latlon):
    print(f"ğŸ“� Site {i+1}: Latitude {lat:.5f}, Longitude {lon:.5f}")



from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load model (takes 15â€“30s the first time)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Convert your RGB numpy array (float32, range [0â€“1]) to PIL image
rgb_uint8 = (rgb * 255).astype('uint8')  # Use 'rgb' from Task 2
image = Image.fromarray(rgb_uint8)

# Define comparison labels
labels = ["ancient mound", "jungle", "earthwork", "natural area"]

# Run CLIP
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]

# Display results
for label, score in zip(labels, probs):
    print(f"ğŸ§  CLIP Score for '{label}': {score:.2f}")



import folium

# Use your top site coordinates
m = folium.Map(location=[-2.41, -62.5], zoom_start=9)

sites = [
    (-2.38827, -62.56480),
    (-2.40401, -62.55536),
    (-2.42244, -62.47143),
    (-2.46316, -62.51000)
]

for i, (lat, lon) in enumerate(sites, 1):
    folium.Marker([lat, lon], popup=f"Site {i}").add_to(m)

m.save("amazon_prediction_map.html")



from IPython.display import HTML

# Open the saved HTML file and display it
with open("/kaggle/input/amazon-earthwork-data-thrisha/amazon_prediction_map.html", "r") as f:
    html = f.read()

HTML(html)


