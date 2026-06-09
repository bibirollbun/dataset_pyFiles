import os
import io
import math
import glob
import json
import base64

import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt

import openai

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
OPENAI_API_KEY = user_secrets.get_secret("OpenAI_key")
openai.api_key = OPENAI_API_KEY


%%capture
! apt-get update 
! apt-get install -y pdal
! pip install --upgrade pip
! pip install uv
! uv pip install rasterio laspy

import rasterio
import laspy


input_files = glob.glob('/kaggle/input/ant-a01-2011-lidar/ANTL/*.laz')
for in_file in input_files:
    out_file = "/kaggle/working/" + os.path.basename(in_file).replace('.laz', '_ground.las')
    cmd = f"pdal translate {in_file} {out_file} filters.smrf filters.range --filters.range.limits=\"Classification[2:2]\" > /dev/null 2>&1"
    print(f"Working with: {in_file} → {out_file}")
    os.system(cmd)


las_files = sorted(glob.glob('/kaggle/working/*_ground.las'))

pipeline = {
    "pipeline": las_files + ["/kaggle/working/ALL_GROUND_POINTS.las"]
}

with open("merge_pipeline.json", "w") as f:
    json.dump(pipeline, f)

print("PDAL pipeline JSON written for merging.")


!pdal pipeline merge_pipeline.json > /dev/null 2>&1
print("✅ Merged with PDAL pipeline: /kaggle/working/ALL_GROUND_POINTS.las")


pipeline = {
    "pipeline": [
        "/kaggle/working/ALL_GROUND_POINTS.las",
        {
            "type": "writers.gdal",
            "filename": "/kaggle/working/ALL_GROUND_POINTS_DEM.tif",
            "resolution": 1.0,          # pixel size in units of your LAS (usually meters)
            "output_type": "min",       # or 'max', 'mean', 'idw', 'count' — 'min' gives ground surface
            "data_type": "float32",
            "nodata": -9999
        }
    ]
}

with open("dem_pipeline.json", "w") as f:
    json.dump(pipeline, f)

!pdal pipeline dem_pipeline.json > /dev/null 2>&1
print("✅ DEM GeoTIFF created: /kaggle/working/ALL_GROUND_POINTS_DEM.tif")


# Read DEM from GeoTIFF
with rasterio.open('/kaggle/working/ALL_GROUND_POINTS_DEM.tif') as src:
    dem = src.read(1)
    profile = src.profile
    nodata = profile.get('nodata', -9999)

# Mask nodata values for visualization
dem_clean = np.where(dem == nodata, np.nan, dem)
vmin = np.nanmin(dem_clean)
vmax = np.nanmax(dem_clean)

# Scale DEM to 0-255 for 8-bit image
dem_scaled = ((dem_clean - vmin) / (vmax - vmin) * 255).astype(np.uint8)

# Save as simple TIFF (no geodata)
img = Image.fromarray(dem_scaled)
img.save('/kaggle/working/ALL_GROUND_POINTS_DEM_simple.tif')
print("✅ Saved simple TIFF: /kaggle/working/ALL_GROUND_POINTS_DEM_simple.tif")

# Show DEM with matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(dem, cmap='terrain')
plt.colorbar(label='Elevation')
plt.title('Digital Elevation Model (DEM) from Ground Points')
plt.axis('off')
plt.show()


# Read DEM from GeoTIFF
with rasterio.open('/kaggle/working/ALL_GROUND_POINTS_DEM.tif') as src:
    dem = src.read(1)
    profile = src.profile
    nodata = profile.get('nodata', -9999)

# Mask nodata values (set NaN for nodata, so we can handle them)
dem_clean = np.where(dem == nodata, np.nan, dem)
vmin = np.nanmin(dem_clean)
vmax = np.nanmax(dem_clean)

# Scale valid DEM to 0-255, and set NaN to 0 (black)
dem_scaled = np.zeros_like(dem_clean, dtype=np.uint8)
valid_mask = ~np.isnan(dem_clean)
dem_scaled[valid_mask] = ((dem_clean[valid_mask] - vmin) / (vmax - vmin) * 255).astype(np.uint8)

# Save as simple grayscale TIFF
img = Image.fromarray(dem_scaled)
img.save('/kaggle/working/ALL_GROUND_POINTS_DEM_simple_gray.tif')
print("✅ Saved simple grayscale TIFF: /kaggle/working/ALL_GROUND_POINTS_DEM_simple_gray.tif")

# Display as large grayscale image
plt.figure(figsize=(20, 12.5))
plt.imshow(dem_scaled, cmap='gray')
plt.title('DEM (grayscale, scaled, nodata=black)')
plt.axis('off')
plt.show()


with rasterio.open('/kaggle/working/ALL_GROUND_POINTS_DEM.tif') as src:
    dem = src.read(1)
    profile = src.profile
    nodata = profile.get('nodata', -9999)

# Replace nodata with np.nan for processing
dem_nan = np.where(dem == nodata, np.nan, dem)

# Make mask of valid (not nan) points
nan_mask = np.isnan(dem_nan)

# Fill NaN using nearest neighbor interpolation
# (distance_transform_edt returns the indices of the nearest non-NaN cell)
filled_dem = dem_nan.copy()
if np.any(nan_mask):
    # Get indices of valid (not NaN) cells
    inds = ndimage.distance_transform_edt(nan_mask,
                                          return_distances=False,
                                          return_indices=True)
    filled_dem = dem_nan[tuple(inds)]

# Now you can rescale and visualize as before
vmin, vmax = np.nanmin(filled_dem), np.nanmax(filled_dem)
dem_scaled = ((filled_dem - vmin) / (vmax - vmin) * 255).astype(np.uint8)

# Save filled TIFF
from PIL import Image
img = Image.fromarray(dem_scaled)
img.save('/kaggle/working/ALL_GROUND_POINTS_DEM_filled_gray.tif')
print("✅ Saved filled grayscale TIFF: /kaggle/working/ALL_GROUND_POINTS_DEM_filled_gray.tif")

# Visualize
plt.figure(figsize=(20, 12.5))
plt.imshow(dem_scaled, cmap='gray')
plt.title('DEM with interpolated (inpainted) nodata')
plt.axis('off')
plt.show()


img = Image.open('/kaggle/working/ALL_GROUND_POINTS_DEM_filled_gray.tif')
print("Image size (width x height):", img.size)  # (width, height)


img = Image.open('/kaggle/working/ALL_GROUND_POINTS_DEM_filled_gray.tif')
tile_size = 1248  
overlap = 128     

width, height = img.size

tiles_dir = '/kaggle/working/tiles'
os.makedirs(tiles_dir, exist_ok=True)

n_x = math.ceil((width - overlap) / (tile_size - overlap))
n_y = math.ceil((height - overlap) / (tile_size - overlap))

tile_idx = 0
for ix in range(n_x):
    for iy in range(n_y):
        x0 = ix * (tile_size - overlap)
        y0 = iy * (tile_size - overlap)
        x1 = min(x0 + tile_size, width)
        y1 = min(y0 + tile_size, height)
        tile = img.crop((x0, y0, x1, y1))
        # Pad tile if it's smaller than tile_size (bottom or right edge)
        padded = Image.new("L", (tile_size, tile_size), 0)
        padded.paste(tile, (0, 0))
        out_path = os.path.join(tiles_dir, f"tile_{ix}_{iy}.tif")
        padded.save(out_path)
        tile_idx += 1
print(f"Saved {tile_idx} tiles to: {tiles_dir}")


import rasterio
import numpy as np

# Read DEM from GeoTIFF
with rasterio.open('/kaggle/working/ALL_GROUND_POINTS_DEM.tif') as src:
    dem = src.read(1)
    profile = src.profile
    nodata = profile.get('nodata', -9999)

# Exclude nodata values
dem_valid = dem[dem != nodata]

# Compute 5th and 95th percentiles (ignore outliers)
p5 = np.percentile(dem_valid, 5)
p95 = np.percentile(dem_valid, 95)

print(f"DEM (5th percentile):  {p5:.2f}")
print(f"DEM (95th percentile): {p95:.2f}")
print(f"Main elevation difference (relief): {p95 - p5:.2f} units")


# Example for a single tile (replace with your actual tile if needed)
tile_path = '/kaggle/working/tiles/tile_0_0.tif'
img = Image.open(tile_path)
width, height = img.size  # in pixels

# Get actual elevation values for this region
with rasterio.open('/kaggle/working/ALL_GROUND_POINTS_DEM.tif') as src:
    dem = src.read(1)
    nodata = src.profile.get('nodata', -9999)
    dem_valid = dem[dem != nodata]
    # Use 5th and 95th percentiles to exclude outliers
    p5 = np.percentile(dem_valid, 5)
    p95 = np.percentile(dem_valid, 95)

pixel_size = 1  # 1 meter per pixel

prompt = (
    f"This is a grayscale digital elevation model (DEM) tile from a LIDAR survey. "
    f"Each pixel represents {pixel_size} meter. "
    f"The tile size is {width} x {height} meters "
    f"(covers {width} meters horizontally and {height} meters vertically). "
    f"The main elevation range for this tile is from {p5:.1f} to {p95:.1f} meters above sea level, excluding outliers.\n\n"
    "Important: The data comes from airborne LIDAR, and all vegetation (tree canopies) has been removed. "
    "You are seeing only the bare earth surface — ground terrain, not vegetation.\n\n"
    "Your tasks:\n"
    "1. Carefully analyze the terrain for hidden or subtle anthropogenic features. These may include old roads, ancient or abandoned earthworks, foundations, geometric clearings, linear ditches, burial mounds, or any human activity traces that may now be overgrown or buried.\n"
    "2. Pay special attention to subtle rectilinear or circular forms, patterns that do not match natural terrain, or straight lines that could indicate archaeological remains.\n"
    "3. If you notice any possible evidence of human activity, describe their location (relative to the tile), their possible nature, and your confidence.\n"
    "4. If you do NOT see any anthropogenic features, state this clearly.\n\n"
    "Do not invent features; be cautious. Your analysis should be based only on the visible relief.\n"
    
)

print(prompt)



tile_path = '/kaggle/working/tiles/tile_2_0.tif' #Choose  tile
img = Image.open(tile_path)

# Convert image to PNG in memory (lossless, recommended for LLM Vision API)
buffered = io.BytesIO()
img.save(buffered, format="PNG")
img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

response = openai.chat.completions.create(
    model="gpt-4.1-2025-04-14",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                        "detail": "high"
                    }
                }
            ]
        }
    ],
    max_tokens=512  # Можно добавить, чтобы ограничить длину ответа
)

print("AI description:\n", response.choices[0].message.content)


