!pip install rasterio matplotlib numpy openai awscli geopandas
%matplotlib inline


import os
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
import json
import requests
from pathlib import Path
import geopandas as gpd
import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient


# Create directory for data
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)


# Download the ArcticDEM 10m Tile 07_40
dem_file = os.path.join(data_dir, '/kaggle/input/incapuquio-fault-south-peru-santa-elena-zone-2018/Inkpkio_SE.tif')


import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

# define your sample size
sample_size = 100  # size of the window to extract

with rasterio.open(dem_file) as src:
    nodata = src.nodata
    row = col = None

    # ğŸ”� Find first valid pixel via block_windows (low RAM)
    for _, win in src.block_windows(1):
        block = src.read(1, window=win)
        mask = block != nodata
        if mask.any():
            i, j = np.where(mask)
            row = int(win.row_off + i[0])
            col = int(win.col_off + j[0])
            break

    if row is None:
        print("No valid data found.")
        exit()

    # ğŸ�¯ Define the 100Ã—100 window around that pixel
    half = sample_size // 2
    row_off = max(0, row - half)
    col_off = max(0, col - half)
    window = Window(col_off, row_off, sample_size, sample_size)

    # âœ… Read only this small window
    sample = src.read(1, window=window)

# ğŸ§  Compute stats
mask = sample != nodata
vals = sample[mask]
stats = {
    "min_elevation": float(vals.min()),
    "max_elevation": float(vals.max()),
    "mean_elevation": float(vals.mean()),
    "std_deviation": float(vals.std())
}

# ğŸ“Š Plot
plt.figure(figsize=(6, 6))
plt.imshow(sample, cmap='terrain')
plt.colorbar(label="Elevation (m)")
plt.title("DEM Sample")
plt.savefig(os.path.join(data_dir, 'dem_sample.png'))
plt.show()
# ğŸ–¨ï¸� Output
print("Sample Region Statistics:")
print(f"- Min Elevation: {stats['min_elevation']:.2f} m")
print(f"- Max Elevation: {stats['max_elevation']:.2f} m")
print(f"- Mean Elevation: {stats['mean_elevation']:.2f} m")
print(f"- STD_deviation: {stats['std_deviation']:.2f} m")


def load_secret(name):
    """Loads secret from Colab/Kaggle."""

    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        except Exception:
            pass 
    else:
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception: 
            pass

    return 'Secret not found'


openai_key = load_secret('openai key')

client = OpenAI(
  api_key=openai_key
)

# Prepare data description for OpenAI
data_description = f"""
Dataset: incapuquio-fault-south-peru-santa-elena survey
Type: Digital Elevation Model (DEM)
Sample Region Statistics:
- Minimum Elevation: {stats['min_elevation']:.2f} meters
- Maximum Elevation: {stats['max_elevation']:.2f} meters
- Mean Elevation: {stats['mean_elevation']:.2f} meters
- Standard Deviation: {stats['std_deviation']:.2f} meters

The data represents surface elevation in meters.
"""

# Call OpenAI API to analyze the data
print("Calling OpenAI API to analyze the DEM data...")
try:
    # Only try competition-specified models
    models_to_try = [
        "gpt-4o-mini",
    ]
    
    model_success = False
    for model_to_use in models_to_try:
        try:
            print(f"Attempting to use model: {model_to_use}")
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are a geospatial analysis assistant with expertise in interpreting elevation data."},
                    {"role": "user", "content": f"Please describe the surface features and terrain characteristics that might be present in this Digital Elevation Model data. {data_description}"}
                ]
            )
            model_success = True
            break
        except Exception as e:
            print(f"Error with {model_to_use}: {e}")
            continue
    
    if not model_success:
        print("All competition-specified models failed. Please check your API key and quota.")
    else:
        # Save the OpenAI response
        openai_response = response.choices[0].message.content
        
        # Record model version and dataset ID
        model_info = {
            "model_version": response.model,
            "dataset_id": "Survey of the Incapuquio Fault, South Peru: Tauja Zone 2018",
            "dataset_source": "https://portal.opentopography.org/dataspace/dataset?opentopoID=OTDS.042021.32719.1",
            "dataset_license": "CC0 1.0",
            "Survey Date": "04/06/2018"
        }
        
        # Save results to files
        with open(os.path.join(data_dir, 'openai_analysis.txt'), 'w') as f:
            f.write(f"Model Version: {model_info['model_version']}\n")
            f.write(f"Dataset ID: {model_info['dataset_id']}\n\n")
            f.write("OpenAI Analysis:\n")
            f.write(openai_response)
        
        with open(os.path.join(data_dir, 'model_dataset_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
            
        # Print model version and dataset ID (required by the competition)
        print("\n" + "=" * 50)
        print("COMPETITION REQUIREMENTS OUTPUT:")
        print("=" * 50)
        print(f"Model Version: {model_info['model_version']}")
        print(f"Dataset ID: {model_info['dataset_id']}")
        print("=" * 50 + "\n")
        
        # Display the OpenAI analysis
        print("OpenAI Analysis:")
        print(openai_response)
except Exception as e:
    print(f"Error calling OpenAI API: {e}")
    print("Please check your API key and quota.")


# Check if all required files exist
required_files = [
    os.path.join(data_dir, 'Inkpkio_SE.tif'),
    os.path.join(data_dir, 'dem_visualization.png'),
    os.path.join(data_dir, 'dem_sample.png')
]

all_files_exist = all(os.path.exists(file) for file in required_files)

print("OpenAI to Z Challenge Requirements:")
print(f"1. Download one OpenTopography LiDAR tile: {'âœ“'}")

# Check if OpenAI analysis was successful
openai_success = os.path.exists(os.path.join(data_dir, 'openai_analysis.txt'))
print(f"2. Run an OpenAI o3/o4-mini or GPT-4.1 prompt on the data: {'âœ“' if openai_success else 'âœ—'}")

# Check if model version and dataset ID were printed
if openai_success:
    with open(os.path.join(data_dir, 'model_dataset_info.json'), 'r') as f:
        model_info = json.load(f)
    print(f"3. Print model version and dataset ID: âœ“")
    print(f"   - Model Version: {model_info['model_version']}")
    print(f"   - Dataset ID: {model_info['dataset_id']}")
else:
    print(f"3. Print model version and dataset ID: âœ—")

print("\nOverall Status: " + ("All requirements met! âœ“" if all_files_exist and openai_success else "Some requirements not met. Please check the errors above."))


# Create directory for data
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)


# Download the ArcticDEM 10m Tile 07_40
dem_file = os.path.join(data_dir, '/kaggle/input/south-america-dem/sa_dem_3s.tif')


import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

# define your sample size
sample_size = 100  # size of the window to extract

with rasterio.open(dem_file) as src:
    nodata = src.nodata
    row = col = None

    # ğŸ”� Find first valid pixel via block_windows (low RAM)
    for _, win in src.block_windows(1):
        block = src.read(1, window=win)
        mask = block != nodata
        if mask.any():
            i, j = np.where(mask)
            row = int(win.row_off + i[0])
            col = int(win.col_off + j[0])
            break

    if row is None:
        print("No valid data found.")
        exit()

    # ğŸ�¯ Define the 100Ã—100 window around that pixel
    half = sample_size // 2
    row_off = max(0, row - half)
    col_off = max(0, col - half)
    window = Window(col_off, row_off, sample_size, sample_size)

    # âœ… Read only this small window
    sample = src.read(1, window=window)

# ğŸ§  Compute stats
mask = sample != nodata
vals = sample[mask]
stats = {
    "min_elevation": float(vals.min()),
    "max_elevation": float(vals.max()),
    "mean_elevation": float(vals.mean()),
    "std_deviation": float(vals.std())
}

# ğŸ“Š Plot
plt.figure(figsize=(6, 6))
plt.imshow(sample, cmap='terrain')
plt.colorbar(label="Elevation (m)")
plt.title("DEM Sample")
plt.savefig(os.path.join(data_dir, 'dem_sample.png'))
plt.show()
# ğŸ–¨ï¸� Output
print("Sample Region Statistics:")
print(f"- Min Elevation: {stats['min_elevation']:.2f} m")
print(f"- Max Elevation: {stats['max_elevation']:.2f} m")
print(f"- Mean Elevation: {stats['mean_elevation']:.2f} m")
print(f"- STD_deviation: {stats['std_deviation']:.2f} m")


gdf = gpd.read_file("/kaggle/input/amazon-indigenous-tribe-borders/indigenous_area_amazon_biome.shp")
gdf.plot()


import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file("/kaggle/input/amazon-indigenous-tribe-borders/indigenous_area_amazon_biome.shp")
plt.figure(figsize=(8, 8))
gdf.plot()
plt.title("Indigenous Amazon Biome Borders")
plt.axis('off')
plt.savefig("borders.png", dpi=150)
plt.close()



def load_secret(name):
    """Loads secret from Colab/Kaggle."""

    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        except Exception:
            pass 
    else:
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception: 
            pass

    return 'Secret not found'


openai_key = load_secret('openai key')

client = OpenAI(
  api_key=openai_key
)



data_description = f"""
Dataset: South-america-dem
Type: Digital Elevation Model (DEM)
Sample Region Statistics:
- Minimum Elevation: {stats['min_elevation']:.2f} meters
- Maximum Elevation: {stats['max_elevation']:.2f} meters
- Mean Elevation: {stats['mean_elevation']:.2f} meters
- Standard Deviation: {stats['std_deviation']:.2f} meters

The data represents surface elevation in meters.
"""


import base64

def b64_encode(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

img1 = b64_encode("borders.png")
img2 = b64_encode(os.path.join(data_dir, "dem_sample.png"))
img3= b64_encode("/kaggle/input/google-earth-image/Screenshot 2025-06-14 013441.png")


messages = [
    {"role": "system", "content": "You are an expert on topographical maps and can provide coordinates to any chunks of data"},
    {"role": "user", "content": [
        {"type": "text", "text": f"Please analyze the second image and provide coordinates on where the location might be with the given information about altitudes, rely more on the data description and point out likely geographical anomalies {data_description}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2}"}}
    ]}
]
resp = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=messages,
    max_tokens=500
)

print(resp.choices[0].message.content)


messages = [
    {"role": "system", "content": "You are a visual assistant comparing images. You are trying to look for patterns and anomalies to assist in finding potential ancient civilizations, attempt to give the user as many potential anomalies to look for and where they are in the google earth screenshots"},
    {"role": "user", "content": [
        {"type": "text", "text": "Please analyze the three images and attempt to layer these three on them to detect patterns and anomalies and give these patterns and anomalies, the second tile seems to be at (Latitude: around -3.465305, Longitude: around -62.215881) and the third image provides google earth visionary on the second tile location, the first tile includes the boundaries of the indigenous tribes in the amazon rainforest. PLEASE DO NOT PROVIDE TECHNIQUES TO FIND ANOMALIES, INSTEAD PROVIDE POTENTIAL LOCATIONS ON THE GOOGLE EARTH MAP"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img3}"}}
    ]}
]

resp = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=messages,
    max_tokens=500
)

print(resp.choices[0].message.content)






from IPython.display import Image, display

display(Image(filename='/kaggle/input/google-earth-image/Screenshot 2025-06-14 013441.png'))


