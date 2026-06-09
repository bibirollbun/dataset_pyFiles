!pip install rasterio


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()


import base64
import urllib.request
import io
from PIL import Image
import matplotlib.pyplot as plt
from openai import OpenAI
import numpy as np
import rasterio
import os
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = user_secrets.get_secret("open_ai_key")
OT_api_key=user_secrets.get_secret("OT_api_key")


import requests

def download_lidar_tile(demtype, south, north, west, east, output_file,api_key=OT_api_key):
    """
    Downloads a LiDAR tile from OpenTopography using the specified parameters.
    """
    # Define the OpenTopography API endpoint
    BASE_URL = "https://portal.opentopography.org/API/globaldem"

    # Parameters for the API request
    params = {
        "demtype": demtype,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key
    }

    # Make the API request
    response = requests.get(BASE_URL, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the downloaded file
        with open(output_file, "wb") as file:
            file.write(response.content)
        print(f"LiDAR data downloaded successfully: {output_file}")
    else:
        print(f"Failed to download data. Status code: {response.status_code}")
        print("Response:", response.text)


# Example usage of the function
download_lidar_tile(
                    demtype='COP90',
                    south=-4.919116467989866,
                    north=-0.7591338330879864,
                    west=-69.26220683008432,
                    east=-57.18603347986936,
                    output_file='lidar_tile.tif')


with rasterio.open('lidar_tile.tif') as src:
    dem = src.read(1)
    plt.figure(figsize=(10, 6))
    plt.imshow(dem, cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title('Downloaded DEM (LiDAR) Data')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()
    # Step 2: Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='JPEG', bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    buf.seek(0)



# Load DEM data
with rasterio.open('lidar_tile.tif') as src:
    dem = src.read(1)
    dem = np.where(dem == src.nodata, np.nan, dem)  # Mask nodata if any

# Normalize for visualization
vmin = np.nanpercentile(dem, 2)
vmax = np.nanpercentile(dem, 98)

# Plot with better visibility
plt.figure(figsize=(10, 6))
plt.imshow(dem, cmap='terrain', vmin=vmin, vmax=vmax)
plt.axis('off')  # Remove axis for cleaner image
plt.tight_layout()

# Save plot to buffer
buf = io.BytesIO()
plt.savefig(buf, format='jpeg', bbox_inches='tight', dpi=150)
plt.close()
buf.seek(0)

# Convert to base64
img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')


client = OpenAI()
CHOSEN_MODEL = "gpt-4.1"

prompt = "Analyze this Digital Elevation Model (DEM) image and describe the terrain and surface features in plain English. Identify features such as valleys, rivers, mountains, flatlands, and changes in elevation.."

response = client.chat.completions.create(
    model=CHOSEN_MODEL,
    messages=[
        {"role": "user", "content": prompt},
        {
            "role": "user",
            "content": [
                
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                
            ]
        }
    ]
)

print(response.choices[0].message.content)


