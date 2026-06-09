# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os

for root, _, files in os.walk('/kaggle/input'):
    for f in files:
        if f.endswith('.tif'):
            print(os.path.join(root, f))



!pip install rasterio --quiet


import rasterio
import numpy as np

# Replace this path with the exact path you just found
tif_path = '/kaggle/input/amazon-basin-dem/SA_srtm_mosaic_30arcsec_reg_hgt.tif'

with rasterio.open(tif_path) as src:
    elevation = src.read(1)  # Read the first band (elevation data)
    elevation = elevation[elevation != src.nodata]  # Remove nodata values

# Calculate elevation statistics
print("Min Elevation:", np.min(elevation))
print("Max Elevation:", np.max(elevation))
print("Mean Elevation:", np.mean(elevation))



import rasterio
import numpy as np
import matplotlib.pyplot as plt

# File path
tif_path = "/kaggle/input/amazon-basin-dem/SA_srtm_mosaic_30arcsec_reg_hgt.tif"

with rasterio.open(tif_path) as src:
    elevation = src.read(1)
    elevation = np.where(elevation == src.nodata, np.nan, elevation)

# Mask NaNs for visualization
masked = np.ma.masked_invalid(elevation)

# Plot with mask
plt.figure(figsize=(10, 6))
plt.imshow(masked, cmap='terrain')
plt.title("Amazon Basin Elevation Map")
plt.colorbar(label='Elevation (m)')
plt.axis('off')
plt.show()


import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient


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


from kaggle_secrets import UserSecretsClient

secret = UserSecretsClient()
openai_key = secret.get_secret("OPEN_API_KEY")


client = OpenAI(
  api_key=openai_key
)

prompt = "Produce a detailed plan for a research scientist and provide recommendations about how and where they could use GPT-4o to analyze multi-spectral satellite imagery with the goal of discovering evidence of ancient civilizations in Brazil."

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message.content);




from kaggle_secrets import UserSecretsClient

secret = UserSecretsClient()
openai_key = secret.get_secret("OPEN_API_KEY")


client = OpenAI(
  api_key=openai_key
)

prompt = "Describe the terrain in plain English based on this elevation summary: min = 0m, max = 6540m, mean = 538m. What might this imply about the region's topography in Brazil?"

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message.content)
print("\nModel: gpt-4o-mini")
print("Dataset ID: amazon-basin-dem/SA_srtm_mosaic_30arcsec_reg_hgt.tif")


