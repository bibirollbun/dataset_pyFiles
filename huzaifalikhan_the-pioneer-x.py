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


import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu

# --- Step 1: Load DEM Data (as image or GeoTIFF) ---
with rasterio.open("/kaggle/input/paraxingu/para(xingu)-3.png") as src:
    dem = src.read(1).astype(float)
    profile = src.profile

# --- Step 2: Clean DEM ---
dem[dem <= 0] = np.nan
dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))  # Fill for filtering

# --- Step 3: Local Relief Modeling (LRM) ---
dem_smooth = gaussian_filter(dem_filled, sigma=15)
lrm = dem - dem_smooth  # LRM emphasizes micro-relief
lrm[np.isnan(lrm)] = 0  # Clean up any remaining NaNs

# --- Step 4: Binary Map from LRM ---
threshold = threshold_otsu(lrm)
binary_lrm = (lrm > threshold).astype(int)

# --- Step 5: Label Features and Compute Region Stats ---
labeled = label(binary_lrm)
regions = regionprops(labeled, intensity_image=lrm)

# --- Step 6: Utility Function for Site-Like Features ---
def utility(region):
    area_score = np.clip(region.area / 200.0, 0, 1)
    roundness_score = 1 - abs(1 - region.extent)
    elevation_score = np.clip(region.mean_intensity / 2.0, 0, 1)
    return area_score * 0.4 + roundness_score * 0.4 + elevation_score * 0.2

# --- Step 7: Score and Collect Candidates ---
candidates = []
for r in regions:
    if r.area > 10:  # Ignore tiny regions
        score = utility(r)
        if score > 0.5:
            candidates.append((r.centroid, score))

# --- Step 8: Plot Results ---
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(lrm, cmap='terrain')
for (y, x), score in candidates:
    ax.plot(x, y, 'ro')
    ax.text(x + 2, y, f"{score:.2f}", color='yellow', fontsize=8)
plt.title("LRM + Utility-Scored Archaeological Candidates")
plt.axis('off')
plt.tight_layout()
plt.show()


import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu

# --- Step 1: Load DEM Data (as image or GeoTIFF) ---
with rasterio.open("/kaggle/input/river-merego-basin/amazon basin(merego)-1.png") as src:
    dem = src.read(1).astype(float)
    profile = src.profile

# --- Step 2: Clean DEM ---
dem[dem <= 0] = np.nan
dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))  # Fill for filtering

# --- Step 3: Local Relief Modeling (LRM) ---
dem_smooth = gaussian_filter(dem_filled, sigma=15)
lrm = dem - dem_smooth  # LRM emphasizes micro-relief
lrm[np.isnan(lrm)] = 0  # Clean up any remaining NaNs

# --- Step 4: Binary Map from LRM ---
threshold = threshold_otsu(lrm)
binary_lrm = (lrm > threshold).astype(int)

# --- Step 5: Label Features and Compute Region Stats ---
labeled = label(binary_lrm)
regions = regionprops(labeled, intensity_image=lrm)

# --- Step 6: Utility Function for Site-Like Features ---
def utility(region):
    area_score = np.clip(region.area / 200.0, 0, 1)
    roundness_score = 1 - abs(1 - region.extent)
    elevation_score = np.clip(region.mean_intensity / 2.0, 0, 1)
    return area_score * 0.4 + roundness_score * 0.4 + elevation_score * 0.2

# --- Step 7: Score and Collect Candidates ---
candidates = []
for r in regions:
    if r.area > 10:  # Ignore tiny regions
        score = utility(r)
        if score > 0.5:
            candidates.append((r.centroid, score))

# --- Step 8: Plot Results ---
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(lrm, cmap='terrain')
for (y, x), score in candidates:
    ax.plot(x, y, 'ro')
    ax.text(x + 2, y, f"{score:.2f}", color='yellow', fontsize=8)
plt.title("LRM + Utility-Scored Archaeological Candidates")
plt.axis('off')
plt.tight_layout()
plt.show()

