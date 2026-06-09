import numpy as np
import matplotlib.pyplot as plt


!pip install rasterio --quiet


import rasterio


from pyproj import CRS, Transformer


from matplotlib.colors import Normalize


def load_raster(file_path):
    with rasterio.open(file_path) as src:
        
        data = src.read(1)
        nodata = src.nodata
        
        profile = src.profile
        bounds = src.bounds
        crs = src.crs
        transform = src.transform
        
        return data, nodata, profile, bounds, crs, transform


def summarize_band(data, nodata):
    band = data.copy()
    # Remove nodata, flattened to 1D
    band = band[band != nodata]
    band_clean = band[np.isfinite(band)]
    return {
        "min": float(np.min(band)),
        "max": float(np.max(band)),
        "mean": float(np.mean(band)),
        "std": float(np.std(band))
    }


def get_coords(bounds, crs):
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    coords = {
        "top_left": transformer.transform(bounds.left, bounds.top),
        "bottom_right": transformer.transform(bounds.right, bounds.bottom)
    }
    return coords


# Visualization
def visualize_img(data_o, nodata, img_title, cmap):
    
    data = data_o.copy()
        
    # Replace nodata values with NaN, retain 2D
    data[data == nodata] = np.nan
    
    # Compute valid elevation range
    valid = data[~np.isnan(data)]
    vmin, vmax = np.percentile(valid, [2, 98])
    
    # Clip and fill NaNs with the minimum valid value
    clipped = np.clip(data, vmin, vmax)
    clipped_filled = np.nan_to_num(clipped, nan=vmin)
    
    # Normalize explicitly
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Final plot using normalized, fully numeric array
    plt.figure(figsize=(8, 5))
    im = plt.imshow(clipped_filled, cmap=cmap, norm=norm)
    
    cbar = plt.colorbar(im)
    cbar.set_label("Elevation (m)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.title(img_title,fontsize=16, fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print("vmin",vmin,"\nvmax",vmax)


file_be = "/kaggle/input/1-open-topography-lidar-tile/output_be.tif"
file_vh = "/kaggle/input/1-open-topography-lidar-tile/output_vh.tif"


# Load both rasters
output_be, nodata_be, profile_be, bounds_be, crs_be, transform_be = load_raster(file_be)
output_vh, nodata_vh, profile_vh, bounds_vh, crs_vh, transform_vh = load_raster(file_vh)


print(profile_be, '\n\n\n', bounds_be, '\n\n\n', crs_be, '\n\n\n', transform_be)


visualize_img(output_be, nodata_be,
              "LiDar Tile - Bare Earth (BE) : Digital Terrain Model (DTM)\n", 
              'terrain')


visualize_img(output_vh, nodata_vh,
              "LiDAR Tile - Vegetation Height (VH) : Canopy Height Model (CHM)\n", 
              'Greens')


# Summarize both bands
summary_be = summarize_band(output_be, nodata_be)
summary_vh = summarize_band(output_vh, nodata_vh)


print(summary_be, '\n\n\n', summary_vh)


# Extract coordinates
coords = get_coords(bounds_be, crs_be)


coords


# To Build prompt
crs_desc = CRS(crs_be).to_string()
x_res = transform_be.a
y_res = abs(transform_be.e)


print(crs_desc, '\n\n\n', x_res,'\n\n\n', y_res)


prompt = (
    f"This is approx 10 km x 10 km terrain tile with two data layers.\n"
    f"CRS: {crs_desc}\n"
    f"Resolution: {x_res:.1f} x {y_res:.1f} meters per pixel\n"
    f"Approximate location: top-left ({coords['top_left'][1]:.4f}°, {coords['top_left'][0]:.4f}°), "
    f"bottom-right ({coords['bottom_right'][1]:.4f}°, {coords['bottom_right'][0]:.4f}°)\n\n"
    f"**Band: Elevation (output_be)**\n"
    f"Min: {summary_be['min']:.2f} m, Max: {summary_be['max']:.2f} m\n"
    f"Mean: {summary_be['mean']:.2f} m, Std Dev: {summary_be['std']:.2f} m\n\n"
    f"**Band: VH Backscatter (output_vh)**\n"
    f"Min: {summary_vh['min']:.2f}, Max: {summary_vh['max']:.2f}\n"
    f"Mean: {summary_vh['mean']:.2f}, Std Dev: {summary_vh['std']:.2f}\n\n"
    "Describe the surface characteristics, likely vegetation cover, and terrain features in plain English."
    "What can you infer about any **potential archaeological features** from the given information"
)


import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient


secret = UserSecretsClient()
openai_key = secret.get_secret("arch-bot")


client = OpenAI(
  api_key=openai_key
)


response = client.chat.completions.create(
  model="o4-mini",
  messages=[{"role": "user", "content": prompt}]
)


print(response.choices[0].message.content)


print("Model version:", response.model)
print("Dataset ID: \n1-open-topography-lidar-tile/output_be.tif ",
"\n1-open-topography-lidar-tile/output_vh.tif")

