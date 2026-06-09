!pip install contextily


import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from kaggle_secrets import UserSecretsClient

# Load boundary data (e.g., Amazon biome, country outlines, etc.)
boundaries = gpd.read_file("/kaggle/input/geographical-boundaries-of-amazonia-by-eva-et-al/amazonia_polygons.shp")

# Load Lidar locations
lidar = gpd.read_file("/kaggle/input/nasa-amazon-lidar-2008-2018/cms_brazil_lidar_tile_inventory.geojson")

# Reproject both to Web Mercator for contextily
boundaries = boundaries.to_crs(epsg=3857)
lidar = lidar.to_crs(epsg=3857)

# Plot the map with contextily basemap
fig, ax = plt.subplots(figsize=(12, 10))

# Plot boundary layer (e.g., Amazon region or political boundaries)
boundaries.plot(ax=ax, facecolor='none', edgecolor='purple', linewidth=2, label="Amazon Boundary")

# Plot Lidar locations
lidar.plot(ax=ax, color='red',  edgecolor='red', alpha=1.0, linewidth=2, label="NASA Lidar Locations")

# Add a basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Create custom legend
legend_elements = [
    Patch(facecolor='none', edgecolor='purple', label='Amazon Boundary'),
    Patch(facecolor='none', edgecolor='red', label='NASA Lidar Locations')

]

ax.legend(handles=legend_elements, loc='upper right')

# Clean up
ax.set_axis_off()
plt.title("Lidar Sites in the Amazon Basin", fontsize=14)
plt.tight_layout()
plt.show()



from glob import glob
from pathlib import Path

redape_lidar_files  = glob("/kaggle/input/2021-redape-lidar/*tif")
redape_lidar_names = [f"{Path(filename).stem}.laz" for filename in redape_lidar_files]




redape_lidar = gpd.GeoDataFrame()
redape_lidar['Name'] = redape_lidar_names


# Some inspiration taken from: https://www.kaggle.com/code/llkh0a/simple-dem-data-gpt-4o-mini-image-prompt
import io
import base64
import rasterio
import numpy as np
import os
from matplotlib import colormaps  # new interface
# Base path to DTM files


import warnings, numpy as np

# Ignore the “invalid value encountered in less” that Matplotlib fires internally
warnings.filterwarnings("ignore",
                        category=RuntimeWarning,
                        module="matplotlib.colors")

# Taken from https://www.neonscience.org/resources/learning-hub/tutorials/create-hillshade-py
# Hillshade calculation
def hillshade(array, azimuth, angle_altitude):
    azimuth = 360.0 - azimuth
    x, y = np.gradient(array)
    slope = np.pi / 2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azm_rad = azimuth * np.pi / 180.
    alt_rad = angle_altitude * np.pi / 180.
    shaded = np.sin(alt_rad) * np.sin(slope) + np.cos(alt_rad) * np.cos(slope) * np.cos((azm_rad - np.pi/2.) - aspect)
    return 255 * (shaded + 1) / 2

# Helper to plot and encode to JPEG base64
def encode_image(array, cmap='terrain', vmin=None, vmax=None):
    """Return a base-64 JPEG with NaNs shown as black."""
    masked = np.ma.masked_invalid(array)          # keep NaNs
    cmap_obj = colormaps[cmap].copy()
    cmap_obj.set_bad('black')                     # draw masked pixels black

    plt.figure(figsize=(10, 6))
    plt.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight', dpi=150)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Main processing function
def process_lidar_tile(laz_filename, base_dtm_path = '/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles'):
    tif_filename = laz_filename.replace('.laz', '.tif')
    tif_path = os.path.join(base_dtm_path, tif_filename)

    with rasterio.open(tif_path) as src:
        dem = src.read(1)
        dem = np.where(dem == src.nodata, np.nan, dem)

    # ――― Terrain-colour DEM ―――
    vmin = np.nanpercentile(dem, 2)
    vmax = np.nanpercentile(dem, 98)
    dem_b64 = encode_image(dem, cmap='terrain', vmin=vmin, vmax=vmax)

    # ――― Hillshade ―――
    mask = np.isnan(dem)                          # remember nodata cells
    dem_filled = np.where(mask, np.nanmean(dem), dem)

    hs = hillshade(dem_filled, azimuth=315, angle_altitude=45)
    hs[mask] = np.nan                             # put the holes back

    hillshade_b64 = encode_image(hs, cmap='gray') # NaNs now render as black

    return dem_b64, hillshade_b64


# Process the first tile as an example
dem_b64, hillshade_b64 = process_lidar_tile(lidar['Name'].values[0])

# Optional: Display inline in notebook
from IPython.display import HTML
HTML(f"""
<h3>DTM (Terrain Color)</h3>
<img src="data:image/jpeg;base64,{dem_b64}"/>
<h3>Hillshade</h3>
<img src="data:image/jpeg;base64,{hillshade_b64}"/>
""")


from openai import OpenAI
import json
from tqdm import tqdm

# Fetch my OpenAI API key which has been stored as a Kaggle secret. 
user_secrets = UserSecretsClient()
OPENAI_KEY = user_secrets.get_secret("OPENAI_KEY")



lidar_example = lidar[lidar['Name'] == 'RIB_A01_2014_laz_2.laz']


client = OpenAI(api_key=OPENAI_KEY)
CHOSEN_MODEL = "o3-2025-04-16"



prompt = """

You are an expert in archaeology and remote sensing with a focus on identifying traces of ancient civilizations in elevation and terrain data.

You are being shown a false-color DTM. 

Your task:
Evaluate the DTM for **possible indicators of pre-Columbian human activity**, particularly features that may suggest the presence of:
- Ancient ruins (e.g., raised platforms, stone walls)
- Anthropogenic earthworks (e.g., mounds, canals, roads, plazas)
- Agricultural or settlement structures (e.g., terraces, rectangular clearings)

Be especially sensitive to **subtle geometric or organized features** that would be unlikely to occur from natural geomorphological processes alone.

Please be aware that the image might be buffered with NODATA at the edges. NODATA is encoded as black. 

If there is evidence of human activity, try and evaluate whether or not it is ancient or modern. 

Return a JSON response like:

{
  "description": "Brief notes on visible features",
  "human_activity": "yes" or "no",
  "ancient": "yes" or "no"
}

Focus on signs like raised platforms, unnatural geometric shapes, mounds, or patterns consistent with ruins or agricultural modification.

Please do not add backticks around the json. The response will loaded in python with json.loads(), so please format accordingly. 

Please answer the "ancient" question in all cases. In case there is no evidence of human activity, just answer "no". 
"""

example_results = []

lidar_filename = lidar_example['Name'].values[0]
dem_b64_example, hillshade_b64_example = process_lidar_tile(lidar_filename)


response = client.chat.completions.create(
    model=CHOSEN_MODEL,
    messages=[
        {"role": "user", "content": prompt},
        {
            "role": "user",
            "content": [
                
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{hillshade_b64_example}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{dem_b64_example}"}}
                    
                
            ]
        }
    ]
)


example_results.append((lidar_filename, json.loads(response.choices[0].message.content), dem_b64_example, hillshade_b64_example))
    



for name, d, _, _ in example_results: 
    print(d['description'])
    print(d['human_activity'])
    print(d['ancient'])

    if not d['ancient'] == 'yes':
        assert(False)


# Display
filename, data, dem_b64, hillshade_b64 =  example_results[0]
print(f"\nTop result: {filename}")
print(f"Human Activity Likely: {data['human_activity']}")
print(f"Ancient: {data['ancient']}")
print(f"Description: {data['description']}")

HTML(f"""
<h3>DTM (Terrain Color)</h3>
<img src="data:image/jpeg;base64,{dem_b64}"/>
<h3>Hillshade</h3>
<img src="data:image/jpeg;base64,{hillshade_b64}"/>
""")


#lidar = lidar.head(5).copy()


results = []

description = []
human_activity = []
ancient = []
for _, row in tqdm(lidar.iterrows()):

    lidar_filename = row['Name']
    dem_b64, hillshade_b64 = process_lidar_tile(lidar_filename)
    
    
    response = client.chat.completions.create(
        model=CHOSEN_MODEL,
        messages=[
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": [
                    
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{hillshade_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{dem_b64}"}}
                        
                    
                ]
            }
        ]
    )

    prediction = json.loads(response.choices[0].message.content)
    results.append((lidar_filename, prediction))#, dem_b64, hillshade_b64))

    description.append(prediction['description'])
    human_activity.append(prediction['human_activity'])
    ancient.append(prediction['ancient'])


lidar.loc[:, 'description'] = description
lidar.loc[:, 'human_activity'] = human_activity
lidar.loc[:, 'ancient']  = ancient


lidar.to_file("predictions_lidar.geojson", driver='GeoJSON')


#redape_lidar = redape_lidar.head(5).copy()


dtm_path = "/kaggle/input/2021-redape-lidar"

results = []

description = []
human_activity = []
ancient = []
for _, row in tqdm(redape_lidar.head().iterrows()):

    lidar_filename = row['Name']
    dem_b64, hillshade_b64 = process_lidar_tile(lidar_filename, dtm_path)
    
    
    response = client.chat.completions.create(
        model=CHOSEN_MODEL,
        messages=[
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": [
                    
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{hillshade_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{dem_b64}"}}
                        
                    
                ]
            }
        ]
    )

    prediction = json.loads(response.choices[0].message.content)
    results.append((lidar_filename, prediction))#, dem_b64, hillshade_b64))

    description.append(prediction['description'])
    human_activity.append(prediction['human_activity'])
    ancient.append(prediction['ancient'])


redape_lidar.loc[:, 'description'] = description
redape_lidar.loc[:, 'human_activity'] = human_activity
redape_lidar.loc[:, 'ancient']  = ancient


redape_lidar.to_csv("predictions_redape_lidar.csv")


print(f"The model used in this notebook was {response.model}")

