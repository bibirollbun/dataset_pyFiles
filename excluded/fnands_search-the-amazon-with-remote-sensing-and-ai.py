!pip install contextily


import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from kaggle_secrets import UserSecretsClient

# Load your GeoJSON files
known_sites = gpd.read_file("/kaggle/input/archaeoblog-amazon-geoglyphs/geoglyph_points.geojson")
predicted_sites = gpd.read_file("/kaggle/input/major-tom-embedding-search/top_500_similar.geojson")

# Load boundary data (e.g., Amazon biome, country outlines, etc.)
boundaries = gpd.read_file("/kaggle/input/geographical-boundaries-of-amazonia-by-eva-et-al/amazonia_polygons.shp")

# Load Lidar locations
lidar = gpd.read_file("/kaggle/input/nasa-amazon-lidar-2008-2018/cms_brazil_lidar_tile_inventory.geojson")

# Reproject both to Web Mercator for contextily
known_sites = known_sites.to_crs(epsg=3857)
predicted_sites = predicted_sites.to_crs(epsg=3857)
boundaries = boundaries.to_crs(epsg=3857)
lidar = lidar.to_crs(epsg=3857)

# Plot the map with contextily basemap
fig, ax = plt.subplots(figsize=(12, 10))

# Plot boundary layer (e.g., Amazon region or political boundaries)
boundaries.plot(ax=ax, facecolor='none', edgecolor='purple', linewidth=2, label="Amazon Boundary")

# Plot your points/polygons
known_sites.plot(ax=ax, color='black', markersize=10, alpha=0.7, label="Known Pre-Columbian Sites")
predicted_sites.plot(ax=ax, color='blue', markersize=10, alpha=0.7, label="Predicted Pre-Columbian Sites")

# Plot Lidar locations
lidar.plot(ax=ax, color='red',  edgecolor='red', alpha=1.0, linewidth=2, label="NASA Lidar Locations")

# Add a basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Create custom legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Known Pre-Columbian Sites',
           markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Predicted Pre-Columbian Sites',
           markerfacecolor='blue', markersize=8),
    Patch(facecolor='none', edgecolor='purple', label='Amazon Boundary'),
    Patch(facecolor='none', edgecolor='red', label='NASA Lidar Locations')

]

ax.legend(handles=legend_elements, loc='upper right')

# Clean up
ax.set_axis_off()
plt.title("Pre-Columbian Sites in the Amazon Basin", fontsize=14)
plt.tight_layout()
plt.show()



# Step 2: Buffer lidar geometries by 5000 meters (5 km)
lidar_buffered = lidar.copy()
lidar_buffered["geometry"] = lidar_buffered.geometry.buffer(5000)

# Step 3: Spatial join â€” points within buffer
nearby_sites = gpd.sjoin(predicted_sites, lidar_buffered, how="inner", predicate="within")

# Step 4: Optionally drop duplicates or extra columns
nearby_sites = nearby_sites.drop(columns=['index_right'])

nearby_sites.head()


nearby_sites = nearby_sites.to_crs(epsg=3857)
# Plot the map with contextily basemap
fig, ax = plt.subplots(figsize=(12, 10))

# Plot boundary layer (e.g., Amazon region or political boundaries)
boundaries.plot(ax=ax, facecolor='none', edgecolor='purple', linewidth=2, label="Amazon Boundary")

# Plot your points/polygons
nearby_sites.plot(ax=ax, color='red', markersize=10, alpha=0.7, label="Candidate Sites")

# Add a basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Create custom legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Candidate Sites',
           markerfacecolor='red', markersize=8),
    Patch(facecolor='none', edgecolor='purple', label='Amazon Boundary'),

]

ax.legend(handles=legend_elements, loc='upper right')

# Clean up
ax.set_axis_off()
plt.title("Candidate Sites in the Amazon Basin", fontsize=14)
plt.tight_layout()
plt.show()


# Some inspiration taken from: https://www.kaggle.com/code/llkh0a/simple-dem-data-gpt-4o-mini-image-prompt
import io
import base64
import rasterio
import numpy as np
import os

# Base path to DTM files
base_dtm_path = '/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles'


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
    clean_arr = np.nan_to_num(array, nan=-1)
    plt.figure(figsize=(10, 6))
    plt.imshow(clean_arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight', dpi=150)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Main processing function
def process_lidar_tile(laz_filename):
    tif_filename = laz_filename.replace('.laz', '.tif')
    tif_path = os.path.join(base_dtm_path, tif_filename)

    with rasterio.open(tif_path) as src:
        dem = src.read(1)
        dem = np.where(dem == src.nodata, np.nan, dem)

    # Normalize DEM for color image
    vmin = np.nanpercentile(dem, 2)
    vmax = np.nanpercentile(dem, 98)
    dem_base64 = encode_image(dem, cmap='terrain', vmin=vmin, vmax=vmax)

    # Generate hillshade from filled DEM
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    hs = hillshade(dem_filled, azimuth=315, angle_altitude=45)
    hillshade_base64 = encode_image(hs, cmap='gray')

    return dem_base64, hillshade_base64


# Process the first tile as an example
dem_b64, hillshade_b64 = process_lidar_tile(nearby_sites['Name'].values[0])

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

# Fetch my OpenAI API key which has been stored as a Kaggle secret. 
user_secrets = UserSecretsClient()
OPENAI_KEY = user_secrets.get_secret("OPENAI_KEY")



client = OpenAI(api_key=OPENAI_KEY)
#CHOSEN_MODEL = "gpt-4o-mini"
CHOSEN_MODEL = "o3"


prompt = """

You are an expert in archaeology and remote sensing with a focus on identifying traces of ancient civilizations in elevation and terrain data.

You are being shown a hillshade image derived from LIDAR elevation data, as well as the DTM that the hillshade was derived from. 
The hillshade image emphasizes terrain features using simulated lighting and is intended to reveal subtle surface anomalies such as raised mounds, geometric depressions, terracing, causeways, or other patterns that may not be visible in standard imagery.

Your task:
Evaluate the hillshade image and DTM for **possible indicators of pre-Columbian human activity**, particularly features that may suggest the presence of:
- Ancient ruins (e.g., raised platforms, stone walls)
- Anthropogenic earthworks (e.g., mounds, canals, roads, plazas)
- Agricultural or settlement structures (e.g., terraces, rectangular clearings)

Be especially sensitive to **subtle geometric or organized features** that would be unlikely to occur from natural geomorphological processes alone.

Please be aware that the image might be buffered with NODATA at the edges. The most likely case is that there is no evidence of human activity.  

Return a JSON response like:

{
  "description": "Brief notes on visible features",
  "human_activity": "yes" or "no",
  "confidence": score from 0 to 100
}

Focus on signs like raised platforms, unnatural geometric shapes, mounds, or patterns consistent with ruins or agricultural modification.

Please do not add backticks around the json. The response will loaded in python with json.loads(), so please format accordingly. 

"""

results = []
for _, row in nearby_sites.head().iterrows():

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

    
    results.append((lidar_filename, json.loads(response.choices[0].message.content), dem_b64, hillshade_b64))
    



# Filter only those with likely human activity
filtered = [entry for entry in results if entry[1]['human_activity'] == 'yes']

# Fallback in case no human activity is found
if len(filtered) == 0:
    filtered = results

# Sort by confidence (highest first)
ranked = sorted(filtered, key=lambda x: x[1]['confidence'], reverse=True)

# Get top-ranked result(s)
top_confidence = ranked[0][1]['confidence']
top_results = [r for r in ranked if r[1]['confidence'] == top_confidence]

# Display
filename, data, dem_b64, hillshade_b64 =  top_results[0]
print(f"\nTop result: {filename}")
print(f"Confidence: {data['confidence']}")
print(f"Human Activity Likely: {data['human_activity']}")
print(f"Description: {data['description']}")

HTML(f"""
<h3>DTM (Terrain Color)</h3>
<img src="data:image/jpeg;base64,{dem_b64}"/>
<h3>Hillshade</h3>
<img src="data:image/jpeg;base64,{hillshade_b64}"/>
""")


import kaggle_session

client = kaggle_session.UserSessionClient()
notebook = client.get_exportable_ipynb()['source']

with open("kaggle.ipynb", "w") as f:
    f.write(notebook)



!jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to markdown kaggle.ipynb


with open("kaggle.md", "r", encoding="utf-8") as f:
    notebook_text = f.read()


prompt = """
You are an expert in remote sensing, digital terrain modeling (DTM), and Mesoamerican archaeology.

Below is a Jupyter notebook containing my code and approach for analyzing airborne LIDAR hillshade data. My objective is to detect signs of ancient (pre-Columbian) human activity in the Amazon â€” including mounds, causeways, terraces, or settlement structures â€” based on elevation anomalies.

Please review my approach critically and do the following:

1. Identify any strengths or sound techniques I am using.
2. Point out any flaws, oversights, or areas where the methodology could be improved.
3. Suggest specific analytical or archaeological techniques (e.g., feature extraction, spatial clustering, topographic indices) that I could incorporate to improve detection of anthropogenic features.
4. If relevant, suggest how machine learning or statistical methods could be applied to help identify ancient human settlement patterns from LIDAR data.
5. Suggest any other improvements or techniques that have not been considered. 
6. Suggest any known literature that might guide future work. 

Be concise but thorough. 

Notebook:

"""

full_prompt = prompt + "\n\n" + notebook_text

client = OpenAI(api_key=OPENAI_KEY)

response = client.chat.completions.create(
    model=CHOSEN_MODEL,
    messages=[
        {"role": "system", "content": "You are an expert in archaeology and remote sensing."},
        {"role": "user", "content": full_prompt}
    ]
)



# From: https://discourse.jupyter.org/t/how-to-add-markdown-inside-a-function-to-pretty-print-equation/2719/2
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))


printmd(response.choices[0].message.content)


print(f"The model used in this notebook was {response.model}")

