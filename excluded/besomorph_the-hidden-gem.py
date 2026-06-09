!pip install rasterio


import geopandas as gpd
import folium
import pandas as pd

# Load the dataset with all the Amazon archaeological sites
# (it's a CSV, so pretty straightforward)
data = pd.read_csv("/kaggle/input/amazon-sites-1/full_amazon_archaeology_sites.csv")

# Create a GeoDataFrame using the longitude and latitude from the data
# Hopefully the coordinate reference system is WGS84...
geo_sites = gpd.GeoDataFrame(
    data,
    geometry = gpd.points_from_xy(data.Longitude, data.Latitude),
    crs ="EPSG:4326"
)

# Load the shapefile for the Amazon basin region
# (I had to look up what 'sensulatissimo' means, lol)
amazon_shape_path = "/kaggle/input/amazonbasinlimits/amazon_sensulatissimo_gmm_v1.shp"
amazon_boundary = gpd.read_file(amazon_shape_path).to_crs("EPSG:4326")

# Filter the sites that are located within the Amazon basin
# spatial join does the job
sites_within_amazon = gpd.sjoin(
    geo_sites,
    amazon_boundary,
    predicate="within"
).drop(columns="index_right")

# Make the base map centered roughly on Amazon region
# Map style is one I like because it's clean
map_ = folium.Map(
    location=[-5, -63],  # kinda middle of Amazon area
    zoom_start=5,
    tiles="CartoDB positron"
)

# Add the Amazon basin geometry to the map (as a translucent green shape)
folium.GeoJson(
    amazon_boundary.geometry,
    name="Amazon Basin",
    style_function=lambda x: {
        "fillColor": "#006400",  # dark green
        "color": "#004400",      # even darker outline
        "weight": 2,
        "fillOpacity": 0.1
    }
).add_to(map_)

# Go through each site and drop a marker with some info
for _, site in sites_within_amazon.iterrows():
    # Making a popup with some basic info about the site
    popup_text = f"""
    <b>{site['Site']}</b><br>
    <b>Detection:</b> {site['Detection Method']}<br>
    <b>Year:</b> {site['Year']}<br>
    <b>Description:</b> {site['Description']}<br>
    """
    
    # Add the marker to the map
    folium.Marker(
        location=[site.geometry.y, site.geometry.x],  # y is lat, x is lon (weird!)
        popup=folium.Popup(popup_text, max_width = 400),
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(map_)

# Optional layers (right now just the Amazon Basin layer)
folium.LayerControl().add_to(map_)

# Display the map (Jupyter or similar will render this)
map_



from IPython.display import display, HTML
import matplotlib.pyplot as plt
import base64
import numpy as np
import io
import os
import rasterio

my_data_folder = "/kaggle/input/sentinel-2-ndvi-2024-dataset"

# Get list of .tif files
tif_files_list = []
for root, _, files in os.walk(my_data_folder):
    for f in files:
        if f.lower().endswith(('.tif', '.tiff')):
            tif_files_list.append(os.path.join(root, f))

tif_files_list = sorted(tif_files_list)

num_imgs = 3
html_parts = []

for i in range(num_imgs):
    tif_file = tif_files_list[i]
    
    with rasterio.open(tif_file) as src:
        img_arr = src.read(1)
        img_arr = np.where((img_arr < -1) | (img_arr > 1), 0, img_arr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_arr, cmap='RdYlGn', vmin  =-1, vmax =1)  # better to fix range
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    b64_img = base64.b64encode(buf.read()).decode()

    # Shorten filename if too long
    short_name = os.path.basename(tif_file)
    if len(short_name) > 20:
        short_name = short_name[:15] + "..."

    html_parts.append(f'''
        <div style="margin:10px; text-align:center;">
            <img src="data:image/png;base64,{b64_img}" 
                 style="width:350px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.2);" />
            <div style="font-size:12px; color:#555; margin-top:4px;">{short_name}</div>
        </div>
    ''')

# Flexbox layout to center everything and wrap if needed
final_html = f'''
    <div style="
        display:flex; 
        justify-content:center; 
        flex-wrap:wrap;
        padding:10px;
    ">
        {''.join(html_parts)}
    </div>
'''

display(HTML(final_html))



import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os
import pandas as pd

# === Basic setup ===
# Path where all the NDVI GeoTIFF files are located
my_data_folder = "/kaggle/input/sentinel-2-ndvi-2024-dataset"

# Thresholds and filters for detecting weird-looking areas
ndvi_cutoff = 0.7             # anything above this might be an "anomaly"
min_shape_area = 50
max_shape_area = 100000
min_polygon_sides = 3
max_polygon_sides = 6
aspect_ratio_min = 0.3
aspect_ratio_max = 3.0

# === Grab all GeoTIFF files ===
# (we go recursively just in case theyâ€™re in subfolders)
tif_files_list = []
for root, _, files in os.walk(my_data_folder):
    for f in files:
        if f.lower().endswith(('.tif', '.tiff')):
            tif_files_list.append(os.path.join(root, f))

# Sort to keep it consistent
tif_files_list = sorted(tif_files_list)

# List to collect all results
detected_anomalies = []

# === Loop over each image and process ===
for one_tif in tif_files_list:
    try:
        with rasterio.open(one_tif) as raster:
            # Read band 1, assuming NDVI is in there
            ndvi_raw = raster.read(1).astype(np.float32)
            
            # Sometimes NDVI can be out of [-1, 1], remove that
            ndvi_raw = np.where((ndvi_raw < -1) | (ndvi_raw > 1), np.nan, ndvi_raw)
    except Exception as err:
        print(f"(!) Couldn't load {one_tif} -> {err}")
        continue  # skip this one

    # Replace nan with zero (maybe not perfect, but works)
    ndvi_fixed = np.nan_to_num(ndvi_raw, nan=0.0)

    # Invert the NDVI, because weâ€™re more interested in low-NDVI regions
    ndvi_inverted = 1 - ndvi_fixed

    # Create binary mask of "suspiciously low" NDVI zones
    binary_mask = (ndvi_inverted > ndvi_cutoff).astype(np.uint8) * 255

    # Clean the mask slightly to remove noise
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations =1)

    # Find all the contours in the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each blob / contour
    for shape in contours:
        shape_area = cv2.contourArea(shape)
        if not (min_shape_area <= shape_area <= max_shape_area):
            continue

        # Approximate the contour into a polygon to count sides
        polygon = cv2.approxPolyDP(shape, 0.02 * cv2.arcLength(shape, True), True)
        num_sides = len(polygon)
        if not (min_polygon_sides <= num_sides <= max_polygon_sides):
            continue

        # Get bounding box to measure aspect ratio
        x, y, w, h = cv2.boundingRect(shape)
        aspect = w / h if h != 0 else 0
        if not (aspect_ratio_min <= aspect <= aspect_ratio_max):
            continue

        # Save the result
        detected_anomalies.append({
            "filename": os.path.basename(one_tif),
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "area": int(w * h),
            "aspect_ratio": round(aspect, 2),
            "sides": num_sides
        })

# === Turn into a DataFrame for display / export ===
anomaly_df = pd.DataFrame(detected_anomalies)

# Show first 10 rows (works in Jupyter etc.)
from IPython.display import display
display(anomaly_df.head(10))

# Save results to CSV (why not?)
anomaly_df.to_csv("ndvi_anomalies_3.csv", index = False)



import geopandas as gpd
import folium
import pandas as pd
import rasterio
import os
from shapely.geometry import box
from rasterio.transform import Affine
from geopandas import GeoDataFrame

# === Some paths to input data ===
ndvi_folder = "/kaggle/input/sentinel-2-ndvi-2024-dataset"
anomalies_csv = pd.read_csv("/kaggle/input/found-anomalies-v3/ndvi_anomalies_3.csv")
amazon_shp = "/kaggle/input/amazonbasinlimits/amazon_sensulatissimo_gmm_v1.shp"
sites_file = "/kaggle/input/amazon-sites-1/full_amazon_archaeology_sites.csv"

# === Read the shapefile for Amazon Basin and reproject it (just in case) ===
amazon_mask = gpd.read_file(amazon_shp).to_crs("EPSG:4326")

# === Georeferencing anomalies using the original TIFFs ===
geo_anomaly_list = []
by_file = anomalies_csv.groupby("filename")

for fname, group in by_file:
    tif_full_path = os.path.join(ndvi_folder, fname)
    try:
        with rasterio.open(tif_full_path) as src:
            tfm: Affine = src.transform  # needed to convert pixel coords to lat/lon

            for _, anomaly in group.iterrows():
                x, y = anomaly["x"], anomaly["y"]
                w, h = anomaly["width"], anomaly["height"]

                # top-left and bottom-right corners in pixels
                px_tl = (x, y)
                px_br = (x + w, y + h)

                # transform to geographic coordinates
                lon1, lat1 = rasterio.transform.xy(tfm, px_tl[1], px_tl[0], offset="ul")
                lon2, lat2 = rasterio.transform.xy(tfm, px_br[1], px_br[0], offset="lr")

                rect_geom = box(lon1, lat1, lon2, lat2)

                geo_anomaly_list.append({
                    "filename": fname,
                    "area_px": int(w * h),
                    "aspect_ratio": round(w / h if h != 0 else 0, 2),
                    "geometry": rect_geom
                })

    except Exception as oops:
        print(f"[X] Failed to process {fname}: {oops}")
        continue  # skip broken file

# Make GeoDataFrame out of it
anomalies_gdf = GeoDataFrame(geo_anomaly_list, crs="EPSG:4326")

# Keep only anomalies that are inside the Amazon mask polygon
anomalies_inside = gpd.sjoin(anomalies_gdf, amazon_mask, predicate="within", how="inner")
anomalies_inside = anomalies_inside.drop(columns=["index_right"]).reset_index(drop=True)

# === Load archaeological sites ===
raw_sites_df = pd.read_csv(sites_file)

# Convert to GeoDataFrame using lat/lon
site_gdf = gpd.GeoDataFrame(
    raw_sites_df,
    geometry=gpd.points_from_xy(raw_sites_df.Longitude, raw_sites_df.Latitude),
    crs="EPSG:4326"
)

# Filter to just the ones inside the Amazon polygon
sites_in_amazon = gpd.sjoin(site_gdf, amazon_mask, predicate="within").drop(columns="index_right")

# === Build interactive map ===
# We'll center roughly on central Amazon
the_map = folium.Map(location=[-4.0, -63.0], zoom_start=5, tiles="CartoDB positron")

# Show the Amazon basin mask layer (just the boundary)
folium.GeoJson(
    amazon_mask.geometry,
    name="Amazon Basin",
    style_function = lambda x: {
        "fillColor": "#006400",  # dark green
        "color": "#004400",      # darker border
        "weight": 2,
        "fillOpacity": 0.1
    }
).add_to(the_map)

# Show detected anomaly rectangles
for _, rec in anomalies_inside.iterrows():
    b = rec.geometry.bounds  # (minx, miny, maxx, maxy)
    folium.Rectangle(
        bounds=[(b[1], b[0]), (b[3], b[2])],  # lat/lon order
        color="red",
        fill=True,
        fill_opacity=0.4,
        tooltip=f"{rec['filename']}<br>Area: {rec['area_px']} pxÂ²<br>Aspect: {rec['aspect_ratio']}"
    ).add_to(the_map)

# Show known archaeological site markers
for _, site in sites_in_amazon.iterrows():
    info = folium.Popup(f"""
        <b>{site['Site']}</b><br>
        Detection: {site['Detection Method']}<br>
        Year: {site['Year']}<br>
        {site['Description']}
    """, max_width = 300)

    folium.Marker(
        location =[site.geometry.y, site.geometry.x],
        popup = info,
        icon = folium.Icon(color="green", icon="info-sign")
    ).add_to(the_map)

# Add layer control for toggling
folium.LayerControl().add_to(the_map)

# Render the map
the_map



"""import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from shapely.geometry import box
import geopandas as gpd
import pandas as pd
import cv2

# === Configuration ===
dem_dir = "GLO30_FILTERED"  # Folder with filtered DEM .tif files
min_flat_slope = 0.0        # Lower slope threshold (flat)
max_flat_slope = 3.0        # Upper slope threshold in degrees
min_area = 500              # Minimum flat region area (in pixels)
min_sides = 4               # Minimum polygon sides for geometric shape

# === Container for results ===
detected_structures = []

# === Iterate through DEM tiles ===
tif_files = [f for f in os.listdir(dem_dir) if f.endswith(".tif")]

for tif in tif_files:
    path = os.path.join(dem_dir, tif)
    try:
        with rasterio.open(path) as src:
            elevation = src.read(1).astype(np.float32)
            transform = src.transform

        # Replace nodata values
        elevation = np.where((elevation == -32768) | (np.isnan(elevation)), np.nan, elevation)
        elevation = np.nan_to_num(elevation, nan = np.nanmean(elevation))

        # === Compute gradient (slope) ===
        gy, gx = np.gradient(elevation)
        slope_deg = np.degrees(np.sqrt(gx**2 + gy**2))

        # === Detect flat areas ===
        flat_mask = ((slope_deg >= min_flat_slope) & (slope_deg <= max_flat_slope)).astype(np.uint8) * 255
        flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(flat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) < min_sides:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            px1, py1 = x, y
            px2, py2 = x + w, y + h

            lon1, lat1 = rasterio.transform.xy(transform, py1, px1, offset="ul")
            lon2, lat2 = rasterio.transform.xy(transform, py2, px2, offset="lr")

            rect = box(lon1, lat1, lon2, lat2)

            detected_structures.append({
                "filename": tif,
                "area": int(area),
                "sides": len(approx),
                "geometry": rect
            })

        # === Optional visualization ===
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(elevation, cmap="terrain")
        ax[0].set_title("Elevation")
        ax[1].imshow(slope_deg, cmap="magma", vmin=0, vmax=20)
        ax[1].set_title("Slope (Â°)")
        plt.suptitle(f"Tile: {tif}")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"[!] Skipped {tif}: {e}")
        continue

# === Convert results to GeoDataFrame and save ===
gdf = gpd.GeoDataFrame(detected_structures, crs="EPSG:4326")
gdf.to_file("detected_structures_dem.geojson", driver="GeoJSON")
print(f"Found {len(gdf)} candidate structures")"""



import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import rasterio
from rasterio.transform import Affine
import folium

# === input paths ===
csv_path = "/kaggle/input/found-anomalies-v3/ndvi_anomalies_3.csv"
tiff_folder = "/kaggle/input/sentinel-2-ndvi-2024-dataset"
dem_path = "/kaggle/input/detected-dem-v7/detected_structures_dem_v7.geojson"
amazon_shp = "/kaggle/input/amazonbasinlimits/amazon_sensulatissimo_gmm_v1.shp"
sites_path = "/kaggle/input/amazon-sites-1/full_amazon_archaeology_sites.csv"

# === load amazon basin shapefile ===
amazon_mask = gpd.read_file(amazon_shp).to_crs("EPSG:4326")

# === load and georeference NDVI anomalies ===
df = pd.read_csv(csv_path)
ndvi_anomalies = []

for fname, group in df.groupby("filename"):
    tif_path = os.path.join(tiff_folder, fname)
    if not os.path.exists(tif_path):
        continue
    try:
        with rasterio.open(tif_path) as src:
            transform = src.transform
            for _, row in group.iterrows():
                x, y = row["x"], row["y"]
                w, h = row["width"], row["height"]
                lon1, lat1 = rasterio.transform.xy(transform, y, x, offset="ul")
                lon2, lat2 = rasterio.transform.xy(transform, y + h, x + w, offset="lr")
                geom = box(lon1, lat1, lon2, lat2)
                ndvi_anomalies.append({
                    "filename": fname,
                    "area_px": row["area"],
                    "aspect_ratio": row["aspect_ratio"],
                    "geometry": geom
                })
    except:
        continue

ndvi_gdf_raw = gpd.GeoDataFrame(ndvi_anomalies, crs="EPSG:4326")
ndvi_gdf = gpd.sjoin(ndvi_gdf_raw, amazon_mask, predicate="within", how="inner").drop(columns="index_right")

# === load DEM detections ===
dem_gdf = gpd.read_file(dem_path).to_crs("EPSG:4326")

# === cross-reference NDVI and DEM ===
cross_hits = gpd.sjoin(ndvi_gdf, dem_gdf, predicate="intersects", how="inner")

# === load known archaeological sites ===
sites_df = pd.read_csv(sites_path)
sites_gdf = gpd.GeoDataFrame(
    sites_df,
    geometry=gpd.points_from_xy(sites_df.Longitude, sites_df.Latitude),
    crs="EPSG:4326"
)
sites_amazon = gpd.sjoin(sites_gdf, amazon_mask, predicate="within").drop(columns="index_right")

# === build map ===
m = folium.Map(location=[-4, -63], zoom_start=5, tiles="CartoDB positron")

folium.GeoJson(
    amazon_mask.geometry,
    name="Amazon Basin",
    style_function=lambda x: {
        "fillColor": "#228B22",
        "color": "#006400",
        "weight": 1,
        "fillOpacity": 0.1
    }
).add_to(m)

# NDVI rectangles
for _, row in ndvi_gdf.iterrows():
    b = row.geometry.bounds
    folium.Rectangle(
        bounds=[(b[1], b[0]), (b[3], b[2])],
        color="red",
        fill=True,
        fill_opacity=0.3,
        tooltip=f"NDVI: {row['filename']}"
    ).add_to(m)

# DEM rectangles
for _, row in dem_gdf.iterrows():
    b = row.geometry.bounds
    folium.Rectangle(
        bounds=[(b[1], b[0]), (b[3], b[2])],
        color="blue",
        fill=False,
        weight=1,
        tooltip=f"DEM: {row['filename']}"
    ).add_to(m)

# NDVI+DEM overlaps
for _, row in cross_hits.iterrows():
    b = row.geometry.bounds
    folium.Rectangle(
        bounds=[(b[1], b[0]), (b[3], b[2])],
        color="orange",
        fill=True,
        fill_opacity=0.5,
        tooltip="Overlap: NDVI + DEM"
    ).add_to(m)

# === Check if any cross-detected zones exist ===
if not cross_hits.empty:
    print(f"Found {len(cross_hits)} overlapping regions between NDVI and DEM detections.")
    cross_hits.to_file("/kaggle/working/cross_hits.geojson", driver="GeoJSON")
    print(" GeoJSON with cross-hits saved successfully.")
else:
    print("âš ï¸� No cross-hits to save.")

# known archaeological sites
for _, row in sites_amazon.iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=folium.Popup(f"""
            <b>{row['Site']}</b><br>
            Detection: {row['Detection Method']}<br>
            Year: {row['Year']}<br>
            {row['Description']}
        """, max_width=300),
        icon=folium.Icon(color="green", icon="info-sign")
    ).add_to(m)

folium.LayerControl().add_to(m)
m




from IPython.display import Image, display
display(Image("/kaggle/input/visual-assets/Exapmle for DEM analysis.png"))


import openai
import geopandas as gpd

# load API key from txt file
with open("/kaggle/input/api-keys/openai_key.txt", "r") as f:
    my_key = f.read().strip()

# init client
client = openai.OpenAI(api_key = my_key)

# load the geojson file with cross detections (NDVI + DEM)
cross_path = "/kaggle/working/cross_hits.geojson"
gdf = gpd.read_file(cross_path)

# loop through each feature and make GPT prompt
all_descriptions = []

for i, row in gdf.iterrows():
    geom = row.geometry
    area = row.get("area_px", "unknown")
    aspect = row.get("aspect_ratio", "unknown")

    # just a simple prompt, nothing fancy
    prompt = f"""
A geospatial anomaly has been detected in both vegetation and elevation data.
Here is some basic information:

- Area in pixels: {area}
- Aspect ratio (width/height): {aspect}
- Approximate coordinates: centroid at ({geom.centroid.y:.4f}, {geom.centroid.x:.4f})

Please generate a short 1â€“2 sentence hypothesis about what this anomaly could represent, based on patterns in vegetation and terrain.
Do not mention NDVI or DEM explicitly.
""".strip()

    try:
        res = client.chat.completions.create(
            model = "gpt-4",
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.5,
            max_tokens = 100
        )
        reply = res.choices[0].message.content.strip()
    except Exception as err:
        reply = f"error: {err}"

    all_descriptions.append(reply)

# put results back into the dataframe
gdf["gpt_description"] = all_descriptions

# save to new file
gdf.to_file("/kaggle/working/cross_hits_described.geojson", driver="GeoJSON")

print("Saved described GeoJSON!")



import geopandas as gpd

described_path = "/kaggle/working/cross_hits_described.geojson"

gdf = gpd.read_file(described_path)

for i, row in gdf.iterrows():
    print(f"ğŸ“� Location (centroid): ({row.geometry.centroid.y:.4f}, {row.geometry.centroid.x:.4f})")
    print(f"ğŸ“� GPT Description: {row.get('gpt_description', 'â€”')}")
    print("-" * 60)



import openai

# === Load API key ===
with open("/kaggle/input/api-keys/openai_key.txt", "r") as f:
    api_key = f.read().strip()

client = openai.OpenAI(api_key=api_key)

# === Updated context for the search ===
site_name = "TeotÃ´nio (Madeira River, RondÃ´nia, Brazil)"
doi = "10.1371/journal.pone.0199868"
coords = "-8.917, -64.0"
year = 2010
method = "Salvage excavation"

# === Refined prompt ===
prompt = f"""
You are an archaeological research assistant.

A site called "{site_name}" was excavated in {year} using a salvage excavation method. It is located at coordinates {coords}.
Please return a concise summary (1â€“3 sentences) of the site's historical or archaeological importance, based on academic references.

If available, quote or paraphrase scholarly sources (e.g. scientific articles, journals, books). Prefer referencing the following DOI:
{doi}

Do NOT invent information. If no information is available, respond clearly that nothing verifiable could be found.

Respond only with the academic citation or passage.
""".strip()

# === API call ===
try:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )
    citation_output = response.choices[0].message.content.strip()
except Exception as e:
    citation_output = f"API error: {e}"

# === Output result ===
print("ğŸ“– Citation or summary:")
print(citation_output)

with open("/kaggle/working/teotonio_academic_reference.txt", "w") as f:
    f.write(citation_output)





