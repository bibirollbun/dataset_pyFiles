import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import geopandas as gpd


from shapely.geometry import box
from shapely import wkt
from pyproj import Geod


import folium


import json


import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient


import re


# Area from bbox coordinates
def compute_area(poly):
    
    # Use WGS84 ellipsoid
    geod = Geod(ellps="WGS84")
    
    # Calculate area (signed) in m² — returns (area, perimeter)
    area, _ = geod.geometry_area_perimeter(poly)
    
    # Convert to km² and get absolute value
    area_km2 = abs(area) / 1e6
    return area_km2 


# visualize bbox
def visualize_bbox(BBOX, zoom_start):

    # Center on AOI
    lat_center = (BBOX[1] + BBOX[3]) / 2
    lon_center = (BBOX[0] + BBOX[2]) / 2
    
    # Create map
    m = folium.Map(location=[lat_center, lon_center], zoom_start=zoom_start)
    
    # Add AOI rectangle
    folium.Rectangle(
        bounds=[[BBOX[1], BBOX[0]], [BBOX[3], BBOX[2]]],
        color='red',
        fill=True,
        fill_opacity=0.2,
        tooltip="Acre AOI"
    ).add_to(m)

    return m


def create_m(centre, zoom_start, titles):
    return folium.Map(location= centre, zoom_start=zoom_start, tiles = titles)


# add legend to folium
def add_legend(m):
    legend_html = """
    <div style="position: fixed;
         bottom: 30px; left: 30px; width: 180px; height: 110px;
         border:2px solid grey; z-index:9999; font-size:14px;
         background-color: white;
         padding: 10px;">
    <b>Legend</b><br>
    <span style="color: red;">&#9632;</span> AOI Boundary<br>
    <span style="color: blue;">&#9632;</span> GEDI Point<br>
    <span style="color: orange;">&#9632;</span> Deforested Area
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


AOI_BBOX = [-70.0, -14.0, -54.0, -1.0]  # xmin, ymin, xmax, ymax
xmin, ymin, xmax, ymax = AOI_BBOX       # lon_min, lat_min, lon_max, lat_max


print(f"Area of Interest (AOI) Area: {compute_area(box(*AOI_BBOX)):.2f} km²")


visualize_bbox(AOI_BBOX, zoom_start = 5)


gedi_path = '/kaggle/input/2-gedi-terrabrasilis-for-aoi/gedi_aoi.csv'


gedi_data = pd.read_csv(gedi_path)


gedi_data = gedi_data[
(gedi_data['sensitivity'] > 0.90) &     # High sensitivity
        (gedi_data['rh100'] > 0) &             # Valid canopy heights
        (gedi_data['pai'] > 0)   
].copy()


gedi_data.describe()


gedi_data.head()


gedi_data.isnull().values.any()


len(gedi_data)


terra_path = '/kaggle/input/2-gedi-terrabrasilis-for-aoi/terrabrasilis_aoi.gpkg'


%%time
terra = gpd.read_file(terra_path, layer="deforestation")


terra.head()


terra.describe()


terra.crs


%%time
# Generate tile grid (0.03° x 0.03°) as WKT-only CSV for external use (e.g., Kaggle)
tile_data = []
tile_size = 0.03
tile_id = 0

for x in np.arange(xmin, xmax, tile_size):
    for y in np.arange(ymin, ymax, tile_size):
        geom = box(x, y, x + tile_size, y + tile_size)
        tile_data.append({
            "tile_id": f"T{tile_id}",
            "geometry": geom.wkt,
            "lat_center": (y + y + tile_size) / 2,
            "lon_center": (x + x + tile_size) / 2
        })
        tile_id += 1

tile_grid_df = pd.DataFrame(tile_data)


tile_grid_df.head()


tile_grid_df.describe()


len(tile_grid_df)


#((ymax-ymin)/0.03)*((xmax-xmin)/0.03)


tile_grid_df["geometry"] = tile_grid_df["geometry"].apply(wkt.loads)


tile_gdf = gpd.GeoDataFrame(tile_grid_df, geometry="geometry", crs="EPSG:4326")


tile_gdf


compute_area(tile_gdf["geometry"][0])


area_km = pd.Series(tile_gdf["geometry"].apply(compute_area))


area_km.mean()


tile_gdf_sample = tile_gdf.sample(n=500, random_state=1)


%%time
# Create a folium map centered on tile grid
map_center = [tile_gdf_sample["lat_center"].mean(), tile_gdf_sample["lon_center"].mean()]
m = create_m(centre=map_center, zoom_start=5, titles="CartoDB positron")


%%time
# Add tile grid polygons
for _, row in tile_gdf_sample.iterrows():
    folium.GeoJson(row.geometry, style_function=lambda x: {
        "color": "red", "weight": 2, "fillOpacity": 0.2
    }).add_to(m)


%%time
# this is a sample plot with only few tiles - tiles actually occupy entire AOI_BBOX
m


gedi_gdf = gpd.GeoDataFrame(
    gedi_data,
    geometry=gpd.points_from_xy(gedi_data.longitude, gedi_data.latitude),
    crs="EPSG:4326"
)


terra_df = terra[['uuid', 'year', 'area_km', 'geometry']]
terra_gdf = gpd.GeoDataFrame(terra_df, geometry='geometry', crs='EPSG:4326')


type(terra_gdf['geometry'])


# 1. Join GEDI to tile grid (within)
gedi_in_tiles = gpd.sjoin(gedi_gdf, tile_gdf, how="inner", predicate="within")


gedi_in_tiles.head()


len(gedi_in_tiles)


# 2. Aggregate GEDI metrics per tile
agg = gedi_in_tiles.groupby("tile_id").agg({
    "rh100": ["count", "mean", "std", "max"],
    "pai": "mean",
    "fhd": "mean"
}).reset_index()

agg.columns = ['tile_id', 'gedi_count', 'rh100_mean', 'rh100_std', 'rh100_max', 'pai_mean', 'fhd_mean']


agg.head()


len(agg)


%%time
# 3. Join TerraBrasilis polygons to tile grid (intersects)
# Skip if terra_gdf is empty
if not terra_gdf.empty:
    tiles_with_deforestation = gpd.sjoin(tile_gdf, terra_gdf, how="left", predicate="intersects")
    tiles_with_deforestation["deforested"] = tiles_with_deforestation["uuid"].notnull()
    deforestation_flag = tiles_with_deforestation[["tile_id", "deforested"]].drop_duplicates()
else:
    deforestation_flag = tile_gdf[["tile_id"]].copy()
    deforestation_flag["deforested"] = False


len(tiles_with_deforestation)-len(deforestation_flag)


%%time
# 4. Merge everything into a single summary per tile
tile_summary = tile_gdf[["tile_id", "lat_center", "lon_center"]].merge(agg, on="tile_id", how="inner")
tile_summary = tile_summary.merge(deforestation_flag, on="tile_id", how="left")
tile_summary["deforested"] = tile_summary["deforested"].fillna(False)


len(tile_summary)


tile_summary.head()


tile_summary.describe()


m1 = visualize_bbox(AOI_BBOX, 5)


# Add tile_summary if exists
if 'tile_summary' in locals():
    tile_gdf = gpd.GeoDataFrame(tile_summary, geometry=gpd.points_from_xy(tile_summary.lon_center, tile_summary.lat_center), crs="EPSG:4326")
    for _, row in tile_gdf.iterrows():
        popup = f"Tile: {row['tile_id']}<br>Deforested: {row['deforested']}"
        color = "green" if not row['deforested'] else "darkred"
        folium.CircleMarker(
            location=[row['lat_center'], row['lon_center']],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=popup
        ).add_to(m1)



m1


df_filtered = tile_summary.dropna(subset=["rh100_mean"])


df_filtered.columns


condition = (
    df_filtered["rh100_mean"] > df_filtered["rh100_mean"].quantile(0.75)
)

df_final = df_filtered[condition].copy()


len(df_final)


df_final.columns


# Save JSON input for GPT
candidates_json = df_final.to_dict(orient="records")
#json_path = "candidates_for_gpt.json"
#with open(json_path, "w") as f:
 #   json.dump(candidates_json, f, indent=2)


feature_explanations = {
    "tile_id": "Unique tile identifier",
    "lat_center": "Latitude of tile center",
    "lon_center": "Longitude of tile center",
    "gedi_count": "Number of GEDI points in the tile",
    "rh100_mean": "Mean canopy height (100th percentile return height)",
    "rh100_std": "Standard deviation of canopy height",
    "rh_max": "Maximum Canopy Height",
    "pai_mean": "Plant Area Index (vegetation density)",
    "fhd_mean": "Fractional cover of vegetation",
    "deforested": "Boolean flag from TerraBrasilis indicating deforestation"
}


explanation_lines = [
    f"{key}: {feature_explanations.get(key, 'No description')}"
    for key in tile_summary.columns
]


explanation_lines


secret = UserSecretsClient()
openai_key = secret.get_secret("arch-bot")


client = OpenAI(
  api_key=openai_key
)


prompt_text = f"""You are an archaeologist.

You are provided with a list of landscape tiles from the Amazon rainforest. Each tile has been preprocessed using GEDI L2B and TerraBrasilis deforestation data.

Your task is to review these tiles and select the **five most surprising or scientifically interesting anomalies** useful to you based on the following features.

Each tile spans approximately 0.03° latitude × 0.03° longitude with mean area of 11 km². Each tile record includes:
{chr(10).join(explanation_lines)}

Return exactly 5 tiles in JSON format, using this structure:
[
  {{
    "tile_id": "...",
    "lat_center": ...,
    "lon_center": ...,
    "confidence": 0-100,
    "reason": "Explain why this tile is anomalous in 3 precise points"
  }},
  ...
]

Avoid extra commentary. Focus on unusual canopy structure, outliers in vegetation metrics, deforestation mismatches, or highly unique patterns.

Total records: {len(candidates_json)}

Tile data:
""" + json.dumps(candidates_json, indent=2)



response = client.chat.completions.create(
  model="gpt-4o-mini",
    temperature=0,
  messages=[{"role": "user", "content": prompt_text}]
)


output = response.choices[0].message.content
print(output)


# Use regex to extract the JSON list from within ```json ... ``` block
json_str = re.search(r"\[.*\]", output, re.DOTALL).group(0)

# Parse to Python object
anomaly_list = json.loads(json_str)

# Convert to DataFrame
anomalies_df = pd.DataFrame(anomaly_list)

# Show result
anomalies_df.head()


tile_gdf2 = gpd.GeoDataFrame(tile_grid_df, geometry="geometry", crs="EPSG:4326")


# Ensure tile_gdf is a GeoDataFrame with 'tile_id' and 'geometry'
joined = anomalies_df.merge(tile_gdf2[["tile_id", "geometry"]], on="tile_id", how="left")


joined


joined = gpd.GeoDataFrame(joined, geometry="geometry", crs="EPSG:4326")

# Make sure 'joined' is a GeoDataFrame and 'geometry' is active geometry
bounds_df = joined.bounds  # This returns a DataFrame with minx, miny, maxx, maxy

# Concatenate bounds back into the original GeoDataFrame
joined = pd.concat([joined, bounds_df], axis=1)

# Optional: create bbox list column
joined["bbox"] = joined.apply(
    lambda row: [row["minx"], row["miny"], row["maxx"], row["maxy"]],
    axis=1
)


# Round each coordinate in bbox list to 3 decimals
joined["bbox"] = joined["bbox"].apply(lambda b: [round(coord, 3) for coord in b])


joined


center_lat = joined["lat_center"].mean()
center_lon = joined["lon_center"].mean()

m_f = create_m(centre = [center_lat, center_lon], zoom_start=8, titles="cartodbpositron")


# Add tiles to the map
for _, row in joined.iterrows():
    geo_json = folium.GeoJson(row["geometry"], 
                               tooltip=folium.Tooltip(f"{row['tile_id']}<br>{row['reason']}"),
                               style_function=lambda x: {"fillColor": "red", "color": "black", "weight": 1, "fillOpacity": 0.4})
    geo_json.add_to(m_f)


m_f


print("Model version:", response.model)


# Dataset log
dataset_id = f"""
    'GEDI': '/kaggle/input/2-gedi-terrabrasilis-for-aoi/gedi_aoi.csv',
    'Terra Brasilis': '/kaggle/input/2-gedi-terrabrasilis-for-aoi/terrabrasilis_aoi.gpkg'
"""
with open("dataset_ids.txt", "w") as f:
    f.write(dataset_id)


# Prompt log
with open("find_top5_anomaly_gpt_prompt.txt", "w") as f:
    f.write(prompt_text)


# Response log
with open("top5_anomaly_result.json", "w") as f:
    json.dump({"response": output}, f, indent=2)


# Output log
output_log = joined[['tile_id', 'bbox', 'geometry', 'confidence', 'reason']]
output_log.to_csv("output_log.csv",index=False)


anomaly_list


followup_prompt = f"""
You are an environmental research assistant helping prioritize future exploration in the Amazon.

Here are 5 anomaly tiles already discovered, along with summary metrics and reasons.

{json.dumps(anomaly_list, indent=2)}

Based on these, do the following:
1. Summarize common patterns or metrics among these tiles.
2. Suggest rules to find similar future anomalies.
3. Describe how this knowledge can help forest monitoring efforts.

Be specific and concise. Return results in Markdown format.
"""


response2 = client.chat.completions.create(
  model="gpt-4o-mini",
    temperature=0,
  messages=[{"role": "user", "content": followup_prompt}]
)


print(response2.choices[0].message.content)

