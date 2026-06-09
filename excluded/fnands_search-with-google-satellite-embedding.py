import geopandas as gpd


points_gdf = gpd.read_file("/kaggle/input/mergedforestedpolygons/forested_merged_soil_points_80.geojson")


points_gdf.head()


import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import math

# Step 1: Add unique ID
points_gdf = points_gdf.copy()
points_gdf['point_id'] = ['P{:05d}'.format(i) for i in range(len(points_gdf))]

# Step 2: Function to generate 9 shifted points
def generate_shifts(row):
    lat = row.geometry.y
    lon = row.geometry.x

    # Approximate degree shifts for ~20 meters
    delta_lat = 20 / 111320
    delta_lon = 20 / (111320 * math.cos(math.radians(lat)))

    shifts = [
        (0, 0),                        # original
        ( delta_lon, 0),              # east
        (-delta_lon, 0),              # west
        (0,  delta_lat),              # north
        (0, -delta_lat),              # south
        ( delta_lon,  delta_lat),     # northeast
        ( delta_lon, -delta_lat),     # southeast
        (-delta_lon,  delta_lat),     # northwest
        (-delta_lon, -delta_lat),     # southwest
    ]

    new_rows = []
    for dx, dy in shifts:
        shifted_point = Point(lon + dx, lat + dy)
        new_row = row.copy()
        new_row['geometry'] = shifted_point
        new_rows.append(new_row)

    return new_rows

# Step 3: Expand original GeoDataFrame
all_rows = []
for _, row in points_gdf.iterrows():
    all_rows.extend(generate_shifts(row))

augmented_gdf = gpd.GeoDataFrame(all_rows, crs='EPSG:4326')

# Optional: Add offset index (0 for original, 1–8 for shifted)
augmented_gdf['offset_index'] = augmented_gdf.groupby('point_id').cumcount()

augmented_gdf.reset_index(drop=True, inplace=True)


augmented_gdf.tail()


import ee
import geopandas as gpd
import geemap
from shapely.geometry import mapping, Point
import pandas as pd
from tqdm import tqdm
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
gee_account = user_secrets.get_secret("gee_account")
gee_credentials = "/kaggle/input/geecreds/geecreds.json" # the file path for the JSON file containing the relevant credentials
ee_creds = ee.ServiceAccountCredentials(gee_account, gee_credentials) # fetch your service account credentials
ee.Initialize(ee_creds) # initialize earth engine using your service account credentials


# Step 2: Convert GeoDataFrame to ee.FeatureCollection
def gdf_to_ee_fc(gdf):
    features = []
    for _, row in gdf.iterrows():
        geom = mapping(row['geometry'])
        props = row.drop('geometry').to_dict()
        features.append(ee.Feature(ee.Geometry.Point(geom['coordinates']), props))
    return ee.FeatureCollection(features)


# Your points_gdf
points_fc = gdf_to_ee_fc(augmented_gdf)




periods = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]

for p in periods: 
    print(p)


periods = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]

for p in periods: 
    # Filter the collection to the year and region of interest
    collection = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
        .filterDate(f'{p[0]}-01-01', f'{p[1]}-01-01') \
        .filterBounds(points_fc)
    
    # Safety check
    if collection.size().getInfo() == 0:
        raise Exception("No embedding image found for the selected year and area.")
    
    # List of all image tiles
    images_list = collection.toList(collection.size())
    
    # Track results
    all_sampled_features = []
    
    print("Sampling embeddings across multiple tiles...")
    
    for i in tqdm(range(collection.size().getInfo())):
        img = ee.Image(images_list.get(i))
        # Find points in the current image tile (using intersection)
        img_geom = img.geometry()
        clipped_points = points_fc.filterBounds(img_geom)
        count = clipped_points.size().getInfo()
        if count == 0:
            continue  # No points in this tile
        
        # Sample the image at the points
        sampled = img.sampleRegions(
            collection=clipped_points,
            scale=10,
            geometries=True,
            properties=['site_type', 'forest_value', 'point_id', 'offset_index']
        )
        
        # Get results and append
        sampled_features = sampled.getInfo()['features']
        all_sampled_features.extend(sampled_features)
    
    # Convert to GeoDataFrame
    data = []
    for f in all_sampled_features:
        coords = f['geometry']['coordinates']
        props = f['properties']
        props['geometry'] = Point(coords)
        data.append(props)
    
    sampled_gdf = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')
    sampled_gdf.to_file(f'ade_training_data_{p[0]}_{p[1]}_80_aug.geojson', driver='GeoJSON')





sampled_gdf




