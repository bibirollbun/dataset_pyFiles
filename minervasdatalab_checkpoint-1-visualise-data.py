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


import pandas as pd





points=pd.read_csv('/kaggle/input/archeologicalartefacts/filtered_points.csv')


points.head()





from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GEEProject")


!pip install geemap


import ee


# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize(project='gee')


import ee

# Convert points to ee.FeatureCollection
features = [ee.Feature(ee.Geometry.Point([row['POINT_X'], row['POINT_Y']])) for _, row in points.iterrows()]
point_fc = ee.FeatureCollection(features)


sentinel_collection = ee.ImageCollection("COPERNICUS/S2_SR") \
    .filterBounds(point_fc) \
    .filterDate('2023-06-01', '2023-06-30') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))



first_image = sentinel_collection.first()
image_id = first_image.get('system:id').getInfo()
print("Scene ID:", image_id)



import ee

# Load the image using its ID
image = ee.Image("COPERNICUS/S2_SR/20230601T133149_20230601T133148_T22MGU")






# SOLUTION 1: Use folium directly (Recommended for Kaggle)
import ee
import folium
import pandas as pd
import matplotlib.pyplot as plt



# Load your data
# points = pd.DataFrame({'type': [...], 'POINT_X': [...], 'POINT_Y': [...]})
unique_types = points['type'].unique()

# Use matplotlib's colormap
colormap = plt.colormaps['tab20']
color_hexes = [
    f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    for r, g, b, _ in colormap(range(len(unique_types)))
]

# Create a folium map directly
Map = folium.Map(location=[-10, -65], zoom_start=5)

# Add Sentinel-2 image using folium
image = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterDate('2021-06-01', '2021-08-31') \
    .filterBounds(ee.Geometry.Point([-65, -10])) \
    .sort('CLOUDY_PIXEL_PERCENTAGE') \
    .first()

rgb = image.select(['B4', 'B3', 'B2'])
vis_params = {'min': 0, 'max': 3000}

# Get the map tiles URL from Earth Engine
map_id_dict = ee.Image(rgb).getMapId(vis_params)
folium.raster_layers.TileLayer(
    tiles=map_id_dict['tile_fetcher'].url_format,
    attr='Google Earth Engine',
    name='Sentinel-2',
    overlay=True,
    control=True
).add_to(Map)

# Add points by type
for idx, type_name in enumerate(unique_types):
    subset = points[points['type'] == type_name]
    for _, row in subset.iterrows():
        folium.CircleMarker(
            location=[row['POINT_Y'], row['POINT_X']],  # Note: folium uses [lat, lon]
            radius=5,
            popup=f"Type: {row['type']}",
            color=color_hexes[idx],
            fill=True,
            fillColor=color_hexes[idx]
        ).add_to(Map)

# Add layer control
folium.LayerControl().add_to(Map)

# Display map
Map















