!pip install utm --quiet

import pandas as pd
import numpy as np
import ee
import json
import re
import utm
from kaggle_secrets import UserSecretsClient

# Authenticate and initialize Earth Engine
user_secrets = UserSecretsClient()
iam_service_account = user_secrets.get_secret('IAM service account')
ee_credentials_json = user_secrets.get_secret('ee creds')
ee_creds = ee.ServiceAccountCredentials(iam_service_account, ee_credentials_json)
ee.Initialize(ee_creds)

# Load new UTM-based CSV
utm_df = pd.read_csv('/kaggle/input/casarabe-bolivia-mounds-list/MoundsOnly.csv')
print(f"Loaded UTM-based CSV: {len(utm_df)} rows")

# Convert UTM X/Y to Latitude/Longitude
def utm_to_latlon(row):
    try:
        lat, lon = utm.to_latlon(row['x (WGS 1984 UTM Zone 20S)'], row['y (WGS 1984 UTM Zone 20S)'], 20, 'K')  # Adjust UTM zone as needed
        return pd.Series({'latitude': lat, 'longitude': lon})
    except:
        return pd.Series({'latitude': np.nan, 'longitude': np.nan})

utm_df[['latitude', 'longitude']] = utm_df.apply(utm_to_latlon, axis=1)
utm_df = utm_df.dropna(subset=['latitude', 'longitude'])

# Define study area polygon
study_coords = [[-65.90750567196693,-15.874873590223393],
                [-63.05106035946693,-15.991080553875655],
                [-63.23782793759193,-12.906236724307375],
                [-66.08328692196693,-12.756269030308728],
                [-65.90750567196693,-15.874873590223393]
]
study_area = ee.Geometry.Polygon([study_coords])

# Keep only features within AOI
def within_study_area(lat, lon):
    try:
        return study_area.contains(ee.Geometry.Point([lon, lat])).getInfo()
    except:
        return False

utm_df = utm_df[utm_df.apply(lambda row: within_study_area(row['latitude'], row['longitude']), axis=1)]
print(f"Sites inside AOI: {len(utm_df)}")

# Generate 100 m buffered geometries
def buffer_geometry(lat, lon, radius_m=100):
    try:
        point = ee.Geometry.Point([float(lon), float(lat)])
        buffer = point.buffer(radius_m)
        return json.dumps(buffer.getInfo())
    except Exception as e:
        return None

utm_df['geometry'] = utm_df.apply(lambda row: buffer_geometry(row['latitude'], row['longitude']), axis=1)
utm_df = utm_df.dropna(subset=['geometry'])

# Assign class label for earthworks
utm_df['class'] = 1

# Limit output columns 
utm_df = utm_df[['latitude', 'longitude', 'geometry', 'class']]

# Save to working directory
utm_df.to_csv('/kaggle/working/mounds_only.csv', index=False)

# Generate random points in study area
study_area_fc = ee.FeatureCollection.randomPoints(region=study_area, points=100, seed=42)

# Extract lon/lat from the feature collection
features = study_area_fc.getInfo()['features']
coords_list = [
    [feat['geometry']['coordinates'][0], feat['geometry']['coordinates'][1]]
    for feat in features
]

# Convert to DataFrame
non_df = pd.DataFrame(coords_list, columns=['longitude', 'latitude'])

# Define buffering function
def buffer_geometry(lat, lon, radius_m=100):
    try:
        point = ee.Geometry.Point([float(lon), float(lat)])
        buffer = point.buffer(radius_m)
        return json.dumps(buffer.getInfo())
    except Exception as e:
        print(f"Buffer error at {lat},{lon}: {e}")
        return None

# Apply 100m buffer and clean
non_df['geometry'] = non_df.apply(lambda row: buffer_geometry(row['latitude'], row['longitude']), axis=1)
non_df = non_df.dropna(subset=['geometry'])
non_df['class'] = 0

# Save result
non_df.to_csv('/kaggle/working/not_mounds.csv', index=False)
print(f"✅ Saved {len(non_df)} non-earthwork points with buffer geometry.")


import ee
import folium
from kaggle_secrets import UserSecretsClient

# --------------------------
# 1. Authenticate and Initialize
# --------------------------
user_secrets = UserSecretsClient()
iam_service_account = user_secrets.get_secret('IAM service account')
ee_credentials_json = user_secrets.get_secret('ee creds')
ee_creds = ee.ServiceAccountCredentials(iam_service_account, ee_credentials_json)
ee.Initialize(ee_creds)

# --------------------------
# 2. Define AOI
# --------------------------
geometry = ee.Geometry.Polygon([
    [[-65.90750567196693,-15.874873590223393],
    [-63.05106035946693,-15.991080553875655],
    [-63.23782793759193,-12.906236724307375],
    [-66.08328692196693,-12.756269030308728],
    [-65.90750567196693,-15.874873590223393]]
])

# --------------------------
# 3. Sentinel-2 Composite
# --------------------------
def maskS2clouds(image):
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
           qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000).select("B.*").copyProperties(image, ["system:time_start"])

S2_col = ee.ImageCollection('COPERNICUS/S2') \
    .filterBounds(geometry) \
    .filterDate('2020-09-01', '2020-09-30') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .map(maskS2clouds)

s2comp = S2_col.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']).mean().clip(geometry)

# --------------------------
# 4. Sentinel-1 Composite
# --------------------------
s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
    .filterBounds(geometry) \
    .filterDate('2020-07-01', '2020-09-30')

def filter_s1(col):
    return col \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW'))

vvvh_asc = filter_s1(s1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')))
vvvh_desc = filter_s1(s1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')))

s1comp = ee.Image.cat([
    vvvh_asc.select('VV').median().rename('s1vva'),
    vvvh_asc.select('VH').median().rename('s1vha'),
    vvvh_desc.select('VV').median().rename('s1vvd'),
    vvvh_desc.select('VH').median().rename('s1vhd'),
]).clip(geometry)

# --------------------------
# 5. DEM + Terrain Features
# --------------------------
dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation').clip(geometry)
slope = ee.Terrain.slope(dem)
aspect = ee.Terrain.aspect(dem)

kernel_radius = 250  # meters
kernel = ee.Kernel.square(radius=kernel_radius, units='meters')
local_mean = dem.reduceNeighborhood(ee.Reducer.mean(), kernel)
local_std = dem.reduceNeighborhood(ee.Reducer.stdDev(), kernel)
zscore_local = dem.subtract(local_mean).divide(local_std).rename('zscore_local')

dem_features = dem.rename('elevation') \
    .addBands(slope.rename('slope')) \
    .addBands(aspect.rename('aspect')) \
    .addBands(zscore_local)

# --------------------------
# 6. Full Composite
# --------------------------
composite = s1comp.addBands(s2comp).addBands(dem_features).multiply(10000).round().divide(10000)

# --------------------------
# 7. Load and Limit Training Data
# --------------------------
sites = ee.FeatureCollection('projects/amazon-starter/assets/mounds_only').limit(2000)
no_mounds = ee.FeatureCollection('projects/amazon-starter/assets/not_mounds').limit(2000)
training_features = sites.merge(no_mounds)

# --------------------------
# 8. Sample Regions
# --------------------------
bands = ['s1vva','s1vha','s1vvd','s1vhd',
         'B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
         'elevation', 'slope', 'aspect', 'zscore_local']

training_samples = composite.select(bands).sampleRegions(
    collection=training_features,
    properties=['class'],
    scale=10,
    geometries=False
).filter(ee.Filter.notNull(bands))

# Print the number of valid training samples
try:
    sample_count = training_samples.size().getInfo()
    print(f"Number of valid training samples: {sample_count}")
except Exception as e:
    print("Could not retrieve sample count:", e)

# --------------------------
# 9. Train and Evaluate Classifier
# --------------------------
with_random = training_samples.randomColumn('random', seed=42)
train_data = with_random.filter(ee.Filter.lt('random', 0.7))
val_data = with_random.filter(ee.Filter.gte('random', 0.7))

classifier = ee.Classifier.smileRandomForest(numberOfTrees=128) \
    .setOutputMode('PROBABILITY') \
    .train(features=train_data, classProperty='class', inputProperties=bands)

label_classifier = ee.Classifier.smileRandomForest(numberOfTrees=128) \
    .setOutputMode('CLASSIFICATION') \
    .train(features=train_data, classProperty='class', inputProperties=bands)

validated = val_data.classify(label_classifier)

conf_matrix = validated.errorMatrix('class', 'classification')
try:
    print('Confusion Matrix:', conf_matrix.getInfo())
    print('Validation Accuracy:', conf_matrix.accuracy().getInfo())
except Exception as e:
    print("Failed to compute matrix:", e)

# --------------------------
# 10. Classify Image and Display Probabilities
# --------------------------

# Classify image
classified = composite.select(bands).classify(classifier)

# Create visualization parameters
prob_vis = {'min': 0.55, 'max': 1, 'palette': ['lightgreen', 'yellow', 'orange', 'red']}
prob_map_id = classified.getMapId(prob_vis)
prob_tile_url = prob_map_id['tile_fetcher'].url_format

# Clip DEM
dem_window = dem.updateMask(dem.gte(120).And(dem.lte(220)))
dem_vis = {'min': 120, 'max': 220, 'palette': ['purple', 'brown', 'white']}
dem_map_id = dem_window.getMapId(dem_vis)

# Style mound points
mounds_fc = sites.style(**{'color': 'red', 'pointSize': 5})
non_mounds_fc = no_mounds.style(**{'color': 'blue', 'pointSize': 5})
mounds_map_id = mounds_fc.getMapId()
non_mounds_map_id = non_mounds_fc.getMapId()

# Add custom polygons to Folium map
polygon_coords = [
    [-64.55510737563252, -15.280079196570364],
    [-64.59905268813252, -15.400596166493742],
    [-64.56884028578877, -15.480019759383284],
    [-64.49193598891377, -15.566027585198999],
    [-64.39305903578877, -15.593807054008936],
    [-64.14723994399189, -15.607034051863216],
    [-64.04424311782002, -15.563381725703367],
    [-64.01677729750752, -15.392652136250803],
    [-64.02776362563252, -15.244308062026802],
    [-64.21453120375752, -15.148888645026467],
    [-64.33263423110127, -15.144911900210754],
    [-64.46035029555439, -15.170096688239843],
    [-64.55510737563252, -15.280079196570364]
]

polygon2_coords = [
    [-64.78973040442102, -15.54077708887319],
    [-64.32281145910852, -15.60956592940302],
    [-64.06188616613977, -15.6201467829625],
    [-63.95751604895228, -15.17528470998708],
    [-63.98223528723353, -15.045355319620718],
    [-63.97674212317103, -14.756045413431329],
    [-64.08935198645227, -14.601943203100088],
    [-64.11956438879602, -14.250826692537794],
    [-64.67712054113977, -14.24017819150337],
    [-64.89135393957727, -14.375908838142403],
    [-65.00945696692102, -14.694948911113258],
    [-65.00396380285852, -14.928616482435693],
    [-64.92705950598352, -15.236244470134201],
    [-64.86938128332727, -15.408426363104162],
    [-64.78973040442102, -15.54077708887319]
]


# Setup Folium map FIRST
center = geometry.centroid().coordinates().getInfo()[::-1]
m = folium.Map(location=center, zoom_start=9)

# Add layers
folium.TileLayer(
    tiles=prob_tile_url,
    attr='GEE RF Classifier',
    name='Mound Probabilities',
    overlay=True,
    control=True
).add_to(m)

#folium.TileLayer(
#    tiles=dem_map_id['tile_fetcher'].url_format,
#    attr='DEM',
#    name='DEM 150–200m',
#    overlay=True,
#    control=True
#).add_to(m)

folium.TileLayer(
    tiles=mounds_map_id['tile_fetcher'].url_format,
    attr='Mounds',
    name='Known Mounds',
    overlay=True,
    control=True
).add_to(m)

folium.TileLayer(
    tiles=non_mounds_map_id['tile_fetcher'].url_format,
    attr='Non-Mounds',
    name='Known Non-Mounds',
    overlay=True,
    control=True
).add_to(m)

# Add polygon with yellow outline
folium.Polygon(
    locations=[(lat, lon) for lon, lat in polygon_coords],
    color='yellow',
    weight=2,
    fill=True,
    fill_color='yellow',
    fill_opacity=0.1,
    popup="Casarabe Area from Denevan"
).add_to(m)

# Add second polygon with yellow outline
folium.Polygon(
    locations=[(lat, lon) for lon, lat in polygon2_coords],
    color='blue',
    weight=2,
    fill=True,
    fill_color='blue',
    fill_opacity=0.1,
    popup="Casarabe Area from Prumers et al 2010"
).add_to(m)


import branca

# Define colorbar HTML
colorbar = branca.element.MacroElement()
colorbar._template = branca.element.Template("""
{% macro html(this, kwargs) %}
<div style="
    position: fixed; 
    bottom: 50px; left: 50px; width: 180px; height: 70px; 
    z-index:9999; font-size:14px;
    background-color: rgba(255, 255, 255, 0.75);
    padding: 10px;
    border-radius: 5px;">
    <b>Mound Probability</b><br>
    <div style="display: flex; align-items: center;">
        <span style="flex: 1;">0.55</span>
        <div style="flex: 8; height: 10px; background: linear-gradient(to right, lightgreen, yellow, orange, red); margin: 0 5px;"></div>
        <span style="flex: 1;">1.0</span>
    </div>
</div>
{% endmacro %}
""")

# Add to map
m.get_root().add_child(colorbar)


folium.LayerControl().add_to(m)
m


import matplotlib.pyplot as plt

# Retrieve feature importance
importance = classifier.explain().get('importance')
importance_dict = ee.Dictionary(importance).getInfo()

# Convert to sorted list
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
features, scores = zip(*sorted_importance)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(features[::-1], scores[::-1], color='teal')
plt.xlabel("Importance Score")
plt.title("Random Forest Feature Importance")
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()




