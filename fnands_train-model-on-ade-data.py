import geopandas as gpd
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd


!pip install rasterio


ade_train_points_2021 = gpd.read_file("/kaggle/input/ade-google-embeddings-training-data/ade_training_data_2021_2022_80_aug.geojson")
ade_train_points_2022 = gpd.read_file("/kaggle/input/ade-google-embeddings-training-data/ade_training_data_2022_2023_80_aug.geojson")
ade_train_points_2023 = gpd.read_file("/kaggle/input/ade-google-embeddings-training-data/ade_training_data_2023_2024_80_aug.geojson")
ade_train_points_2024 = gpd.read_file("/kaggle/input/ade-google-embeddings-training-data/ade_training_data_2024_2025_80_aug.geojson")


ade_train_points = gpd.GeoDataFrame(pd.concat([ade_train_points_2021, ade_train_points_2022, ade_train_points_2023, ade_train_points_2024], ignore_index=True))


ade_train_points = ade_train_points[ade_train_points['forest_value'] > 95]


ade_train_points.head()


len(ade_train_points['point_id'].unique())


embedding_cols = [f"A{str(i).zfill(2)}" for i in range(64)]

# Group by stable location, compute std dev across years
feature_stability = ade_train_points.groupby('point_id')[embedding_cols].std().mean()

# Select most stable features
stable_features = feature_stability.sort_values()
selected_features = stable_features[stable_features < 0.05].index.tolist()


stable_features.hist()


ade_train_points = ade_train_points[selected_features + ['site_type', 'point_id', 'offset_index']]


ade_train_points[ade_train_points['site_type'] == 'ADE'].head()


# Group by point_id and aggregate
result = ade_train_points.groupby('point_id').agg({
    # Average all A00-A63 columns
    **{col: 'mean' for col in ade_train_points.columns if col.startswith('A')},
    # Take first value for site_type (should be same for all points with same point_id)
    'site_type': 'first',
    # Take first geometry
    #'geometry': 'first'
})

# Convert back to GeoDataFrame and reset index
ade_train_points = gpd.GeoDataFrame(result).reset_index()


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb


from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Step 1: Prepare data
#embedding_cols = [f"A{str(i).zfill(2)}" for i in range(64)]
embedding_cols = selected_features
X = ade_train_points[embedding_cols].values
y = ade_train_points['site_type'].apply(lambda x: 1 if x == 'ADE' else 0).values
groups = ade_train_points['point_id'].values

# Step 2: Normalize (XGBoost doesn't require it, but won't hurt)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Group-aware train/test split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_scaled, y, groups=groups))
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train locations: {len(set(groups[train_idx]))}")
print(f"Test locations: {len(set(groups[test_idx]))}")

# Step 4: Define and train XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()  # handle class imbalance
)

xgb_model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["non-ADE", "ADE"]))

print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")



from shapely.geometry import box
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





buffer_m = 1000  # meters
scale = 10  # meters per pixel

#lat, lon = -2.132385, -78.105309
lat, lon = -2.50309, -64.63575
point = ee.Geometry.Point([lon, lat])
region = point.buffer(buffer_m).bounds()

# Load embedding image
image = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
    .filterDate('2021-01-01', '2022-01-01') \
    .filterBounds(point) \
    .first()

# Get pixel values in region
scale = 500  # meters
region = point.buffer(10000).bounds()  # ~2km window around your point

# Sample points every 100m
sampled = image.sample(
    region=region,
    scale=scale,
    geometries=True
)




features = sampled.getInfo()['features']

from shapely.geometry import Point
import geopandas as gpd
import pandas as pd

rows = []
for f in features:
    props = f['properties']
    lon, lat = f['geometry']['coordinates']
    props['geometry'] = Point(lon, lat)
    rows.append(props)

gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs='EPSG:4326')



gdf


# Step 1: Extract features
X = gdf[embedding_cols].values

# Step 2: Apply same scaler (optional, but consistent)
X_scaled = scaler.transform(X)

# Step 3: Predict probabilities (ADE likelihood)
logits = xgb_model.predict_proba(X_scaled)[:, 1]  # shape: (N,)

# Add to GeoDataFrame
gdf['logits'] = logits



logits_scaled = np.clip((logits * 100).round(), 0, 100).astype(np.uint8)
gdf['logits_uint8'] = logits_scaled


import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize

# Convert to UTM (for equal spacing in meters)
gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())

# Get bounds and resolution
xmin, ymin, xmax, ymax = gdf_utm.total_bounds
pixel_size = 500
width = int((xmax - xmin) / pixel_size)
height = int((ymax - ymin) / pixel_size)

transform = from_origin(xmin, ymax, pixel_size, pixel_size)

# Rasterize
shapes = zip(gdf_utm.geometry, gdf_utm['logits_uint8'])

raster = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    fill=0,
    transform=transform,
    dtype='uint8'
)

# Save to GeoTIFF
out_meta = {
    'driver': 'GTiff',
    'height': height,
    'width': width,
    'count': 1,
    'dtype': 'uint8',
    'crs': gdf_utm.crs,
    'transform': transform
}

with rasterio.open("ade_logits_500m.tif", "w", **out_meta) as dst:
    dst.write(raster, 1)



logits.max()


import math
import itertools

amazon_bounds = gpd.read_file('/kaggle/input/geographical-boundaries-of-amazonia-by-eva-et-al/amazonia_polygons.shp')

# amazon_bounds is a GeoDataFrame containing polygons that outline the AOI
amazon_poly = amazon_bounds.unary_union     # dissolve to one polygon

# Re-project to an equal-distance CRS (EPSG:3857) so 20 km really is 20 km
amazon_3857 = gpd.GeoSeries([amazon_poly], crs="EPSG:4326").to_crs("EPSG:3857")[0]
xmin, ymin, xmax, ymax = amazon_3857.bounds

tile_size_m = 20_000        # 20 km
x_steps = math.ceil((xmax - xmin)/tile_size_m)
y_steps = math.ceil((ymax - ymin)/tile_size_m)

tiles = []
for ix, iy in itertools.product(range(x_steps), range(y_steps)):
    x0 = xmin + ix*tile_size_m
    y0 = ymin + iy*tile_size_m
    x1, y1 = x0 + tile_size_m, y0 + tile_size_m
    tile_geom = box(x0, y0, x1, y1)
    if tile_geom.intersects(amazon_3857):          # keep only tiles that intersect AOI
        tiles.append(tile_geom)

tiles_gdf = gpd.GeoDataFrame(geometry=tiles, crs="EPSG:3857").to_crs("EPSG:4326")
print(f"{len(tiles_gdf)} tiles to process")



tiles_gdf.head().iloc[0].geometry.centroid.coords[0]#.to_file("tiles.geojson")


#tiles_gdf = tiles_gdf[0:500]


len(tiles_gdf)


def tile_max_logit(tile_geom, year_from="2021-01-01", year_to="2022-01-01",
                   pixel_scale=500):
    """
    Returns max ADE probability (0-1) for a single 20 × 20 km tile.
    """
    # EE region
    #region = ee.Geometry.Polygon(tile_geom.bounds)
    buffer_m = 10000  # meters

    lon, lat = tile_geom.centroid.coords[0]
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_m).bounds()
    
    # Pick first annual embedding image intersecting the tile
    image = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
               .filterDate(year_from, year_to)
               .filterBounds(region)
               .first())
    if image is None:
        print("Warning: image is None!")
        return np.nan


    # Get pixel values in region
    #pixel_scale = pixel_scale  # meters
    region = point.buffer(buffer_m).bounds()  # ~2km window around your point
    
    # Sample points every 100m
    sampled = image.sample(
        region=region,
        scale=pixel_scale,
        geometries=False
    )
    # Sample at 500 m
    #ampled = (image.sample(region=region,
    #                        scale=pixel_scale,
    #                        geometries=False)  # no need for geometry; speeds up
    #                  .limit(5000))             # safety cap (5000 < EE limit)

    # Pull to client
    try:
        array_dict = sampled.getInfo()
    except Exception as e:
        print("Warning, more than 5k features")
        # happens if >5000 features or quota; just skip
        return np.nan
    
    if not array_dict['features']:
        return np.nan
    
    # Build DataFrame
    df = pd.DataFrame([f['properties'] for f in array_dict['features']])
    X = df[embedding_cols].values
    X_scaled = scaler.transform(X)
    probs = xgb_model.predict_proba(X_scaled)[:, 1]
    return float(np.nanmax(np.array(probs)))



def worker(tile_idx, tile_geom):
    """
    Called in parallel. Returns (index, max_logit).
    Re-initialises EE in the thread the first time it’s needed.
    """
    # EE may not be initialised inside new threads → lazy init
    if not ee.data._initialized:
        ee.Initialize(ee.ServiceAccountCredentials(gee_account, gee_credentials))

    try:
        max_log = tile_max_logit(tile_geom)
    except Exception as e:
        print(f"Tile {tile_idx}: {e}")
        max_log = np.nan
    return tile_idx, max_log



from concurrent.futures import ThreadPoolExecutor, as_completed

tiles_gdf["max_logit"] = np.nan        # pre-allocate column

max_workers = min(8, len(tiles_gdf))   # stay under EE concurrency quota

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(worker, idx, geom): idx
        for idx, geom in tiles_gdf.geometry.items()
    }

    for fut in tqdm(as_completed(futures), total=len(futures)):
        idx, val = fut.result()
        tiles_gdf.at[idx, "max_logit"] = val



tiles_gdf.to_file('prediction_logits.geojson', driver='GeoJSON')


#results = []
#for idx, row in tqdm(tiles_gdf.iterrows(), total=len(tiles_gdf)):
#    max_log = tile_max_logit(row.geometry)
#    results.append(max_log)






# take the original logit column as a NumPy array
vals = tiles_gdf["max_logit"].to_numpy()

# prepare an output float array (same length)
scaled = np.empty_like(vals, dtype="float64")

# mask of finite (non-NaN, non-Inf) elements
finite_mask = np.isfinite(vals)

# ── 1. scale the finite values ─────────────────────────
scaled[finite_mask] = (vals[finite_mask] * 100).round().clip(0, 100)

# ── 2. mark non-finite as 255 ──────────────────────────
scaled[~finite_mask] = 255                  # special “nodata” marker

# ── 3. cast once everything is clean ───────────────────
tiles_gdf["max_uint8"] = scaled.astype("uint8")



#tiles_gdf["sq_max_uint8"] = (tiles_gdf["max_logit"]*tiles_gdf["max_logit"]*100).round().clip(0, 100).astype("uint8")


tiles_gdf.head()


import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

# ── 1.  CRS-aligned grid  ─────────────────────────────────────
tiles_3857 = tiles_gdf.to_crs("EPSG:3857")

xmin, ymin, xmax, ymax = tiles_3857.total_bounds
pixel_size = tile_size_m
width  = int((xmax - xmin) / pixel_size)
height = int((ymax - ymin) / pixel_size)
transform = from_origin(xmin, ymax, pixel_size, pixel_size)

# ── 2.  rasterise with fill = 255  ────────────────────────────
NODATA = 255

raster = rasterize(
    shapes=zip(tiles_3857.geometry, tiles_3857["max_uint8"].astype("uint8")),
    out_shape=(height, width),
    fill=NODATA,                 # ← background = nodata
    transform=transform,
    dtype="uint8"
)

# ── 3.  write GeoTIFF with nodata flag  ───────────────────────
meta = {
    "driver": "GTiff",
    "height": height,
    "width":  width,
    "count":  1,
    "dtype":  "uint8",
    "crs":    "EPSG:3857",
    "transform": transform,
    "nodata": NODATA            # ← register nodata in metadata
}

with rasterio.open("amazon_ade_max_50km_uint8.tif", "w", **meta) as dst:
    dst.write(raster, 1)






