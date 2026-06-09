!pip install --no-cache-dir laspy[lazrs,laszip] numpy matplotlib open3d scipy
!pip show laspy lazrs laszip
!pip install laspy matplotlib numpy scipy
!pip install rasterio
!pip install sentinelhub


import pandas as pd
from shapely.geometry import box
import geopandas as gpd
from pyproj import Transformer, CRS
import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import rasterio
import rasterio.mask
import requests
from kaggle_secrets import UserSecretsClient
from osgeo import gdal
import json, logging, hashlib, random, tempfile, shutil, subprocess
import ee
import os
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline      import make_pipeline
from sklearn import set_config
from sklearn.metrics import mean_absolute_error
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sentinelhub import SHConfig, BBox, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions
import datetime


import logging
from io import StringIO

log_stream = StringIO()
logger = logging.getLogger("GlyphTrack")
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

memory_handler = logging.StreamHandler(log_stream)
memory_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(memory_handler)

file_handler = logging.FileHandler("glyphtrack_OpenAItoZ.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logger.info("Logging setup is done.\n")


def load_secret(name):
    return UserSecretsClient().get_secret(name)


# This LiDAR process might take a while during running
laz_file_path = "/kaggle/input/lidar-brazil-geoglyph/RIB_A01_2014_laz_2.laz"
logger.info(f"Loading LiDAR file: {laz_file_path} Citation: LiDAR Surveys over Selected Forest Research Sites, Brazilian Amazon, 2008â€“2018 dos-Santos, M.N., Keller, M.M., & Morton, D.C. (2019) ORNL DAAC. https://doi.org/10.3334/ORNLDAAC/1644")

print("Reading ground-classified points...")
with laspy.open(laz_file_path) as f:
    las = f.read()
    x, y, z = las.x, las.y, las.z
    if hasattr(las, "classification"):
        ground = las.classification == 2  # ASPRS ground code
        x, y, z = x[ground], y[ground], z[ground]
print(f"âœ… Ground points extracted. ({len(x)} points)")

print("Setting up CRS transformer...")
src_crs = las.header.parse_crs()
if src_crs is None:
    src_crs = CRS.from_epsg(32719)  # fallback
transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
print("âœ… Transformer ready.")

print("Interpolating to DEM grids...")
grid_spacing = 1  # meters
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
grid_x, grid_y = np.meshgrid(
    np.arange(x_min, x_max, grid_spacing),
    np.arange(y_min, y_max, grid_spacing)
)

lidar_dem_ground = griddata((x, y), z, (grid_x, grid_y), method="linear")
print("âœ… DEMs interpolated.")

print("Converting extent to geographic coordinates...")
lon_c, lat_c = transformer.transform(
    [x_min, x_max, x_min, x_max],
    [y_min, y_min, y_max, y_max]
)
lon_min, lon_max = min(lon_c), max(lon_c)
lat_min, lat_max = min(lat_c), max(lat_c)
print("âœ… Geographic bounds computed.")

tiles_df = pd.read_csv("/kaggle/input/lidar-brazil-geoglyph/cms_brazil_lidar_tile_inventory.csv")

tiles_df["geometry"] = [
    box(row.min_lon, row.min_lat, row.max_lon, row.max_lat)
    for _, row in tiles_df.iterrows()
]

gdf_tiles = gpd.GeoDataFrame(
    tiles_df[["filename", "geometry"]],
    geometry="geometry",
    crs="EPSG:4326"
)

tile = gdf_tiles[gdf_tiles["filename"] == "RIB_A01_2014_laz_2.laz"]
tile_4326_bb = tile["geometry"].bounds.iloc[0]
bbox_coords = [tile_4326_bb["minx"], tile_4326_bb["miny"], tile_4326_bb["maxx"], tile_4326_bb["maxy"]]

from sentinelhub import CRS

logger.info(f"Loading Sentinel: {laz_file_path} Citation: LiDAR Surveys over Selected Forest Research Sites, Brazilian Amazon, 2008â€“2018 dos-Santos, M.N., Keller, M.M., & Morton, D.C. (2019) ORNL DAAC. https://doi.org/10.3334/ORNLDAAC/1644")


bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
config = SHConfig()
config.sh_client_id = load_secret("SENTINEL_CLIENT_ID")
config.sh_client_secret = load_secret("SENTINEL_CLIENT_SECRET")
resolution = 10  # meters
bbox_size = bbox_to_dimensions(bbox, resolution=resolution)

# Define evalscript to extract True Color
evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"],
    output: { bands: 3 }
  };
}

function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02];
}
"""

# Request data
request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=("2018-01-01", "2023-12-31"),
            mosaicking_order="leastCC"
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=bbox,
    size=bbox_size,
    config=config
)

# Get data and visualize
image = request.get_data()[0]


fig, axes = plt.subplots(1, 2, figsize=(16, 8))

im0 = axes[0].imshow(
    lidar_dem_ground, cmap="terrain",
    extent=(lon_min, lon_max, lat_min, lat_max),
    origin="lower"
)
axes[0].set_title("Ground Surface DEM")
axes[0].set_xlabel("Longitude (Â°)")
axes[0].set_ylabel("Latitude (Â°)")

factor = 3.5 / 255
axes[1].imshow(np.clip(image * factor, 0, 1), extent=(lon_min, lon_max, lat_min, lat_max), vmax=None)
axes[1].set_title("Canopy Level")

fig.colorbar(im0, ax=axes[0], label="Elevation (m)")

plt.tight_layout()
plt.savefig('lidar_dem_vs_satellite_data.png')
plt.show()


ee.Authenticate() # This part takes a while... To actually extract GEDI data a GEE registered Google Cloud project and authentication is needed.
ee.Initialize(project=load_secret('ee')) # Our registered project's name. Stored as Kaggle secret.


def apply_quality_mask(img):  # Applying quality mask to make sure we are using useful data.
    good = img.select('quality_flag').eq(1)   
    nodeg = img.select('degrade_flag').eq(0)
    return img.updateMask(good.And(nodeg))


gedi_path = "LARSE/GEDI/GEDI02_A_002_MONTHLY"

logger.info(f"""Loading GEDI - {gedi_path} data for coordinates: {bbox_coords} 
Citation: Dubayah, R., et al. (2021). GEDI L2A Global Footprint Data, V002 NASA LP DAAC. https://doi.org/10.5067/GEDI/GEDI02_A.002 Accessed: June 24, 2025""")

roi = ee.Geometry.Rectangle(bbox_coords)
gedi = (ee.ImageCollection(gedi_path)\
         .filterBounds(roi)\
         .filterDate('2019-04-01', '2025-06-17')\
         .map(apply_quality_mask))

img = gedi.median().select(["elev_lowestmode", "digital_elevation_model", "rh98", "modis_treecover"])

pts_fc  = img.sample(region=roi,
                     scale=25,
                     projection=img.projection(),
                     geometries=True)

url = pts_fc.getDownloadURL(
        'geojson',         
        filename='gedi_points'
)

gdf = gpd.read_file(url)

logger.info(f"""Loading NASADEM data for coordinates: {bbox_coords} 
Citation: NASADEM Merged DEM Global 1 Arc-Second, Version 001 NASA Jet Propulsion Laboratory (2021) OpenTopography. https://doi.org/10.5069/G93T9FD9 Accessed: June 24, 2025""")

DEM_TYPE   = "NASADEM"
AOI        = bbox_coords
API_KEY    = load_secret("open_topography")
OUT_DEM30 = f"{DEM_TYPE.lower()}_dem.tif"

def load_dem_data(dem_type, aoi, out_file):
    url = "https://portal.opentopography.org/API/globaldem"
    params = dict(
        demtype       = DEM_TYPE,
        west          = aoi[0],
        south         = aoi[1],
        east          = aoi[2],
        north         = aoi[3],
        outputFormat  = "GTiff",
        API_Key       = API_KEY,
    )
    print("â†’ Requesting DEMâ€¦")
    response = requests.get(url, params=params, timeout=180)
    response.raise_for_status()
    
    with open(out_file, "wb") as f:
        f.write(response.content)
        print("Written to: ", out_file)
    return out_file

out_dem_file = load_dem_data(DEM_TYPE, AOI, OUT_DEM30)

def extract_topography_features_gdal(geo_gdf, dem_path):
    with tempfile.TemporaryDirectory() as tmp:
        with rasterio.open(dem_path) as src:
            lon, lat = src.bounds.left + src.width*src.res[0]/2, \
                       src.bounds.bottom + src.height*src.res[1]/2
            utm_zone = int((lon + 180)//6) + 1
            dst_crs  = f"EPSG:{32700+utm_zone if lat<0 else 32600+utm_zone}"
        proj_dem  = os.path.join(tmp, "dem_utm.tif")
        gdal.Warp(proj_dem, dem_path, dstSRS=dst_crs)

        slope_path  = os.path.join(tmp, "slope.tif")
        aspect_path = os.path.join(tmp, "aspect.tif")
        tri_path = os.path.join(tmp, "tri.tif")
        gdal.DEMProcessing(slope_path,  proj_dem, "slope",
                           slopeFormat="degree", computeEdges=True)
        gdal.DEMProcessing(aspect_path, proj_dem, "aspect",
                           computeEdges=True)
        gdal.DEMProcessing(tri_path, proj_dem, "TRI", computeEdges=True)
        # 3. Sample values
        with rasterio.open(proj_dem) as elev,\
             rasterio.open(slope_path) as slope,\
             rasterio.open(aspect_path) as aspect, \
             rasterio.open(tri_path)  as tri:

            if geo_gdf.crs != elev.crs:
                geo_gdf = geo_gdf.to_crs(elev.crs)

            coords       = [(pt.x, pt.y) for pt in geo_gdf.geometry]
            elev_vals    = [v[0] for v in elev.sample(coords)]
            slope_vals   = [v[0] for v in slope.sample(coords)]
            aspect_vals  = [v[0] for v in aspect.sample(coords)]
            tri_vals  = [v[0] for v in tri.sample(coords)]
                 
    out = geo_gdf.copy()
    out["elevation_m"] = elev_vals
    out["slope_deg"]   = slope_vals
    out["aspect_deg"]  = aspect_vals
    out["tri"] = tri_vals
    return out

gdf_spatial = extract_topography_features_gdal(gdf, out_dem_file)
gdf_spatial["ground_diff"] = gdf_spatial["digital_elevation_model"] - gdf_spatial["elev_lowestmode"]

set_config(display='diagram')

features = [
    "rh98", "modis_treecover",
]

X = gdf_spatial[features].values
y = gdf_spatial["ground_diff"].values      #  (TanDEM-X â€“ GEDI ground)

base_est = make_pipeline(
    StandardScaler(),
    HuberRegressor(epsilon=1.5, max_iter=200)
)

model = RANSACRegressor(
    base_estimator      = base_est,
    min_samples         = 0.5,
    residual_threshold  = 3.0,
    max_trials          = 1_000,
    random_state        = 42
).fit(X, y)

huber   = model.estimator_.named_steps['huberregressor']
beta    = dict(zip(features, huber.coef_))
beta0   = huber.intercept_
print("\nRobust fit:")
for k,v in beta.items():
    print(f"  {v:7.3f} Ã— {k}")
print(f"  {beta0:7.3f}  (intercept)")

gdf_spatial["expected"] = model.predict(X)
gdf_spatial["residual"] = y - gdf_spatial["expected"]
print("\nMAE (in-liers):",
      mean_absolute_error(
          y[model.inlier_mask_],
          gdf_spatial.loc[model.inlier_mask_, "expected"]))

med   = np.nanmedian(gdf_spatial["residual"])
mad   = np.nanmedian(np.abs(gdf_spatial["residual"] - med))
sigma = 1.4826 * mad

gdf_spatial["z"] = (gdf_spatial["residual"] - med) / sigma


gdf_spatial_4326 = gdf_spatial.to_crs(4326)

fig, ax = plt.subplots(figsize=(10, 8))

# DEM background (produced in the previous step)
im = ax.imshow(
    lidar_dem_ground,
    cmap="terrain",
    extent=(lon_min, lon_max, lat_min, lat_max),
    origin="lower"
)
fig.colorbar(im, ax=ax, label="DEM elevation (m)")

# Normalise z so colour is proportional across its range
norm = Normalize(vmin=gdf_spatial_4326["z"].min(), vmax=gdf_spatial_4326["z"].max())

# Scatter points coloured by z
sc = ax.scatter(
    gdf_spatial_4326.geometry.x, gdf_spatial_4326.geometry.y,
    c=gdf_spatial_4326["z"],
    cmap="plasma",
    norm=norm,
    s=3,   
    edgecolor="k",
    linewidth=0.2,
    alpha=0.8
)

ax.set_xlabel("Longitude (Â°)")
ax.set_ylabel("Latitude (Â°)")
ax.set_title("DEM with points coloured by z")
plt.savefig('lidar_dem_with_gedi.png')
plt.tight_layout()
plt.show()


Z_BIAS_SAT      = np.percentile(np.abs(gdf_spatial_4326["z"]), 95)
SLOPE_GENTLE_MAX= np.percentile(gdf_spatial_4326["slope_deg"], 90)
TRI_LEVEL_MAX   = np.percentile(gdf_spatial_4326["tri"], 90)

def fuzzy_scores(row, z_sat, slope_max, tri_max):
    # individual fuzzy factors (clipped to 0â€“1)
    Sz   = max(min(abs(row["z"]) / z_sat, 1.0), 0.0)
    Sflat= max(0.0, 1.0 - row["slope_deg"] / slope_max)
    Stri = max(0.0, 1.0 - row["tri"]        / tri_max)

    Pglyph = (Sz + Sflat + Stri) / 3

    return pd.Series({
        "Sz": Sz, "Sflat": Sflat, "Stri": Stri,
        "Pglyph": Pglyph
    })

gdf_new_scores = gdf_spatial_4326.join(
    gdf_spatial_4326.apply(
        lambda row: fuzzy_scores(row, Z_BIAS_SAT, SLOPE_GENTLE_MAX, TRI_LEVEL_MAX),
        axis=1
    )
)

# inspect top-ranked
gdf_new_scores.sort_values("Pglyph", ascending=False).head(10)[
    ["id","Pglyph","z","slope_deg","tri","residual"]]


def high_pglyph_outliers(gdf, col="Pglyph", z_thresh=2.5):
    """
    Returns gdf with a new Boolean column `is_high_pglyph`.
    Uses a robust Z-score (MAD-based) so it adapts to any skewed distribution.
    """
    median = gdf[col].median()
    mad    = (np.abs(gdf[col] - median)).median()
    robust_z = 0.6745 * (gdf[col] - median) / mad          # 0.6745 â‰ˆ Î¦â�»Â¹(0.75)

    gdf = gdf.copy()
    gdf["is_high_pglyph"] = robust_z > z_thresh
    return gdf, robust_z

cand, rz = high_pglyph_outliers(gdf_new_scores, col="Pglyph", z_thresh=2.5)
high_p   = cand[cand["is_high_pglyph"]]

fignd, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(
    lidar_dem_ground,
    cmap="terrain",
    extent=(lon_min, lon_max, lat_min, lat_max),
    origin="lower"
)
fig.colorbar(im, ax=ax, label="DEM elevation (m)")

for _, row in high_p.iterrows():
    ax.text(
        row.geometry.x, row.geometry.y,
        str(row["id"]),
        fontsize=6,     
        ha="center", va="center",
        color="white",  
        zorder=5,    
        path_effects=[ 
            pe.withStroke(linewidth=1.2, foreground="black")
        ]
    )

ax.set_xlabel("Longitude (Â°)")
ax.set_ylabel("Latitude (Â°)")
ax.set_title("DEM with exact points on geoglyph")
plt.savefig('lidar_dem_with_gedi_on_geoglyph.png')
plt.tight_layout()
plt.show()


!pip install xarray rioxarray


import xarray as xr
from shapely.geometry import Point
from scipy.spatial import cKDTree
from geopandas.tools import sjoin_nearest
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
import joblib
from sklearn.metrics import accuracy_score
from shapely.geometry import Polygon
from sklearn.preprocessing import MinMaxScaler
from pyproj import CRS
import numpy as np, pandas as pd, joblib, lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin, clone


AOI_BBOX = [-73.991, -10.94, -66.88, -7.61]


def load_amazon_geoglyphs_sites():
    geoglyph_path = "/kaggle/input/amazon-geoglyphs-sites/amazon_geoglyphs_sites.csv"
    logger.info(f"""Loading Amazon geoglyph data, {geoglyph_path}
    Citation: Jacobs, J. Q. (2023). Ancient Human Settlement Patterns in Amazonia. Personal Academic Blog.""")
    geoglyphs_df = pd.read_csv(geoglyph_path)
    geoglyphs_df['latitude'] = pd.to_numeric(geoglyphs_df['latitude'].str.strip(), errors='coerce') 
    
    geoglyphs_gdf = gpd.GeoDataFrame(
        geoglyphs_df,
        geometry=gpd.points_from_xy(geoglyphs_df['longitude'], geoglyphs_df['latitude']),
        crs="EPSG:4326",
    )
    geoglyphs_aoi_filtered = geoglyphs_gdf.cx[AOI_BBOX[0]:AOI_BBOX[2], AOI_BBOX[1]:AOI_BBOX[3]]
    geoglyphs_gdf = geoglyphs_aoi_filtered.dropna()
    gdf_web = geoglyphs_gdf.to_crs("EPSG:3857")

    return gdf_web
    
gdf_web = load_amazon_geoglyphs_sites()


def load_main_rivers():
    RIVERS_PATH = "/kaggle/input/hydrorivers-dataset/HydroRIVERS.gdb"
    logger.info(f"""Loading HydroRivers data, {RIVERS_PATH}
    Citation: HydroRivers database. Lehner, B., Grill G. (2013): Global river hydrography and network routing: baseline data and new approaches to study the worldâ€™s large river systems. Hydrological Processes, 27(15): 2171â€“2186. Data is available at www.hydrosheds.org.""")
    rivers = gpd.read_file(RIVERS_PATH, bbox=(AOI_BBOX[0], AOI_BBOX[1], AOI_BBOX[2], AOI_BBOX[3]))
    rivers = rivers.to_crs("EPSG:4326")
    rivers_proj = rivers.to_crs("EPSG:3857")

    return rivers_proj

main_rivers_web = load_main_rivers()

def get_distances_to_rivers_between(geo_gdf, riv_gdf):
    def get_two_closest_distances(point):
        distances = riv_gdf.distance(point).sort_values().values
        # Handle edge case: fewer than 2 rivers
        if len(distances) == 0:
            return (None, None)
        elif len(distances) == 1:
            return (distances[0], None)
        else:
            return (distances[0], distances[1])

    # Apply function and unpack two distances
    dists = geo_gdf.geometry.apply(get_two_closest_distances)
    geo_gdf["dist_to_river_1_m"] = dists.apply(lambda x: x[0])
    geo_gdf["dist_to_river_2_m"] = dists.apply(lambda x: x[1])

    return geo_gdf

main_rivers_metres = main_rivers_web.to_crs("EPSG:32719")
geoglyphs_metres = gdf_web.to_crs("EPSG:32719")
river_distances_metres = get_distances_to_rivers_between(geoglyphs_metres, main_rivers_metres)


fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot 1: Distance to Nearest River
axes[0].hist(river_distances_metres["dist_to_river_1_m"], bins=40, color="mediumseagreen", edgecolor="black")
axes[0].set_xlabel("Distance to Nearest River")
axes[0].set_ylabel("Number of Geoglyphs")
axes[0].set_title("Geoglyph Distance to Nearest River")
axes[0].grid(True)

# Plot 2: Distance to Second Nearest River
axes[1].hist(river_distances_metres["dist_to_river_2_m"], bins=40, color="mediumseagreen", edgecolor="black")
axes[1].set_xlabel("Distance to Second Nearest River")
axes[1].set_title("Geoglyph Distance to Second Nearest River")
axes[1].grid(True)

plt.tight_layout()
plt.show()


p_path = "/kaggle/input/soil-phosphorus/pforms_den.nc"
logger.info(f"""Loading Soil Phosphorus data, {p_path}
    Citation: Yang, X., W.M. Post, P.E. Thornton, and A. Jain. 2014. Global Gridded Soil Phosphorus Distribution Maps at 0.5-degree Resolution. Data set. Available on-line [http://daac.ornl.gov] from Oak Ridge National Laboratory Distributed Active Archive Center, Oak Ridge, Tennessee, USA. http://dx.doi.org/10.3334/ORNLDAAC/1223""")

ds = xr.open_dataset(p_path)
df = ds.to_dataframe().reset_index()

# Filter out rows with NaNs (optional)
df = df.dropna(subset=['lat', 'lon', 'tot'])

# Create geometry column
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
p_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
aoi_geom = box(*AOI_BBOX)
p_gdf = p_gdf[p_gdf.geometry.within(aoi_geom)]

GRID_SIZE_M = 3000        # 3-km tiles
K           = 10          # use the 10 nearest stations
POWER       = 2           # IDW exponent

target_crs = CRS.from_epsg(32719)
if p_gdf.crs is None:
    p_gdf = p_gdf.set_crs(4326)

p_metric = p_gdf.to_crs(target_crs)
minx, miny, maxx, maxy = Transformer.from_crs(4326, target_crs, always_xy=True)\
                                       .transform_bounds(*AOI_BBOX)

xs = np.arange(minx, maxx, GRID_SIZE_M)
ys = np.arange(miny, maxy, GRID_SIZE_M)

tiles = gpd.GeoDataFrame(
    geometry=[box(x, y, x + GRID_SIZE_M, y + GRID_SIZE_M) for x in xs for y in ys],
    crs=target_crs
)
tiles['centroid'] = tiles.geometry.centroid
centres = np.column_stack((tiles.centroid.x.values, tiles.centroid.y.values))

coords   = np.column_stack((p_metric.geometry.x.values, p_metric.geometry.y.values))
tot_vals = p_metric['tot'].to_numpy()
tree     = cKDTree(coords)

dists, idxs = tree.query(centres, k=K)        # each row: k distances & indices
# handle centres with <K neighbours (happens only beyond data edge)
valid_mask  = np.isfinite(dists)
weights     = np.zeros_like(dists)
weights[valid_mask] = 1.0 / (dists[valid_mask] ** POWER + 1e-12)   # avoid /0

numer = (weights * tot_vals[idxs]).sum(axis=1)
denom = weights.sum(axis=1)
tiles['tot_idw'] = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom>0)

ax = tiles.boundary.plot(figsize=(8, 6), linewidth=0.1, color='lightgrey')
tiles.plot(column='tot_idw',
           cmap='Blues',
           edgecolor='none',
           legend=True,
           missing_kwds={'color': 'white'},
           ax=ax)
ax.set_title('3 km Ã— 3 km tiles â€“ IDW-smoothed total phosphorus')
ax.set_axis_off()


ax.get_figure().savefig(
    "phosphorus_tiles.png",
    dpi=300,  
    bbox_inches="tight",
    facecolor="white" 
)


DEM_TYPE   = "NASADEM"
AOI        = AOI_BBOX
API_KEY    = load_secret("open_topography")
OUT_DEM90 = f"{DEM_TYPE.lower()}_dem_part_2.tif"

logger.info(f"""Loading NASADEM data again for coordinates: {AOI_BBOX}""")
out_dem_file = load_dem_data(DEM_TYPE, AOI_BBOX, OUT_DEM90)

def extract_topography_features_gdal_new(geo_gdf, dem_path):
    with tempfile.TemporaryDirectory() as tmp:
        with rasterio.open(dem_path) as src:
            lon, lat = src.bounds.left + src.width*src.res[0]/2, \
                       src.bounds.bottom + src.height*src.res[1]/2
            utm_zone = int((lon + 180)//6) + 1
            dst_crs  = f"EPSG:{32700+utm_zone if lat<0 else 32600+utm_zone}"
        proj_dem  = os.path.join(tmp, "dem_utm.tif")
        gdal.Warp(proj_dem, dem_path, dstSRS=dst_crs)

        with rasterio.open(proj_dem) as elev:
            if geo_gdf.crs != elev.crs:
                geo_gdf = geo_gdf.to_crs(elev.crs)

            coords       = [(pt.x, pt.y) for pt in geo_gdf.geometry]
            elev_vals    = [v[0] for v in elev.sample(coords)]

    out = geo_gdf.copy()
    out["elevation_m"] = elev_vals
    return out

gdf_with_river_distances = river_distances_metres.to_crs("EPSG:4326")
gdf_spatial_new = extract_topography_features_gdal_new(gdf_with_river_distances, out_dem_file)


if tiles.crs != gdf_spatial_new.crs:
    gdf_spatial_new = gdf_spatial_new.to_crs(tiles_gdf.crs)

sites_with_tot = gpd.sjoin(
    gdf_spatial_new,
    tiles[['tot_idw', 'geometry']],
    how='left',
    predicate='within'
).drop(columns='index_right')

pos = sites_with_tot.to_crs(4326)
pos = pos.dropna()

pos["label"] = 1


xmin, ymin, xmax, ymax = AOI_BBOX

sampling_box = box(xmin, ymin, xmax, ymax)

neg_n   = len(pos) * 3
rng     = np.random.default_rng(42)

xs = rng.uniform(xmin, xmax, size=neg_n)
ys = rng.uniform(ymin, ymax, size=neg_n)
neg_pts = [Point(x, y) for x, y in zip(xs, ys)]

neg = gpd.GeoDataFrame(geometry=neg_pts, crs="EPSG:4326")
neg["label"] = 0

assert neg.geometry.within(sampling_box).all()

print(f"Generated {len(neg)} NEG points inside padded AOI.")

neg_metres = neg.to_crs("EPSG:32719")
neg_distances = get_distances_to_rivers_between(neg_metres, main_rivers_metres)
neg_4326 =  neg_distances.to_crs("EPSG: 4326")
neg_spatial = extract_topography_features_gdal_new(neg_4326, out_dem_file)

# Removing duplicates if any and extracting values that are not near to the actual positives.
dup_mask = neg_spatial.geometry.isin(pos.geometry)
neg = neg_spatial.loc[~dup_mask].reset_index(drop=True)

neg_proj = neg.to_crs("EPSG:32719")
pos_proj = pos.to_crs("EPSG:32719")

joined = sjoin_nearest(
    neg_proj,                 
    pos_proj[["geometry"]],   
    how="left",
    max_distance=50,     
    distance_col="dist_m" 
)

near_mask = joined["index_right"].notna()
neg_clean = neg.loc[~near_mask].reset_index(drop=True)

print(f"Removed {near_mask.sum()} negatives; {len(neg_clean)} remain.")

neg = neg_clean.to_crs("EPSG: 4326")
pos = pos_proj.to_crs("EPSG: 4326")

if tiles.crs != neg.crs:
    neg = neg.to_crs(tiles.crs)

neg = gpd.sjoin(
    neg,
    tiles[['tot_idw', 'geometry']],
    how='left',
    predicate='within'
).drop(columns='index_right')

neg = neg.to_crs(4326)


class PUBagger(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=30,
                 u_frac=0.6, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators   = n_estimators
        self.u_frac         = u_frac
        self.random_state   = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []

        # split once so every bag sees the same pools
        pos_mask = np.asarray(y) == 1
        X_pos, X_unl = X[pos_mask], X[~pos_mask]

        for i in range(self.n_estimators):
            rs = rng.randint(0, 2**32 - 1)
            unl_idx = rng.choice(len(X_unl),
                                 size=int(self.u_frac * len(X_unl)),
                                 replace=False)
            X_bag = pd.concat([X_pos, X_unl.iloc[unl_idx]], ignore_index=True)
            y_bag = np.concatenate([np.ones(len(X_pos), dtype=int),
                                    np.zeros(len(unl_idx), dtype=int)])

            est = clone(self.base_estimator)
            est.set_params(random_state=rs)
            est.fit(X_bag, y_bag)
            self.estimators_.append(est)
        return self

    def _bag_probas(self, X):
        return np.mean([est.predict_proba(X)[:, 1]
                        for est in self.estimators_], axis=0)

    def predict_proba(self, X):
        p = self._bag_probas(X)
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return (self._bag_probas(X) >= threshold).astype(int)


TRAIN = pd.concat([pos, neg], ignore_index=True)          # y: 1=glyph, 0=Unidentified
features = ["elevation_m", "tot_idw",
            "dist_to_river_1_m", "dist_to_river_2_m"]
X, y = TRAIN[features], TRAIN["label"]

Xtr, Xva, ytr, yva = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

lgb_params = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, n_estimators=800,
    subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
    random_state=42, n_jobs=-1
)

M, u_frac = 30, 0.6
pu_model = PUBagger(
    base_estimator=lgb.LGBMClassifier(**lgb_params),
    n_estimators=M, u_frac=u_frac, random_state=42
).fit(Xtr, ytr)

y_val_pred = pu_model.predict_proba(Xva)[:, 1]
auc_val = roc_auc_score(yva, y_val_pred)
ap_val  = average_precision_score(yva, y_val_pred)
acc_val = accuracy_score(yva, (y_val_pred >= 0.5).astype(int))

logger.info("PU-Bagging LGBM Validation Metrics:")
logger.info("  AUC : %.4f", auc_val)
logger.info("  AP  : %.4f", ap_val)
logger.info("  Acc : %.4f (threshold = 0.5)", acc_val)

print(f"PU-Bag  M={M}")
print(f"  AUC : {auc_val:.4f}")
print(f"  AP  : {ap_val:.4f}")
print(f"  Acc : {acc_val:.4f}  (thr = 0.5)")

joblib.dump(pu_model, "lightgbm_geoglyph_pu_bag.pkl")
print("Saved ensemble â†’ lightgbm_geoglyph_pu_bag.pkl")


tiles = tiles.to_crs(4326)
tiles = tiles.drop("centroid", axis=1)
centroids = tiles.to_crs(32719)
centroids.geometry = centroids.geometry.centroid
centroids = centroids.reset_index(drop=True)
print(f"Total of {len(centroids)} centroids generated.")


batch_size = 1000
results = []

for start in tqdm(range(0, len(centroids), batch_size), desc="Processing in batches"):
    end = min(start + batch_size, len(centroids))
    batch =  centroids.iloc[start:end].copy() 
    result = get_distances_to_rivers_between(batch, main_rivers_metres)
    results.append(result)

centroids_river_distances = pd.concat(results).reset_index(drop=True)


centroids_river_distances_4326 =  centroids_river_distances.to_crs("EPSG: 4326")

results = []

for start in tqdm(range(0, len(centroids_river_distances_4326), batch_size), desc="Processing in batches"):
    end = min(start + batch_size, len(centroids_river_distances_4326))
    batch = centroids_river_distances_4326.iloc[start:end]
    result = extract_topography_features_gdal_new(batch, out_dem_file)
    results.append(result)

centroids_river_distances_4326 = pd.concat(results).reset_index(drop=True)
centroids_river_distances_4326 = centroids_river_distances_4326.to_crs(4326)
centroids_river_distances_4326 = centroids_river_distances_4326[centroids_river_distances_4326["elevation_m"] != -32768].copy()  # dropping nan values


clf    = joblib.load("lightgbm_geoglyph_pu_bag.pkl")

batch   = 20_000
proba   = np.empty(len(centroids_river_distances_4326), dtype="float32")

for start in tqdm(range(0, len(centroids_river_distances_4326), batch)):
    end = min(start + batch, len(centroids_river_distances_4326))
    blk = centroids_river_distances_4326.iloc[start:end]

    X_blk = blk[[
        "elevation_m", "tot_idw", "dist_to_river_1_m", "dist_to_river_2_m"]]
    proba[start:end] = clf.predict_proba(X_blk.values)[:, 1]


!pip install contextily
import contextily as cx


gdf = centroids_river_distances_4326.copy()
gdf["proba"] = proba
hits = gdf[gdf.proba >= 0.90].copy()
gdf_geog = pos.to_crs(32719)
gdf_cand = hits.to_crs(32719)

geo_union = gdf_geog.unary_union
gdf_cand["dist_to_geog_m"] = gdf_cand.geometry.distance(geo_union)

gdf_far = gdf_cand[gdf_cand.dist_to_geog_m >= 3_000].copy() # predicted sites far from already discovered ones

gdf_far_wm = gdf_far.to_crs(3857)
fig, ax = plt.subplots(figsize=(10, 10))
gdf_far_wm.plot(ax=ax, color="red", markersize=10, label="Candidate Sites (p >= 0.90) and far from known sites (> 3 km")
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
ax.set_axis_off()
plt.legend()
plt.show()


aoi_ll = gpd.GeoDataFrame(geometry=[box(*AOI_BBOX)], crs="EPSG:4326")
aoi_m  = aoi_ll.to_crs("EPSG:32719")
minx, miny, maxx, maxy = aoi_m.total_bounds

step = 20_000
grid_polys = []
for x0 in np.arange(minx, maxx, step):
    for y0 in np.arange(miny, maxy, step):
        poly = Polygon([
            (x0,     y0),
            (x0+step,y0),
            (x0+step,y0+step),
            (x0,     y0+step)
        ])
        grid_polys.append(poly)

grid = gpd.GeoDataFrame(geometry=grid_polys, crs="EPSG:32719")
grid["tile_id"] = np.arange(len(grid))
gdf = gdf_far.to_crs("EPSG:32719")
joined = gpd.sjoin(gdf, grid, how="inner", predicate="within")
tile_stats = (joined.groupby("tile_id")
                         .agg(tile_score=("proba","mean"), 
                              n_points =("proba", "size"))
                         .reset_index())
grid = grid.merge(tile_stats, on="tile_id", how="inner")


# store the point information relates to each tile 
gdf_pts = gdf_far.to_crs("EPSG:32719")
points   = (
    gpd.sjoin(gdf_pts, # the points
              grid[["geometry", "tile_id"]],  # just geometry + id from the grid
              how="left",
              predicate="within")
    .drop(columns="index_right")
)

#  find the most optimal candidate tiles according to tile scores and how many points they include
gdf_opt = grid.copy()

scaler = MinMaxScaler()
gdf_opt[["score_norm", "points_norm"]] = scaler.fit_transform(gdf_opt[["tile_score", "n_points"]])

gdf_opt["combined_score"] = 0.5 * gdf_opt["score_norm"] + 0.5 * gdf_opt["points_norm"]

# define a threshold as one standard deviation above the mean of the combined score
# this effectively selects the top ~16% of points assuming a normal distribution,
# helping to filter out the most promising candidates
threshold = gdf_opt["combined_score"].mean() + gdf_opt["combined_score"].std()
gdf_filtered = gdf_opt[gdf_opt["combined_score"] >= threshold]

fig, ax = plt.subplots(figsize=(10,10))
gdf_opt.plot(column="combined_score", cmap="Reds", linewidth=0.2,
          edgecolor="k", legend=True, ax=ax,
          legend_kwds=dict(label="Weighted mean probability"))
cx.add_basemap(ax, crs=grid.crs.to_string())
ax.set_axis_off()
plt.title("Anomaly-candidate heat-map (20 Ã— 20 km tiles)")
plt.savefig("anomaly_candidate_heatmap.png")
plt.show()


fig, ax = plt.subplots(figsize=(10,10))
gdf_filtered.plot(column="combined_score", cmap="Reds", linewidth=0.2,
          edgecolor="k", legend=True, ax=ax,
          legend_kwds=dict(label="Weighted mean probability"))
cx.add_basemap(ax, crs=grid.crs.to_string())
ax.set_axis_off()
plt.title("Anomaly-candidate heat-map after thresholding")
plt.savefig("anomaly_candidate_heatmap_threshold.png")
plt.show()


import json
from pathlib import Path
from pydantic import BaseModel
from openai import OpenAI


openai_key = load_secret('openai_api_key')

client = OpenAI(
  api_key=openai_key
)


gdf_filtered = gdf_filtered.to_crs(4326)
points = points.to_crs(4326)

points["point_id"] = points.index 
points["coords"]   = points.geometry.apply(lambda g: [g.x, g.y])

pt_cols = ["point_id", "proba", "tot_idw", "elevation_m", "dist_to_river_1_m","dist_to_river_2_m", "coords"]

point_nests = (
    points.groupby("tile_id")[pt_cols]
          .apply(lambda df: df.to_dict(orient="records"))
          .to_dict()
)

features = []
for _, row in gdf_filtered.iterrows():
    features.append({
        "tile_id"      : int(row.tile_id),
        "tile_score"   : float(row.tile_score),
        "n_points"     : int(row.n_points),
        "bounding_box" : list(row.geometry.bounds),   # [minx, miny, maxx, maxy]
        "points"       : point_nests.get(row.tile_id, [])
    })

out_path = Path("tiles_with_points.json")
out_path.write_text(json.dumps(features, indent=2))
print(f"JSON written to {out_path.resolve()}")


prompt = f"""
You are an Amazonian field-archaeologist and remote-sensing specialist with years of experience studying pre-Columbian geoglyphs, particularly those documented in Acre, Brazil.

You receive a JSON array called **tiles**. Each element has this structure:

{{
  "tile_id": int,                  // unique ID for a 20 km Ã— 20 km tile
  "tile_score": float 0â€“1,         // mean ML confidence that the tile contains a geoglyph
  "n_points": int,                 // number of high-confidence sample points in the tile
  "bounding_box": [minLon, minLat, maxLon, maxLat],
  "points": [                      // point-level observations
    {{
      "point_id": int,
      "proba": float 0â€“1,           // per-pixel geoglyph likelihood
      "tot_idw": float,            // interpolated local phosphorus value
      "elevation_m": int,          // surface elevation (m a.s.l.)
      "dist_to_river_1_m": float,  // distance to nearest major river (m)
      "dist_to_river_2_m": float,  // distance to nearest secondary river (m)
      "coords": [lon, lat]         // WGS-84 coordinates
    }}
  ]
}}

---

### Your task

1. **Select the single most promising tile which that has the most possibility to include geoglyphs** and return its `tile_id` as **best_tile_id**.  
2. **Explain your choice** step by step, citing the numbers you use and drawing on:  
   - Landscape setting, elevation, and hydrology  
   - Spatial clustering/dispersion statistics of the points  
   - Comparisons with known Acre geoglyph contexts  
   - Relevant historical, archaeological and mythological insights

---

### Output format

Return **only** a JSON object with these keys â€” no additional text:

"best_tile_id": <int>,
"explanation": "<string with detailed reasoning>"

tiles:
{features}
"""


logger.info(f"ğŸ§  The GPT-triage prompt is : \n================\n{prompt}\n================\n")

reasoning_model = "o3"
parsing_model = "gpt-4o"

class Tile(BaseModel):
    tile_id: int

response = client.responses.create(
  model=reasoning_model,
  input=prompt
)

logger.info(f"Model {reasoning_model} returned response")
logger.info(f"Response : \n================\n{response.output_text}\n================\n")
logger.info(f"Parsing output via structrued outputs with model: {parsing_model}")

system_msg = "Extract the tile id which is given as best."
logger.info(f"Parsing system message is : \n================\n{system_msg}\n================\n")

response_parsed = client.responses.parse(
    model=parsing_model,
    tools=[{"type": "web_search_preview"}],
    input=[
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": response.output_text,
        },
    ],
    text_format=Tile,
)

logger.info(f"Parsing is completed, the response is: \n================\n{response_parsed.output_parsed}\n================\n")

tile_id = response_parsed.output_parsed
tile = features["tile_id" == tile_id.tile_id]
tile_gdf = gdf_filtered[gdf_filtered["tile_id"] == tile_id.tile_id]


bounds = tile_gdf.geometry.bounds.iloc[0]
bbox_coords = [bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"]]
tile_id = tile_id.tile_id


roi = ee.Geometry.Rectangle(bbox_coords)
gedi = (ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY")\
         .filterBounds(roi)\
         .filterDate('2019-04-01', '2025-06-17')\
         .map(apply_quality_mask))

logger.info(f"""Loading GEDI data again for coordinates: {bbox_coords}""")

img = gedi.median().select(["elev_lowestmode", "digital_elevation_model", "rh98", "modis_treecover"])

pts_fc  = img.sample(region=roi,
                     scale=25,
                     projection=img.projection(),
                     geometries=True)

url = pts_fc.getDownloadURL(
        'geojson',         
        filename='gedi_points'
)

gdf = gpd.read_file(url)
OUT_DEM = f"{DEM_TYPE.lower()}_dem_tile_{tile_id}.tif"
logger.info(f"""Loading NASADEM data again for coordinates: {bbox_coords}""")
out_dem_file = load_dem_data(DEM_TYPE, bbox_coords, OUT_DEM)

gdf_spatial = extract_topography_features_gdal(gdf, out_dem_file)
gdf_spatial = gdf_spatial[gdf_spatial["elevation_m"] != -32768].copy()
gdf_spatial = gdf_spatial[gdf_spatial["aspect_deg"] != -9999.0].copy()
gdf_spatial["ground_diff"] = gdf_spatial["digital_elevation_model"] - gdf_spatial["elev_lowestmode"]
print("Extracted topography features...")

set_config(display='diagram')

features = [
    "rh98", "modis_treecover"
]

X = gdf_spatial[features].values
y = gdf_spatial["ground_diff"].values      #  (TanDEM-X â€“ GEDI ground)

base_est = make_pipeline(
    StandardScaler(),
    HuberRegressor(epsilon=1.5, max_iter=200)
)

model = RANSACRegressor(
    base_estimator      = base_est,
    min_samples         = 0.5,
    residual_threshold  = 3.0,
    max_trials          = 1_000,
    random_state        = 42
).fit(X, y)
print("Trained the model...")
huber   = model.estimator_.named_steps['huberregressor']
beta    = dict(zip(features, huber.coef_))
beta0   = huber.intercept_
print("\nRobust fit:")
for k,v in beta.items():
    print(f"  {v:7.3f} Ã— {k}")
print(f"  {beta0:7.3f}  (intercept)")

gdf_spatial["expected"] = model.predict(X)
gdf_spatial["residual"] = y - gdf_spatial["expected"]
print("\nMAE (in-liers):",
      mean_absolute_error(
          y[model.inlier_mask_],
          gdf_spatial.loc[model.inlier_mask_, "expected"]))

med   = np.nanmedian(gdf_spatial["residual"])
mad   = np.nanmedian(np.abs(gdf_spatial["residual"] - med))
sigma = 1.4826 * mad

gdf_spatial["z"] = (gdf_spatial["residual"] - med) / sigma
gdf_spatial_4326 = gdf_spatial.to_crs(4326)

Z_BIAS_SAT      = np.percentile(np.abs(gdf_spatial_4326["z"]), 95)
SLOPE_GENTLE_MAX= np.percentile(gdf_spatial_4326["slope_deg"], 90)
TRI_LEVEL_MAX   = np.percentile(gdf_spatial_4326["tri"], 90)

gdf_new_scores = gdf_spatial_4326.join(
gdf_spatial_4326.apply(
        lambda row: fuzzy_scores(row, Z_BIAS_SAT, SLOPE_GENTLE_MAX, TRI_LEVEL_MAX),
        axis=1
    )
)

cand, rz = high_pglyph_outliers(gdf_new_scores, col="Pglyph", z_thresh=2.5)
high_p   = cand[cand["is_high_pglyph"]]


high_p_meters = high_p.to_crs("EPSG:32719")
high_p_meters_rivers = get_distances_to_rivers_between(high_p_meters, main_rivers_metres)

if tiles.crs != high_p_meters_rivers.crs:
    high_p_meters_rivers = high_p_meters_rivers.to_crs(tiles.crs)

sites_with_tot = gpd.sjoin(
    high_p_meters_rivers,
    tiles[['tot_idw', 'geometry']],
    how='left',
    predicate='within'
).drop(columns='index_right')

clf = joblib.load("lightgbm_geoglyph_pu_bag.pkl")

X = sites_with_tot[[
    "elevation_m", "tot_idw", "dist_to_river_1_m", "dist_to_river_2_m"
]]

sites_with_tot["proba"] = clf.predict_proba(X.values)[:, 1]


sites_with_tot["score"] = (sites_with_tot["Pglyph"] + sites_with_tot["proba"]) / 2


gdf = sites_with_tot

if gdf.crs != "EPSG:4326" :
    gdf = gdf.to_crs("EPSG:4326")

with rasterio.open(out_dem_file) as src:
    dem = src.read(1) 

scores = gdf["score"]

fig, ax = plt.subplots(figsize=(8, 8))

extent = [bbox_coords[0], bbox_coords[2], bbox_coords[1], bbox_coords[3]]
ax.imshow(dem, extent=extent,
          origin="upper") 

scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y,
                     s=10, 
                     c=scores, cmap="Reds",
                     edgecolor="black", linewidth=0.5,
                     alpha=0.8)

for _, row in gdf.iterrows():
    ax.text(
        row.geometry.x, row.geometry.y + 0.002,
        str(row["id"]),
        fontsize=7,     
        ha="center", va="bottom",
        color="white",  
        zorder=5,    
        path_effects=[ 
            pe.withStroke(linewidth=1.5, foreground="black")
        ]
    )

cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
cbar.set_label("Score")

ax.set_xlim(bbox_coords[0], bbox_coords[2])
ax.set_ylim(bbox_coords[1], bbox_coords[3])
ax.set_xlabel("Longitude (Â°)")
ax.set_ylabel("Latitude (Â°)")
ax.set_title("Candidate Anomaly Points on DEM")

plt.tight_layout()
plt.savefig('candidate_anomaly_points_on_dem.png')
plt.show()



cols_wanted = [
    "id", "proba", "tot_idw",
    "elevation_m",
    "dist_to_river_1_m", "dist_to_river_2_m",
    "Pglyph",
    "aspect_deg", "slope_deg", "tri",
    "z"
]

final_gdf = (
    sites_with_tot[cols_wanted + ["geometry"]]
        .assign(     
            coords=lambda df: df.geometry.apply(lambda p: [p.x, p.y])
        )
        .drop(columns="geometry")
)

dtype_map = {
    "id": "int64",
    "proba": "float64",
    "tot_idw": "float64",
    "elevation_m": "int64",
    "dist_to_river_1_m": "float64",
    "dist_to_river_2_m": "float64",
    "Pglyph": "float64",
    "aspect_deg": "float64",
    "slope_deg": "float64",
    "tri": "float64",
    "z": "float64"
}

final_gdf = final_gdf.astype(dtype_map)


prompt_final = f"""
You are an Amazonian field-archaeologist and remote-sensing specialist with years of experience studying pre-Columbian geoglyphs, particularly those documented in Acre, Brazil.

You will receive candidate anomalies that might on anthropogenic ancient sites such as geoglyphs. The points and their properties will be in the following format
      "id": int,
      "proba": float 0â€“1, 
      "tot_idw": float,
      "elevation_m": int,
      "dist_to_river_1_m": float,
      "dist_to_river_2_m": float,
      "Pglyph": float 0â€“1,
      "aspect_deg": float 0-360,
      "slope_deg": float 0-90,
      "tri": float
      "z":float
      "coords": [lon, lat]

The given properties explanation is as follows:
"id": The id of the given point.
"proba": Score from the LGBM prediction that uses tot_idw, elevation_m, dist_to_river_1_m, dist_to_river_2_m.
"tot_idw": the total phophorus value of the soil.
"elevation_m": Elevation of the point in meters.
"dist_to_river_1_m": Distance to first nearest river in meters
"dist_to_river_2_m": Distance to second nearest river in meters
"Pglyph": Probability according to z, tri and slope_deg values.
"aspect_deg": Aspect degree of the given point.
"slope_deg": Slope degree of the given point.
"tri": Terrain ruggedness index.
"z": Standardized residual elevation anomaly, defined as:
  > z = (residual - median(residuals)) / (1.4826 Ã— MAD)
  This value captures how unusual the vertical offset is between LiDAR-predicted ground elevation (GEDI) and the DEM-derived elevation, after accounting for expected vegetation-induced elevation bias. The "expected" elevation difference is modeled via a robust RANSAC regression using predictors such as rh98 (canopy height), tree cover, slope, and TRI. The z-score highlights locations with significantly depressed ground that cannot be explained by vegetation â€” an indicator of anthropogenic features like ditches or earthworks.
"coords": longitude and latitude of the given point.

I will also provide the given points with their ids on the DEM image. 

Use all of your inner knowledge as an archeologist and do a research on colonial diaries, Indigenous oral maps, past documentaries, archaeological survey papers. According to all of your findings and your internal reasoning,
analyze each point with the gathered knowledge and on both geomorphological and archeological level. 
Give the points which are most likely to be on a geoglyph and support your reasoning with solid research foundings, you can include mythological research at this point.  

Here are the points:
{final_gdf}

Remember, those points are also ones in the given DEM that their ids are shown.
"""


import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_path_dem = "candidate_anomaly_points_on_dem.png"
base64_image = encode_image(image_path_dem)
reasoning_model = "o3"
response = client.responses.create(
    model=reasoning_model,
    tools=[{"type": "web_search_preview"}],
    input=[
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": prompt_final },
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{base64_image}",
                }
            ],
        }
    ],
)

logger.info(f" The final GPT-archeology assistant prompt is : \n================\n{prompt_final}\n================\n")
logger.info(f"ğŸ“� The OpenAI model {reasoning_model} produced the following response:\n================\n{response.output_text}\n================\n")
print(response.output_text)

