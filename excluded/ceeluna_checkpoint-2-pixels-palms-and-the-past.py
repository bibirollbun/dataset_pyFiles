!pip install -q contextily rasterio pygbif


import numpy as np
import pandas as pd
import ee
import io
import os
import matplotlib.pyplot as plt
import urllib.request
from kaggle_secrets import UserSecretsClient
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
import pyproj
import contextily as cx
import requests
import rasterio
import rasterio.mask
from shapely.geometry import mapping
import seaborn as sns
from pygbif import species as species
from pygbif import occurrences as occ
from sklearn.cluster import DBSCAN
from shapely.geometry import box
import hdbscan
from shapely.ops import unary_union
from openai import OpenAI
from pydantic import BaseModel
import json, logging, hashlib, random, tempfile, shutil, subprocess
from datetime import datetime
import scipy.stats as st
import logging, hashlib, json, tempfile, shutil
from pathlib import Path
from functools import wraps
import fiona


logging.getLogger("fiona").setLevel(logging.ERROR)
fiona.Env().logging_level = logging.ERROR


SEED = 42                    # one seed to rule them all
np.random.seed(SEED)

def load_secret(name):
    return UserSecretsClient().get_secret(name)


RUN_ID = Path("artefacts") / pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
RUN_ID.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=RUN_ID / "checkpoint2.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True
)


AOI_CRS = "EPSG:4326"
METRIC_CRS = "EPSG:3857"
#Acre state
AOI_BBOX = [-73.991, -10.94, -66.88, -7.61]   # xmin, ymin   # xmax, ymax


DATASETS = {
    "amazon-geoglyphs-sites": {
        "type": "file",
        "source": "/kaggle/input/amazon-geoglyphs-sites/amazon_geoglyphs_sites.csv",
        "description": (
            "Geoglyph sites in Amazon area. Citation: Jacobs, J. Q. (2023). Ancient Human Settlement Patterns in Amazonia. Personal Academic Blog."
        )
    },
    "hydrorivers": {
        "type": "file",
        "source": "/kaggle/input/hydrorivers-dataset/HydroRIVERS.gdb",
        "description": (
            " HydroRivers database. Lehner, B., Grill G. (2013): Global river hydrography and network routing: baseline data and new approaches to study the worldâ€™s large river systems. Hydrological Processes, 27(15): 2171â€“2186. Data is available at www.hydrosheds.org."
        )
    },
    "dem": {
        "type": "url",
        "source": "https://portal.opentopography.org/API/globaldem",
        "description": (
            "SRTM DEM data for elevation. NASA Shuttle Radar Topography Mission (SRTM)(2013). Shuttle Radar Topography Mission (SRTM) Global. Distributed by OpenTopography. https://doi.org/10.5069/G9445JDF. Accessed: 2025-06-09"
        )
    },
    "prediction-sites": {
        "type": "file",
        "source": "/kaggle/input/archaeological-survey-data/submit.csv",
        "description": (
            "Ancient site data for elevation. Walker RS, Ferguson JR, Olmeda A, Hamilton MJ, Elghammer J, Buchanan B. Predicting the geographic distribution of ancient Amazonian archaeological sites with machine learning. PeerJ. 2023 Mar 31;11:e15137. doi: 10.7717/peerj.15137. PMID: 37020851; PMCID: PMC10069417."
        )
    },
    "palm-tree-data": {
        "type": "API",
        "source": "GBIF",
        "description": (
            "Via pygbif to gather palm tree occurences. Rights holders for the related AOI: Erlene Carneiro, Elizangela Oliveira, Ilune Mesquita, Edson Guilherme, The New York Botanical Garden, Instituto Nacional de Pesquisas da AmazÃ´nia, Marllus Rafael Almeida."
        ) #We could not find a related DOI, because of that we share each rights holder for the related AOI. If you have any comments regarding to this issue, please share your thoughts in the comments.
    },
    "gedi-data": {
        "type": "file",
        "source": "/kaggle/input/gedi-cluster0-2019-2021/gedi_cluster0_2019_2021.geojson",
        "description": (
            "Exported from Google Drive to Kaggle as dataset. Code snippet to download the data is shared in notebook. DOI: 10.5067/GEDI/GEDI02_A.002"
        )
    },
}


def log_dataset(name: str):
    meta = DATASETS[name]
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            gdf = func(*args, **kwargs)
            ds_type = meta["type"]
            src     = meta["source"]
            desc    = meta["description"]
            logging.info(
                "DATASET | name=%s | type=%s | source=%s | desc=%s",
                name, ds_type, src, desc
            )
            return gdf
        return wrapper
    return decorator


@log_dataset("amazon-geoglyphs-sites")
def load_amazon_geoglyphs_sites():
    geoglyphs_df = pd.read_csv(DATASETS["amazon-geoglyphs-sites"]["source"])
    geoglyphs_df['latitude'] = pd.to_numeric(geoglyphs_df['latitude'].str.strip(), errors='coerce') 
    
    geoglyphs_gdf = gpd.GeoDataFrame(
        geoglyphs_df,
        geometry=gpd.points_from_xy(geoglyphs_df['longitude'], geoglyphs_df['latitude']),
        crs=AOI_CRS,
    )
    geoglyphs_aoi_filtered = geoglyphs_gdf.cx[AOI_BBOX[0]:AOI_BBOX[2], AOI_BBOX[1]:AOI_BBOX[3]]
    geoglyphs_gdf = geoglyphs_aoi_filtered.dropna()
    gdf_web = geoglyphs_gdf.to_crs(METRIC_CRS)

    return gdf_web
    
gdf_web = load_amazon_geoglyphs_sites()


fig, ax = plt.subplots(figsize=(10, 10))
gdf_web.plot(ax=ax, alpha=0.7, edgecolor='k')

cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

ax.set_axis_off()
plt.tight_layout()
plt.show()


@log_dataset("hydrorivers")
def load_main_rivers():
    RIVERS_PATH = DATASETS["hydrorivers"]["source"]
    
    rivers = gpd.read_file(RIVERS_PATH, bbox=(AOI_BBOX[0], AOI_BBOX[1], AOI_BBOX[2], AOI_BBOX[3]))
    rivers = rivers.to_crs(AOI_CRS)
    rivers_proj = rivers.to_crs(METRIC_CRS)
    
    # Filter only the main rivers (order â‰¤ 6)
    main_rivers = rivers[rivers["ORD_FLOW"].astype(float) <= 6].copy()
    
    # Reproject to Web Mercator for plotting with basemap
    main_rivers_web = main_rivers.to_crs(METRIC_CRS)

    return main_rivers_web

main_rivers_web = load_main_rivers()


fig, ax = plt.subplots(figsize=(10, 10))
main_rivers_web.plot(ax=ax, linewidth=1.0, color="blue", label="Main Rivers")
gdf_web.plot(ax=ax, color="red", markersize=10, label="Geoglyphs")
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
ax.set_axis_off()
plt.legend()
plt.show()


def get_distance_to_rivers(geo_gdf, riv_gdf):
    geo_gdf["dist_to_river_m"] = geo_gdf.geometry.apply(
        lambda point: riv_gdf.distance(point).min()
    )

    return geo_gdf


gdf_web = get_distance_to_rivers(gdf_web, main_rivers_web)


plt.figure(figsize=(10, 6))
plt.hist(gdf_web["dist_to_river_m"] / 1000, bins=40, color="mediumseagreen", edgecolor="black")
plt.xlabel("Distance to Nearest River (km)")
plt.ylabel("Number of Geoglyphs")
plt.title("Geoglyph Distance to Nearest River")
plt.grid(True)
plt.tight_layout()
plt.show()


RIVERS_BUFFER = 9_000


buffers_9_km = main_rivers_web.buffer(RIVERS_BUFFER)
buffers_gdf = gpd.GeoDataFrame(geometry=buffers_9_km, crs=METRIC_CRS)


TILE_SIZE_KM = 9
OUT_GRID     = "tiles_9_km.gpkg"

def generate_tiles(tile_size_km, out_grid, in_gdf):
    in_gdf = in_gdf.to_crs(AOI_CRS)
    lon_min, lat_min, lon_max, lat_max = AOI_BBOX[0], AOI_BBOX[1], AOI_BBOX[2], AOI_BBOX[3]
    tile_deg = tile_size_km * 0.008333
    
    tiles = []
    tid   = 0
    lat = lat_max
    
    footprint = in_gdf.unary_union
    
    while lat - tile_deg > lat_min:
        lon = lon_min
        while lon + tile_deg < lon_max:
            geom = box(lon, lat - tile_deg, lon + tile_deg, lat)
            if geom.intersects(footprint):
                tiles.append({"tile_id": tid, "geometry": geom})
            tid += 1
            lon += tile_deg
        lat -= tile_deg
    
    tiles_gdf = gpd.GeoDataFrame(tiles, crs=AOI_CRS, geometry="geometry")
    print(f"â€¢ {len(tiles_gdf):,} tiles intersect the 9-km buffer")

    tiles_gdf.to_file(out_grid, driver="GPKG")
    print(f"âœ“ Grid written â†’ {out_grid}")
    tiles_gdf = tiles_gdf.to_crs(METRIC_CRS)
    return tiles_gdf


tiles_gdf = generate_tiles(TILE_SIZE_KM, OUT_GRID, buffers_gdf)


tiles_web = tiles_gdf.to_crs(METRIC_CRS)
fig, ax = plt.subplots(figsize=(10, 10))

# Tile outlines on top
tiles_web.plot(ax=ax,
               linewidth=0.6,
               edgecolor="red",
               facecolor="none")

# Basemap
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

ax.set_title("9 Ã— 9 km candidate tiles (within 9 km of order-6 rivers)",
             fontsize=13)
ax.set_axis_off()
plt.tight_layout()
plt.show()


def tag_tiles_by_geoglyphs(tiles_gdf, geoglyphs_gdf):
    tiles_gdf = tiles_gdf.to_crs(AOI_CRS)
    geoglyphs_gdf = geoglyphs_gdf.to_crs(AOI_CRS)
    
    hits = gpd.sjoin(
        tiles_gdf[["tile_id", "geometry"]],
        geoglyphs_gdf[["name", "geometry"]],
        how="left",           # keep all tiles
        predicate="intersects"
    ).drop(columns="index_right")
    
    summary = (
        hits.groupby("tile_id")
            .agg(n_geoglyphs=("name", "count"))      # count non-null hits
            .reset_index()
    )
    
    tiles_tagged = tiles_gdf.merge(summary, on="tile_id", how="left")
    tiles_tagged["n_geoglyphs"] = tiles_tagged["n_geoglyphs"].fillna(0).astype(int)
    tiles_tagged["has_geoglyph"] = tiles_tagged["n_geoglyphs"] > 0
    tiles_hit = tiles_tagged[tiles_tagged["has_geoglyph"] == True].copy()

    print(f"{len(tiles_hit)} tiles contain â‰¥1 geoglyph")
    
    tiles_tagged.to_file("tiles_tagged_geoglyphs.gpkg", driver="GPKG")

    return tiles_tagged


tiles_tagged = tag_tiles_by_geoglyphs(tiles_gdf, gdf_web)

tiles_hit = tiles_tagged[tiles_tagged["has_geoglyph"] == True].copy()
tiles_not_hit = tiles_tagged[tiles_tagged["has_geoglyph"] == False].copy()

tiles_hit_web = tiles_hit.to_crs(METRIC_CRS)
tiles_not_hit_web = tiles_not_hit.to_crs(METRIC_CRS)

fig, ax = plt.subplots(figsize=(10, 10))

tiles_hit_web.plot(ax=ax,
                   edgecolor="green",
                   facecolor="none",
                   linewidth=1.2,
                   label="Tiles with geoglyphs")

tiles_not_hit_web.plot(ax=ax,
                      edgecolor="red",
                      facecolor="none",
                      linewidth=1.2,
                      label="Tiles without geoglyphs")

cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

ax.set_title("Tiles Containing Confirmed Geoglyphs", fontsize=14)
ax.set_axis_off()
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()


DEM_TYPE   = "SRTMGL3"
AOI        = AOI_BBOX
API_KEY    = load_secret("open_topography")
OUT_DEM90 = f"{DEM_TYPE.lower()}_90m.tif"

@log_dataset("dem")
def load_dem_data(dem_type, aoi, out_file):
    url = DATASETS["dem"]["source"]
    params = dict(
        demtype       = DEM_TYPE,
        west          = AOI[0],
        south         = AOI[1],
        east          = AOI[2],
        north         = AOI[3],
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


out_dem_file = load_dem_data(DEM_TYPE, AOI_BBOX, OUT_DEM90)

with rasterio.open(out_dem_file) as src:
    dem = src.read(1)
    plt.imshow(dem, cmap="terrain")
    plt.colorbar(label="Elevation (m)")
    plt.title("DEM from OpenTopography")
    plt.show()


def _masked_vals(geom, raster_path):
    with rasterio.open(raster_path) as r:
        arr, _ = rasterio.mask.mask(r, [mapping(geom)],
                                    crop=True, filled=False)
    return arr[0]


def append_mean_elev(gdf, out_dem_file):
    mean_slope, mean_elev = [], []
    for g in tiles_tagged.geometry:
        mean_elev.append(np.nanmean(_masked_vals(g, out_dem_file)))

    new_gdf = gdf.copy()
    new_gdf["mean_elev_m"] = mean_elev
    return new_gdf

tiles_tagged_with_elev = append_mean_elev(tiles_tagged, OUT_DEM90)


@log_dataset("prediction-sites")
def load_archeological_prediction_data():
    submit_df = pd.read_csv(DATASETS["prediction-sites"]["source"])
    return submit_df


submit_df = load_archeological_prediction_data()
submit_df.keys()


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(data=submit_df, x="wc2.1_30s_elev", bins=40, kde=True, color="teal")

plt.title("Elevation Distribution (wc2.1_30s_elev)", fontsize=14)
plt.xlabel("Elevation (meters)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()


def generate_tiles_after_dem(tiles_tagged_with_elev):
    elev = submit_df["wc2.1_30s_elev"].astype(float)

    min_elev = elev.min()
    
    # Mean of top 10 %
    cutoff   = elev.quantile(0.90)
    mean_top = elev[elev >= cutoff].mean()
    
    print(f"Mean of top 10 %       : {mean_top:7.2f} m")

    tiles_after_dem = (
        tiles_tagged_with_elev
        .query("mean_elev_m >= @min_elev "
               "and mean_elev_m <= @mean_top")
        .copy()
    )
    
    print(f"âœ“ {len(tiles_after_dem)} / {len(tiles_tagged)} tiles survive elev triage")
    
    tiles_after_dem_web = tiles_after_dem.to_crs(METRIC_CRS)

    return tiles_after_dem_web

tiles_after_dem = generate_tiles_after_dem(tiles_tagged_with_elev)

fig, ax = plt.subplots(figsize=(10, 10))

tiles_after_dem.plot(ax=ax,linewidth=0.6,edgecolor="red",facecolor="none")
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

ax.set_title("9 Ã— 9 km candidate tiles after DEM processing",
             fontsize=13)
ax.set_axis_off()
plt.tight_layout()
plt.show()


@log_dataset("palm-tree-data")
def load_palm_tree_dataset():
    splist = ['Bactris gasipaes']
    keys = [ species.name_backbone(x)['usageKey'] for x in splist ]
    taxon_key = keys[0]

    lon_min, lat_min, lon_max, lat_max = AOI[0], AOI[1], AOI[2], AOI[3]
    geom_wkt = (
        f"POLYGON(({lon_min} {lat_max}, {lon_max} {lat_max}, "
        f"{lon_max} {lat_min}, {lon_min} {lat_min}, {lon_min} {lat_max}))"
    )

    records = []
    offset  = 0
    while True:
        page = occ.search(
            taxonKey      = taxon_key,
            geometry      = geom_wkt,
            hasCoordinate = True,
            limit         = 300,
            offset        = offset
        )
        records.extend(page["results"])
        if len(page["results"]) < 300:
            break
        offset += 300

    df  = pd.DataFrame([
        {
            "lat":  r["decimalLatitude"],
            "lon":  r["decimalLongitude"],
            "date": r.get("eventDate"),
            "basis": r.get("basisOfRecord"),
            "id":   r["key"],
        }
        for r in records
    ])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat, crs=AOI_CRS)
    )

    return gdf


palm_gdf = load_palm_tree_dataset()
print(f"{len(palm_gdf):,} *Bactris gasipaes* points inside Acre")
palm_gdf


palm_gdf_web = palm_gdf.to_crs(METRIC_CRS)

fig, ax = plt.subplots(figsize=(8, 8))
palm_gdf_web.plot(
    ax=ax,
    markersize=15,
    color="goldenrod",
    edgecolor="black",
    linewidth=0.4,
    alpha=0.9)

cx.add_basemap(
    ax,
    crs=palm_gdf_web.crs,
    source=cx.providers.CartoDB.Positron,
    attribution_size=8
)

ax.set_title("Bactris gasipaes occurrences in Acre (GBIF)", fontsize=13, pad=12)
ax.set_axis_off()                      
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(10, 10))

tiles_after_dem.plot(ax=ax,linewidth=0.6, edgecolor="blue",facecolor="none")

geoglyphs_proj = gdf_web.to_crs(METRIC_CRS)

geoglyphs_proj.plot(ax=ax, color="red", markersize=5, label="Geoglyphs")

palm_gdf_web.plot(ax=ax, color="green", markersize=30,
                  facecolor="green")

cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

ax.set_title("Tiles after DEM processing with Palm Trees",
             fontsize=13)
ax.set_axis_off()
plt.tight_layout()
plt.show()


PALMS_BUFFER = 20_000

buffers_20_km = palm_gdf_web.buffer(PALMS_BUFFER)
buffers_gdf_20_km = gpd.GeoDataFrame(geometry=buffers_20_km, crs=METRIC_CRS)


TILE_SIZE_KM = 9
BUFFER_GDF    = buffers_gdf_20_km.to_crs(AOI_CRS)
DEM_GDF       = tiles_after_dem.to_crs(AOI_CRS)


def generate_palm_dem_intersected_tiles(aoi, tile_size, buffer_gdf, dem_gdf):
    lon_min, lat_min, lon_max, lat_max = aoi[0], aoi[1], aoi[2], aoi[3]
    tile_deg = tile_size * 0.008333
    
    buf_union = buffer_gdf.unary_union
    dem_union = dem_gdf.unary_union
    
    tiles = []
    tid   = 0
    lat   = lat_max
    
    while lat - tile_deg > lat_min:
        lon = lon_min
        while lon + tile_deg < lon_max:
            geom = box(lon, lat - tile_deg, lon + tile_deg, lat)
            if geom.intersects(buf_union) and geom.intersects(dem_union):
                tiles.append({"tile_id": tid, "geometry": geom})
            tid += 1
            lon += tile_deg
        lat -= tile_deg
    
    tiles_gdf = gpd.GeoDataFrame(tiles, crs=AOI_CRS, geometry="geometry")
    print(f"â€¢ {len(tiles_gdf):,} tiles intersect both buffers and DEMâ€�filtered tiles")
    
    tiles_gdf.to_file(OUT_GRID, driver="GPKG")
    print(f"âœ“ Grid written â†’ {OUT_GRID}")
    return tiles_gdf


palm_dem_intersected_tiles = generate_palm_dem_intersected_tiles(AOI, TILE_SIZE_KM, BUFFER_GDF, DEM_GDF)


fig, ax = plt.subplots(figsize=(10, 10))

palm_dem_intersected_tiles.plot(ax=ax, linewidth=0.6, edgecolor="blue",  facecolor="none")
cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, crs=AOI_CRS)

ax.set_title("20 km Buffers Around Detected Palm Trees")
ax.set_axis_off()
plt.tight_layout()
plt.show()


def clusters_from_tiles_gdf(tiles_gdf):
    tiles_gdf = tiles_gdf.sort_values("tile_id").reset_index(drop=True) #sort tile ids to ensure persistent results
    
    tiles_m   = tiles_gdf.to_crs(METRIC_CRS)
    centroids = np.array([[p.x, p.y] for p in tiles_m.geometry.centroid])
    
    clustering = DBSCAN(eps=20_000, min_samples=1).fit(centroids)
    tiles_gdf["cluster"] = clustering.labels_
    
    clusters = (
        tiles_gdf
        .dissolve(by="cluster", as_index=True)   
        .reset_index()                           
    )
    
    print(f"{len(clusters)} clusters built from {len(tiles_gdf)} tiles")
    
    clusters_4326 = clusters.to_crs(AOI_CRS)
    bounds = clusters_4326.geometry.bounds
    clusters_bb = pd.concat([clusters_4326.reset_index(drop=True), bounds], axis=1)
    
    # Rename for clarity
    clusters_bb = clusters_bb.rename(
        columns={"minx": "min_lon", "miny": "min_lat",
                 "maxx": "max_lon", "maxy": "max_lat"}
    )
    
    # Add the actual rectangle geometry
    clusters_bb["bbox_geom"] = clusters_bb.apply(
        lambda r: box(r.min_lon, r.min_lat, r.max_lon, r.max_lat), axis=1
    )
    clusters_bb = gpd.GeoDataFrame(clusters_bb,
                                   geometry="bbox_geom",
                                   crs=AOI_CRS)
    return clusters_bb


clusters_bb = clusters_from_tiles_gdf(palm_dem_intersected_tiles)


fig, ax = plt.subplots(figsize=(6, 6))
bb_clip = clusters_bb.to_crs(METRIC_CRS)
bb_clip.boundary.plot(ax=ax, edgecolor="firebrick", linewidth=1.5)
cx.add_basemap(ax, crs=bb_clip.crs)
ax.set_axis_off()
plt.tight_layout()
plt.show()


min_lon = clusters_bb.loc[[0]]["min_lon"].iloc[0]
min_lat = clusters_bb.loc[[0]]["min_lat"].iloc[0]
max_lon = clusters_bb.loc[[0]]["max_lon"].iloc[0]
max_lat = clusters_bb.loc[[0]]["max_lat"].iloc[0]
aoi = [min_lon, min_lat, max_lon, max_lat]
aoi


# This part is for downloading GEDI data for related AOI. You can authenticate with your own Google Cloud account after registering to GEE.

ee.Authenticate()
ee.Initialize(project=load_secret('ee'))
aoi = ee.Geometry.Rectangle(aoi)

GEDI_ID  = "LARSE/GEDI/GEDI02_A_002_MONTHLY"
gedi_img = (ee.ImageCollection(GEDI_ID)
              .filterBounds(aoi)
              .filterDate("2019-04-01", "2021-03-31")
              .select(["rh95", "rh100", "quality_flag"])
              .median())

samples_fc = (gedi_img.sample(region = aoi,
                              scale = 25,          # GEDI footprint
                              geometries = True)
                        .filter(ee.Filter.eq("quality_flag", 1))
                        .map(lambda f: f.set("cluster", 0)))   # tag cluster ID

task = ee.batch.Export.table.toDrive(
        collection     = samples_fc,
        description    = "gedi_cluster0",
        folder         = "EE_exports",
        fileNamePrefix = "gedi_cluster0_2019_2021",
        fileFormat     = "GeoJSON")
task.start()
print("â�³  Export started â€“ monitor the EE Tasks tabâ€¦")


@log_dataset("gedi-data")
def gedi_clustered_anomalies():
    gedi0 = gpd.read_file(DATASETS["gedi-data"]["source"])
    gedi0["canopy_h"] = gedi0["rh100"].astype(float) - gedi0["rh95"].astype(float)
    gedi0 = gedi0.dropna(subset=["canopy_h"])
    
    coords = np.vstack([gedi0.geometry.y,
                        gedi0.geometry.x,
                        gedi0.canopy_h]).T
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
    labels    = clusterer.fit_predict(coords)
    
    gedi0["outlier_score"] = clusterer.outlier_scores_
    gedi0["is_anomaly"]    = gedi0.outlier_score > 0.9
    
    gedi0.to_file("gedi_cluster0_anomalies.gpkg", driver="GPKG") 
    anomalies = gedi0[gedi0.is_anomaly == True]
    g3857 = anomalies.to_crs(METRIC_CRS)
    return g3857


anomalies_3857 = gedi_clustered_anomalies()
anomalies_3857


fig, ax = plt.subplots(figsize=(8, 8))

anomalies_3857.plot(ax=ax,
                    column = "outlier_score",
                    cmap = "Reds",
                    markersize = 15,
                    alpha = 0.8,
                    legend = True)

cx.add_basemap(ax,crs=METRIC_CRS, source=cx.providers.CartoDB.Positron)
ax.set_axis_off()
plt.tight_layout()
plt.savefig("cluster0_anomalies.png", dpi=300)
plt.show()


def clustered_anomaly_points(anomalies_3857):
    anomalies_3857 = (
        anomalies_3857
        .sort_values("id")
        .reset_index(drop=True)
    )
    coords = np.vstack(anomalies_3857.geometry.apply(lambda p: (p.x, p.y)))
    db = DBSCAN(eps=4000, min_samples=1).fit(coords)
    anomalies_3857["cluster"] = db.labels_
    keep = anomalies_3857[anomalies_3857.cluster != -1].copy()
    clusters = (
        keep.groupby("cluster", sort=False)
            .agg(
                pt_count      = ("cluster", "size"),
                geom          = ("geometry", lambda g: unary_union(g.tolist())),
                rh100_vals    = ("rh100", list),
                rh95_vals     = ("rh95",  list),
                canopy_h_vals = ("canopy_h", list),
                outlier_vals  = ("outlier_score", list),
            )
            .sort_values("pt_count", ascending=False)
            .reset_index()
            .rename(columns={"geom": "geometry"})
    )
    
    clusters_gdf = (
    clusters.reset_index()                         
            .rename(columns={"geom": "geometry"}) 
    )
    
    clusters_gdf = gpd.GeoDataFrame(clusters_gdf,
                                    geometry="geometry",
                                    crs=METRIC_CRS)
    
    return clusters_gdf


clusters_gdf = clustered_anomaly_points(anomalies_3857)
clusters_gdf


BUFFER_CLUSTER_HU = 1_000

def make_bbox(row):
    geom = row.geometry
    if row.pt_count == 1:
        # a lone GEDI spike â†’ circular buffer then bbox
        geom = geom.buffer(BUFFER_CLUSTER_HU)
    else:
        # >1 points â†’ tight hull around the group
        geom = geom.convex_hull.buffer(BUFFER_CLUSTER_HU)
    return geom.envelope      # final rectangle (Polygon)

clusters_gdf["footprint"] = clusters_gdf.apply(make_bbox, axis=1)


footprints = clusters_gdf.set_geometry("footprint")

fig, ax = plt.subplots(figsize=(10, 10))

clusters_gdf.set_geometry("geometry").plot(
    ax=ax,
    column="cluster",
    categorical=True,
    markersize=4,
    alpha=0.6,
)

footprints.boundary.plot(ax=ax, linewidth=2)

cx.add_basemap(
    ax,
    source=cx.providers.CartoDB.Positron,
    crs=clusters_gdf.crs,
    attribution=False
)

ax.set_axis_off()
ax.set_title("GEDI anomaly clusters (EPSG:3857)", fontsize=14)
plt.tight_layout()
plt.show()


def get_elev_for_bb(dem_path, clusters_gdf):
    with rasterio.open(dem_path) as src:
        clusters_in_raster_crs = clusters_gdf.to_crs(src.crs)

        elev_lists = []
        for geom in clusters_in_raster_crs.geometry:
            if geom.is_empty:
                elev_lists.append([])
                continue

            if isinstance(geom, Point):
                pts = [geom]
            elif isinstance(geom, MultiPoint):
                pts = sorted(list(geom.geoms), key=lambda p: (p.x, p.y))
            else:
                raise TypeError(f"Unsupported geometry type: {geom.geom_type}")

            coords = [(p.x, p.y) for p in pts]
            elev = [val[0] if val[0] is not None else np.nan
                    for val in src.sample(coords)]
            elev_lists.append(elev)
        out = clusters_gdf.copy()
    out["elev_m_vals"] = elev_lists
    return out


clusters_with_elev = get_elev_for_bb(OUT_DEM90, clusters_gdf)

print(clusters_with_elev.head())


clusters_with_elev = clusters_with_elev.to_crs(AOI_CRS)

clusters_with_elev["footprint"] = (
    gpd.GeoSeries(clusters_with_elev["footprint"], crs=METRIC_CRS)
    .to_crs(AOI_CRS)
)

clusters_with_elev = clusters_with_elev.set_geometry("footprint")
clusters_with_elev


def generate_bounding_box_column(clusters_with_elev):
    bounds_df = clusters_with_elev['footprint'].bounds
    clusters_with_elev["bounding_box_arr"] = bounds_df.apply(
        lambda r: [r.minx, r.miny, r.maxx, r.maxy], axis=1
    )

    return clusters_with_elev


final_anomalies = generate_bounding_box_column(clusters_with_elev)
final_anomalies


final_anomalies_string = final_anomalies.to_string(index=False)
logging.info("SHOWCASE FINAL_ANOMALY_CLUSTERS:\n%s",final_anomalies_string)


def generate_prompt(gdf_to_gpt):
    return f"""
You are **ADE-Scout-GPT**, an archaeology-oriented geo-analyst.

Task  
Transform every DBSCAN-labelled cluster of GEDI L2A canopy-height anomalies (Acre, Brazil) into clear, actionable insights that guide the next stage of remote-sensing anomaly detection.

Data context  
â€¢ CRS = EPSG:3857 (metres)  
â€¢ Table columns  
  - id  
  - clusterâ€ƒ(DBSCAN label)  
  - geometryâ€ƒ(point or multipoint)  
  - pt_countâ€ƒ(points in footprint bbox)  
  - rh100, rh95, canopy_h (= rh100 âˆ’ rh95)  
  - outlierâ€ƒ(score 0â€“1)  
  - footprintâ€ƒ(polygon)  
  - elev_mâ€ƒ(DEM ground elevation, m asl)
  - bounding-box (bounding box array for footprint)

Output for **each cluster**  
1. Summarise canopy-height anomaly pattern and spatial extent.  
2. State confidence level (low / medium / high) based on pt_count and outlier scores.  
3. Suggest the most relevant follow-up remote-sensing actions (e.g., LiDAR strip flight, multispectral revisit, field survey).  
4. Flag any potential archaeological significance or data quality concerns.

Table  
{gdf_to_gpt}
"""


openai_key = load_secret('openai_api_key')

client = OpenAI(
  api_key=openai_key
)


prompt = generate_prompt(final_anomalies)


def gpt_for_deductions(prompt, client):
    completion = client.chat.completions.create(
      model="gpt-4.1",
      store=True,
      messages=[
        {"role": "user", "content": prompt}
      ]
    )

    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


gpt_response = gpt_for_deductions(prompt, client)


logging.info("SHOWCASE PROMPT:\n%s", prompt)
        
raw_json = json.dumps(gpt_response, indent=2)
logging.info("SHOWCASE GPT_RESPONSE:\n%s", raw_json)


def verify_bboxes(initial_bboxes, current_bboxes, tol=1e-6):
    assert len(initial_bboxes) == len(current_bboxes), (
        f"Number of bounding boxes changed: "
        f"{len(initial_bboxes)} â†’ {len(current_bboxes)}"
    )
    
    arr0 = np.array(initial_bboxes, dtype=float)
    arr1 = np.array(current_bboxes, dtype=float)
    
    if not np.allclose(arr0, arr1, atol=tol):
        diffs = np.abs(arr0 - arr1)
        idx, coord = np.where(diffs > tol)[0][0], np.where(diffs > tol)[1][0]
        raise AssertionError(
            f"BBox #{idx} coordinate {['min_lon','min_lat','max_lon','max_lat'][coord]} "
            f"drifted by {diffs[idx,coord]:.6f}Â° (> {tol})"
        )
    
    print("All bounding boxes match within Â±{:.6f}Â°".format(tol))

def verification(n=3):
    for i in range(n):
        geoglyph_gdf_web = load_amazon_geoglyphs_sites()
        rivers_gdf_web = load_main_rivers()
        buffers_9_km = rivers_gdf_web.buffer(RIVERS_BUFFER)
        buffers_gdf = gpd.GeoDataFrame(geometry=buffers_9_km, crs=METRIC_CRS)
        tiles_gdf = generate_tiles(TILE_SIZE_KM, OUT_GRID, buffers_gdf)
        tiles_tagged = tag_tiles_by_geoglyphs(tiles_gdf, geoglyph_gdf_web)
        tiles_tagged_with_elev = append_mean_elev(tiles_tagged, OUT_DEM90)
        submit_df = load_archeological_prediction_data() #we did not use directly this data for the implementation part, but we used it for deciding the max mean elevation value.
        tiles_after_dem = generate_tiles_after_dem(tiles_tagged_with_elev)
        palm_gdf = load_palm_tree_dataset()
        palm_gdf_web = palm_gdf.to_crs(METRIC_CRS)
        buffers_20_km = palm_gdf_web.buffer(PALMS_BUFFER)
        buffers_gdf_20_km = gpd.GeoDataFrame(geometry=buffers_20_km, crs=METRIC_CRS)
        buffers_gdf_20_km_web = buffers_gdf_20_km.to_crs(AOI_CRS)
        tiles_after_dem_web = tiles_after_dem.to_crs(AOI_CRS)
        palm_dem_intersected_tiles = generate_palm_dem_intersected_tiles(AOI, TILE_SIZE_KM, buffers_gdf_20_km_web, tiles_after_dem_web)
        clusters_bb = clusters_from_tiles_gdf(palm_dem_intersected_tiles)
        min_lon = clusters_bb.loc[[0]]["min_lon"].iloc[0]
        min_lat = clusters_bb.loc[[0]]["min_lat"].iloc[0]
        max_lon = clusters_bb.loc[[0]]["max_lon"].iloc[0]
        max_lat = clusters_bb.loc[[0]]["max_lat"].iloc[0]
        aoi = [min_lon, min_lat, max_lon, max_lat]
        # After this part, reviewer can use GEE service to download the GEDI dataset using aoi value. Since it requires transferring data from Google Drive to here, 
        # we used already existing data we previously downloaded as dataset for convenience.
        anomalies_3857 = gedi_clustered_anomalies()
        clusters_gdf = clustered_anomaly_points(anomalies_3857)
        clusters_gdf["footprint"] = clusters_gdf.apply(make_bbox, axis=1)
        clusters_with_elev = get_elev_for_bb(OUT_DEM90, clusters_gdf)
        clusters_with_elev = clusters_with_elev.to_crs(AOI_CRS)
    
        clusters_with_elev["footprint"] = (
            gpd.GeoSeries(clusters_with_elev["footprint"], crs=METRIC_CRS)
            .to_crs(AOI_CRS)
        )
        
        clusters_with_elev = clusters_with_elev.set_geometry("footprint")

        final_anomalies = generate_bounding_box_column(clusters_with_elev)

        prompt = generate_prompt(final_anomalies)
        logging.info("ITER %d PROMPT:\n%s", i, prompt)
        
        gpt_response = gpt_for_deductions(prompt, client)

        logging.info("ITER %d FINAL_ANOMALY_CLUSTERS:\n%s", i, final_anomalies.to_string(index=False))

        if i == 0:
            initial_bboxes = final_anomalies["bounding_box_arr"].tolist()
        else:
            current_bboxes = final_anomalies["bounding_box_arr"].tolist()
            verify_bboxes(initial_bboxes, current_bboxes, tol=1e-6)

verification()


re_prompt = f"""

You are an **archaeological-analytics assistant**.  
Our goal is to identify previously unknown ancient sites in the Acre region of Brazil.

**Datasets available**  
{DATASETS}

**Workflow already completed**

1. **Hydrology & buffers** â€“ Created river buffers using the maximum observed distance between known geoglyphs and rivers.  
2. **Candidate tiles** â€“ Exported all image tiles that intersect these buffers.  
3. **Elevation filtering (DEM)** â€“  
   â€¢ Obtained predicted site elevations from the prediction-sites dataset.  
   â€¢ For our AOIâ€™s DEM, retained only tiles whose mean elevation falls between the min and the top-10 % max of predicted elevations.  
4. **Bactris gasipaes proxy** â€“ Pulled GBIF occurrences of *Bactris gasipaes* (pejibaye palm) within the AOI, buffered them by 20 km, and intersected with the surviving DEM-filtered tiles.  
5. **GEDI canopy structure** â€“  
   â€¢ Focused on Cluster 0 (the most promising tile cluster).  
   â€¢ Downloaded GEDI L2A metrics for its bounding box from Google Earth Engine.  
   â€¢ Applied HDBSCAN to flag statistical outliers.  
   â€¢ From outliers with a quality score > 90, ran DBSCAN to group them into candidate bounding boxes.  
6. **Resulting dataframe sent to GPT** .
GPT prompt:
{prompt}
GPT response:
{gpt_response}
**Your tasks**

Critically assess the **strengths** of our current pipeline.  
Identify the most **error-prone steps** and recommend concrete improvements.  
Propose how this general workflow could be **adapted to detect ancient sites in other regions**.  
List additional **datasets or remote-sensing products** that could enhance future searches, explaining the specific value each would add. What extra evidence could tighten validation?
Summarize your advice in a clear **step-by-step roadmap** suitable for a research team.
"""


completion = client.chat.completions.create(
      model="gpt-4.1",
      store=True,
      messages=[
        {"role": "user", "content": re_prompt}
      ],
    )

print(completion.choices[0].message.content)

