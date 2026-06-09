!pip install /kaggle/input/rasterio/rasterio-1.4.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl > /dev/null 2>&1
!pip install /kaggle/input/pyogrio/pyogrio-0.11.0-cp311-cp311-manylinux_2_28_x86_64.whl > /dev/null 2>&1
!pip install /kaggle/input/openai/openai-1.84.0-py3-none-any.whl > /dev/null 2>&1


import matplotlib
matplotlib.use('Agg')


import os
import openai
import ee
import gc
import base64
import geemap
import psutil
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import cupy as cp
import matplotlib.pyplot as plt
import json
import pyogrio
import zipfile
from glob import glob
from google.oauth2 import service_account
from kaggle_secrets import UserSecretsClient
from shapely.geometry import box, mapping
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio.windows import Window
from affine import Affine
from PIL import Image
from multiprocessing import Pool, cpu_count


user_secrets = UserSecretsClient()
openai.api_key = user_secrets.get_secret("openai_key")
service_account_json = user_secrets.get_secret("GEE_SERVICE_ACCOUNT")
credentials_dict = json.loads(service_account_json)
credentials = service_account.Credentials.from_service_account_info(
    credentials_dict,
    scopes=['https://www.googleapis.com/auth/earthengine']
)
#ee.Initialize(credentials)


input_dir = '/kaggle/input'
output_dir = '/kaggle/working'
topo_features = f'{output_dir}/topo_features'
merged_tiff = f'{topo_features}/merged_tiff'
output_images = f'{topo_features}/output_images'
meta_data_json = f'{topo_features}/metadata'
opentopography_path = f'{input_dir}/opentopography-data/opentopography'
usgs_path = f'{input_dir}/usgs-sa-data/usgs_SA_data'
amazon_deforestation_path = f'{input_dir}/amazon-deforestation/amazon-deforestation-tif'
amazon_shp_path = f'{input_dir}/amazon-shp'

# Create directories if they don't exist
for dir_path in [topo_features, merged_tiff, output_images, meta_data_json]:
    os.makedirs(dir_path, exist_ok=True)


models = ['gpt-4o']
model_results = {}


# Function to list all TIFF files in a directory and its subdirectories
def find_tiff_files(directory):
    tiff_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tif') or file.endswith('.tiff'):
                tiff_files.append(os.path.join(root, file))
    return tiff_files

def get_tiff_files(directories):
    tiff_files = []
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.tif') or file.endswith('.tiff'):
                    tiff_files.append(os.path.join(root, file))
    return tiff_files

def find_shp_files(directory):
    shp_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.shp'):
                shp_files.append(os.path.join(root, file))
    return shp_files


# Print tifs for Debug
def print_tifs(tif_files, tif_type: str):
    print(f"TIFF files in {tif_type}-data:")
    for tif in tif_files[:5]:
        print(tif)

def print_shps(shp_files, shp_type: str):
    print(f"SHP files in {shp_type}-data:")
    for shp in shp_files:
        print(shp)


def extract_zip_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(root)
                print(f"Extracted: {zip_path}")


# Extract any ZIP files in opentopography-data
extract_zip_files(opentopography_path)
extract_zip_files(usgs_path)
extract_zip_files(amazon_deforestation_path)
extract_zip_files(amazon_shp_path)


# List TIFF and SHP files
opentopography_tiffs = find_tiff_files(opentopography_path)
usgs_tiffs = find_tiff_files(usgs_path)
amazon_deforestation_tiffs = find_tiff_files(amazon_deforestation_path)
amazon_shps = find_shp_files(amazon_shp_path)


# Print TIFF and SHP files
print_tifs(opentopography_tiffs, "opentopography")
print_tifs(usgs_tiffs, "usgs-sa")
print_tifs(amazon_deforestation_tiffs, "amazon-deforestation")
print_shps(amazon_shps, "amazon-shp")


# Define the Amazon biome bounding box for Northern South America
amazon_bbox = box(-81, -10, -50, 5)


# Function to load a single shapefile with optimizations
def load_shapefile(args):
    key, shp_path, simplify_tolerance = args
    try:
        # Check for associated files
        base_path = os.path.splitext(shp_path)[0]
        shx_path = f"{base_path}.shx"
        dbf_path = f"{base_path}.dbf"

        if not os.path.exists(shp_path):
            print(f"Shapefile not found: {shp_path}")
            return key, None
        if not os.path.exists(shx_path):
            print(f"Missing .shx file for {key}: {shx_path}")
            return key, None
        if not os.path.exists(dbf_path):
            print(f"Missing .dbf file for {key}: {dbf_path}")
            return key, None

        gdf = gpd.read_file(shp_path, engine='pyogrio', columns=['geometry'])
        if simplify_tolerance > 0:
            gdf['geometry'] = gdf['geometry'].simplify(simplify_tolerance, preserve_topology=True)

        print(f"Loaded and simplified shapefile: {key} with {len(gdf)} features")
        return key, gdf

    except Exception as e:
        print(f"Error loading shapefile {key}: {e}")
        return key, None


# Load and combine shapefiles in parallel
shapefiles = {
    'biome': f'{amazon_shp_path}/LimBiogeografico.shp',
    'indigenous': f'{amazon_shp_path}/Tis_TerritoriosIndigenas.shp',
    'fires': f'{amazon_shp_path}/quemas.shp',
    'oil': f'{amazon_shp_path}/petroleo.shp',
    'mining': f'{amazon_shp_path}/mineria_pl.shp',
    'illegal_mining': f'{amazon_shp_path}/MineriaIlegal_pol.shp',
    'hydroelectric': f'{amazon_shp_path}/hidroeletricas.shp',
    'protected_national': f'{amazon_shp_path}/ANP_Nacional.shp',
    'protected_departmental': f'{amazon_shp_path}/ANP_Departamental.shp',
    'protected_forest': f'{amazon_shp_path}/ANP_BosqueProtector.shp',
    'protected_reserve': f'{amazon_shp_path}/ANP_ReservaFlorestal.shp',
    'roads_national': f'{amazon_shp_path}/vias_nacional.shp',
    'roads_departmental': f'{amazon_shp_path}/vias_departamental.shp',
    'roads_railway': f'{amazon_shp_path}/vias_ferrea.shp'
}

simplify_tolerance = 0.01
tasks = [(key, path, simplify_tolerance) for key, path in shapefiles.items()]
num_processes = cpu_count() - 1
print(f"Loading shapefiles using {num_processes} processes...")
with Pool(processes=num_processes) as pool:
    results = pool.map(load_shapefile, tasks)

gdfs = {key: gdf for key, gdf in results}

# Filter fires to only those intersecting the Amazon biome bounding box
amazon_bbox_gdf = gpd.GeoDataFrame({'geometry': [amazon_bbox]}, crs=gdfs['biome'].crs)
if 'fires' in gdfs and gdfs['fires'] is not None:
    if gdfs['fires'].crs != amazon_bbox_gdf.crs:
        print(f"Reprojecting fires from {gdfs['fires'].crs} to {amazon_bbox_gdf.crs}...")
        gdfs['fires'] = gdfs['fires'].to_crs(amazon_bbox_gdf.crs)
    gdfs['fires'] = gpd.overlay(gdfs['fires'], amazon_bbox_gdf, how='intersection')
    print(f"Filtered fires to {len(gdfs['fires'])} features within Amazon biome bounding box.")

# Combine protected areas into a single GeoDataFrame
protected_keys = ['protected_national', 'protected_departmental', 'protected_forest', 'protected_reserve']
protected_gdfs = [gdfs[key] for key in protected_keys if gdfs[key] is not None]
if protected_gdfs:
    try:
        gdfs['protected'] = gpd.GeoDataFrame(pd.concat(protected_gdfs, ignore_index=True), crs=protected_gdfs[0].crs)
        print("Combined protected areas into a single layer.")
    except Exception as e:
        print(f"Error combining protected areas: {e}")
        gdfs['protected'] = None
else:
    gdfs['protected'] = None
    print("No protected areas shapefiles loaded.")

roads_keys = ['roads_national', 'roads_departmental', 'roads_railway']
roads_gdfs = [gdfs[key] for key in roads_keys if gdfs[key] is not None]
if roads_gdfs:
    try:
        gdfs['roads'] = gpd.GeoDataFrame(pd.concat(roads_gdfs, ignore_index=True), crs=roads_gdfs[0].crs)
        print("Combined roads into a single layer.")
    except Exception as e:
        print(f"Error combining roads: {e}")
        gdfs['roads'] = None
else:
    gdfs['roads'] = None
    print("No roads shapefiles loaded.")


# Add Sentinel-2 NDVI for vegetation analysis
def get_sentinel2_ndvi():
    try:
        sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterBounds(ee.Geometry.Rectangle(-81, -10, -50, 5)) \
            .filterDate('2020-01-01', '2020-12-31') \
            .median()
        ndvi = sentinel2.normalizedDifference(['B8', 'B4']).rename('NDVI')
        geemap.ee_export_image(ndvi, filename=f'{merged_tiff}/sentinel2_ndvi.tif', scale=10, region=ee.Geometry.Rectangle(-81, -10, -50, 5))
        print("Exported Sentinel-2 NDVI to sentinel2_ndvi.tif")
    except Exception as e:
        print(f"Error exporting Sentinel-2 NDVI: {e}")


#get_sentinel2_ndvi()


tiff_files = get_tiff_files([opentopography_path, usgs_path, amazon_deforestation_path])
print(f"Found {len(tiff_files)} TIFF files.")


# Function to check if a TIFF file intersects with the Amazon biome
def intersects_amazon(tiff_path, target_bbox):
    try:
        with rasterio.open(tiff_path) as src:
            bounds = src.bounds 
            tiff_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            return tiff_bbox.intersects(target_bbox)
    except Exception as e:
        print(f"Error reading {tiff_path}: {e}")
        return False


# Filter TIFF files
relevant_tiffs = [tiff for tiff in tiff_files if intersects_amazon(tiff, amazon_bbox)]
print(f"Found {len(relevant_tiffs)} TIFF files overlapping the Amazon biome.")


# Calculate hillshade
def hillshade(array, azimuth=315, altitude=45):
    array = cp.asarray(array)
    azimuth_rad = cp.radians(azimuth)
    altitude_rad = cp.radians(altitude)
    dx, dy = cp.gradient(array)
    slope = cp.arctan(cp.sqrt(dx**2 + dy**2))
    aspect = cp.arctan2(-dx, dy)
    hillshade = 255.0 * ((cp.cos(altitude_rad) * cp.cos(slope)) +
                         (cp.sin(altitude_rad) * cp.sin(slope) * cp.cos(azimuth_rad - aspect)))
    hillshade = cp.clip(hillshade, 0, 255)
    return cp.asnumpy(hillshade)


# Function to process a single batch
def process_batch(batch_args):
    batch_idx, batch_tiffs, gdfs, merged_tiff, output_images, meta_data_json = batch_args
    gpu_id = batch_idx % 2
    cp.cuda.runtime.setDevice(gpu_id)
    print(f"Processing batch {batch_idx + 1} on GPU {gpu_id}: {len(batch_tiffs)} files")
    #print(f"Memory usage before batch: {psutil.virtual_memory().used / 1024**3:.2f} GB")

    target_crs = 'EPSG:4326'

    src_files_to_mosaic = []
    for tiff_path in batch_tiffs:
        try:
            src = rasterio.open(tiff_path)
            data = src.read(
                out_shape=(src.count, src.height // 8, src.width // 8),
                resampling=rasterio.enums.Resampling.bilinear
            )
            transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2])
            )

            memfile = rasterio.MemoryFile()
            with memfile.open(
                driver='GTiff',
                height=data.shape[1],
                width=data.shape[2],
                count=data.shape[0],
                dtype=data.dtype,
                crs=src.crs,
                transform=transform,
                nodata=src.nodata
            ) as mem_dst:
                mem_dst.write(data)

            src_downsampled = memfile.open()
            reprojected_memfile = rasterio.MemoryFile()
            with reprojected_memfile.open(
                driver='GTiff',
                height=data.shape[1],
                width=data.shape[2],
                count=data.shape[0],
                dtype=data.dtype,
                crs=target_crs,
                transform=transform,
                nodata=src.nodata
            ) as reprojected_dst:
                rasterio.warp.reproject(
                    source=rasterio.band(src_downsampled, list(range(1, src_downsampled.count + 1))),
                    destination=rasterio.band(reprojected_dst, list(range(1, src_downsampled.count + 1))),
                    src_transform=src_downsampled.transform,
                    src_crs=src_downsampled.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=rasterio.enums.Resampling.bilinear
                )
            src_downsampled.close()
            src_reprojected = reprojected_memfile.open()
            src_files_to_mosaic.append(src_reprojected)
            src.close()
        except Exception as e:
            print(f"Error opening or reprojecting {tiff_path}: {e}")

    if not src_files_to_mosaic:
        print(f"No valid TIFF files in batch {batch_idx + 1}. Skipping...")
        return

    try:
        mosaic, out_trans = merge(src_files_to_mosaic)
        memory_after_merge = psutil.virtual_memory().used / 1024**3
        #print(f"Memory usage after merge: {memory_after_merge:.2f} GB")

        memory_threshold = 5.0
        if memory_after_merge > memory_threshold:
            #print(f"Memory usage ({memory_after_merge:.2f} GB) exceeds threshold ({memory_threshold} GB). Skipping batch {batch_idx + 1}.")
            print(f"Batch {batch_idx + 1} (indices {batch_idx*batch_size}:{(batch_idx + 1)*batch_size}) skipped due to memory usage: {memory_after_merge:.2f} GB\n")
            for src in src_files_to_mosaic:
                src.close()
            src_files_to_mosaic = []
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            return

        out_meta = src_files_to_mosaic[0].meta.copy()
        nodata = src_files_to_mosaic[0].nodata if src_files_to_mosaic[0].nodata is not None else -9999
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw",
            "nodata": nodata
        })

        mosaic_gpu = cp.asarray(mosaic)

        if 'biome' in gdfs and gdfs['biome'] is not None:
            gdf_biome = gdfs['biome']
            geom = [mapping(g) for g in gdf_biome.geometry]
            mask = geometry_mask(
                geom,
                out_shape=(mosaic.shape[1], mosaic.shape[2]),
                transform=out_trans,
                invert=True
            )
            mask_gpu = cp.asarray(mask[np.newaxis, :, :])
            mosaic_gpu[~mask_gpu] = nodata
            rows, cols = np.where(mask)
            if len(rows) > 0 and len(cols) > 0:
                row_start, row_stop = rows.min(), rows.max() + 1
                col_start, col_stop = cols.min(), cols.max() + 1
                mosaic_gpu = mosaic_gpu[:, row_start:row_stop, col_start:col_stop]
                window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
                a, b, c, d, e, f, _, _, _ = out_trans
                new_c = c + col_start * a + row_start * b
                new_f = f + col_start * d + row_start * e
                out_trans = Affine(a, b, new_c, d, e, new_f)
                out_meta.update({
                    "height": mosaic_gpu.shape[1],
                    "width": mosaic_gpu.shape[2],
                    "transform": out_trans
                })
            else:
                print(f"Warning: No overlap with biome for batch {batch_idx + 1}. Skipping clipping.")
        else:
            print(f"Warning: Biome shapefile not loaded. Skipping clipping for batch {batch_idx + 1}.")

        mosaic = cp.asnumpy(mosaic_gpu)
        merged_path = f"{merged_tiff}/batch_{batch_idx}_merged.tif"
        with rasterio.open(merged_path, 'w', **out_meta) as dest:
            dest.write(mosaic)

        elevation = cp.asarray(mosaic[0])
        dx, dy = cp.gradient(elevation)
        slope = cp.arctan(cp.sqrt(dx**2 + dy**2)) * (180 / cp.pi)
        hs = hillshade(elevation)

        slope = cp.asnumpy(slope)
        elevation = cp.asnumpy(elevation)

        batch_bounds = box(*rasterio.transform.array_bounds(mosaic.shape[1], mosaic.shape[2], out_trans))
        metadata = {
            "batch_id": batch_idx,
            "indigenous_overlap": any(gdfs['indigenous'].geometry.intersects(batch_bounds)) if 'indigenous' in gdfs and gdfs['indigenous'] is not None else False,
            "protected_overlap": any(gdfs['protected'].geometry.intersects(batch_bounds)) if 'protected' in gdfs and gdfs['protected'] is not None else False,
            "fires_overlap": any(gdfs['fires'].geometry.intersects(batch_bounds)) if 'fires' in gdfs and gdfs['fires'] is not None else False,
            "oil_overlap": any(gdfs['oil'].geometry.intersects(batch_bounds)) if 'oil' in gdfs and gdfs['oil'] is not None else False,
            "mining_overlap": any(gdfs['mining'].geometry.intersects(batch_bounds)) if 'mining' in gdfs and gdfs['mining'] is not None else False,
            "illegal_mining_overlap": any(gdfs['illegal_mining'].geometry.intersects(batch_bounds)) if 'illegal_mining' in gdfs and gdfs['illegal_mining'] is not None else False,
            "hydroelectric_overlap": any(gdfs['hydroelectric'].geometry.intersects(batch_bounds)) if 'hydroelectric' in gdfs and gdfs['hydroelectric'] is not None else False,
            "roads_overlap": any(gdfs['roads'].geometry.intersects(batch_bounds)) if 'roads' in gdfs and gdfs['roads'] is not None else False
        }

        #print(f"Batch {batch_idx + 1} metadata: {metadata}")
        #print(f"Memory usage after overlap checks: {psutil.virtual_memory().used / 1024**3:.2f} GB")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(elevation, cmap='terrain')
        axes[0].set_title(f'Batch {batch_idx} Elevation')
        axes[0].axis('off')
        axes[1].imshow(slope, cmap='viridis')
        axes[1].set_title(f'Batch {batch_idx} Slope')
        axes[1].axis('off')
        axes[2].imshow(hs, cmap='gray')
        axes[2].set_title(f'Batch {batch_idx} Hillshade')
        axes[2].axis('off')
        plt.tight_layout()
        output_path = f"{output_images}/batch_{batch_idx}_visualizations.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        with open(f"{meta_data_json}/batch_{batch_idx}_metadata.json", 'w') as f:
            json.dump(metadata, f)

        print(f"Batch {batch_idx + 1} processed. Visualizations saved.")

    except Exception as e:
        print(f"Error processing batch {batch_idx + 1}: {e}")

    finally:
        for src in src_files_to_mosaic:
            src.close()
        src_files_to_mosaic = []
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        #print(f"Memory usage after batch cleanup: {psutil.virtual_memory().used / 1024**3:.2f} GB")


# Parallelize batch processing
batch_size = 5
batches = [(batch_idx, relevant_tiffs[i:i + batch_size], gdfs, merged_tiff, output_images, meta_data_json)
           for batch_idx, i in enumerate(range(0, len(relevant_tiffs), batch_size))]
# Stop until 175, after 176 memory consumption is high.
for batch in batches[:175]:
    process_batch(batch)


png_files = sorted(glob(os.path.join(output_images, '*.png')))


# Group PNG files by batch (elevation, slope, hillshade for each batch)
batches = {}
for png in png_files:
    batch_num = os.path.basename(png).split('_')[1]
    if batch_num not in batches:
        batches[batch_num] = {}
    if 'elevation' in png:
        batches[batch_num]['elevation'] = png
    elif 'slope' in png:
        batches[batch_num]['slope'] = png
    elif 'hillshade' in png:
        batches[batch_num]['hillshade'] = png


# Encode image for OpenAI
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


# Downsample image for OpenAI analysis
def downsample_image(image_path, max_size=512):
    try:
        with Image.open(image_path) as img:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            downsized_path = image_path.replace('.png', '_downsized.png')
            img.save(downsized_path, 'PNG')
            return downsized_path
    except Exception as e:
        print(f"Error downsizing image {image_path}: {e}")
        return None


# Analyze each batch with OpenAI models
results = {}
for batch_num in batches.keys():
    print(f"Analyzing batch {batch_num} for archaeological sites...")
    
    # Load metadata
    metadata_path = f"{meta_data_json}/batch_{batch_num}_metadata.json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            indigenous_overlap = metadata.get('indigenous_overlap', False)
            protected_overlap = metadata.get('protected_overlap', False)
            fires_overlap = metadata.get('fires_overlap', False)
            oil_overlap = metadata.get('oil_overlap', False)
            mining_overlap = metadata.get('mining_overlap', False)
            illegal_mining_overlap = metadata.get('illegal_mining_overlap', False)
            roads_overlap = metadata.get('roads_overlap', False)
        except Exception as e:
            print(f"Error loading metadata for batch {batch_num}: {e}")
            indigenous_overlap = False
            protected_overlap = False
            fires_overlap = False
            oil_overlap = False
            mining_overlap = False
            illegal_mining_overlap = False
            roads_overlap = False
    else:
        print(f"Metadata not found for batch {batch_num}")
        indigenous_overlap = False
        protected_overlap = False
        fires_overlap = False
        oil_overlap = False
        mining_overlap = False
        illegal_mining_overlap = False
        roads_overlap = False

    # Update image path to match the 'visualizations' suffix
    image_path = f"{output_images}/batch_{batch_num}_visualizations.png"
    if os.path.exists(image_path):
        downsized_image = downsample_image(image_path)
        encoded_image = encode_image(downsized_image)
        print(f"Found valid image at {image_path}")
    else:
        print(f"Image not found: {image_path}")
        encoded_image = None

    # Skip batch if no image is available
    if not encoded_image:
        print(f"No valid image for batch {batch_num}, skipping...")
        model_results[batch_num] = {model: "Error: No valid image provided" for model in models}
        continue

    # Construct the prompt for archaeological site discovery
    base_prompt = (
        f"I am a leading expert in archaeological site detection using topographic data, specializing in discovering previously unknown sites. "
        f"I have provided a single combined image for batch {batch_num} derived from SRTM elevation data in the Amazon biome, "
        f"containing elevation, slope, and hillshade visualizations in a single plot.\n\n"
        f"Additional context:\n"
        f"- This batch {'overlaps' if indigenous_overlap else 'does not overlap'} with indigenous territories.\n"
        f"- This batch {'overlaps' if protected_overlap else 'does not overlap'} with protected areas.\n"
        f"- This batch {'overlaps' if fires_overlap else 'does not overlap'} with areas affected by recent fires.\n"
        f"- This batch {'overlaps' if oil_overlap else 'does not overlap'} with oil extraction areas.\n"
        f"- This batch {'overlaps' if mining_overlap else 'does not overlap'} with mining areas.\n"
        f"- This batch {'overlaps' if illegal_mining_overlap else 'does not overlap'} with illegal mining areas.\n"
        f"- This batch {'overlaps' if roads_overlap else 'does not overlap'} with roads or railways.\n\n"
        f"Analyze the provided image to identify potential previously unknown archaeological sites. Look for subtle anomalies such as:\n"
        f"- Raised platforms or mounds (elevation changes of 1-5m indicating possible structures).\n"
        f"- Linear features (terraces, canals, or ancient roads, typically 50-500m long).\n"
        f"- Circular or geometric patterns (potential settlements, geoglyphs, or ceremonial sites).\n\n"
        f"Provide a detailed report for each potential site, including:\n"
        f"- Approximate location within the image (e.g., top-left quadrant, center-right).\n"
        f"- Type of feature and estimated size.\n"
        f"- Reasoning for suspecting an archaeological origin, considering topographic evidence and context.\n"
        f"- Assessment of preservation likelihood based on socio-environmental factors (e.g., better preservation in protected areas, disturbance in mining or fire-affected zones).\n"
        f"- Recommendation to avoid disturbance in active indigenous territories.\n"
        f"If no sites are identified, explain the reasoning (e.g., uniform terrain, heavy disturbance)."
    )

    # Analyze with each model
    for model in models:
        print(f"Testing model: {model}")
        messages = []
        if model == "o4-mini":
            messages = [
                {"role": "user", "content": "You are an expert archaeologist. " + base_prompt}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are an expert archaeologist."},
                {"role": "user", "content": base_prompt}
            ]

        try:
            token_param = 'max_completion_tokens'
            kwargs = {token_param: 1000}

            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            model_results[model] = response.choices[0].message.content
            print(f"{model} analysis completed.")
        except Exception as e:
            model_results[model] = f"Error: {str(e)}"
            print(f"Error with {model}: {str(e)}")

    # Clean up downsized image
    if 'downsized_image' in locals() and os.path.exists(downsized_image):
        try:
            os.remove(downsized_image)
        except Exception as e:
            print(f"Error removing {downsized_image}: {e}")

    # Store results for this batch
    results[batch_num] = {model: model_results.get(model, "No analysis performed") for model in models}


# Compare the results
print("\nModel Comparison:")
with open('model_comparison.txt', 'w') as f:
    for model, result in model_results.items():
        print(f"\nModel: {model}")
        print(result)
        print("-" * 50)
        f.write(f"Model: {model}\n{result}\n{'-'*50}\n")
print("Comparison saved to model_comparison.txt")

