!pip install boto3 rasterio --quiet


import os
import requests
import rasterio
from rasterio.merge import merge
from kaggle_secrets import UserSecretsClient
import geopandas as gpd
from rasterio.mask import mask
from rasterio.io import MemoryFile

# ğŸ”� Load credentials from Kaggle secrets
secrets = UserSecretsClient()
MERIT_USER = secrets.get_secret("MERIT_USER")
MERIT_PASS = secrets.get_secret("MERIT_PASS")

# ğŸŒ� Correct MERIT DEM base URL (5-degree tiles)
BASE_URL = "http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/distribute/v1.0.2/5deg"

# Amazon Basin bounding box (in 5Â° steps)
min_lon, max_lon = -80, -50
min_lat, max_lat = -20, 10

# ğŸ”¤ Create lowercase tile names (e.g. s10w075_dem.tif)
def merit_tile_name(lon, lat):
    lon_prefix = f"{'w' if lon < 0 else 'e'}{abs(lon):03d}"
    lat_prefix = f"{'s' if lat < 0 else 'n'}{abs(lat):02d}"
    return f"{lat_prefix}{lon_prefix}_dem.tif"

tile_names = []
for lat in range(min_lat, max_lat, 5):
    for lon in range(min_lon, max_lon, 5):
        tile_names.append(merit_tile_name(lon, lat))

# ğŸ“� Create local tile directory
tile_dir = "merit_tiles"
os.makedirs(tile_dir, exist_ok=True)

# â¬‡ï¸� Download tiles with basic HTTP auth
# ğŸ“� Define source (dataset) and working (target) directories
kaggle_dataset_dir = "/kaggle/input/amazon-meri-dem-tiles/merit_tiles"
tile_dir = "merit_tiles"
os.makedirs(tile_dir, exist_ok=True)

downloaded = []

for tile in tile_names:
    source_path = os.path.join(kaggle_dataset_dir, tile)
    target_path = os.path.join(tile_dir, tile)

    if os.path.exists(target_path):
        print(f"âœ… Found locally: {tile}")
        downloaded.append(target_path)
        continue

    elif os.path.exists(source_path):
        print(f"ğŸ“¦ Copying from Kaggle dataset: {tile}")
        os.system(f"cp {source_path} {target_path}")
        downloaded.append(target_path)
        continue

    # Otherwise download from MERIT server
    url = f"{BASE_URL}/{tile}"
    print(f"â¬‡ï¸� Downloading {tile}...")
    try:
        response = requests.get(url, auth=(MERIT_USER, MERIT_PASS), stream=True)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
            downloaded.append(target_path)
            print("  âœ… Saved")
        else:
            print(f"  âš ï¸� Skipped {tile}: HTTP {response.status_code}")
    except Exception as e:
        print(f"  â�Œ Error downloading {tile}: {e}")


# ğŸ§© Merge all downloaded tiles into a single raster
if downloaded:
    srcs = [rasterio.open(fp) for fp in downloaded]
    mosaic, transform = merge(srcs, method='first')

    meta = srcs[0].meta.copy()
    meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
        "driver": "GTiff",
        "compress": "deflate"
    })

    os.makedirs("output", exist_ok=True)
    
    # Load basin polygon
    basin_fp = "/kaggle/input/make-amazon-polygon/amazon_basin_polygon.geojson"
    print("ğŸ“‚ Loading basin polygon for clipping...")
    basin_gdf = gpd.read_file(basin_fp)
    
    # Reproject basin to DEM CRS if needed
    if basin_gdf.crs != meta["crs"]:
        basin_gdf = basin_gdf.to_crs(meta["crs"])
    
    # Open in-memory raster from mosaic
    meta.update({"count": 1})
    with MemoryFile() as memfile:
        with memfile.open(**meta) as tmp:
            tmp.write(mosaic[0], 1)
    
            # Simplify geometry before clipping
            print("âœ‚ï¸� Simplifying basin polygon for faster clipping...")
            simplified_geom = basin_gdf.simplify(0.05)  # degrees; adjust for your use case
            
            print("âœ‚ï¸� Clipping DEM to simplified Amazon basin polygon...")
            clipped_dem, clipped_transform = mask(tmp, simplified_geom.geometry, crop=True)

    
    # Update metadata for clipped version
    meta.update({
        "height": clipped_dem.shape[1],
        "width": clipped_dem.shape[2],
        "transform": clipped_transform
    })
    
    # Save clipped raster
    clipped_fp = "output/amazon_merit_dem_clipped.tif"
    with rasterio.open(clipped_fp, "w", **meta) as dst:
        dst.write(clipped_dem)
    
    print(f"âœ… Clipped MERIT DEM saved to: {clipped_fp}")

    # ğŸ§¹ Delete individual MERIT tiles after merging
    for fp in downloaded:
        try:
            os.remove(fp)
        except Exception as e:
           print(f"âš ï¸� Could not delete {fp}: {e}")
    
    print("ğŸ§¹ Removed all individual tile .tif files.")

else:
    print("âš ï¸� No tiles were downloaded â€” check your credentials or region.")





