!pip install rasterio geopandas scipy --quiet


# ==============================================================================
# SCRIPT: Downsample MERIT DEM Raster
# PURPOSE: Reduce DEM resolution by a specified factor using average resampling
# ==============================================================================

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
import rasterio
from rasterio.enums import Resampling
import numpy as np
import os

# --------------------------------------------------------------------------
# Input and output file paths
# --------------------------------------------------------------------------
dem_in = "/kaggle/input/download-merit-dem-amazon/output/amazon_merit_dem_clipped.tif"
dem_out = "amazon_merit_dem_downsampled.tif"
downsample_factor = 3  # Change to 2, 4, etc. to reduce file sizes

print(f"ğŸ“‰ Downsampling DEM by {downsample_factor}Ã—...")

# --------------------------------------------------------------------------
# Open input DEM and apply resampling
# --------------------------------------------------------------------------
with rasterio.open(dem_in) as src:
    # Compute new raster dimensions
    new_height = src.height // downsample_factor
    new_width = src.width // downsample_factor

    # Compute new affine transform for rescaled resolution
    new_transform = src.transform * src.transform.scale(
        src.width / new_width,
        src.height / new_height
    )

    # Resample DEM using average pooling
    dem_data = src.read(
        1,
        out_shape=(new_height, new_width),
        resampling=Resampling.average
    )

    # ----------------------------------------------------------------------
    # Update metadata for output raster
    # ----------------------------------------------------------------------
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": new_height,
        "width": new_width,
        "transform": new_transform,
        "compress": "deflate",
        "dtype": "float32",
        "nodata": np.nan
    })

    # ----------------------------------------------------------------------
    # Write downsampled DEM to disk
    # ----------------------------------------------------------------------
    with rasterio.open(dem_out, "w", **out_meta) as dst:
        dst.write(dem_data.astype("float32"), 1)

print(f"âœ… Downsampled DEM saved to: {dem_out}")



# ==============================================================================
# SCRIPT: Prepare Inputs for Elevation Above River Analysis
# PURPOSE: Align CRS between DEM and basin polygon before hydrological modeling
# ==============================================================================

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
import os
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
import geopandas as gpd

# --------------------------------------------------------------------------
# Define input and output file paths
# --------------------------------------------------------------------------
dem_path   = "/kaggle/working/amazon_merit_dem_downsampled.tif"
river_fp   = "/kaggle/input/amazon-river-map-geojson/Amazon River.geojson"
basin_fp   = "/kaggle/input/make-amazon-polygon/amazon_basin_polygon.geojson"
out_fp     = "amazon_elevation_above_river.tif"

# --------------------------------------------------------------------------
# Load Amazon Basin polygon from GeoJSON
# --------------------------------------------------------------------------
basin_gdf = gpd.read_file(basin_fp)

# --------------------------------------------------------------------------
# Check and reproject basin polygon to match DEM's CRS
# --------------------------------------------------------------------------
with rasterio.open(dem_path) as src:
    if basin_gdf.crs != src.crs:
        print(f"ğŸ”„ Reprojecting basin polygon from {basin_gdf.crs} to {src.crs}")
        basin_gdf = basin_gdf.to_crs(src.crs)



# ==============================================================================
# STEP: Compute Elevation Above River Surface Using Distance Propagation
# ==============================================================================

# --------------------------------------------------------------------------
# Step 1: Load the DEM
# --------------------------------------------------------------------------
with rasterio.open(dem_path) as src:
    dem = src.read(1).astype("float32")
    transform = src.transform
    crs = src.crs
    meta = src.meta.copy()

print("ğŸ“‚ Loaded DEM with shape:", dem.shape)

# --------------------------------------------------------------------------
# Step 2: Load and rasterize the river geometry to match DEM resolution
# --------------------------------------------------------------------------
print("ğŸ“‚ Loading and rasterizing river GeoJSON...")
rivers = gpd.read_file(river_fp)

# Reproject rivers if CRS does not match DEM
if rivers.crs != crs:
    print(f"ğŸ”„ Reprojecting river CRS from {rivers.crs} to {crs}")
    rivers = rivers.to_crs(crs)

# Create binary raster mask (1 = river, 0 = non-river)
river_mask = rasterize(
    [(geom, 1) for geom in rivers.geometry],
    out_shape=dem.shape,
    transform=transform,
    fill=0,
    dtype="uint8"
)

# --------------------------------------------------------------------------
# Step 3: Extract elevation values where river pixels are located
# --------------------------------------------------------------------------
river_dem = np.where(river_mask == 1, dem, np.nan)
river_pixel_count = np.isfinite(river_dem).sum()
print(f"ğŸ§® River pixels found: {river_pixel_count:,}")

if river_pixel_count == 0:
    raise ValueError("â�Œ No valid river pixels found in DEM. Check alignment or CRS.")

# --------------------------------------------------------------------------
# Step 4: Use distance transform to propagate nearest river elevation
# --------------------------------------------------------------------------
print("ğŸ“¡ Propagating river elevation to all land pixels...")
mask = ~np.isfinite(river_dem)  # Mask: True where river elevation is NaN
dist, indices = distance_transform_edt(mask, return_indices=True)
nearest_river_elevation = river_dem[tuple(indices)]  # Lookup nearest river elevation per cell

# --------------------------------------------------------------------------
# Step 5: Subtract nearest river elevation from DEM to get height above river
# --------------------------------------------------------------------------
print("â�– Calculating elevation above river...")
above_river = np.where(np.isfinite(dem), dem - nearest_river_elevation, np.nan)

# --------------------------------------------------------------------------
# Step 6: Apply basin mask to limit results to defined region
# --------------------------------------------------------------------------
from rasterio.mask import mask
from shapely.geometry import mapping

print("ğŸ§¼ Applying basin mask to elevation-above-river result...")
with rasterio.open(dem_path) as dem_src:
    masked_result, _ = mask(
        dem_src,
        [mapping(geom) for geom in basin_gdf.geometry],
        crop=False,
        filled=False
    )

# Apply rasterio mask to above_river array
above_river = np.where(masked_result[0].mask, np.nan, above_river)

# --------------------------------------------------------------------------
# Step 7: Save output as compressed GeoTIFF
# --------------------------------------------------------------------------
print("ğŸ’¾ Saving elevation-above-river raster...")
os.makedirs("output", exist_ok=True)

meta.update({
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": np.nan,
    "compress": "deflate"
})

with rasterio.open(out_fp, "w", **meta) as dst:
    dst.write(above_river, 1)

print(f"âœ… Elevation-above-river raster saved to: {out_fp}")


# ==============================================================================
# SCRIPT: Downsample Elevation-Above-River Raster
# PURPOSE: Reduce resolution of elevation-above-river GeoTIFF using average resampling
# ==============================================================================

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
import rasterio
from rasterio.enums import Resampling
import numpy as np

# --------------------------------------------------------------------------
# Define input/output file paths and target width (pixels)
# --------------------------------------------------------------------------
input_fp    = "/kaggle/working/amazon_elevation_above_river.tif"
downsamp_fp = "amazon_elevation_above_river_downsamp.tif"
target_width = 3000  # Desired output width in pixels (change as needed)

# --------------------------------------------------------------------------
# Open the source raster and compute new dimensions
# --------------------------------------------------------------------------
with rasterio.open(input_fp) as src:
    scale = src.width / target_width
    target_height = int(src.height / scale)

    # Compute new transform for scaled resolution
    transform = src.transform * src.transform.scale(
        src.width / target_width,
        src.height / target_height
    )

    # ----------------------------------------------------------------------
    # Resample data using average pooling
    # ----------------------------------------------------------------------
    data = src.read(
        1,
        out_shape=(target_height, target_width),
        resampling=Resampling.average
    )

    # ----------------------------------------------------------------------
    # Update metadata to match new shape and transform
    # ----------------------------------------------------------------------
    meta = src.meta.copy()
    meta.update({
        "height": target_height,
        "width": target_width,
        "transform": transform
    })

    # ----------------------------------------------------------------------
    # Write the downsampled raster to disk
    # ----------------------------------------------------------------------
    with rasterio.open(downsamp_fp, "w", **meta) as dst:
        dst.write(data.astype("float32"), 1)

# --------------------------------------------------------------------------
# Confirm output
# --------------------------------------------------------------------------
print(f"âœ… Downsampled raster saved: {downsamp_fp}")




# ==============================================================================
# SCRIPT: Elevation Classification and Interactive Map Rendering
# PURPOSE: Colorize elevation-above-river raster and overlay it on a Folium map
# ==============================================================================

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
import os
import numpy as np
import rasterio
import folium
from rasterio.warp import transform_bounds
import imageio

# --------------------------------------------------------------------------
# Input and output file paths
# --------------------------------------------------------------------------
raster_path = "/kaggle/working/amazon_elevation_above_river_downsamp.tif"
png_path = "elevation_above_water_classified.png"

# --------------------------------------------------------------------------
# Load the downsampled elevation-above-river raster
# --------------------------------------------------------------------------
with rasterio.open(raster_path) as src:
    dem_data = src.read(1)            # Read first band
    bounds = src.bounds               # Get spatial bounds
    transform = src.transform         # Affine transform
    crs = src.crs                     # Coordinate reference system

# --------------------------------------------------------------------------
# Function: Apply RGB color classification based on elevation thresholds
# --------------------------------------------------------------------------
def classify_elevation(val):
    if np.isnan(val):
        return (0, 0, 0, 0)                # transparent
    elif val < 0:
        return (128, 0, 128, 255)          # purple (below river)
    elif val < 0.5:
        return (0, 0, 128, 255)            # navy
    elif val < 1:
        return (0, 0, 180, 255)            # deep blue
    elif val < 2:
        return (0, 100, 200, 255)          # medium blue
    elif val < 5:
        return (173, 216, 230, 255)        # light blue
    elif val < 8:
        return (60, 179, 113, 255)         # medium green
    elif val < 12:
        return (34, 139, 34, 255)          # dark green
    elif val < 20:
        return (189, 183, 107, 255)        # khaki
    elif val < 30:
        return (205, 133, 63, 255)         # tan
    else:
        return (139, 69, 19, 255)          # brown (high)

# --------------------------------------------------------------------------
# Generate RGBA image from elevation raster using classification function
# --------------------------------------------------------------------------
h, w = dem_data.shape
rgba = np.zeros((h, w, 4), dtype=np.uint8)

for i in range(h):
    for j in range(w):
        rgba[i, j] = classify_elevation(dem_data[i, j])

# --------------------------------------------------------------------------
# Save RGBA image as PNG for use in Folium overlay
# --------------------------------------------------------------------------
os.makedirs("output", exist_ok=True)
imageio.imwrite(png_path, rgba)

# --------------------------------------------------------------------------
# Reproject raster bounds from source CRS to WGS84 (lat/lon)
# --------------------------------------------------------------------------
latlon_bounds = transform_bounds(crs, "EPSG:4326", *bounds)
minx, miny, maxx, maxy = latlon_bounds
center = [(miny + maxy) / 2, (minx + maxx) / 2]

# --------------------------------------------------------------------------
# Create interactive Folium map and add the PNG overlay
# --------------------------------------------------------------------------
m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

folium.raster_layers.ImageOverlay(
    image=png_path,
    bounds=[[miny, minx], [maxy, maxx]],
    opacity=0.8,
    name="Elevation Above Water",
).add_to(m)

# Add layer control toggle
folium.LayerControl().add_to(m)

# --------------------------------------------------------------------------
# Save the map as a standalone HTML file
# --------------------------------------------------------------------------
map_output_path = "elevation_above_water_map.html"
m.save(map_output_path)

print(f"âœ… Map saved to: {map_output_path}")



