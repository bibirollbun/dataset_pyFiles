# ==============================================================================
# SCRIPT: Generate Convex Hull Basin from River Geometry
# PURPOSE: Estimate a simplified Amazon Basin polygon by buffering a river network
# ==============================================================================

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
import os
import geopandas as gpd
from shapely.geometry import MultiLineString

# --------------------------------------------------------------------------
# Define input/output file paths
# --------------------------------------------------------------------------
river_fp = "/kaggle/input/amazon-river-map-geojson/Amazon River.geojson"  # Input: river lines
basin_out_fp = "amazon_basin_polygon.geojson"                             # Output: generated basin polygon

# --------------------------------------------------------------------------
# Step 1: Load Amazon River dataset (as line features)
# --------------------------------------------------------------------------
print("ðŸ“‚ Loading river data...")
rivers = gpd.read_file(river_fp)

# --------------------------------------------------------------------------
# Step 2: Reproject to WGS84 if needed (standard for global analysis)
# --------------------------------------------------------------------------
if rivers.crs != "EPSG:4326":
    print(f"ðŸ”„ Reprojecting from {rivers.crs} to EPSG:4326...")
    rivers = rivers.to_crs("EPSG:4326")

# --------------------------------------------------------------------------
# Step 3: Generate convex-hull-based polygon with buffer (~50km radius)
# --------------------------------------------------------------------------
print("ðŸ›  Generating basin polygon from river geometry...")
merged = rivers.geometry.unary_union                # Combine all geometries into one
basin_geom = merged.convex_hull.buffer(0.5)         # Add 0.5-degree (~50 km) buffer around convex hull

# --------------------------------------------------------------------------
# Step 4: Save resulting basin polygon as GeoJSON
# --------------------------------------------------------------------------
basin_gdf = gpd.GeoDataFrame(geometry=[basin_geom], crs="EPSG:4326")
os.makedirs("output", exist_ok=True)
basin_gdf.to_file(basin_out_fp, driver="GeoJSON")

print(f"âœ… Basin polygon saved to: {basin_out_fp}")






