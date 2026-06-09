!pip install -q contextily
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import contextily as ctx
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Libraries imported successfully!")
print(f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load the datasets exactly like in the original notebook
print("Loading indigenous territories and communities data...")

# Load territories data
territories = gpd.read_file('/kaggle/input/indigenous-territories-amazon-geojson/brazilian_indigenous_territories_2024.geojson')
print(f"Indigenous Territories: {len(territories)} features loaded")

# Load communities data
communities = gpd.read_file('/kaggle/input/indigenous-territories-amazon-geojson/brazilian_indigenous_communities_2024.geojson')
print(f"Indigenous Communities: {len(communities)} features loaded")

# Load geoglyphs data for cross-reference (using the complete dataset like in original)
geoglyphs = gpd.read_file('/kaggle/input/amazon-geoglyphs-jqjacobs/amazon_geoglyphs_complete.geojson')
print(f"Geoglyphs: {len(geoglyphs)} features loaded")

print("\n=== Data Loading Complete ===")
print(f"Total features: {len(territories) + len(communities) + len(geoglyphs):,}")

# Display basic information about the datasets
print("=== TERRITORIES DATA OVERVIEW ===")
print(f"Shape: {territories.shape}")
print(f"CRS: {territories.crs}")
print(f"Countries represented: {territories['pais'].unique()}")
print(f"Geometry types: {territories.geometry.geom_type.value_counts().to_dict()}")

print("\n=== GEOGLYPHS DATA OVERVIEW ===")
print(f"Shape: {geoglyphs.shape}")
print(f"CRS: {geoglyphs.crs}")
print(f"Categories: {geoglyphs['category'].value_counts().to_dict()}")

# Use ALL archaeological site categories (excluding temporal data)
exclude_categories = ['temporal_data', 'Temporal Data', 'temporal']
all_categories = [cat for cat in sorted(geoglyphs['category'].unique()) if cat not in exclude_categories]
geoglyphs_filtered = geoglyphs[geoglyphs['category'].isin(all_categories)]  # Exclude temporal data
print(f"Using archaeological site categories (excluding temporal data): {all_categories}")
print(f"Adding {len(geoglyphs_filtered)} archaeological sites from {len(all_categories)} categories")

# Perform spatial analysis to find territories with sites
print("Performing spatial analysis...")

# Ensure both datasets are in WGS84 first
if territories.crs != 'EPSG:4326':
    territories_wgs84 = territories.to_crs(epsg=4326)
else:
    territories_wgs84 = territories.copy()

if geoglyphs_filtered.crs != 'EPSG:4326':
    geoglyphs_wgs84 = geoglyphs_filtered.to_crs(epsg=4326)
else:
    geoglyphs_wgs84 = geoglyphs_filtered.copy()

# Project to a suitable projected CRS for accurate spatial operations (like SIRGAS 2000 / Brazil Polyconic)
territories_proj = territories_wgs84.to_crs(epsg=5880)
geoglyphs_proj = geoglyphs_wgs84.to_crs(epsg=5880)

# Find territories containing archaeological sites (within boundaries)
geoglyphs_in_territories = gpd.sjoin(geoglyphs_proj, territories_proj, how='inner', predicate='within')

# Get unique territories with sites inside
territories_with_sites_ids = geoglyphs_in_territories.index_right.unique()
territories_with_sites = territories_proj.loc[territories_with_sites_ids].copy()

print(f"Found {len(territories_with_sites)} territories containing archaeological sites")

# Find territories within 8km of archaeological sites
print("Finding territories within 8km of archaeological sites...")

# Create an 8km buffer around the ARCHAEOLOGICAL SITES
sites_buffered_8km = geoglyphs_proj.buffer(8000)

# Find which territories intersect with this buffered zone
# This is more efficient than buffering every territory
intersecting_indices = []
for index, territory in territories_proj.iterrows():
    # Check if the territory's geometry intersects with any of the buffered sites
    if sites_buffered_8km.intersects(territory.geometry).any():
        intersecting_indices.append(index)

# Get the unique territories that are near sites
territories_near_sites_ids = pd.Index(intersecting_indices).unique()
territories_near_sites = territories_proj.loc[territories_near_sites_ids].copy()

# Territories with sites within 8km but NOT within boundaries
territories_near_only_ids = set(territories_near_sites_ids) - set(territories_with_sites_ids)
territories_near_only = territories_proj.loc[list(territories_near_only_ids)].copy()

print(f"Found {len(territories_near_sites)} territories with sites inside or within 8km")
print(f"Found {len(territories_near_only)} territories with sites within 8km (but not inside boundaries)")

# Convert back to WGS84 for plotting and saving
territories_with_sites_wgs84 = territories_with_sites.to_crs(epsg=4326)
territories_near_only_wgs84 = territories_near_only.to_crs(epsg=4326)
all_territories_wgs84 = territories_proj.to_crs(epsg=4326)


# --- START: CODE ADDED TO SAVE THE FILE ---

print("\n=== Preparing and Saving Output File ===")

# Add a 'proximity_status' column to each GeoDataFrame to describe why it was included
# This makes the output file more informative
territories_with_sites_wgs84['proximity_status'] = 'Contains Site'
territories_near_only_wgs84['proximity_status'] = 'Within 8km of Site'

# Combine the two groups of territories (those with sites inside and those with sites nearby)
# into a single GeoDataFrame.
territories_to_export = pd.concat([
    territories_with_sites_wgs84,
    territories_near_only_wgs84
])

# Define the output path for the Kaggle working directory
output_filename = '/kaggle/working/territories_with_archaeological_proximity.geojson'

# Save the combined GeoDataFrame to a GeoJSON file
try:
    territories_to_export.to_file(output_filename, driver='GeoJSON')
    print(f"Successfully outlined and saved {len(territories_to_export)} territories.")
    print(f"File saved to: {output_filename}")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")

# --- END: CODE ADDED TO SAVE THE FILE ---


# Create static map exactly like in the original notebook
print("\nGenerating map...")
fig, ax = plt.subplots(figsize=(20, 16))

# Plot all territories in light gray
all_territories_wgs84.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.7, label='All Other Territories')

# Highlight territories with sites within 8km (but not inside) in light blue
if len(territories_near_only_wgs84) > 0:
    territories_near_only_wgs84.plot(ax=ax, color='lightblue', edgecolor='blue', linewidth=0.8, alpha=0.7, label='Territories within 8km of Sites')

# Highlight territories with archaeological sites inside boundaries in dark blue
territories_with_sites_wgs84.plot(ax=ax, color='steelblue', edgecolor='navy', linewidth=1, alpha=0.9, label='Territories Containing Sites')

# Plot archaeological sites by category - create dynamic color mapping for static map
# Create a color palette for all categories
n_categories = len(all_categories)
# Use a visually distinct color map like 'tab20' or 'Set3'
colors_list = cm.get_cmap('tab20', n_categories)
colors = {category: colors_list(i) for i, category in enumerate(all_categories)}

# Override specific known categories with preferred colors for emphasis
color_overrides = {
    'earthworks': 'orange',
    'geoglyphs': 'purple',
    'mound_sites': 'green',
    'other': 'gray'
}
colors.update(color_overrides)

for category in all_categories:
    sites_in_category = geoglyphs_wgs84[geoglyphs_wgs84['category'] == category]
    if len(sites_in_category) > 0:
        # Use geometry directly for plotting, as Geopandas handles Points and Centroids
        sites_in_category.plot(ax=ax, marker='o', color=colors[category], markersize=20, alpha=0.8,
                               label=f'{category.replace("_", " ").title()} ({len(sites_in_category)})',
                               edgecolor='black', linewidth=0.4)

# Add basemap context
try:
    ctx.add_basemap(ax, crs=all_territories_wgs84.crs.to_string(), source=ctx.providers.CartoDB.Positron, alpha=0.6)
except Exception as e:
    print(f"Note: Could not add basemap - {e}. Displaying without context tiles.")

# Set map extent to focus on areas with data
bounds = all_territories_wgs84.total_bounds
margin = 1.0  # degrees
ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
ax.set_ylim(bounds[1] - margin, bounds[3] + margin)

# Customize map exactly like the original
total_territories_with_proximity = len(territories_with_sites) + len(territories_near_only)
ax.set_title(f'Indigenous Territories and Archaeological Sites\n{len(territories_with_sites)} territories contain sites, {len(territories_near_only)} are within 8km\n{total_territories_with_proximity} of {len(territories)} territories have archaeological proximity',
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Longitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)

# Add legend
legend = ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), fontsize=12, framealpha=0.95)
legend.set_title('Legend', prop={'size': 14, 'weight': 'bold'})

# Add grid
ax.grid(True, alpha=0.3)

# Remove axis spines for cleaner look
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()

# Print summary statistics exactly like the original
total_sites_in_territories = len(gpd.sjoin(geoglyphs_proj, territories_with_sites, how='inner'))
total_sites_near_only = len(gpd.sjoin(geoglyphs_proj, territories_near_only, how='inner'))

print("\n--- SUMMARY STATISTICS ---")
print(f"Total Indigenous Territories Analyzed: {len(territories)}")
print(f"Total Archaeological Sites Analyzed: {len(geoglyphs_filtered)}")
print("-" * 30)
print(f"Territories containing sites: {len(territories_with_sites)} ({len(territories_with_sites)/len(territories):.2%})")
print(f"Territories with sites only within 8km: {len(territories_near_only)} ({len(territories_near_only)/len(territories):.2%})")
print(f"Total territories with archaeological proximity: {total_territories_with_proximity} ({total_territories_with_proximity/len(territories):.2%})")
print("-" * 30)
print(f"Total sites located INSIDE territories: {total_sites_in_territories}")
print(f"Total sites located within 8km of (but not in) territories: {total_sites_near_only}")


# Breakdown by category for sites within territories
print(f"\n--- Archaeological Sites Inside Territories by Category ---")
sites_in_territory_by_cat = geoglyphs_in_territories['category'].value_counts()
for category in all_categories:
    count = sites_in_territory_by_cat.get(category, 0)
    if count > 0:
        print(f"- {category.replace('_', ' ').title()}: {count}")

# We need to recalculate this join specifically for the 'near_only' territories
geoglyphs_near_only_territories = gpd.sjoin(geoglyphs_proj, territories_near_only, how='inner', predicate='intersects')
print(f"\n--- Archaeological Sites within 8km of Territories (but not inside) by Category ---")
sites_near_territory_by_cat = geoglyphs_near_only_territories['category'].value_counts()
for category in all_categories:
    count = sites_near_territory_by_cat.get(category, 0)
    if count > 0:
        print(f"- {category.replace('_', ' ').title()}: {count}")


"""
Finding the Area of Interest (AOI) for Archaeological Ruins - RASTERIZED & OPTIMIZED
Kaggle Version - Lean GeoJSON Output Only (Fixed for Combined Data)

This script efficiently identifies potential areas of interest for archaeological discoveries
by analyzing the spatial relationship between known archaeological sites and recent deforestation.

Core Methodology:
1.  Loads archaeological sites and combined deforestation data efficiently.
2.  Cleans and validates input geometries to prevent topological errors.
3.  Converts vector data (polygons) to a raster grid (pixels) for high-speed analysis.
4.  Uses morphological operations (dilation) on the grid to create buffer zones.
5.  Identifies AOI by finding the intersection of the two buffered raster zones.
6.  Converts the resulting AOI pixels back to vector polygons for GeoJSON output.
"""

# Import required libraries for geospatial analysis
import geopandas as gpd  # For reading/writing geospatial data
import json             # For JSON file operations
import os               # For file system operations
from datetime import datetime  # For timestamping outputs
from pathlib import Path      # For modern path handling
import pandas as pd           # For data manipulation
import numpy as np           # For numerical operations
import rasterio             # For raster data operations
from rasterio import features  # For raster-vector conversion
from rasterio.transform import from_origin  # For setting up raster coordinate systems
from scipy import ndimage    # For morphological operations (buffering)
from shapely.geometry import shape  # For geometric operations
import warnings
warnings.filterwarnings('ignore')  # Suppress minor warnings for cleaner output

# =============================================================================
# CONFIGURATION - Adjust these parameters for your analysis
# =============================================================================
CONFIG = {
    "files": {
        # Path to archaeological territories with confirmed sites
        "archaeological_sites": "/kaggle/working/territories_with_archaeological_proximity.geojson",
        # Path to combined deforestation data file
        "deforestation_file": "/kaggle/input/deforestation-data-2000-2023/TerraBrasilis_combined_2008_2023.gpkg",
        # Where to save the output GeoJSON file
        "output_dir": "/kaggle/working",
    },
    "parameters": {
        # Initial search radius around archaeological sites to load deforestation data (km) Purposefully low since this is not running on local
        "search_buffer_km": 20,
        # Buffer around archaeological sites for potential discovery zones (km)
        "archaeological_buffer_km": 15,
        # Buffer around deforestation areas where artifacts might be exposed (km)  
        "deforestation_buffer_km": 3,
        # Pixel size for raster analysis - smaller = more detail but slower (meters)
        "raster_resolution_m": 100,
        # Minimum size for AOI polygons to include in results (km²)
        "min_aoi_area_km2": 1.0,
    },
    "crs": {
        "geographic": "EPSG:4326",  # Standard lat/lon coordinate system
        "projected": "EPSG:31984",  # UTM Zone 20S - accurate for distance measurements in Brazil
    }
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def convert_numpy_types(obj):
    """
    Convert numpy data types to native Python types for JSON serialization.
    This is needed because numpy types can't be directly saved to JSON files.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(element) for element in obj]
    else:
        return obj

def load_archaeological_sites(path: str) -> gpd.GeoDataFrame:
    """
    Load the GeoJSON file containing territories with confirmed archaeological sites.
    These serve as the 'seed points' for our analysis - areas where we know artifacts exist.
    """
    print("📍 Loading confirmed archaeological territories...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archaeological territories file not found: {path}")
    gdf = gpd.read_file(path)
    print(f"   Found {len(gdf)} territories with confirmed archaeological sites.")
    return gdf

def load_combined_deforestation_data(
    archaeological_gdf: gpd.GeoDataFrame,
    config: dict
) -> gpd.GeoDataFrame:
    """
    Load the combined deforestation data file, but only within a search radius 
    of known archaeological sites. This speeds up processing by ignoring 
    deforestation data that's too far away to be relevant.
    
    The theory: Recent deforestation may expose previously hidden archaeological 
    features by removing vegetation cover.
    """
    deforestation_file = Path(config['files']['deforestation_file'])
    buffer_km = config['parameters']['search_buffer_km']
    proj_crs = config['crs']['projected']

    print(f"\n🌳 Loading combined deforestation data within a {buffer_km}km radius...")

    # Step 1: Create a search area by buffering around all archaeological sites
    print("   Creating accurate search area bounding box...")
    arch_projected = archaeological_gdf.to_crs(proj_crs)  # Convert to projected CRS for accurate distance measurements
    
    # Use geometry.unary_union for compatibility with older GeoPandas versions
    search_area_geom = arch_projected.geometry.unary_union.buffer(buffer_km * 1000)  # Buffer in meters
    bbox_geo = gpd.GeoSeries([search_area_geom], crs=proj_crs).to_crs(archaeological_gdf.crs)
    bbox_latlon = bbox_geo.total_bounds  # Get bounding box coordinates

    print(f"   Search BBox (Lat/Lon): ({bbox_latlon[0]:.3f}, {bbox_latlon[1]:.3f}) to ({bbox_latlon[2]:.3f}, {bbox_latlon[3]:.3f})")

    # Step 2: Load the combined deforestation data, filtered to our search area
    print(f"   Loading and filtering combined deforestation data from {deforestation_file.name}...")
    
    if not deforestation_file.exists():
        # Try alternative file formats
        alt_geojson = deforestation_file.with_suffix('.geojson')
        if alt_geojson.exists():
            deforestation_file = alt_geojson
        else:
            raise FileNotFoundError(f"Deforestation file not found: {deforestation_file} or {alt_geojson}")
    
    try:
        # Use bbox parameter to only load data within our search area - much faster!
        deforestation_gdf = gpd.read_file(deforestation_file, bbox=tuple(bbox_latlon))
        
        if deforestation_gdf.empty:
            raise ValueError("No deforestation data found within the search area.")
        
        # Clip to the exact search area (bbox is rectangular, this makes it precise)
        deforestation_gdf = gpd.clip(deforestation_gdf, bbox_geo)
        
        if deforestation_gdf.empty:
            raise ValueError("No deforestation data remains after clipping to search area.")
            
        print(f"      Found {len(deforestation_gdf)} deforestation polygons within the search area.")
        
        # Ensure same CRS as archaeological data
        deforestation_gdf = deforestation_gdf.to_crs(archaeological_gdf.crs)
        
        return deforestation_gdf
        
    except Exception as e:
        raise RuntimeError(f"Could not process deforestation file: {e}")

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_rasterized_analysis(config: dict):
    """
    Main pipeline for the high-speed rasterized AOI analysis.
    
    This function converts vector polygons to raster grids for fast spatial analysis.
    Why raster? Vector operations (like buffering and intersecting thousands of polygons) 
    can be very slow. Raster operations work on pixels and use optimized array operations.
    """
    output_dir = Path(config['files']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {}

    try:
        # =================================================================
        # STEP 1: LOAD AND PREPARE DATA
        # =================================================================
        print("🔄 STEP 1: Loading and preparing spatial data...")
        
        # Load archaeological sites (our "seed points")
        archaeological_gdf = load_archaeological_sites(config['files']['archaeological_sites'])
        
        # Load combined deforestation data (filtered to relevant area only)
        deforestation_gdf = load_combined_deforestation_data(archaeological_gdf, config)
        
        # Dissolve deforestation polygons to create one unified shape per analysis
        # This speeds up rasterization and reduces memory usage
        print("   Dissolving deforestation polygons for efficient processing...")
        dissolved_deforestation = deforestation_gdf.dissolve().reset_index(drop=True)

        # =================================================================
        # STEP 2: SETUP RASTER GRID FOR PIXEL-BASED ANALYSIS
        # =================================================================
        print("\n🔲 STEP 2: Setting up raster grid for high-speed analysis...")
        proj_crs = config['crs']['projected']  # Use projected CRS for accurate distance measurements
        resolution = config['parameters']['raster_resolution_m']  # Size of each pixel in meters

        # Convert both datasets to projected coordinate system
        arch_proj = archaeological_gdf.to_crs(proj_crs)
        defor_proj = dissolved_deforestation.to_crs(proj_crs)

        # Clean geometries to prevent topology errors (invalid polygons can crash rasterization)
        print("   Cleaning geometries to resolve potential topological errors...")
        arch_proj.geometry = arch_proj.geometry.buffer(0)  # buffer(0) fixes invalid geometries
        defor_proj.geometry = defor_proj.geometry.buffer(0)
        arch_proj = arch_proj[~arch_proj.is_empty]  # Remove any empty geometries
        defor_proj = defor_proj[~defor_proj.is_empty]
        print("   Geometry cleaning complete.")

        # Calculate the bounding box that covers both datasets
        print("   Calculating total bounds for raster grid...")
        arch_bounds = arch_proj.total_bounds
        defor_bounds = defor_proj.total_bounds
        xmin = min(arch_bounds[0], defor_bounds[0])
        ymin = min(arch_bounds[1], defor_bounds[1])
        xmax = max(arch_bounds[2], defor_bounds[2])
        ymax = max(arch_bounds[3], defor_bounds[3])
        
        # Add a small buffer around the bounds to avoid edge effects
        buffer_m = 5000  # 5km buffer
        total_bounds = (xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m)
        grid_xmin, grid_ymin, grid_xmax, grid_ymax = total_bounds
        
        # Calculate raster dimensions (number of pixels in each direction)
        width = int((grid_xmax - grid_xmin) / resolution)
        height = int((grid_ymax - grid_ymin) / resolution)
        
        # Create coordinate transformation for the raster
        transform = from_origin(grid_xmin, grid_ymax, resolution, resolution)
        print(f"   Raster grid: {width}x{height} pixels at {resolution}m resolution.")
        
        # =================================================================
        # STEP 3: CONVERT VECTORS TO RASTERS AND CREATE BUFFER ZONES
        # =================================================================
        print("\n🔥 STEP 3: Converting polygons to pixels and creating buffer zones...")
        
        # "Burn" the vector polygons into raster grids (1 = inside polygon, 0 = outside)
        arch_raster = features.rasterize(
            shapes=arch_proj.geometry, 
            out_shape=(height, width), 
            transform=transform, 
            fill=0,  # Background value
            default_value=1,  # Value for pixels inside polygons
            dtype=np.uint8  # Use 8-bit integers to save memory
        )
        defor_raster = features.rasterize(
            shapes=defor_proj.geometry, 
            out_shape=(height, width), 
            transform=transform, 
            fill=0, 
            default_value=1, 
            dtype=np.uint8
        )

        # Calculate buffer sizes in pixels
        arch_buffer_pixels = int(config['parameters']['archaeological_buffer_km'] * 1000 / resolution)
        defor_buffer_pixels = int(config['parameters']['deforestation_buffer_km'] * 1000 / resolution)
        print(f"   Archaeological buffer: {config['parameters']['archaeological_buffer_km']}km ≈ {arch_buffer_pixels} pixels")
        print(f"   Deforestation buffer: {config['parameters']['deforestation_buffer_km']}km ≈ {defor_buffer_pixels} pixels")

        # Apply morphological dilation to create buffer zones
        # This expands the "1" pixels outward by the specified number of pixels
        arch_buffered = ndimage.binary_dilation(arch_raster, iterations=arch_buffer_pixels).astype(np.uint8)
        defor_buffered = ndimage.binary_dilation(defor_raster, iterations=defor_buffer_pixels).astype(np.uint8)

        # =================================================================
        # STEP 4: FIND AREAS OF INTEREST (AOI) 
        # =================================================================
        print("\n🔍 STEP 4: Identifying Areas of Interest through raster intersection...")
        
        # Find pixels that are "1" in BOTH buffered rasters
        # These are areas that are within buffer distance of BOTH archaeological sites AND deforestation
        aoi_raster = np.logical_and(arch_buffered, defor_buffered).astype(np.uint8)
        
        # Check if we found any AOI pixels
        if np.sum(aoi_raster) == 0:
            print("   ⚠️ No AOI areas found after intersection!")
            return None

        # =================================================================
        # STEP 5: CONVERT AOI PIXELS BACK TO VECTOR POLYGONS
        # =================================================================
        print("\n📐 STEP 5: Converting AOI pixels back to vector polygons...")
        
        # Extract polygon shapes from the AOI raster (where pixels = 1)
        aoi_shapes = features.shapes(aoi_raster, mask=(aoi_raster == 1), transform=transform)
        aoi_polygons = [shape(geom) for geom, val in aoi_shapes]
        
        if not aoi_polygons:
            print("   ⚠️ No vector polygons could be extracted!")
            return None

        # Create a GeoDataFrame from the polygons
        aoi_gdf = gpd.GeoDataFrame(geometry=aoi_polygons, crs=proj_crs)
        print(f"   Extracted {len(aoi_gdf)} raw AOI polygons.")

        # =================================================================
        # STEP 6: CLEAN AND FILTER FINAL AOI POLYGONS
        # =================================================================
        print("   Cleaning and filtering final AOI polygons...")
        
        # Calculate area of each polygon and filter by minimum size
        min_area_m2 = config['parameters']['min_aoi_area_km2'] * 1e6  # Convert km² to m²
        aoi_gdf['area_m2'] = aoi_gdf.area
        final_aoi_proj = aoi_gdf[aoi_gdf['area_m2'] >= min_area_m2].reset_index(drop=True)
        
        # Convert back to geographic coordinates (lat/lon) for output
        final_aoi = final_aoi_proj.to_crs(config['crs']['geographic'])

        if final_aoi.empty:
            print(f"   ⚠️ No AOI areas remain after filtering by size (min {config['parameters']['min_aoi_area_km2']} km²).")
            return None
            
        # Calculate summary statistics
        total_aoi_area_km2 = final_aoi_proj.area.sum() / 1e6
        avg_island_size = total_aoi_area_km2 / len(final_aoi) if len(final_aoi) > 0 else 0
        print(f"   Final AOI: {len(final_aoi)} islands, {total_aoi_area_km2:.1f} km² total, {avg_island_size:.1f} km² avg size.")

        # =================================================================
        # STEP 7: SAVE RESULTS AS GEOJSON
        # =================================================================
        print("\n💾 STEP 7: Saving AOI results...")
        
        # Save the AOI polygons as a GeoJSON file
        aoi_path = output_dir / f"archaeological_aoi_{timestamp}.geojson"
        final_aoi.to_file(aoi_path, driver='GeoJSON')
        print(f"   ✅ Saved AOI polygons: {aoi_path}")

        # Create a summary of the analysis results
        summary = {
            "analysis_type": "Archaeological AOI - Rasterized & Optimized (Combined Data)",
            "timestamp": timestamp,
            "description": "Areas of Interest for potential archaeological discoveries based on proximity to known sites and deforestation",
            "methodology": {
                "step_1": "Load archaeological sites and combined deforestation data",
                "step_2": "Convert vector data to raster grid for fast processing",
                "step_3": "Create buffer zones around both datasets using morphological dilation",
                "step_4": "Find intersection of buffered zones as potential discovery areas",
                "step_5": "Convert result back to vector polygons",
                "step_6": "Filter by minimum area and export as GeoJSON"
            },
            "parameters": config['parameters'],
            "results": {
                "aoi_polygons_found": len(final_aoi),
                "aoi_total_area_km2": total_aoi_area_km2,
                "average_aoi_size_km2": avg_island_size,
                "archaeological_sites_analyzed": len(archaeological_gdf),
                "deforestation_polygons_processed": len(deforestation_gdf)
            },
            "output_files": {
                "aoi_geojson": str(aoi_path),
                "analysis_summary": f"aoi_analysis_summary_{timestamp}.json"
            }
        }
        
        # Convert numpy types to regular Python types for JSON serialization
        summary = convert_numpy_types(summary)
        
        # Save the summary as JSON
        summary_path = output_dir / f"aoi_analysis_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   ✅ Saved analysis summary: {summary_path}")

        return summary

    except Exception as e:
        print(f"❌ An unexpected error occurred during analysis: {e}")
        print(f"   Error details: {str(e)}")
        
        # Save error information if we have a summary started
        if summary:
            summary['error'] = str(e)
            summary['status'] = 'failed'
            summary_path = output_dir / f"aoi_analysis_summary_{timestamp}_ERROR.json"
            with open(summary_path, 'w') as f:
                json.dump(convert_numpy_types(summary), f, indent=2)
            print(f"   Error details saved to: {summary_path}")
        return None

# =============================================================================
# EXECUTE THE ANALYSIS
# =============================================================================

print("=" * 80)
print("🏛️  ARCHAEOLOGICAL AREAS OF INTEREST FINDER")
print("   Rasterized Analysis - Combined Data - Lean GeoJSON Output")
print("=" * 80)


input_dir = Path("/kaggle/input")
if input_dir.exists():
    pass
else:
    # Fallback to local file paths for testing outside Kaggle
    CONFIG['files']['archaeological_sites'] = "notebooks/output/gee_geoglyph_analysis/territories_with_archaeological_sites_20250627_114748.geojson"
    CONFIG['files']['deforestation_file'] = "data/TerraBrasilis/geojson/TerraBrasilis_combined_2008_2023.gpkg"
    CONFIG['files']['output_dir'] = "output/archaeological_aoi_kaggle"

print(f"\n🚀 Starting analysis with configuration:")
print(f"   • Archaeological buffer: {CONFIG['parameters']['archaeological_buffer_km']} km")
print(f"   • Deforestation buffer: {CONFIG['parameters']['deforestation_buffer_km']} km")
print(f"   • Raster resolution: {CONFIG['parameters']['raster_resolution_m']} m")
print(f"   • Minimum AOI size: {CONFIG['parameters']['min_aoi_area_km2']} km²")

# =============================================================================
# RUN THE MAIN ANALYSIS
# =============================================================================

result = run_rasterized_analysis(CONFIG)

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

if result:
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📊 Results Summary:")
    print(f"   • AOI Polygons Found: {result['results']['aoi_polygons_found']}")
    print(f"   • Total AOI Area: {result['results']['aoi_total_area_km2']:.1f} km²")
    print(f"   • Average AOI Size: {result['results']['average_aoi_size_km2']:.1f} km²")
    print(f"   • Archaeological Sites Analyzed: {result['results']['archaeological_sites_analyzed']}")
    print(f"   • Deforestation Polygons Processed: {result['results']['deforestation_polygons_processed']}")
    
    print(f"\n📂 Output Files:")
    for file_type, file_path in result['output_files'].items():
        print(f"   • {file_type}: {Path(file_path).name}")
        
else:
    print("\n" + "="*60)
    print("❌ ANALYSIS FAILED")
    print("="*60)
    print("No AOI areas were identified. This could be due to:")
    print("   • No deforestation within buffer distance of archaeological sites")
    print("   • All potential AOI areas were smaller than the minimum size threshold")
    print("   • Data loading issues")
    print("Check the detailed output above for more information.")

# Display all output files available for download
print(f"\n📁 All files available in output directory:")
working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(CONFIG['files']['output_dir'])
if working_dir.exists():
    output_files = list(working_dir.glob("*"))
    if output_files:
        for file in output_files:
            if file.is_file():
                size_kb = file.stat().st_size / 1024
                print(f"   📄 {file.name} ({size_kb:.1f} KB)")
    else:
        print("   No output files found.")
else:
    print("   Output directory not found.")

print(f"\n🎯 Analysis complete! The GeoJSON file contains the Areas of Interest for potential archaeological discoveries.")



!pip install -q contextily
# Create a static map showing the generated AOI polygons
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from pathlib import Path
import glob
import json

def visualize_aoi_results():
    """
    Create a static map showing the Areas of Interest generated from the analysis
    """
    print("🗺️ Creating AOI Visualization Map...")
    
    # Try to find the most recent AOI file
    working_dir = Path("/kaggle/working")
    if working_dir.exists():
        aoi_files = list(working_dir.glob("archaeological_aoi_*.geojson"))
        if aoi_files:
            # Get the most recent AOI file
            latest_aoi_file = max(aoi_files, key=lambda x: x.stat().st_mtime)
            print(f"📁 Loading AOI from: {latest_aoi_file.name}")
        else:
            print("❌ No AOI files found. Please run the AOI analysis first.")
            return
    else:
        print("❌ Working directory not found. Please run the AOI analysis first.")
        return
    
    try:
        # Load the AOI polygons
        aoi_gdf = gpd.read_file(latest_aoi_file)
        print(f"✅ Loaded {len(aoi_gdf)} AOI polygons")
        
        # Load indigenous territories for context (if available)
        territories_gdf = None
        try:
            territories_file = working_dir / "territories_with_archaeological_proximity.geojson"
            if territories_file.exists():
                territories_gdf = gpd.read_file(territories_file)
                print(f"✅ Loaded {len(territories_gdf)} indigenous territories for context")
        except Exception as e:
            print(f"⚠️ Could not load territories: {e}")
        
        # Create the visualization
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Plot indigenous territories first (as background context)
        if territories_gdf is not None:
            territories_gdf.plot(ax=ax, color='lightblue', alpha=0.3, edgecolor='blue', 
                               linewidth=0.5, label='Indigenous Territories with Archaeological Proximity')
        
        # Plot AOI polygons prominently
        aoi_gdf.plot(ax=ax, color='red', alpha=0.7, edgecolor='darkred', 
                     linewidth=2, label='Areas of Interest (AOI)')
        
        # Add basemap for geographic context
        try:
            ctx.add_basemap(ax, crs=aoi_gdf.crs.to_string(), 
                           source=ctx.providers.CartoDB.Positron, alpha=0.7)
            print("✅ Added basemap context")
        except Exception as e:
            print(f"⚠️ Could not add basemap: {e}")
        
        # Calculate and display AOI statistics
        if aoi_gdf.crs.to_string() != "EPSG:4326":
            aoi_proj = aoi_gdf.to_crs("EPSG:3857")  # Web Mercator for area calculation
        else:
            aoi_proj = aoi_gdf.to_crs("EPSG:3857")
        
        total_area_km2 = aoi_proj.area.sum() / 1e6  # Convert m² to km²
        avg_polygon_size = total_area_km2 / len(aoi_gdf)
        
        # Set map extent
        bounds = aoi_gdf.total_bounds
        margin = 0.5  # degrees
        ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
        ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
        
        # Customize the map
        ax.set_title(f'Archaeological Areas of Interest (AOI)\n'
                    f'{len(aoi_gdf)} priority zones identified • {total_area_km2:.1f} km² total area\n'
                    f'Generated from intersection of archaeological sites and deforestation data',
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Add legend
        legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        legend.set_title('Map Layers', prop={'size': 12, 'weight': 'bold'})
        
        # Add statistics text box
        stats_text = f"""AOI Analysis Results:
• Priority Zones: {len(aoi_gdf)}
• Total Area: {total_area_km2:.1f} km²
• Average Size: {avg_polygon_size:.1f} km²
• Analysis Method: Rasterized intersection"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add grid and clean up
        ax.grid(True, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Display summary information
        print(f"\n📊 AOI ANALYSIS SUMMARY")
        print(f"{'='*40}")
        print(f"Priority zones identified: {len(aoi_gdf)}")
        print(f"Total area coverage: {total_area_km2:.1f} km²")
        print(f"Average polygon size: {avg_polygon_size:.1f} km²")
        print(f"Largest polygon: {aoi_proj.area.max()/1e6:.1f} km²")
        print(f"Smallest polygon: {aoi_proj.area.min()/1e6:.1f} km²")
        
        # Load and display analysis summary if available
        try:
            summary_files = list(working_dir.glob("aoi_analysis_summary_*.json"))
            if summary_files:
                latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
                with open(latest_summary) as f:
                    summary = json.load(f)
                
                print(f"\n🔧 ANALYSIS PARAMETERS")
                print(f"{'='*40}")
                params = summary.get('parameters', {})
                print(f"Archaeological buffer: {params.get('archaeological_buffer_km', 'N/A')} km")
                print(f"Deforestation buffer: {params.get('deforestation_buffer_km', 'N/A')} km")
                print(f"Minimum AOI size: {params.get('min_aoi_area_km2', 'N/A')} km²")
                print(f"Raster resolution: {params.get('raster_resolution_m', 'N/A')} meters")
        except Exception as e:
            print(f"⚠️ Could not load analysis summary: {e}")
        
    except Exception as e:
        print(f"❌ Error creating AOI visualization: {e}")
        return

# Generate the AOI visualization
visualize_aoi_results()



import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Adjust for speed vs. density trade-off
# =============================================================================

# 🚀 KAGGLE SPEED OPTIMIZATION: Adjust these values to run faster with fewer points
SAMPLING_CONFIG = {
    "point_spacing_meters": 500,  # 500m between sample points.
                                  # 🚀 SPEED UP: Increase to 1000-2000m for faster processing with fewer points
    "min_points_per_polygon": 3,   # Minimum coverage for small polygons.
                                   # 🚀 SPEED UP: Keep low (1-3) for good coverage
    "max_points_per_polygon": 500, # Safety valve to prevent excessive API calls.
                                   # 🚀 SPEED UP: Lower to 50-100 for much faster processing
    "buffer_inside_polygon_m": -50 # Negative buffer to stay 50m inside polygon edges.
                                   # 🚀 SPEED UP: Keep this value, it's already optimized
}

# 🚀 KAGGLE QUICK CONFIG - Uncomment these lines for much faster processing:
SAMPLING_CONFIG_FAST = {
    "point_spacing_meters": 1500,      # 1.5km spacing = fewer points
    "min_points_per_polygon": 1,       # Just 1 point minimum  
    "max_points_per_polygon": 50,      # Max 50 points per polygon
    "buffer_inside_polygon_m": -50     # Keep buffer
}
# This will run 5-10x faster but with lower sampling density

# Choose configuration: Use SAMPLING_CONFIG_FAST for speed, SAMPLING_CONFIG for density
USE_FAST_CONFIG = True  # Set to False for higher density, True for speed
CONFIG = SAMPLING_CONFIG_FAST if USE_FAST_CONFIG else SAMPLING_CONFIG

# CRS Configuration - Should match your analysis script
CRS_CONFIG = {
    "projected": "EPSG:31984", # UTM Zone 20S for accurate meter-based spacing
    "geographic": "EPSG:4326"  # Standard Lat/Lon for output
}

# =============================================================================
# MAIN SAMPLING FUNCTION
# =============================================================================

def generate_sampling_points(input_geojson_path: Path, output_geojson_path: Path):
    """
    Generates a grid of sampling points within each polygon of an input GeoJSON file.
    Optimized for Kaggle notebook execution.
    """
    if not input_geojson_path.exists():
        print(f"❌ Input file not found: {input_geojson_path}")
        return None

    print(f"📍 Loading {input_geojson_path.name}...")
    aoi_gdf = gpd.read_file(input_geojson_path)
    
    # Add polygon statistics
    if 'area_m2' in aoi_gdf.columns:
        total_area = aoi_gdf['area_m2'].sum() / 1e6  # Convert to km²
        print(f"   {len(aoi_gdf)} polygons, {total_area:.1f} km² total area")

    # Project to the projected CRS for accurate distance calculations
    aoi_proj = aoi_gdf.to_crs(CRS_CONFIG['projected'])

    all_points = []
    total_points_generated = 0
    polygons_processed = 0

    print(f"🔥 Processing {len(aoi_proj)} polygons...")
    
    for index, polygon_row in aoi_proj.iterrows():
        polygon = polygon_row.geometry
        
        # Skip invalid polygons
        if polygon.is_empty or not polygon.is_valid:
            continue
            
        # Apply buffer
        try:
            polygon_buffered = polygon.buffer(CONFIG['buffer_inside_polygon_m'])
        except Exception as e:
            polygon_buffered = polygon
            
        if polygon_buffered.is_empty:
            continue
            
        xmin, ymin, xmax, ymax = polygon_buffered.bounds
        
        # Generate grid coordinates
        spacing = CONFIG['point_spacing_meters']
        x_coords = np.arange(xmin, xmax, spacing)
        y_coords = np.arange(ymin, ymax, spacing)
        
        # Create a list of all potential points in the bounding box
        grid_points = [Point(x, y) for x in x_coords for y in y_coords]

        if not grid_points:
            points_within_poly = [polygon.centroid]
        else:
            # Filter points to only those inside the actual polygon
            points_within_poly = []
            for point in grid_points:
                try:
                    if polygon_buffered.contains(point):
                        points_within_poly.append(point)
                except Exception as e:
                    continue

        num_points = len(points_within_poly)
        
        # Apply min/max constraints
        if 0 < num_points < CONFIG['min_points_per_polygon']:
            pass
        elif num_points == 0:
            points_within_poly = [polygon.centroid]
            num_points = 1
            
        if num_points > CONFIG['max_points_per_polygon']:
            points_within_poly = np.random.choice(points_within_poly, CONFIG['max_points_per_polygon'], replace=False).tolist()
            num_points = CONFIG['max_points_per_polygon']
        
        total_points_generated += num_points
        polygons_processed += 1
        
        # Show progress every 10 polygons
        if polygons_processed % 10 == 0:
            print(f"   Processed {polygons_processed}/{len(aoi_proj)} polygons, {total_points_generated} points so far")

        # Store points with metadata
        for i, point in enumerate(points_within_poly):
            all_points.append({
                "geometry": point,
                "polygon_id": index,  # Original polygon's index
                "point_index": i,
                "polygon_area_m2": polygon_row.get('area_m2', 0)  # Include original area
            })

    if not all_points:
        print("❌ No sampling points were generated. Check input data and configuration.")
        return None

    print(f"\n📊 Summary: {polygons_processed} polygons → {total_points_generated} points ({total_points_generated / polygons_processed:.1f} avg/polygon)")

    # Create the final GeoDataFrame and convert to lat/lon
    points_final_proj = gpd.GeoDataFrame(all_points, crs=CRS_CONFIG['projected'])
    points_final_geo = points_final_proj.to_crs(CRS_CONFIG['geographic'])

    print(f"💾 Saving to {output_geojson_path.name}...")
    points_final_geo.to_file(output_geojson_path, driver='GeoJSON')
    
    # Create a summary report
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_polygons": len(aoi_gdf),
        "processed_polygons": polygons_processed,
        "total_points_generated": total_points_generated,
        "average_points_per_polygon": total_points_generated / polygons_processed if polygons_processed > 0 else 0,
        "estimated_api_calls": total_points_generated,
        "sampling_config": CONFIG,
        "config_type": "FAST" if USE_FAST_CONFIG else "DENSE",
        "input_file": str(input_geojson_path),
        "output_file": str(output_geojson_path)
    }
    
    summary_file = output_geojson_path.parent / "sampling_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

# =============================================================================
# EXECUTE THE SAMPLING
# =============================================================================

print("🗺️ AOI POLYGON SAMPLER")
print(f"   {'FAST' if USE_FAST_CONFIG else 'DENSE'} Configuration")

# =============================================================================
# FILE CONFIGURATION
# =============================================================================

# Direct file paths - no auto-detection needed
input_file = Path("/kaggle/working/archaeological_aoi_20250629_200533.geojson")
output_dir = Path("/kaggle/working")
output_file = output_dir / "sampling_points.geojson"

print("📂 Configuration:")
print(f"   Input:  {input_file.name}")
print(f"   Output: {output_file.name}")
print(f"   Config: {'FAST' if USE_FAST_CONFIG else 'DENSE'} ({CONFIG['point_spacing_meters']}m spacing, max {CONFIG['max_points_per_polygon']} pts/polygon)")

# =============================================================================
# RUN THE SAMPLING
# =============================================================================

if input_file.exists():
    result = generate_sampling_points(input_geojson_path=input_file, output_geojson_path=output_file)
    
    if result:
        print(f"\n✅ SUCCESS! Generated {result['total_points_generated']} sampling points from {result['processed_polygons']} polygons")
        print(f"📁 Output: {output_file.name} ({result['config_type']} config)")
    else:
        print("\n❌ SAMPLING FAILED - Check error messages above")
else:
    print(f"\n❌ Input file not found: {input_file}")

print(f"\n🎯 Complete! Use {output_file.name} for satellite tile download.")



# Skip this cell if the satellite images have already been generated.
import os
import sys
import requests
import json
import re
import math
from pathlib import Path
from typing import List, Dict, Optional

# --- START: CORRECT WAY TO GET KAGGLE SECRETS ---
# Use the kaggle_secrets library instead of os.getenv
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GOOGLE_MAPS_API_KEY = user_secrets.get_secret("GOOGLE_MAPS_API_KEY")
    print("✅ Successfully retrieved Google Maps API key from Kaggle Secrets.")
except Exception as e:
    GOOGLE_MAPS_API_KEY = ''
    print("⚠️ WARNING: Could not retrieve Google Maps API key from Kaggle Secrets.")
    print("   Please ensure you have added a secret named 'GOOGLE_MAPS_API_KEY'.")
    print("   Go to Add-ons -> Secrets -> Add a new secret.")
# --- END: CORRECT WAY TO GET KAGGLE SECRETS ---

# Configuration for Kaggle
# NOTE: Using the file generated by the previous step
DEFAULT_GEOJSON_FILE = "/kaggle/working/sampling_points.geojson" 
DEFAULT_OUTPUT_DIR = "/kaggle/working/satellite_images"
DEFAULT_ZOOM = 18
DEFAULT_MAX_IMAGES = 25 # Increased for better coverage
DEFAULT_START_INDEX = 0

# Check if we should skip execution
output_dir_path = Path(DEFAULT_OUTPUT_DIR)
if output_dir_path.exists() and any(output_dir_path.iterdir()):
    print(f"\nOutput directory '{DEFAULT_OUTPUT_DIR}' already contains files. Skipping image generation.")
    # Set a flag or exit if running as a script, for notebooks we just print and continue
    # to allow inspection. In a real pipeline, you might 'sys.exit()' here.
    execution_should_continue = False
else:
    execution_should_continue = True


# --- Function definitions (no changes needed here) ---

def clean_description(desc):
    """Clean and format the description text for use in filenames."""
    if not desc: return ""
    clean_text = re.sub(r'<[^>]+>', ' ', str(desc))
    clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text[:50].replace(' ', '_').lower()

def parse_geojson(geojson_file):
    """Parse a GeoJSON file and extract coordinates and descriptions from features."""
    locations = []
    with open(geojson_file, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    features = geojson_data.get('features', [])
    for feature in features:
        try:
            geometry = feature.get('geometry', {})
            properties = feature.get('properties', {})
            # Use the polygon ID and status from the sampling points file
            desc = f"poly_{properties.get('polygon_id', 'N/A')}_{properties.get('proximity_status', 'N/A')}"
            lon, lat = geometry.get('coordinates', [None, None])
            if lat is not None and lon is not None:
                locations.append({'lat': lat, 'lon': lon, 'description': clean_description(desc)})
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error parsing feature: {e}")
    return locations

def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x_tile, y_tile

def get_tile_image(api_key, x, y, zoom, map_type="satellite", session=None):
    base_url = f"https://tile.googleapis.com/v1/2dtiles/{zoom}/{x}/{y}"
    params = {"key": api_key, "session": session}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"    ❌ Error fetching tile: {e}")
        return None

def save_image(image_data, output_path):
    try:
        with open(output_path, 'wb') as f: f.write(image_data)
        return True
    except IOError as e:
        print(f"    ❌ Error saving image {output_path}: {e}")
        return False

# --- Main execution block ---
if execution_should_continue:
    print("\n🛰️ Starting Google Maps Satellite Image Download")
    
    GEOJSON_FILE = DEFAULT_GEOJSON_FILE
    OUTPUT_DIR = Path(DEFAULT_OUTPUT_DIR)
    ZOOM = DEFAULT_ZOOM
    MAX_IMAGES = DEFAULT_MAX_IMAGES
    START_INDEX = DEFAULT_START_INDEX
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not GOOGLE_MAPS_API_KEY:
        print("❌ CRITICAL: No API key available. Stopping execution.")
    elif not Path(GEOJSON_FILE).exists():
        print(f"❌ CRITICAL: Input file not found: {GEOJSON_FILE}")
        print("   Please run the previous cell (the polygon sampler) to generate it.")
    else:
        # Create a new session for this run
        print("Creating new session for Google Maps Tiles API...")
        try:
            resp = requests.post(
                "https://tile.googleapis.com/v1/createSession",
                params={"key": GOOGLE_MAPS_API_KEY},
                json={"mapType": "satellite", "language": "en-US", "region": "US"},
                headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            session_token = resp.json()["session"]
            print(f"✅ Session created successfully.")
        except requests.exceptions.RequestException as e:
            print(f"❌ Error creating session: {e.response.text if e.response else e}")
            print("   Please check your API key and ensure the Maps Tiles API is enabled in your Google Cloud project.")
            session_token = None

        if session_token:
            print(f"📋 Parsing sampling points from: {GEOJSON_FILE}")
            locations = parse_geojson(GEOJSON_FILE)
            if not locations:
                print("❌ No locations found in the GeoJSON file.")
            else:
                print(f"✅ Found {len(locations)} sampling points.")
                
                # Limit the number of locations to process
                end_index = min(START_INDEX + MAX_IMAGES, len(locations))
                locations_to_process = locations[START_INDEX:end_index]

                print(f"🔄 Downloading {len(locations_to_process)} satellite images (from index {START_INDEX} to {end_index-1})...")
                
                successful_downloads = 0
                for i, location in enumerate(locations_to_process, START_INDEX):
                    lat, lon, desc = location['lat'], location['lon'], location['description']
                    x_tile, y_tile = latlon_to_tile(lat, lon, ZOOM)
                    
                    print(f"\n📍 Processing point {i+1}/{len(locations)} (Lat: {lat:.5f}, Lon: {lon:.5f})")
                    
                    safe_desc = f"_{desc}" if desc else ""
                    output_file_path = OUTPUT_DIR / f"tile_{i:04d}_z{ZOOM}_x{x_tile}_y{y_tile}{safe_desc}.png"
                    
                    if output_file_path.exists():
                        print(f"   ⏭️ File already exists, skipping.")
                        successful_downloads += 1
                        continue
                    
                    image_data = get_tile_image(GOOGLE_MAPS_API_KEY, x_tile, y_tile, ZOOM, session=session_token)
                    
                    if image_data:
                        if save_image(image_data, output_file_path):
                            print(f"   ✅ Saved: {output_file_path.name}")
                            successful_downloads += 1

                print(f"\n🎉 Processing complete!")
                print(f"✅ Successfully downloaded {successful_downloads}/{len(locations_to_process)} images.")
                print(f"📁 Images saved to: {OUTPUT_DIR}")


# Check if output file already exists and skip if it does
output_file = "geoglyph_classification_results_4.1.jsonl"
input_dir = "/kaggle/input/geoglyph-classification-results-4-1"
if os.path.exists(os.path.join(input_dir, output_file)):
    print(f"Output file {output_file} already exists in {input_dir}. Skipping classification.")
else:
    # Otherwise, call OpenAI to classify the results.
    openai.api_key = os.getenv("OPENAI_API_KEY")
    MAX_RPS = 83  # Tier 3
    MAX_WORKERS = 50  # Tune as needed
    
    rate_lock = threading.Lock()
    last_request_times = []
    
    def get_lat_lon_from_filename(filename):
        m = re.search(r'_z(\d+)_x(\d+)_y(\d+)', filename)
        if not m:
            raise ValueError(f"Filename does not match expected pattern: {filename}")
        z, x, y = map(int, m.groups())
        n = 2.0 ** z
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg
    
    def extract_json_from_response(response_text):
        match = re.search(r"```json(.*)```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            match = re.search(r"({.*})", response_text, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                json_str = response_text.strip()
        return json_str
    
    def rate_limited():
        with rate_lock:
            now = time.time()
            # Remove timestamps older than 1 second
            while last_request_times and now - last_request_times[0] > 1:
                last_request_times.pop(0)
            if len(last_request_times) >= MAX_RPS:
                sleep_time = 1 - (now - last_request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            last_request_times.append(time.time())
    
    def classify_image_worker(args):
        image_path, filename, lat, lon = args
        retries = 0
        while retries < 5:
            try:
                rate_limited()
                with open(image_path, "rb") as img_file:
                    img_bytes = img_file.read()
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    prompt = (
                        "You are an expert in Amazonian archaeology. "
                        "Given the attached satellite image, return a JSON object with the following keys: "
                        "'filename' (string), 'latitude' (float), 'longitude' (float), "
                        "'description' (string, 1-2 sentences evaluating the likelihood of a geoglyph or earthwork), "
                        "and 'rating' (float between 0.00 and 1.00, where 0.00 means 'no chance' and 1.00 means 'certain'). "
                        "Be consistent in using the hundreds decimal places."
                        "Do not include any text outside the JSON object. "
                        "Here is the metadata for this image: "
                        f"filename: {filename}, latitude: {lat}, longitude: {lon}."
                    )
                    response = openai.chat.completions.create(
                        model="gpt-4.1-2025-04-14",
                        messages=[
                            {"role": "system", "content": "You are an expert in Amazonian archaeology."},
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                            ]}
                        ],
                        max_completion_tokens=300
                    )
                return filename, response.choices[0].message.content
            except openai.RateLimitError:
                retries += 1
                time.sleep(2 ** retries)  # Exponential backoff
            except Exception as e:
                return filename, f"ERROR: {e}"
        return filename, "ERROR: Max retries exceeded"
    
    def classify_geoglyph_tiles_parallel(directory="/kaggle/working/", num_files=5, output_file="geoglyph_results_4.1.jsonl"):
        files = [f for f in os.listdir(directory) if f.endswith(".png")]
        args_list = []
        for filename in files[:num_files]:
            lat, lon = get_lat_lon_from_filename(filename)
            image_path = os.path.join(directory, filename)
            args_list.append((image_path, filename, lat, lon))
    
        with open(output_file, "a") as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, tqdm(total=len(args_list), desc="Classifying tiles") as pbar:
            futures = {executor.submit(classify_image_worker, args): args[1] for args in args_list}
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    filename, result = future.result()
                    json_str = extract_json_from_response(result)
                    try:
                        result_json = json.loads(json_str)
                        f.write(json.dumps(result_json) + "\n")
                        f.flush()
                    except Exception as e:
                        print(f"Failed to parse result for {filename}: {result}\nError: {e}")
                except Exception as e:
                    print(f"Worker failed for {filename}: {e}")
                pbar.update(1)
    
    classify_geoglyph_tiles_parallel(num_files=30000)


from typing import List, Tuple
import json

def extract_high_rating_coordinates(jsonl_file: str, min_rating: float = 0.8) -> List[Tuple[float, float, str, float, str]]:
    """
    Extract latitude, longitude, description, rating, and filename from JSONL file for entries with rating >= min_rating.
    
    Args:
        jsonl_file: Path to the JSONL file
        min_rating: Minimum rating threshold (default: 0.8)
    
    Returns:
        List of tuples containing (latitude, longitude, description, rating, filename)
    """
    coordinates = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # Check if required fields exist
                if 'latitude' in data and 'longitude' in data and 'rating' in data and 'description' in data and 'filename' in data:
                    rating = data['rating']
                    
                    # Filter by rating threshold
                    if rating >= min_rating:
                        lat = data['latitude']
                        lon = data['longitude']
                        description = data['description']
                        filename = data['filename']
                        coordinates.append((lat, lon, description, rating, filename))
                        
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue
            except (KeyError, TypeError) as e:
                print(f"Warning: Missing required field on line {line_num}: {e}")
                continue
    
    return coordinates

def main():
    """Main function to run the coordinate extraction."""
    jsonl_file = "/kaggle/input/geoglyph-classification-results-4-1/geoglyph_classification_results_4.1.jsonl"
    min_rating = 0.8
    
    print(f"Extracting coordinates from {jsonl_file} with rating >= {min_rating}")
    
    coordinates = extract_high_rating_coordinates(jsonl_file, min_rating)
    
    print(f"Found {len(coordinates)} entries with rating >= {min_rating}")
    
    # Print first 5 results as example
    if coordinates:
        print("\nFirst 5 entries (latitude, longitude, rating, filename, description):")
        for i, (lat, lon, description, rating, filename) in enumerate(coordinates[:5], 1):
            print(f"{i}. ({lat:.6f}, {lon:.6f}) - Rating: {rating:.3f} - File: {filename}")
            print(f"   Description: {description[:100]}{'...' if len(description) > 5 else ''}")
            print()
        
        if len(coordinates) > 5:
            print(f"... and {len(coordinates) - 5} more")
    
    return coordinates

if __name__ == "__main__":
    coordinates = main()


print(coordinates[0])


import folium
from folium import plugins
import base64
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import os
import io
import re
import math
from typing import Dict, Tuple, Optional, List

def tile_to_latlon(x: int, y: int, z: int) -> Tuple[float, float]:
    """
    Convert tile coordinates (x, y, z) to latitude and longitude.
    Uses the standard Web Mercator / Google Maps tile system.
    """
    n = 2.0 ** z
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def parse_coordinates_from_filename(filename: str) -> Optional[Tuple[float, float]]:
    """
    Parse coordinates from different filename formats:
    - tile_8754_z18_x88167_y137765_mt.png
    - geoglyph_1_z17_x42229_y69015_linear_and_geometric_markings_.png
    """
    # Try tile format first
    tile_match = re.search(r'tile_\d+_z(\d+)_x(\d+)_y(\d+)', filename)
    if tile_match:
        z, x, y = map(int, tile_match.groups())
        return tile_to_latlon(x, y, z)
    
    # Try geoglyph format
    geoglyph_match = re.search(r'geoglyph_\d+_z(\d+)_x(\d+)_y(\d+)', filename)
    if geoglyph_match:
        z, x, y = map(int, geoglyph_match.groups())
        return tile_to_latlon(x, y, z)
    
    return None

def create_image_coordinate_mapping(image_directory: str) -> Dict[str, Tuple[float, float]]:
    """
    Create a mapping of image filenames to their coordinates.
    """
    image_mapping = {}
    
    if not os.path.exists(image_directory):
        print(f"Warning: Image directory not found: {image_directory}")
        return image_mapping
    
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            coords = parse_coordinates_from_filename(filename)
            if coords:
                image_mapping[filename] = coords
                
    print(f"Found {len(image_mapping)} images with parseable coordinates in {image_directory}")
    return image_mapping

def match_results_to_images(coordinates: List[Tuple], image_mapping: Dict[str, Tuple[float, float]], 
                          tolerance: float = 0.01) -> Dict[int, str]:
    """
    Match classification results to available images based on coordinate proximity.
    """
    matches = {}
    
    for i, (result_lat, result_lon, description, rating, original_filename) in enumerate(coordinates):
        best_match = None
        min_distance = float('inf')
        
        for img_filename, (img_lat, img_lon) in image_mapping.items():
            # Calculate simple Euclidean distance
            distance = math.sqrt((result_lat - img_lat)**2 + (result_lon - img_lon)**2)
            
            if distance < tolerance and distance < min_distance:
                min_distance = distance
                best_match = img_filename
        
        if best_match:
            matches[i] = best_match
            
    print(f"Matched {len(matches)}/{len(coordinates)} results to available images")
    return matches

def create_interactive_map(coordinates, 
                          kaggle_image_path="/kaggle/input/top-candidates",
                          fallback_image_directory="/kaggle/input/TerraBrasilis-tiles-2023/TerraBrasilis"):
    """
    Create an enhanced interactive map showing high-probability archaeological sites
    with popup images from Kaggle dataset, falling back to original images if needed.
    """
    if not coordinates:
        print("No coordinates provided for mapping.")
        return None
    
    # Create image coordinate mapping from Kaggle dataset
    print("🔍 Connecting results to Kaggle images...")
    kaggle_image_mapping = create_image_coordinate_mapping(kaggle_image_path)
    
    # Match results to Kaggle images
    result_to_kaggle_image = match_results_to_images(coordinates, kaggle_image_mapping)
    
    # Calculate map center
    lats = [coord[0] for coord in coordinates]
    lons = [coord[1] for coord in coordinates]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Add satellite layer as option
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add markers for each high-probability site
    for i, (lat, lon, description, rating, filename) in enumerate(coordinates, 1):
        # Determine marker color based on rating
        if rating >= 0.95:
            color = 'red'
        elif rating >= 0.9:
            color = 'orange'
        else:
            color = 'green'
        
        # Create popup content
        popup_html = f"""
        <div style="width: 300px;">
            <h4>Site #{i}</h4>
            <p><strong>Coordinates:</strong> {lat:.6f}, {lon:.6f}</p>
            <p><strong>AI Rating:</strong> {rating:.3f}/1.00</p>
            <p><strong>Description:</strong> {description}</p>
            <p><strong>Original File:</strong> {filename}</p>
        """
        
        # Try to add the matched Kaggle image first
        image_added = False
        if i-1 in result_to_kaggle_image:  # i-1 because we're using 0-based indexing for matches
            kaggle_image_filename = result_to_kaggle_image[i-1]
            kaggle_image_path_full = os.path.join(kaggle_image_path, kaggle_image_filename)
            
            if os.path.exists(kaggle_image_path_full):
                try:
                    # Convert image to base64 for embedding
                    with open(kaggle_image_path_full, 'rb') as img_file:
                        img_data = img_file.read()
                        img_b64 = base64.b64encode(img_data).decode()
                        popup_html += f'<br><p><strong>Best Candidate Image:</strong> {kaggle_image_filename}</p>'
                        popup_html += f'<br><img src="data:image/png;base64,{img_b64}" style="width:280px;height:auto;">'
                        image_added = True
                except Exception as e:
                    popup_html += f'<br><p style="color:red;">Error loading candidate image: {e}</p>'
        
        # Fallback to original image if no Kaggle image matched
        if not image_added:
            fallback_image_path = os.path.join(fallback_image_directory, filename)
            if os.path.exists(fallback_image_path):
                try:
                    with open(fallback_image_path, 'rb') as img_file:
                        img_data = img_file.read()
                        img_b64 = base64.b64encode(img_data).decode()
                        popup_html += f'<br><p><strong>Original Image:</strong> {filename}</p>'
                        popup_html += f'<br><img src="data:image/png;base64,{img_b64}" style="width:280px;height:auto;">'
                        image_added = True
                except Exception as e:
                    popup_html += f'<br><p style="color:red;">Error loading original image: {e}</p>'
        
        if not image_added:
            popup_html += '<br><p style="color:orange;">No matching image found</p>'
        
        popup_html += "</div>"
        
        # Add marker to map
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"Site #{i} (Rating: {rating:.3f})",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Calculate statistics for the legend
    ratings = [coord[3] for coord in coordinates]
    very_high_count = sum(1 for r in ratings if r >= 0.95)
    high_count = sum(1 for r in ratings if 0.9 <= r < 0.95)
    moderate_count = sum(1 for r in ratings if 0.8 <= r < 0.9)
    total_sites = len(coordinates)
    matched_images = len(result_to_kaggle_image)
    
    # Add an enhanced legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 20px; left: 20px; width: 280px; height: auto; 
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border: 2px solid #dee2e6; 
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                z-index: 9999; 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 13px; 
                padding: 16px;
                line-height: 1.4;">
        
        <div style="text-align: center; margin-bottom: 12px;">
            <h3 style="margin: 0; color: #2c3e50; font-size: 16px; font-weight: 600;">
                🏛️ Archaeological Sites Discovery
            </h3>
            <p style="margin: 4px 0 0 0; color: #6c757d; font-size: 11px;">
                AI-Powered Amazon Analysis
            </p>
        </div>
        
        <div style="border-top: 1px solid #dee2e6; padding-top: 10px; margin-bottom: 10px;">
            <p style="margin: 0 0 6px 0; color: #495057; font-weight: 500;">AI Confidence Levels:</p>
            
            <div style="display: flex; align-items: center; margin: 4px 0;">
                <span style="color: #dc3545; font-size: 16px; margin-right: 8px;">●</span>
                <span style="color: #2c3e50;">≥ 0.95 - Highest ({very_high_count} sites)</span>
            </div>
            
            <div style="display: flex; align-items: center; margin: 4px 0;">
                <span style="color: #fd7e14; font-size: 16px; margin-right: 8px;">●</span>
                <span style="color: #2c3e50;">≥ 0.90 - High ({high_count} sites)</span>
            </div>
            
            <div style="display: flex; align-items: center; margin: 4px 0;">
                <span style="color: #28a745; font-size: 16px; margin-right: 8px;">●</span>
                <span style="color: #2c3e50;">≥ 0.80 - Moderate ({moderate_count} sites)</span>
            </div>
        </div>
        
        <div style="border-top: 1px solid #dee2e6; padding-top: 8px; font-size: 11px; color: #6c757d;">
            <div style="margin: 2px 0;">📍 Total Sites: {total_sites}</div>
            <div style="margin: 2px 0;">🖼️ Images Matched: {matched_images}/{total_sites}</div>
            <div style="margin: 2px 0;">🤖 Model: GPT-4.1 Vision</div>
            <div style="margin: 6px 0 0 0; font-size: 10px; color: #adb5bd;">
                Click markers for satellite images and details
            </div>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def display_summary_statistics(coordinates):
    """Display summary statistics about the discovered sites."""
    if not coordinates:
        print("No sites found with rating ≥ 0.8")
        return
    
    ratings = [coord[3] for coord in coordinates]
    
    print(f"🎯 ARCHAEOLOGICAL SITE DISCOVERY SUMMARY")
    print(f"{'='*50}")
    print(f"Total high-probability sites found: {len(coordinates)}")
    print(f"Average AI confidence rating: {sum(ratings)/len(ratings):.3f}")
    print(f"Highest confidence rating: {max(ratings):.3f}")
    print(f"Lowest confidence rating: {min(ratings):.3f}")
    print()
    
    # Rating distribution
    very_high = sum(1 for r in ratings if r >= 0.95)
    high = sum(1 for r in ratings if 0.9 <= r < 0.95)
    moderate = sum(1 for r in ratings if 0.8 <= r < 0.9)
    
    print(f"Rating Distribution:")
    print(f"  Very High (≥0.95): {very_high} sites")
    print(f"  High (0.90-0.94):  {high} sites")
    print(f"  Moderate (0.80-0.89): {moderate} sites")
    print()
    
    # Show top 5 sites
    sorted_coords = sorted(coordinates, key=lambda x: x[3], reverse=True)
    print(f"🏆 TOP 5 MOST PROMISING SITES:")
    print(f"{'='*50}")
    for i, (lat, lon, description, rating, filename) in enumerate(sorted_coords[:5], 1):
        print(f"{i}. Rating: {rating:.3f} | ({lat:.6f}, {lon:.6f})")
        print(f"   {description[:80]}{'...' if len(description) > 80 else ''}")
        print()

def display_individual_images(coordinates, image_directory="/kaggle/input/TerraBrasilis-tiles-2023/TerraBrasilis", max_display=5):
    """Display individual images for the top-rated sites."""
    if not coordinates:
        print("No coordinates provided.")
        return
    
    # Sort by rating (highest first)
    sorted_coords = sorted(coordinates, key=lambda x: x[3], reverse=True)
    
    print(f"📸 DISPLAYING TOP {min(max_display, len(sorted_coords))} SITES")
    print(f"{'='*60}")
    
    for i, (lat, lon, description, rating, filename) in enumerate(sorted_coords[:max_display], 1):
        image_path = os.path.join(image_directory, filename)
        
        print(f"\n🎯 SITE #{i} - RATING: {rating:.3f}")
        print(f"Coordinates: {lat:.6f}, {lon:.6f}")
        print(f"Filename: {filename}")
        print(f"Description: {description}")
        print("-" * 40)
        
        if os.path.exists(image_path):
            try:
                img = PILImage.open(image_path)
                plt.figure(figsize=(10, 10))
                plt.imshow(img)
                plt.title(f"Site #{i}: {filename}\nRating: {rating:.3f} | Lat: {lat:.6f}, Lon: {lon:.6f}", 
                         fontsize=12, pad=20)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"❌ Error displaying image: {e}")
        else:
            print(f"❌ Image file not found: {image_path}")
        
        print()

# Execute the visualization functions
print("🗺️ Creating interactive map and visualizations...")
print("This may take a moment to process all the high-probability sites...")

# Display summary statistics first
display_summary_statistics(coordinates)

# Create and display the interactive map
interactive_map = create_interactive_map(coordinates)
if interactive_map:
    print("🗺️ Displaying interactive map below...")
    display(interactive_map)
else:
    print("❌ Failed to create interactive map")

# Display top 5 most promising sites with images
print(f"\n" + "="*60)
print("🏆 TOP 5 MOST PROMISING ARCHAEOLOGICAL SITES")
print("="*60)

def display_top_sites_with_images(coordinates, kaggle_image_path="/kaggle/input/top-candidates", max_display=5):
    """
    Display the top-rated sites with their images inline
    """
    if not coordinates:
        print("No coordinates provided.")
        return
    
    # Sort by rating (highest first)
    sorted_coords = sorted(coordinates, key=lambda x: x[3], reverse=True)
    
    # Create image coordinate mapping for matching
    kaggle_image_mapping = create_image_coordinate_mapping(kaggle_image_path)
    result_to_kaggle_image = match_results_to_images(sorted_coords, kaggle_image_mapping)
    
    for i, (lat, lon, description, rating, filename) in enumerate(sorted_coords[:max_display], 1):
        print(f"\n🎯 SITE #{i} - AI CONFIDENCE: {rating:.3f}/1.00")
        print(f"📍 Coordinates: {lat:.6f}, {lon:.6f}")
        print(f"📄 Original File: {filename}")
        print(f"📝 Description: {description}")
        print("-" * 50)
        
        # Try to find and display the matched Kaggle image
        image_displayed = False
        
        # Check if we have a matched Kaggle image
        if i-1 in result_to_kaggle_image:
            kaggle_image_filename = result_to_kaggle_image[i-1]
            kaggle_image_path_full = os.path.join(kaggle_image_path, kaggle_image_filename)
            
            if os.path.exists(kaggle_image_path_full):
                try:
                    img = PILImage.open(kaggle_image_path_full)
                    plt.figure(figsize=(12, 8))
                    plt.imshow(img)
                    plt.title(f"🏛️ Site #{i}: {kaggle_image_filename}\n"
                             f"AI Confidence: {rating:.3f} | Coordinates: {lat:.6f}, {lon:.6f}\n"
                             f"Status: Best Candidate Image from Kaggle Dataset", 
                             fontsize=14, pad=20, fontweight='bold')
                    plt.axis('off')
                    
                    # Add a colored border based on confidence level
                    if rating >= 0.95:
                        plt.gca().add_patch(plt.Rectangle((0, 0), img.width, img.height, 
                                                        fill=False, edgecolor='red', linewidth=8))
                    elif rating >= 0.9:
                        plt.gca().add_patch(plt.Rectangle((0, 0), img.width, img.height, 
                                                        fill=False, edgecolor='orange', linewidth=8))
                    else:
                        plt.gca().add_patch(plt.Rectangle((0, 0), img.width, img.height, 
                                                        fill=False, edgecolor='green', linewidth=8))
                    
                    plt.tight_layout()
                    plt.show()
                    image_displayed = True
                    print(f"✅ Displayed Kaggle image: {kaggle_image_filename}")
                except Exception as e:
                    print(f"❌ Error loading Kaggle image: {e}")
        
        # Fallback to original image if no Kaggle image was found/displayed
        if not image_displayed:
            fallback_image_path = os.path.join("/kaggle/input/TerraBrasilis-tiles-2023/TerraBrasilis", filename)
            if os.path.exists(fallback_image_path):
                try:
                    img = PILImage.open(fallback_image_path)
                    plt.figure(figsize=(12, 8))
                    plt.imshow(img)
                    plt.title(f"🏛️ Site #{i}: {filename}\n"
                             f"AI Confidence: {rating:.3f} | Coordinates: {lat:.6f}, {lon:.6f}\n"
                             f"Status: Original Dataset Image", 
                             fontsize=14, pad=20, fontweight='bold')
                    plt.axis('off')
                    
                    # Add a colored border based on confidence level
                    if rating >= 0.95:
                        plt.gca().add_patch(plt.Rectangle((0, 0), img.width, img.height, 
                                                        fill=False, edgecolor='red', linewidth=8))
                    elif rating >= 0.9:
                        plt.gca().add_patch(plt.Rectangle((0, 0), img.width, img.height, 
                                                        fill=False, edgecolor='orange', linewidth=8))
                    else:
                        plt.gca().add_patch(plt.Rectangle((0, 0), img.width, img.height, 
                                                        fill=False, edgecolor='green', linewidth=8))
                    
                    plt.tight_layout()
                    plt.show()
                    print(f"✅ Displayed original image: {filename}")
                except Exception as e:
                    print(f"❌ Error loading original image: {e}")
            else:
                print(f"❌ No image found for this site")
        
        print()

# Display the top 5 sites with images
display_top_sites_with_images(coordinates)

print("🎯 Analysis complete! The most promising sites have been displayed above.")
print("💡 Sites are ranked by AI confidence level and show the best available images.")
print("🔴 Red border = Highest confidence (≥0.95) | 🟠 Orange = High (≥0.90) | 🟢 Green = Moderate (≥0.80)")

