!pip install osmnx geopandas shapely -q
!pip install rasterio -q
!pip install elevation -q
!pip install rtree -q
!sudo apt-get install osmium-tool -q


import ee
import geemap
import ipywidgets as widgets
from IPython.display import display, clear_output
import re
import csv

# =========================================================================================
# 0. Helper Functions & Site Data Loading
# =========================================================================================
def parse_source_date_py(source_string, site_code_for_debug="N/A"):
    if not source_string or not isinstance(source_string, str): return None
    date_part = source_string.replace('GE ', '').strip().replace('.', '-').replace('/', '-')
    try:
        parts = date_part.split('-')
        if len(parts) == 3:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            if not (1000 <= year <= 3000 and 1 <= month <= 12 and 1 <= day <= 31): return None
            date_part = f"{year:04d}-{month:02d}-{day:02d}"
        else: return None
    except ValueError: return None
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_part): return None
    try:
        if not ee.data._initialized:
            print("Warning (parse_source_date_py): GEE not initialized. Cannot parse date string for GEE.")
            return None
        return ee.Date.parse('yyyy-MM-dd', date_part)
    except Exception: return None

def load_sites_from_csv(csv_filepath):
    sites_list = []
    try:
        with open(csv_filepath, mode='r', newline='', encoding='utf-8-sig') as infile:
            reader = csv.reader(infile)
            header_row_index, raw_headers_from_file = -1, []
            for i, row in enumerate(reader):
                if not row: continue
                temp_headers = [h.strip().lower() for h in row]
                # MODIFICATION START: Only require 'lat' and 'lon' in headers
                if 'lat' in temp_headers and 'lon' in temp_headers:
                    raw_headers_from_file, header_row_index = [h.strip() for h in row], i
                    break
            if header_row_index == -1:
                print(f"ERROR (load_sites_from_csv): No valid header (lat, lon) in {csv_filepath}.") # Updated error message
                return []
            # MODIFICATION END

            actual_headers_for_dict_keys = [h.lower() for h in raw_headers_from_file if h.strip()]
            data_offset = 1 if raw_headers_from_file and (raw_headers_from_file[0] == '' or raw_headers_from_file[0].isdigit()) else 0
            if data_offset: actual_headers_for_dict_keys = [h.lower() for h in raw_headers_from_file[1:] if h.strip()]
            if not actual_headers_for_dict_keys:
                 print(f"ERROR (load_sites_from_csv): No valid header columns after processing {csv_filepath}.")
                 return []
            infile.seek(0)
            for _ in range(header_row_index + 1): next(reader)
            site_counter = 0 # To generate codes if missing
            for i, row in enumerate(reader):
                if not any(field.strip() for field in row): continue
                data_values = row[data_offset:]
                if len(data_values) < len(actual_headers_for_dict_keys): data_values.extend([None] * (len(actual_headers_for_dict_keys) - len(data_values)))
                elif len(data_values) > len(actual_headers_for_dict_keys): data_values = data_values[:len(actual_headers_for_dict_keys)]
                site_dict = dict(zip(actual_headers_for_dict_keys, data_values))

                # MODIFICATION START: Ensure 'code' is always present, even if generated
                site_code_val = site_dict.get('code')
                if site_code_val is None or str(site_code_val).strip() == '':
                    site_code_val = f"SITE_{site_counter:05d}"
                    site_counter += 1
                # MODIFICATION END

                final_site = {
                    'code': site_code_val, # Use the extracted or generated code
                    'place': site_dict.get('place'),
                    'lat': None, 'lon': None, 'elev': None,
                    'source': site_dict.get('source'),
                    'a_width': None, 'b_width': None, 'form': site_dict.get('form')
                }
                try:
                    if site_dict.get('lat') and str(site_dict.get('lat')).strip(): final_site['lat'] = float(site_dict.get('lat'))
                    if site_dict.get('lon') and str(site_dict.get('lon')).strip(): final_site['lon'] = float(site_dict.get('lon'))                   
                    
                except (ValueError, TypeError): print(f"Warning (load_sites_from_csv): Value parsing error for site {site_code_val}.") # Use generated code for warning
                if final_site['lat'] is not None and final_site['lon'] is not None: sites_list.append(final_site)
                else: print(f"Info (load_sites_from_csv): Site {final_site.get('code', 'N/A')} skipped (missing lat/lon).")
    except FileNotFoundError: print(f"ERROR (load_sites_from_csv): File not found: {csv_filepath}"); return []
    except Exception as e: print(f"ERROR (load_sites_from_csv): Reading {csv_filepath}: {e}"); return []
    return sites_list

# =========================================================================================
# 1. Known Archaeological Sites Data - LOADED FROM CSV (Keep as is, but add check for empty FC later)
# =========================================================================================
CSV_FILENAME = '/kaggle/input/amazon-geoglyphs/all_geoglyphs.csv' # ADJUSTED TO ALL_GEOGLYPHS.CSV
sites_data_array_py = load_sites_from_csv(CSV_FILENAME)
if not sites_data_array_py: print(f"CRITICAL ERROR: No sites loaded from {CSV_FILENAME}.")
else: print(f"Successfully loaded {len(sites_data_array_py)} sites from {CSV_FILENAME}")


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from shapely.geometry import Point
from shapely.geometry import box
import geopandas as gpd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV # Import GridSearchCV if using for bandwidth

def filter_low_density_sites_kde(sites_data, density_threshold_percentile=10):
    """
    Filters sites based on local density estimated using Kernel Density Estimation (KDE),
    plots before and after maps. Sites in areas with density *below*
    this percentile will be kept (low density areas).

    Args:
        sites_data (list): A list of dictionaries, where each dictionary
                           represents a site with 'lat' and 'lon' keys.
        density_threshold_percentile (int): The percentile value to use as the
                                            site density threshold for filtering.
                                            Sites in areas with density *below*
                                            this percentile will be kept.
    Returns:
        pandas.DataFrame: DataFrame containing the sites that remain after filtering,
                          including original columns.
    """
    if not sites_data:
        print("No site data provided.")
        return pd.DataFrame()

    print(f"Total sites loaded: {len(sites_data)}")

    # Convert sites data to a pandas DataFrame for easier processing
    # IMPORTANT CHANGE: Do not filter columns here. Keep all original columns.
    df = pd.DataFrame(sites_data)

    # Drop rows with missing lat/lon as these cannot be used for spatial density
    df_valid_coords = df.dropna(subset=['lat', 'lon']).copy()

    if df_valid_coords.empty:
        print("No sites with valid lat/lon coordinates after dropping NAs.")
        # If no sites remain after dropping NAs, return an empty DataFrame with original columns
        return df.head(0) # Return an empty DataFrame with the original columns

    print(f"Sites with valid lat/lon coordinates: {len(df_valid_coords)}")

    # Plot original sites (using only those with valid coords for plotting)
    # plt.figure(figsize=(10, 8))
    # plt.scatter(df_valid_coords['lon'], df_valid_coords['lat'], color='blue', s=10, label='Original Sites (with valid coords)', alpha=0.5)
    # plt.title('Original Site Locations (with valid coordinates)')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # --- Estimate density using KDE ---
    X = df_valid_coords[['lon', 'lat']].values

    # Using a fixed bandwidth for simplicity and speed
    bandwidth = 0.05 # Example bandwidth in degrees, adjust as needed
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(X)

    # Get density estimate for each site location with valid coordinates
    site_density_scores = np.exp(kde.score_samples(X)) # score_samples returns log-likelihood

    # Add density scores back to the DataFrame containing only sites with valid coordinates
    df_valid_coords['density_score'] = site_density_scores

    # Calculate the density threshold value based on the specified percentile
    if df_valid_coords['density_score'].empty:
         density_threshold_value = 0
         print("No density scores calculated. Cannot filter.")
    else:
        density_threshold_value = np.percentile(df_valid_coords['density_score'], density_threshold_percentile)
        print(f"\nCalculated {density_threshold_percentile}th percentile density threshold (KDE score): {density_threshold_value:.6f}")


    # --- Filter sites ---
    # Keep sites from the valid coordinates DataFrame where density is *below* the calculated threshold percentile
    filtered_df_sites = df_valid_coords[df_valid_coords['density_score'] <= density_threshold_value].copy()
    # Get sites
    print(f"\nSites remaining after filtering (KDE density <= {density_threshold_value:.6f}): {len(filtered_df_sites)}")
    print(f"Number of sites filtered out: {len(df_valid_coords) - len(filtered_df_sites)}")


    # --- Plot remaining sites ---
    plt.figure(figsize=(10, 8))
    plt.scatter(filtered_df_sites['lon'], filtered_df_sites['lat'], color='green', s=10, label='Remaining Sites (Low Density)', alpha=0.6)
    plt.title('Site Locations After Filtering by Low Density (KDE)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    # The returned DataFrame includes all original columns plus 'density_score' for filtered sites
    return filtered_df_sites

# Example usage with the loaded data:
# Keep the original sites_data_array_py loaded from CSV
# sites_data_array_py = load_sites_from_csv(CSV_FILENAME) # Assuming this was done earlier

remaining_sites_kde = filter_low_density_sites_kde(sites_data_array_py, density_threshold_percentile=99)

# drop_

# Assuming remaining_sites_kde is a pandas DataFrame as returned by filter_low_density_sites_kde
if remaining_sites_kde is not None and not remaining_sites_kde.empty:
  # Convert back to list of dictionaries if needed for subsequent steps
  # Select the desired columns to include in the output list of dictionaries
  # This is where we specify which columns from the filtered DataFrame to keep
  # Ensure these columns exist in the DataFrame returned by filter_low_density_sites_kde
  output_columns = ['code', 'place', 'lat', 'lon', 'elev', 'source', 'a_width', 'b_width', 'form']
  # Filter the remaining_sites_kde DataFrame to only include columns that are actually present
  present_output_columns = [col for col in output_columns if col in remaining_sites_kde.columns]

  remaining_sites_data_array_py_filtered = remaining_sites_kde[present_output_columns].to_dict('records')
  print(f"\nConverted remaining_sites_kde DataFrame to list of dictionaries: {len(remaining_sites_data_array_py_filtered)}")
  # Optional: Print the first few entries to verify
  # print(remaining_sites_data_array_py_filtered[:5])
else:
  remaining_sites_data_array_py_filtered = []
  print("\nremaining_sites_kde DataFrame is empty or None. remaining_sites_data_array_py_filtered is an empty list.")



!wget https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_sa.gdb.zip -q
!unzip -o /kaggle/working/HydroRIVERS_v10_sa.gdb.zip 


import geopandas as gpd
import pyproj
from shapely.ops import unary_union
from tqdm import tqdm
import os

# Download from: https://www.hydrosheds.org/products/hydrorivers
# Get "RiverATLAS_v10_sa.gdb" (South America subset)
HYDRO_RIVERS_PATH = '/kaggle/working/HydroRIVERS_v10_sa.gdb'

# Load rivers - this contains discharge data (DIS_AV_CMS)
rivers = gpd.read_file(HYDRO_RIVERS_PATH)

# Filter for Amazon basin countries and minimum discharge
riv_ord_list= [1,2,3,4]
amazon_rivers = rivers[
    (rivers['ORD_FLOW'].isin(riv_ord_list))
].copy()

# Use South America Albers Equal Area Conic (EPSG:102033)
project_crs = 'EPSG:5641'
buffer_distance = 10000  # 10km in meters

# Project and buffer
amazon_rivers_proj = amazon_rivers.to_crs(project_crs)
amazon_rivers_proj['geometry'] = amazon_rivers_proj.buffer(buffer_distance)

# Merge all buffers
merged_buffer = unary_union(amazon_rivers_proj.geometry)

# Simplify (tolerance in meters)
simplified_buffer = merged_buffer.simplify(500)  # 500m tolerance

# Convert back to GeoDataFrame
final_gdf = gpd.GeoDataFrame(geometry=[simplified_buffer], crs=project_crs).to_crs('EPSG:4326')

import pandas as pd
# Define the Amazon bounding box (lon_min, lat_min, lon_max, lat_max)
AMAZON_BBOX = (-80, -25, -45, 10)

# Create a shapely box from the bounding box coordinates
amazon_box = box(*AMAZON_BBOX)

# Check if final_gdf is not empty and has a geometry column
if not final_gdf.empty and 'geometry' in final_gdf.columns:
    # Clip the geometry of final_gdf to the Amazon bounding box
    # Note: .clip() is a method of GeoDataFrames and Series
    final_gdf_clipped = final_gdf.clip(amazon_box)
    print(f"Geometry clipped to Amazon BBOX: {AMAZON_BBOX}")
else:
    print("final_gdf is empty or does not have a 'geometry' column. Cannot clip.")
    final_gdf_clipped = gpd.GeoDataFrame(columns=final_gdf.columns) # Create an empty GeoDataFrame with the same columns


import pandas as pd
# Assuming final_gdf is a GeoDataFrame with a 'geometry' column containing polygons or multipolygons
# and is already loaded and processed as in the preceding code.

import matplotlib.pyplot as plt
import geopandas as gpd

final_gdf=final_gdf_clipped

# Check if final_gdf is not empty
if not final_gdf.empty:
    # Define the extent for a rough South America view (adjust as needed)
    # This is a simpler approach than loading country boundaries just for plotting extent
    # You could refine this by loading a South America shapefile if more precision is needed
    south_america_extent = {
        'minx': -85,
        'maxx': -30,
        'miny': -30,
        'maxy': 15
    }

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot the final_gdf GeoDataFrame
    # Use a specific color and alpha for the buffer zone
    final_gdf.plot(ax=ax, color='blue', alpha=0.5, label='River Buffer Zone')

    # Optionally, if you want to plot the remaining sites on the same map
    # Assuming remaining_sites_kde is available and is a GeoDataFrame or convertible
    if 'remaining_sites_kde' in locals() and not remaining_sites_kde.empty:
        # Convert pandas DataFrame to GeoDataFrame if it's not already
        if not isinstance(remaining_sites_kde, gpd.GeoDataFrame):
            # Assuming 'lon' and 'lat' columns exist
            gdf_sites = gpd.GeoDataFrame(
                remaining_sites_kde, geometry=gpd.points_from_xy(remaining_sites_kde['lon'], remaining_sites_kde['lat']), crs="EPSG:4326"
            )
        else:
            gdf_sites = remaining_sites_kde

        # Plot the sites on the same axes
        gdf_sites.plot(ax=ax, marker='o', color='red', markersize=10, label='Remaining Sites', alpha=0.7)


    # Set title and labels
    ax.set_title('River Buffer Zone and Filtered Sites on South America Map')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Set plot limits to the approximate South America extent
    ax.set_xlim(south_america_extent['minx'], south_america_extent['maxx'])
    ax.set_ylim(south_america_extent['miny'], south_america_extent['maxy'])

    # Add a legend
    ax.legend()

    # Add a grid
    ax.grid(True)

    # Show the plot
    plt.show()
else:
    print("final_gdf is empty. Cannot plot on a map.")



import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box
from tqdm import tqdm
import json
import math

# Add this to the top
from rtree import index

# Modify generate_grid_cells function
def generate_grid_cells_rtree(multipolygon, cell_size=0.01):
    minx, miny, maxx, maxy = multipolygon.bounds
    width_deg = maxx - minx
    height_deg = maxy - miny

    # Create spatial index
    idx = index.Index()
    for i, cell in enumerate(multipolygon.geoms if hasattr(multipolygon, 'geoms') else [multipolygon]):
        idx.insert(i, cell.bounds)

    # Generate grid
    valid_cells = []
    x_coords = np.arange(minx, maxx, cell_size)
    y_coords = np.arange(miny, maxy, cell_size)
    total_cells = len(x_coords) * len(y_coords)

    with tqdm(total=total_cells, desc="Processing grid cells") as pbar:
        for x in x_coords:
            for y in y_coords:
                cell = box(x, y, x+cell_size, y+cell_size)

                # Fast R-tree intersection check
                if not list(idx.intersection(cell.bounds)):
                    pbar.update(1)
                    continue

                # Precise containment check
                if multipolygon.contains(cell):
                    valid_cells.append({
                        "minx": x, "miny": y,
                        "maxx": x+cell_size, "maxy": y+cell_size
                    })
                pbar.update(1)

    return valid_cells

def save_grid_cells(grid_cells, output_file):
    """Save grid cells to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(grid_cells, f)
    print(f"Saved {len(grid_cells)} grid cells to {output_file}")


gdf = final_gdf

# Extract multipolygon (assuming first feature is the main one)
multipolygon = gdf.geometry.iloc[0]

# Generate grid cells
grid_cells = generate_grid_cells_rtree(multipolygon)

# Save results
output_file = "/kaggle/working/grid_cells.json"
save_grid_cells(grid_cells, output_file)

print(f"Processed {len(grid_cells)} grid cells within river buffers")


# Load the filtered grid cells data
grid_cells_no_sites = grid_cells

# Load the original remaining sites data (the ones we kept after density filtering)
# remaining_sites_data_array_py_filtered is assumed to be already loaded

if not remaining_sites_data_array_py_filtered:
    print("Warning: remaining_sites_data_array_py_filtered is empty. Cannot calculate distances.")

# Create a list of shapely Point objects for the remaining sites
site_points = []
if remaining_sites_data_array_py_filtered:
    for site in remaining_sites_data_array_py_filtered:
        try:
            site_points.append(Point(float(site['lon']), float(site['lat'])))
        except (ValueError, TypeError):
            print(f"Warning: Skipping site {site.get('code', 'N/A')} due to invalid lat/lon.")

# Create a spatial index for the site points
site_point_idx = index.Index()
if site_points:
    for i, point in enumerate(site_points):
        site_point_idx.insert(i, point.bounds)

# Define the distance threshold in degrees (approximate)
# Need to convert km to degrees. This is highly approximate and varies by latitude.
# A rough conversion: 1 degree lat ~ 111 km. 1 degree lon ~ 111*cos(lat) km.
# Since our area is near the equator, we can roughly assume 1 degree ~ 111km.
# So, 50 km is approximately 50 / 111 degrees.
distance_threshold_km = 50
# Using a central latitude for a slightly better approximation
central_lat = -5 # Assuming the center of the Amazon basin is around 5 S
degrees_per_km_lat = 1 / 111.32 # At equator
degrees_per_km_lon = 1 / (111.32 * math.cos(math.radians(central_lat)))
# Let's use the larger of the two to be safe (approximate the diagonal of a square cell)
approx_degrees_per_km = max(degrees_per_km_lat, degrees_per_km_lon) # Using the lat factor is simple and works for distance
distance_threshold_degrees = distance_threshold_km * approx_degrees_per_km

print(f"\nUsing distance threshold: {distance_threshold_km} km â‰ˆ {distance_threshold_degrees:.6f} degrees")


# Function to check if a grid cell is further than the threshold from ALL sites
def is_far_from_all_sites(grid_cell_bounds, site_point_index, sites_points_list, threshold_degrees):
    """
    Checks if a grid cell (defined by its bounds) is further than the threshold
    from *all* sites using a spatial index for efficient querying.
    Returns True if the grid cell is far from ALL sites, False otherwise.
    """
    minx, miny, maxx, maxy = grid_cell_bounds
    cell_center_x = (minx + maxx) / 2.0
    cell_center_y = (miny + miny) / 2.0 # Use miny for consistent approach, or average of the two. Let's use the center.
    cell_center_y = (miny + maxy) / 2.0
    cell_center_point = Point(cell_center_x, cell_center_y)

    # Create a buffer around the grid cell center
    buffer_around_cell = cell_center_point.buffer(threshold_degrees)

    # Use the spatial index to find potential site intersections with the buffer
    # This is slightly optimized as it checks the buffer's bounds against site bounds
    # We need to expand the grid cell bounds by the threshold to query the index
    buffered_bounds = (minx - threshold_degrees, miny - threshold_degrees,
                       maxx + threshold_degrees, maxy + threshold_degrees)
    potential_site_indices = list(site_point_index.intersection(buffered_bounds))


    if not potential_site_indices:
        # No sites within the expanded bounds, so the cell is likely far from all sites.
        # Perform a more rigorous check if necessary, but R-tree intersection is good.
        # If the R-tree intersection is empty for the buffered bounds,
        # it means no site bounding box overlaps with the buffer's bounding box.
        # This strongly implies the cell center is further than the threshold from all sites.
        return True

    # Perform precise distance check only for potential sites
    for site_index in potential_site_indices:
        site_point = sites_points_list[site_index]
        # Check if the buffer around the cell center contains the site point
        if buffer_around_cell.contains(site_point):
             return False # Found a site within the threshold distance

    # If the loop completes without finding any site within the threshold distance,
    # it means the grid cell is far from all sites.
    return True


# Filter out grid cells that are far from all sites
interesting_grid_cells = []
total_grid_cells_no_sites = len(grid_cells_no_sites)

print(f"\nStarting second filtering of {total_grid_cells_no_sites} grid cells (by distance from sites)...")

with tqdm(total=total_grid_cells_no_sites, desc="Filtering grid cells by distance") as pbar:
    for cell_data in grid_cells_no_sites:
        cell_bounds = (cell_data['minx'], cell_data['miny'], cell_data['maxx'], cell_data['maxy'])
        # Keep the grid cell if it is *NOT* far from all sites
        if not is_far_from_all_sites(cell_bounds, site_point_idx, site_points, distance_threshold_degrees):
             interesting_grid_cells.append(cell_data)
        pbar.update(1)

print(f"\nFinished second filtering.")
print(f"Initial number of grid cells (no sites): {total_grid_cells_no_sites}")
print(f"Number of interesting grid cells remaining: {len(interesting_grid_cells)}")

# Save the interesting grid cells to a new JSON file
output_interesting_grid_file = "/kaggle/working/interesting_grid_cells.json"
save_grid_cells(interesting_grid_cells, output_interesting_grid_file)


import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

# Create a list of Shapely Polygon objects for the grid cells
grid_polygons = [box(cell['minx'], cell['miny'], cell['maxx'], cell['maxy']) for cell in tqdm(interesting_grid_cells)]

# Create a list of Shapely Point objects for the site centers
site_points = [Point(site['lon'], site['lat']) for site in tqdm(remaining_sites_data_array_py_filtered) if site.get('lat') is not None and site.get('lon') is not None]

# Extract grid cell centers
grid_centers = [cell.centroid for cell in grid_polygons]
grid_center_x = [point.x for point in grid_centers]
grid_center_y = [point.y for point in grid_centers]

# Extract site coordinates
site_x = [point.x for point in site_points]
site_y = [point.y for point in site_points]

# Plotting
plt.figure(figsize=(12, 10))

# Plot grid cell centers
plt.scatter(grid_center_x, grid_center_y, color='blue', s=1, label='Grid Cell Centers', alpha=0.5)

# Plot site locations
plt.scatter(site_x, site_y, color='red', s=10, label='Remaining Sites (Low Density)', alpha=0.8)

plt.title('Grid Cell Centers and Remaining Sites on a Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()




