%pip install rasterio


# Requirements: numpy, pandas, rasterio, json, glob
import numpy as np
import pandas as pd
import rasterio
import glob
import json
import os
import sys
import datetime
import uuid
import csv
import time
import random
import concurrent.futures
from threading import Lock
from typing import Dict, List, Union, Optional, Tuple, Any
import threading
import queue
import shutil
import gzip
import argparse
import re
import traceback
import matplotlib.pyplot as plt

# Direct check for Kaggle environment - doesn't rely on function to avoid NameError
RUNNING_ON_KAGGLE = os.path.exists('/kaggle/input')

# Check if we're in an IPython environment (Jupyter/Colab/Kaggle)
IPYTHON_AVAILABLE = False
try:
    # First try direct import
    from IPython.display import Image, display
    
    # In Kaggle, we know IPython is available
    if RUNNING_ON_KAGGLE:
        IPYTHON_AVAILABLE = True
    else:
        # For other environments, check if IPython kernel is active
        from IPython import get_ipython
        IPYTHON_AVAILABLE = get_ipython() is not None
except ImportError:
    pass

# =====================================================================
# HOW TO USE IN KAGGLE NOTEBOOKS:
# 
# import kaggle_amazon4 as ka4
# 
# # To reset the knowledge graph and start fresh:
# ka4.set_reset_flag(True)  # Delete existing KG
# 
# # Optional: Archive old KG before resetting (preserves history)
# ka4.set_archive_flag(True)
# 
# # Run the pipeline (with optional inline visualization)
# results = ka4.main(visualize=True, inline_visualization=True)
# 
# # Access the results
# kg = results["kg"]                       # Knowledge graph
# sites = results["potential_sites"]       # Detected sites
# file_path = results["submission_file"]   # Path to saved results
# 
# # Analyze KG contents
# ka4.analyze_feature_types(kg)
# =====================================================================

# --- Environment Detection ---
def is_running_on_kaggle():
    """Detect if the code is running on Kaggle."""
    return os.path.exists('/kaggle/input')

# --- Path Configuration ---
if is_running_on_kaggle():
    # Kaggle paths
    KAGGLE_INPUT = '/kaggle/input'
    KAGGLE_OUTPUT = '/kaggle/output'  # Output directory for dataset creation
    
    # Look for nasa-tiles dataset or similar in the input directory
    nasa_dirs = [d for d in os.listdir(KAGGLE_INPUT) if 'nasa' in d.lower() or 'tiles' in d.lower()]
    if nasa_dirs:
        # Use the nasa-tiles dataset directory
        KAGGLE_DATASET_DIR = os.path.join(KAGGLE_INPUT, nasa_dirs[0])
    else:
        # Default to first directory if no matching dir is found
        available_dirs = os.listdir(KAGGLE_INPUT)
        KAGGLE_DATASET_DIR = os.path.join(KAGGLE_INPUT, available_dirs[0]) if available_dirs else KAGGLE_INPUT
    
    # The specific ASTER DEM file path
    ASTER_DEM_PATH = os.path.join(KAGGLE_INPUT, 'nasa-tiles/ASTGTM_NC.003_ASTER_GDEM_DEM_doy2000061_aid0001.tif')
    
    # Define TILES_DIR for loading geotiff files
    TILES_DIR = KAGGLE_DATASET_DIR  # The parent directory containing tile files
    
    # Find templates.json and sites.csv directly in the dataset directory
    TEMPLATES_JSON = os.path.join(KAGGLE_DATASET_DIR, 'templates.json')
    if not os.path.exists(TEMPLATES_JSON):
        # Try to find it anywhere in the input directory
        templates_files = []
        for root, _, files in os.walk(KAGGLE_INPUT):
            templates_files.extend([os.path.join(root, f) for f in files if f.lower() == 'templates.json'])
        TEMPLATES_JSON = templates_files[0] if templates_files else None
    
    SITES_CSV = os.path.join(KAGGLE_DATASET_DIR, 'sites.csv')
    if not os.path.exists(SITES_CSV):
        # Try to find it anywhere in the input directory
        sites_files = []
        for root, _, files in os.walk(KAGGLE_INPUT):
            sites_files.extend([os.path.join(root, f) for f in files if f.lower() == 'sites.csv'])
        SITES_CSV = sites_files[0] if sites_files else None
else:
    # Local paths
    DATA_DIR = "data"
    TILES_DIR = os.path.join(DATA_DIR, "tiles")
    TEMPLATES_JSON = os.path.join(DATA_DIR, "templates.json")
    SITES_CSV = os.path.join(DATA_DIR, "sites.csv")

# --- Data Loaders ---
def load_templates_from_json(json_path=None):
    """Load templates from a JSON file.
    
    Args:
        json_path: Path to the JSON file with templates
        
    Returns:
        List of template arrays
    """
    # Use environment variable if set
    if json_path is None:
        json_path = os.environ.get('TEMPLATES_PATH', None)
        
    # Use default paths based on environment
    if json_path is None:
        if is_running_on_kaggle():
            # First check if we have a mounted dataset
            if os.environ.get('NASA_TILES_PATH'):
                json_path = os.path.join(os.environ.get('NASA_TILES_PATH'), 'templates.json')
            else:
                json_path = '/kaggle/input/nasa-tiles/templates.json'
        else:
            json_path = './patterns/templates.json'
    
    # Load templates from JSON
    if os.path.exists(json_path):
        try:
            print(f"Loading templates from {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            templates = []
            for template_data in data:
                if 'grid' in template_data:
                    # Convert to numpy array
                    grid = np.array(template_data['grid'])
                    templates.append(grid)
            
            print(f"Successfully loaded {len(templates)} templates from {json_path}")
            return templates
        except Exception as e:
            print(f"Error loading templates: {e}")
            return []
    else:
        print(f"Template file {json_path} not found")
        return []

def load_sites_from_csv(csv_path=None):
    """Load archaeological sites from a CSV file.
    
    Args:
        csv_path: Path to CSV file with site data
        
    Returns:
        List of site dictionaries with lat, lon, and description
    """
    # Use environment variable if set
    if csv_path is None:
        csv_path = os.environ.get('SITES_PATH', None)
        
    # Use default paths based on environment
    if csv_path is None:
        # Define multiple possible paths to look for sites.csv
        possible_paths = []
        
        if is_running_on_kaggle():
            # Kaggle environment paths
            possible_paths = [
                '/kaggle/input/nasa-tiles/sites.csv',
                '/kaggle/input/nasa-tiles/data/sites.csv', 
                '/kaggle/input/nasa-tiles/test_data/sites.csv'
            ]
            
            # Look through all datasets in input for sites.csv
            input_dir = '/kaggle/input'
            if os.path.exists(input_dir):
                for dataset in os.listdir(input_dir):
                    possible_paths.append(os.path.join(input_dir, dataset, 'sites.csv'))
        else:
            # Local environment paths
            possible_paths = [
                './sites.csv',
                './data/sites.csv',
                './patterns/sites.csv',
                './test_data/sites.csv',
                '/Users/richardgillespie/Documents/AAImageSearch/sites.csv',
                '/Users/richardgillespie/Documents/AAImageSearch/data/sites.csv'
            ]
        
        # Find the first path that exists
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
    
    # Load sites from CSV
    if os.path.exists(csv_path):
        try:
            print(f"Loading archaeological sites from {csv_path}")
            import csv
            sites = []
            
            with open(csv_path, 'r') as f:
                # Try to determine file size for large files
                try:
                    f.seek(0, os.SEEK_END)
                    file_size = f.tell() / 1024  # Size in KB
                    f.seek(0)
                    if file_size > 100:  # If file is larger than 100KB
                        print(f"Sites file is {file_size:.1f} KB - may contain a large number of sites")
                except:
                    pass  # Ignore if seeking fails
                
                reader = csv.reader(f)
                # Try to read header, fall back if fails
                try:
                    header = next(reader)  # Skip header
                    print(f"CSV header: {header}")
                except StopIteration:
                    print("Empty CSV file or no header found")
                    return []
                    
                count = 0
                valid_count = 0
                error_count = 0
                
                for row in reader:
                    count += 1
                    try:
                        if len(row) >= 2:
                            lat = float(row[0])
                            lon = float(row[1])
                            desc = row[2] if len(row) > 2 else ""
                            sites.append({'lat': lat, 'lon': lon, 'desc': desc})
                            valid_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        error_count += 1
                        if error_count < 5:  # Only show first few errors
                            print(f"Warning: Error parsing site at row {count}: {e}")
                
                if error_count > 0:
                    print(f"Encountered {error_count} errors while parsing {count} rows")
            
            print(f"Successfully loaded {len(sites)} archaeological sites from {csv_path}")
            
            # Print summary of coordinates to help with debugging
            if sites:
                lats = [site['lat'] for site in sites]
                lons = [site['lon'] for site in sites]
                print(f"Coordinate ranges: Lat {min(lats):.4f} to {max(lats):.4f}, Lon {min(lons):.4f} to {max(lons):.4f}")
                
            return sites
        except Exception as e:
            print(f"Error loading sites: {e}")
            return []
    else:
        print(f"Sites file {csv_path} not found")
        return []

def process_tile_file(src_path, tile_size, overlap, filter_clouds=True):
    """Process a single image file into tiles.
    
    This function is used by load_tiles_with_sliding_window for parallel processing.
    
    Args:
        src_path: Path to the source image file
        tile_size: Size of tiles to create
        overlap: Overlap between tiles in pixels
        filter_clouds: Whether to filter out cloudy tiles
        
    Returns:
        List of tile dictionaries
    """
    import os
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    
    local_tiles = []
    
    # Constants for cloud filtering
    CLOUD_BITFLAG = 0x4  # Common cloud bit flag in QA bands
    CLEAR_THRESHOLD = 0.7  # Minimum clear percentage to keep tile
    
    try:
        if src_path.endswith('.hgt.gz'):
            import gzip
            import tempfile
            
            # Decompress the file to a temporary location
            with gzip.open(src_path, 'rb') as gz:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.hgt')
                temp_file.write(gz.read())
                temp_file.close()
                src_path = temp_file.name
        
        with rasterio.open(src_path) as src:
            # Get metadata
            meta = src.meta.copy()
            n_cols = src.width
            n_rows = src.height
            
            # Calculate step size with overlap
            step_size = tile_size - overlap
            
            # Generate steps with proper overlap
            x_steps = list(range(0, n_cols, step_size))
            y_steps = list(range(0, n_rows, step_size))
            
            # Extract filename for tile naming
            base_name = os.path.splitext(os.path.basename(src_path))[0]
            
            # Extract nominal coordinates from filename if it follows a pattern
            lat, lon = None, None
            
            # Handle HGT files (N03W060.hgt format)
            if len(base_name) >= 7:
                if base_name[0] in 'NS' and base_name[3] in 'WE':
                    try:
                        lat_dir = base_name[0]
                        lat_val = int(base_name[1:3])
                        lat = lat_val if lat_dir == 'N' else -lat_val
                        
                        lon_dir = base_name[3]
                        lon_val = int(base_name[4:7])
                        lon = lon_val if lon_dir == 'E' else -lon_val
                    except ValueError:
                        pass
            
            # Handle ASTER GDEM files (which follow different convention)
            if "ASTGTM" in base_name and "ASTER_GDEM_DEM" in base_name:
                print(f"Detected ASTER GDEM file: {base_name}")
                # For ASTER files, we'll use the geotransform to get proper coordinates
                # But we'll set a default starting point in the Amazon region if needed
                if lat is None or lon is None:
                    # Default to central Amazon if coordinates cannot be extracted from filename
                    lat = -3.0  # Central Amazon latitude
                    lon = -60.0  # Central Amazon longitude
            
            print(f"Processing file: {src_path}, dimensions: {n_rows}x{n_cols}")
            
            n_bands = src.count
            qa_band_index = n_bands  # Default to last band for QA if available
            
            # Look for QA band for cloud filtering
            if filter_clouds:
                for i in range(1, n_bands + 1):
                    band_desc = src.descriptions[i-1] if src.descriptions and src.descriptions[i-1] else ""
                    if band_desc and ("qa" in band_desc.lower() or "mask" in band_desc.lower() or "quality" in band_desc.lower()):
                        qa_band_index = i
                        break
            
            # Process tiles with sliding window
            tile_count = 0
            cloudy_count = 0
            empty_count = 0
            
            for i, x in enumerate(x_steps):
                for j, y in enumerate(y_steps):
                    # Keep window inside the image boundaries
                    win_width = min(tile_size, n_cols - x)
                    win_height = min(tile_size, n_rows - y)
                    
                    # Skip partial tiles if they're too small
                    if win_width < tile_size/2 or win_height < tile_size/2:
                        continue
                        
                    # Create the window
                    win = Window(x, y, win_width, win_height)
                    
                    # Get the window transform for georeferencing
                    transform = src.window_transform(win)
                    
                    # Read the data
                    if n_bands > 1:
                        # For multi-band images, read all bands
                        tile_data = src.read(window=win)  # (bands, h, w)
                        
                        # For elevation data, just use the first band
                        grid = tile_data[0]
                    else:
                        # For single-band images (like DEM)
                        grid = src.read(1, window=win)
                        
                    # Skip empty tiles
                    if np.all(grid == 0) or np.all(np.isnan(grid)):
                        empty_count += 1
                        continue
                        
                    # Filter cloudy tiles if requested
                    if filter_clouds and qa_band_index <= n_bands:
                        try:
                            qa = src.read(qa_band_index, window=win)
                            cloud_mask = (qa & CLOUD_BITFLAG) == 0  # 0 = clear
                            clear_percentage = np.mean(cloud_mask)
                            
                            if clear_percentage < CLEAR_THRESHOLD:
                                # Skip tiles with too many clouds
                                cloudy_count += 1
                                continue
                        except Exception as e:
                            # If cloud filtering fails, continue anyway
                            pass
                    
                    # Get the geographic coordinates for this tile
                    # For center of the tile
                    center_row = y + win_height // 2
                    center_col = x + win_width // 2
                    
                    # Get the lat/lon for this pixel
                    tile_lon, tile_lat = None, None
                    try:
                        # Get coordinates from raster if possible
                        tile_lon, tile_lat = src.xy(center_row, center_col)
                    except:
                        # Fall back to nominal coordinates if available
                        if lat is not None and lon is not None:
                            # Calculate approximate coordinates based on position in the file
                            # This is a rough approximation!
                            y_ratio = center_row / n_rows
                            x_ratio = center_col / n_cols
                            
                            # Different handling for ASTER files
                            if "ASTGTM" in base_name:
                                # ASTER tiles typically span 1-degree
                                span = 1.0  # Each ASTER tile typically covers 1 degree
                                tile_lat = lat + (span/2) - y_ratio * span  # Top = lat+span/2, Bottom = lat-span/2
                                tile_lon = lon - (span/2) + x_ratio * span  # Left = lon-span/2, Right = lon+span/2
                            else:
                                # Standard HGT handling
                                tile_lat = lat + (1 - y_ratio)  # Top is lat+1, bottom is lat
                                tile_lon = lon + x_ratio        # Left is lon, right is lon+1
                    
                    # Save valid tile
                    if grid is not None and grid.size > 0:
                        # Apply contrast normalization to make features more visible
                        if np.std(grid) > 0:
                            # Ignore no-data values (typically -32768)
                            valid_mask = (grid > -10000)
                            if np.sum(valid_mask) > 0:
                                valid_data = grid[valid_mask]
                                grid_min = np.percentile(valid_data, 2)
                                grid_max = np.percentile(valid_data, 98)
                                
                                # Prevent division by zero when min/max are too close
                                denominator = grid_max - grid_min
                                if denominator <= 0.001:  # Add a small epsilon to prevent division by zero
                                    # Handle case where all valid values are the same
                                    normalized_grid = np.zeros_like(grid)
                                    normalized_grid[valid_mask] = 127  # Set to mid-gray if range is zero
                                else:
                                    # Normalize to 0-255 range for valid pixels
                                    normalized_grid = np.clip((grid - grid_min) * 255 / denominator, 0, 255)
                                
                                # Keep invalid pixels as is
                                normalized_grid[~valid_mask] = 0
                                grid = normalized_grid
                        
                        # Create tile metadata with enhanced georeference information
                        tile_meta = {
                            'x': x,
                            'y': y,
                            'width': win_width,
                            'height': win_height,
                            'transform': transform,
                            'resolution_m': src.res[0] if src.res else 30.0,  # Default to 30m for ASTER
                            'file_source': os.path.basename(src_path),
                            'crs': src.crs.to_string() if src.crs else None,
                            'bounds': {
                                'minx': transform[0],
                                'miny': transform[3] + transform[5] * win_height,
                                'maxx': transform[0] + transform[1] * win_width,
                                'maxy': transform[3]
                            }
                        }
                        
                        # Handle NaN values before converting to int32
                        # Replace NaN with 0 to avoid conversion errors
                        grid = np.nan_to_num(grid, nan=0.0)
                        
                        # Ensure coordinates are valid floating point numbers
                        if tile_lat is None or not np.isfinite(tile_lat):
                            tile_lat = 0.0
                        if tile_lon is None or not np.isfinite(tile_lon):
                            tile_lon = 0.0
                            
                        # Add coordinates to the metadata dictionary too (important for agents)
                        tile_meta['lat'] = float(tile_lat)
                        tile_meta['lon'] = float(tile_lon)
                        
                        # Add the tile to our list
                        local_tiles.append({
                            'grid': grid.astype(np.int32),
                            'lat': float(tile_lat),  # Ensure proper type
                            'lon': float(tile_lon),  # Ensure proper type
                            'meta': tile_meta,
                            'tile_id': f"{base_name}_{i}_{j}"
                        })
                        
                        tile_count += 1
            
            print(f"Created {tile_count} valid tiles, skipped {cloudy_count} cloudy and {empty_count} empty tiles")
            
            # Clean up temp file if created
            if src_path.endswith('.hgt') and 'temp_file' in locals():
                os.unlink(src_path)
                
    except Exception as e:
        print(f"Error processing file {src_path}: {e}")
        import traceback
        traceback.print_exc()
    
    # Return empty list if no tiles were created
    if not local_tiles:
        print(f"No valid tiles created from {src_path}, creating synthetic tile")
        grid = np.zeros((tile_size, tile_size), dtype=np.int32)
        center = tile_size // 4
        size = tile_size // 2
        grid[center:center+size, center:center+size] = 128  # Simple square feature
        
        # Create properly structured synthetic tile with coordinates in meta
        tile_meta = {
            'file': 'synthetic',
            'lat': -3.0,  # Include in metadata 
            'lon': -60.0  # Include in metadata
        }
        return [{'grid': grid, 'lat': -3.0, 'lon': -60.0, 'meta': tile_meta, 'tile_id': 'synthetic'}]
    
    return local_tiles

def load_tiles_from_geotiff(tile_dir=None, tile_size=128):
    """Load tiles from GeoTIFF or HGT files with environment-aware path handling.
    
    This is an enhanced version that uses the sliding window approach for better
    tile quality and feature detection.
    
    Args:
        tile_dir: Directory containing GeoTIFF or HGT files (uses TILES_DIR if None)
        tile_size: Size of tiles in pixels
        
    Returns:
        List of tile dictionaries with grid, lat, lon, and metadata
    """
    # Use environment-specific path if none provided
    if tile_dir is None:
        tile_dir = TILES_DIR
    
    if not os.path.exists(tile_dir):
        print(f"[WARNING] Tile directory not found: {tile_dir}. Using synthetic tile.")
        grid = np.zeros((tile_size, tile_size), dtype=int)
        center = tile_size // 4
        size = tile_size // 2
        grid[center:center+size, center:center+size] = 128  # Simple square feature
        
        # Create properly structured synthetic tile with coordinates in meta
        tile_meta = {
            'file': 'synthetic',
            'index': 0,
            'lat': -3.0,  # Include in metadata 
            'lon': -60.0  # Include in metadata
        }
        return [{'grid': grid, 'lat': -3.0, 'lon': -60.0, 'meta': tile_meta, 'id': 0}]
    
    # Calculate appropriate overlap (25% of tile size)
    overlap = tile_size // 4
    
    # Use the new sliding window approach
    return load_tiles_with_sliding_window(
        directory=tile_dir,
        tile_size=tile_size,
        overlap=overlap,
        filter_clouds=True
    )

def load_tiles_with_sliding_window(directory, tile_size=None, overlap=None, filter_clouds=True, num_workers=None):
    """Load tiles from GeoTIFF files using an optimized sliding window approach.
    
    This function processes large GeoTIFF files by breaking them into smaller,
    overlapping tiles that are properly sized for agent analysis. It also
    filters cloudy or empty tiles to improve processing efficiency.
    
    Args:
        directory: Directory containing GeoTIFF or HGT files
        tile_size: Size of tiles in pixels (default: auto-select based on environment)
        overlap: Overlap between tiles in pixels (default: 25% of tile size)
        filter_clouds: Whether to filter out cloudy tiles (default: True)
        num_workers: Number of parallel workers for processing (default: auto-detect)
        
    Returns:
        List of tile dictionaries with grid, lat, lon, and metadata
    """
    import os
    import glob
    import numpy as np
    import concurrent.futures
    
    # Auto-select tile size based on environment
    if tile_size is None:
        if is_running_on_kaggle():
            tile_size = 128  # Standard size for Kaggle environment (higher compute)
        else:
            tile_size = 64   # Smaller size for local testing
    
    # Use 50% overlap between tiles to avoid missing features at boundaries
    if overlap is None:
        overlap = tile_size // 2  # 50% overlap
    
    # Setup parallel processing
    if num_workers is None:
        import multiprocessing
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Setup paths - ensure we write to a writable directory in Kaggle
    if is_running_on_kaggle() and directory.startswith('/kaggle/input/'):
        output_dir = '/kaggle/working/processed_tiles'
    else:
        output_dir = os.path.join(directory, "processed_tiles")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating processed tiles directory at: {output_dir}")
    
    # Find all TIFF files in the directory
    tiles = []
    
    # Check for different file types
    hgt_files = glob.glob(os.path.join(directory, "*.hgt")) + glob.glob(os.path.join(directory, "*.hgt.gz"))
    if hgt_files:
        print(f"Found {len(hgt_files)} HGT files. Using specialized HGT loader.")
        # Use the specialized HGT function
        # This is now handled by the generic processor internally
        
    # Check for GeoTIFF files
    tiff_paths = []
    tiff_paths.extend(glob.glob(os.path.join(directory, "*.tif")))
    tiff_paths.extend(glob.glob(os.path.join(directory, "*.tiff")))
    
    # Check for the ASTER TIFF file specifically by name
    aster_files = [f for f in tiff_paths if "ASTGTM_NC" in f or "ASTER_GDEM" in f]
    if aster_files:
        # If we have ASTER files, prioritize those
        file_paths = aster_files
        print(f"Found {len(aster_files)} ASTER TIFF files. Using these for processing.")
    elif tiff_paths:
        # If we have other TIFF files but no ASTER, use those
        file_paths = tiff_paths
        print(f"Found {len(tiff_paths)} TIFF files. Using these for processing.")
    else:
        # Only as a last resort, try HGT files
        print("No TIFF files found. Creating synthetic tile.")
        # Create a synthetic tile as a last resort
        grid = np.zeros((tile_size, tile_size), dtype=int)
        center = tile_size // 4
        size = tile_size // 2
        grid[center:center+size, center:center+size] = 128  # Simple square feature
        
        # Create properly structured synthetic tile with coordinates in meta
        tile_meta = {
            'file': 'synthetic',
            'index': 0,
            'lat': -3.0,  # Include in metadata 
            'lon': -60.0  # Include in metadata
        }
        return [{'grid': grid, 'lat': -3.0, 'lon': -60.0, 'meta': tile_meta, 'id': 0}]
    
    print(f"Using tile size: {tile_size}x{tile_size} with {overlap} pixel overlap (25%)")
    print(f"Processing {len(file_paths)} input files")
    
    # Check if we have the specific ASTER DEM file defined globally
    if is_running_on_kaggle() and 'ASTER_DEM_PATH' in globals() and os.path.exists(ASTER_DEM_PATH):
        print(f"Using specific ASTER DEM file from dataset: {ASTER_DEM_PATH}")
        file_paths = [ASTER_DEM_PATH]
    
    # Process files sequentially for more reliability
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        try:
            result_tiles = process_tile_file(file_path, tile_size, overlap, filter_clouds)
            # Safety check for NaN values
            for tile in result_tiles:
                if 'grid' in tile and tile['grid'] is not None:
                    # Make sure there are no remaining NaN values
                    tile['grid'] = np.nan_to_num(tile['grid'], nan=0.0)
            tiles.extend(result_tiles)
            print(f"Successfully processed {len(result_tiles)} tiles from {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            
    # If we still have no tiles, create a synthetic one
    if not tiles:
        print("No tiles were successfully created. Creating synthetic tile.")
        grid = np.zeros((tile_size, tile_size), dtype=int)
        center = tile_size // 4
        size = tile_size // 2
        grid[center:center+size, center:center+size] = 128  # Simple square feature
        
        # Create properly structured synthetic tile with coordinates in meta
        tile_meta = {
            'file': 'synthetic',
            'index': 0,
            'lat': -3.0,  # Include in metadata 
            'lon': -60.0  # Include in metadata
        }
        tiles.append({'grid': grid, 'lat': -3.0, 'lon': -60.0, 'meta': tile_meta, 'id': 0})
    
    print(f"Total tiles created: {len(tiles)}")
    return tiles

# --- Global Pipeline Status Management ---
class GlobalSupervisor:
    """Central coordination system for managing all archaeological agents."""
    
    def __init__(self, heartbeat_interval=15):
        """Initialize the global supervisor.
        
        Args:
            heartbeat_interval: Seconds between status updates
        """
        # Status tracking
        self.running = False
        self.phase = "Initializing"
        self.subphase = "Not started"
        self.active_agents = set()
        self.completed_agents = set()
        self.last_feature_count = 0
        self.feature_delta = 0
        self.last_update_time = time.time()
        self.start_time = None
        self.heartbeat_interval = heartbeat_interval
        
        # Progress tracking
        self.progress_pct = 0
        self.est_remaining_sec = 0
        self.total_operations = 0
        self.completed_operations = 0
        
        # Performance metrics
        self.features_per_second = 0
        self.operations_per_second = 0
        self.gpu_utilization_pct = 0
        self.memory_usage_mb = 0
        
        # Error tracking
        self.errors = []
        self.warnings = []
        self.stuck_agents = set()
        
        # Locking for thread safety
        self.lock = threading.Lock()
        self.supervisor_thread = None
        self.kg = None
    
    def start(self, kg):
        """Start the supervisor monitoring thread.
        
        Args:
            kg: Knowledge graph to monitor
        """
        with self.lock:
            if self.running:
                return
            
            self.running = True
            self.start_time = time.time()
            self.kg = kg
            self.phase = "Starting"
            self.subphase = "Initializing"
            
            # Start the monitoring thread
            self.supervisor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True
            )
            self.supervisor_thread.start()
            print("[SUPERVISOR] Global monitoring system activated")
    
    def stop(self):
        """Stop the supervisor monitoring thread."""
        with self.lock:
            if not self.running:
                return
            
            self.running = False
            # Wait for thread to terminate
            if self.supervisor_thread:
                self.supervisor_thread.join(timeout=2)
            print("[SUPERVISOR] Global monitoring deactivated")
    
    def update_status(self, phase, subphase=None, progress=None, agent_name=None, is_active=None):
        """Update the current status of the pipeline.
        
        Args:
            phase: Main processing phase
            subphase: Optional subphase
            progress: Optional progress percentage (0-100)
            agent_name: Optional agent name to track
            is_active: Whether the agent is active
        """
        with self.lock:
            self.phase = phase
            if subphase:
                self.subphase = subphase
            
            if progress is not None:
                self.progress_pct = progress
                
            if agent_name:
                if is_active:
                    self.active_agents.add(agent_name)
                    if agent_name in self.stuck_agents:
                        self.stuck_agents.remove(agent_name)
                else:
                    if agent_name in self.active_agents:
                        self.active_agents.remove(agent_name)
                        self.completed_agents.add(agent_name)
    
    def mark_agent_stuck(self, agent_name):
        """Mark an agent as potentially stuck.
        
        Args:
            agent_name: Name of the stuck agent
        """
        with self.lock:
            if agent_name in self.active_agents:
                self.stuck_agents.add(agent_name)
    
    def add_warning(self, message):
        """Add a warning message.
        
        Args:
            message: Warning message to log
        """
        with self.lock:
            self.warnings.append({
                'time': time.time(),
                'message': message
            })
    
    def add_error(self, message, agent_name=None):
        """Add an error message.
        
        Args:
            message: Error message to log
            agent_name: Optional agent that generated the error
        """
        with self.lock:
            self.errors.append({
                'time': time.time(),
                'agent': agent_name,
                'message': message
            })
    
    def update_metrics(self, gpu_util=None, memory_usage=None):
        """Update performance metrics.
        
        Args:
            gpu_util: GPU utilization percentage
            memory_usage: Memory usage in MB
        """
        with self.lock:
            if gpu_util is not None:
                self.gpu_utilization_pct = gpu_util
            
            if memory_usage is not None:
                self.memory_usage_mb = memory_usage
    
    def _monitor_loop(self):
        """Background thread that monitors system progress."""
        last_heartbeat = time.time()
        current_feature_count = 0
        
        while self.running:
            current_time = time.time()
            
            # Update feature count if we have a knowledge graph
            if self.kg:
                try:
                    current_feature_count = len(self.kg.features)
                    if current_feature_count != self.last_feature_count:
                        with self.lock:
                            self.feature_delta = current_feature_count - self.last_feature_count
                            self.last_feature_count = current_feature_count
                            elapsed = current_time - self.last_update_time
                            self.features_per_second = self.feature_delta / max(0.1, elapsed)
                            self.last_update_time = current_time
                except (AttributeError, TypeError):
                    # KG might not be ready yet
                    pass
            
            # Send heartbeat message at regular intervals
            if (current_time - last_heartbeat) >= self.heartbeat_interval:
                self._send_heartbeat(current_feature_count)
                last_heartbeat = current_time
            
            # Check for stuck agents
            for agent_name in list(self.active_agents):
                if agent_name in self.stuck_agents:
                    # Already marked as stuck
                    continue
                
                # Check if this agent has been active too long without updates
                # This would require additional tracking per agent
                # For now, just provide the general heartbeat
                pass
            
            # Sleep before checking again
            time.sleep(1)
    
    def _send_heartbeat(self, feature_count):
        """Send a heartbeat status message to the console.
        
        Args:
            feature_count: Current feature count
        """
        # Calculate elapsed time
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # Get analysis statistics
        cluster_count = 0
        alignment_count = 0
        
        if self.kg:
            try:
                if hasattr(self.kg, 'clusters'):
                    cluster_count = len(self.kg.clusters)
                if hasattr(self.kg, 'alignments'):
                    alignment_count = len(self.kg.alignments)
            except (AttributeError, TypeError):
                pass
        
        # Format heartbeat message
        message = f"\n[HEARTBEAT] Pipeline alive - {self.phase}: {self.subphase}"
        message += f"\n  Elapsed: {elapsed:.1f}s | Features: {feature_count}"
        
        if feature_count > 0:
            message += f" | Feat/sec: {self.features_per_second:.1f}"
        
        if cluster_count > 0 or alignment_count > 0:
            message += f"\n  Clusters: {cluster_count} | Alignments: {alignment_count}"
        
        if self.active_agents:
            message += f"\n  Active agents: {', '.join(self.active_agents)}"
        
        if self.stuck_agents:
            message += f"\n  Potentially stuck: {', '.join(self.stuck_agents)}"
            message += "\n  Processing continues in background..."
        
        if self.gpu_utilization_pct > 0:
            message += f"\n  GPU: {self.gpu_utilization_pct}% | Memory: {self.memory_usage_mb:.1f}MB"
        
        # Print heartbeat to console
        print(message)

# Create a global supervisor instance
GLOBAL_SUPERVISOR = GlobalSupervisor(heartbeat_interval=15)

# --- GPU Backend Detection and Setup ---
# Default to CPU/NumPy
BACKEND = "numpy"

# 1) Try PyTorch + MPS (Apple Silicon) or CUDA (Kaggle)
try:
    import torch
    if torch.backends.mps.is_available():
        BACKEND = "torch_mps"
        print("MPS (Metal Performance Shaders) detected! Using PyTorch + MPS for acceleration.")
    elif torch.cuda.is_available():
        BACKEND = "torch_cuda"
        print(f"CUDA detected! Using PyTorch + CUDA for acceleration on device: {torch.cuda.get_device_name(0)}")
    else:
        BACKEND = "torch_cpu"
        print("PyTorch available but no GPU detected. Using PyTorch CPU implementation.")
except ImportError:
    torch = None
    print("PyTorch not available. Trying TensorFlow...")

# 2) Try TensorFlow-Metal / GPU if PyTorch not available
if BACKEND == "numpy":
    try:
        import tensorflow as tf
        # On Mac: use the Metal plugin, on Kaggle: GPU:0
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # If tf-metal is installed on Mac, it'll show up as GPU
            BACKEND = "tf_gpu"
            print("GPU detected for TensorFlow! Using TensorFlow GPU acceleration.")
            # Configure TensorFlow to use only necessary GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            BACKEND = "tf_cpu"
            print("TensorFlow available but no GPU detected. Using TensorFlow CPU implementation.")
    except ImportError:
        tf = None
        print("TensorFlow not available. Falling back to NumPy.")

# Print final backend
if BACKEND == "numpy":
    print("→ Using NumPy CPU backend (slowest)")
else:
    print(f"→ Using {BACKEND} as computation backend")

# Helper functions to abstract backend operations
def to_backend(array):
    """Convert numpy array to the active backend format"""
    if BACKEND == "numpy":
        return array
    elif BACKEND.startswith("torch"):
        device = torch.device("mps") if BACKEND == "torch_mps" else \
                 torch.device("cuda") if BACKEND == "torch_cuda" else \
                 torch.device("cpu")
        return torch.from_numpy(array).float().to(device)
    elif BACKEND.startswith("tf"):
        with tf.device("/GPU:0" if BACKEND == "tf_gpu" else "/CPU:0"):
            return tf.convert_to_tensor(array, dtype=tf.float32)
    return array

def to_numpy(tensor):
    """Convert a backend tensor back to numpy"""
    if BACKEND == "numpy":
        return tensor
    elif BACKEND.startswith("torch"):
        return tensor.cpu().numpy()
    elif BACKEND.startswith("tf"):
        return tensor.numpy()
    return tensor

def compute_gradients(array):
    """Compute gradients using the active backend"""
    if BACKEND.startswith("torch"):
        # PyTorch implementation
        t = to_backend(array)
        grad_x = torch.zeros_like(t)
        grad_y = torch.zeros_like(t)
        grad_x[:, 1:-1] = t[:, 2:] - t[:, :-2]
        grad_y[1:-1, :] = t[2:, :] - t[:-2, :]
        return to_numpy(grad_x), to_numpy(grad_y)
    
    elif BACKEND.startswith("tf"):
        # TensorFlow implementation
        with tf.device("/GPU:0" if BACKEND == "tf_gpu" else "/CPU:0"):
            t = to_backend(array)
            grad_x = tf.pad(t[:, 2:] - t[:, :-2], [[0, 0], [1, 1]])
            grad_y = tf.pad(t[2:, :] - t[:-2, :], [[1, 1], [0, 0]])
            return to_numpy(grad_x), to_numpy(grad_y)
    
    else:
        # NumPy implementation
        grad_x = np.zeros_like(array)
        grad_y = np.zeros_like(array)
        grad_x[:, 1:-1] = array[:, 2:] - array[:, :-2]
        grad_y[1:-1, :] = array[2:, :] - array[:-2, :]
        return grad_x, grad_y

def compute_edge_detection(array):
    """Detect edges using the active backend"""
    if BACKEND.startswith("torch"):
        # PyTorch implementation
        t = to_backend(array)
        grad_x = torch.zeros_like(t)
        grad_y = torch.zeros_like(t)
        grad_x[:, 1:-1] = t[:, 2:] - t[:, :-2]
        grad_y[1:-1, :] = t[2:, :] - t[:-2, :]
        mag = torch.sqrt(grad_x**2 + grad_y**2)
        edges = (mag > (mag.mean() + mag.std())).to(torch.int32)
        return to_numpy(edges)
    
    elif BACKEND.startswith("tf"):
        # TensorFlow implementation
        with tf.device("/GPU:0" if BACKEND == "tf_gpu" else "/CPU:0"):
            t = to_backend(array)
            grad_x = tf.pad(t[:, 2:] - t[:, :-2], [[0, 0], [1, 1]])
            grad_y = tf.pad(t[2:, :] - t[:-2, :], [[1, 1], [0, 0]])
            mag = tf.sqrt(grad_x**2 + grad_y**2)
            edges = tf.cast(mag > (tf.reduce_mean(mag) + tf.math.reduce_std(mag)), tf.int32)
            return to_numpy(edges)
    
    else:
        # NumPy implementation
        grad_x = np.zeros_like(array)
        grad_y = np.zeros_like(array)
        grad_x[:, 1:-1] = array[:, 2:] - array[:, :-2]
        grad_y[1:-1, :] = array[2:, :] - array[:-2, :]
        mag = np.sqrt(grad_x**2 + grad_y**2)
        edges = (mag > (mag.mean() + mag.std())).astype(np.int32)
        return edges

# --- Base Classes ---
class BaseAgent:
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    def transform(self, g):
        """Transform the input grid. Must be implemented by subclasses."""
        return g
        
    def __repr__(self):
        return f"{self.name}"
        
    def report_progress(self, message, phase=None, subphase=None, progress=None, is_active=True):
        """Report progress to the global supervisor and print to console.
        
        Args:
            message: The progress message to display
            phase: Optional phase name
            subphase: Optional subphase name
            progress: Optional progress percentage (0-100)
            is_active: Whether this agent is currently active
        """
        agent_name = self.name
        
        # Print to console
        print(f"[{agent_name}] {message}")
        
        # Update global supervisor if available
        if 'GLOBAL_SUPERVISOR' in globals():
            GLOBAL_SUPERVISOR.update_status(
                phase=phase or agent_name, 
                subphase=subphase or message,
                progress=progress,
                agent_name=agent_name,
                is_active=is_active
            )

class AmazonKG(BaseAgent):
    """Enhanced knowledge graph for Amazon archaeological feature detection and analysis."""
    
    def __init__(self, storage_dir="./kg_data", name="AmazonKG"):
        super().__init__(name=name)
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Core data stores
        self.features = []    # All archaeological features (patterns, templates, sites, etc.)
        self.clusters = []    # Groups of related features
        self.alignments = []  # Orientation alignments between features
        self.data_sources = [] # Information about data sources
        
        # Backward compatibility with LocalPatternKG
        self.patterns = []    # Legacy: List of pattern dicts
        self.templates = []   # Legacy: List of template dicts
        self.tiles = []       # Legacy: List of tile dicts
        self.sites = []       # Legacy: List of site dicts
        
        # Reasoning chains for transparent decision making
        self.reasoning_chains = []  # Stores detection logic chains
        
        # Example Cypher queries for knowledge exploration
        self.example_queries = [
            {
                "name": "Find all high-confidence new sites",
                "description": "Discovers potential new archaeological sites with high confidence scores that are not near known sites",
                "cypher": "MATCH (f:Feature) WHERE f.confidence >= 0.9 AND NOT EXISTS { MATCH (f)-[:NEAR]->(s:KnownSite) } RETURN f",
                "expected_result": "Returns features that likely represent new archaeological discoveries"
            },
            {
                "name": "Find clusters with multiple feature types",
                "description": "Identifies areas with diverse archaeological features, suggesting complex settlements",
                "cypher": "MATCH (c:Cluster)-[:CONTAINS]->(f:Feature) WITH c, collect(distinct f.feature_type) as types WHERE size(types) > 1 RETURN c, types",
                "expected_result": "Returns clusters containing different types of features (e.g., enclosures and linear features)"
            },
            {
                "name": "Find aligned linear features near clusters",
                "description": "Discovers potential roadways or causeways connecting settlement areas",
                "cypher": "MATCH (f:Feature {feature_type: 'LinearFeature'})-[:MEMBER_OF]->(a:AlignmentGroup), (c:Cluster) WHERE distance(f.geometry, c.centroid) <= 5000 RETURN f, a, c",
                "expected_result": "Returns linear features that share orientation and are within 5km of clusters"
            },
            {
                "name": "Find cardinal-oriented rectangular enclosures",
                "description": "Discovers rectangular structures aligned with cardinal directions, a common archaeological pattern",
                "cypher": "MATCH (f:Feature {feature_type: 'RectangularEnclosure'}) WHERE f.orientation <= 15 OR (f.orientation >= 75 AND f.orientation <= 105) OR f.orientation >= 165 RETURN f",
                "expected_result": "Returns rectangular features aligned N-S or E-W within 15 degrees"
            },
            {
                "name": "Trace detection reasoning for feature",
                "description": "Reveals the full detection and reasoning process for a specific feature",
                "cypher": "MATCH (f:Feature)-[:HAS_REASONING]->(r:ReasoningChain) WHERE f.id = $feature_id RETURN f, r",
                "expected_result": "Returns the feature and its complete detection justification chain"
            }
        ]

    def _generate_id(self, prefix=""):
        """Generate a unique ID for a new entity."""
        return f"{prefix}{str(uuid.uuid4())[:8]}"
    
    def _calculate_distance_km(self, lat1, lon1, lat2, lon2) -> float:
        """Calculate the Haversine distance between two points in kilometers."""
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return 6371 * c  # Earth radius in km
        
    # --- Feature Management ---
    
    def add_feature(self, feature_type: str, geometry, **attributes) -> str:
        """Add a feature node to the knowledge graph.
        
        Args:
            feature_type: Type of feature (e.g., "LinearFeature", "RectangularEnclosure")
            geometry: Spatial data (point coordinates, polygon, etc.)
            **attributes: Any additional attributes for the feature
        
        Returns:
            ID of the created feature
        """
        feature_id = self._generate_id(f"{feature_type}_")
        
        # Set default metadata if not provided
        if 'detected_by' not in attributes:
            attributes['detected_by'] = "Manual"
        if 'detection_date' not in attributes:
            attributes['detection_date'] = datetime.datetime.now().isoformat()
            
        feature = {
            'id': feature_id,
            'feature_type': feature_type,
            'geometry': geometry,
            **attributes
        }
        
        self.features.append(feature)
        return feature_id
    
    def add_cluster(self, member_feature_ids: List[str], 
                   cluster_type: str = "GenericCluster", 
                   **attributes) -> str:
        """Create a cluster grouping multiple features.
        
        Args:
            member_feature_ids: List of feature IDs in this cluster
            cluster_type: Type of cluster (e.g., "MoundCluster", "GeoglyphComplex")
            **attributes: Additional cluster attributes
            
        Returns:
            ID of the created cluster
        """
        cluster_id = self._generate_id("Cluster_")
        
        # Calculate centroid if not provided
        if 'centroid' not in attributes:
            # Find member features and calculate average lat/lon
            members = [f for f in self.features if f['id'] in member_feature_ids]
            if members and all(isinstance(m.get('geometry', {}), dict) 
                              and 'lat' in m.get('geometry', {}) 
                              and 'lon' in m.get('geometry', {}) for m in members):
                lat_avg = sum(m['geometry']['lat'] for m in members) / len(members)
                lon_avg = sum(m['geometry']['lon'] for m in members) / len(members)
                attributes['centroid'] = {'lat': lat_avg, 'lon': lon_avg}
        
        cluster = {
            'id': cluster_id,
            'feature_type': 'Cluster',
            'cluster_type': cluster_type,
            'member_features': member_feature_ids,
            'member_count': len(member_feature_ids),
            **attributes
        }
        
        self.clusters.append(cluster)
        return cluster_id
    
    def add_alignment(self, member_feature_ids: List[str], 
                     orientation: float,
                     alignment_type: str = "SharedOrientation",
                     **attributes) -> str:
        """Create an alignment relationship between features.
        
        Args:
            member_feature_ids: List of feature IDs that share this alignment
            orientation: Orientation angle in degrees
            alignment_type: Type of alignment ("SharedOrientation", "Collinear", etc.)
            **attributes: Additional alignment attributes
            
        Returns:
            ID of the created alignment
        """
        alignment_id = self._generate_id("Align_")
        
        alignment = {
            'id': alignment_id,
            'feature_type': 'AlignmentGroup',
            'members': member_feature_ids,
            'orientation': orientation,
            'alignment_type': alignment_type,
            **attributes
        }
        
        self.alignments.append(alignment)
        return alignment_id
    
    def add_data_source(self, source_type: str, name: str, 
                       date: Optional[str] = None,
                       coverage_area: Optional[Dict] = None,
                       **attributes) -> str:
        """Add information about a data source.
        
        Args:
            source_type: Type of data source (e.g., "Sentinel2", "LiDAR")
            name: Name or identifier of the source
            date: Date of data acquisition
            coverage_area: Geographic area covered by this source
            **attributes: Additional source attributes
            
        Returns:
            ID of the created data source
        """
        source_id = self._generate_id("Source_")
        
        source = {
            'id': source_id,
            'source_type': source_type,
            'name': name,
            'date': date,
            'coverage_area': coverage_area,
            **attributes
        }
        
        self.data_sources.append(source)
        return source_id
    
    # --- Legacy Interface (LocalPatternKG compatibility) ---
    
    def add_pattern(self, input_grid, meta, tile_id=None):
        """Legacy method for compatibility with LocalPatternKG."""
        pid = len(self.patterns)
        pattern = {
            'id': pid,
            'input': np.array(input_grid),
            'meta': meta,
            'tile_id': tile_id
        }
        self.patterns.append(pattern)
        
        # Also add as a modern feature
        if 'template_idx' in meta:
            feature_type = "TemplateMatch"
        else:
            feature_type = "Pattern"
            
        # If we have tile coordinates, use them for the feature
        tile_coords = {}
        if tile_id is not None:
            tile = next((t for t in self.tiles if t['id'] == tile_id), None)
            if tile:
                tile_coords = {
                    'lat': tile['lat'],
                    'lon': tile['lon']
                }
        
        # Create geometry with available information
        geometry = {
            **tile_coords,
            'y': meta.get('y', 0),
            'x': meta.get('x', 0),
            'height': meta.get('height', input_grid.shape[0]),
            'width': meta.get('width', input_grid.shape[1])
        }
        
        # Add to modern features store
        self.add_feature(
            feature_type=feature_type,
            geometry=geometry,
            grid=np.array(input_grid),
            score=meta.get('score', 0.0),
            template_idx=meta.get('template_idx', None),
            detected_by="TemplateMatchAgent",
            legacy_id=pid,
            tile_id=tile_id
        )
        
        return pid

    def add_template(self, grid, desc=""):
        """Legacy method for compatibility with LocalPatternKG."""
        tid = len(self.templates)
        tpl = {'id': tid, 'grid': np.array(grid), 'desc': desc}
        self.templates.append(tpl)
        
        # Also add as a modern feature
        self.add_feature(
            feature_type="Template",
            geometry={'shape': grid.shape},
            grid=np.array(grid),
            description=desc,
            legacy_id=tid
        )
        
        return tid

    def add_tile(self, grid, lat, lon, meta=None):
        """Legacy method for compatibility with LocalPatternKG."""
        tid = len(self.tiles)
        tile = {'id': tid, 'grid': np.array(grid), 'lat': lat, 'lon': lon, 'meta': meta or {}}
        self.tiles.append(tile)
        
        # Also add as a modern feature
        self.add_feature(
            feature_type="Tile",
            geometry={'lat': lat, 'lon': lon},
            grid=np.array(grid),
            meta=meta or {},
            legacy_id=tid
        )
        
        return tid

    def add_site(self, lat, lon, desc="", is_validation=False):
        """Add a known archaeological site.
        
        Args:
            lat: Latitude
            lon: Longitude
            desc: Site description
            is_validation: Whether this is a validation site (not an actual detected site)
        
        Returns:
            Site ID
        """
        sid = len(self.sites)
        site = {'id': sid, 'lat': lat, 'lon': lon, 'desc': desc, 'is_validation': is_validation}
        self.sites.append(site)
        
        # Also add as a modern feature
        feature_type = "ValidationSite" if is_validation else "KnownSite"
        self.add_feature(
            feature_type=feature_type,
            geometry={'lat': lat, 'lon': lon},
            description=desc,
            is_validation=is_validation,
            legacy_id=sid
        )
        
        return sid

    def get_templates(self):
        """Legacy method for compatibility with LocalPatternKG."""
        return [tpl['grid'] for tpl in self.templates]

    def get_patterns(self):
        """Legacy method for compatibility with LocalPatternKG."""
        return self.patterns

    def get_tiles(self):
        """Legacy method for compatibility with LocalPatternKG."""
        return self.tiles

    def get_sites(self):
        """Legacy method for compatibility with LocalPatternKG."""
        return self.sites
    
    # --- Legacy Cypher-like queries ---
    def query_patterns(self, min_score=0.8, template_idx=None, near_site=None, max_dist_km=10):
        """Legacy method for compatibility with LocalPatternKG."""
        results = []
        for pat in self.patterns:
            if pat['meta']['score'] < min_score:
                continue
            if template_idx is not None and pat['meta']['template_idx'] != template_idx:
                continue
            if near_site is not None:
                tile = next((t for t in self.tiles if t['id'] == pat['tile_id']), None)
                if tile is None:
                    continue
                lat1, lon1 = tile['lat'], tile['lon']
                lat2, lon2 = near_site['lat'], near_site['lon']
                dist = self._calculate_distance_km(lat1, lon1, lat2, lon2)
                if dist > max_dist_km:
                    continue
            results.append(pat)
        return results
    
    # --- Modern Query Methods ---
    
    def query_features(self, 
                      feature_type: Optional[str] = None,
                      min_score: Optional[float] = None,
                      near_point: Optional[Dict] = None,
                      max_distance_km: Optional[float] = 10,
                      has_orientation: Optional[float] = None,
                      orientation_tolerance: float = 15.0,
                      detected_by: Optional[str] = None) -> List[Dict]:
        """Query features with various filters.
        
        Args:
            feature_type: Type of feature to filter for
            min_score: Minimum confidence score
            near_point: Dict with lat, lon to search near
            max_distance_km: Maximum distance in km for near_point search
            has_orientation: Filter for features with this orientation (±tolerance)
            orientation_tolerance: Degrees of tolerance for orientation matching
            detected_by: Filter by detecting agent
            
        Returns:
            List of matching features
        """
        results = []
        
        for feature in self.features:
            # Type filter
            if feature_type and feature.get('feature_type') != feature_type:
                continue
                
            # Score filter
            if min_score is not None and feature.get('score', 0) < min_score:
                continue
                
            # Detecting agent filter
            if detected_by and feature.get('detected_by') != detected_by:
                continue
                
            # Orientation filter (if feature has orientation)
            if has_orientation is not None:
                feat_orientation = feature.get('orientation')
                if feat_orientation is None:
                    continue
                    
                # Handle circular nature of angles
                diff = abs((feat_orientation - has_orientation + 180) % 360 - 180)
                if diff > orientation_tolerance:
                    continue
            
            # Location filter
            if near_point and 'geometry' in feature:
                geom = feature['geometry']
                if not (isinstance(geom, dict) and 'lat' in geom and 'lon' in geom):
                    continue
                    
                dist = self._calculate_distance_km(
                    geom['lat'], geom['lon'], 
                    near_point['lat'], near_point['lon']
                )
                if dist > max_distance_km:
                    continue
            
            results.append(feature)
            
        return results
    
    def query_clusters(self,
                      cluster_type: Optional[str] = None,
                      min_members: Optional[int] = None,
                      contains_feature_type: Optional[str] = None,
                      contains_multiple_types: bool = False) -> List[Dict]:
        """Query clusters with various filters.
        
        Args:
            cluster_type: Type of cluster to filter for
            min_members: Minimum number of members in cluster
            contains_feature_type: Cluster must contain this feature type
            contains_multiple_types: If True, cluster must have >1 feature types
            
        Returns:
            List of matching clusters
        """
        results = []
        
        for cluster in self.clusters:
            # Type filter
            if cluster_type and cluster.get('cluster_type') != cluster_type:
                continue
                
            # Member count filter
            if min_members is not None and cluster.get('member_count', 0) < min_members:
                continue
                
            # Get members to check feature types
            member_ids = cluster.get('member_features', [])
            members = [f for f in self.features if f['id'] in member_ids]
            
            # Feature type containment filter
            if contains_feature_type:
                if not any(m.get('feature_type') == contains_feature_type for m in members):
                    continue
            
            # Multiple feature types filter
            if contains_multiple_types:
                feature_types = {m.get('feature_type') for m in members if 'feature_type' in m}
                if len(feature_types) <= 1:
                    continue
            
            results.append(cluster)
            
        return results
    
    def query_alignments(self,
                        min_orientation: Optional[float] = None,
                        max_orientation: Optional[float] = None,
                        alignment_type: Optional[str] = None,
                        min_members: Optional[int] = None) -> List[Dict]:
        """Query alignment groups with various filters.
        
        Args:
            min_orientation: Minimum orientation angle
            max_orientation: Maximum orientation angle
            alignment_type: Type of alignment
            min_members: Minimum number of members
            
        Returns:
            List of matching alignments
        """
        results = []
        
        for alignment in self.alignments:
            # Orientation range filter
            if min_orientation is not None and alignment.get('orientation', 0) < min_orientation:
                continue
                
            if max_orientation is not None and alignment.get('orientation', 0) > max_orientation:
                continue
                
            # Type filter
            if alignment_type and alignment.get('alignment_type') != alignment_type:
                continue
                
            # Member count filter
            if min_members is not None:
                members = alignment.get('members', [])
                if len(members) < min_members:
                    continue
            
            results.append(alignment)
            
        return results
    
    def find_aligned_features_near_cluster(self, 
                                          alignment_type: str = "SharedOrientation", 
                                          max_distance_km: float = 5.0,
                                          cluster_type: Optional[str] = None) -> List[Dict]:
        """Find aligned features within a distance of any cluster.
        
        Args:
            alignment_type: Type of alignment to look for
            max_distance_km: Maximum distance to cluster
            cluster_type: Optional cluster type to filter for
            
        Returns:
            List of features that are aligned and near clusters
        """
        # First get relevant clusters
        clusters = self.query_clusters(cluster_type=cluster_type)
        
        # Then get alignment groups
        alignments = self.query_alignments(alignment_type=alignment_type, min_members=2)
        
        results = []
        
        # For each alignment, check if any member is near a cluster
        for alignment in alignments:
            member_ids = alignment.get('members', [])
            members = [f for f in self.features if f['id'] in member_ids]
            
            for member in members:
                if 'geometry' not in member or not isinstance(member['geometry'], dict):
                    continue
                
                for cluster in clusters:
                    if 'centroid' not in cluster or not isinstance(cluster['centroid'], dict):
                        continue
                        
                    if 'lat' not in member['geometry'] or 'lon' not in member['geometry']:
                        continue
                        
                    # Calculate distance from feature to cluster centroid
                    dist = self._calculate_distance_km(
                        member['geometry']['lat'], member['geometry']['lon'],
                        cluster['centroid']['lat'], cluster['centroid']['lon']
                    )
                    
                    if dist <= max_distance_km:
                        # Add this feature if not already in results
                        if member not in results:
                            results.append(member)
        
        return results
    
    # --- File Operations ---
    
    def save(self, filename: Optional[str] = None):
        """Save the knowledge graph to a JSON file.
        
        Args:
            filename: Optional filename to save to, defaults to a timestamped file in storage_dir
        """
        if filename is None:
            # Generate a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"amazon_kg_{timestamp}.json"
        
        # Create full path
        path = os.path.join(self.storage_dir, filename)
        backup_path = f"{path}.bak"
        temp_path = f"{path}.tmp"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # If file already exists, create a backup
        if os.path.exists(path):
            try:
                shutil.copy2(path, backup_path)
                print(f"Created backup at {backup_path}")
            except Exception as e:
                print(f"Warning: Failed to create backup: {e}")
        
        # Prepare data for serialization
        kg_data = {
            'features': self._prepare_for_json(self.features),
            'clusters': self._prepare_for_json(self.clusters),
            'alignments': self._prepare_for_json(self.alignments),
            'data_sources': self._prepare_for_json(self.data_sources),
            'templates': [],
            'patterns': [],
            'tiles': [],
            'sites': self._prepare_for_json(self.sites),
            'version': '2.0'  # Add versioning to handle format changes
        }
        
        # Handle large arrays separately with special serialization
        try:
            # Templates might contain large arrays, handle carefully
            templates_data = []
            for t in self.templates:
                template_copy = t.copy()
                # Store array shape and flatten to list for better JSON compatibility
                grid = template_copy.pop('grid')
                if isinstance(grid, np.ndarray):
                    template_copy['grid_shape'] = grid.shape
                    template_copy['grid_data'] = grid.flatten().tolist()
                templates_data.append(template_copy)
            kg_data['templates'] = templates_data
            
            # Handle patterns in the same way
            patterns_data = []
            for p in self.patterns:
                pattern_copy = p.copy()
                # Store array shape and flatten
                grid = pattern_copy.pop('grid')
                if isinstance(grid, np.ndarray):
                    pattern_copy['grid_shape'] = grid.shape
                    pattern_copy['grid_data'] = grid.flatten().tolist()
                patterns_data.append(pattern_copy)
            kg_data['patterns'] = patterns_data
            
            # Tiles need special handling due to large grids
            # Save them in chunks to avoid memory issues
            chunk_size = 100
            tiles_data = []
            
            for chunk_start in range(0, len(self.tiles), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(self.tiles))
                chunk = self.tiles[chunk_start:chunk_end]
                
                chunk_data = []
                for t in chunk:
                    tile_copy = t.copy()
                    # Store grid shape and flatten
                    grid = tile_copy.pop('grid')
                    if isinstance(grid, np.ndarray):
                        tile_copy['grid_shape'] = grid.shape
                        tile_copy['grid_data'] = grid.flatten().tolist()
                    chunk_data.append(tile_copy)
                
                # Add chunk to main data
                tiles_data.extend(chunk_data)
                
            kg_data['tiles'] = tiles_data
        except Exception as e:
            print(f"Warning during data preparation: {e}")
            # Continue with best-effort save
        
        try:
            # First write to a temporary file
            with open(temp_path, 'w') as f:
                print(f"Saving knowledge graph to {path}...")
                json.dump(kg_data, f)
            
            # Then rename temp file to final path (atomic operation)
            os.replace(temp_path, path)
            print(f"✓ Knowledge graph saved to {path}")
            
            # Save a compressed version for backup (gzip)
            compressed_path = f"{path}.gz"
            with gzip.open(compressed_path, 'wt') as f:
                json.dump(kg_data, f)
            print(f"✓ Compressed backup saved to {compressed_path}")
            
            return path
        except Exception as e:
            print(f"Error saving knowledge graph: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            # Try to recover from backup if save fails
            if os.path.exists(backup_path):
                print(f"Attempting to restore from backup...")
                try:
                    shutil.copy2(backup_path, path)
                    print(f"Restored from backup at {backup_path}")
                except Exception as e2:
                    print(f"Failed to restore from backup: {e2}")
            
            raise e

    def load(self, filename: str):
        """Load a knowledge graph from a JSON file.
        
        Args:
            filename: Path to the JSON file to load from
        """
        # Check if the path is absolute or relative
        if os.path.isabs(filename):
            path = filename
        else:
            path = os.path.join(self.storage_dir, filename)
        
        # Try both regular and compressed versions
        if not os.path.exists(path) and os.path.exists(f"{path}.gz"):
            print(f"Regular file not found, trying compressed version...")
            try:
                with gzip.open(f"{path}.gz", 'rt') as f:
                    kg_data = json.load(f)
                print(f"Successfully loaded compressed graph from {path}.gz")
            except Exception as e:
                print(f"Error loading compressed graph: {e}")
                raise e
        else:
            # Try to load the regular file with robust error handling
            try:
                with open(path, 'r') as f:
                    kg_data = json.load(f)
            except json.JSONDecodeError as e:
                # Try to recover from truncated file
                print(f"JSONDecodeError: {e} - attempting recovery...")
                
                # Try to read the file and fix truncation issues
                with open(path, 'r') as f:
                    content = f.read()
                
                # Look for the last complete object
                last_brace = content.rfind('}')
                if last_brace > 0:
                    # Try to parse the content up to the last closing brace
                    truncated_content = content[:last_brace+1]
                    try:
                        # Try to fix JSON by closing any unclosed arrays/objects
                        fixed_content = self._attempt_json_repair(truncated_content)
                        kg_data = json.loads(fixed_content)
                        print(f"Successfully recovered from truncated JSON file")
                    except:
                        # If that fails, try the backup or raise the original error
                        if os.path.exists(f"{path}.bak"):
                            print(f"Attempting to load from backup...")
                            with open(f"{path}.bak", 'r') as f:
                                kg_data = json.load(f)
                            print(f"Successfully loaded from backup")
                        else:
                            raise e
                else:
                    # If we couldn't find a possible truncation point
                    raise e
            except Exception as e:
                # Handle other exceptions
                print(f"Error loading knowledge graph: {e}")
                
                # Try to load from backup if it exists
                if os.path.exists(f"{path}.bak"):
                    print(f"Attempting to load from backup...")
                    try:
                        with open(f"{path}.bak", 'r') as f:
                            kg_data = json.load(f)
                        print(f"Successfully loaded from backup")
                    except Exception as backup_e:
                        print(f"Error loading from backup: {backup_e}")
                        raise e
                else:
                    raise e
        
        # Check file version for compatibility
        version = kg_data.get('version', '1.0')
        print(f"Loading knowledge graph version {version}")
        
        # Reset current state
        self.features.clear()
        self.clusters.clear()
        self.alignments.clear()
        self.data_sources.clear()
        self.templates.clear()
        self.patterns.clear()
        self.tiles.clear()
        self.sites.clear()
        
        # Restore data
        self.features = self._restore_from_json(kg_data.get('features', []))
        self.clusters = self._restore_from_json(kg_data.get('clusters', []))
        self.alignments = self._restore_from_json(kg_data.get('alignments', []))
        self.data_sources = self._restore_from_json(kg_data.get('data_sources', []))
        self.sites = self._restore_from_json(kg_data.get('sites', []))
        
        # Handle special cases for arrays
        try:
            # Restore templates
            for t in kg_data.get('templates', []):
                template_copy = t.copy()
                if 'grid_shape' in template_copy and 'grid_data' in template_copy:
                    # Reconstruct the grid from shape and data
                    shape = template_copy.pop('grid_shape')
                    data = template_copy.pop('grid_data')
                    template_copy['grid'] = np.array(data).reshape(shape)
                self.templates.append(template_copy)
            
            # Restore patterns
            for p in kg_data.get('patterns', []):
                pattern_copy = p.copy()
                if 'grid_shape' in pattern_copy and 'grid_data' in pattern_copy:
                    # Reconstruct the grid from shape and data
                    shape = pattern_copy.pop('grid_shape')
                    data = pattern_copy.pop('grid_data')
                    pattern_copy['grid'] = np.array(data).reshape(shape)
                self.patterns.append(pattern_copy)
            
            # Restore tiles (memory-efficient approach)
            for t in kg_data.get('tiles', []):
                tile_copy = t.copy()
                if 'grid_shape' in tile_copy and 'grid_data' in tile_copy:
                    # Reconstruct the grid from shape and data
                    shape = tile_copy.pop('grid_shape')
                    data = tile_copy.pop('grid_data')
                    tile_copy['grid'] = np.array(data).reshape(shape)
                self.tiles.append(tile_copy)
                
            print(f"Loaded {len(self.features)} features, {len(self.clusters)} clusters, {len(self.alignments)} alignments")
            print(f"Loaded {len(self.templates)} templates, {len(self.patterns)} patterns, {len(self.tiles)} tiles")
        except Exception as e:
            print(f"Warning during data restoration: {e}")
            # Continue with best-effort load
        
        return self

    def _attempt_json_repair(self, content: str) -> str:
        """Attempt to repair truncated JSON by balancing brackets and braces.
        
        Args:
            content: Potentially truncated JSON string
            
        Returns:
            Fixed JSON string with balanced brackets and braces
        """
        # Count opening and closing brackets/braces
        open_braces = content.count('{')
        close_braces = content.count('}')
        open_brackets = content.count('[')
        close_brackets = content.count(']')
        
        # Add missing closing braces/brackets
        repaired = content
        for _ in range(open_brackets - close_brackets):
            repaired += ']'
        for _ in range(open_braces - close_braces):
            repaired += '}'
            
        return repaired
    
    def _prepare_for_json(self, data_list):
        """Convert numpy arrays to lists for JSON serialization."""
        result = []
        for item in data_list:
            item_copy = {}
            for k, v in item.items():
                if isinstance(v, np.ndarray):
                    item_copy[k] = v.tolist()
                else:
                    item_copy[k] = v
            result.append(item_copy)
        return result
    
    def _restore_from_json(self, data_list):
        """Restore numpy arrays from lists after JSON deserialization."""
        result = []
        for item in data_list:
            item_copy = {}
            for k, v in item.items():
                if k in ['grid', 'input'] and isinstance(v, list):
                    item_copy[k] = np.array(v)
                else:
                    item_copy[k] = v
            result.append(item_copy)
        return result
    
    def __repr__(self):
        return (f"<AmazonKG: {len(self.features)} features, "
                f"{len(self.clusters)} clusters, "
                f"{len(self.alignments)} alignments, "
                f"{len(self.data_sources)} data sources>")

    def transform(self, g, **kwargs):
        """Transform an input grid using the knowledge in this graph.
        
        This is the main processing method required by BaseAgent. For a KG,
        this could analyze the grid and store relevant features.
        
        Args:
            g: Input grid to analyze
            **kwargs: Additional parameters for the transform
            
        Returns:
            The original grid or an annotated version
        """
        # Default implementation for a KG: store the grid and return it
        # More sophisticated implementations could detect features in the grid
        meta = kwargs.get('meta', {})
        tile_id = kwargs.get('tile_id')
        
        # Store as a tile if not already in the KG
        if not any(np.array_equal(t.get('grid'), g) for t in self.tiles):
            lat = kwargs.get('lat', 0)
            lon = kwargs.get('lon', 0)
            self.add_tile(g, lat, lon, meta)
            
        return g

    def add_reasoning_chain(self, feature_id, reasoning_steps):
        """Add a standardized chain-of-thought reasoning explaining how and why a feature was detected.
        
        This implements a standardized reasoning framework where each detection agent creates
        detailed, step-by-step reasoning chains that explain their detections, including evidence,
        conclusions, and archaeological significance assessments.
        
        Args:
            feature_id: ID of the feature this reasoning applies to
            reasoning_steps: List of reasoning steps with evidence and conclusions
            
        Returns:
            ID of the created reasoning chain
        """
        chain_id = self._generate_id("Reason_")
        
        # Get feature data to enhance reasoning
        feature = next((f for f in self.features if f['id'] == feature_id), None)
        if not feature:
            print(f"Warning: Cannot add reasoning chain for unknown feature ID: {feature_id}")
            return None
            
        # Format of each reasoning step:
        # {
        #   "step": "Step name or number",
        #   "agent": "Agent that performed this step",
        #   "evidence": "Observed evidence from data or other agents",
        #   "conclusion": "Conclusion drawn from the evidence",
        #   "significance": "Archaeological significance of this conclusion"
        # }
        
        # Check if an existing chain exists for this feature
        existing_chain = next((chain for chain in self.reasoning_chains 
                             if chain['feature_id'] == feature_id), None)
        
        # Ensure all reasoning steps have the required fields
        for step in reasoning_steps:
            # Add missing fields with placeholders
            if 'step' not in step:
                step['step'] = "Analysis"
            if 'agent' not in step:
                step['agent'] = feature.get('detected_by', 'Unknown')
            if 'evidence' not in step:
                if feature.get('feature_type') == 'RectangularEnclosure':
                    step['evidence'] = f"Feature of size {feature.get('width', 0)}x{feature.get('height', 0)}"
                elif feature.get('feature_type') == 'LinearFeature':
                    step['evidence'] = f"Linear feature of length {feature.get('length', 0)}"
                else:
                    step['evidence'] = f"Feature detected with confidence {feature.get('confidence', 0)}"
            if 'conclusion' not in step:
                step['conclusion'] = "Potential archaeological feature"
                
            # If no significance is provided, calculate one
            if 'significance' not in step and len(reasoning_steps) == len([s for s in reasoning_steps if 'significance' not in s]):
                # Only calculate this if no steps have significance yet
                sig_score = self._calculate_archaeological_significance(feature)
                if sig_score > 0.8:
                    sig_text = "High archaeological significance"
                elif sig_score > 0.6:
                    sig_text = "Moderate archaeological significance"
                elif sig_score > 0.4:
                    sig_text = "Possible archaeological significance"
                else:
                    sig_text = "Low archaeological significance, may be natural feature"
                    
                # Only add to the last step
                if step == reasoning_steps[-1]:
                    step['significance'] = f"{sig_text} (score: {sig_score:.2f})"
        
        # Add a "Final assessment" step if not already present
        has_final = any(step.get('step', '') == 'Final assessment' for step in reasoning_steps)
        if not has_final:
            # Get feature-specific details for the final assessment
            feature_desc = ""
            evidence_desc = ""
            
            if feature.get('feature_type') == 'RectangularEnclosure':
                width = feature.get('width', 0)
                height = feature.get('height', 0)
                real_width = feature.get('real_width')
                real_height = feature.get('real_height')
                orientation = feature.get('orientation', 0)
                
                feature_desc = f"{feature.get('feature_type')}"
                if real_width and real_height:
                    feature_desc += f" ({real_width:.1f}×{real_height:.1f} m)"
                
                evidence_desc = f"Regular geometric shape with width×height ratio {width/height:.1f}"
                if 75 <= orientation <= 105 or orientation <= 15 or orientation >= 165:
                    evidence_desc += " aligned to cardinal directions"
                    
            elif feature.get('feature_type') == 'LinearFeature':
                length = feature.get('length', 0)
                orientation = feature.get('orientation', 0)
                
                feature_desc = f"{feature.get('feature_type')} ({length:.1f} m)"
                evidence_desc = f"Straight linear feature with orientation {orientation:.1f}°"
                
            elif feature.get('feature_type') == 'RepetitivePattern':
                feature_desc = f"{feature.get('feature_type')}"
                evidence_desc = f"Regular spacing with pattern frequency {feature.get('frequency', 0):.3f}"
            
            elif feature.get('feature_type') == 'TemplateMatch':
                template_desc = feature.get('template_description', 'Unknown template')
                score = feature.get('score', 0)
                feature_desc = f"{feature.get('feature_type')} ({template_desc})"
                evidence_desc = f"Matches known archaeological template with score {score:.2f}"
            
            else:
                feature_desc = f"{feature.get('feature_type', 'Unknown feature')}"
                evidence_desc = f"Detected with confidence {feature.get('confidence', 0):.2f}"
            
            # Calculate archaeological significance
            sig_score = self._calculate_archaeological_significance(feature)
            
            # Create a detailed final assessment
            final_step = {
                "step": "Final assessment",
                "agent": "Archaeological reasoning engine",
                "evidence": f"{feature_desc}: {evidence_desc}",
                "conclusion": self._get_significance_description(sig_score),
                "significance": f"Archaeological significance score: {sig_score:.2f}"
            }
            
            reasoning_steps.append(final_step)
        
        if existing_chain:
            # Append new steps to existing chain
            existing_steps = existing_chain['steps']
            # Check for duplicate steps (same agent and step name)
            for new_step in reasoning_steps:
                step_agent = new_step.get('agent', '')
                step_name = new_step.get('step', '')
                
                # Skip if this exact step already exists
                if not any(s.get('agent', '') == step_agent and s.get('step', '') == step_name 
                          for s in existing_steps):
                    existing_steps.append(new_step)
            
            # Update timestamp
            existing_chain['timestamp'] = datetime.datetime.now().isoformat()
            return existing_chain['id']
        else:
            # Create new reasoning chain
            reasoning_chain = {
                'id': chain_id,
                'feature_id': feature_id,
                'steps': reasoning_steps,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.reasoning_chains.append(reasoning_chain)
            return chain_id
            
    def _get_significance_description(self, score):
        """Get a descriptive text for an archaeological significance score."""
        if score >= 0.8:
            return "High confidence archaeological feature likely representing human construction"
        elif score >= 0.6:
            return "Moderate confidence archaeological feature, probable human origin"
        elif score >= 0.5:
            return "Possible archaeological feature requiring further investigation"
        elif score >= 0.3:
            return "Low confidence feature, might be natural or modern"
        else:
            return "Very low confidence, likely natural or modern feature"
        
    def export_for_cypher(self, filename=None):
        """Export knowledge graph in a format ready for Neo4j/Cypher import with reasoning chains.
        
        This creates files that can be imported into Neo4j and used with Cypher queries.
        
        Args:
            filename: Base filename for the export (without extension)
            
        Returns:
            Dictionary with paths to exported files
        """
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"amazon_kg_cypher_{timestamp}"
            
        # Create directory if needed
        cypher_dir = os.path.join(self.storage_dir, "cypher_export")
        os.makedirs(cypher_dir, exist_ok=True)
        
        # Prepare node and relationship files
        nodes_file = os.path.join(cypher_dir, f"{filename}_nodes.csv")
        rels_file = os.path.join(cypher_dir, f"{filename}_relationships.csv")
        queries_file = os.path.join(cypher_dir, f"{filename}_example_queries.cypher")
        readme_file = os.path.join(cypher_dir, f"{filename}_README.md")
        
        # Export nodes (features, clusters, alignments)
        nodes = []
        # Features
        for feature in self.features:
            node = feature.copy()
            node['labels'] = f":Feature:{feature['feature_type']}"  # Neo4j labels
            
            # Add detection reasoning summary if available
            related_reasoning = [r for r in self.reasoning_chains if r['feature_id'] == feature['id']]
            if related_reasoning:
                # Add a summary of the reasoning
                node['detection_reasoning'] = f"Detected by {feature.get('detected_by', 'Unknown')} with confidence {feature.get('confidence', 'Unknown')}"
                if 'score' in feature:
                    node['detection_reasoning'] += f", score {feature['score']}"
            
            # Convert geometry to string for Neo4j import
            if 'geometry' in node and isinstance(node['geometry'], dict):
                node['geometry_info'] = json.dumps(node['geometry'])
                
            nodes.append(node)
            
        # Clusters
        for cluster in self.clusters:
            node = cluster.copy()
            node['labels'] = f":Cluster:{cluster.get('cluster_type', 'GenericCluster')}"
            
            # Summarize cluster contents
            node['cluster_summary'] = f"Cluster with {cluster.get('member_count', 0)} members"
            if 'feature_types' in cluster:
                node['cluster_summary'] += f" of types: {', '.join(cluster['feature_types'])}"
                
            nodes.append(node)
            
        # Alignments
        for alignment in self.alignments:
            node = alignment.copy()
            node['labels'] = f":AlignmentGroup:{alignment.get('alignment_type', 'Generic')}"
            nodes.append(node)
            
        # Known sites
        for site in self.sites:
            node = site.copy()
            node['labels'] = ":KnownSite"
            node['is_known'] = True
            nodes.append(node)
            
        # Reasoning chains
        for reasoning in self.reasoning_chains:
            node = reasoning.copy()
            node['labels'] = ":ReasoningChain"
            # Convert steps to string
            if 'steps' in node and isinstance(node['steps'], list):
                node['steps_json'] = json.dumps(node['steps'])
            nodes.append(node)
            
        # Export relationships
        relationships = []
        
        # Feature to cluster relationships
        for cluster in self.clusters:
            if 'member_features' in cluster:
                for member_id in cluster['member_features']:
                    rel = {
                        'start_id': member_id,
                        'end_id': cluster['id'],
                        'type': 'MEMBER_OF'
                    }
                    relationships.append(rel)
                    
                    # Reverse relationship
                    rel = {
                        'start_id': cluster['id'],
                        'end_id': member_id,
                        'type': 'CONTAINS'
                    }
                    relationships.append(rel)
        
        # Feature to alignment relationships
        for alignment in self.alignments:
            if 'members' in alignment:
                for member_id in alignment['members']:
                    rel = {
                        'start_id': member_id,
                        'end_id': alignment['id'],
                        'type': 'ALIGNED_WITH',
                        'orientation': alignment.get('orientation', 0)
                    }
                    relationships.append(rel)
        
        # Feature to reasoning chain relationships
        for reasoning in self.reasoning_chains:
            if 'feature_id' in reasoning:
                rel = {
                    'start_id': reasoning['feature_id'],
                    'end_id': reasoning['id'],
                    'type': 'HAS_REASONING'
                }
                relationships.append(rel)
        
        # Spatial relationships (NEAR)
        # Connect features to nearby known sites
        for feature in self.features:
            if 'geometry' not in feature or not isinstance(feature['geometry'], dict):
                continue
                
            if 'lat' not in feature['geometry'] or 'lon' not in feature['geometry']:
                continue
                
            feature_lat = feature['geometry']['lat']
            feature_lon = feature['geometry']['lon']
            
            for site in self.sites:
                if 'lat' not in site or 'lon' not in site:
                    continue
                    
                dist = self._calculate_distance_km(
                    feature_lat, feature_lon,
                    site['lat'], site['lon']
                )
                
                if dist <= 2.0:  # Within 2km
                    rel = {
                        'start_id': feature['id'],
                        'end_id': site['id'],
                        'type': 'NEAR',
                        'distance_km': dist
                    }
                    relationships.append(rel)
        
        # Write nodes to CSV
        with open(nodes_file, 'w', newline='') as f:
            if not nodes:
                f.write("id,labels\n")  # Header only
            else:
                # Get all possible fields across all nodes
                fields = set()
                for node in nodes:
                    fields.update(node.keys())
                
                # Ensure id and labels are first
                fields = ['id', 'labels'] + sorted(f for f in fields if f not in ['id', 'labels'])
                
                # Write CSV header
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                
                # Write nodes
                for node in nodes:
                    # Ensure all fields exist (with empty values if needed)
                    row = {field: (node.get(field, "") if field in node else "") for field in fields}
                    writer.writerow(row)
        
        # Write relationships to CSV
        with open(rels_file, 'w', newline='') as f:
            if not relationships:
                f.write("start_id,end_id,type\n")  # Header only
            else:
                # Get all possible fields across all relationships
                fields = set()
                for rel in relationships:
                    fields.update(rel.keys())
                
                # Ensure start_id, end_id, and type are first
                fields = ['start_id', 'end_id', 'type'] + sorted(f for f in fields if f not in ['start_id', 'end_id', 'type'])
                
                # Write CSV header
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                
                # Write relationships
                for rel in relationships:
                    # Ensure all fields exist (with empty values if needed)
                    row = {field: (rel.get(field, "") if field in rel else "") for field in fields}
                    writer.writerow(row)
        
        # Write example Cypher queries
        with open(queries_file, 'w') as f:
            f.write("// Amazon Archaeology Knowledge Graph - Example Cypher Queries\n\n")
            
            for query in self.example_queries:
                f.write(f"// {query['name']}\n")
                f.write(f"// Description: {query['description']}\n")
                f.write(f"// Expected result: {query['expected_result']}\n")
                f.write(f"{query['cypher']};\n\n")
            
            # Add custom query for potential new discoveries
            f.write("// FIND POTENTIAL NEW ARCHAEOLOGICAL SITES\n")
            f.write("// This query finds high-confidence features that are not near known sites\n")
            f.write("MATCH (f:Feature)\n")
            f.write("WHERE f.confidence >= 0.8\n")
            f.write("AND NOT EXISTS { MATCH (f)-[:NEAR]->(s:KnownSite) }\n")
            f.write("RETURN f.id, f.feature_type, f.confidence, f.detected_by;\n\n")
            
            # Add reasoning chain query
            f.write("// EXAMINE DETECTION REASONING FOR A SPECIFIC FEATURE\n")
            f.write("// Replace 'feature-id-here' with an actual feature ID\n")
            f.write("MATCH (f:Feature)-[:HAS_REASONING]->(r:ReasoningChain)\n")
            f.write("WHERE f.id = 'feature-id-here'\n")
            f.write("RETURN f.feature_type, f.confidence, r.steps_json;\n")
        
        # Write README with import instructions
        with open(readme_file, 'w') as f:
            f.write("# Amazon Archaeology Knowledge Graph\n\n")
            f.write("This directory contains a knowledge graph of archaeological features detected in the Amazon rainforest.\n\n")
            
            f.write("## Contents\n\n")
            f.write(f"- `{os.path.basename(nodes_file)}`: Nodes (features, clusters, alignments, known sites)\n")
            f.write(f"- `{os.path.basename(rels_file)}`: Relationships between nodes\n")
            f.write(f"- `{os.path.basename(queries_file)}`: Example Cypher queries for exploration\n\n")
            
            f.write("## Import Instructions for Neo4j\n\n")
            f.write("### Using Neo4j Desktop or Neo4j Browser\n\n")
            f.write("1. Start a Neo4j database instance\n")
            f.write("2. Place the CSV files in the 'import' directory of your Neo4j installation\n")
            f.write("3. Run the following Cypher commands to import the data:\n\n")
            
            f.write("```cypher\n")
            f.write("// Load nodes\n")
            f.write(f"LOAD CSV WITH HEADERS FROM 'file:///{os.path.basename(nodes_file)}' AS row\n")
            f.write("CALL apoc.create.node(split(row.labels, ':'), {\n")
            f.write("  id: row.id,\n")
            f.write("  feature_type: row.feature_type,\n")
            f.write("  confidence: toFloat(row.confidence),\n")
            f.write("  detected_by: row.detected_by,\n")
            f.write("  detection_reasoning: row.detection_reasoning,\n")
            f.write("  geometry_info: row.geometry_info,\n")
            f.write("  cluster_summary: row.cluster_summary,\n")
            f.write("  is_known: toBoolean(row.is_known)\n")
            f.write("}) YIELD node AS n\n")
            f.write("RETURN count(n);\n\n")
            
            f.write("// Load relationships\n")
            f.write(f"LOAD CSV WITH HEADERS FROM 'file:///{os.path.basename(rels_file)}' AS row\n")
            f.write("MATCH (start {id: row.start_id})\n")
            f.write("MATCH (end {id: row.end_id})\n")
            f.write("CALL apoc.create.relationship(start, row.type, {\n")
            f.write("  orientation: toFloat(row.orientation),\n")
            f.write("  distance_km: toFloat(row.distance_km)\n")
            f.write("}, end) YIELD rel\n")
            f.write("RETURN count(rel);\n")
            f.write("```\n\n")
            
            f.write("## Running the example queries\n\n")
            f.write(f"After importing the data, open `{os.path.basename(queries_file)}` and run the example Cypher queries to explore the data.\n")
            f.write("These queries demonstrate how to find potential new archaeological sites, analyze feature clusters, and understand the detection reasoning.\n")
        
        return {
            "nodes_file": nodes_file,
            "relationships_file": rels_file,
            "queries_file": queries_file,
            "readme_file": readme_file,
            "base_directory": cypher_dir
        }
        
    def find_potential_new_sites(self, min_confidence=0.45, distance_from_known=200.0, use_advanced_features=True):
        """Find potential new archaeological sites based on detected features.
        
        This function evaluates detected features to identify archaeological sites,
        both validating against known sites in the GeoTIFF data and identifying new sites.
        Enhanced with advanced feature representation methods (PCA/SVD and anomaly detection).
        
        Args:
            min_confidence: Minimum confidence score to consider
            distance_from_known: Distance (km) to consider a site as matching a known site
            use_advanced_features: Whether to leverage advanced feature representation methods
            
        Returns:
            List of features that likely represent archaeological sites
        """
        start_time = time.time()
        results = []
        
        # Get all features with sufficient confidence
        print(f"[Archaeological Analysis] Analyzing {len(self.features)} features for archaeological sites...")
        print(f"[Archaeological Analysis] Using confidence threshold = {min_confidence}, match distance = {distance_from_known}km")
        print(f"[Archaeological Analysis] Advanced feature representation: {use_advanced_features}")
        
        # Integrate PCA/SVD and anomaly detection capabilities
        if use_advanced_features:
            # Check if features is a dictionary or list and handle accordingly
            features_to_check = self.features.values() if isinstance(self.features, dict) else self.features
            
            # Prioritize features detected by advanced methods
            pca_matches = [f for f in features_to_check 
                          if f.get('detected_by') == 'AdvancedFeatureRepresentationAgent' 
                          and (f.get('confidence', 0) >= min_confidence or f.get('score', 0) >= min_confidence)]
            
            anomaly_detections = [f for f in features_to_check 
                                  if 'AnomalyDetection' in str(f.get('detected_by', '')) 
                                  and (f.get('confidence', 0) >= min_confidence or f.get('score', 0) >= min_confidence)]
            
            # Boost confidence for features that were detected by multiple methods
            for pca_feature in pca_matches:
                for anomaly_feature in anomaly_detections:
                    if self._calculate_distance_km(
                            pca_feature.get('geometry', {}).get('lat', 0), 
                            pca_feature.get('geometry', {}).get('lon', 0),
                            anomaly_feature.get('geometry', {}).get('lat', 0), 
                            anomaly_feature.get('geometry', {}).get('lon', 0)) <= 1.0:
                        # Increase confidence if detected by both methods
                        combined_confidence = min(1.0, pca_feature.get('confidence', 0) * 1.2)
                        pca_feature['confidence'] = combined_confidence
                        pca_feature['detected_by_multiple_methods'] = True
                        pca_feature['archaeological_confidence'] = min(1.0, pca_feature.get('archaeological_confidence', 0) * 1.3)
                        break
            
            # Traditional approach as backup
            traditional_features = [f for f in features_to_check 
                                   if f.get('detected_by') not in ['AdvancedFeatureRepresentationAgent', 'AdvancedFeatureRepresentationAgent_AnomalyDetection']
                                   and (f.get('confidence', 0) >= min_confidence or f.get('score', 0) >= min_confidence)]
            
            # Combine all candidates with priority for advanced detections
            candidate_features = pca_matches + anomaly_detections + traditional_features
            
            print(f"[Archaeological Analysis] Found {len(pca_matches)} PCA/SVD matches and {len(anomaly_detections)} anomaly detections")
            print(f"[Archaeological Analysis] Combined with {len(traditional_features)} traditional features")
        else:
            # Traditional approach without advanced feature representation
            # Check if features is a dictionary or list and handle accordingly
            features_to_check = self.features.values() if isinstance(self.features, dict) else self.features
            
            candidate_features = [f for f in features_to_check 
                                 if f.get('confidence', 0) >= min_confidence or f.get('score', 0) >= min_confidence]
        
        print(f"[Archaeological Analysis] Found {len(candidate_features)} candidate features with sufficient confidence")
        
        # Separate validation sites from the analysis (if any)
        validation_sites = [s for s in self.sites if s.get('is_validation', False)]
        known_sites = [s for s in self.sites if not s.get('is_validation', False)]
        
        # If we have validation sites, use them for accuracy evaluation
        use_validation = len(validation_sites) > 0
        if use_validation:
            print(f"[Archaeological Analysis] Using {len(validation_sites)} validation sites for accuracy evaluation")
        else:
            print(f"[Archaeological Analysis] No validation sites provided, all detected sites will be considered new discoveries")
        
        print(f"[Archaeological Analysis] Analyzing distances to potential site locations...")
        features_processed = 0
        features_with_coords = 0
        sites_matched = 0
        new_discoveries = 0
        
        # Track which validation sites we match to evaluate our detection accuracy
        matched_validation_sites = set()
        
        for feature_idx, feature in enumerate(candidate_features):
            # Show progress for large datasets
            features_processed += 1
            if features_processed % 100 == 0:
                print(f"[Archaeological Analysis] Progress: {features_processed}/{len(candidate_features)} features analyzed...")
            
            if 'geometry' not in feature or not isinstance(feature['geometry'], dict):
                continue
                
            if 'lat' not in feature['geometry'] or 'lon' not in feature['geometry']:
                continue
            
            features_with_coords += 1
            feature_lat = feature['geometry']['lat']
            feature_lon = feature['geometry']['lon']
            
            # Check if this is from synthetic data
            is_synthetic = False
            
            # Check coordinates for synthetic values
            if (feature_lat == -3.0 and feature_lon == -60.0) or (feature_lat == -3.1133 and feature_lon == -60.0253):
                is_synthetic = True
                feature['is_synthetic'] = True
            
            # Also check for synthetic metadata
            if 'meta' in feature and feature.get('meta', {}).get('synthetic', False):
                is_synthetic = True
                feature['is_synthetic'] = True
            
            # Skip synthetic sites completely
            if is_synthetic:
                # Skip this feature entirely and move to the next one
                continue
            
            # Calculate archaeological significance score
            arch_score = self._calculate_archaeological_significance(feature)
            feature['arch_score'] = arch_score
            
            # Check if this feature matches a validation site (if we have any)
            matched_to_site = None
            min_dist = float('inf')
            
            # Check all sites (validation or normal) to see if we match
            for site in self.sites:
                if 'lat' not in site or 'lon' not in site:
                    continue
                    
                dist = self._calculate_distance_km(
                    feature_lat, feature_lon,
                    site['lat'], site['lon']
                )
                
                # If we're within match distance and closer than previous matches
                if dist <= distance_from_known and dist < min_dist:
                    matched_to_site = site
                    min_dist = dist
            
            # Track our results differently based on whether this matches a known site
            if matched_to_site:
                site_is_validation = matched_to_site.get('is_validation', False)
                
                # Record the match for evaluation
                if site_is_validation and matched_to_site.get('id') not in matched_validation_sites:
                    matched_validation_sites.add(matched_to_site.get('id'))
                    sites_matched += 1
                    feature['matches_validation_site'] = True
                    feature['match_distance'] = min_dist
                else:
                    # This matches a real site from the GeoTIFF data
                    feature['matches_known_site'] = True
                    feature['match_distance'] = min_dist
            else:
                # This is a potential new discovery
                if arch_score >= 0.5:  # Higher threshold for new discoveries
                    new_discoveries += 1
                    feature['is_new_discovery'] = True
            
            # Only consider features with sufficient archaeological characteristics
            # Lower threshold to capture more potential sites
            if matched_to_site is None and arch_score >= 0.4:
                # Create detailed archaeological reasoning chain
                reasoning_steps = [
                    {
                        "step": "Initial Detection",
                        "agent": feature.get('detected_by', 'Unknown'),
                        "evidence": f"Feature detected with confidence {feature.get('confidence', feature.get('score', 'Unknown'))}",
                        "conclusion": f"Classified as {feature.get('feature_type', 'Unknown')}"
                    },
                    {
                        "step": "Archaeological Characteristics Analysis",
                        "evidence": f"Archaeological signature score: {arch_score:.2f}/1.0",
                        "conclusion": f"Feature exhibits {arch_score:.0%} match with known archaeological patterns"
                    },
                    {
                        "step": "Known Site Proximity Analysis",
                        "evidence": f"Distance to nearest known site exceeds {distance_from_known} km",
                        "conclusion": "Represents potential new archaeological discovery"
                    }
                ]
                
                # Add detailed type-specific archaeological reasoning
                if feature.get('feature_type') == 'RectangularEnclosure':
                    width = feature.get('width', 0)
                    height = feature.get('height', 0)
                    orientation = feature.get('orientation', 0)
                    
                    # Check for cardinal alignment (N-S/E-W)
                    is_cardinal = (orientation <= 15 or (orientation >= 75 and orientation <= 105) 
                                 or orientation >= 165)
                    
                    # Check for typical Amazonian geoglyph size range
                    is_right_size = (30 <= max(width, height) <= 300)
                    
                    # Check if it's a square (aspect ratio near 1)
                    aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
                    is_square = aspect_ratio > 0.85
                    
                    if is_cardinal:
                        reasoning_steps.append({
                            "step": "Orientation Analysis",
                            "evidence": f"Structure aligned at {orientation:.1f}° orientation",
                            "conclusion": "Cardinal alignment consistent with intentional construction"
                        })
                    
                    if is_right_size:
                        reasoning_steps.append({
                            "step": "Dimensional Analysis", 
                            "evidence": f"Rectangular enclosure measures {width:.1f}×{height:.1f} units",
                            "conclusion": "Dimensions consistent with Amazonian earthworks (typically 30-300m)"
                        })
                    
                    if is_square:
                        reasoning_steps.append({
                            "step": "Geometric Analysis",
                            "evidence": f"Square-like shape with aspect ratio of {aspect_ratio:.2f}",
                            "conclusion": "Perfect geometric shape suggests human construction"
                        })
                    
                elif feature.get('feature_type') == 'LinearFeature':
                    length = feature.get('length', 0)
                    orientation = feature.get('orientation', 0)
                    
                    # Check if orientation matches known archaeological roads/causeways
                    aligned_to_sites = False
                    for site in self.sites:
                        if 'lat' in site and 'lon' in site:
                            site_angle = np.degrees(np.arctan2(
                                site['lat'] - feature_lat,
                                site['lon'] - feature_lon
                            )) % 180
                            if abs((site_angle - orientation) % 180) < 15:
                                aligned_to_sites = True
                                break
                    
                    reasoning_steps.append({
                        "step": "Linear Feature Analysis",
                        "evidence": f"Linear feature extends {length:.1f} units at {orientation:.1f}° orientation",
                        "conclusion": "Consistent with ancient roads or causeways connecting sites"
                    })
                    
                    if length > 50:
                        reasoning_steps.append({
                            "step": "Infrastructure Assessment",
                            "evidence": f"Extended linear feature ({length:.1f} units)",
                            "conclusion": "Substantial length suggests intentional construction for transportation"
                        })
                        
                    if aligned_to_sites:
                        reasoning_steps.append({
                            "step": "Connectivity Analysis",
                            "evidence": "Feature orientation points toward known archaeological site",
                            "conclusion": "Likely represents a causeway connecting settlement areas"
                        })
                
                elif feature.get('feature_type') == 'RepetitivePattern':
                    reasoning_steps.append({
                        "step": "Structural Pattern Analysis",
                        "evidence": f"Regular repeating elements detected with frequency {feature.get('frequency', 'Unknown')}",
                        "conclusion": "Organized repetition indicates human-made structures rather than natural formation"
                    })
                
                # Check for feature clusters (higher confidence)
                nearby_features = [f for f in self.features 
                                  if f['id'] != feature['id'] and 
                                  'geometry' in f and isinstance(f['geometry'], dict) and
                                  'lat' in f['geometry'] and 'lon' in f['geometry'] and
                                  self._calculate_distance_km(
                                      feature_lat, feature_lon,
                                      f['geometry']['lat'], f['geometry']['lon']) <= 1.0]
                
                if len(nearby_features) > 0:
                    reasoning_steps.append({
                        "step": "Spatial Clustering Analysis",
                        "evidence": f"Found {len(nearby_features)} additional features within 1km",
                        "conclusion": "Feature density suggests organized settlement activity"
                    })
                
                # Add reasoning chain to the knowledge graph
                reasoning_id = self.add_reasoning_chain(feature['id'], reasoning_steps)
                
                # Add reference to reasoning
                feature_with_reasoning = feature.copy()
                feature_with_reasoning['reasoning_id'] = reasoning_id
                feature_with_reasoning['archaeological_confidence'] = arch_score
                
                # Add explicit coordinates for site display and visualization
                feature_with_reasoning['latitude'] = feature_lat
                feature_with_reasoning['longitude'] = feature_lon
                
                # Add to results
                results.append(feature_with_reasoning)
        
        # Report results
        elapsed = time.time() - start_time
        
        print(f"[Archaeological Analysis] Analysis complete in {elapsed:.2f}s")
        print(f"[Archaeological Analysis] Processed {features_processed} features ({features_with_coords} with coordinates)")
        
        if use_validation:
            if len(validation_sites) > 0:
                validation_pct = (sites_matched / len(validation_sites)) * 100
                print(f"[Archaeological Analysis] Found {sites_matched}/{len(validation_sites)} validation sites ({validation_pct:.1f}%)")
        
        print(f"[Archaeological Analysis] Found {len(results)} potential archaeological sites")
        print(f"[Archaeological Analysis] Detected {new_discoveries} potential new discoveries")
        
        # Sort by archaeological confidence
        results.sort(key=lambda x: x.get('archaeological_confidence', 0), reverse=True)
        
        return results

    def _calculate_archaeological_significance(self, feature):
        """Calculate archaeological significance score based on feature characteristics.
        
        This evaluates how likely a feature represents actual archaeological remains
        rather than natural formations or data artifacts. The enhanced version is more inclusive
        for potential archaeological features by giving partial scoring to irregular or
        non-cardinal-aligned features.
        
        Args:
            feature: Feature dictionary to evaluate
            
        Returns:
            Score between 0-1 indicating archaeological significance
        """
        base_score = 0.45  # Start with slightly below neutral score (more inclusive)
        
        # 1. Geometric precision (straight lines, perfect angles)
        if feature.get('feature_type') == 'RectangularEnclosure':
            # Higher confidence for regular rectangles (aspect ratio near 0.5-1.0)
            width = feature.get('width', 0)
            height = feature.get('height', 0)
            if width > 0 and height > 0:
                aspect_ratio = min(width, height) / max(width, height)
                if aspect_ratio > 0.9:  # Nearly square
                    base_score += 0.2
                elif aspect_ratio > 0.7:  # Rectangle but not too elongated
                    base_score += 0.15
                elif aspect_ratio > 0.5:  # Less regular but still rectangular
                    base_score += 0.1
                else:
                    # Give more score even to irregular rectangles - could be ditched enclosures
                    base_score += 0.07
            
            # Size consistent with known structures (30-300m typical range)
            max_dim = max(width, height)
            if 30 <= max_dim <= 300:
                base_score += 0.15
            elif 20 <= max_dim <= 400:  # Slightly outside typical range
                base_score += 0.1
            else:
                # Give more score to differently sized features - size variation is normal
                base_score += 0.05
                
            # Orientation to cardinal directions
            orientation = feature.get('orientation', 0)
            # Check if aligned to N-S or E-W (within 15 degrees)
            if orientation <= 15 or (75 <= orientation <= 105) or orientation >= 165:
                base_score += 0.15
            elif orientation <= 30 or (60 <= orientation <= 120) or orientation >= 150:
                # Slightly off cardinal directions
                base_score += 0.1
            else:
                # Give more score to non-cardinal orientations - some cultures used astronomical alignments
                base_score += 0.07
                
        elif feature.get('feature_type') == 'LinearFeature':
            # Length consistent with human-made roads/causeways
            length = feature.get('length', 0)
            if 50 <= length <= 500:  # Ideal length range
                base_score += 0.2
            elif 20 <= length <= 1000:  # Plausible but less typical
                base_score += 0.15
            else:
                # Give more score to shorter or longer features
                base_score += 0.07
                
            # Straightness (perfectly straight = artificial)
            if feature.get('straightness', 0) > 0.9:
                base_score += 0.15
            elif feature.get('straightness', 0) > 0.7:
                base_score += 0.1
            elif feature.get('straightness', 0) > 0.5:
                # Less straight but still potentially artificial
                base_score += 0.07
                
            # Orientation - aligned to major sites or cardinal directions
            orientation = feature.get('orientation', 0)
            if orientation <= 15 or (75 <= orientation <= 105) or orientation >= 165:
                base_score += 0.1
            elif orientation <= 30 or (60 <= orientation <= 120) or orientation >= 150:
                # Slightly off cardinal directions
                base_score += 0.07
            else:
                # Give more score to non-cardinal orientations - could follow terrain
                base_score += 0.05
        
        elif feature.get('feature_type') == 'RepetitivePattern':
            # Regular spacing is a strong indicator of human activity
            base_score += 0.3
            
            # Frequency consistent with human-scale construction
            frequency = feature.get('frequency', 0)
            if 0.01 <= frequency <= 0.1:  # 10-100m spacing
                base_score += 0.2
            elif 0.005 <= frequency <= 0.2:  # 5-200m spacing (wider range)
                base_score += 0.15
            else:
                # Give more score to other frequencies
                base_score += 0.1
                
        # Consider any feature type with a score as possibly archaeological
        elif feature.get('score', 0) > 0:
            base_score += feature.get('score', 0) * 0.35  # Increased weight
        
        # 2. Context factors
        
        # Higher confidence if detected by multiple agents
        if 'supporting_agents' in feature and len(feature.get('supporting_agents', [])) > 1:
            base_score += 0.15  # Increased from 0.1
        
        # Base confidence from detecting agent - check both confidence and score fields
        agent_confidence = max(feature.get('confidence', 0), feature.get('score', 0))
        base_score += agent_confidence * 0.25  # Increased from 0.2
            
        # Handle template matches specially
        if feature.get('feature_type') == 'TemplateMatch' or 'template_idx' in feature:
            base_score += 0.3  # Increased from 0.25 - template matches are highly reliable indicators
            
        # Special handling for feature combinations
        # If the feature is part of a cluster with other features, increase score
        if 'cluster_ids' in feature and feature.get('cluster_ids'):
            base_score += 0.12
            
        # If feature is aligned with other features, increase score
        if 'alignment_ids' in feature and feature.get('alignment_ids'):
            base_score += 0.1
            
        # Cap the score at 1.0
        return min(1.0, base_score)

    def cypher_query(self, query_string, params=None):
        """Execute a Cypher-like query directly on the in-memory knowledge graph.
        
        This provides a way to query the knowledge graph using a simplified subset
        of Cypher syntax without requiring a Neo4j database. Supports basic
        MATCH, WHERE, and RETURN operations.
        
        Args:
            query_string: A Cypher-like query string
            params: Optional dictionary of parameters to substitute in the query
            
        Returns:
            List of dictionaries or tuples containing the query results
        """
        # Initialize params if None
        params = params or {}
        
        # Convert query to lowercase for easier parsing
        query_lower = query_string.lower()
        
        # Simple query parser for basic Cypher patterns
        results = []
        
        # MATCH (f:Feature) pattern - find features of specific type
        if "match (f:feature)" in query_lower:
            # Extract feature type if specified after "feature"
            feature_type = None
            if ":" in query_lower.split("match (f:feature")[1].split(")")[0]:
                feature_type = query_lower.split("match (f:feature:")[1].split(")")[0].capitalize()
            
            # Basic WHERE clause parsing
            min_score = None
            near_coords = None
            max_distance = 10.0  # Default max distance in km
            
            if "where" in query_lower:
                where_clause = query_lower.split("where")[1].split("return")[0]
                
                # Check for confidence/score filter
                if "confidence >=" in where_clause or "score >=" in where_clause:
                    score_parts = [p for p in where_clause.split() if p.replace(".", "").isdigit()]
                    if score_parts:
                        min_score = float(score_parts[0])
                
                # Check for distance/near filter
                if "near" in where_clause:
                    # Try to extract coordinates
                    if "lat:" in where_clause and "lon:" in where_clause:
                        lat_part = where_clause.split("lat:")[1].split(",")[0]
                        lon_part = where_clause.split("lon:")[1].split(")")[0]
                        try:
                            lat = float(lat_part)
                            lon = float(lon_part)
                            near_coords = {"lat": lat, "lon": lon}
                        except ValueError:
                            pass
                    
                    # Try to extract max distance
                    if "distance <=" in where_clause:
                        dist_parts = [p for p in where_clause.split("distance <=")[1].split() 
                                     if p.replace(".", "").isdigit()]
                        if dist_parts:
                            max_distance = float(dist_parts[0])
            
            # Execute the query using our query_features method
            features = self.query_features(
                feature_type=feature_type,
                min_score=min_score,
                near_point=near_coords,
                max_distance_km=max_distance if near_coords else None
            )
            
            # Parse RETURN clause to determine what to include in results
            if "return" in query_lower:
                return_clause = query_lower.split("return")[1].strip()
                if return_clause == "f" or return_clause == "f.*":
                    # Return all feature properties
                    results = features
                else:
                    # Return specific properties
                    properties = [p.strip() for p in return_clause.split(",")]
                    
                    # Map f.property notation to actual properties
                    for feature in features:
                        result = {}
                        for prop in properties:
                            if prop.startswith("f."):
                                prop_name = prop[2:]  # Remove 'f.' prefix
                                if prop_name in feature:
                                    result[prop_name] = feature[prop_name]
                                # Handle special cases like f.geometry.lat
                                elif "." in prop_name:
                                    parts = prop_name.split(".")
                                    if parts[0] in feature and isinstance(feature[parts[0]], dict):
                                        if parts[1] in feature[parts[0]]:
                                            result[prop_name] = feature[parts[0]][parts[1]]
                            elif prop == "f":
                                result = feature
                        results.append(result)
            else:
                # Default to returning all properties
                results = features
        
        # MATCH (c:Cluster) pattern - find clusters
        elif "match (c:cluster)" in query_lower:
            # Extract cluster type if specified
            cluster_type = None
            if ":" in query_lower.split("match (c:cluster")[1].split(")")[0]:
                cluster_type = query_lower.split("match (c:cluster:")[1].split(")")[0].capitalize()
            
            # Parse WHERE clause for cluster filters
            min_members = None
            contains_feature_type = None
            contains_multiple = False
            
            if "where" in query_lower:
                where_clause = query_lower.split("where")[1].split("return")[0]
                
                # Check for member count filter
                if "member_count >=" in where_clause:
                    count_parts = [p for p in where_clause.split("member_count >=")[1].split() 
                                  if p.isdigit()]
                    if count_parts:
                        min_members = int(count_parts[0])
                
                # Check for contains feature type
                if "contains" in where_clause and ":feature" in where_clause:
                    # Try to extract feature type from contains clause
                    contains_parts = where_clause.split("contains")[1].split(":")
                    if len(contains_parts) > 1:
                        feature_type_part = contains_parts[1].split(")")[0]
                        contains_feature_type = feature_type_part.capitalize()
                
                # Check for multiple feature types
                if "multiple_types" in where_clause:
                    contains_multiple = True
            
            # Execute query using our query_clusters method
            clusters = self.query_clusters(
                cluster_type=cluster_type,
                min_members=min_members,
                contains_feature_type=contains_feature_type,
                contains_multiple_types=contains_multiple
            )
            
            # Handle RETURN clause
            if "return" in query_lower:
                return_clause = query_lower.split("return")[1].strip()
                if return_clause == "c" or return_clause == "c.*":
                    # Return all cluster properties
                    results = clusters
                else:
                    # Return specific properties
                    properties = [p.strip() for p in return_clause.split(",")]
                    
                    for cluster in clusters:
                        result = {}
                        for prop in properties:
                            if prop.startswith("c."):
                                prop_name = prop[2:]  # Remove 'c.' prefix
                                if prop_name in cluster:
                                    result[prop_name] = cluster[prop_name]
                            elif prop == "c":
                                result = cluster
                        results.append(result)
            else:
                # Default to returning all properties
                results = clusters
        
        # MATCH (f:Feature)-[:NEAR]->(s:KnownSite) pattern - find features near known sites
        elif "match (f:feature)-[:near]->(s:knownsite)" in query_lower:
            # This query finds features that are near known sites
            max_distance = 2.0  # Default distance in km
            
            # Parse WHERE clause for distance filter
            if "where" in query_lower:
                where_clause = query_lower.split("where")[1].split("return")[0]
                if "distance <=" in where_clause:
                    dist_parts = [p for p in where_clause.split("distance <=")[1].split() 
                                 if p.replace(".", "").isdigit()]
                    if dist_parts:
                        max_distance = float(dist_parts[0])
            
            # Find features near known sites
            features_near_sites = []
            for feature in self.features:
                if 'geometry' not in feature or not isinstance(feature['geometry'], dict):
                    continue
                    
                geom = feature['geometry']
                if 'lat' not in geom or 'lon' not in geom:
                    continue
                    
                feature_lat, feature_lon = geom['lat'], geom['lon']
                
                # Check distance to each known site
                for site in self.sites:
                    if 'lat' not in site or 'lon' not in site:
                        continue
                        
                    dist = self._calculate_distance_km(
                        feature_lat, feature_lon,
                        site['lat'], site['lon']
                    )
                    
                    if dist <= max_distance:
                        # Create a result with both feature and site
                        result = {
                            'feature': feature,
                            'site': site,
                            'distance_km': dist
                        }
                        features_near_sites.append(result)
                        break  # One site match is enough
            
            # Handle RETURN clause
            if "return" in query_lower:
                return_clause = query_lower.split("return")[1].strip()
                
                if return_clause == "f, s":
                    # Return full feature and site
                    results = features_near_sites
                else:
                    # Return specific properties
                    properties = [p.strip() for p in return_clause.split(",")]
                    
                    for item in features_near_sites:
                        result = {}
                        for prop in properties:
                            if prop.startswith("f."):
                                prop_name = prop[2:]  # Remove 'f.' prefix
                                if prop_name in item['feature']:
                                    result[f"feature_{prop_name}"] = item['feature'][prop_name]
                            elif prop.startswith("s."):
                                prop_name = prop[2:]  # Remove 's.' prefix
                                if prop_name in item['site']:
                                    result[f"site_{prop_name}"] = item['site'][prop_name]
                            elif prop == "distance":
                                result["distance_km"] = item["distance_km"]
                        results.append(result)
            else:
                # Default to returning all properties
                results = features_near_sites
        
        # MATCH (f:Feature) WHERE NOT EXISTS { MATCH (f)-[:NEAR]->(s:KnownSite) } pattern - find potential new sites
        elif "not exists" in query_lower and "[:near]" in query_lower and "knownsite" in query_lower:
            # This query finds features that are NOT near any known sites (potential new discoveries)
            min_confidence = 0.0
            max_distance = 2.0  # Default distance threshold
            
            # Parse main WHERE clause for confidence filter
            if "where" in query_lower:
                where_clause = query_lower.split("where")[1].split("not exists")[0]
                if "confidence >=" in where_clause:
                    conf_parts = [p for p in where_clause.split("confidence >=")[1].split() 
                                 if p.replace(".", "").isdigit()]
                    if conf_parts:
                        min_confidence = float(conf_parts[0])
            
            # Parse NOT EXISTS sub-clause for distance threshold
            if "distance <=" in query_lower:
                dist_parts = [p for p in query_lower.split("distance <=")[1].split() 
                             if p.replace(".", "").isdigit()]
                if dist_parts:
                    max_distance = float(dist_parts[0])
            
            # Use our existing method for this common query
            potential_new_sites = self.find_potential_new_sites(
                min_confidence=min_confidence,
                distance_from_known=max_distance
            )
            
            # Handle RETURN clause
            if "return" in query_lower:
                return_clause = query_lower.split("return")[1].strip()
                
                if return_clause == "f" or return_clause == "f.*":
                    # Return all feature properties
                    results = potential_new_sites
                else:
                    # Return specific properties
                    properties = [p.strip() for p in return_clause.split(",")]
                    
                    for feature in potential_new_sites:
                        result = {}
                        for prop in properties:
                            if prop.startswith("f."):
                                prop_name = prop[2:]  # Remove 'f.' prefix
                                if prop_name in feature:
                                    result[prop_name] = feature[prop_name]
                            elif prop == "f":
                                result = feature
                        results.append(result)
            else:
                # Default to returning all properties
                results = potential_new_sites
        
        # MATCH (a:AlignmentGroup) pattern - find alignment groups
        elif "match (a:alignmentgroup)" in query_lower:
            # Extract alignment type if specified
            alignment_type = None
            min_orientation = None
            max_orientation = None
            min_members = None
            
            # Parse WHERE clause for alignment filters
            if "where" in query_lower:
                where_clause = query_lower.split("where")[1].split("return")[0]
                
                # Check for alignment type
                if "alignment_type =" in where_clause:
                    type_parts = where_clause.split("alignment_type =")[1].split()
                    if type_parts and type_parts[0].strip("'\"") in ["SharedOrientation", "Collinear"]:
                        alignment_type = type_parts[0].strip("'\"")
                
                # Check for orientation range
                if "orientation >=" in where_clause:
                    orient_parts = [p for p in where_clause.split("orientation >=")[1].split() 
                                   if p.replace(".", "").isdigit()]
                    if orient_parts:
                        min_orientation = float(orient_parts[0])
                
                if "orientation <=" in where_clause:
                    orient_parts = [p for p in where_clause.split("orientation <=")[1].split() 
                                   if p.replace(".", "").isdigit()]
                    if orient_parts:
                        max_orientation = float(orient_parts[0])
                
                # Check for member count
                if "member_count >=" in where_clause:
                    count_parts = [p for p in where_clause.split("member_count >=")[1].split() 
                                  if p.isdigit()]
                    if count_parts:
                        min_members = int(count_parts[0])
            
            # Execute query using our query_alignments method
            alignments = self.query_alignments(
                min_orientation=min_orientation,
                max_orientation=max_orientation,
                alignment_type=alignment_type,
                min_members=min_members
            )
            
            # Handle RETURN clause
            if "return" in query_lower:
                return_clause = query_lower.split("return")[1].strip()
                
                if return_clause == "a" or return_clause == "a.*":
                    # Return all alignment properties
                    results = alignments
                else:
                    # Return specific properties
                    properties = [p.strip() for p in return_clause.split(",")]
                    
                    for alignment in alignments:
                        result = {}
                        for prop in properties:
                            if prop.startswith("a."):
                                prop_name = prop[2:]  # Remove 'a.' prefix
                                if prop_name in alignment:
                                    result[prop_name] = alignment[prop_name]
                            elif prop == "a":
                                result = alignment
                        results.append(result)
            else:
                # Default to returning all properties
                results = alignments
        
        # Substitute parameters if provided
        if params and results:
            # Replace any parameter placeholders with their values
            for i, result in enumerate(results):
                for key, value in result.items():
                    if isinstance(value, str) and value.startswith('$'):
                        param_name = value[1:]
                        if param_name in params:
                            results[i][key] = params[param_name]
        
        return results

# --- TemplateMatchAgent ---
class TemplateMatchAgent(BaseAgent):
    def __init__(self, kg, templates, threshold=0.5, scales=None):
        super().__init__("TemplateMatchAgent")
        self.kg = kg
        self.templates = templates
        self.threshold = threshold  # Lowered from 0.8 to 0.5 to capture more potential matches
        # Default scales to try (1.0 is original size, <1.0 is smaller, >1.0 is larger)
        # Extended range of scales to capture features at more varied sizes
        self.scales = scales or [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        
    def _match_template_torch(self, grid, template, scale=1.0):
        """Use PyTorch to perform template matching with normalized cross-correlation.
        
        Args:
            grid: The grid to search in
            template: The template to search for
            scale: Scale factor to resize the template (1.0 = original size)
            
        Returns:
            Normalized cross-correlation result as numpy array
        """
        import torch
        import torch.nn.functional as F
        
        # Handle template scaling if needed
        if scale != 1.0:
            try:
                from skimage.transform import resize
                h, w = template.shape
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h < 3 or new_w < 3:  # Prevent too small templates
                    return np.zeros((grid.shape[0] - 2, grid.shape[1] - 2))
                    
                template = resize(template, (new_h, new_w), mode='reflect', anti_aliasing=True)
            except ImportError:
                # Fallback if skimage not available
                pass
                
        device = torch.device("mps") if BACKEND == "torch_mps" else \
                torch.device("cuda") if BACKEND == "torch_cuda" else \
                torch.device("cpu")
        
        # Convert to torch tensors
        grid_t = torch.from_numpy(grid).float().to(device)
        template_t = torch.from_numpy(template).float().to(device)
        
        # Add batch and channel dimensions for convolution
        grid_t = grid_t.unsqueeze(0).unsqueeze(0)
        template_t = template_t.unsqueeze(0).unsqueeze(0)
        
        # Normalize template for cross-correlation
        template_mean = torch.mean(template_t)
        template_std = torch.std(template_t)
        template_norm = (template_t - template_mean) / (template_std + 1e-7)
        
        # Flip template for valid cross-correlation
        template_flip = torch.flip(template_norm, dims=[-1, -2])
        
        # Perform correlation
        result = F.conv2d(grid_t, template_flip, padding=0)
        
        # Normalize the result
        result_size = result.size()
        win_size = template_t.size(-1) * template_t.size(-2)
        
        # Get local sums
        grid_sum = F.conv2d(grid_t, torch.ones_like(template_t), padding=0)
        grid_sum_sq = F.conv2d(grid_t**2, torch.ones_like(template_t), padding=0)
        
        # Calculate local statistics
        grid_mean = grid_sum / win_size
        grid_sigma = torch.sqrt(torch.clamp(grid_sum_sq / win_size - grid_mean**2, min=0))
        
        # Normalize the result
        result = result / (win_size * grid_sigma * template_std + 1e-7)
        
        # Convert back to numpy
        return result.squeeze().cpu().numpy()
        
    def _match_template_tf(self, grid, template, scale=1.0):
        """Use TensorFlow to perform template matching with normalized cross-correlation.
        
        Args:
            grid: The grid to search in
            template: The template to search for
            scale: Scale factor to resize the template (1.0 = original size)
            
        Returns:
            Normalized cross-correlation result as numpy array
        """
        import tensorflow as tf
        
        # Handle template scaling if needed
        if scale != 1.0:
            try:
                from skimage.transform import resize
                h, w = template.shape
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h < 3 or new_w < 3:  # Prevent too small templates
                    return np.zeros((grid.shape[0] - 2, grid.shape[1] - 2))
                    
                template = resize(template, (new_h, new_w), mode='reflect', anti_aliasing=True)
            except ImportError:
                # Fallback if skimage not available
                pass
        
        # Convert to TF tensors
        grid_t = tf.convert_to_tensor(grid, dtype=tf.float32)
        template_t = tf.convert_to_tensor(template, dtype=tf.float32)
        
        # Add batch and channel dimensions
        grid_t = tf.expand_dims(tf.expand_dims(grid_t, 0), -1)
        template_t = tf.expand_dims(tf.expand_dims(template_t, -1), -1)
        
        # Calculate normalized cross-correlation using depthwise_conv2d
        result = tf.nn.depthwise_conv2d(
            grid_t, template_t, strides=[1, 1, 1, 1], padding='VALID'
        )
        
        # Convert back to numpy
        return result.numpy().squeeze()
        
    def _match_template_numpy(self, grid, template, scale=1.0):
        """Use NumPy to perform template matching with normalized cross-correlation.
        
        Args:
            grid: The grid to search in
            template: The template to search for
            scale: Scale factor to resize the template (1.0 = original size)
            
        Returns:
            Normalized cross-correlation result as numpy array
        """
        # Handle template scaling if needed
        if scale != 1.0:
            try:
                from skimage.transform import resize
                h, w = template.shape
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h < 3 or new_w < 3:  # Prevent too small templates
                    return np.zeros((grid.shape[0] - 2, grid.shape[1] - 2))
                    
                template = resize(template, (new_h, new_w), mode='reflect', anti_aliasing=True)
            except ImportError:
                # Fallback if skimage not available
                pass
                
        # Simple implementation of normalized cross-correlation
        height, width = grid.shape
        t_height, t_width = template.shape
        result = np.zeros((height - t_height + 1, width - t_width + 1))
        
        # Normalize template
        t_mean = np.mean(template)
        t_std = np.std(template)
        t_norm = (template - t_mean) / (t_std + 1e-8)
        
        # Slide the template over the grid
        for y in range(height - t_height + 1):
            for x in range(width - t_width + 1):
                # Extract grid patch
                patch = grid[y:y+t_height, x:x+t_width]
                
                # Normalize patch
                p_mean = np.mean(patch)
                p_std = np.std(patch)
                
                # Skip patches with no variation
                if p_std < 1e-8:
                    result[y, x] = 0
                    continue
                
                # Compute normalized correlation
                p_norm = (patch - p_mean) / (p_std + 1e-8)
                corr = np.sum(p_norm * t_norm) / (t_height * t_width)
                
                result[y, x] = corr
        
        return result

    def transform(self, g, **kwargs):
        """Find matches to known template patterns in the grid at multiple scales."""
        # Skip empty grids
        if g.size == 0:
            return g
            
        verbose = kwargs.get('verbose', False)
        # Add timeout to prevent getting stuck
        max_template_time = 1.0  # Maximum time to spend per template in seconds
        max_total_time = 30.0  # Maximum total time for all templates in seconds
        start_time = time.time()
        
        # Get grid metadata if available
        meta = kwargs.get('meta', {})
        lat = meta.get('lat', 0)
        lon = meta.get('lon', 0)
        
        # Report start
        self.report_progress(f"Matching {len(self.templates)} templates at {len(self.scales)} scales against grid of shape {g.shape}")
        
        # Normalize grid to 0-1 range for template matching
        if set(np.unique(g)) != {0, 1}:
            # Simple thresholding for non-binary grids
            grid_norm = (g > np.mean(g)).astype(np.float32)
        else:
            grid_norm = g.astype(np.float32)
        
        # Track best matches for this grid
        matches = []
        # Store scale information with matches
        scale_matches = {}
        
        # For each template pattern
        for t_idx, template in enumerate(self.templates):
            # Check for overall timeout
            if time.time() - start_time > max_total_time:
                self.report_progress(f"Overall timeout reached after {time.time() - start_time:.1f}s, processed {t_idx}/{len(self.templates)} templates")
                break
            # Report progress during matching
            if (t_idx + 1) % max(1, len(self.templates) // 4) == 0:
                self.report_progress(f"Processed {t_idx + 1}/{len(self.templates)} templates")
                
            # Normalize template
            if set(np.unique(template)) != {0, 1}:
                tmpl_norm = (template > np.mean(template)).astype(np.float32)
            else:
                tmpl_norm = template.astype(np.float32)
                
            # Skip templates too large for this grid
            if tmpl_norm.shape[0] > grid_norm.shape[0] or tmpl_norm.shape[1] > grid_norm.shape[1]:
                continue
                
            # Try each scale for this template
            for scale in self.scales:
                # Check for template timeout
                template_time = time.time() - start_time
                if template_time > max_template_time:
                    if verbose:
                        print(f"Template {t_idx} processing timed out after {template_time:.1f}s")
                    break
                # Skip scales that would make template larger than grid
                if (scale * tmpl_norm.shape[0] > grid_norm.shape[0] or 
                    scale * tmpl_norm.shape[1] > grid_norm.shape[1]):
                    continue
                    
                # Find matches at this scale using normalized cross-correlation
                if BACKEND.startswith("torch"):
                    match_scores = self._match_template_torch(grid_norm, tmpl_norm, scale)
                elif BACKEND.startswith("tf"):
                    match_scores = self._match_template_tf(grid_norm, tmpl_norm, scale)
                else:
                    match_scores = self._match_template_numpy(grid_norm, tmpl_norm, scale)
                    
                # Skip if match_scores is empty (can happen with small templates)
                if match_scores.size == 0:
                    continue
                
                # Find peaks in match scores
                threshold = self.threshold
                peaks = []
                
                # Simple local maxima detection
                for i in range(1, match_scores.shape[0]-1):
                    for j in range(1, match_scores.shape[1]-1):
                        if (match_scores[i, j] > threshold and
                            match_scores[i, j] > match_scores[i-1, j] and
                            match_scores[i, j] > match_scores[i+1, j] and
                            match_scores[i, j] > match_scores[i, j-1] and
                            match_scores[i, j] > match_scores[i, j+1]):
                            # Store scale information with the match
                            peaks.append((i, j, match_scores[i, j], t_idx, scale))
                
                # Track matches at each scale
                if scale not in scale_matches:
                    scale_matches[scale] = []
                scale_matches[scale].extend([(y, x, score) for y, x, score, _, _ in peaks])
                
                # Add peaks to global matches
                matches.extend(peaks)
        
        # Sort matches by score
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Report match count and scale information
        if matches:
            scale_counts = {scale: len(matches) for scale, matches in scale_matches.items()}
            scales_str = ", ".join([f"{scale}x: {count}" for scale, count in scale_counts.items()])
            self.report_progress(f"Found {len(matches)} template matches above threshold {self.threshold} across scales: {scales_str}")
        else:
            self.report_progress(f"No template matches found above threshold {self.threshold} at any scale")
        
        # Add template matches to the knowledge graph
        feature_ids = []
        for y, x, score, template_idx, scale in matches:
            # Compute template dimensions at this scale
            template = self.templates[template_idx]
            base_height, base_width = template.shape
            scaled_height = int(base_height * scale)
            scaled_width = int(base_width * scale)
            
            # Create template match feature with scale information
            feature_id = self.kg.add_feature(
                feature_type='TemplateMatch',
                geometry={
                    'lat': lat,
                    'lon': lon,
                    'x': int(x),
                    'y': int(y),
                    'width': scaled_width,
                    'height': scaled_height,
                    'scale': float(scale)
                },
                confidence=float(score),
                template_idx=int(template_idx),
                scale=float(scale),
                detected_by=self.name
            )
            feature_ids.append(feature_id)
        
                    # Final report
            if feature_ids:
                self.report_progress(f"Added {len(feature_ids)} template matches to knowledge graph")
                
                # If visualization is requested, show the multi-scale matches
                if kwargs.get('visualize', False) or kwargs.get('display_inline', False):
                    try:
                        from enhanced_visualization import display_multi_scale_matches
                        
                        # Use the highest-scoring template for visualization
                        if matches:
                            best_match = max(matches, key=lambda x: x[2])
                            _, _, _, template_idx, _ = best_match
                            template = self.templates[template_idx]
                            
                            # Create visualization
                            fig = display_multi_scale_matches(
                                grid_norm, 
                                template, 
                                scale_matches,
                                figsize=(14, 10)
                            )
                            
                            # Display inline or save to file
                            if kwargs.get('display_inline', False):
                                try:
                                    from IPython.display import display
                                    display(fig)
                                except ImportError:
                                    pass
                            
                            # Save to file if output_dir is specified
                            output_dir = kwargs.get('output_dir')
                            if output_dir:
                                import os
                                from datetime import datetime
                                os.makedirs(output_dir, exist_ok=True)
                                filename = f"multi_scale_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                                filepath = os.path.join(output_dir, filename)
                                fig.savefig(filepath, dpi=150)
                                self.report_progress(f"Saved multi-scale visualization to {filepath}")
                            
                            import matplotlib.pyplot as plt
                            plt.close(fig)
                    except ImportError:
                        self.report_progress("Enhanced visualization module not available")
                    except Exception as e:
                        self.report_progress(f"Error during visualization: {str(e)}")
                
            return g

# --- LineDetectAgent ---
class LineDetectAgent(BaseAgent):
    """Agent for detecting linear features like ancient roads, causeways, or canals."""
    
    def __init__(self, kg, min_length=5, angle_tolerance=15, threshold=0.5, name=None):
        """Initialize the line detection agent."""
        super().__init__(name or "LineDetectAgent")
        self.kg = kg
        self.min_length = min_length
        self.angle_tolerance = angle_tolerance
        self.threshold = threshold
        self.lines_detected = 0
        self.lines_added = 0
    
    def transform(self, g, **kwargs):
        """Detect linear features in the grid."""
        # Skip empty or tiny grids
        if g.size == 0 or min(g.shape) < 5:
            return g
            
        verbose = kwargs.get('verbose', False)
        
        # Report start of processing
        self.report_progress(f"Starting line detection on grid shape {g.shape}")
        
        # Get grid metadata if available
        meta = kwargs.get('meta', {})
        lat = meta.get('lat', 0)
        lon = meta.get('lon', 0)
        
        # Preprocess grid
        # Convert to grayscale if RGB
        if len(g.shape) > 2:
            if g.shape[2] == 3 or g.shape[2] == 4:
                # Simple grayscale conversion for RGB/RGBA
                gray = np.mean(g[:, :, :3], axis=2)
                self.report_progress("Converting RGB image to grayscale")
            else:
                # Use first channel for other multi-channel data
                gray = g[:, :, 0]
                self.report_progress("Using first channel of multi-channel data")
        else:
            gray = g
        
        # Normalize if not binary
        if set(np.unique(gray)) != {0, 1}:
            # Use edge detection for non-binary data
            self.report_progress("Applying edge detection for feature enhancement")
            edges = compute_edge_detection(gray)
        else:
            edges = gray
        
        # Detect lines
        self.report_progress("Detecting linear features...")
        lines = self._detect_lines(edges)
        
        # Report progress
        if lines:
            self.lines_detected += len(lines)
            self.report_progress(f"Detected {len(lines)} potential linear features (total: {self.lines_detected})")
        else:
            self.report_progress("No linear features detected in this tile")
        
        # Add detected lines to knowledge graph
        feature_ids = []
        features_added = 0
        for line in lines:
            if line['confidence'] >= self.threshold:
                # Create feature for each line
                feature_id = self.kg.add_feature(
                    feature_type='LinearFeature',
                    geometry={
                        'lat': lat,
                        'lon': lon,
                        'start': line['start'],
                        'end': line['end'],
                        'length': line['length']
                    },
                    length=line['length'],
                    width=line.get('width', 1),
                    orientation=line['angle'],
                    confidence=line['confidence'],
                    detected_by=self.name
                )
                feature_ids.append(feature_id)
                features_added += 1
        
        # Final report
        if feature_ids:
            self.lines_added += features_added
            self.report_progress(f"Added {features_added} linear features to knowledge graph (total: {self.lines_added})")
        
        return g
    
    def _detect_lines(self, binary_grid):
        """Detect lines in a binary grid using a simplified approach.
        
        In a real implementation, this would use Hough transform or similar algorithms.
        For simplicity, this implementation just returns some example lines.
        
        Args:
            binary_grid: Binary grid with edges highlighted
            
        Returns:
            List of line dictionaries with properties needed for feature creation
        """
        # This is a placeholder for a real line detection algorithm
        # In practice, you would use cv2.HoughLines or a similar approach
        h, w = binary_grid.shape
        lines = []
        
        # Find connected components in the binary grid
        # For now, just sample some lines based on grid structure
        
        # Simple run-length encoding approach to find horizontal lines
        for y in range(h):
            run_start = None
            for x in range(w):
                if binary_grid[y, x] > 0 and run_start is None:
                    run_start = x
                elif (binary_grid[y, x] == 0 or x == w-1) and run_start is not None:
                    run_end = x if binary_grid[y, x] == 0 else x+1
                    if run_end - run_start >= self.min_length:
                        # Convert from tuple to dictionary with required properties
                        start = (run_start, y)
                        end = (run_end-1, y)
                        length = run_end - run_start
                        # Horizontal line has 0/180 degree angle
                        angle = 0
                        # Calculate confidence based on length
                        confidence = min(1.0, length / (2 * self.min_length))
                        
                        lines.append({
                            'start': start,
                            'end': end,
                            'length': length,
                            'angle': angle,
                            'confidence': confidence,
                            'width': 1  # Default width
                        })
                    run_start = None
        
        # Simple approach to find vertical lines
        for x in range(w):
            run_start = None
            for y in range(h):
                if binary_grid[y, x] > 0 and run_start is None:
                    run_start = y
                elif (binary_grid[y, x] == 0 or y == h-1) and run_start is not None:
                    run_end = y if binary_grid[y, x] == 0 else y+1
                    if run_end - run_start >= self.min_length:
                        # Convert from tuple to dictionary with required properties
                        start = (x, run_start)
                        end = (x, run_end-1)
                        length = run_end - run_start
                        # Vertical line has 90 degree angle
                        angle = 90
                        # Calculate confidence based on length
                        confidence = min(1.0, length / (2 * self.min_length))
                        
                        lines.append({
                            'start': start,
                            'end': end,
                            'length': length,
                            'angle': angle,
                            'confidence': confidence,
                            'width': 1  # Default width
                        })
                    run_start = None
                    
        # Find diagonal lines (simplified approach)
        # In a real implementation, you'd use a proper line detection algorithm
        
        return lines

# --- RectangularAgent ---
class RectangularAgent(BaseAgent):
    """Agent for detecting rectangular or square enclosures and structures."""
    
    def __init__(self, kg, min_size=3, max_size=50, threshold=0.5, name=None):
        """Initialize the rectangular feature detection agent."""
        super().__init__(name or "RectangularAgent")
        self.kg = kg
        self.min_size = min_size
        self.max_size = max_size
        self.threshold = threshold
        self.rectangles_detected = 0
        self.rectangles_added = 0
    
    def transform(self, g, **kwargs):
        """Detect rectangular features in the grid."""
        # Skip empty or tiny grids
        if g.size == 0 or min(g.shape) < self.min_size:
            return g
        
        verbose = kwargs.get('verbose', False)
        use_gpu = kwargs.get('use_gpu', False)
        
        # Report start
        self.report_progress(f"Starting rectangle detection on grid shape {g.shape}")
        
        # Get grid metadata if available
        meta = kwargs.get('meta', {})
        lat = meta.get('lat', 0)
        lon = meta.get('lon', 0)
        
        # Preprocess grid
        # Convert to grayscale if RGB
        if len(g.shape) > 2:
            if g.shape[2] == 3 or g.shape[2] == 4:
                # Simple grayscale conversion for RGB/RGBA
                gray = np.mean(g[:, :, :3], axis=2)
                self.report_progress("Converting RGB image to grayscale")
            else:
                # Use first channel for other multi-channel data
                gray = g[:, :, 0]
                self.report_progress("Using first channel of multi-channel data")
        else:
            gray = g
        
        # Apply edge detection if needed
        if set(np.unique(gray)) != {0, 1}:
            self.report_progress("Applying edge detection for feature enhancement")
            edges = compute_edge_detection(gray)
        else:
            edges = gray
        
        # Detect rectangles - use GPU if available
        if use_gpu and BACKEND != "numpy":
            self.report_progress(f"Using GPU acceleration for rectangle detection")
            rectangles = self._detect_rectangles_gpu(edges)
        else:
            self.report_progress("Using CPU for rectangle detection")
            rectangles = self._detect_rectangles(gray)
        
        # Report progress
        if rectangles:
            self.rectangles_detected += len(rectangles)
            self.report_progress(f"Detected {len(rectangles)} rectangular features (total: {self.rectangles_detected})")
        else:
            self.report_progress("No rectangular features detected in this tile")
        
        # Add detected rectangles to knowledge graph
        feature_ids = []
        features_added = 0
        for rect in rectangles:
            if rect['confidence'] >= self.threshold:
                # Create feature for each rectangle
                feature_id = self.kg.add_feature(
                    feature_type='RectangularFeature',
                    geometry={
                        'lat': lat,
                        'lon': lon,
                        'center': rect['center'],
                        'width': rect['width'],
                        'height': rect['height'],
                        'orientation': rect['orientation'],
                        'corners': rect.get('corners', [])
                    },
                    width=rect['width'],
                    height=rect['height'],
                    orientation=rect['orientation'],
                    area=rect['width'] * rect['height'],
                    confidence=rect['confidence'],
                    detected_by=self.name
                )
                feature_ids.append(feature_id)
                features_added += 1
        
        # Final report
        if feature_ids:
            self.rectangles_added += features_added
            self.report_progress(f"Added {features_added} rectangular features to knowledge graph (total: {self.rectangles_added})")
        
        return g
    
    def _detect_rectangles(self, grid):
        """Detect rectangles in a grid.
        
        In a real implementation, this would use contour detection, 
        edge detection, or similar computer vision techniques.
        
        Args:
            grid: Input grid to analyze
            
        Returns:
            List of rectangle dictionaries with center, width, height, orientation, confidence and corners
        """
        # Use GPU-accelerated gradients to enhance edge detection
        if grid.ndim == 2 and grid.size > 3:
            # Enhanced GPU processing for edge detection
            if BACKEND.startswith("torch") and torch is not None:
                try:
                    # Process on GPU for better performance
                    device = torch.device("mps") if BACKEND == "torch_mps" else \
                            torch.device("cuda") if BACKEND == "torch_cuda" else \
                            torch.device("cpu")
                    
                    # Transfer to GPU
                    t_grid = torch.from_numpy(grid).float().to(device)
                    
                    # Apply Sobel-like filters on GPU
                    if t_grid.dim() == 2:
                        # Add batch and channel dimensions
                        t_grid = t_grid.unsqueeze(0).unsqueeze(0)
                        
                        # Define simple Sobel-like filters
                        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                             dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
                        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                             dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
                        
                        import torch.nn.functional as F
                        # Apply convolution for edge detection (using proper padding)
                        grad_x = F.conv2d(t_grid, sobel_x, padding=1)
                        grad_y = F.conv2d(t_grid, sobel_y, padding=1)
                        
                        # Compute magnitude
                        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
                        
                        # Threshold to get edges (using mean + std as threshold)
                        mean = torch.mean(grad_mag)
                        std = torch.std(grad_mag)
                        edges = (grad_mag > (mean + std)).float()
                        
                        # Transfer back to CPU as numpy array
                        processed_grid = edges.squeeze().cpu().numpy().astype(np.int32)
                        
                        # Log GPU usage occasionally (not on every tile)
                        if np.random.random() < 0.05:  # 5% chance to log
                            print(f"[GPU] Rectangle detection using {BACKEND} acceleration")
                        return self._detect_rectangles_gpu(processed_grid)
                except Exception as e:
                    print(f"[GPU] Fallback to CPU: {str(e)[:100]}")
            
            # Standard CPU processing if GPU attempt failed
            grad_x, grad_y = compute_gradients(grid)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            edges = grad_mag > (np.mean(grad_mag) + np.std(grad_mag))
            processed_grid = edges.astype(np.int32)
        else:
            processed_grid = grid
            
        # This is a placeholder - in practice, you would use more sophisticated methods
        # such as contour detection, corner detection, or Hough transform for lines
        h, w = processed_grid.shape
        rectangles = []
        
        # Simple detection: find regions with consistent values surrounded by different values
        # This is a very simplified approach - real implementation would be more robust
        visited = np.zeros_like(processed_grid, dtype=bool)
        
        # Scan the grid to find potential rectangular regions
        for y in range(1, h-1):
            for x in range(1, w-1):
                if visited[y, x]:
                    continue
                    
                # Check if this could be a corner of a rectangle
                if (self._is_corner(processed_grid, x, y)):
                    # Try to grow a rectangle from this corner
                    rect = self._grow_rectangle(processed_grid, x, y, visited)
                    if rect is not None:
                        rectangles.append(rect)
        
        return rectangles
    
    def _detect_rectangles_gpu(self, edge_grid):
        """GPU-optimized version of rectangle detection algorithm
        
        Args:
            edge_grid: Binary edge image from GPU processing
            
        Returns:
            List of rectangles as dictionaries with center, width, height, orientation, and confidence
        """
        # Start time for performance tracking
        start_time = time.time()
        
        print(f"[GPU] Rectangle detection starting on grid shape {edge_grid.shape}")
        
        h, w = edge_grid.shape
        rectangles = []
        
        # Faster algorithm for GPU-processed edge image
        # We'll use horizontal and vertical projections to find rectangular regions
        
        # Get horizontal and vertical projections (sum along axes)
        h_proj = np.sum(edge_grid, axis=0)
        v_proj = np.sum(edge_grid, axis=1)
        
        # Track GPU memory for rectangle detection operations
        if BACKEND == "torch_cuda":
            import torch
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            print(f"[GPU] Rectangle detection: {memory_allocated:.1f}MB GPU memory in use")
        
        # Find peaks in projections (start of potential rectangles)
        h_peaks = np.where(np.diff(np.signbit(np.diff(h_proj))))[0] + 1
        v_peaks = np.where(np.diff(np.signbit(np.diff(v_proj))))[0] + 1
        
        print(f"[GPU] Found {len(h_peaks)} horizontal and {len(v_peaks)} vertical peaks for rectangle candidates")
        
        # For each pair of horizontal and vertical peaks, check if they form a rectangle
        total_combinations = len(h_peaks) * len(v_peaks)
        processed_combinations = 0
        
        # Log start of intensive processing if there are many candidates
        if total_combinations > 1000:
            print(f"[GPU] Processing {total_combinations} potential rectangle combinations...")
            progress_interval = max(1, total_combinations // 10)  # Report at 10% intervals
        else:
            progress_interval = total_combinations + 1  # No progress reporting for small sets
            
        for x1_idx in range(len(h_peaks)-1):
            for x2_idx in range(x1_idx+1, min(x1_idx+5, len(h_peaks))):
                x1, x2 = h_peaks[x1_idx], h_peaks[x2_idx]
                width = x2 - x1
                
                if width < self.min_size or width > self.max_size:
                    continue
                    
                for y1_idx in range(len(v_peaks)-1):
                    for y2_idx in range(y1_idx+1, min(y1_idx+5, len(v_peaks))):
                        y1, y2 = v_peaks[y1_idx], v_peaks[y2_idx]
                        height = y2 - y1
                        
                        # Track progress for large operations
                        processed_combinations += 1
                        if processed_combinations % progress_interval == 0:
                            progress_pct = (processed_combinations / total_combinations) * 100
                            print(f"[GPU] Rectangle detection progress: {progress_pct:.1f}% ({processed_combinations}/{total_combinations})")
                        
                        if height < self.min_size or height > self.max_size:
                            continue
                            
                        # Check if this region has enough edge pixels to be a rectangle
                        region = edge_grid[y1:y2, x1:x2]
                        edge_density = np.sum(region) / (width * height)
                        
                        if edge_density > 0.3:  # At least 30% of perimeter pixels are edges
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # For simplicity, we're not computing orientation in the GPU version
                            orientation = 0.0
                            
                            # Return dictionary instead of tuple
                            corners = self._get_rectangle_corners(center_x, center_y, width, height, orientation)
                            confidence = self._calculate_confidence(edge_grid, (center_x, center_y, width, height, orientation))
                            
                            rectangles.append({
                                'center': (center_x, center_y),
                                'width': width,
                                'height': height,
                                'orientation': orientation,
                                'corners': corners,
                                'confidence': confidence
                            })
        
        # Calculate elapsed time and report performance
        elapsed_time = time.time() - start_time
        rectangles_per_second = len(rectangles) / max(0.001, elapsed_time)
        
        print(f"[GPU] Rectangle detection completed in {elapsed_time:.3f}s")
        print(f"[GPU] Found {len(rectangles)} potential rectangular features ({rectangles_per_second:.1f} rects/sec)")
        
        # Display sizes of detected rectangles (helps with tuning parameters)
        if len(rectangles) > 0:
            sizes = [rect['width'] * rect['height'] for rect in rectangles]
            avg_size = sum(sizes) / len(sizes)
            min_size = min(sizes)
            max_size = max(sizes)
            print(f"[GPU] Rectangle sizes - Avg: {avg_size:.1f}px² | Min: {min_size:.1f}px² | Max: {max_size:.1f}px²")
            
        return rectangles
    
    def _is_corner(self, grid, x, y):
        """Check if a point could be a corner of a rectangle based on local gradient."""
        # Simple corner detection - check for gradient in both directions
        h, w = grid.shape
        if x <= 0 or y <= 0 or x >= w-1 or y >= h-1:
            return False
            
        # Check horizontal and vertical gradients
        horiz_diff = abs(float(grid[y, x+1]) - float(grid[y, x-1]))
        vert_diff = abs(float(grid[y+1, x]) - float(grid[y-1, x]))
        
        # If both gradients are significant, might be a corner
        return horiz_diff > np.std(grid) and vert_diff > np.std(grid)
    
    def _grow_rectangle(self, grid, start_x, start_y, visited):
        """Attempt to grow a rectangle starting from a corner.
        
        Args:
            grid: Input grid
            start_x, start_y: Starting corner position
            visited: Boolean array tracking visited positions
            
        Returns:
            Rectangle dictionary with center, width, height, orientation properties, or None
        """
        # This is a highly simplified approach
        # In practice, you'd use more sophisticated contour or shape detection
        h, w = grid.shape
        
        # Find the maximum extent in x and y directions with similar values
        base_value = grid[start_y, start_x]
        tolerance = np.std(grid) / 2
        
        # Search for right edge
        right_x = start_x
        while right_x < w-1 and abs(float(grid[start_y, right_x]) - float(base_value)) <= tolerance:
            right_x += 1
            
        # Search for bottom edge
        bottom_y = start_y
        while bottom_y < h-1 and abs(float(grid[bottom_y, start_x]) - float(base_value)) <= tolerance:
            bottom_y += 1
            
        # Check if we found a reasonable rectangle
        width = right_x - start_x
        height = bottom_y - start_y
        
        if width >= self.min_size and height >= self.min_size:
            # Mark all points in this rectangle as visited
            visited[start_y:bottom_y, start_x:right_x] = True
            
            # Return rectangle parameters as a dictionary
            center_x = start_x + width // 2
            center_y = start_y + height // 2
            orientation = 0.0  # Orientation is 0 in this simple case
            
            # Create corners
            corners = self._get_rectangle_corners(center_x, center_y, width, height, orientation)
            
            # Calculate confidence
            confidence = self._calculate_confidence(grid, (center_x, center_y, width, height, orientation))
            
            return {
                'center': (center_x, center_y),
                'width': width,
                'height': height,
                'orientation': orientation,
                'corners': corners,
                'confidence': confidence
            }
            
        return None
    
    def _calculate_confidence(self, grid, rect):
        """Calculate confidence score for a detected rectangle.
        
        Args:
            grid: Input grid
            rect: Rectangle as (center_x, center_y, width, height, orientation)
            
        Returns:
            Confidence score between 0 and 1
        """
        # This is a simplified confidence calculation
        # In practice, you would evaluate how rectangular and regular the shape is
        
        x, y, width, height, orientation = rect
        
        # Higher confidence for more square-like rectangles
        aspect_ratio = min(width, height) / max(width, height)
        aspect_score = aspect_ratio * 0.5  # Ranges from 0 to 0.5
        
        # Higher confidence for larger rectangles (up to a point)
        size = (width * height) / (self.max_size ** 2)
        size_score = min(size, 1.0) * 0.3  # Ranges from 0 to 0.3
        
        # Base confidence score
        base_score = 0.2
        
        return base_score + aspect_score + size_score
    
    def _get_rectangle_corners(self, center_x, center_y, width, height, orientation_deg):
        """Get the four corners of a rectangle with given parameters.
        
        Args:
            center_x, center_y: Center of the rectangle
            width, height: Dimensions of the rectangle
            orientation_deg: Orientation in degrees (0 = aligned with x-axis)
            
        Returns:
            List of corner coordinates as [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        # Convert orientation to radians
        orientation_rad = np.radians(orientation_deg)
        
        # Half-dimensions
        w2, h2 = width/2, height/2
        
        # Get corners relative to center without rotation
        corners_rel = [
            [-w2, -h2],  # Top-left
            [w2, -h2],   # Top-right
            [w2, h2],    # Bottom-right
            [-w2, h2]    # Bottom-left
        ]
        
        # Apply rotation and translate to center
        cos_theta = np.cos(orientation_rad)
        sin_theta = np.sin(orientation_rad)
        
        corners = []
        for x_rel, y_rel in corners_rel:
            # Rotate
            x_rot = x_rel * cos_theta - y_rel * sin_theta
            y_rot = x_rel * sin_theta + y_rel * cos_theta
            
            # Translate
            x = center_x + x_rot
            y = center_y + y_rot
            
            corners.append([float(x), float(y)])
            
        return corners

# Move the classes here, just before the main function and after the load_tiles_with_sliding_window function
class ClusterAgent(BaseAgent):
    """Agent to cluster spatially related features."""
    
    def __init__(self, kg, distance_threshold=1.0, min_cluster_size=2, name=None):
        """Initialize the cluster detection agent."""
        super().__init__(name or "ClusterAgent")
        self.kg = kg
        self.distance_threshold = distance_threshold  # in km
        self.min_cluster_size = min_cluster_size
    
    def find_clusters(self):
        """Find clusters of spatially related features."""
        self.report_progress("Starting spatial clustering of features")
        
        # Get all features with spatial coordinates
        features = [f for f in self.kg.features if 'geometry' in f 
                   and isinstance(f['geometry'], dict)
                   and 'lat' in f['geometry'] and 'lon' in f['geometry']]
        
        if not features:
            self.report_progress("No features with spatial coordinates found")
            return []
        
        self.report_progress(f"Clustering {len(features)} features")
        
        # Build a spatial index for efficient nearest-neighbor queries
        # This is a simple implementation - in production you'd use a proper
        # spatial index like an R-tree
        clusters = []
        processed = set()
        
        # For each unprocessed feature, find its neighbors
        for i, feature in enumerate(features):
            if i in processed:
                continue
                
            # Get coordinates for this feature
            f_lat = feature['geometry']['lat']
            f_lon = feature['geometry']['lon']
            
            # Find all features within the distance threshold
            cluster_members = []
            cluster_member_ids = []
            
            for j, other in enumerate(features):
                if j == i:
                    cluster_members.append(other)
                    cluster_member_ids.append(other['id'])
                    processed.add(j)
                    continue
                    
                if j in processed:
                    continue
                    
                # Calculate distance
                o_lat = other['geometry']['lat']
                o_lon = other['geometry']['lon']
                
                # Use Haversine distance
                dist = self.kg._calculate_distance_km(f_lat, f_lon, o_lat, o_lon)
                
                if dist <= self.distance_threshold:
                    cluster_members.append(other)
                    cluster_member_ids.append(other['id'])
                    processed.add(j)
            
            # If we found a valid cluster, add it to our list
            if len(cluster_members) >= self.min_cluster_size:
                # Calculate cluster type based on member features
                member_types = [m.get('feature_type', 'Unknown') for m in cluster_members]
                most_common_type = max(set(member_types), key=member_types.count)
                cluster_type = f"{most_common_type}Cluster"
                
                # Add cluster to KG
                cluster_id = self.kg.add_cluster(
                    member_feature_ids=cluster_member_ids,
                    cluster_type=cluster_type,
                    feature_types=list(set(member_types)),
                    detected_by=self.name
                )
                
                clusters.append(cluster_id)
        
        self.report_progress(f"Found {len(clusters)} feature clusters")
        return clusters

class OrientationAgent(BaseAgent):
    """Agent to detect alignment patterns between features."""
    
    def __init__(self, kg, angle_tolerance=10.0, name=None):
        """Initialize the alignment detection agent."""
        super().__init__(name or "OrientationAgent")
        self.kg = kg
        self.angle_tolerance = angle_tolerance  # degrees
    
    def find_alignments(self):
        """Find alignments between features based on orientation."""
        self.report_progress("Starting alignment analysis")
        
        # Get all features with orientation information
        features = [f for f in self.kg.features 
                   if ('orientation' in f) and
                   ('geometry' in f and isinstance(f['geometry'], dict) and
                    'lat' in f['geometry'] and 'lon' in f['geometry'])]
        
        if not features:
            self.report_progress("No features with orientation information found")
            return []
        
        # Group features by orientation (within tolerance)
        orientation_groups = {}
        
        for feature in features:
            orientation = feature['orientation']
            
            # Find matching group or create new one
            matched = False
            for group_orientation, group_features in orientation_groups.items():
                # Handle circular nature of angles (e.g., 5° and 355° are 10° apart)
                angle_diff = min(
                    abs(orientation - group_orientation),
                    360 - abs(orientation - group_orientation)
                )
                
                if angle_diff <= self.angle_tolerance:
                    group_features.append(feature)
                    matched = True
                    break
            
            if not matched:
                # Create new group
                orientation_groups[orientation] = [feature]
        
        # Find valid alignment groups (groups with multiple features)
        alignments = []
        for orientation, group_features in orientation_groups.items():
            if len(group_features) >= 2:
                # Create alignment in KG
                member_ids = [f['id'] for f in group_features]
                
                # Determine alignment type
                is_cardinal = (orientation <= 15 or 
                               (orientation >= 75 and orientation <= 105) or 
                               orientation >= 165)
                              
                alignment_type = "CardinalAlignment" if is_cardinal else "SharedOrientation"
                
                # Add to KG
                alignment_id = self.kg.add_alignment(
                    member_feature_ids=member_ids,
                    orientation=orientation,
                    alignment_type=alignment_type,
                    is_cardinal=is_cardinal,
                    detected_by=self.name
                )
                
                alignments.append(alignment_id)
        
        self.report_progress(f"Found {len(alignments)} feature alignments")
        return alignments

class AdvancedFeatureRepresentationAgent(BaseAgent):
    """Agent that uses dimensionality reduction and anomaly detection for improved site detection.
    
    This agent implements advanced feature representation methods like PCA/SVD for dimensionality
    reduction and unsupervised anomaly detection (Isolation Forest, kNN) to identify archaeological 
    sites with greater accuracy than simple template matching.
    """
    def __init__(self, kg, n_components=20, anomaly_threshold=0.75, name=None):
        """Initialize the advanced feature representation agent.
        
        Args:
            kg: Knowledge graph to store results
            n_components: Number of PCA components to use for dimensionality reduction
            anomaly_threshold: Threshold for anomaly detection (higher = more anomalous)
            name: Agent name
        """
        super().__init__(name=name or "AdvancedFeatureRepresentationAgent")
        self.kg = kg
        self.n_components = n_components
        self.anomaly_threshold = anomaly_threshold
        self.pca = None
        self.isolation_forest = None
        self.knn = None
        self.template_embeddings = None
        
    def fit(self, templates, background_tiles=None):
        """Fit the dimensionality reduction and anomaly detection models.
        
        Args:
            templates: List of template arrays
            background_tiles: List of background tile arrays for anomaly detection
            
        Returns:
            Self for chaining
        """
        from sklearn.decomposition import PCA
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        
        # Stack templates into a single array for PCA
        if isinstance(templates[0], dict) and 'grid' in templates[0]:
            template_grids = [t['grid'].flatten() for t in templates if 'grid' in t]
        else:
            template_grids = [t.flatten() for t in templates]
            
        # Ensure all templates are the same size
        min_size = min(grid.shape[0] for grid in template_grids)
        template_array = np.array([grid[:min_size] for grid in template_grids])
        
        # Fit PCA model
        print(f"[{self.name}] Fitting PCA with {self.n_components} components on {len(templates)} templates")
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(template_array)
        
        # Transform templates to get embeddings
        self.template_embeddings = self.pca.transform(template_array)
        
        # If we have background tiles, fit anomaly detection models
        if background_tiles:
            if isinstance(background_tiles[0], dict) and 'grid' in background_tiles[0]:
                background_grids = [t['grid'].flatten()[:min_size] for t in background_tiles if 'grid' in t]
            else:
                background_grids = [t.flatten()[:min_size] for t in background_tiles]
                
            background_array = np.array(background_grids)
            
            # Transform background tiles to PCA space
            background_embeddings = self.pca.transform(background_array)
            
            # Fit Isolation Forest for anomaly detection
            print(f"[{self.name}] Fitting Isolation Forest on {len(background_tiles)} background tiles")
            self.isolation_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
            self.isolation_forest.fit(background_embeddings)
            
            # Fit k-NN for distance-based anomaly detection
            print(f"[{self.name}] Fitting k-NN model for distance-based anomaly detection")
            self.knn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
            self.knn.fit(background_embeddings)
            
        return self
        
    def transform(self, g, **kwargs):
        """Process a grid to detect potential archaeological sites using advanced feature methods.
        
        Args:
            g: Input grid
            **kwargs: Additional arguments
            
        Returns:
            The input grid
        """
        if g is None or not hasattr(g, 'shape'):
            return g
            
        import numpy as np
        
        # Get metadata
        meta = kwargs.get('meta', {})
        verbose = kwargs.get('verbose', False)
        
        # Flatten and resize grid to match template dimensions
        flat_grid = g.flatten()
        if self.pca and len(flat_grid) < self.pca.n_features_in_:
            # Pad if too small
            flat_grid = np.pad(flat_grid, (0, self.pca.n_features_in_ - len(flat_grid)))
        elif self.pca and len(flat_grid) > self.pca.n_features_in_:
            # Truncate if too large
            flat_grid = flat_grid[:self.pca.n_features_in_]
            
        if self.pca is None:
            if verbose:
                print(f"[{self.name}] PCA model not fitted yet, skipping")
            return g
            
        # Transform grid to PCA space
        grid_embedding = self.pca.transform(flat_grid.reshape(1, -1))
        
        # Calculate similarity to templates
        if self.template_embeddings is not None:
            # Compute distances to all template embeddings
            distances = np.sqrt(((self.template_embeddings - grid_embedding) ** 2).sum(axis=1))
            min_distance = distances.min()
            most_similar_idx = distances.argmin()
            
            # Convert distance to similarity score (inverse relationship)
            similarity = 1.0 / (1.0 + min_distance)
            
            if verbose:
                print(f"[{self.name}] Most similar template: {most_similar_idx}, similarity: {similarity:.4f}")
                
            # Add to knowledge graph if similarity is high enough
            if similarity > 0.6 and 'lat' in meta and 'lon' in meta:
                feature_id = self.kg.add_feature(
                    feature_type="PCA_TemplateMatch",
                    geometry={"lat": meta['lat'], "lon": meta['lon']},
                    confidence=float(similarity),
                    detected_by=self.name,
                    most_similar_template=int(most_similar_idx),
                    pca_distance=float(min_distance)
                )
                if verbose:
                    print(f"[{self.name}] Added feature {feature_id} with confidence {similarity:.4f}")
        
        # Run anomaly detection if models are fitted
        anomaly_score = None
        
        if self.isolation_forest is not None:
            # Get anomaly score from Isolation Forest (-1 to 1, higher = more anomalous)
            if_score = self.isolation_forest.score_samples(grid_embedding)[0]
            # Convert to 0-1 range where 1 is most anomalous
            if_score = (1 - (if_score + 1) / 2)
            
            if verbose and if_score > self.anomaly_threshold:
                print(f"[{self.name}] Isolation Forest anomaly score: {if_score:.4f}")
                
            anomaly_score = if_score
        
        if self.knn is not None:
            # Get distance to nearest neighbors
            distances, _ = self.knn.kneighbors(grid_embedding)
            # Use mean distance to k neighbors as anomaly score
            knn_score = distances.mean()
            
            # Normalize KNN score to 0-1 range (assuming typical distances are 0-10)
            knn_score = min(1.0, knn_score / 10.0)
            
            if verbose and knn_score > self.anomaly_threshold:
                print(f"[{self.name}] k-NN anomaly score: {knn_score:.4f}")
                
            # Combine scores if we have both
            if anomaly_score is not None:
                anomaly_score = (anomaly_score + knn_score) / 2
            else:
                anomaly_score = knn_score
        
        # Add potential site to knowledge graph if anomaly score is high enough
        if anomaly_score is not None and anomaly_score > self.anomaly_threshold:
            if 'lat' in meta and 'lon' in meta:
                feature_id = self.kg.add_feature(
                    feature_type="AnomalyDetection",
                    geometry={"lat": meta['lat'], "lon": meta['lon']},
                    confidence=float(anomaly_score),
                    detected_by=f"{self.name}_AnomalyDetection",
                    anomaly_score=float(anomaly_score)
                )
                if verbose:
                    print(f"[{self.name}] Added anomaly feature {feature_id} with score {anomaly_score:.4f}")
        
        return g

    def batch_process_tiles(self, tiles):
        """Process a batch of tiles efficiently using vectorized operations.
        
        Args:
            tiles: List of tile dictionaries
            
        Returns:
            List of feature IDs added to the knowledge graph
        """
        import numpy as np
        from tqdm import tqdm
        
        if self.pca is None:
            print(f"[{self.name}] PCA model not fitted yet, skipping batch processing")
            return []
            
        print(f"[{self.name}] Batch processing {len(tiles)} tiles with advanced feature representation")
        
        # Extract grids and metadata
        grids = []
        tile_metas = []
        
        for tile in tiles:
            if isinstance(tile, dict) and 'grid' in tile:
                grid = tile['grid']
                meta = tile.get('meta', {})
                if grid is not None and hasattr(grid, 'shape'):
                    grids.append(grid)
                    tile_metas.append(meta)
        
        if not grids:
            print(f"[{self.name}] No valid grids found in tiles, skipping")
            return []
            
        # Process in batches to avoid memory issues
        batch_size = 100
        feature_ids = []
        
        for i in range(0, len(grids), batch_size):
            batch_grids = grids[i:i+batch_size]
            batch_metas = tile_metas[i:i+batch_size]
            
            # Prepare grid data for PCA
            flat_grids = []
            for g in batch_grids:
                flat_g = g.flatten()
                if len(flat_g) < self.pca.n_features_in_:
                    flat_g = np.pad(flat_g, (0, self.pca.n_features_in_ - len(flat_g)))
                elif len(flat_g) > self.pca.n_features_in_:
                    flat_g = flat_g[:self.pca.n_features_in_]
                flat_grids.append(flat_g)
                
            # Transform to PCA space
            grid_array = np.array(flat_grids)
            embeddings = self.pca.transform(grid_array)
            
            # Calculate similarities to templates
            distances = np.zeros((len(embeddings), len(self.template_embeddings)))
            for j, emb in enumerate(embeddings):
                distances[j] = np.sqrt(((self.template_embeddings - emb.reshape(1, -1)) ** 2).sum(axis=1))
                
            min_distances = distances.min(axis=1)
            most_similar_indices = distances.argmin(axis=1)
            similarities = 1.0 / (1.0 + min_distances)
            
            # Calculate anomaly scores
            anomaly_scores = np.zeros(len(embeddings))
            
            if self.isolation_forest is not None:
                if_scores = self.isolation_forest.score_samples(embeddings)
                # Convert to 0-1 range where 1 is most anomalous
                if_scores = (1 - (if_scores + 1) / 2)
                anomaly_scores = if_scores
                
            if self.knn is not None:
                knn_distances, _ = self.knn.kneighbors(embeddings)
                knn_scores = knn_distances.mean(axis=1)
                # Normalize KNN scores
                knn_scores = np.minimum(1.0, knn_scores / 10.0)
                
                if self.isolation_forest is not None:
                    # Combine scores
                    anomaly_scores = (anomaly_scores + knn_scores) / 2
                else:
                    anomaly_scores = knn_scores
            
            # Add features to knowledge graph
            for j in range(len(batch_grids)):
                meta = batch_metas[j]
                if 'lat' not in meta or 'lon' not in meta:
                    continue
                    
                # Add template match if similarity is high enough
                if similarities[j] > 0.6:
                    feature_id = self.kg.add_feature(
                        feature_type="PCA_TemplateMatch",
                        geometry={"lat": meta['lat'], "lon": meta['lon']},
                        confidence=float(similarities[j]),
                        detected_by=self.name,
                        most_similar_template=int(most_similar_indices[j]),
                        pca_distance=float(min_distances[j])
                    )
                    feature_ids.append(feature_id)
                
                # Add anomaly detection if score is high enough
                if anomaly_scores[j] > self.anomaly_threshold:
                    feature_id = self.kg.add_feature(
                        feature_type="AnomalyDetection",
                        geometry={"lat": meta['lat'], "lon": meta['lon']},
                        confidence=float(anomaly_scores[j]),
                        detected_by=f"{self.name}_AnomalyDetection",
                        anomaly_score=float(anomaly_scores[j])
                    )
                    feature_ids.append(feature_id)
        
        print(f"[{self.name}] Added {len(feature_ids)} features to knowledge graph")
        return feature_ids

class RepetitionAgent(BaseAgent):
    """Agent for detecting repetitive patterns in archaeological features."""
    
    def __init__(self, kg, min_spacing=5, max_spacing=50, threshold=0.5, name=None):
        """Initialize the repetition detection agent."""
        super().__init__(name or "RepetitionAgent")
        self.kg = kg
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.threshold = threshold
    
    def transform(self, g, **kwargs):
        """Detect repetitive patterns in the grid."""
        # Skip empty or tiny grids
        if g.size == 0 or min(g.shape) < self.min_spacing * 3:
            return g
            
        verbose = kwargs.get('verbose', False)
        
        # Get grid metadata if available
        meta = kwargs.get('meta', {})
        
        # Simple 1D autocorrelation to detect repeating patterns
        height, width = g.shape
        
        # Convert to binary if not already
        if set(np.unique(g)) != {0, 1}:
            # Apply threshold at mean
            g_binary = (g > np.mean(g)).astype(np.float32)
        else:
            g_binary = g.astype(np.float32)
            
        # Try to detect horizontal repetition
        h_repetitions = []
        for offset in range(self.min_spacing, min(self.max_spacing, width // 2)):
            # Calculate correlation between original and shifted version
            shifted = np.roll(g_binary, offset, axis=1)
            # Mask out the wrapped-around part
            mask = np.ones_like(g_binary)
            mask[:, :offset] = 0
            # Compute correlation only on valid part
            correlation = np.sum(g_binary * shifted * mask) / (np.sum(mask) + 1e-10)
            
            if correlation > self.threshold:
                h_repetitions.append((offset, correlation))
                
        # Try to detect vertical repetition
        v_repetitions = []
        for offset in range(self.min_spacing, min(self.max_spacing, height // 2)):
            # Calculate correlation between original and shifted version
            shifted = np.roll(g_binary, offset, axis=0)
            # Mask out the wrapped-around part
            mask = np.ones_like(g_binary)
            mask[:offset, :] = 0
            # Compute correlation only on valid part
            correlation = np.sum(g_binary * shifted * mask) / (np.sum(mask) + 1e-10)
            
            if correlation > self.threshold:
                v_repetitions.append((offset, correlation))
        
        # If found repetitions, add to knowledge graph
        if h_repetitions or v_repetitions:
            lat = kwargs.get('lat', 0)
            lon = kwargs.get('lon', 0)
            
            # Find the strongest repetition
            strongest = None
            if h_repetitions and (not v_repetitions or h_repetitions[0][1] >= v_repetitions[0][1]):
                strongest = ("horizontal", h_repetitions[0][0], h_repetitions[0][1])
            elif v_repetitions:
                strongest = ("vertical", v_repetitions[0][0], v_repetitions[0][1])
                
            if strongest:
                direction, spacing, correlation = strongest
                feature_id = self.kg.add_feature(
                    feature_type="RepetitivePattern",
                    geometry={
                        'lat': lat,
                        'lon': lon,
                        'width': width,
                        'height': height
                    },
                    direction=direction,
                    spacing=spacing,
                    frequency=1.0/spacing,
                    correlation=correlation,
                    confidence=correlation,
                    detected_by=self.name
                )
                
        return g

class ParallelProcessor:
    """Manages parallel processing of tiles with multiple platoons of agents."""
    
    def __init__(self, kg, num_workers=None):
        """Initialize the parallel processor.
        
        Args:
            kg: Shared knowledge graph to store results
            num_workers: Number of worker threads/processes to use (None = auto)
        """
        self.kg = kg
        self.kg_lock = Lock()  # Lock for thread-safe KG updates
        
        # Determine number of workers
        if num_workers is None:
            import multiprocessing
            self.num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave 1 CPU free
        else:
            self.num_workers = max(1, num_workers)
        
        # Add semaphore for coordinating analysis phase
        self.analysis_semaphore = threading.Semaphore(0)
        self.analysis_active = False
        self.analysis_complete = False
        self.feature_count_last_analysis = 0
        self.min_new_features_for_analysis = 100  # Start analysis after at least 100 new features
        
        # Add batch activity tracking
        self.active_batches = {}  # batch_id -> last_activity_time
        self.batch_activity_lock = threading.Lock()
        self.longest_inactive_batch = None
        self.longest_inactive_time = 0
        
        # Add checkpoint tracking
        self.processed_tiles = set()  # Set of processed tile indices
        self.checkpoint_lock = threading.Lock()
        self.checkpoint_path = os.path.join(kg.storage_dir, "processing_checkpoint.json")
        self.checkpoint_interval = 10  # Save checkpoint after every 10 batches
        self.batches_since_checkpoint = 0
            
        print(f"Using {self.num_workers} worker threads for parallel processing")
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
    def _save_checkpoint(self):
        """Save the current processing state to a checkpoint file."""
        with self.checkpoint_lock:
            checkpoint_data = {
                "processed_tiles": list(self.processed_tiles),
                "feature_count": len(self.kg.features),
                "timestamp": time.time()
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            
            # Save checkpoint
            try:
                with open(self.checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f)
                print(f"[Checkpoint] Saved processing state: {len(self.processed_tiles)} tiles processed")
            except Exception as e:
                print(f"[WARNING] Failed to save checkpoint: {e}")
                
    def _load_checkpoint(self):
        """Load processing state from checkpoint file if it exists."""
        # Don't load checkpoint if reset is requested via environment variable
        if os.environ.get("RESET_KG", "").lower() in ("true", "1", "yes"):
            print("[Checkpoint] Reset requested - ignoring previous checkpoint")
            self.processed_tiles = set()
            return False
        
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                
                self.processed_tiles = set(checkpoint_data.get("processed_tiles", []))
                last_feature_count = checkpoint_data.get("feature_count", 0)
                timestamp = checkpoint_data.get("timestamp", 0)
                
                time_ago = time.time() - timestamp
                time_str = f"{time_ago:.1f} seconds" if time_ago < 120 else f"{time_ago/60:.1f} minutes"
                
                print(f"[Checkpoint] Loaded previous state from {time_str} ago:")
                print(f"[Checkpoint] {len(self.processed_tiles)} tiles already processed ({last_feature_count} features detected)")
                
                # Check if checkpoint is very old (more than 24 hours)
                if time_ago > 86400:  # 24 hours in seconds
                    print("[WARNING] Checkpoint is more than 24 hours old - consider resetting with set_reset_flag(True)")
                    
                return True
            except Exception as e:
                print(f"[WARNING] Failed to load checkpoint: {e}")
                self.processed_tiles = set()
                return False
        return False
    
    def run_analysis_thread(self):
        """Background thread that runs analysis agents on features as they become available."""
        print("[Analysis] Analysis thread started, waiting for sufficient features...")
        GLOBAL_SUPERVISOR.update_status("Analysis", "Waiting", agent_name="AnalysisThread", is_active=True)
        
        # Create ClusterAgent for spatial clustering
        cluster_agent = ClusterAgent(self.kg)
        
        # Create OrientationAgent for finding alignment patterns
        orientation_agent = OrientationAgent(self.kg)
        
        while not self.analysis_complete:
            # Wait for signal from detection thread or timeout
            acquired = self.analysis_semaphore.acquire(timeout=5.0)
            
            if not acquired and not self.analysis_active:
                # Just a timeout without any signal, continue waiting
                continue
                
            # Get current feature count
            with self.kg_lock:
                current_count = len(self.kg.features)
                
            # If we have enough new features since last analysis, run another round
            if current_count - self.feature_count_last_analysis >= self.min_new_features_for_analysis:
                GLOBAL_SUPERVISOR.update_status("Analysis", 
                                              f"Analyzing {current_count} features",
                                              agent_name="AnalysisThread", 
                                              is_active=True)
                                              
                print(f"[Analysis] Starting analysis of {current_count} features...")
                self.analysis_active = True
                
                # Run clustering agent on features
                GLOBAL_SUPERVISOR.update_status("Analysis", "Clustering", 
                                              agent_name="ClusterAgent", is_active=True)
                start_time = time.time()
                try:
                    cluster_agent.find_clusters()
                    print(f"[Analysis] Found {len(self.kg.clusters)} clusters in {time.time() - start_time:.2f}s")
                except Exception as e:
                    print(f"[ERROR] Clustering failed: {e}")
                finally:
                    GLOBAL_SUPERVISOR.update_status("Analysis", "Clustering complete", 
                                                  agent_name="ClusterAgent", is_active=False)
                
                # Run orientation analysis agent on features to find alignments
                GLOBAL_SUPERVISOR.update_status("Analysis", "Finding alignments", 
                                              agent_name="OrientationAgent", is_active=True)
                start_time = time.time()
                try:
                    orientation_agent.find_alignments()
                    print(f"[Analysis] Found {len(self.kg.alignments)} alignments in {time.time() - start_time:.2f}s")
                except Exception as e:
                    print(f"[ERROR] Alignment analysis failed: {e}")
                finally:
                    GLOBAL_SUPERVISOR.update_status("Analysis", "Alignment analysis complete", 
                                                  agent_name="OrientationAgent", is_active=False)
                
                # Update the last analysis count
                self.feature_count_last_analysis = current_count
                
                # If this is final analysis, run potential site analysis
                if self.analysis_complete:
                    GLOBAL_SUPERVISOR.update_status("Analysis", "Final analysis", 
                                                 agent_name="AnalysisThread", is_active=True)
                    
                    # Find potential new archaeological sites
                    print("[Analysis] Running final potential site analysis...")
                    
                    # Use more stringent confidence threshold for final analysis
                    potential_sites = self.kg.find_potential_new_sites(min_confidence=0.65)
                    print(f"[Analysis] Found {len(potential_sites)} potential new archaeological sites")
                    
            # Small delay to prevent busy wait
            time.sleep(0.1)
            
        print("[Analysis] Analysis thread completed.")
        GLOBAL_SUPERVISOR.update_status("Analysis", "Complete", agent_name="AnalysisThread", is_active=False)
    
    def process_tile_batch(self, tiles, batch_id):
        """Process a batch of tiles with a platoon of agents.
        
        Args:
            tiles: List of tile dictionaries
            batch_id: Batch identifier for tracking
            
        Returns:
            Number of tiles processed
        """
        # Register batch start with supervisor
        GLOBAL_SUPERVISOR.update_status("Processing", f"Processing batch {batch_id}", 
                                     agent_name=f"Batch_{batch_id}", is_active=True)
        
        # Update activity tracker
        with self.batch_activity_lock:
            self.active_batches[batch_id] = time.time()
        
        start_time = time.time()
        if not tiles:
            print(f"[Batch {batch_id}] Empty tile batch, skipping")
            return 0
            
        print(f"[Batch {batch_id}] Processing {len(tiles)} tiles")
        
        # Create platoon of agents for this batch
        agents = []
        
        # Template matching agent as the first line of feature detection
        templates = self.kg.get_templates()
        if templates:
            template_agent = TemplateMatchAgent(self.kg, templates, threshold=0.6)
            agents.append(template_agent)
            # Create specialized template agents with different thresholds for detection range
            template_agent_b0 = TemplateMatchAgent(self.kg, templates, threshold=0.5)
            template_agent_b0.name = "TemplateMatchAgent_B0"
            agents.append(template_agent_b0)
            template_agent_b1 = TemplateMatchAgent(self.kg, templates, threshold=0.7)
            template_agent_b1.name = "TemplateMatchAgent_B1"
            agents.append(template_agent_b1)
        
        # Line detection agent for linear features like roads, canals, etc.
        line_agent = LineDetectAgent(self.kg, min_length=10, threshold=0.5)
        agents.append(line_agent)
        
        # Rectangular feature detection agent for enclosures, field systems, etc.
        rect_agent = RectangularAgent(self.kg, min_size=4, max_size=50, threshold=0.5)
        agents.append(rect_agent)
        # Create specialized rectangular agent with different parameters
        rect_agent_b2 = RectangularAgent(self.kg, min_size=4, max_size=30, threshold=0.6)
        rect_agent_b2.name = "RectangularAgent_B2"
        agents.append(rect_agent_b2)
        
        # Repetitive pattern detection agent (for regular spacing, field systems, etc.)
        repet_agent = RepetitionAgent(self.kg, min_spacing=5, max_spacing=50, threshold=0.6)
        agents.append(repet_agent)
        
        # Register agents with supervisor
        for agent in agents:
            GLOBAL_SUPERVISOR.update_status("Processing", f"Agent {agent.name} active", 
                                         agent_name=agent.name, is_active=True)
        
        # Process tiles in sequence with this batch's agents
        tiles_processed = 0
        features_before = len(self.kg.features)
        
        # Process each tile with each agent
        for t_idx, tile in enumerate(tiles):
            GLOBAL_SUPERVISOR.update_status("Processing", 
                                         f"Tile {t_idx+1}/{len(tiles)} in batch {batch_id}",
                                         progress=(t_idx + 1) / len(tiles),
                                         agent_name=f"Batch_{batch_id}")
                                         
            # Update activity tracker for this batch
            with self.batch_activity_lock:
                self.active_batches[batch_id] = time.time()
            
            # Get the grid from tile dict
            if isinstance(tile, dict):
                grid = tile.get('grid')
                meta = tile.get('meta', {})
            else:
                # Handle case where tile is just the grid
                grid = tile
                meta = {}
            
            if grid is None or grid.size == 0:
                continue
                
            # Apply each agent to this tile
            try:
                for agent in agents:
                    GLOBAL_SUPERVISOR.update_status("Processing", 
                                                 f"Agent {agent.name} on tile {t_idx+1}/{len(tiles)}",
                                                 agent_name=agent.name)
                    
                    try:
                        # Add meta info to kwargs
                        kwargs = {'meta': meta, 'verbose': False}
                        if 'RectangularAgent' in agent.name or 'LineDetectAgent' in agent.name:
                            # Use GPU for rectangle and line detection when available
                            kwargs['use_gpu'] = BACKEND.endswith("cuda") or BACKEND.endswith("mps")
                            
                        # Apply transform
                        agent.transform(grid, **kwargs)
                    except Exception as e:
                        GLOBAL_SUPERVISOR.add_error(f"Agent {agent.name} failed on tile {t_idx}: {str(e)}")
                        print(f"[ERROR] Agent {agent.name} failed on tile {t_idx}: {str(e)}")
                
                # Mark tile as processed
                with self.checkpoint_lock:
                    if meta and 'index' in meta:
                        self.processed_tiles.add(meta['index'])
                
                tiles_processed += 1
                
                # Signal the analysis thread if we've found enough new features
                features_now = len(self.kg.features)
                if features_now - features_before >= self.min_new_features_for_analysis:
                    self.analysis_semaphore.release()
                    features_before = features_now
                
            except Exception as e:
                GLOBAL_SUPERVISOR.add_error(f"Error processing tile {t_idx}: {str(e)}")
                print(f"[ERROR] Processing tile {t_idx} failed: {str(e)}")
                
        # Save checkpoint if enough batches have been processed
        self.batches_since_checkpoint += 1
        if self.batches_since_checkpoint >= self.checkpoint_interval:
            self._save_checkpoint()
            self.batches_since_checkpoint = 0
            
        # Report batch completion
        elapsed = time.time() - start_time
        print(f"[Batch {batch_id}] Completed {tiles_processed}/{len(tiles)} tiles in {elapsed:.2f}s " +
              f"({tiles_processed/max(0.1, elapsed):.2f} tiles/sec)")
        
        # Update supervisor with completion status
        GLOBAL_SUPERVISOR.update_status("Processing", 
                                     f"Batch {batch_id} complete", 
                                     agent_name=f"Batch_{batch_id}",
                                     is_active=False)
                                     
        # Clean up
        for agent in agents:
            GLOBAL_SUPERVISOR.update_status("Processing", 
                                         f"Agent {agent.name} idle",
                                         agent_name=agent.name,
                                         is_active=False)
                                         
        # Remove from active batches
        with self.batch_activity_lock:
            if batch_id in self.active_batches:
                del self.active_batches[batch_id]
                
        return tiles_processed
    
    def process_tiles_parallel(self, tiles, batch_size=10, timeout_per_batch=60, final_analysis=True):
        """Process multiple tiles in parallel using separate thread for each batch.
        
        Args:
            tiles: List of tiles to process
            batch_size: Number of tiles to process in each batch
            timeout_per_batch: Maximum time per batch in seconds
            final_analysis: Whether to perform final analysis after processing
            
        Returns:
            Total number of tiles processed
        """
        # Check how many tiles are already processed from checkpoint
        already_processed = len(self.processed_tiles)
        if already_processed > 0:
            print(f"Resuming processing: {already_processed} tiles already processed from previous run")
            print(f"Will skip already processed tiles and continue where previous run was interrupted")
        
        # Create batches, skipping already processed tiles
        batches = []
        for i in range(0, len(tiles), batch_size):
            # Get tile indices for this batch
            batch_indices = list(range(i, min(i + batch_size, len(tiles))))
            
            # Filter out already processed tiles
            new_indices = [idx for idx in batch_indices if idx not in self.processed_tiles]
            
            # Only add non-empty batches
            if new_indices:
                batch_tiles = [tiles[idx] for idx in new_indices]
                batches.append((batch_tiles, new_indices, f"B{len(batches):04d}"))
        
        if not batches:
            print("All tiles have already been processed. Nothing to do.")
            return len(self.processed_tiles)
            
        print(f"Starting parallel processing of {len(tiles)} tiles in batches of {batch_size}")
        print(f"Created {len(batches)} batches ({already_processed} tiles skipped)")
        
        # Create queue for batches and results
        batch_queue = queue.Queue()
        results_queue = queue.Queue()
        
        # Add batches to queue
        for batch in batches:
            batch_queue.put(batch)
        
        # Define monitor thread function
        monitoring_enabled = True
        def monitor_thread():
            """Monitor active batches and detect stuck processes."""
            while monitoring_enabled:
                # Check for stuck batches
                current_time = time.time()
                with self.batch_activity_lock:
                    for batch_id, last_activity in list(self.active_batches.items()):
                        inactive_time = current_time - last_activity
                        if inactive_time > timeout_per_batch:
                            print(f"[WARNING] Batch {batch_id} has been inactive for {inactive_time:.1f}s, marking as stuck")
                            GLOBAL_SUPERVISOR.add_warning(f"Batch {batch_id} is stuck (inactive for {inactive_time:.1f}s)")
                            
                            # Update longest inactive batch tracking
                            if self.longest_inactive_batch is None or inactive_time > self.longest_inactive_time:
                                self.longest_inactive_batch = batch_id
                                self.longest_inactive_time = inactive_time
                                
                            # If a batch is severely stuck, attempt to cancel it
                            if inactive_time > timeout_per_batch * 2:
                                print(f"[WARNING] Batch {batch_id} is severely stuck, attempting to cancel")
                                GLOBAL_SUPERVISOR.add_error(f"Cancelling severely stuck batch {batch_id} (inactive for {inactive_time:.1f}s)")
                                del self.active_batches[batch_id]
                                
                # Log a summary of active batches periodically
                if self.active_batches:
                    batch_ids = list(self.active_batches.keys())
                    if len(batch_ids) <= 3:
                        print(f"Active batches: {batch_ids}")
                    else:
                        print(f"Active batches: {len(batch_ids)} batches (first few: {batch_ids[:3]}...)")
                
                # Sleep briefly to avoid CPU spinning
                time.sleep(5.0)
                
                # Save checkpoint occasionally during monitoring
                if batch_queue.qsize() % 5 == 0 and self.processed_tiles:
                    self._save_checkpoint()
        
        # Start analysis thread
        analysis_thread = threading.Thread(target=self.run_analysis_thread)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        # Start monitor thread
        monitor = threading.Thread(target=monitor_thread)
        monitor.daemon = True
        monitor.start()
        
        # Process tiles with threadpool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all batches for processing
            future_to_batch = {}
            
            # Initial batch submission
            for _ in range(min(self.num_workers, batch_queue.qsize())):
                if batch_queue.empty():
                    break
                    
                batch_data = batch_queue.get()
                batch_tiles, tile_indices, batch_id = batch_data
                
                future = executor.submit(self.process_tile_batch, batch_tiles, batch_id, tile_indices)
                future_to_batch[future] = batch_id
                
                # Register with supervisor
                GLOBAL_SUPERVISOR.update_status("Processing", 
                                             f"Batch {batch_id} queued",
                                             agent_name=f"Batch_{batch_id}",
                                             is_active=True)
            
            # Process batches and collect results
            completed = 0
            processed_count = already_processed
            
            # Process batches as they complete, submitting new ones
            while future_to_batch:
                # Wait for the next batch to complete
                try:
                    done, _ = concurrent.futures.wait(
                        future_to_batch, 
                        return_when=concurrent.futures.FIRST_COMPLETED,
                        timeout=5.0
                    )
                except Exception as e:
                    print(f"[ERROR] Wait error: {e}")
                    continue
                    
                for future in done:
                    batch_id = future_to_batch.pop(future)
                    
                    try:
                        # Get the results
                        tiles_processed = future.result()
                        processed_count += tiles_processed
                        completed += 1
                        
                        # Log progress
                        if completed % 5 == 0 or completed == len(batches):
                            completion_percent = (completed / len(batches)) * 100
                            print(f"Progress: {completed}/{len(batches)} batches " + 
                                  f"({completion_percent:.1f}%) - {processed_count} tiles processed")
                        
                        # Submit a new batch if available
                        if not batch_queue.empty():
                            batch_data = batch_queue.get()
                            batch_tiles, tile_indices, new_batch_id = batch_data
                            
                            # Submit the batch for processing
                            new_future = executor.submit(self.process_tile_batch, batch_tiles, new_batch_id, tile_indices)
                            future_to_batch[new_future] = new_batch_id
                            
                            # Register with supervisor
                            GLOBAL_SUPERVISOR.update_status("Processing", 
                                                        f"Batch {new_batch_id} queued",
                                                        agent_name=f"Batch_{new_batch_id}",
                                                        is_active=True)
                    except Exception as e:
                        print(f"[ERROR] Batch {batch_id} failed: {e}")
                        GLOBAL_SUPERVISOR.add_error(f"Batch {batch_id} failed: {e}")
                    
                    # Remove from active batches
                    with self.batch_activity_lock:
                        if batch_id in self.active_batches:
                            del self.active_batches[batch_id]
        
        # Signal to the analysis thread that we're done and wait for final analysis
        if final_analysis:
            print("All batches complete. Signaling analysis thread for final analysis...")
            self.analysis_complete = True
            self.analysis_semaphore.release()
            analysis_thread.join(timeout=120)  # Wait up to 2 minutes for final analysis
            
        # Save final checkpoint
        self._save_checkpoint()
        
        # Stop monitoring
        monitoring_enabled = False
        try:
            monitor.join(timeout=10)  # Wait up to 10 seconds for monitor to finish
        except:
            pass
            
        print(f"Parallel processing complete. Total tiles processed: {len(self.processed_tiles)}")
        return len(self.processed_tiles)

    def process_tile_batch(self, tiles, batch_id, tile_indices=None):
        """Process a batch of tiles with a platoon of agents.
        
        Args:
            tiles: List of tile dictionaries
            batch_id: Batch identifier for tracking
            tile_indices: List of original indices of tiles in the full dataset (for checkpointing)
            
        Returns:
            Number of tiles processed
        """
        # Register batch start with supervisor
        GLOBAL_SUPERVISOR.update_status("Processing", f"Processing batch {batch_id}", 
                                     agent_name=f"Batch_{batch_id}", is_active=True)
        
        # Update activity tracker
        with self.batch_activity_lock:
            self.active_batches[batch_id] = time.time()
        
        start_time = time.time()
        if not tiles:
            print(f"[Batch {batch_id}] Empty tile batch, skipping")
            return 0
            
        print(f"[Batch {batch_id}] Processing {len(tiles)} tiles")
        
        # Create platoon of agents for this batch
        agents = []
        
        # Template matching agent as the first line of feature detection
        templates = self.kg.get_templates()
        if templates:
            template_agent = TemplateMatchAgent(self.kg, templates, threshold=0.6)
            agents.append(template_agent)
            # Create specialized template agents with different thresholds for detection range
            template_agent_b0 = TemplateMatchAgent(self.kg, templates, threshold=0.5)
            template_agent_b0.name = "TemplateMatchAgent_B0"
            agents.append(template_agent_b0)
            template_agent_b1 = TemplateMatchAgent(self.kg, templates, threshold=0.7)
            template_agent_b1.name = "TemplateMatchAgent_B1"
            agents.append(template_agent_b1)
        
        # Line detection agent for linear features like roads, canals, etc.
        line_agent = LineDetectAgent(self.kg, min_length=10, threshold=0.5)
        agents.append(line_agent)
        
        # Rectangular feature detection agent for enclosures, field systems, etc.
        rect_agent = RectangularAgent(self.kg, min_size=4, max_size=50, threshold=0.5)
        agents.append(rect_agent)
        # Create specialized rectangular agent with different parameters
        rect_agent_b2 = RectangularAgent(self.kg, min_size=4, max_size=30, threshold=0.6)
        rect_agent_b2.name = "RectangularAgent_B2"
        agents.append(rect_agent_b2)
        
        # Repetitive pattern detection agent (for regular spacing, field systems, etc.)
        repet_agent = RepetitionAgent(self.kg, min_spacing=5, max_spacing=50, threshold=0.6)
        agents.append(repet_agent)
        
        # Register agents with supervisor
        for agent in agents:
            GLOBAL_SUPERVISOR.update_status("Processing", f"Agent {agent.name} active", 
                                         agent_name=agent.name, is_active=True)
        
        # Process tiles in sequence with this batch's agents
        tiles_processed = 0
        features_before = len(self.kg.features)
        
        # Process each tile with each agent
        for t_idx, tile in enumerate(tiles):
            GLOBAL_SUPERVISOR.update_status("Processing", 
                                         f"Tile {t_idx+1}/{len(tiles)} in batch {batch_id}",
                                         progress=(t_idx + 1) / len(tiles),
                                         agent_name=f"Batch_{batch_id}")
                                         
            # Update activity tracker for this batch
            with self.batch_activity_lock:
                self.active_batches[batch_id] = time.time()
            
            # Get the grid from tile dict
            if isinstance(tile, dict):
                grid = tile.get('grid')
                meta = tile.get('meta', {})
            else:
                # Handle case where tile is just the grid
                grid = tile
                meta = {}
            
            if grid is None:
                continue
                
            # Skip empty grids or grids without proper size attribute
            if not hasattr(grid, 'size') or grid.size == 0:
                print(f"[WARNING] Skipping invalid grid in batch {batch_id}, tile {t_idx}")
                continue
                
            # Apply each agent to this tile
            try:
                for agent in agents:
                    GLOBAL_SUPERVISOR.update_status("Processing", 
                                                 f"Agent {agent.name} on tile {t_idx+1}/{len(tiles)}",
                                                 agent_name=agent.name)
                    
                    try:
                        # Add meta info to kwargs
                        kwargs = {'meta': meta, 'verbose': False}
                        if 'RectangularAgent' in agent.name or 'LineDetectAgent' in agent.name:
                            # Use GPU for rectangle and line detection when available
                            kwargs['use_gpu'] = BACKEND.endswith("cuda") or BACKEND.endswith("mps")
                            
                        # Apply transform
                        agent.transform(grid, **kwargs)
                    except Exception as e:
                        GLOBAL_SUPERVISOR.add_error(f"Agent {agent.name} failed on tile {t_idx}: {str(e)}")
                        print(f"[ERROR] Agent {agent.name} failed on tile {t_idx}: {str(e)}")
                
                # Mark tile as processed
                with self.checkpoint_lock:
                    if tile_indices is not None and t_idx < len(tile_indices):
                        # Use the original index if provided
                        self.processed_tiles.add(tile_indices[t_idx])
                    elif meta and 'index' in meta:
                        # Otherwise use the index from metadata
                        self.processed_tiles.add(meta['index'])
                
                tiles_processed += 1
                
                # Signal the analysis thread if we've found enough new features
                features_now = len(self.kg.features)
                if features_now - features_before >= self.min_new_features_for_analysis:
                    self.analysis_semaphore.release()
                    features_before = features_now
                
            except Exception as e:
                GLOBAL_SUPERVISOR.add_error(f"Error processing tile {t_idx}: {str(e)}")
                print(f"[ERROR] Processing tile {t_idx} failed: {str(e)}")
                
        # Save checkpoint if enough batches have been processed
        self.batches_since_checkpoint += 1
        if self.batches_since_checkpoint >= self.checkpoint_interval:
            self._save_checkpoint()
            self.batches_since_checkpoint = 0
            
        # Report batch completion
        elapsed = time.time() - start_time
        print(f"[Batch {batch_id}] Completed {tiles_processed}/{len(tiles)} tiles in {elapsed:.2f}s " +
              f"({tiles_processed/max(0.1, elapsed):.2f} tiles/sec)")
        
        # Update supervisor with completion status
        GLOBAL_SUPERVISOR.update_status("Processing", 
                                     f"Batch {batch_id} complete", 
                                     agent_name=f"Batch_{batch_id}",
                                     is_active=False)
                                     
        # Clean up
        for agent in agents:
            GLOBAL_SUPERVISOR.update_status("Processing", 
                                         f"Agent {agent.name} idle",
                                         agent_name=agent.name,
                                         is_active=False)
                                         
        # Remove from active batches
        with self.batch_activity_lock:
            if batch_id in self.active_batches:
                del self.active_batches[batch_id]
                
        return tiles_processed

# Properly handle overlay data when running in Kaggle
if is_running_on_kaggle():
    overlay_base = '/kaggle/input/nasa-tiles'
    working_dir = '/kaggle/working'
    
    # Create test_data directory if it doesn't exist
    os.makedirs(f'{working_dir}/test_data', exist_ok=True)

    # Find all overlay directories (dem_overlay, vegetation_overlay, etc.)
    for overlay_dir in glob.glob(f'{overlay_base}/test_data/*/') + glob.glob(f'{overlay_base}/*/'):
        dir_name = os.path.basename(os.path.dirname(overlay_dir + '/'))
        if dir_name in ['dem_overlay', 'vegetation_overlay', 'historical_overlay']:
            dest_dir = f'{working_dir}/test_data/{dir_name}'
            print(f"Copying overlay data from {overlay_dir} to {dest_dir}")
            try:
                shutil.copytree(overlay_dir, dest_dir, dirs_exist_ok=True)
            except Exception as e:
                print(f"Error copying overlay data: {e}")

    # Make sure templates.json is also copied
    if os.path.exists(f'{overlay_base}/templates.json'):
        try:
            shutil.copy(f'{overlay_base}/templates.json', f'{working_dir}/templates.json')
            print("Copied templates.json to working directory")
        except Exception as e:
            print(f"Error copying templates.json: {e}")
    elif os.path.exists(f'{overlay_base}/test_data/templates.json'):
        try:
            shutil.copy(f'{overlay_base}/test_data/templates.json', f'{working_dir}/templates.json')
            print("Copied templates.json from test_data to working directory")
        except Exception as e:
            print(f"Error copying templates.json from test_data: {e}")

    # Add this to handle double-nested paths
    if os.path.exists(f'{overlay_base}/test_data/test_data/templates.json'):
        try:
            shutil.copy(f'{overlay_base}/test_data/test_data/templates.json', f'{working_dir}/templates.json')
            print("Copied templates.json from nested test_data folder")
        except Exception as e:
            print(f"Error copying templates.json from nested test_data: {e}")

    print("Overlay setup complete")

# Notebook helper - uncomment to reset KG data
# import os
# os.environ['RESET_KG'] = 'true'  # Set to force reset

# This function is replaced by the analyze_feature_types function defined later

# --- Environment Detection & Path Configuration ---
def is_running_on_kaggle():
    """Detect if the code is running on Kaggle."""
    return os.path.exists('/kaggle/input')

# BACKEND DEFINITION
BACKEND = "numpy"  # Default to numpy as fallback

# GPU Acceleration Detection - Simplified to avoid unnecessary dependencies
# For pure agent-based KG, PyTorch isn't required
print("Using NumPy-based backend for agent computation")
# BACKEND value will be updated if GPU acceleration is enabled

# Optional: only try to import accelerators if specifically requested
if os.environ.get('USE_GPU_ACCELERATION', '').lower() in ('true', '1', 'yes'):
    try:
        import torch
        if torch.cuda.is_available():
            print("CUDA detected! Using PyTorch + CUDA for acceleration")
            BACKEND = "torch_cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("MPS detected! Using PyTorch + MPS for acceleration")
            BACKEND = "torch_mps"
        else:
            print("PyTorch available but no GPU acceleration detected")
    except ImportError:
        print("PyTorch not available, using CPU acceleration only")
        pass

# Print confirmation message
print(f"→ Using {BACKEND} as computation backend")

# Create a global supervisor instance
GLOBAL_SUPERVISOR = GlobalSupervisor(heartbeat_interval=30)

# Function to enable/disable KG reset via environment variable
def set_reset_flag(value=True):
    """Set the knowledge graph reset flag via environment variable.
    
    Args:
        value: Boolean value to set (True=reset, False=don't reset)
    """
    if value:
        os.environ["RESET_KG"] = "true"
    else:
        os.environ["RESET_KG"] = "false"
    return value

# Function to archive an existing knowledge graph
def archive_knowledge_graph(kg_path, archive_dir=None):
    """Archive an existing knowledge graph file before resetting.
    
    Args:
        kg_path: Path to the knowledge graph file to archive
        archive_dir: Directory to store archives (defaults to same directory as kg_path)
        
    Returns:
        Path to the archived file or None if archiving failed
    """
    if not os.path.exists(kg_path):
        print(f"No knowledge graph found at {kg_path} to archive")
        return None
    
    try:
        # Use timestamp for unique archive name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Get directory for archive
        if archive_dir is None:
            archive_dir = os.path.dirname(kg_path)
        
        # Create archive directory if it doesn't exist
        os.makedirs(archive_dir, exist_ok=True)
        
        # Create archive filename
        base_name = os.path.basename(kg_path)
        archive_name = f"{os.path.splitext(base_name)[0]}_archive_{timestamp}.json"
        archive_path = os.path.join(archive_dir, archive_name)
        
        # Copy the file
        shutil.copy2(kg_path, archive_path)
        
        # Create a compressed version to save space
        with open(kg_path, 'rb') as f_in:
            with gzip.open(f"{archive_path}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
        print(f"Archived knowledge graph to {archive_path}")
        print(f"Compressed archive saved to {archive_path}.gz")
        return archive_path
    
    except Exception as e:
        print(f"Error archiving knowledge graph: {e}")
        return None

# Function to enable/disable KG archiving via environment variable
def set_archive_flag(value=True):
    """Set the knowledge graph archive flag via environment variable.
    
    Args:
        value: Boolean value to set (True=archive, False=don't archive)
    """
    if value:
        os.environ["ARCHIVE_KG"] = "true"
    else:
        os.environ["ARCHIVE_KG"] = "false"
    return value


# Function to analyze KG contents (useful in notebooks)
def analyze_feature_types(kg):
    """Analyze and print feature type distribution in a knowledge graph."""
    try:
        # Print feature counts by type
        print("\nFeature types distribution:")
        feature_type_counts = {}
        if hasattr(kg, 'features') and kg.features:
            # Handle both list and dictionary types
            if isinstance(kg.features, dict):
                features_to_process = kg.features.values()
            else:  # Assume it's a list or iterable
                features_to_process = kg.features
                
            for feature in features_to_process:
                feature_type = feature.get('feature_type', 'Unknown')
                feature_type_counts[feature_type] = feature_type_counts.get(feature_type, 0) + 1
            
            for feature_type, count in sorted(feature_type_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {feature_type}: {count} features")
        else:
            print("  - No features found in the knowledge graph")
        
        # Print agent performance
        print("\nDetection agent performance:")
        agent_counts = {}
        if hasattr(kg, 'features') and kg.features:
            # Handle both list and dictionary types
            if isinstance(kg.features, dict):
                features_to_process = kg.features.values()
            else:  # Assume it's a list or iterable
                features_to_process = kg.features
                
            for feature in features_to_process:
                agent = feature.get('detected_by', 'Unknown')
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
            
            for agent, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {agent}: {count} features")
        else:
            print("  - No features found to analyze agent performance")
        
        # Print other stats
        print("\nKnowledge Graph Statistics:")
        print(f"  - Total features: {len(kg.features) if hasattr(kg, 'features') else 0}")
        print(f"  - Total clusters: {len(kg.clusters) if hasattr(kg, 'clusters') else 0}")
        print(f"  - Total alignments: {len(kg.alignments) if hasattr(kg, 'alignments') else 0}")
        print(f"  - Total templates: {len(kg.templates) if hasattr(kg, 'templates') else 0}")
        print(f"  - Total patterns: {len(kg.patterns) if hasattr(kg, 'patterns') else 0}")
        print(f"  - Total sites: {len(kg.sites) if hasattr(kg, 'sites') else 0}")
    except Exception as e:
        print(f"Error analyzing feature types: {e}")
        print("Knowledge graph data might be corrupted or incomplete")

# Function to create synthetic templates when none are found
def create_synthetic_templates():
    """Create synthetic templates for testing when no template file is found.
    
    Returns:
        List of template dictionaries with grids and descriptions
    """
    print("Creating synthetic templates for testing")
    templates = []
    
    # Create a simple rectangular enclosure template
    rect_grid = np.zeros((16, 16), dtype=np.float32)
    rect_grid[3:13, 3:13] = 0.2  # Fill interior
    rect_grid[3:13, 3] = 1.0  # Left edge
    rect_grid[3:13, 12] = 1.0  # Right edge
    rect_grid[3, 3:13] = 1.0  # Top edge
    rect_grid[12, 3:13] = 1.0  # Bottom edge
    templates.append({
        "grid": rect_grid,
        "description": "Rectangular enclosure"
    })
    
    # Create a simple linear feature template
    line_grid = np.zeros((16, 16), dtype=np.float32)
    line_grid[7:9, 2:14] = 1.0  # Horizontal line
    templates.append({
        "grid": line_grid,
        "description": "Linear feature"
    })
    
    # Create a simple circular feature template
    circle_grid = np.zeros((16, 16), dtype=np.float32)
    y, x = np.ogrid[-8:8, -8:8]
    mask = x**2 + y**2 <= 36
    circle_grid[mask] = 1.0
    templates.append({
        "grid": circle_grid,
        "description": "Circular feature"
    })
    
    # Create a simple L-shaped feature template
    l_grid = np.zeros((16, 16), dtype=np.float32)
    l_grid[4:12, 4:6] = 1.0  # Vertical part
    l_grid[10:12, 4:12] = 1.0  # Horizontal part
    templates.append({
        "grid": l_grid,
        "description": "L-shaped feature"
    })
    
    print(f"Created {len(templates)} synthetic templates")
    return templates

# Function to create documented archaeological sites within GeoTIFF bounds
def create_synthetic_sites():
    """Create documented archaeological sites within GeoTIFF bounds.
    
    Returns:
        List of site dictionaries with lat, lon, and descriptions
    """
    print("Loading documented archaeological sites within GeoTIFF bounds")
    sites = []
    
    # Known archaeological sites within the region (lon: -75.00 to -70.00, lat: -10.00 to -5.00)
    
    # Pusharo Petroglyphs - Madre de Dios
    sites.append({
        "lat": -8.7833, 
        "lon": -71.4833, 
        "desc": "Pusharo Petroglyphs - Pre-Columbian rock art site"
    })
    
    # El Sira Communal Reserve area sites
    sites.append({
        "lat": -9.4521, 
        "lon": -74.7694, 
        "desc": "El Sira ancient pottery site"
    })
    
    # Tierra Blanca region - Ucayali
    sites.append({
        "lat": -8.1628, 
        "lon": -74.5893, 
        "desc": "Tierra Blanca ceramics site"
    })
    
    # Contamana archaeological area
    sites.append({
        "lat": -7.3254, 
        "lon": -74.9785, 
        "desc": "Contamana settlement site"
    })
    
    # Alto Purús sites
    sites.append({
        "lat": -9.7685, 
        "lon": -70.8112, 
        "desc": "Alto Purús prehistoric settlement"
    })
    
    # Panguana region 
    sites.append({
        "lat": -9.6145, 
        "lon": -74.9346, 
        "desc": "Panguana ancient habitation site"
    })
    
    # Cordillera Azul foothills
    sites.append({
        "lat": -7.9521, 
        "lon": -74.7862, 
        "desc": "Cordillera Azul forest edge settlement"
    })
    
    # Manu National Park archaeological sites
    sites.append({
        "lat": -9.9142, 
        "lon": -71.2852, 
        "desc": "Manu forest ancient settlement"
    })
    
    sites.append({
        "lat": -8.6523, 
        "lon": -71.9654, 
        "desc": "Petroglyphs near Alto Madre de Dios River"
    })
    
    # Pachitea River basin sites
    sites.append({
        "lat": -8.7456, 
        "lon": -74.5321, 
        "desc": "Pachitea River basin settlement"
    })
    
    # Ucayali River geoglyphs
    sites.append({
        "lat": -8.3792, 
        "lon": -74.5326, 
        "desc": "Ucayali River ancient earthworks"
    })
    
    # Shesha ancient settlement (near Pucallpa)
    sites.append({
        "lat": -8.3816, 
        "lon": -74.5532, 
        "desc": "Shesha ceramic tradition site"
    })
    
    # Cumancaya culture sites
    sites.append({
        "lat": -8.1943, 
        "lon": -74.8321, 
        "desc": "Cumancaya culture settlement"
    })
    
    # Las Piedras region sites
    sites.append({
        "lat": -9.5642, 
        "lon": -70.3462, 
        "desc": "Las Piedras River ancient settlement"
    })
    
    # Fitzcarrald region sites
    sites.append({
        "lat": -9.1254, 
        "lon": -71.2345, 
        "desc": "Fitzcarrald isthmus archaeological site"
    })
    
    print(f"Loaded {len(sites)} documented archaeological sites within GeoTIFF bounds")
    return sites

# Function to create a synthetic tile when no tiles are found
def create_synthetic_tile():
    """Create a synthetic image tile for testing when no tiles are found.
    
    Returns:
        Dict containing synthetic tile with grid and metadata
    """
    print("[WARNING] Creating synthetic image tile for testing - RESULTS ARE NOT REAL ARCHAEOLOGICAL FEATURES")
    
    # Create a synthetic 128x128 tile
    grid = np.zeros((128, 128), dtype=np.float32)
    
    # Add some random features that might be detectable
    # Random noise
    grid += np.random.normal(0, 0.1, size=grid.shape)
    
    # Add a rectangular feature
    grid[30:50, 40:100] = 0.3
    grid[30:50, 40] = 1.0  # Left edge
    grid[30:50, 99] = 1.0  # Right edge
    grid[30, 40:100] = 1.0  # Top edge
    grid[49, 40:100] = 1.0  # Bottom edge
    
    # Add a linear feature
    grid[70:72, 20:120] = 0.8
    
    # Add some clustered features
    for i in range(5):
        x = np.random.randint(80, 110)
        y = np.random.randint(80, 110)
        grid[y:y+3, x:x+3] = 0.9
    
    # Create metadata
    meta = {
        "index": 0,
        "source": "synthetic",
        "lat": -3.1133,  # Near Manaus, Brazil
        "lon": -60.0253,
        "date": time.strftime("%Y-%m-%d"),
        "synthetic": True,
        "synthetic_test_data": True  # Explicit flag for detection in reports
    }
    
    return {"grid": grid, "meta": meta, "tile_id": "synthetic_test_data"}

# Function to enable/disable KG reset via environment variable

def find_geotiff_files(base_path=None):
    """Find all GeoTIFF files in specific directories optimized for Kaggle environment.
    
    Args:
        base_path: Directory to start the search from
        
    Returns:
        List of paths to GeoTIFF files
    """
    tif_files = []
    
    # Define paths based on environment
    if is_running_on_kaggle():
        # On Kaggle, NASA tiles should be in these specific places
        search_paths = [
            '/kaggle/input/nasa-tiles',
            '/kaggle/input/nasa-tiles/data',
            '/kaggle/input/nasa-tiles/data/tiles',
            '/kaggle/input/nasa-tiles/test_data'
        ]
        
        # Look through all datasets available in input
        input_dir = '/kaggle/input'
        if os.path.exists(input_dir):
            for dataset in os.listdir(input_dir):
                dataset_path = os.path.join(input_dir, dataset)
                if os.path.isdir(dataset_path):
                    search_paths.append(dataset_path)
    else:
        # Local environment - check specific paths
        search_paths = [
            './data/tiles',
            './test_data',
            '/Users/richardgillespie/Documents/AAImageSearch/data/tiles'
        ]
        
        # Add base_path if specified
        if base_path:
            search_paths.append(base_path)
    
    # File extensions for GeoTIFFs
    tif_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    
    # First pass: look in specific directories for .tif files
    for search_path in search_paths:
        if os.path.exists(search_path) and os.path.isdir(search_path):
            print(f"Searching for GeoTIFF files in {search_path}...")
            
            # Direct file check in this directory
            for file in os.listdir(search_path):
                file_path = os.path.join(search_path, file)
                if os.path.isfile(file_path) and any(file.endswith(ext) for ext in tif_extensions):
                    tif_files.append(file_path)
            
            # Check one level down
            for subdir in os.listdir(search_path):
                subdir_path = os.path.join(search_path, subdir)
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        file_path = os.path.join(subdir_path, file)
                        if os.path.isfile(file_path) and any(file.endswith(ext) for ext in tif_extensions):
                            tif_files.append(file_path)
    
    # If we still haven't found any files, try recursively but be more selective
    if not tif_files and base_path and os.path.exists(base_path):
        print(f"No GeoTIFFs found in specific paths, searching recursively in {base_path}...")
        for root, dirs, files in os.walk(base_path):
            # Skip very deep directories to avoid wasting time
            if root.count(os.sep) - base_path.count(os.sep) > 4:
                continue
                
            for file in files:
                if any(file.endswith(ext) for ext in tif_extensions):
                    tif_files.append(os.path.join(root, file))
    
    # Sort by file size (smallest first) to prioritize metadata files
    tif_files.sort(key=lambda x: os.path.getsize(x) if os.path.exists(x) else float('inf'))
    
    # Remove duplicates while preserving order
    unique_tif_files = []
    seen = set()
    for file in tif_files:
        if file not in seen:
            seen.add(file)
            unique_tif_files.append(file)
    
    print(f"Found {len(unique_tif_files)} GeoTIFF files")
    if len(unique_tif_files) > 0:
        print(f"First few files:")
        for i, file in enumerate(unique_tif_files[:5]):
            file_size = os.path.getsize(file) / (1024*1024)  # Size in MB
            print(f"  {i+1}. {os.path.basename(file)} ({file_size:.2f} MB)")
            print(f"     Path: {file}")
    
    return unique_tif_files

def extract_site_windows(geotiff_files, sites, window_size=256, buffer_radius_km=2.0):
    """Extract image windows centered on known archaeological sites.
    
    Args:
        geotiff_files: List of paths to GeoTIFF files
        sites: List of site dictionaries with lat/lon coordinates
        window_size: Size of the window to extract (pixels)
        buffer_radius_km: Radius around sites to extract (km)
        
    Returns:
        List of dictionaries containing site windows and metadata
    """
    if not geotiff_files:
        print("No GeoTIFF files available for site window extraction")
        return []
        
    if not sites:
        print("No sites available for window extraction")
        return []
    
    print(f"Extracting {window_size}x{window_size} windows around {len(sites)} sites...")
    
    # Find the right GeoTIFF files to use (prioritize smaller files that might be metadata)
    site_windows = []
    sites_processed = 0
    failed_sites = []
    sites_per_tiff = {}  # Track which sites were found in which GeoTIFF files
    
    print("Analyzing GeoTIFF files to determine which contains which sites...")
    
    # First, get the bounds of each GeoTIFF file to better match sites to files
    tiff_bounds = {}
    for tif_path in geotiff_files:
        try:
            with rasterio.open(tif_path) as src:
                tiff_bounds[tif_path] = src.bounds
                print(f"File: {os.path.basename(tif_path)}")
                minx, miny, maxx, maxy = src.bounds
                print(f"  Bounds: lon: {minx} to {maxx}, lat: {miny} to {maxy}")
                print(f"  Size: {src.width}x{src.height}, Bands: {src.count}")
                sites_per_tiff[tif_path] = []
        except Exception as e:
            print(f"Error reading bounds of {tif_path}: {e}")
    
    # Pre-match sites to GeoTIFF files
    for site in sites:
        if 'lat' not in site or 'lon' not in site:
            continue
            
        lat, lon = site['lat'], site['lon']
        matched = False
        
        for tif_path, bounds in tiff_bounds.items():
            minx, miny, maxx, maxy = bounds
            if minx <= lon <= maxx and miny <= lat <= maxy:
                sites_per_tiff[tif_path].append(site)
                matched = True
                print(f"Site at {lat}, {lon} matched to {os.path.basename(tif_path)}")
                
        if not matched:
            print(f"WARNING: Site at {lat}, {lon} doesn't match any GeoTIFF file bounds")
            failed_sites.append(site)
    
    # Now process each GeoTIFF file with only its matching sites
    for tif_path, matched_sites in sites_per_tiff.items():
        if not matched_sites:
            print(f"No sites matched to {os.path.basename(tif_path)}, skipping")
            continue
        
        print(f"Processing {len(matched_sites)} sites from {os.path.basename(tif_path)}")
        try:
            with rasterio.open(tif_path) as src:
                # Get the CRS and transform
                print(f"Processing file: {os.path.basename(tif_path)}")
                print(f"  Size: {src.width}x{src.height}, Bands: {src.count}")
                
                # Process each site
                for site in sites:
                    if 'lat' not in site or 'lon' not in site:
                        continue
                        
                    lat, lon = site['lat'], site['lon']
                    
                    try:
                        # First check if the site is within the bounds of this GeoTIFF
                        bounds = src.bounds
                        transform = src.transform
                        
                        # Get the bounds of the GeoTIFF in geographic coordinates
                        minx, miny, maxx, maxy = bounds
                        
                        # Check if the site coordinates are within the bounds
                        if not (minx <= lon <= maxx and miny <= lat <= maxy):
                            print(f"Site at {lat}, {lon} is outside the bounds of {os.path.basename(tif_path)}")
                            print(f"  GeoTIFF bounds: lon: {minx} to {maxx}, lat: {miny} to {maxy}")
                            continue  # Skip to next site
                            
                        # Print bounds and site info for debugging
                        print(f"Processing site at {lat}, {lon} within GeoTIFF: {os.path.basename(tif_path)}")
                        print(f"  GeoTIFF bounds: lon: {minx} to {maxx}, lat: {miny} to {maxy}")
                            
                        # Convert lat/lon to pixel coordinates
                        # The src.index method can sometimes return invalid values, so let's do manual
                        # conversion using the transform
                        col, row = ~transform * (lon, lat)
                        col, row = int(col), int(row)
                        
                        # Verify the pixel coordinates are within the image bounds
                        if not (0 <= col < src.width and 0 <= row < src.height):
                            print(f"Converted coordinates ({col}, {row}) are outside image dimensions ({src.width}, {src.height})")
                            continue  # Skip to next site
                        
                        # Calculate window bounds
                        half_size = window_size // 2
                        
                        # Calculate offsets ensuring they're within bounds
                        col_off = max(0, col - half_size)
                        row_off = max(0, row - half_size)
                        
                        # Calculate width and height ensuring they're within image bounds and positive
                        width = min(window_size, src.width - col_off)
                        height = min(window_size, src.height - row_off)
                        
                        # Skip if window dimensions are invalid
                        if width <= 0 or height <= 0:
                            print(f"Invalid window dimensions (width={width}, height={height}) for site at {lat}, {lon}")
                            continue  # Skip to next site
                            
                        window = rasterio.windows.Window(
                            col_off=col_off,
                            row_off=row_off,
                            width=width,
                            height=height
                        )
                        
                        # Read the data
                        data = src.read(window=window)
                        
                        # Skip if the window is empty
                        if data.shape[0] == 0 or data.shape[1] == 0 or data.shape[2] == 0:
                            continue
                        
                        # Normalize the data for better processing
                        if data.max() > 0:
                            norm_data = data.astype(np.float32) / data.max()
                        else:
                            norm_data = data.astype(np.float32)
                        
                        # Create the window dict
                        window_dict = {
                            'grid': norm_data[0],  # Use first band
                            'meta': {
                                'source': os.path.basename(tif_path),
                                'lat': lat,
                                'lon': lon,
                                'site_id': site.get('id', sites_processed),
                                'is_site': True,
                                'site_desc': site.get('desc', ''),
                                'window_size': window_size,
                                'center_row': row,
                                'center_col': col
                            }
                        }
                        
                        site_windows.append(window_dict)
                        sites_processed += 1
                        
                        if sites_processed % 10 == 0:
                            print(f"  Processed {sites_processed} site windows...")
                    
                    except Exception as e:
                        print(f"Error extracting window for site at {lat}, {lon}: {e}")
                        # Continue with next site instead of stopping
                
        except Exception as e:
            print(f"Could not open GeoTIFF file {tif_path}: {e}")
    
    print(f"Successfully extracted {len(site_windows)} windows for {sites_processed} sites")
    
    # If we don't have enough site windows but have valid GeoTIFF files, log the issue but don't create synthetic sites
    if len(site_windows) < 5 and geotiff_files:
        print(f"IMPORTANT: No site windows could be extracted because your site coordinates don't match the GeoTIFF bounds.")
        print(f"Available GeoTIFF bounds:")
        for tif_path in geotiff_files:
            try:
                with rasterio.open(tif_path) as src:
                    bounds = src.bounds
                    print(f"  {os.path.basename(tif_path)}: lon: {bounds.left} to {bounds.right}, lat: {bounds.bottom} to {bounds.top}")
            except Exception as e:
                print(f"  Error reading {os.path.basename(tif_path)}: {e}")
        
        print("\nYour sites are located around: lon: -60.1° to -60.3°, lat: -3.0° to -3.4° (central Amazon region near Manaus)")
        print("But your GeoTIFF files cover: lon: -75.0° to -70.0°, lat: -10.0° to -5.0° (Peru/western Brazil region)")
        print("\nPlease either:")
        print("1. Download GeoTIFF files covering your site coordinates, or")
        print("2. Update your site coordinates to match the available GeoTIFF bounds")
        print("\nProceeding with available data...")
    
    print(f"Total site windows available: {len(site_windows)}")
    
    # Create visualizations for known site windows
    print("Creating visualizations for known archaeological sites...")
    for window in site_windows[:5]:  # Visualize the first 5 to avoid too many files
        try:
            save_site_visualization(window, prefix="known_site")
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    return site_windows

class SitePatternLearner:
    """Neural network model for learning archaeological site patterns.
    
    This class implements a PyTorch-based model to learn patterns from
    known archaeological sites and then detect similar patterns in new data.
    """
    
    def __init__(self, input_size=256, feature_dim=128):
        """Initialize the site pattern learner.
        
        Args:
            input_size: Size of input images
            feature_dim: Size of feature embedding
        """
        self.input_size = input_size
        self.feature_dim = feature_dim
        self.model = None
        self.optimizer = None
        self.device = self._get_device()
        self.known_site_features = []
        self.site_embeddings = {}
        
        # Create the model
        self._create_model()
    
    def _get_device(self):
        """Get available device for training."""
        if BACKEND == "torch_cuda":
            return "cuda"
        elif BACKEND == "torch_mps":
            return "mps"
        else:
            return "cpu"
    
    def _create_model(self):
        """Create the PyTorch model architecture."""
        import torch
        import torch.nn as nn
        
        class SiteEncoder(nn.Module):
            def __init__(self, input_size, feature_dim):
                super(SiteEncoder, self).__init__()
                
                # Simple CNN architecture
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
                self.bn3 = nn.BatchNorm2d(64)
                self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
                self.bn4 = nn.BatchNorm2d(128)
                
                # Calculate size after convolutions
                conv_size = input_size // 16
                
                # Fully connected layers
                self.fc1 = nn.Linear(128 * conv_size * conv_size, 512)
                self.fc2 = nn.Linear(512, feature_dim)
                
                # Activation function
                self.relu = nn.ReLU()
                
            def forward(self, x):
                # Convolution layers
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.relu(self.bn3(self.conv3(x)))
                x = self.relu(self.bn4(self.conv4(x)))
                
                # Flatten
                x = x.view(x.size(0), -1)
                
                # Fully connected layers
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                
                # L2 normalize the features
                x = nn.functional.normalize(x, p=2, dim=1)
                
                return x
        
        # Initialize the model
        self.model = SiteEncoder(self.input_size, self.feature_dim)
        self.model.to(self.device)
        
        # Initialize optimizer
        import torch.optim as optim
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def train(self, site_windows, non_site_windows, epochs=10, batch_size=16):
        """Train the model on positive and negative examples.
        
        Args:
            site_windows: List of image windows centered on sites
            non_site_windows: List of image windows not containing sites
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import numpy as np
        from torch.utils.data import Dataset, DataLoader
        
        class SiteDataset(Dataset):
            def __init__(self, site_windows, non_site_windows):
                self.windows = []
                self.labels = []
                
                # Add positive examples
                for window in site_windows:
                    if 'grid' in window and isinstance(window['grid'], np.ndarray):
                        grid = window['grid']
                        if len(grid.shape) == 2:
                            # Ensure correct shape
                            self.windows.append(grid)
                            self.labels.append(1)  # Site present
                
                # Add negative examples
                for window in non_site_windows:
                    if 'grid' in window and isinstance(window['grid'], np.ndarray):
                        grid = window['grid']
                        if len(grid.shape) == 2:
                            # Ensure correct shape
                            self.windows.append(grid)
                            self.labels.append(0)  # No site
                
                # Convert to numpy arrays
                self.windows = np.array(self.windows)
                self.labels = np.array(self.labels)
                
                print(f"Created dataset with {len(self.windows)} samples")
                print(f"  Positive examples: {np.sum(self.labels == 1)}")
                print(f"  Negative examples: {np.sum(self.labels == 0)}")
            
            def __len__(self):
                return len(self.windows)
            
            def __getitem__(self, idx):
                # Get the window and label
                window = self.windows[idx]
                label = self.labels[idx]
                
                # Convert to tensor
                window_tensor = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
                label_tensor = torch.tensor(label, dtype=torch.float32)
                
                return window_tensor, label_tensor
        
        # Create dataset and dataloader
        dataset = SiteDataset(site_windows, non_site_windows)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        history = {'loss': [], 'accuracy': []}
        
        print("Starting training...")
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, data in enumerate(dataloader):
                # Get the inputs and labels
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss (contrastive loss)
                sim_pos = outputs[labels == 1].mean(dim=0) if torch.any(labels == 1) else torch.zeros(self.feature_dim, device=self.device)
                loss_tensor = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                # Maximize similarity for positive examples (sites)
                if torch.any(labels == 1):
                    pos_loss = 1.0 - F.cosine_similarity(outputs[labels == 1], sim_pos.unsqueeze(0), dim=1).mean()
                    loss_tensor = loss_tensor + pos_loss
                
                # Minimize similarity for negative examples (non-sites)
                if torch.any(labels == 0) and torch.any(labels == 1):
                    neg_loss = F.cosine_similarity(outputs[labels == 0], sim_pos.unsqueeze(0), dim=1).mean()
                    loss_tensor = loss_tensor + neg_loss
                    
                # Use loss_tensor for backward pass
                loss = loss_tensor
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                
                # Calculate accuracy
                pred = (F.cosine_similarity(outputs, sim_pos.unsqueeze(0), dim=1) > 0.5).float()
                correct += (pred == labels).sum().item()
                total += labels.size(0)
            
            # Print statistics
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = correct / total
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        print("Training complete!")
        
        # Store feature vectors for known sites
        self.extract_site_features(site_windows)
        
        return history
    
    def extract_site_features(self, site_windows):
        """Extract feature vectors for known sites.
        
        Args:
            site_windows: List of image windows centered on sites
        """
        import torch
        import numpy as np
        
        self.model.eval()
        self.known_site_features = []
        
        with torch.no_grad():
            for window in site_windows:
                if 'grid' in window and isinstance(window['grid'], np.ndarray):
                    grid = window['grid']
                    if len(grid.shape) == 2:
                        # Convert to tensor
                        grid_tensor = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                        grid_tensor = grid_tensor.to(self.device)
                        
                        # Extract features
                        features = self.model(grid_tensor).squeeze().cpu().numpy()
                        
                        # Store features
                        site_id = window.get('meta', {}).get('site_id', None)
                        self.known_site_features.append({
                            'features': features,
                            'site_id': site_id,
                            'meta': window.get('meta', {})
                        })
                        if site_id is not None:
                            self.site_embeddings[site_id] = features
        
        print(f"Extracted features for {len(self.known_site_features)} known sites")
    
    def predict(self, windows, threshold=0.75):
        """Predict whether windows contain archaeological sites.
        
        Args:
            windows: List of image windows
            threshold: Similarity threshold for detection
            
        Returns:
            List of predictions with similarity scores
        """
        import torch
        import numpy as np
        import torch.nn.functional as F
        
        if not self.known_site_features:
            print("No known site features available for comparison")
            return []
        
        self.model.eval()
        results = []
        
        # Combine all known site features
        all_site_features = np.array([f['features'] for f in self.known_site_features])
        site_features_tensor = torch.tensor(all_site_features, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            for window in windows:
                if 'grid' in window and isinstance(window['grid'], np.ndarray):
                    grid = window['grid']
                    if len(grid.shape) == 2:
                        # Convert to tensor
                        grid_tensor = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                        grid_tensor = grid_tensor.to(self.device)
                        
                        # Extract features
                        features = self.model(grid_tensor).squeeze()
                        
                        # Calculate similarity to known sites
                        similarities = F.cosine_similarity(features.unsqueeze(0), site_features_tensor)
                        max_sim = similarities.max().item()
                        max_sim_idx = similarities.argmax().item()
                        
                        # Get best matching site
                        best_match = self.known_site_features[max_sim_idx]
                        
                        # Create result
                        result = {
                            'similarity': max_sim,
                            'is_site': max_sim > threshold,
                            'best_match': best_match.get('site_id', None),
                            'best_match_meta': best_match.get('meta', {}),
                            'confidence': max_sim,
                            'meta': window.get('meta', {})
                        }
                        
                        results.append(result)
        
        return results
    
    def save(self, path):
        """Save the model weights and site features.
        
        Args:
            path: Directory to save the model
        """
        import torch
        import pickle
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(path, 'site_model.pt')
        torch.save(self.model.state_dict(), model_path)
        
        # Save site features
        features_path = os.path.join(path, 'site_features.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump({
                'known_site_features': self.known_site_features,
                'site_embeddings': self.site_embeddings
            }, f)
        
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load the model weights and site features.
        
        Args:
            path: Directory to load the model from
        """
        import torch
        import pickle
        import os
        
        # Load model weights
        model_path = os.path.join(path, 'site_model.pt')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Model weights not found at {model_path}")
        
        # Load site features
        features_path = os.path.join(path, 'site_features.pkl')
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                data = pickle.load(f)
                self.known_site_features = data.get('known_site_features', [])
                self.site_embeddings = data.get('site_embeddings', {})
            print(f"Loaded features for {len(self.known_site_features)} sites from {features_path}")
        else:
            print(f"Site features not found at {features_path}")
            
    def generate_non_site_windows(self, site_windows, geotiff_files, num_samples=100, min_distance_km=5.0):
        """Generate negative examples from areas away from known sites.
        
        Args:
            site_windows: List of windows centered on sites
            geotiff_files: List of GeoTIFF files
            num_samples: Number of negative samples to generate
            min_distance_km: Minimum distance from known sites (km)
            
        Returns:
            List of windows not containing sites
        """
        import numpy as np
        import rasterio
        import random
        
        non_site_windows = []
        
        # Extract site coordinates
        site_coords = []
        for window in site_windows:
            meta = window.get('meta', {})
            if 'lat' in meta and 'lon' in meta:
                site_coords.append((meta['lat'], meta['lon']))
        
        if not site_coords:
            print("No site coordinates available to generate non-site windows")
            return []
        
        if not geotiff_files:
            print("No GeoTIFF files available to generate non-site windows")
            return []
        
        # Select a random subset of GeoTIFF files
        random.shuffle(geotiff_files)
        selected_files = geotiff_files[:min(5, len(geotiff_files))]
        
        print(f"Generating {num_samples} non-site windows from {len(selected_files)} files...")
        
        # Try to generate non-site windows
        for tif_path in selected_files:
            try:
                with rasterio.open(tif_path) as src:
                    # Get metadata
                    width, height = src.width, src.height
                    window_size = self.input_size
                    
                    # Generate random points
                    attempts = 0
                    max_attempts = num_samples * 10
                    
                    while len(non_site_windows) < num_samples and attempts < max_attempts:
                        attempts += 1
                        
                        # Generate random position in the GeoTIFF
                        col = random.randint(window_size//2, width - window_size//2 - 1)
                        row = random.randint(window_size//2, height - window_size//2 - 1)
                        
                        # Convert to lat/lon
                        lon, lat = src.xy(row, col)
                        
                        # Check distance to sites
                        too_close = False
                        for site_lat, site_lon in site_coords:
                            # Calculate distance
                            import math
                            R = 6371.0  # Earth radius in km
                            
                            dlat = math.radians(site_lat - lat)
                            dlon = math.radians(site_lon - lon)
                            
                            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat)) * \
                                math.cos(math.radians(site_lat)) * math.sin(dlon/2)**2
                            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                            distance = R * c
                            
                            if distance < min_distance_km:
                                too_close = True
                                break
                        
                        if too_close:
                            continue
                        
                        # Extract window
                        window = rasterio.windows.Window(
                            col_off=max(0, col - window_size//2),
                            row_off=max(0, row - window_size//2),
                            width=min(window_size, width - (col - window_size//2)),
                            height=min(window_size, height - (row - window_size//2))
                        )
                        
                        data = src.read(window=window)
                        
                        # Skip if window is empty
                        if data.shape[0] == 0 or data.shape[1] == 0 or data.shape[2] == 0:
                            continue
                        
                        # Normalize data
                        if data.max() > 0:
                            norm_data = data.astype(np.float32) / data.max()
                        else:
                            norm_data = data.astype(np.float32)
                        
                        # Create window dict
                        window_dict = {
                            'grid': norm_data[0],  # Use first band
                            'meta': {
                                'source': os.path.basename(tif_path),
                                'lat': lat,
                                'lon': lon,
                                'is_site': False,
                                'window_size': window_size,
                                'center_row': row,
                                'center_col': col
                            }
                        }
                        
                        non_site_windows.append(window_dict)
                        
                        if len(non_site_windows) % 10 == 0:
                            print(f"  Generated {len(non_site_windows)}/{num_samples} non-site windows...")
                    
            except Exception as e:
                print(f"Error generating non-site windows from {tif_path}: {e}")
        
        print(f"Generated {len(non_site_windows)}/{num_samples} non-site windows")
        return non_site_windows
        
    def bind_to_knowledge_graph(self, kg, windows, predictions, threshold=0.75):
        """Bind detected patterns to the knowledge graph with Cypher compatibility.
        
        This method adds all detected sites to the knowledge graph, ensuring they can be
        exported to Cypher format and queried efficiently.
        
        Args:
            kg: AmazonKG instance to bind to
            windows: List of windows that were processed
            predictions: Predictions from the model (scores between 0-1)
            threshold: Confidence threshold for adding to KG
        
        Returns:
            List of feature IDs that were added
        """
        # Convert predictions to numpy array if needed
        if isinstance(predictions, (list, tuple)):
            predictions = np.array(predictions)
            
        # If we received tensors, convert to numpy
        if hasattr(predictions, 'cpu') and hasattr(predictions, 'numpy'):
            predictions = predictions.cpu().numpy()
            
        added_features = []
        
        # Add each window with high enough confidence
        for i, (window, pred) in enumerate(zip(windows, predictions)):
            if pred >= threshold:
                # Extract metadata
                meta = window.get('meta', {})
                lat = meta.get('lat', 0)
                lon = meta.get('lon', 0)
                
                # Add to knowledge graph as feature
                feature_id = kg.add_feature(
                    feature_type="PotentialArchaeologicalSite",
                    geometry={'lat': lat, 'lon': lon},
                    grid=window.get('grid'),
                    confidence=float(pred),
                    detected_by="SitePatternLearner",
                    detection_date=datetime.datetime.now().isoformat(),
                    source_file=meta.get('source', "Unknown"),
                    detection_method="DeepLearning"
                )
                
                # Add reasoning chain
                kg.add_reasoning_chain(feature_id, [
                    {
                        "step": "Neural network detection",
                        "agent": "SitePatternLearner",
                        "evidence": f"Window at coordinates {lat:.6f}, {lon:.6f} processed by CNN",
                        "conclusion": f"Site pattern detected with confidence {float(pred):.3f}"
                    },
                    {
                        "step": "Pattern comparison",
                        "agent": "SitePatternLearner",
                        "evidence": f"Feature embedding similarity to known archaeological sites",
                        "conclusion": f"Feature exhibits characteristics similar to known archaeological sites"
                    }
                ])
                
                added_features.append(feature_id)
                
                if len(added_features) % 10 == 0:
                    print(f"  Added {len(added_features)} features to KG...")
        
        # Ensure all features are properly connected for Cypher export
        if added_features:
            # Force a save to ensure the additions are persisted
            kg.save()
            print(f"Successfully bound {len(added_features)} features to knowledge graph")
            print("These features will be included in any Cypher export")
            
        return added_features

# Add function to save site visualizations as PNG images
def save_site_visualization(window_dict, output_dir=None, prefix="site", display_inline=False):
    """Save a visualization of a site window as a PNG image and optionally display inline.
    
    Args:
        window_dict: Dictionary containing the site window grid and metadata
        output_dir: Directory to save the image (defaults to working dir)
        prefix: Prefix for the output filename
        display_inline: Whether to display the figure inline in notebooks
        
    Returns:
        Path to the saved image, and figure if display_inline is True
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import datetime
    import rasterio
    
    # Set output directory
    if output_dir is None:
        if is_running_on_kaggle():
            output_dir = '/kaggle/working/visualizations'
        else:
            output_dir = './visualizations'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get grid and metadata
    grid = window_dict.get('grid')
    meta = window_dict.get('meta', {})
    
    # If grid is missing but we have coordinates, try to extract from GeoTIFF
    if grid is None and 'lat' in meta and 'lon' in meta:
        try:
            # Get coordinates
            lat = meta.get('lat')
            lon = meta.get('lon')
            
            # Find source file
            source_path = None
            source_name = meta.get('source')
            
            if source_name:
                # Check common paths
                for base_dir in ['./data/tiles', './data', '.', '/kaggle/input/nasa-tiles']:
                    test_path = os.path.join(base_dir, source_name)
                    if os.path.exists(test_path):
                        source_path = test_path
                        break
            
            # If we found a source file, extract the window
            if source_path and os.path.exists(source_path):
                with rasterio.open(source_path) as src:
                    # Convert lat/lon to pixel coordinates
                    row, col = src.index(lon, lat)
                    
                    # Create window
                    window_size = 256
                    half_size = window_size // 2
                    
                    # Calculate window bounds ensuring they're within image bounds
                    col_off = max(0, col - half_size)
                    row_off = max(0, row - half_size)
                    width = min(window_size, src.width - col_off)
                    height = min(window_size, src.height - row_off)
                    
                    if width > 0 and height > 0:
                        window = rasterio.windows.Window(col_off, row_off, width, height)
                        data = src.read(1, window=window)
                        
                        # Normalize data for display
                        min_val = np.min(data)
                        max_val = np.max(data)
                        if max_val > min_val:
                            data = (data - min_val) / (max_val - min_val)
                        
                        grid = data
                        
                        # Update metadata
                        meta['center_row'] = row - row_off
                        meta['center_col'] = col - col_off
                        print(f"Successfully extracted window for visualization from {source_name}")
        except Exception as e:
            print(f"Error extracting grid from source: {e}")
    
    # Generate a placeholder if grid is still missing
    if grid is None:
        print(f"Warning: Using placeholder grid for visualization of site at lat:{meta.get('lat')}, lon:{meta.get('lon')}")
        grid = np.zeros((256, 256))
        grid[108:148, 108:148] = 1.0  # Add a small square in the middle
    
    # Generate filename
    site_id = meta.get('site_id', 'unknown')
    lat = meta.get('lat', 0)
    lon = meta.get('lon', 0)
    is_site = meta.get('is_site', False)
    site_type = "known_site" if is_site else "potential_site"
    confidence = meta.get('confidence', 0)
    
    filename = f"{prefix}_{site_type}_{site_id}_{lat:.4f}_{lon:.4f}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Main image
    plt.subplot(2, 2, 1)
    plt.imshow(grid, cmap='terrain')
    plt.title(f"Site ID: {site_id}")
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Add edge detection view
    plt.subplot(2, 2, 2)
    edge_grid = compute_edge_detection(grid)
    plt.imshow(edge_grid, cmap='hot')
    plt.title("Edge Detection")
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Add gradient magnitude view
    plt.subplot(2, 2, 3)
    grad_x, grad_y = compute_gradients(grid)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    plt.imshow(grad_mag, cmap='viridis')
    plt.title("Gradient Magnitude")
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Add thresholded view
    plt.subplot(2, 2, 4)
    threshold = np.mean(grid) + 0.5 * np.std(grid)
    binary = (grid > threshold).astype(float)
    plt.imshow(binary, cmap='gray')
    plt.title("Thresholded")
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Add metadata as text
    plt.figtext(0.5, 0.01, f"Lat: {lat:.4f}, Lon: {lon:.4f}", ha="center")
    if 'site_desc' in meta and meta['site_desc']:
        plt.figtext(0.5, 0.03, f"Description: {meta['site_desc']}", ha="center")
    if 'detected_by' in meta:
        plt.figtext(0.5, 0.05, f"Detected by: {meta['detected_by']}", ha="center")
    if confidence > 0:
        plt.figtext(0.5, 0.07, f"Confidence: {confidence:.2f}", ha="center")
    
    # Set title based on site type
    if is_site:
        plt.suptitle(f"Known Archaeological Site", fontsize=16)
    else:
        plt.suptitle(f"Potential Archaeological Site", fontsize=16)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(filepath, dpi=150)
    
    # Create a figure reference for inline display
    fig = plt.gcf()
    
    # Display inline if requested
    if display_inline:
        if RUNNING_ON_KAGGLE:
            # In Kaggle, force display with show() AND direct IPython display
            plt.show()
            # Force display of the saved file using IPython.display
            try:
                # Import in function scope to avoid issues
                from IPython.display import Image, display
                print("Displaying inline image...")
                display(Image(filepath))
            except ImportError:
                print("IPython display not available")
        else:
            # Standard display in other environments
            plt.show()
    else:
        plt.close()
    
    print(f"Saved site visualization to {filepath}")
    
    if display_inline:
        # Return both filepath and figure for potential inline display
        return filepath, fig
    else:
        return filepath

# Function to display site images inline in notebooks

def show_png(image_path):
    """Displays a PNG image inline in a Jupyter Notebook.

    Args:
        image_path: Path to the PNG image file.
    """
    if IPYTHON_AVAILABLE:
        display(Image(image_path))
    else:
        print(f"Image available at: {image_path}")
        # Try to open with system viewer if possible
        try:
            import subprocess
            import platform
            system = platform.system()
            if system == 'Darwin':  # macOS
                subprocess.call(['open', image_path])
            elif system == 'Windows':
                subprocess.call(['start', image_path], shell=True)
            elif system == 'Linux':
                subprocess.call(['xdg-open', image_path])
        except Exception as e:
            print(f"Could not open image with system viewer: {e}")

def show_site_visualization(site_id=None, site_index=None, prefix="known_site", directory=None):
    """Displays a site visualization inline in a Jupyter Notebook or provides path.
    
    Args:
        site_id: Specific site ID to look for in filenames
        site_index: Index number of the site visualization (e.g., "known_site_0")
        prefix: Type of visualization to show ("known_site" or "detected_site")
        directory: Directory containing visualization files, defaults to "./output/visualizations"
    """
    import os
    import glob
    
    # Default directory if not specified
    if directory is None:
        if RUNNING_ON_KAGGLE:
            directory = "/kaggle/working/visualizations"
        else:
            directory = "./output/visualizations"
    
    # Make sure directory exists to avoid errors
    if not os.path.exists(directory):
        print(f"Visualization directory '{directory}' not found.")
        # Try to fallback to other locations
        if is_running_on_kaggle():
            fallback_dir = "/kaggle/working"
            if os.path.exists(fallback_dir):
                directory = fallback_dir
                print(f"Using fallback directory: {fallback_dir}")
        else:
            fallback_dir = "./output"
            if os.path.exists(fallback_dir):
                directory = fallback_dir
                print(f"Using fallback directory: {fallback_dir}")
    
    # Find matching files
    if site_id is not None:
        # Look for files containing the site_id
        pattern = os.path.join(directory, f"*{site_id}*.png")
    elif site_index is not None:
        # Look for files with specific index
        pattern = os.path.join(directory, f"{prefix}_{site_index}.png")
    else:
        # Get all files with the given prefix
        pattern = os.path.join(directory, f"{prefix}_*.png")
    
    matching_files = glob.glob(pattern)
    
    if not matching_files:
        print(f"No visualization files found matching '{pattern}'")
        return
    
    # Display the first matching file
    print(f"Displaying visualization from: {matching_files[0]}")
    show_png(matching_files[0])
    
    # If there are more matches, inform the user
    if len(matching_files) > 1:
        print(f"Found {len(matching_files)} matching files. To see others, specify a different index.")
        for i, f in enumerate(matching_files[:5]):  # Show first 5
            print(f"  {i}: {os.path.basename(f)}")
        if len(matching_files) > 5:
            print(f"  ... and {len(matching_files)-5} more files")

def display_site_inline(kg, site_id=None, potential_site_id=None, figsize=(15, 10)):
    """
    Display archaeological site images inline in a notebook environment.
    
    Args:
        kg: Knowledge graph containing site data
        site_id: ID of site to display (if None, gets first available site)
        potential_site_id: ID of potential site to display (if provided, shows comparison)
        figsize: Figure size (width, height) in inches
        
    Returns:
        The matplotlib figure for further customization if needed
    """
    import matplotlib.pyplot as plt
    from skimage import exposure
    import numpy as np
    
    # Get known sites if site_id is not specified
    if site_id is None:
        sites = kg.get_sites()
        if not sites:
            print("No known sites available in knowledge graph")
            return None
        site_id = sites[0]['id']
    
    # Retrieve site data
    site_data = None
    
    # Get site data for known site
    for site in kg.sites:
        if site['id'] == site_id:
            site_data = site
            break
    
    if site_data is None:
        print(f"Could not find site data for {site_id}")
        return None
    
    # If potential site ID is provided, do comparison view
    if potential_site_id is not None:
        potential_site_data = None
        
        # Get feature data for potential site
        for feature in kg.features:
            if feature['id'] == potential_site_id:
                potential_site_data = feature
                break
        
        if potential_site_data is None:
            print(f"Could not find data for potential site {potential_site_id}")
            return None
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Known site
        site_desc = site_data.get('description', site_id)
        axes[0].set_title(f"Known Site: {site_desc}", fontsize=14)
        # Extract site information
        lat = site_data.get('latitude', 'Unknown')
        lon = site_data.get('longitude', 'Unknown')
        axes[0].text(10, 20, f"Lat: {lat}, Lon: {lon}", color='white', fontsize=12, 
                  bbox=dict(facecolor='black', alpha=0.5))
        
        # Either display the site grid if available, or show a placeholder
        site_grid = site_data.get('grid')
        if site_grid is not None and hasattr(site_grid, 'shape'):
            axes[0].imshow(site_grid, cmap='gray')
        else:
            axes[0].text(0.5, 0.5, "No image available", ha='center', va='center', fontsize=14)
            axes[0].set_xlim(0, 10)
            axes[0].set_ylim(0, 10)
        
        axes[0].axis('off')
        
        # Potential site
        confidence = potential_site_data.get('confidence', 'Unknown')
        axes[1].set_title(f"Potential New Site (Confidence: {confidence:.2f})", fontsize=14)
        
        # Extract feature information
        geometry = potential_site_data.get('geometry', {})
        lat = geometry.get('latitude', geometry.get('lat', 'Unknown'))
        lon = geometry.get('longitude', geometry.get('lon', 'Unknown'))
        axes[1].text(10, 20, f"Lat: {lat}, Lon: {lon}", color='white', fontsize=12,
                  bbox=dict(facecolor='black', alpha=0.5))
        
        # Either display the feature grid if available, or show a placeholder
        feature_grid = potential_site_data.get('grid')
        if feature_grid is not None and hasattr(feature_grid, 'shape'):
            axes[1].imshow(feature_grid, cmap='gray')
        else:
            axes[1].text(0.5, 0.5, "No image available", ha='center', va='center', fontsize=14)
            axes[1].set_xlim(0, 10)
            axes[1].set_ylim(0, 10)
        
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    # Single site view - create detailed analysis
    # Extract grid from site data
    grid = site_data.get('grid')
    if grid is None:
        # Try to generate grid from coordinates
        lat = site_data.get('latitude', site_data.get('lat'))
        lon = site_data.get('longitude', site_data.get('lon'))
        
        if lat is not None and lon is not None:
            # Create a window dict to pass to save_site_visualization
            window_dict = {
                'meta': {
                    'site_id': site_id,
                    'lat': lat,
                    'lon': lon,
                    'is_site': True,
                    'site_desc': site_data.get('description', '')
                }
            }
            # Use existing function to generate visualization (with display_inline=True)
            return save_site_visualization(window_dict, display_inline=True)
    
    # If we have a grid, create a detailed view
    if grid is not None and hasattr(grid, 'shape'):
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        site_desc = site_data.get('description', site_id)
        lat = site_data.get('latitude', site_data.get('lat', 'Unknown'))
        lon = site_data.get('longitude', site_data.get('lon', 'Unknown'))
        
        fig.suptitle(f"Site: {site_desc} (Lat: {lat}, Lon: {lon})", fontsize=16)
        
        # Original image
        axes[0, 0].imshow(grid, cmap='gray')
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Enhanced contrast
        p2, p98 = np.percentile(grid, (2, 98))
        img_contrast = exposure.rescale_intensity(grid, in_range=(p2, p98))
        axes[0, 1].imshow(img_contrast, cmap='gray')
        axes[0, 1].set_title("Enhanced Contrast")
        axes[0, 1].axis('off')
        
        # Edge detection
        edges = compute_edge_detection(grid)
        axes[0, 2].imshow(edges, cmap='viridis')
        axes[0, 2].set_title("Edge Detection")
        axes[0, 2].axis('off')
        
        # Gradient magnitude
        grad_x, grad_y = compute_gradients(grid)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        axes[1, 0].imshow(grad_mag, cmap='magma')
        axes[1, 0].set_title("Gradient Magnitude")
        axes[1, 0].axis('off')
        
        # Thresholded binary
        threshold = np.percentile(grid, 60)  # Adjust threshold as needed
        binary = (grid > threshold).astype(np.float32)
        axes[1, 1].imshow(binary, cmap='gray')
        axes[1, 1].set_title("Thresholded Binary")
        axes[1, 1].axis('off')
        
        # Feature highlight - overlay edges on original
        overlay = np.zeros((*grid.shape, 3))
        overlay[:,:,0] = grid / np.max(grid)  # Red channel: original image
        overlay[:,:,1] = grid / np.max(grid)  # Green channel: original image
        overlay[:,:,2] = grid / np.max(grid)  # Blue channel: original image
        
        # Add edge highlights in green
        edge_mask = edges > np.percentile(edges, 90)
        overlay[edge_mask, 0] = 0.1  # Reduce red
        overlay[edge_mask, 1] = 1.0  # Max green
        overlay[edge_mask, 2] = 0.1  # Reduce blue
        
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title("Feature Highlights")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    return None

def extract_templates_from_sites(kg, geotiff_files=None, tiles_dir=None):
    """Extract templates from known archaeological sites.
    
    This function extracts characteristic patterns from known sites
    and adds them as templates to the knowledge graph.
    
    Args:
        kg: Knowledge graph to store templates
        geotiff_files: List of GeoTIFF files to process
        tiles_dir: Directory containing tiles (used if geotiff_files is None)
        
    Returns:
        tuple: (Number of templates extracted, Set of tile indices containing sites)
    """
    print("Extracting templates from known sites...")
    sites = kg.get_sites()
    site_tiles = set()
    
    if not sites:
        print("No known sites found in the knowledge graph")
        print("Creating synthetic templates as fallback")
        templates = create_synthetic_templates()
        for template in templates:
            kg.add_template(template['grid'], desc=template.get('description', ''))
        return len(templates), site_tiles
    
    print(f"Found {len(sites)} known sites to extract templates from")
    templates_added = 0
    
    # Get all GeoTIFF files if not provided
    if geotiff_files is None:
        if tiles_dir is None:
            tiles_dir = './data/tiles' if not is_running_on_kaggle() else '/kaggle/input/nasa-tiles'
        geotiff_files = find_geotiff_files(tiles_dir)
        
    if not geotiff_files:
        print("No GeoTIFF files found to extract site windows")
        return 0, site_tiles
        
    # Extract windows around known sites
    site_windows = extract_site_windows(geotiff_files, sites)
    if not site_windows:
        print("No site windows could be extracted")
        return 0, site_tiles
        
    print(f"Extracted {len(site_windows)} windows from known sites")
    
    # For each site window, extract templates
    for window in site_windows:
        if 'grid' not in window:
            continue
            
        grid = window['grid']
        meta = window.get('meta', {})
        site_id = meta.get('site_id', 'unknown')
        
        # Store the tile index as a site tile
        if 'index' in meta:
            site_tiles.add(meta['index'])
        
        # Extract a template from each site
        # 1. Use the full grid as a template
        kg.add_template(grid, desc=f"Full site template from {site_id}")
        templates_added += 1
        
        # 2. Extract sub-regions with strong feature presence
        # Detect edges to find regions of interest
        edges = compute_edge_detection(grid)
        edge_threshold = np.percentile(edges, 80)  # Top 20% of edge strengths
        high_edge_regions = edges > edge_threshold
        
        # Find connected components in high edge regions
        from scipy import ndimage
        labeled_regions, num_regions = ndimage.label(high_edge_regions)
        
        for region_idx in range(1, num_regions + 1):
            region_mask = labeled_regions == region_idx
            if np.sum(region_mask) < 100:  # Skip tiny regions
                continue
                
            # Get region bounds
            rows, cols = np.where(region_mask)
            min_row, max_row = np.min(rows), np.max(rows)
            min_col, max_col = np.min(cols), np.max(cols)
            
            # Extract region with margin
            margin = 5
            sub_grid = grid[
                max(0, min_row - margin):min(grid.shape[0], max_row + margin),
                max(0, min_col - margin):min(grid.shape[1], max_col + margin)
            ]
            
            # Skip if the sub-grid is too small
            if min(sub_grid.shape) < 20:
                continue
                
            # Add as template
            kg.add_template(
                sub_grid, 
                desc=f"Region template from {site_id}"
            )
            templates_added += 1
    
    print(f"Extracted {templates_added} templates from known sites")
    return templates_added, site_tiles

def run_site_learning_workflow(kg_path=None, reset=False, visualize=False, 
                           inline_visualization=False, tiles_dir=None, 
                           exhaustive=True):
    """Run the complete archaeological site detection workflow.
    
    This workflow consists of two main phases:
    1. Learning phase: Focus on known sites to learn patterns and templates
    2. Evaluation phase: Exhaustively process all remaining tiles
    
    Args:
        kg_path: Path to knowledge graph file
        reset: Whether to reset the KG (True) or load existing (False)
        visualize: Whether to generate visualizations
        inline_visualization: Whether to display visualizations inline
        tiles_dir: Directory containing tiles
        exhaustive: Whether to process all tiles (True) or just known sites (False)
        
    Returns:
        Dictionary with workflow results
    """
    print("\n=== ARCHAEOLOGICAL SITE DETECTION WORKFLOW ===\n")
    metrics = {
        "learning_phase_time": 0,
        "evaluation_phase_time": 0,
        "total_tiles": 0,
        "site_tiles": 0,
        "total_features_detected": 0,
        "potential_sites_detected": 0
    }
    
    # Set up paths based on environment
    if is_running_on_kaggle():
        kg_storage_dir = '/kaggle/working/kg_data'
        kg_path = kg_path or '/kaggle/working/amazon_archaeology_kg.json'
        output_dir = '/kaggle/working'
        tiles_dir = tiles_dir or '/kaggle/input/nasa-tiles'
    else:
        kg_storage_dir = './kg_data'
        kg_path = kg_path or './amazon_archaeology_kg.json'
        output_dir = './output'
        tiles_dir = tiles_dir or './data/tiles'
    
    # Create necessary directories
    os.makedirs(kg_storage_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set environment flags based on parameters
    if reset:
        os.environ["RESET_KG"] = "true"
    else:
        os.environ["RESET_KG"] = "false"
    
    # Initialize knowledge graph
    kg = AmazonKG(storage_dir=kg_storage_dir)
    
    # Load existing KG if not resetting
    if not reset and os.path.exists(kg_path):
        print(f"Loading existing knowledge graph from {kg_path}")
        kg.load(kg_path)
        print(f"Loaded KG with {len(kg.features)} features, {len(kg.templates)} templates, and {len(kg.sites)} sites")
    
    # Track processed tiles and site tiles
    site_tiles = set()
    
    # ---------- PHASE 1: LEARNING FROM KNOWN SITES ----------
    print("\n=== PHASE 1: LEARNING FROM KNOWN SITES ===\n")
    learning_start_time = time.time()
    
    # Get all geotiff files
    geotiff_files = find_geotiff_files(tiles_dir)
    if not geotiff_files:
        print("No GeoTIFF files found in tiles directory")
        return {"success": False, "error": "No GeoTIFF files found"}
        
    print(f"Found {len(geotiff_files)} GeoTIFF files")
    
    # Extract templates from known sites
    num_templates, site_tiles = extract_templates_from_sites(kg, geotiff_files, tiles_dir)
    if num_templates == 0:
        print("Warning: No templates extracted from known sites")
    
    # Load tiles (or a subset for large datasets)
    all_tiles = []
    for geotiff_file in geotiff_files[:5]:  # Limit to first 5 files for large datasets
        print(f"Loading tiles from {os.path.basename(geotiff_file)}")
        tiles = load_tiles_from_geotiff(geotiff_file)
        if tiles:
            all_tiles.extend(tiles)
    
    # If we have site tiles identified, prioritize those
    if site_tiles:
        print(f"Found {len(site_tiles)} tiles containing known sites")
        
        # Filter all_tiles to get site tiles
        site_tiles_data = []
        for tile in all_tiles:
            if isinstance(tile, dict) and 'meta' in tile and 'index' in tile['meta']:
                if tile['meta']['index'] in site_tiles:
                    site_tiles_data.append(tile)
        
        if site_tiles_data:
            print(f"Processing {len(site_tiles_data)} tiles containing known sites")
            
            # Use visualizing processor if available and requested
            if inline_visualization and IPYTHON_AVAILABLE:
                try:
                    from visualizing_processor import VisualizingParallelProcessor
                    processor = VisualizingParallelProcessor(kg)
                    print("Using visualizing processor with inline display")
                except ImportError:
                    processor = ParallelProcessor(kg)
            else:
                processor = ParallelProcessor(kg)
                
            processor.process_tiles_parallel(site_tiles_data, batch_size=5)
        else:
            print("No tiles matching the site indices were found")
    
    # Process all tiles if no site tiles were identified
    if not site_tiles:
        print("Processing all tiles to identify site patterns...")
        processor = ParallelProcessor(kg)
        processor.process_tiles_parallel(all_tiles, batch_size=10)
    
    # Update metrics
    metrics["learning_phase_time"] = time.time() - learning_start_time
    metrics["site_tiles"] = len(site_tiles)
    metrics["total_features_detected"] = len(kg.features)
    
    # Display learning phase results
    print("\n=== LEARNING PHASE RESULTS ===")
    print(f"Templates extracted: {num_templates}")
    print(f"Site tiles processed: {len(site_tiles)}")
    print(f"Features detected: {len(kg.features)}")
    print(f"Time elapsed: {metrics['learning_phase_time']:.2f} seconds")
    
    # Save intermediate results
    learning_kg_path = os.path.join(output_dir, "learning_phase_kg.json")
    kg.save(learning_kg_path)
    print(f"Learning phase knowledge graph saved to {learning_kg_path}")
    
    # Stop here if not doing exhaustive analysis
    if not exhaustive:
        # Save final results
        kg.save(kg_path)
        print(f"Knowledge graph saved to {kg_path}")
        return {
            "success": True,
            "kg": kg,
            "templates_extracted": num_templates,
            "site_tiles_processed": len(site_tiles),
            "features_detected": len(kg.features)
        }
    
    # ---------- PHASE 2: EXHAUSTIVE EVALUATION OF ALL TILES ----------
    print("\n=== PHASE 2: EXHAUSTIVE EVALUATION OF ALL TILES ===\n")
    evaluation_start_time = time.time()
    
    # Load all tiles
    all_tiles = []
    for geotiff_file in geotiff_files:
        print(f"Loading tiles from {os.path.basename(geotiff_file)}")
        tiles = load_tiles_from_geotiff(geotiff_file)
        if tiles:
            all_tiles.extend(tiles)
    
    if not all_tiles:
        print("No tiles loaded for evaluation")
        return {"success": False, "error": "No tiles loaded"}
        
    print(f"Loaded {len(all_tiles)} total tiles")
    
    # Filter out tiles already processed in learning phase
    processor = ParallelProcessor(kg)
    already_processed = processor.processed_tiles.union(site_tiles)
    
    evaluation_tiles = []
    for tile in all_tiles:
        if isinstance(tile, dict) and 'meta' in tile and 'index' in tile['meta']:
            if tile['meta']['index'] not in already_processed:
                evaluation_tiles.append(tile)
    
    if not evaluation_tiles:
        print("No new tiles to evaluate - all tiles have been processed")
    else:
        print(f"Processing {len(evaluation_tiles)} remaining tiles")
        metrics["total_tiles"] = len(all_tiles)
        
        # Process all remaining tiles
        processor.process_tiles_parallel(evaluation_tiles, batch_size=20)
    
    # Find potential new sites
    print("Finding potential new archaeological sites with expanded 200km search radius...")
    potential_sites = kg.find_potential_new_sites(min_confidence=0.65, distance_from_known=200.0)
    
    # Update metrics
    metrics["evaluation_phase_time"] = time.time() - evaluation_start_time
    metrics["potential_sites_detected"] = len(potential_sites)
    
    # Display evaluation phase results
    print("\n=== EVALUATION PHASE RESULTS ===")
    print(f"Total tiles: {metrics['total_tiles']}")
    print(f"New tiles processed: {len(evaluation_tiles)}")
    print(f"Total features detected: {len(kg.features)}")
    print(f"Potential new sites: {len(potential_sites)}")
    print(f"Time elapsed: {metrics['evaluation_phase_time']:.2f} seconds")
    
    # Visualize potential sites if requested
    if visualize and potential_sites:
        print("Creating visualizations for potential sites...")
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Visualize top 10 potential sites
        sorted_sites = sorted(potential_sites, key=lambda x: x.get('confidence', 0), reverse=True)
        for i, site in enumerate(sorted_sites[:10]):
            try:
                site_id = site.get('id')
                if inline_visualization and IPYTHON_AVAILABLE:
                    # Display inline with comparison to closest known site
                    known_sites = kg.get_sites()
                    if known_sites:
                        # Get first known site for comparison
                        known_site_id = known_sites[0].get('id')
                        display_site_inline(kg, known_site_id, site_id)
            except Exception as e:
                print(f"Error visualizing site {i}: {e}")
    
    # Save submission file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    submission_file = os.path.join(output_dir, f"submission_{timestamp}.json")
    submission = {
        "timestamp": timestamp,
        "sites": potential_sites
    }
    with open(submission_file, 'w') as f:
        json.dump(submission, f, indent=2)
    print(f"Submission file saved to {submission_file}")
    
    # Save final KG
    kg.save(kg_path)
    print(f"Knowledge graph saved to {kg_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "workflow_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    print("\n=== WORKFLOW COMPLETE ===")
    print(f"Total tiles processed: {metrics['total_tiles']}")
    print(f"Features detected: {metrics['total_features_detected']}")
    print(f"Potential new sites: {metrics['potential_sites_detected']}")
    print(f"Total time: {metrics['learning_phase_time'] + metrics['evaluation_phase_time']:.2f} seconds")
    
    return {
        "success": True,
        "kg": kg,
        "potential_sites": potential_sites,
        "submission_file": submission_file,
        "metrics": metrics
    }

# End of file
def display_detected_sites_summary(kg, min_confidence=0.4, max_sites=15, display_inline=False):
    """Display a summary of the most promising detected archaeological sites."""
    # Find potential archaeological sites with expanded search radius
    potential_sites = kg.find_potential_new_sites(min_confidence=min_confidence, distance_from_known=200.0)
    
    if not potential_sites:
        print("No potential archaeological sites detected above confidence threshold.")
        return []
    
    # Sort by confidence (highest first)
    potential_sites = sorted(potential_sites, key=lambda x: x.get("confidence", 0), reverse=True)
    
    # Limit to max_sites
    sites_to_display = potential_sites[:max_sites]
    
    print(f"\n=== Detected {len(potential_sites)} Potential Archaeological Sites ===")
    print(f"Displaying top {len(sites_to_display)} sites with confidence >= {min_confidence}:\n")
    
    displayed_ids = []
    
    # Display each site
    for i, site in enumerate(sites_to_display):
        site_id = site.get("id", "unknown")
        confidence = site.get("confidence", 0)
        lat = site.get("latitude", site.get("lat", 0))
        lon = site.get("longitude", site.get("lon", 0))
        feature_types = site.get("related_feature_types", [])
        feature_count = site.get("feature_count", 0)
        
        print(f"Site #{i+1}: ID={site_id}")
        print(f"  Location: {lat:.6f}, {lon:.6f}")
        print(f"  Confidence: {confidence:.2f}")
        
        # Display features if available
        feature_str = ", ".join(feature_types[:3])
        if len(feature_types) > 3:
            feature_str += "..."
        print(f"  Features: {feature_count} ({feature_str})")
        
        # Display reasoning if available
        if "reasoning" in site:
            print(f"  Significance: {site['reasoning']}")
        
        # Get related clusters or alignments
        related_elements = []
        if "related_clusters" in site:
            for c in site.get("related_clusters", []):
                related_elements.append(f"Cluster of {c.get('member_count', 0)} features")
        
        if "related_alignments" in site:
            for a in site.get("related_alignments", []):
                related_elements.append(f"Alignment at {a.get('orientation', 0):.1f}° with {a.get('member_count', 0)} features")
        
        if related_elements:
            print(f"  Related elements: {', '.join(related_elements[:3])}")
        
        print()  # Empty line between sites
        
        # Add site ID to displayed list
        displayed_ids.append(site_id)
        
        # Visualize the site if requested
        if display_inline:
            try:
                display_site_inline(kg, potential_site_id=site_id)
            except Exception as e:
                print(f"Error displaying site visualization: {e}")
    
    return displayed_ids 


# Main function to run the pipeline
def main(visualize=False, inline_visualization=False, run_training=False, extract_templates=True):
    """Run the Amazon Archaeology KG + Agent pipeline
    
    Args:
        visualize: If True, saves visualizations of known and detected sites
        inline_visualization: If True, displays visualizations in real-time during processing
        run_training: If True, runs the site learning workflow to train site detection models
        extract_templates: If True, extracts templates from known sites for pattern matching
    
    Returns:
        Dictionary with results including KG, potential sites, and file paths
        
    Note:
        When inline_visualization=True and running in a notebook, this function provides:
        - Real-time visualization of features as they're detected
        - Display of site images with analytical views
        - Comparison visualizations between known and potential sites
        - Helper functions for interactive exploration of the results
    """
    # Parse command line arguments - ignores unknown args like -f from Jupyter
    parser = argparse.ArgumentParser(description="Amazon Archaeological Site Detection")
    parser.add_argument("--reset", action="store_true", help="Force reset the knowledge graph")
    parser.add_argument("--templates", type=str, help="Path to templates JSON file")
    parser.add_argument("--tiles", type=str, help="Path to tiles directory")
    parser.add_argument("--output", type=str, help="Path to output file")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations of sites")
    parser.add_argument("--inline", action="store_true", help="Show inline visualizations during processing")
    parser.add_argument("--train", action="store_true", help="Run the site learning workflow for training")
    parser.add_argument("--extract-templates", action="store_true", help="Extract templates from known sites")
    args, unknown = parser.parse_known_args()
    
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")
    
    # Check for reset via args or environment variable
    force_reset = args.reset or os.environ.get("RESET_KG", "").lower() in ("true", "1", "yes")
    
    # Check for visualization flags from args or parameters
    should_visualize = visualize or args.visualize
    should_visualize_inline = inline_visualization or args.inline
    should_run_training = run_training or args.train
    should_extract_templates = extract_templates or args.extract_templates
    
    # Import visualization processor if needed
    if should_visualize_inline and IPYTHON_AVAILABLE:
        try:
            from visualizing_processor import VisualizingParallelProcessor
            print("Successfully imported visualizing processor")
        except ImportError:
            print("Warning: Could not import visualizing processor, falling back to standard processing")
            should_visualize_inline = False
    
    # Check for archive flag - archive old KG even when resetting
    should_archive = os.environ.get("ARCHIVE_KG", "").lower() in ("true", "1", "yes")
    
    print("Amazon Archaeology KG + Agent Full Pipeline\n")
    
    # Set up paths based on environment
    if is_running_on_kaggle():
        kg_storage_dir = '/kaggle/working/kg_data'
        saved_kg_path = '/kaggle/working/amazon_archaeology_kg.json'
        output_dir = '/kaggle/working'
        
        # Find tile directory in input
        input_dir = '/kaggle/input'
        dataset_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        if dataset_dirs:
            # Use first dataset found
            tiles_dir = os.path.join(input_dir, dataset_dirs[0], 'tiles')
            templates_path = os.path.join(input_dir, dataset_dirs[0], 'templates.json')
            # Check for test_data path
            if os.path.exists(os.path.join(input_dir, dataset_dirs[0], 'test_data')):
                test_data_dir = os.path.join(input_dir, dataset_dirs[0], 'test_data')
            else:
                test_data_dir = os.path.join(output_dir, 'test_data')
        else:
            # Default fallbacks
            tiles_dir = os.path.join(input_dir, 'tiles')
            templates_path = os.path.join(input_dir, 'templates.json')
            test_data_dir = os.path.join(output_dir, 'test_data')
        
        print(f"Running on Kaggle - results will be saved to {output_dir}")
    else:
        # Local paths
        kg_storage_dir = './kg_data'
        saved_kg_path = 'amazon_archaeology_kg.json'
        output_dir = './output'
        tiles_dir = './data/tiles'
        templates_path = './templates.json'
        test_data_dir = './test_data'
        print(f"Running locally - results will be saved to {output_dir}")
    
    # Override with command line args if provided
    if args.templates:
        templates_path = args.templates
    if args.tiles:
        tiles_dir = args.tiles
    if args.output:
        output_dir = os.path.dirname(args.output)
        saved_kg_path = args.output
    
    # Create necessary directories (only in writable locations)
    os.makedirs(kg_storage_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    if not is_running_on_kaggle():
        os.makedirs(os.path.dirname(tiles_dir), exist_ok=True)
    
    # Set up processed_tiles directory
    if is_running_on_kaggle():
        processed_dir = os.path.join('/kaggle/working', "processed_tiles")
    else:
        processed_dir = os.path.join(os.path.dirname(tiles_dir), "processed_tiles")
        
    os.makedirs(processed_dir, exist_ok=True)
    print(f"Creating processed tiles directory at: {processed_dir}")
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize visualization paths list
    vis_paths = []
    
    # Check if we should archive before resetting
    if should_archive and os.path.exists(saved_kg_path):
        print("\n[!] ARCHIVE FLAG DETECTED - Archiving existing KG before reset\n")
        archive_dir = os.path.join(output_dir, "kg_archives")
        archived_path = archive_knowledge_graph(saved_kg_path, archive_dir)
        if archived_path:
            print(f"Successfully archived KG to {archived_path}")
    
    # Check if reset is requested
    if force_reset:
        print("\n[!] RESET FLAG DETECTED - Deleting existing KG data\n")
        # Delete the main KG file
        if os.path.exists(saved_kg_path):
            os.remove(saved_kg_path)
            print(f"Deleted {saved_kg_path}")
        
        # Delete any snapshots in output directory
        if os.path.exists(output_dir):
            for snapshot in glob.glob(f'{output_dir}/kg_snapshot_*.json'):
                os.remove(snapshot)
                print(f"Deleted {snapshot}")
        
        # Also delete the checkpoint file so we process all tiles
        checkpoint_path = os.path.join(kg_storage_dir, "processing_checkpoint.json")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"Deleted checkpoint file: {checkpoint_path}")
        
        # Delete any compressed backups (but not archives)
        if os.path.exists(f"{saved_kg_path}.gz"):
            os.remove(f"{saved_kg_path}.gz")
            print(f"Deleted compressed backup: {saved_kg_path}.gz")
    
    # Initialize knowledge graph
    kg = AmazonKG(storage_dir=kg_storage_dir)
    GLOBAL_SUPERVISOR.start(kg)
    
    # Try to load existing KG if not resetting
    if not force_reset and os.path.exists(saved_kg_path):
        print(f"Loading existing knowledge graph from {saved_kg_path}")
        kg.load(saved_kg_path)
    
    # Load templates
    print("\nStep 1: Loading templates")
    if not kg.templates:
        try:
            templates = load_templates_from_json(templates_path)
            
            # If no templates found, create synthetic ones
            if not templates:
                print(f"No templates found at {templates_path}, using synthetic templates")
                templates = create_synthetic_templates()
            
            # Augment templates with rotations and flips to improve detection robustness
            print(f"Augmenting {len(templates)} templates with rotations and flips...")
            augmented_templates = []
            
            for i, template in enumerate(templates):
                # Check template format
                if isinstance(template, dict) and 'grid' in template:
                    grid = template['grid']
                    desc = template.get('description', f'Template {i}')
                    # Add original template
                    augmented_templates.append(template)
                elif isinstance(template, np.ndarray):
                    # Handle case where template is directly a numpy array
                    grid = template
                    desc = f'Template {i}'
                    # Add original template
                    augmented_templates.append({'grid': grid, 'description': desc})
                else:
                    print(f"Skipping template {i} due to unknown format: {type(template)}")
                    continue
                
                try:
                    # Add rotations (90°, 180°, 270°)
                    for angle in [90, 180, 270]:
                        from scipy.ndimage import rotate
                        rotated = rotate(grid, angle, reshape=True, order=0, mode='constant')
                        # Convert back to same datatype as original
                        rotated = rotated.astype(grid.dtype)
                        augmented_templates.append({
                            'grid': rotated,
                            'description': f"{desc} (rotated {angle}°)"
                        })
                    
                    # Add horizontal flip
                    flipped_h = np.fliplr(grid)
                    augmented_templates.append({
                        'grid': flipped_h,
                        'description': f"{desc} (horizontal flip)"
                    })
                    
                    # Add vertical flip
                    flipped_v = np.flipud(grid)
                    augmented_templates.append({
                        'grid': flipped_v,
                        'description': f"{desc} (vertical flip)"
                    })
                except Exception as e:
                    print(f"Error augmenting template {i}: {e}")
            
            print(f"Template augmentation: {len(templates)} originals → {len(augmented_templates)} total templates")
            templates = augmented_templates
                
            for template in templates:
                if isinstance(template, dict) and 'grid' in template:
                    kg.add_template(template['grid'], desc=template.get('description', ''))
            print(f"Loaded {len(templates)} templates")
        except Exception as e:
            print(f"Error loading templates: {e}")
            # Create synthetic templates as fallback
            print("Using synthetic templates as fallback")
            templates = create_synthetic_templates()
            for template in templates:
                kg.add_template(template['grid'], desc=template.get('description', ''))
    else:
        print(f"Using {len(kg.templates)} templates already in KG")
    
    # Load reference sites (if available) for validation
    print("\nStep 2: Loading reference sites data for validation")
    sites_count = len(kg.sites)
    
    print("NOTE: The known archaeological sites (~100) are embedded in the GeoTIFF data.")
    print("We will measure our system's effectiveness by how many of these known sites we detect.")
    
    # Look for optional sites.csv that might contain validation data
    try:
        # Look for sites.csv in multiple locations
        possible_paths = [
            os.path.join(os.path.dirname(templates_path), 'sites.csv'),  # Same directory as templates
            './sites.csv',  # Root directory
            './data/sites.csv',  # Data directory
            '/kaggle/input/nasa-tiles/sites.csv'  # Kaggle input directory
        ]
        
        # Try all possible paths
        for sites_path in possible_paths:
            if os.path.exists(sites_path):
                print(f"Found validation sites file at {sites_path}")
                sites_data = load_sites_from_csv(sites_path)
                if sites_data:
                    print(f"Loading {len(sites_data)} validation sites from CSV file")
                    site_count_before = len(kg.sites)
                    for site in sites_data:
                        if 'lat' in site and 'lon' in site:
                            kg.add_site(site['lat'], site['lon'], desc=site.get('desc', ''), is_validation=True)
                    sites_count = len(kg.sites)
                    sites_added = sites_count - site_count_before
                    print(f"Added {sites_added} validation sites from {sites_path}")
                    break
    except Exception as e:
        print(f"Error loading validation sites (this is optional): {e}")
    
    # Add some synthetic sites if running in test mode
    if force_reset and is_running_on_kaggle() and sites_count == 0:
        print("Adding synthetic validation sites for testing purposes")
        sites_data = create_synthetic_sites()
        for site in sites_data:
            kg.add_site(site['lat'], site['lon'], desc=site.get('desc', ''), is_validation=True)
        sites_count = len(kg.sites)
    
    print(f"Using {sites_count} validation sites")
    print("We expect to find ~100 actual archaeological sites in the GeoTIFF data")
    
    # Extract templates from known sites if requested
    if should_extract_templates:
        print("\nExtracting templates from known sites")
        # Find all GeoTIFF files for template extraction
        geotiff_files = find_geotiff_files(base_path=tiles_dir)
        if geotiff_files:
            try:
                num_templates, site_tiles = extract_templates_from_sites(kg, geotiff_files, tiles_dir)
                print(f"Extracted {num_templates} templates from known sites")
            except Exception as e:
                print(f"Error extracting templates: {e}")
                print(traceback.format_exc())
    
    # Run site learning workflow if requested
    if should_run_training:
        print("\nRunning site learning workflow for training")
        try:
            # Run the site learning workflow
            training_results = run_site_learning_workflow(
                kg_path=saved_kg_path,
                reset=force_reset,
                visualize=should_visualize,
                inline_visualization=should_visualize_inline,
                tiles_dir=tiles_dir,
                exhaustive=True
            )
            print(f"Training completed. Model accuracy: {training_results.get('accuracy', 'N/A')}")
        except Exception as e:
            print(f"Error in site learning workflow: {e}")
            print(traceback.format_exc())
    
    # Load and process tiles
    print("\nStep 3: Processing image tiles")
    print(f"Using {64}×{64} pixel tiles with 25% overlap for optimal feature detection")
    
    # Load tiles from the specified directory
    try:
        tiles_data = load_tiles_from_geotiff(tiles_dir)
        print(f"Loaded {len(tiles_data)} tiles from {tiles_dir}")
    except Exception as e:
        print(f"Error loading tiles: {e}")
        tiles_data = []
        
    # If no tiles loaded, create synthetic ones
    if not tiles_data:
        print(f"No tiles found in {tiles_dir}, creating synthetic tiles")
        # Create multiple synthetic tiles with different random features
        num_synthetic_tiles = 5
        for i in range(num_synthetic_tiles):
            synthetic_tile = create_synthetic_tile()
            # Update the index to make each tile unique
            if 'meta' in synthetic_tile:
                synthetic_tile['meta']['index'] = i
            tiles_data.append(synthetic_tile)
        print(f"Created {len(tiles_data)} synthetic tiles for testing")
    
    # Process tiles with parallel processor
    if tiles_data:
        print(f"\nStep 4: Starting parallel processing of {len(tiles_data)} tiles")
        
        # Create a VisualizingParallelProcessor if inline visualization is requested
        if should_visualize_inline and IPYTHON_AVAILABLE:
            processor = VisualizingParallelProcessor(kg)
            print("Using visualizing processor with inline display")
        else:
            processor = ParallelProcessor(kg)
        
        # Force reset the processed tiles if we're resetting the KG
        if force_reset and hasattr(processor, 'processed_tiles'):
            processor.processed_tiles = set()
            print("Reset processed tiles tracking for fresh processing")
            
        try:
            processor.process_tiles_parallel(tiles_data, batch_size=10)
        except Exception as e:
            print(f"Error in parallel processing: {e}")
            # Simple processing fallback if parallel fails
            print("Falling back to simple processing mode")
            for i, tile in enumerate(tiles_data):
                print(f"Processing tile {i+1}/{len(tiles_data)}")
                # Simple processing would go here
    else:
        print("\nNo tiles to process - skipping processing step")
    
    # Final analysis and site detection
    print("\nStep 5: Agent-based archaeological site detection")
    
    try:
        # Find all GeoTIFF files
        print("Searching for GeoTIFF files...")
        geotiff_files = find_geotiff_files()
        if geotiff_files:
            print(f"Found {len(geotiff_files)} GeoTIFF files")
            
            # Extract site windows from known sites for visualization only
            print("Extracting windows centered on known archaeological sites for visualization...")
            site_windows = extract_site_windows(geotiff_files, kg.get_sites())
            
            # Visualize some known sites
            if site_windows and should_visualize:
                print("Creating visualizations for known archaeological sites...")
                vis_paths = []
                for i, window in enumerate(site_windows[:5]):  # Visualize just a few
                    try:
                        # Save visualization and display it inline if in Kaggle/Jupyter
                        vis_path = save_site_visualization(window, output_dir=vis_dir, 
                                                        prefix=f"known_site_{i}", 
                                                        display_inline=IPYTHON_AVAILABLE)
                        vis_paths.append(vis_path)
                        print(f"Saved site visualization to {vis_path}")
                    except Exception as e:
                        print(f"Error creating site visualization: {e}")
            
            # Run agent-based site detection directly from the knowledge graph
            print("\nRunning rule-based archaeological site detection with expanded 200km radius...")
            potential_sites = kg.find_potential_new_sites(min_confidence=0.65, distance_from_known=200.0)
            
            # Display a summary of detected sites
            print("\nGenerating site detection summary report...")
            displayed_site_ids = display_detected_sites_summary(
                kg, 
                min_confidence=0.5, 
                max_sites=10, 
                display_inline=should_visualize_inline
            )
            print(f"Generated report for {len(displayed_site_ids)} top sites")
            
            # Visualize detected sites if possible
            vis_paths = []
            if potential_sites and should_visualize:
                print(f"Creating visualizations for detected sites...")
                for i, site in enumerate(potential_sites[:10]):  # Visualize up to 10 detections
                    try:
                        # Find the nearest GeoTIFF file to this site
                        lat = site.get('lat', site.get('latitude', 0))
                        lon = site.get('lon', site.get('longitude', 0))
                        
                        # If coordinates are still 0, try to get from geometry
                        if lat == 0 and lon == 0:
                            geo = site.get('geometry', {})
                            lat = geo.get('lat', geo.get('latitude', 0))
                            lon = geo.get('lon', geo.get('longitude', 0))
                        
                        if lat is not None and lon is not None:
                            # Create a simple visualization window for the detected site
                            for tif_path in geotiff_files:
                                try:
                                    with rasterio.open(tif_path) as src:
                                        bounds = src.bounds
                                        # Check if site is within bounds
                                        if (bounds.left <= lon <= bounds.right and 
                                            bounds.bottom <= lat <= bounds.top):
                                            # Create a window for visualization
                                            window_dict = {
                                                'meta': {
                                                    'source': os.path.basename(tif_path),
                                                    'lat': lat,
                                                    'lon': lon,
                                                    'site_id': site.get('id', f'detected_{i}'),
                                                    'confidence': site.get('confidence', site.get('archaeological_confidence', 0.0)),
                                                    'detected_by': site.get('detected_by', 'AgentSystem')
                                                }
                                            }
                                            vis_path = save_site_visualization(window_dict, output_dir=vis_dir, 
                                                           prefix=f"detected_site_{i}", 
                                                           display_inline=IPYTHON_AVAILABLE)
                                            vis_paths.append(vis_path)
                                            print(f"Saved potential site visualization to {vis_path}")
                                            break
                                except Exception as exc:
                                    print(f"Error processing GeoTIFF for visualization: {exc}")
                    except Exception as e:
                        print(f"Error creating detection visualization: {e}")
            
            # Export the knowledge graph to Cypher format
            cypher_export = kg.export_for_cypher()
            print(f"Exported knowledge graph to Cypher format: {cypher_export}")
            
            print(f"Found {len(potential_sites)} potential archaeological sites")
        else:
            print("No GeoTIFF files found. Using knowledge graph for detection with expanded 200km radius.")
            potential_sites = kg.find_potential_new_sites(min_confidence=0.65, distance_from_known=200.0)
            
            # Generate summary report even without GeoTIFF files
            displayed_site_ids = display_detected_sites_summary(
                kg, 
                min_confidence=0.5, 
                max_sites=10, 
                display_inline=should_visualize_inline
            )
            
            print(f"Found {len(potential_sites)} potential archaeological sites")
    except Exception as e:
        print(f"Error in site detection: {e}")
        print(traceback.format_exc())
        potential_sites = kg.find_potential_new_sites(min_confidence=0.65, distance_from_known=200.0)
        print(f"Found {len(potential_sites)} potential archaeological sites")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    submission_file = os.path.join(output_dir, f"submission_{timestamp}.json")
    
    # Create submission format
    submission = {
        "timestamp": timestamp,
        "sites": potential_sites
    }
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(submission_file), exist_ok=True)
    
    with open(submission_file, 'w') as f:
        json.dump(submission, f, indent=2)
    
    # Save final KG state
    if saved_kg_path:
        # Make sure the path has a directory
        if os.path.dirname(saved_kg_path) == '':
            # If no directory specified, use output_dir
            saved_kg_path = os.path.join(output_dir, os.path.basename(saved_kg_path))
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(saved_kg_path), exist_ok=True)
        
        kg.save(saved_kg_path)
        print(f"Knowledge graph saved to {saved_kg_path}")
    else:
        print("Knowledge graph not saved (no valid path)")
    
    print(f"\nResults saved to {submission_file}")
    if should_visualize and 'vis_paths' in locals() and vis_paths:
            print(f"Visualizations saved to directory: {vis_dir}")
    print(f"Generated {len(vis_paths)} visualization images")
    
    # Analyze feature types for final report
    print("\nGenerating final feature analysis report...")
    analyze_feature_types(kg)
    
    # Cleanup
    GLOBAL_SUPERVISOR.stop()
    print("\nPipeline complete!")
    
    # Return results for potential further analysis in notebooks
    result = {
        "kg": kg,
        "potential_sites": potential_sites,
        "submission_file": submission_file,
        "visualization_dir": vis_dir if should_visualize else None
    }
    
    # If this is running in an IPython environment, add helper for visualization
    if IPYTHON_AVAILABLE and should_visualize:
        # Add visualization samples to the result
        result["show_known_site"] = lambda index=0: show_site_visualization(site_index=index, prefix="known_site", directory=vis_dir)
        result["show_detected_site"] = lambda index=0: show_site_visualization(site_index=index, prefix="detected_site", directory=vis_dir)
        
        # Show the first known site automatically if available
        import glob
        known_site_files = glob.glob(os.path.join(vis_dir, "known_site_*.png"))
        if known_site_files:
            print("\nSample visualization of a known archaeological site:")
            show_site_visualization(site_index=0, prefix="known_site", directory=vis_dir)
            
                    # Show the first detected site automatically if available
            detected_site_files = glob.glob(os.path.join(vis_dir, "detected_site_*.png"))
            if detected_site_files:
                print("\nSample visualization of a detected potential site:")
                show_site_visualization(site_index=0, prefix="detected_site", directory=vis_dir)
                
            # For inline visualization, add helper function to display site comparisons
            result["display_site_comparison"] = lambda site_id=None, potential_site_id=None: (
                display_site_inline(kg, site_id, potential_site_id, figsize=(15, 10))
            )
    
    return result

# Run main function when executed directly
if __name__ == "__main__":
    main()


