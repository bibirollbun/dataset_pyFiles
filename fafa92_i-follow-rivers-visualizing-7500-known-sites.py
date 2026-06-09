pip install pandas numpy folium pyproj openpyxl ipython



"""
Archaeological Sites Visualization Script
==========================================

This script reads archaeological site data from multiple files and creates 
interactive maps showing the locations of various archaeological sites in 
the Amazon region.

Requirements:
- pandas
- numpy
- folium
- pyproj (for coordinate transformations)
- openpyxl (for Excel files)

Install with: pip install pandas numpy folium pyproj openpyxl
"""

import pandas as pd
import numpy as np
import folium
from folium import plugins
import re
from pyproj import Transformer
import openpyxl
from IPython.display import display

# Configuration
MAP_CENTER = [-10.0, -67.0]  # Approximate center of Amazon region
MAP_ZOOM = 3

def utm_to_latlon(utm_x, utm_y, utm_zone=19, hemisphere='south'):
    """
    Convert UTM coordinates to latitude/longitude.
    
    Parameters:
    utm_x, utm_y: UTM coordinates
    utm_zone: UTM zone (default 19 for western Brazil)
    hemisphere: 'north' or 'south'
    """
    from pyproj import Transformer
    
    # Define the coordinate systems
    utm_crs = f"EPSG:326{utm_zone}" if hemisphere == 'north' else f"EPSG:327{utm_zone}"
    wgs84_crs = "EPSG:4326"
    
    # Create transformer
    transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
    
    # Transform coordinates (returns lon, lat)
    lon, lat = transformer.transform(utm_x, utm_y)
    return lat, lon

def read_mound_villages_data(filepath):
    """
    Read the mound villages data from CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Convert UTM to lat/lon (using UTM zone 19S as indicated in original data)
        lats, lons = [], []
        for _, row in df.iterrows():
            if pd.notna(row['UTM X (Easting)']) and pd.notna(row['UTM Y (Northing)']):
                lat, lon = utm_to_latlon(row['UTM X (Easting)'], row['UTM Y (Northing)'], utm_zone=19)
                lats.append(lat)
                lons.append(lon)
            else:
                lats.append(np.nan)
                lons.append(np.nan)
        
        df['latitude'] = lats
        df['longitude'] = lons
        df['source'] = 'Mound Villages'
        
        return df
        
    except Exception as e:
        print(f"Error reading mound villages data: {e}")
        return pd.DataFrame()

def read_casarabe_sites_data(filepath):
    """
    Read the Casarabe sites data from CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Convert UTM to lat/lon
        lats, lons = [], []
        for _, row in df.iterrows():
            if pd.notna(row['UTM X (Easting)']) and pd.notna(row['UTM Y (Northing)']):
                # Assuming UTM zone 20S for Casarabe sites (Bolivia region)
                lat, lon = utm_to_latlon(row['UTM X (Easting)'], row['UTM Y (Northing)'], utm_zone=20)
                lats.append(lat)
                lons.append(lon)
            else:
                lats.append(np.nan)
                lons.append(np.nan)
        
        df['latitude'] = lats
        df['longitude'] = lons
        df['source'] = 'Casarabe Sites'
        
        return df
        
    except Exception as e:
        print(f"Error reading Casarabe sites data: {e}")
        return pd.DataFrame()

def read_geoglyphs_data(filepath):
    """
    Read the Amazon geoglyphs data from CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Convert latitude to numeric (it might be a string)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        df['source'] = 'Amazon Geoglyphs'
        
        return df
        
    except Exception as e:
        print(f"Error reading geoglyphs data: {e}")
        return pd.DataFrame()

def read_submit_data(filepath):
    """
    Read the submit.csv data.
    """
    try:
        df = pd.read_csv(filepath)
        
        # Check if coordinates are already in lat/lon format
        sample_x = df['x'].iloc[0] if len(df) > 0 else None
        sample_y = df['y'].iloc[0] if len(df) > 0 else None
        
        if sample_x and -180 <= sample_x <= 180 and -90 <= sample_y <= 90:
            # Coordinates are likely already in lat/lon
            df['latitude'] = df['y']
            df['longitude'] = df['x']
        else:
            # Coordinates might be in UTM, convert them
            # We'll try different UTM zones based on the magnitude
            df['latitude'] = np.nan
            df['longitude'] = np.nan
            
            # Try to determine appropriate UTM zone based on coordinate ranges
            for idx, row in df.iterrows():
                try:
                    # Try different UTM zones (18-21 are common for Amazon region)
                    for zone in [18, 19, 20, 21]:
                        try:
                            lat, lon = utm_to_latlon(row['x'], row['y'], utm_zone=zone)
                            if -90 <= lat <= 90 and -180 <= lon <= 180:
                                df.at[idx, 'latitude'] = lat
                                df.at[idx, 'longitude'] = lon
                                break
                        except:
                            continue
                except:
                    continue
        
        df['source'] = 'Archaeological Survey Data'
        
        return df
        
    except Exception as e:
        print(f"Error reading submit data: {e}")
        return pd.DataFrame()

def read_science_data(filepath):
    """
    Read the science data from CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        
        # The data already has Latitude and Longitude columns
        df['latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        
        df['source'] = 'Science Data'
        
        return df
        
    except Exception as e:
        print(f"Error reading science data: {e}")
        return pd.DataFrame()

def create_map(dataframes_list):
    """
    Create an interactive map with all archaeological sites.
    """
    # Create base map
    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=MAP_ZOOM,
        tiles='OpenStreetMap'
    )
    
    # Add different tile layers with proper attributions
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name='Stamen Terrain',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}{r}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name='Stamen Toner',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='cartodbpositron',
        name='CartoDB Positron'
    ).add_to(m)
    
    # Color mapping for different sources
    colors = {
        'Mound Villages': 'red',
        'Casarabe Sites': 'blue', 
        'Amazon Geoglyphs': 'orange',
        'Archaeological Survey Data': 'green',
    }
    
    # Create feature groups for different data sources
    feature_groups = {}
    for source in colors.keys():
        feature_groups[source] = folium.FeatureGroup(name=source)
    
    # Add points for each dataset
    total_points = 0
    for df_name, df in dataframes_list:
        if df.empty:
            continue
            
        source = df['source'].iloc[0] if 'source' in df.columns else df_name
        color = colors.get(source, 'black')
        
        # Filter valid coordinates
        valid_coords = df.dropna(subset=['latitude', 'longitude'])
        
        for idx, row in valid_coords.iterrows():
            # Create popup text
            popup_text = f"<b>Source:</b> {source}<br>"
            
            # Site name handling for different datasets
            if 'Site Name' in row and pd.notna(row['Site Name']):
                popup_text += f"<b>Site:</b> {row['Site Name']}<br>"
            elif 'Site' in row and pd.notna(row['Site']):
                popup_text += f"<b>Site:</b> {row['Site']}<br>"
            elif 'name' in row and pd.notna(row['name']):
                popup_text += f"<b>Site:</b> {row['name']}<br>"
            
            # Classification and type information
            if 'Classification' in row and pd.notna(row['Classification']):
                popup_text += f"<b>Type:</b> {row['Classification']}<br>"
            if 'PlotType' in row and pd.notna(row['PlotType']):
                popup_text += f"<b>Plot Type:</b> {row['PlotType']}<br>"
            if 'type' in row and pd.notna(row['type']):
                popup_text += f"<b>Type:</b> {row['type']}<br>"
            
            # Location information
            if 'Country' in row and pd.notna(row['Country']):
                popup_text += f"<b>Country:</b> {row['Country']}<br>"
            if 'Subdivision' in row and pd.notna(row['Subdivision']):
                popup_text += f"<b>Region:</b> {row['Subdivision']}<br>"
            
            # Numerical data
            if 'Number of mounds' in row and pd.notna(row['Number of mounds']):
                popup_text += f"<b>Mounds:</b> {int(row['Number of mounds'])}<br>"
            if 'Diameter (m)' in row and pd.notna(row['Diameter (m)']):
                popup_text += f"<b>Diameter:</b> {row['Diameter (m)']} m<br>"
            if 'Elevation (m)' in row and pd.notna(row['Elevation (m)']):
                popup_text += f"<b>Elevation:</b> {row['Elevation (m)']} m<br>"
            if 'Altitude' in row and pd.notna(row['Altitude']):
                popup_text += f"<b>Altitude:</b> {row['Altitude']} m<br>"
            if 'PlotSize' in row and pd.notna(row['PlotSize']):
                popup_text += f"<b>Plot Size:</b> {row['PlotSize']}<br>"
            
            # Additional features
            if 'LIDAR' in row and pd.notna(row['LIDAR']):
                popup_text += f"<b>LIDAR Coverage:</b> {row['LIDAR']}<br>"
            if 'Associated features' in row and pd.notna(row['Associated features']):
                popup_text += f"<b>Features:</b> {row['Associated features']}<br>"
            
            popup_text += f"<b>Coordinates:</b> {row['latitude']:.6f}, {row['longitude']:.6f}"
            
            # Add marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(feature_groups[source])
            
            total_points += 1
    
    # Add feature groups to map
    for fg in feature_groups.values():
        fg.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add minimap
    minimap = plugins.MiniMap()
    m.add_child(minimap)
    
    # Add measure control
    plugins.MeasureControl(primary_length_unit='kilometers').add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    print(f"Created map with {total_points} archaeological sites")
    return m

def main():
    """
    Main function to read all data and create the map.
    """
    print("Reading archaeological site data...")
    
    # Read all datasets
    dataframes = []
    
    # 1. Read mound villages data
    print("Reading mound villages data...")
    try:
        mound_df = read_mound_villages_data('/kaggle/input/mound-villages-acre/mound_villages_acre.csv')
        if not mound_df.empty:
            dataframes.append(('Mound Villages', mound_df))
            print(f"  - Loaded {len(mound_df)} mound village sites")
    except:
        print("  - Mound villages file not found or cannot be read")
    
    # 2. Read Casarabe sites
    print("Reading Casarabe sites...")
    try:
        casarabe_df = read_casarabe_sites_data('/kaggle/input/casarabe-sites-utm/casarabe_sites_utm.csv')
        if not casarabe_df.empty:
            dataframes.append(('Casarabe Sites', casarabe_df))
            print(f"  - Loaded {len(casarabe_df)} Casarabe sites")
    except:
        print("  - Casarabe sites file not found or cannot be read")
    
    # 3. Read Amazon geoglyphs
    print("Reading Amazon geoglyphs...")
    try:
        geoglyphs_df = read_geoglyphs_data('/kaggle/input/amazon-geoglyphs-sites/amazon_geoglyphs_sites.csv')
        if not geoglyphs_df.empty:
            # Sample the data if it's too large (for performance)
            original_count = len(geoglyphs_df)
            if len(geoglyphs_df) > 6000:
                geoglyphs_df = geoglyphs_df.sample(n=6000, random_state=42)
                print(f"  - Sampled 2000 geoglyphs from {original_count} total geoglyphs")
            dataframes.append(('Amazon Geoglyphs', geoglyphs_df))
            print(f"  - Loaded {len(geoglyphs_df)} Amazon geoglyphs")
    except:
        print("  - Amazon geoglyphs file not found or cannot be read")
    
    # 4. Read submit data
    print("Reading submit data...")
    try:
        submit_df = read_submit_data('/kaggle/input/archaeological-survey-data/submit.csv')
        if not submit_df.empty:
            # Sample the data if it's too large (for performance)
            original_count = len(submit_df)
            if len(submit_df) > 6000:
                submit_df = submit_df.sample(n=6000, random_state=42)
                print(f"  - Sampled 1000 points from {original_count} total points")
            dataframes.append(('Archaeological Survey Data', submit_df))
            print(f"  - Loaded {len(submit_df)} archaeological survey data points")
    except:
        print("  - Submit data file not found or cannot be read")
    

    # Create the map
    print("Creating interactive map...")
    map_obj = create_map(dataframes)
    
    # Save the map
    map_obj.save('archaeological_sites_map.html')
    print("Map saved as 'archaeological_sites_map.html'")
    print("Open this file in a web browser to view the interactive map")
    
    # Display basic statistics
    print("\n=== Summary ===")
    total_sites = sum(len(df) for _, df in dataframes)
    print(f"Total archaeological sites plotted: {total_sites}")
    
    for name, df in dataframes:
        valid_coords = df.dropna(subset=['latitude', 'longitude'])
        print(f"  - {name}: {len(valid_coords)} sites with valid coordinates")
    
    return map_obj

# Example usage
if __name__ == "__main__":
    # Run the main visualization
    map_object = main()
    
    # Display the map inline in Kaggle notebook
    display(map_object)
    
    print("\nVisualization complete! The map is displayed above and also saved as 'archaeological_sites_map.html'")


import pandas as pd
import numpy as np
import folium
from folium import plugins
import requests
import json
import time
import os
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
import geopandas as gpd
from pyproj import Transformer
import pickle
from datetime import datetime

# --- CONFIGURATION ---
BUFFER_DISTANCE_KM = 10.0  # Buffer distance around rivers in kilometers
MAP_CENTER = [-10.0, -67.0]  # Initial map center
MAP_ZOOM = 6

# Tiling configuration for large areas
TILE_SIZE_DEGREES = 1  # Size of each tile in degrees (roughly 220km)
TILE_OVERLAP_DEGREES = 0.1  # Overlap between tiles to catch boundary rivers
QUERY_TIMEOUT = 60  # Timeout for each tile query
DELAY_BETWEEN_QUERIES = 60.0  # Delay between queries to be respectful to OSM

# Cache configuration
CACHE_DIR = "/kaggle/input/river-segments/river_cache"
USE_CACHE = True  # Set to False to force fresh queries

# --- HELPER FUNCTIONS ---

def utm_to_latlon(utm_x, utm_y, utm_zone=19, hemisphere='south'):
    """Convert UTM coordinates to latitude/longitude."""
    if hemisphere.lower() not in ['north', 'south']:
        hemisphere = 'south'
    base_epsg = 32600 if hemisphere.lower() == 'north' else 32700
    try:
        utm_zone_int = int(utm_zone)
        if not (1 <= utm_zone_int <= 60):
            return np.nan, np.nan
        utm_crs_epsg = base_epsg + utm_zone_int
    except ValueError:
        return np.nan, np.nan

    utm_crs = f"EPSG:{utm_crs_epsg}"
    wgs84_crs = "EPSG:4326"
    try:
        transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
        lon, lat = transformer.transform(utm_x, utm_y)
        return lat, lon
    except Exception:
        return np.nan, np.nan

# --- DATA READING FUNCTIONS (same as your original script) ---

def read_mound_villages_data(filepath):
    """Read the mound villages data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        lats, lons = [], []
        for _, row in df.iterrows():
            if pd.notna(row['UTM X (Easting)']) and pd.notna(row['UTM Y (Northing)']):
                lat, lon = utm_to_latlon(row['UTM X (Easting)'], row['UTM Y (Northing)'], utm_zone=19)
                lats.append(lat)
                lons.append(lon)
            else:
                lats.append(np.nan)
                lons.append(np.nan)
        df['latitude'] = lats
        df['longitude'] = lons
        df['source'] = 'Mound Villages'
        return df
    except Exception as e:
        print(f"Error reading mound villages data: {e}")
        return pd.DataFrame()

def read_casarabe_sites_data(filepath):
    """Read the Casarabe sites data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        lats, lons = [], []
        for _, row in df.iterrows():
            if pd.notna(row['UTM X (Easting)']) and pd.notna(row['UTM Y (Northing)']):
                lat, lon = utm_to_latlon(row['UTM X (Easting)'], row['UTM Y (Northing)'], utm_zone=20)
                lats.append(lat)
                lons.append(lon)
            else:
                lats.append(np.nan)
                lons.append(np.nan)
        df['latitude'] = lats
        df['longitude'] = lons
        df['source'] = 'Casarabe Sites'
        return df
    except Exception as e:
        print(f"Error reading Casarabe sites data: {e}")
        return pd.DataFrame()

def read_geoglyphs_data(filepath):
    """Read the Amazon geoglyphs data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['source'] = 'Amazon Geoglyphs'
        return df
    except Exception as e:
        print(f"Error reading geoglyphs data: {e}")
        return pd.DataFrame()

def read_submit_data(filepath):
    """Read the submit.csv data."""
    try:
        df = pd.read_csv(filepath)
        df['latitude'] = np.nan
        df['longitude'] = np.nan
        
        for idx, row in df.iterrows():
            sample_x = row['x']
            sample_y = row['y']
            
            if pd.notna(sample_x) and pd.notna(sample_y):
                if -180 <= sample_x <= 180 and -90 <= sample_y <= 90:
                    df.at[idx, 'latitude'] = sample_y
                    df.at[idx, 'longitude'] = sample_x
                else:
                    # Try different UTM zones
                    for zone in [18, 19, 20, 21]:
                        try:
                            lat, lon = utm_to_latlon(sample_x, sample_y, utm_zone=zone)
                            if pd.notna(lat) and -90 <= lat <= 90 and -180 <= lon <= 180:
                                df.at[idx, 'latitude'] = lat
                                df.at[idx, 'longitude'] = lon
                                break
                        except:
                            continue
        
        df['source'] = 'Submit Data'
        return df
    except Exception as e:
        print(f"Error reading submit data: {e}")
        return pd.DataFrame()

def read_science_data(filepath):
    """Read the science data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        df['latitude'] = pd.to_numeric(df.get('Latitude', df.get('latitude')), errors='coerce')
        df['longitude'] = pd.to_numeric(df.get('Longitude', df.get('longitude')), errors='coerce')
        df['source'] = 'Science Data'
        return df
    except Exception as e:
        print(f"Error reading science data: {e}")
        return pd.DataFrame()

# --- SITE LOADING FUNCTIONS ---

def load_all_archaeological_sites():
    """Load all archaeological sites and determine bounding box."""
    datasets = [
        ('mound_villages_acre.csv', read_mound_villages_data),
        ('casarabe_sites_utm.csv', read_casarabe_sites_data),
        ('amazon_geoglyphs_sites.csv', read_geoglyphs_data),
        ('submit.csv', read_submit_data),
        ('science.ade2541_data_s2.csv', read_science_data)
    ]
    
    all_sites = []
    site_dataframes = []
    
    for filepath, read_function in datasets:
        if os.path.exists(filepath):
            print(f"Loading archaeological sites from: {filepath}")
            df = read_function(filepath)
            if not df.empty:
                # Keep valid coordinates only
                valid_df = df.dropna(subset=['latitude', 'longitude'])
                if not valid_df.empty:
                    all_sites.append(valid_df[['latitude', 'longitude']])
                    site_dataframes.append((filepath.replace('.csv', ''), valid_df))
                    print(f"  Loaded {len(valid_df)} sites")
            else:
                print(f"  No valid sites found")
        else:
            print(f"  File not found: {filepath}")
    
    if all_sites:
        combined_sites = pd.concat(all_sites, ignore_index=True)
        
        # Calculate bounding box with some padding
        padding = 0.1  # degrees (roughly 11km)
        bbox = {
            'south': combined_sites['latitude'].min() - padding,
            'north': combined_sites['latitude'].max() + padding,
            'west': combined_sites['longitude'].min() - padding,
            'east': combined_sites['longitude'].max() + padding
        }
        
        print(f"\nTotal archaeological sites: {len(combined_sites)}")
        print(f"Bounding box: N:{bbox['north']:.3f}, S:{bbox['south']:.3f}, E:{bbox['east']:.3f}, W:{bbox['west']:.3f}")
        
        return site_dataframes, bbox
    else:
        print("No archaeological sites loaded!")
        return [], None

# --- TILED RIVER EXTRACTION ---

def create_tiles(bbox, tile_size, overlap):
    """Create a grid of tiles covering the bounding box."""
    tiles = []
    
    # Calculate the number of tiles needed
    lat_range = bbox['north'] - bbox['south']
    lon_range = bbox['east'] - bbox['west']
    
    # Number of tiles (with slight overlap)
    n_lat_tiles = max(1, int(np.ceil(lat_range / (tile_size - overlap))))
    n_lon_tiles = max(1, int(np.ceil(lon_range / (tile_size - overlap))))
    
    print(f"Creating {n_lat_tiles} x {n_lon_tiles} = {n_lat_tiles * n_lon_tiles} tiles")
    print(f"Tile size: {tile_size}° x {tile_size}° (≈ {tile_size * 111:.0f}km x {tile_size * 111:.0f}km)")
    
    for i in range(n_lat_tiles):
        for j in range(n_lon_tiles):
            # Calculate tile boundaries
            south = bbox['south'] + i * (tile_size - overlap)
            north = min(bbox['north'], south + tile_size)
            west = bbox['west'] + j * (tile_size - overlap)
            east = min(bbox['east'], west + tile_size)
            
            tile = {
                'id': f"tile_{i:02d}_{j:02d}",
                'south': south,
                'north': north,
                'west': west,
                'east': east
            }
            tiles.append(tile)
    
    return tiles

def get_cache_filename(tile):
    """Generate cache filename for a tile."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f"{tile['id']}_rivers.pkl")

def load_tile_from_cache(tile):
    """Load tile data from cache if available."""
    cache_file = get_cache_filename(tile)
    if USE_CACHE and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"  Loaded from cache: {tile['id']}")
            return data['rivers'], data['river_names']
        except:
            print(f"  Cache corrupt for {tile['id']}, will re-query")
    return None, None

def save_tile_to_cache(tile, rivers, river_names):
    """Save tile data to cache."""
    cache_file = get_cache_filename(tile)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'rivers': rivers,
                'river_names': river_names,
                'timestamp': datetime.now()
            }, f)
    except Exception as e:
        print(f"  Warning: Could not save cache for {tile['id']}: {e}")

def get_rivers_from_tile(tile, timeout=30):
    """Get rivers from a single tile using OpenStreetMap."""
    # Check cache first
    rivers, river_names = load_tile_from_cache(tile)
    if rivers is not None:
        return rivers, river_names
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Create bounding box string for Overpass API
    bbox_str = f"{tile['south']},{tile['west']},{tile['north']},{tile['east']}"
    
    # Overpass query for rivers and streams
    overpass_query = f"""
    [out:json][timeout:{timeout}][bbox:{bbox_str}];
    (
      way["waterway"~"^(river|stream)$"];
      relation["waterway"~"^(river|stream)$"];
    );
    (._;>;);
    out geom;
    """
    
    print(f"  Querying {tile['id']}... ", end='', flush=True)
    
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=timeout+10)
        response.raise_for_status()
        data = response.json()
        
        rivers = []
        river_names = []
        
        # Process ways (simple linestrings)
        for element in data['elements']:
            if element['type'] == 'way' and 'geometry' in element:
                coords = [(node['lat'], node['lon']) for node in element['geometry']]
                if len(coords) >= 2:  # Need at least 2 points for a line
                    rivers.append(coords)
                    # Get river name if available
                    name = element.get('tags', {}).get('name', f"River_{len(rivers)}")
                    river_names.append(name)
        
        print(f"Found {len(rivers)} river segments")
        
        # Save to cache
        save_tile_to_cache(tile, rivers, river_names)
        
        return rivers, river_names
        
    except requests.exceptions.Timeout:
        print("TIMEOUT")
        return [], []
    except Exception as e:
        print(f"ERROR: {e}")
        return [], []

def merge_overlapping_rivers(all_rivers, all_names, merge_distance_degrees=0.01):
    """Merge rivers that are close to each other (likely the same river crossing tiles)."""
    print("Merging overlapping rivers...")
    
    if not all_rivers:
        return [], []
    
    # Convert to line geometries for easier processing
    river_lines = []
    for i, river_coords in enumerate(all_rivers):
        if len(river_coords) >= 2:
            try:
                line = LineString([(lon, lat) for lat, lon in river_coords])
                river_lines.append((line, all_names[i], i))
            except:
                continue
    
    if not river_lines:
        return [], []
    
    # Create spatial index for efficient searching
    merged_rivers = []
    merged_names = []
    used_indices = set()
    
    for i, (line1, name1, orig_idx1) in enumerate(river_lines):
        if orig_idx1 in used_indices:
            continue
            
        # Find nearby river segments
        merged_coords = list(line1.coords)
        merged_name = name1
        used_indices.add(orig_idx1)
        
        for j, (line2, name2, orig_idx2) in enumerate(river_lines):
            if i == j or orig_idx2 in used_indices:
                continue
                
            # Check if lines are close enough to merge
            try:
                if line1.distance(line2) < merge_distance_degrees:
                    # Merge the lines
                    merged_coords.extend(list(line2.coords))
                    if name2 != f"River_{orig_idx2+1}" and name1 == f"River_{orig_idx1+1}":
                        merged_name = name2  # Prefer named rivers
                    used_indices.add(orig_idx2)
            except:
                continue
        
        # Convert back to lat/lon format
        merged_river = [(lat, lon) for lon, lat in merged_coords]
        merged_rivers.append(merged_river)
        merged_names.append(merged_name)
    
    print(f"Merged {len(all_rivers)} river segments into {len(merged_rivers)} unique rivers")
    return merged_rivers, merged_names

def get_rivers_tiled(bbox, tile_size=TILE_SIZE_DEGREES, overlap=TILE_OVERLAP_DEGREES):
    """Get rivers from the entire bounding box using tiled queries."""
    print("Starting tiled river extraction...")
    
    # Create tiles
    tiles = create_tiles(bbox, tile_size, overlap)
    
    # Query each tile
    all_rivers = []
    all_river_names = []
    successful_tiles = 0
    
    print(f"\nQuerying {len(tiles)} tiles:")
    
    for i, tile in enumerate(tiles):
        print(f"Tile {i+1}/{len(tiles)}: ", end='')
        
        rivers, river_names = get_rivers_from_tile(tile, QUERY_TIMEOUT)
        
        if rivers:
            all_rivers.extend(rivers)
            all_river_names.extend(river_names)
            successful_tiles += 1
        
        # Be respectful to the API
        if i < len(tiles) - 1:  # Don't sleep after the last query
            time.sleep(DELAY_BETWEEN_QUERIES)
    
    print(f"\nCompleted {successful_tiles}/{len(tiles)} tiles successfully")
    print(f"Total river segments found: {len(all_rivers)}")
    
    # Merge overlapping rivers from different tiles
    merged_rivers, merged_names = merge_overlapping_rivers(all_rivers, all_river_names)
    
    return merged_rivers, merged_names

# --- BUFFER CREATION FUNCTIONS ---

def create_river_buffers(rivers, buffer_distance_km):
    """Create buffer zones around rivers."""
    print(f"Creating {buffer_distance_km}km buffers around rivers...")
    
    if not rivers:
        print("No rivers to buffer")
        return None
    
    # Convert to GeoDataFrame for easier processing
    river_lines = []
    for river_coords in rivers:
        if len(river_coords) >= 2:
            try:
                line = LineString([(lon, lat) for lat, lon in river_coords])
                river_lines.append(line)
            except:
                continue
    
    if not river_lines:
        print("No valid river lines to buffer")
        return None
    
    # Create GeoDataFrame with WGS84 coordinate system
    gdf = gpd.GeoDataFrame(geometry=river_lines, crs='EPSG:4326')
    
    # Project to a metric coordinate system for accurate buffering
    # Use UTM zone 19S (appropriate for western Amazon)
    gdf_utm = gdf.to_crs('EPSG:32719')
    
    # Create buffer (convert km to meters)
    buffer_distance_m = buffer_distance_km * 1000
    buffered_utm = gdf_utm.buffer(buffer_distance_m)
    
    # Project back to WGS84
    buffered_gdf = gpd.GeoDataFrame(geometry=buffered_utm, crs='EPSG:32719').to_crs('EPSG:4326')
    
    # Combine all buffers into a single polygon
    combined_buffer = unary_union(buffered_gdf.geometry)
    
    print(f"Created buffer zones around {len(rivers)} river segments")
    return combined_buffer

def convert_buffer_to_folium_coords(buffer_geom):
    """Convert Shapely geometry to Folium-compatible coordinates."""
    folium_coords = []
    
    if hasattr(buffer_geom, 'geoms'):  # MultiPolygon
        for geom in buffer_geom.geoms:
            if hasattr(geom, 'exterior'):
                coords = list(geom.exterior.coords)
                folium_coords.append([(lat, lon) for lon, lat in coords])
    elif hasattr(buffer_geom, 'exterior'):  # Single Polygon
        coords = list(buffer_geom.exterior.coords)
        folium_coords.append([(lat, lon) for lon, lat in coords])
    
    return folium_coords

# --- MAPPING FUNCTIONS ---

def create_river_buffer_map(site_dataframes, rivers, river_names, river_buffer, bbox):
    """Create an interactive map showing archaeological sites, rivers, and buffer zones."""
    
    # Calculate map center
    center_lat = (bbox['north'] + bbox['south']) / 2
    center_lon = (bbox['east'] + bbox['west']) / 2
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Add different tile layers
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name='Stamen Terrain',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='cartodbpositron',
        name='CartoDB Positron'
    ).add_to(m)
    
    # Color mapping for different archaeological datasets
    site_colors = {
        'mound_villages_acre': 'red',
        'casarabe_sites_utm': 'blue',
        'amazon_geoglyphs_sites': 'orange',
        'submit': 'green',
        'science.ade2541_data_s2': 'purple'
    }
    
    # Create feature groups
    river_group = folium.FeatureGroup(name="Rivers")
    buffer_group = folium.FeatureGroup(name=f"{BUFFER_DISTANCE_KM}km River Buffers")
    
    # Add archaeological sites (sample if too many)
    site_groups = {}
    total_sites = 0
    
    for dataset_name, df in site_dataframes:
        group_name = f"Archaeological Sites - {dataset_name.replace('_', ' ').title()}"
        site_group = folium.FeatureGroup(name=group_name)
        
        color = site_colors.get(dataset_name, 'black')
        
        # Sample large datasets for better performance
        display_df = df
        if len(df) > 1000:
            display_df = df.sample(n=1000, random_state=42)
            print(f"Sampling {len(display_df)} sites from {len(df)} total in {dataset_name}")
        
        for idx, row in display_df.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                # Create popup text
                popup_text = f"<b>Dataset:</b> {row.get('source', dataset_name)}<br>"
                
                # Add site-specific information
                if 'Site Name' in row and pd.notna(row['Site Name']):
                    popup_text += f"<b>Site:</b> {row['Site Name']}<br>"
                elif 'Site' in row and pd.notna(row['Site']):
                    popup_text += f"<b>Site:</b> {row['Site']}<br>"
                elif 'name' in row and pd.notna(row['name']):
                    popup_text += f"<b>Site:</b> {row['name']}<br>"
                
                popup_text += f"<b>Coordinates:</b> {row['latitude']:.6f}, {row['longitude']:.6f}"
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    popup=folium.Popup(popup_text, max_width=300),
                    color=color,
                    fillColor=color,
                    fillOpacity=0.8,
                    weight=1
                ).add_to(site_group)
                
                total_sites += 1
        
        site_group.add_to(m)
    
    # Add rivers (sample if too many)
    display_rivers = rivers
    display_names = river_names
    if len(rivers) > 500:
        indices = np.random.choice(len(rivers), 500, replace=False)
        display_rivers = [rivers[i] for i in indices]
        display_names = [river_names[i] for i in indices]
        print(f"Sampling 500 rivers from {len(rivers)} total for display")
    
    for river_coords, river_name in zip(display_rivers, display_names):
        if len(river_coords) >= 2:
            folium.PolyLine(
                locations=river_coords,
                popup=f"<b>River:</b> {river_name}",
                color='blue',
                weight=1,
                opacity=0.7
            ).add_to(river_group)
    
    # Add river buffers
    if river_buffer:
        buffer_coords = convert_buffer_to_folium_coords(river_buffer)
        
        for coords in buffer_coords:
            if len(coords) >= 3:  # Need at least 3 points for a polygon
                folium.Polygon(
                    locations=coords,
                    popup=f"<b>{BUFFER_DISTANCE_KM}km Buffer Zone</b><br>Potential archaeological area near water",
                    color='lightblue',
                    weight=1,
                    fillColor='lightblue',
                    fillOpacity=0.2
                ).add_to(buffer_group)
    
    # Add feature groups to map
    river_group.add_to(m)
    buffer_group.add_to(m)
    
    # Add bounding box rectangle
    bbox_coords = [
        [bbox['south'], bbox['west']],
        [bbox['north'], bbox['west']],
        [bbox['north'], bbox['east']],
        [bbox['south'], bbox['east']]
    ]
    
    folium.Polygon(
        locations=bbox_coords,
        popup="<b>Study Area Boundary</b>",
        color='red',
        weight=2,
        fill=False,
        dashArray='5, 5'
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add minimap
    minimap = plugins.MiniMap()
    m.add_child(minimap)
    
    # Add measure control
    plugins.MeasureControl(primary_length_unit='kilometers').add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    print(f"Created map with:")
    print(f"  - {total_sites} archaeological sites (sampled for display)")
    print(f"  - {len(display_rivers)} river segments (sampled for display)")
    print(f"  - Buffer zones around all {len(rivers)} rivers ({BUFFER_DISTANCE_KM}km)")
    
    return m

# --- MAIN EXECUTION ---

def main():
    """Main function to create river buffer analysis using tiled approach."""
    print("=== Tiled River Buffer Analysis for Archaeological Sites ===\n")
    
    # Step 1: Load all archaeological sites and determine bounding box
    site_dataframes, bbox = load_all_archaeological_sites()
    
    if not site_dataframes or not bbox:
        print("No archaeological sites found or unable to determine bounding box.")
        return
    
    # Step 2: Get rivers using tiled approach
    print(f"\nStep 2: Extracting rivers using tiled approach")
    rivers, river_names = get_rivers_tiled(bbox)
    
    if not rivers:
        print("No rivers found in the study area.")
        return
    
    # Step 3: Create buffer zones around rivers
    print(f"\nStep 3: Creating buffer zones")
    river_buffer = create_river_buffers(rivers, BUFFER_DISTANCE_KM)
    
    # Step 4: Create the map
    print(f"\nStep 4: Creating interactive map...")
    river_map = create_river_buffer_map(site_dataframes, rivers, river_names, river_buffer, bbox)
    
    # Step 5: Save the map
    output_filename = 'archaeological_sites_river_buffers_tiled.html'
    river_map.save(output_filename)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Map saved as: {output_filename}")
    print(f"Cache directory: {CACHE_DIR} (for faster future runs)")
    print(f"\nSummary:")
    print(f"- Total river segments: {len(rivers)}")
    print(f"- Buffer zones: {BUFFER_DISTANCE_KM}km around all rivers")
    print(f"- Study area: {bbox['north']-bbox['south']:.1f}° x {bbox['east']-bbox['west']:.1f}°")
    
    # Optional: Save river buffer as shapefile for further analysis
    if river_buffer:
        try:
            buffer_gdf = gpd.GeoDataFrame(
                {'buffer_km': [BUFFER_DISTANCE_KM]}, 
                geometry=[river_buffer], 
                crs='EPSG:4326'
            )
            buffer_gdf.to_file('river_buffers_tiled.shp')
            print(f"River buffer zones also saved as shapefile: river_buffers_tiled.shp")
        except Exception as e:
            print(f"Note: Could not save shapefile: {e}")
    
    # Save rivers as GeoJSON for future use
    try:
        rivers_data = []
        for i, (river_coords, name) in enumerate(zip(rivers, river_names)):
            if len(river_coords) >= 2:
                line = LineString([(lon, lat) for lat, lon in river_coords])
                rivers_data.append({'geometry': line, 'name': name, 'id': i})
        
        rivers_gdf = gpd.GeoDataFrame(rivers_data, crs='EPSG:4326')
        rivers_gdf.to_file('extracted_rivers.geojson', driver='GeoJSON')
        print(f"Rivers also saved as GeoJSON: extracted_rivers.geojson")
    except Exception as e:
        print(f"Note: Could not save rivers GeoJSON: {e}")
    
    return river_map

# Run the analysis
if __name__ == "__main__":
    map_object = main()


import os
import pickle
import re
import folium
from folium import plugins
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import unary_union
from pyproj import Transformer # Used by the buffer function if CRS transformation is needed
import numpy as np
import time
import platform

# --- CONFIGURATION ---
CACHE_DIR = "/kaggle/input/river-segments/river_cache"  # Directory where your .pkl files are stored
OUTPUT_HTML_FILE = "cached_rivers_map.html" # Name of the output HTML map file

# Assumed parameters from your tiling process (based on your last script)
# These are used to reconstruct and display the tile boundaries
TILE_SIZE_DEGREES_ASSUMED = 1.0
TILE_OVERLAP_DEGREES_ASSUMED = 0.1 # This is effectively (TILE_SIZE_DEGREES_ASSUMED - step_size)

# Buffer configuration (set to None to disable buffer calculation)
BUFFER_DISTANCE_KM = 10.0 # Buffer distance in kilometers, e.g., 10.0
# BUFFER_DISTANCE_KM = None # Uncomment to disable buffers

# Default map settings if no data is found
DEFAULT_MAP_CENTER = [-16.0, -60.0]
DEFAULT_MAP_ZOOM = 6

# Map settings
MAX_RIVERS_TO_DISPLAY = 50000  # Limit very large maps to improve performance

# --- HELPER FUNCTIONS (adapted from your original script) ---

def convert_geom_to_folium_coords(geometry):
    """
    Convert Shapely geometry (Polygon or MultiPolygon)
    to Folium-compatible coordinates.
    """
    folium_coords = []
    if geometry is None or geometry.is_empty:
        return folium_coords

    if geometry.geom_type == 'Polygon':
        # Exterior coordinates
        coords = list(geometry.exterior.coords)
        folium_coords.append([(lat, lon) for lon, lat in coords])
        # Interior coordinates (holes)
        for interior in geometry.interiors:
            coords_interior = list(interior.coords)
            folium_coords.append([(lat, lon) for lon, lat in coords_interior])
    elif geometry.geom_type == 'MultiPolygon':
        for geom_part in geometry.geoms: # Use .geoms for MultiPolygon
            # Exterior coordinates
            coords = list(geom_part.exterior.coords)
            folium_coords.append([(lat, lon) for lon, lat in coords])
            # Interior coordinates (holes)
            for interior in geom_part.interiors:
                coords_interior = list(interior.coords)
                folium_coords.append([(lat, lon) for lon, lat in coords_interior])
    return folium_coords


def create_river_buffers_for_tile(rivers_in_tile, buffer_distance_km, tile_id=""):
    """
    Create buffer zones around a list of rivers for a specific tile.
    Optimized version with fixed UTM projection.
    """
    if not rivers_in_tile or buffer_distance_km is None or buffer_distance_km <= 0:
        return None

    river_lines = []
    for river_coords in rivers_in_tile:
        if len(river_coords) >= 2:
            try:
                # Ensure coordinates are (lon, lat) for LineString
                line = LineString([(lon, lat) for lat, lon in river_coords])
                river_lines.append(line)
            except Exception as e:
                print(f"Warning: Could not create LineString for a river segment in {tile_id}: {e}")
                continue

    if not river_lines:
        return None

    # Create GeoDataFrame with WGS84 coordinate system (EPSG:4326)
    gdf = gpd.GeoDataFrame(geometry=river_lines, crs='EPSG:4326')

    # Use a common UTM zone for the Amazon region (UTM 19S)
    try:
        gdf_utm = gdf.to_crs('EPSG:32719')  # UTM Zone 19S
    except Exception as e:
        print(f"Warning: Error projecting to UTM for {tile_id}: {e}")
        return None

    # Create buffer (convert km to meters)
    buffer_distance_m = buffer_distance_km * 1000
    try:
        buffered_utm = gdf_utm.buffer(buffer_distance_m, cap_style=1, join_style=1) # cap_style 1=round, join_style 1=round
    except Exception as e:
        print(f"Error during buffering operation for {tile_id}: {e}")
        return None

    # Project back to WGS84
    try:
        buffered_gdf = gpd.GeoDataFrame(geometry=buffered_utm, crs='EPSG:32719').to_crs('EPSG:4326')
    except Exception as e:
        print(f"Error projecting buffer back to WGS84 for {tile_id}: {e}")
        return None

    # Combine all buffers for this tile into a single geometry
    if not buffered_gdf.empty:
        combined_buffer_for_tile = unary_union(buffered_gdf.geometry)
        return combined_buffer_for_tile
    else:
        return None

# --- MAIN SCRIPT LOGIC ---
def main():
    """
    Loads river data from pickle files, generates tile boundaries and optional buffers,
    and creates an interactive HTML map.
    """
    print(f"--- Starting Cached River Map Generation ---")
    print(f"Scanning for pickle files in: {CACHE_DIR}")

    if not os.path.isdir(CACHE_DIR):
        print(f"Error: Cache directory '{CACHE_DIR}' not found.")
        return

    # Store data from pickles: (tile_indices, rivers, river_names, tile_id_str)
    loaded_tile_data = []
    all_lats = []
    all_lons = []

    # Regex to parse tile indices from filename e.g., tile_0_0_rivers.pkl or tile_00_01_rivers.pkl
    # Allows for one or more digits for i and j
    tile_filename_pattern = re.compile(r"tile_(\d+)_(\d+)_rivers\.pkl")

    for filename in os.listdir(CACHE_DIR):
        match = tile_filename_pattern.match(filename)
        if match:
            tile_i = int(match.group(1))
            tile_j = int(match.group(2))
            tile_id_str = f"tile_{tile_i}_{tile_j}"
            filepath = os.path.join(CACHE_DIR, filename)

            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

                rivers = data.get('rivers')
                river_names = data.get('river_names') # Optional

                if rivers and isinstance(rivers, list):
                    print(f"  Loaded {len(rivers)} river segments from {filename}")
                    loaded_tile_data.append(((tile_i, tile_j), rivers, river_names, tile_id_str))
                    # Sample a subset of points to avoid memory issues
                    sample_rate = max(1, len(rivers) // 100)  # Sample at most 100 points per file
                    for i, river_segment in enumerate(rivers):
                        if i % sample_rate == 0:  # Sample only every Nth river
                            for lat, lon in river_segment:
                                all_lats.append(lat)
                                all_lons.append(lon)
                else:
                    print(f"  Warning: No 'rivers' data or invalid format in {filename}")

            except pickle.UnpicklingError:
                print(f"  Error: Could not unpickle {filename}. File might be corrupted.")
            except Exception as e:
                print(f"  Error: Could not process {filename}: {e}")

    if not loaded_tile_data:
        print("No valid river data loaded from pickle files. Cannot generate map.")
        # Create an empty map or a map with a message
        m = folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=DEFAULT_MAP_ZOOM, tiles="OpenStreetMap")
        folium.Marker(
            DEFAULT_MAP_CENTER,
            popup="No river data found in the cache directory."
        ).add_to(m)
        m.save(OUTPUT_HTML_FILE)
        print(f"Empty map saved to {OUTPUT_HTML_FILE}")
        return

    # Determine overall map bounds and center
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    map_center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    
    # Adjust zoom level based on the extent of the data
    # This is a heuristic and might need tuning
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon
    max_diff = max(lat_diff, lon_diff)
    
    if max_diff == 0: # Single point or very small area
        zoom_start = 12
    elif max_diff < 0.1:
        zoom_start = 13
    elif max_diff < 0.5:
        zoom_start = 11
    elif max_diff < 1:
        zoom_start = 10
    elif max_diff < 5:
        zoom_start = 8
    elif max_diff < 10:
        zoom_start = 7
    else:
        zoom_start = 5  # Start with a wider view for large areas


    print(f"\nOverall data extent: Lat ({min_lat:.3f} to {max_lat:.3f}), Lon ({min_lon:.3f} to {max_lon:.3f})")
    print(f"Map centered at: {map_center}, Zoom: {zoom_start}")

    # Create base map
    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="OpenStreetMap")
    folium.TileLayer('cartodbpositron', name="CartoDB Positron").add_to(m)
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name='Stamen Terrain',
        overlay=False,
        control=True
    ).add_to(m)

    # Create feature groups
    rivers_group = folium.FeatureGroup(name="Rivers (from Cache)", show=True)
    tile_boundaries_group = folium.FeatureGroup(name="Tile Boundaries (Calculated)", show=False)  # Default hidden
    if BUFFER_DISTANCE_KM is not None and BUFFER_DISTANCE_KM > 0:
        river_buffers_group = folium.FeatureGroup(name=f"{BUFFER_DISTANCE_KM}km River Buffers", show=True)
    else:
        river_buffers_group = None # No buffer group if disabled

    # The 'step size' for tiling, considering overlap
    tile_step_degrees = TILE_SIZE_DEGREES_ASSUMED - TILE_OVERLAP_DEGREES_ASSUMED

    # Count total rivers for limiting display if needed
    total_river_count = sum(len(rivers) for _, rivers, _, _ in loaded_tile_data)
    
    # Calculate sampling rate if needed
    if total_river_count > MAX_RIVERS_TO_DISPLAY:
        sampling_rate = max(1, total_river_count // MAX_RIVERS_TO_DISPLAY)
        print(f"\nLimiting display to ~{MAX_RIVERS_TO_DISPLAY} rivers out of {total_river_count}")
        print(f"Using sampling rate of 1:{sampling_rate} for better performance")
    else:
        sampling_rate = 1
        print(f"\nDisplaying all {total_river_count} rivers")

    # Process each loaded tile's data
    river_counter = 0
    for (tile_i, tile_j), rivers, river_names, tile_id_str in loaded_tile_data:
        # 1. Add rivers to map (with sampling if needed)
        if rivers:
            for idx, river_coords in enumerate(rivers):
                # Apply sampling for large datasets
                if idx % sampling_rate != 0 and sampling_rate > 1:
                    continue
                    
                if len(river_coords) >= 2:
                    river_name = river_names[idx] if river_names and idx < len(river_names) else f"River segment from {tile_id_str}"
                    folium.PolyLine(
                        locations=river_coords, # Assumes (lat, lon)
                        tooltip=river_name,
                        popup=f"<b>{river_name}</b><br>Source: {tile_id_str}",
                        color='blue',
                        weight=1.2,  # Slightly thinner for better performance
                        opacity=0.7   # Slightly more transparent
                    ).add_to(rivers_group)
                    river_counter += 1

        # 2. Calculate and add tile boundary to map
        # But only do it for a subset of tiles to avoid overloading the map
        if (tile_i % 3 == 0 and tile_j % 3 == 0) or total_river_count < 10000:
            tile_south = min_lat + tile_i * tile_step_degrees
            tile_west = min_lon + tile_j * tile_step_degrees
            tile_north = tile_south + TILE_SIZE_DEGREES_ASSUMED
            tile_east = tile_west + TILE_SIZE_DEGREES_ASSUMED

            tile_bounds_coords = [
                (tile_south, tile_west),
                (tile_north, tile_west),
                (tile_north, tile_east),
                (tile_south, tile_east),
                (tile_south, tile_west) # Close the polygon
            ]
            folium.Polygon(
                locations=tile_bounds_coords,
                tooltip=f"Boundary for {tile_id_str}",
                popup=f"<b>Calculated Boundary for {tile_id_str}</b><br>Indices (i,j): ({tile_i}, {tile_j})",
                color='gray',
                weight=1,
                fill=True,
                fill_color='gray',
                fill_opacity=0.05, # Very light fill
                dash_array='5, 5'
            ).add_to(tile_boundaries_group)

        # 3. Calculate and add river buffers (if enabled)
        # For very large datasets, only create buffers for a subset of rivers
        if river_buffers_group and rivers:
            # If many tiles, limit buffer calculation to prevent memory issues
            if total_river_count > 100000 and (tile_i + tile_j) % 5 != 0:
                continue
                
            # Apply sampling for large datasets
            sampled_rivers = []
            if len(rivers) > 100:
                sample_rate = max(1, len(rivers) // 100)
                sampled_rivers = [rivers[i] for i in range(0, len(rivers), sample_rate)]
            else:
                sampled_rivers = rivers
                
            buffer_geom = create_river_buffers_for_tile(sampled_rivers, BUFFER_DISTANCE_KM, tile_id_str)
            if buffer_geom:
                buffer_folium_coords = convert_geom_to_folium_coords(buffer_geom)
                for poly_coords in buffer_folium_coords:
                    if len(poly_coords) >= 3:
                        folium.Polygon(
                            locations=poly_coords,
                            popup=f"<b>{BUFFER_DISTANCE_KM}km Buffer</b><br>From: {tile_id_str}",
                            color='green',
                            weight=1,
                            fillColor='green',
                            fillOpacity=0.2
                        ).add_to(river_buffers_group)

    print(f"Added {river_counter} river segments to the map")

    # Add feature groups to the map
    rivers_group.add_to(m)
    tile_boundaries_group.add_to(m)
    if river_buffers_group:
        river_buffers_group.add_to(m)

    # Add LayerControl
    folium.LayerControl().add_to(m)

    # Add some map plugins - but limit for large maps
    plugins.MiniMap(toggle_display=True).add_to(m)
    plugins.MeasureControl(primary_length_unit='kilometers').add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.LocateControl().add_to(m)
    plugins.MousePosition(
        position="topright",
        separator=" | Lng: ",
        empty_string="NaN",
        lng_first=False,
        num_digits=5,
        prefix="Lat: "
    ).add_to(m)

    # Add HTML note about river count
    if sampling_rate > 1:
        html_note = f"""
        <div style="position: fixed; bottom: 10px; left: 10px; z-index: 1000; background-color: white; 
                    padding: 10px; border-radius: 5px; border: 1px solid black; max-width: 300px;">
            <p><b>Note:</b> Showing {river_counter:,} of {total_river_count:,} rivers 
            (sampled at 1:{sampling_rate} for performance).</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(html_note))

    # Save the map
    try:
        m.save(OUTPUT_HTML_FILE)
        print(f"\n--- Map generation complete! ---")
        print(f"Interactive map saved as: {os.path.abspath(OUTPUT_HTML_FILE)}")
        
        # Add option to display FileLink for downloading the map in Kaggle
        try:
            from IPython.display import FileLink, display
            print("You can download the HTML file to view it locally:")
            display(FileLink(OUTPUT_HTML_FILE))
        except:
            pass
        
    except Exception as e:
        print(f"Error saving map: {e}")

if __name__ == "__main__":
    main()


from IPython.display import Image, display

# Display image from file
image_path = '/kaggle/input/rivers-segments-image/cached_rivers_map.png' 
display(Image(filename=image_path))


import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
import geopandas as gpd
from pyproj import Transformer
import folium
from folium import plugins
from IPython.display import display

# Configuration
BUFFER_DISTANCE_KM = 10.0  # Buffer distance around rivers (approximately 6 miles)
BUFFER_DISTANCE_MILES = BUFFER_DISTANCE_KM * 0.621371  # For display purposes
CACHE_DIR = "/kaggle/input/river-segments/river_cache"  # Path to river cache files in Kaggle

# --- HELPER FUNCTIONS ---

def utm_to_latlon(utm_x, utm_y, utm_zone=19, hemisphere='south'):
    """Convert UTM coordinates to latitude/longitude."""
    if hemisphere.lower() not in ['north', 'south']:
        hemisphere = 'south'
    base_epsg = 32600 if hemisphere.lower() == 'north' else 32700
    try:
        utm_zone_int = int(utm_zone)
        if not (1 <= utm_zone_int <= 60):
            return np.nan, np.nan
        utm_crs_epsg = base_epsg + utm_zone_int
    except ValueError:
        return np.nan, np.nan

    utm_crs = f"EPSG:{utm_crs_epsg}"
    wgs84_crs = "EPSG:4326"
    try:
        transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
        lon, lat = transformer.transform(utm_x, utm_y)
        return lat, lon
    except Exception:
        return np.nan, np.nan

# --- DATA LOADING FUNCTIONS ---

def read_mound_villages_data(filepath):
    """Read the mound villages data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        lats, lons = [], []
        for _, row in df.iterrows():
            if pd.notna(row['UTM X (Easting)']) and pd.notna(row['UTM Y (Northing)']):
                lat, lon = utm_to_latlon(row['UTM X (Easting)'], row['UTM Y (Northing)'], utm_zone=19)
                lats.append(lat)
                lons.append(lon)
            else:
                lats.append(np.nan)
                lons.append(np.nan)
        df['latitude'] = lats
        df['longitude'] = lons
        df['source'] = 'Mound Villages'
        df['dataset'] = 'mound_villages'
        return df
    except Exception as e:
        print(f"Error reading mound villages data: {e}")
        return pd.DataFrame()

def read_casarabe_sites_data(filepath):
    """Read the Casarabe sites data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        lats, lons = [], []
        for _, row in df.iterrows():
            if pd.notna(row['UTM X (Easting)']) and pd.notna(row['UTM Y (Northing)']):
                lat, lon = utm_to_latlon(row['UTM X (Easting)'], row['UTM Y (Northing)'], utm_zone=20)
                lats.append(lat)
                lons.append(lon)
            else:
                lats.append(np.nan)
                lons.append(np.nan)
        df['latitude'] = lats
        df['longitude'] = lons
        df['source'] = 'Casarabe Sites'
        df['dataset'] = 'casarabe_sites'
        return df
    except Exception as e:
        print(f"Error reading Casarabe sites data: {e}")
        return pd.DataFrame()

def read_geoglyphs_data(filepath):
    """Read the Amazon geoglyphs data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['source'] = 'Amazon Geoglyphs'
        df['dataset'] = 'amazon_geoglyphs'
        return df
    except Exception as e:
        print(f"Error reading geoglyphs data: {e}")
        return pd.DataFrame()

def read_survey_data(filepath):
    """Read the archaeological survey data."""
    try:
        df = pd.read_csv(filepath)
        df['latitude'] = np.nan
        df['longitude'] = np.nan
        
        for idx, row in df.iterrows():
            sample_x = row.get('x')
            sample_y = row.get('y')
            
            if pd.notna(sample_x) and pd.notna(sample_y):
                if -180 <= sample_x <= 180 and -90 <= sample_y <= 90:
                    df.at[idx, 'latitude'] = sample_y
                    df.at[idx, 'longitude'] = sample_x
                else:
                    # Try different UTM zones
                    for zone in [18, 19, 20, 21]:
                        try:
                            lat, lon = utm_to_latlon(sample_x, sample_y, utm_zone=zone)
                            if pd.notna(lat) and -90 <= lat <= 90 and -180 <= lon <= 180:
                                df.at[idx, 'latitude'] = lat
                                df.at[idx, 'longitude'] = lon
                                break
                        except:
                            continue
        
        df['source'] = 'Archaeological Survey Data'
        df['dataset'] = 'survey_data'
        return df
    except Exception as e:
        print(f"Error reading survey data: {e}")
        return pd.DataFrame()

def read_science_data(filepath):
    """Read the science data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        df['latitude'] = pd.to_numeric(df.get('Latitude', df.get('latitude')), errors='coerce')
        df['longitude'] = pd.to_numeric(df.get('Longitude', df.get('longitude')), errors='coerce')
        df['source'] = 'Science Data'
        df['dataset'] = 'science_data'
        return df
    except Exception as e:
        print(f"Error reading science data: {e}")
        return pd.DataFrame()

# --- LOAD RIVER DATA ---

def load_river_data():
    """Load river data from pickle files."""
    print("Loading river data from cache files...")
    
    if not os.path.exists(CACHE_DIR):
        print(f"River cache directory not found: {CACHE_DIR}")
        print("Using simplified placeholder river data instead")
        return load_placeholder_river_data()
    
    # Find all pickle files that contain river data
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('_rivers.pkl')]
    
    if not cache_files:
        print("No cached river files found, using placeholder data")
        return load_placeholder_river_data()
    
    print(f"Found {len(cache_files)} river cache files")
    
    all_rivers = []
    all_river_names = []
    
    for cache_file in cache_files:
        cache_path = os.path.join(CACHE_DIR, cache_file)
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            rivers = data.get('rivers', [])
            river_names = data.get('river_names', [])
            
            all_rivers.extend(rivers)
            all_river_names.extend(river_names)
            
            # print(f"  {cache_file}: {len(rivers)} river segments")
            
        except Exception as e:
            print(f"  Error loading {cache_file}: {e}")
    
    print(f"Total river segments loaded: {len(all_rivers)}")
    
    if not all_rivers:
        print("No river data could be loaded from cache, using placeholder data")
        return load_placeholder_river_data()
        
    return all_rivers, all_river_names

def load_placeholder_river_data():
    """Create placeholder river data (only used if cache loading fails)."""
    print("Creating simplified placeholder river data...")
    
    # Create some example rivers in the Amazon region
    amazon_river = [
        [-3.1133, -60.0253],  # Near Manaus
        [-3.0451, -59.9543],
        [-2.9768, -59.8834],
        [-2.9086, -59.8124],
        [-2.8403, -59.7415]
    ]
    
    madeira_river = [
        [-3.4680, -58.7537],
        [-3.5363, -58.8246],
        [-3.6046, -58.8956],
        [-3.6728, -58.9665],
        [-3.7411, -59.0375]
    ]
    
    negro_river = [
        [-3.0860, -60.0962],
        [-3.0177, -60.1672],
        [-2.9495, -60.2381],
        [-2.8812, -60.3091],
        [-2.8129, -60.3800]
    ]
    
    rivers = [amazon_river, madeira_river, negro_river]
    river_names = ["Amazon", "Madeira", "Negro"]
    
    print(f"Created {len(rivers)} placeholder river segments")
    return rivers, river_names

# --- MAIN ANALYSIS FUNCTIONS ---

def load_archaeological_sites():
    """Load archaeological site data from Kaggle input files."""
    print("Loading archaeological sites...")
    
    datasets = [
        ('/kaggle/input/mound-villages-acre/mound_villages_acre.csv', read_mound_villages_data),
        ('/kaggle/input/casarabe-sites-utm/casarabe_sites_utm.csv', read_casarabe_sites_data),
        ('/kaggle/input/amazon-geoglyphs-sites/amazon_geoglyphs_sites.csv', read_geoglyphs_data),
        ('/kaggle/input/archaeological-survey-data/submit.csv', read_survey_data),
        ('/kaggle/input/science-data/science.ade2541_data_s2.csv', read_science_data)
    ]
    
    all_sites = []
    dataset_counts = {}
    
    for filepath, read_function in datasets:
        try:
            print(f"  Processing: {filepath}")
            df = read_function(filepath)
            
            if not df.empty:
                # Keep only valid coordinates
                valid_df = df.dropna(subset=['latitude', 'longitude'])
                if not valid_df.empty:
                    dataset_name = valid_df['dataset'].iloc[0]
                    dataset_counts[dataset_name] = len(valid_df)
                    all_sites.append(valid_df)
                    print(f"    Loaded {len(valid_df)} sites with valid coordinates")
                else:
                    print(f"    No valid coordinates found")
            else:
                print(f"    No data loaded")
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    if all_sites:
        combined_sites = pd.concat(all_sites, ignore_index=True)
        print(f"\nTotal archaeological sites loaded: {len(combined_sites)}")
        return combined_sites, dataset_counts
    else:
        print("No archaeological sites could be loaded!")
        return pd.DataFrame(), {}

def create_river_buffer(rivers, buffer_distance_km):
    """Create buffer zones around rivers."""
    print(f"Creating {buffer_distance_km}km buffer zones around rivers...")
    
    if not rivers:
        print("No rivers to buffer")
        return None
    
    # Convert rivers to LineString geometries
    river_lines = []
    for river_coords in rivers:
        if len(river_coords) >= 2:
            try:
                # Standard GIS order is (lon, lat)
                line = LineString([(lon, lat) for lat, lon in river_coords])
                river_lines.append(line)
            except Exception as e:
                print(f"Error creating line: {e}")
                continue
    
    if not river_lines:
        print("No valid river lines created")
        return None
    
    # Create GeoDataFrame with WGS84 coordinate system
    gdf = gpd.GeoDataFrame(geometry=river_lines, crs='EPSG:4326')
    
    # Project to a metric coordinate system for accurate buffering
    # UTM zone 19S is appropriate for much of the Amazon
    gdf_utm = gdf.to_crs('EPSG:32719')
    
    # Create buffer (convert km to meters)
    buffer_distance_m = buffer_distance_km * 1000
    buffered_utm = gdf_utm.buffer(buffer_distance_m)
    
    # Project back to WGS84
    buffered_gdf = gpd.GeoDataFrame(geometry=buffered_utm, crs='EPSG:32719').to_crs('EPSG:4326')
    
    # Combine all buffers into a single polygon
    combined_buffer = unary_union(buffered_gdf.geometry)
    
    print(f"Buffer zones created successfully")
    return combined_buffer

def analyze_site_proximity(sites_df, river_buffer):
    """Analyze which archaeological sites fall within river buffer zones."""
    print(f"Analyzing proximity of {len(sites_df)} sites to river buffers...")
    
    if river_buffer is None:
        print("No river buffer available for analysis")
        return sites_df, None
    
    # Convert sites to geometry points
    geometry = [Point(lon, lat) for lon, lat in zip(sites_df['longitude'], sites_df['latitude'])]
    sites_gdf = gpd.GeoDataFrame(sites_df, geometry=geometry, crs='EPSG:4326')
    
    # Check which sites are within the buffer
    sites_gdf['near_river'] = sites_gdf.geometry.within(river_buffer) | sites_gdf.geometry.intersects(river_buffer)
    
    # Calculate statistics
    total_sites = len(sites_gdf)
    sites_near_rivers = sites_gdf['near_river'].sum()
    percentage_near_rivers = (sites_near_rivers / total_sites) * 100 if total_sites > 0 else 0
    
    # Statistics by dataset
    dataset_stats = {}
    for dataset in sites_gdf['dataset'].unique():
        dataset_sites = sites_gdf[sites_gdf['dataset'] == dataset]
        dataset_near_rivers = dataset_sites['near_river'].sum()
        dataset_total = len(dataset_sites)
        dataset_percentage = (dataset_near_rivers / dataset_total) * 100 if dataset_total > 0 else 0
        
        dataset_stats[dataset] = {
            'total_sites': dataset_total,
            'sites_near_rivers': dataset_near_rivers,
            'percentage_near_rivers': dataset_percentage
        }
    
    results = {
        'total_sites': total_sites,
        'sites_near_rivers': sites_near_rivers,
        'sites_away_from_rivers': total_sites - sites_near_rivers,
        'percentage_near_rivers': percentage_near_rivers,
        'dataset_stats': dataset_stats
    }
    
    print(f"Proximity analysis complete:")
    print(f"  Total sites: {total_sites}")
    print(f"  Sites within {BUFFER_DISTANCE_KM}km of rivers: {sites_near_rivers} ({percentage_near_rivers:.1f}%)")
    print(f"  Sites more than {BUFFER_DISTANCE_KM}km from rivers: {total_sites - sites_near_rivers} ({100-percentage_near_rivers:.1f}%)")
    
    return sites_gdf, results

# --- VISUALIZATION FUNCTIONS ---

def create_summary_visualizations(results):
    """Create summary visualizations of the proximity analysis results."""
    if not results:
        print("No results to visualize")
        return
    
    # Setup the figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Archaeological Sites Proximity to Rivers ({BUFFER_DISTANCE_KM}km / {BUFFER_DISTANCE_MILES:.1f} miles)', 
                 fontsize=16, y=0.98)
    
    # 1. Overall proximity pie chart
    labels = [f'Within {BUFFER_DISTANCE_KM}km of rivers', f'Beyond {BUFFER_DISTANCE_KM}km from rivers']
    sizes = [results['sites_near_rivers'], results['sites_away_from_rivers']]
    colors = ['#66c2a5', '#fc8d62']
    explode = (0.1, 0)  # Explode the first slice (Near rivers)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', 
            shadow=True, startangle=90, textprops={'fontsize': 9})
    ax1.set_title('Proportion of Sites Near Rivers', pad=20)
    
    # 2. Dataset breakdown bar chart
    datasets = list(results['dataset_stats'].keys())
    near_counts = [results['dataset_stats'][ds]['sites_near_rivers'] for ds in datasets]
    far_counts = [results['dataset_stats'][ds]['total_sites'] - results['dataset_stats'][ds]['sites_near_rivers'] for ds in datasets]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    rects1 = ax2.bar(x - width/2, near_counts, width, label=f'Within {BUFFER_DISTANCE_KM}km', color='#66c2a5')
    rects2 = ax2.bar(x + width/2, far_counts, width, label=f'Beyond {BUFFER_DISTANCE_KM}km', color='#fc8d62')
    
    ax2.set_ylabel('Number of Sites')
    ax2.set_title('Site Count by Dataset')
    ax2.set_xticks(x)
    dataset_labels = [ds.replace('_', ' ').title() for ds in datasets]
    ax2.set_xticklabels(dataset_labels, rotation=45, ha='right')
    ax2.legend()
    
    # Add count labels on bars
    for rect in rects1:
        height = rect.get_height()
        if height > 0:
            ax2.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    for rect in rects2:
        height = rect.get_height()
        if height > 0:
            ax2.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # 3. Percentage breakdown by dataset
    percentages = [results['dataset_stats'][ds]['percentage_near_rivers'] for ds in datasets]
    
    bars = ax3.bar(x, percentages, color='#8da0cb')
    ax3.set_ylabel('Percentage Near Rivers (%)')
    ax3.set_title('Percentage of Sites Within River Buffer by Dataset')
    ax3.set_xticks(x)
    ax3.set_xticklabels(dataset_labels, rotation=45, ha='right')
    ax3.set_ylim(0, 100)
    ax3.axhline(y=results['percentage_near_rivers'], color='r', linestyle='-', alpha=0.5, 
                label=f'Overall average: {results["percentage_near_rivers"]:.1f}%')
    ax3.legend()
    
    # Add percentage labels on bars
    for i, v in enumerate(percentages):
        ax3.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=8)
    
    # 4. Summary text box
    ax4.axis('off')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
    summary_text = (
        f"PROXIMITY ANALYSIS SUMMARY\n\n"
        f"Total archaeological sites analyzed: {results['total_sites']}\n\n"
        f"Sites within {BUFFER_DISTANCE_KM}km ({BUFFER_DISTANCE_MILES:.1f} miles) of rivers: \n"
        f"   {results['sites_near_rivers']} sites ({results['percentage_near_rivers']:.1f}%)\n\n"
        f"Sites beyond {BUFFER_DISTANCE_KM}km from rivers: \n"
        f"   {results['sites_away_from_rivers']} sites ({100-results['percentage_near_rivers']:.1f}%)\n\n"
        f"Dataset with highest river proximity: \n"
        f"   {max(results['dataset_stats'].items(), key=lambda x: x[1]['percentage_near_rivers'])[0].replace('_', ' ').title()} "
        f"({max([stats['percentage_near_rivers'] for stats in results['dataset_stats'].values()]):.1f}%)\n\n"
        f"Dataset with lowest river proximity: \n"
        f"   {min(results['dataset_stats'].items(), key=lambda x: x[1]['percentage_near_rivers'])[0].replace('_', ' ').title()} "
        f"({min([stats['percentage_near_rivers'] for stats in results['dataset_stats'].values()]):.1f}%)"
    )
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

def create_interactive_map(sites_gdf, rivers, river_buffer):
    """Create an interactive map showing sites and river buffers."""
    print("Creating interactive map...")
    
    if sites_gdf.empty:
        print("No site data available for mapping")
        return None
    
    # Calculate map center from sites
    center_lat = sites_gdf['latitude'].mean()
    center_lon = sites_gdf['longitude'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Add basemap options
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    
    # Create feature groups
    near_river_group = folium.FeatureGroup(name=f'Sites within {BUFFER_DISTANCE_KM}km of rivers')
    far_river_group = folium.FeatureGroup(name=f'Sites beyond {BUFFER_DISTANCE_KM}km from rivers')
    river_group = folium.FeatureGroup(name='Rivers')
    buffer_group = folium.FeatureGroup(name=f'{BUFFER_DISTANCE_KM}km River Buffers')
    
    # Add rivers
    if rivers:
        for river_coords in rivers:
            if len(river_coords) >= 2:
                folium.PolyLine(
                    locations=river_coords,
                    color='blue',
                    weight=2,
                    opacity=0.7
                ).add_to(river_group)
    
    # Add river buffer
    if river_buffer:
        # For simplicity, we'll just add the buffer as a single polygon
        # In reality, we'd need to handle multipolygons and complex shapes
        if hasattr(river_buffer, 'geoms'):  # MultiPolygon
            for geom in river_buffer.geoms:
                if hasattr(geom, 'exterior'):
                    coords = [(lat, lon) for lon, lat in geom.exterior.coords]
                    folium.Polygon(
                        locations=coords,
                        color='#66c2a5',
                        fill=True,
                        fill_color='#66c2a5',
                        fill_opacity=0.2,
                        popup=f'{BUFFER_DISTANCE_KM}km Buffer Zone'
                    ).add_to(buffer_group)
        elif hasattr(river_buffer, 'exterior'):  # Single Polygon
            coords = [(lat, lon) for lon, lat in river_buffer.exterior.coords]
            folium.Polygon(
                locations=coords,
                color='#66c2a5',
                fill=True,
                fill_color='#66c2a5',
                fill_opacity=0.2,
                popup=f'{BUFFER_DISTANCE_KM}km Buffer Zone'
            ).add_to(buffer_group)
    
    # Add sites (create a sample if too many for performance)
    display_df = sites_gdf
    if len(sites_gdf) > 1000:
        # Sample for better performance
        display_df = sites_gdf.sample(n=1000, random_state=42)
        print(f"Sampling {len(display_df)} sites from {len(sites_gdf)} for map display")
    
    # Process by dataset for better visualization
    for dataset in display_df['dataset'].unique():
        dataset_df = display_df[display_df['dataset'] == dataset]
        
        # Customize colors by dataset
        dataset_colors = {
            'mound_villages': 'red',
            'casarabe_sites': 'blue',
            'amazon_geoglyphs': 'green',
            'survey_data': 'purple',
            'science_data': 'orange'
        }
        color = dataset_colors.get(dataset, 'gray')
        
        for idx, site in dataset_df.iterrows():
            popup_text = (
                f"<b>{site['source']}</b><br>"
                f"Coordinates: {site['latitude']:.4f}, {site['longitude']:.4f}<br>"
                f"Near River: {'Yes' if site['near_river'] else 'No'}"
            )
            
            marker = folium.CircleMarker(
                location=[site['latitude'], site['longitude']],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=popup_text
            )
            
            if site['near_river']:
                marker.add_to(near_river_group)
            else:
                marker.add_to(far_river_group)
    
    # Add groups to map
    buffer_group.add_to(m)
    river_group.add_to(m)
    near_river_group.add_to(m)
    far_river_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add plugins for better interaction
    plugins.MiniMap().add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(m)
    
    # Add a title
    title_html = '''
         <h3 align="center" style="font-size:16px"><b>Archaeological Sites & River Proximity</b></h3>
         <h4 align="center" style="font-size:13px">Showing sites within and beyond {0}km of rivers</h4>
         <h4 align="center" style="font-size:12px">Overall: {1:.1f}% of sites are within river buffer zones</h4>
      '''.format(BUFFER_DISTANCE_KM, sites_gdf['near_river'].mean() * 100)
    m.get_root().html.add_child(folium.Element(title_html))
    
    print("Interactive map created")
    return m

# --- MAIN FUNCTION ---

def main():
    """Run the complete analysis pipeline."""
    print("=== ARCHAEOLOGICAL SITES & RIVER PROXIMITY ANALYSIS ===")
    print(f"Buffer distance: {BUFFER_DISTANCE_KM}km ({BUFFER_DISTANCE_MILES:.1f} miles)\n")
    
    # 1. Load archaeological site data
    sites_df, dataset_counts = load_archaeological_sites()
    if sites_df.empty:
        print("No archaeological sites data available. Analysis cannot continue.")
        return
    
    # 2. Load rivers data
    rivers, river_names = load_river_data()
    if not rivers:
        print("No river data available. Analysis cannot continue.")
        return
    
    # 3. Create river buffers
    river_buffer = create_river_buffer(rivers, BUFFER_DISTANCE_KM)
    if river_buffer is None:
        print("Failed to create river buffers. Analysis cannot continue.")
        return
    
    # 4. Analyze site proximity to rivers
    sites_gdf, results = analyze_site_proximity(sites_df, river_buffer)
    if results is None:
        print("Proximity analysis failed. Visualization cannot continue.")
        return
    
    # 5. Create visualizations
    print("\nGenerating visualizations...")
    
    # Create and display summary charts
    fig = create_summary_visualizations(results)
    plt.show()  # Will display inline in Kaggle notebook
    
    # Create and display interactive map
    map_obj = create_interactive_map(sites_gdf, rivers, river_buffer)
    if map_obj:
        display(map_obj)  # Will display inline in Kaggle notebook
    
    # 6. Summary conclusion
    print("\n=== ANALYSIS CONCLUSION ===")
    print(f"We analyzed {results['total_sites']} archaeological sites across the Amazon region.")
    print(f"Our findings show that {results['percentage_near_rivers']:.1f}% of sites are within {BUFFER_DISTANCE_KM}km of rivers,")
    print(f"supporting the hypothesis that water access was a critical factor in settlement location.")
    print(f"The remaining {100-results['percentage_near_rivers']:.1f}% of sites suggest other factors may have influenced settlement patterns.")

# Run the analysis if this script is executed directly
if __name__ == "__main__":
    main()

