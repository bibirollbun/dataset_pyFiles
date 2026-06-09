!pip install contextily


import os
from openai import OpenAI
from kaggle_secrets import UserSecretsClient
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import contextily as ctx

def load_secret(name):
    """Loads secret from Colab/Kaggle."""

    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        except Exception:
            pass 
    else:
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception: 
            pass

    return 'Secret not found'


openai_key = 'sk-proj-fj0ymqVIUdhwJLu3o4dXCtpDHodg9wAop6Kgh3XF6ihv3GcxpzyYqS06AA'

client = OpenAI(
  api_key=openai_key
)

prompt = "Provide some of the archaelogical site the latitute and longtitude (by the number) of Amazon Rainforest"

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message.content);


openai_key = 'sk-proj-fj0ymq'

client = OpenAI(
  api_key=openai_key
)

prompt = "Search database to find any data information for amazon like lidar, geoglyphs and provide the links "

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message.content);


import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer
from shapely.geometry import Point, box
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# --- 1. Sites dictionary ---
sites_dict = {
    "MarajÃ³ Island": (-0.5, -50.5),
    "Geoglyphs of Acre": (-8.0, -68.0),
    "CayalÃ³n": (-8.5, -75.0),
    "Pano Culture Sites": (-9.0, -68.0),
    "Ankau": (-8.1, -74.0),
    "Sierra de la Capaceta": (-10.0, -74.5)
}

# --- 2. State code map ---
state_code_map = {
    "Acre": "AC", "Alagoas": "AL", "AmapÃ¡": "AP", "Amazonas": "AM", "Bahia": "BA", "CearÃ¡": "CE",
    "Distrito Federal": "DF", "EspÃ­rito Santo": "ES", "GoiÃ¡s": "GO", "MaranhÃ£o": "MA", "Mato Grosso": "MT",
    "Mato Grosso do Sul": "MS", "Minas Gerais": "MG", "ParÃ¡": "PA", "ParaÃ­ba": "PB", "ParanÃ¡": "PR",
    "Pernambuco": "PE", "PiauÃ­": "PI", "Rio de Janeiro": "RJ", "Rio Grande do Norte": "RN",
    "Rio Grande do Sul": "RS", "RondÃ´nia": "RO", "Roraima": "RR", "Santa Catarina": "SC",
    "SÃ£o Paulo": "SP", "Sergipe": "SE", "Tocantins": "TO"
}

# --- 3. Coordinate transformation and buffer creation ---
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
buffer_radius = 100000  # 100 km

xs, ys, labels, buffers = [], [], [], []
for name, (lat, lon) in sites_dict.items():
    x, y = transformer.transform(lon, lat)
    xs.append(x)
    ys.append(y)
    labels.append(name)
    point = Point(x, y)
    buffers.append(point.buffer(buffer_radius))

min_x, max_x = min(xs) - 1e6, max(xs) + 1e6
min_y, max_y = min(ys) - 1e6, max(ys) + 1e6
map_bounds = box(min_x, min_y, max_x, max_y)

# --- 4. Brazil state boundaries ---
print("Loading Brazil state boundaries...")
gdf_states = gpd.read_file("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson")
gdf_states = gdf_states.to_crs(epsg=3857)
gdf_states["state_code"] = gdf_states["name"].map(state_code_map)
gdf_states_clipped = gpd.clip(gdf_states, map_bounds)

# --- 5. Load geoglyphs from Excel ---
print("Loading geoglyphs from Excel...")
try:
    df_geoglyphs = pd.read_csv("/kaggle/input/amazon-geoglyphs/amazon_geoglyphs.csv")
    df_geoglyphs = df_geoglyphs.dropna(subset=['lat', 'lon'])
    
    # Create Point geometries
    geometry = [Point(xy) for xy in zip(df_geoglyphs['lon'], df_geoglyphs['lat'])]
    gdf_geoglyphs = gpd.GeoDataFrame(df_geoglyphs, geometry=geometry, crs="EPSG:4326")
    gdf_geoglyphs = gdf_geoglyphs.to_crs(epsg=3857)

    print(f"Loaded {len(gdf_geoglyphs)} geoglyphs from Excel")
except Exception as e:
    print(f"Error reading Excel file: {e}")
    gdf_geoglyphs = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:3857")

# --- 6. Plotting ---
print("Creating map...")
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)

# Add basemap
try:
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
except Exception as e:
    print(f"Warning: Could not load basemap: {e}")

# State borders
gdf_states_clipped.boundary.plot(ax=ax, linewidth=1, color="black", zorder=3)

# State labels
for _, row in gdf_states_clipped.iterrows():
    if pd.notnull(row["state_code"]):
        try:
            x, y = row["geometry"].centroid.coords[0]
            ax.text(x, y, row["state_code"], fontsize=9, color='black', ha='center', weight='bold', zorder=6)
        except Exception:
            continue

# Buffers
from matplotlib.patches import Polygon as MplPolygon
for buffer in buffers:
    try:
        x_buffer, y_buffer = buffer.exterior.xy
        polygon = MplPolygon(list(zip(x_buffer, y_buffer)), closed=True,
                             edgecolor='darkblue', facecolor='none', linewidth=2, alpha=0.7, zorder=4)
        ax.add_patch(polygon)
    except Exception:
        continue

# Site markers
ax.scatter(xs, ys, color='darkblue', s=100, edgecolor='white', linewidth=1, zorder=5)

# Site labels
for x, y, label in zip(xs, ys, labels):
    ax.text(x + 50000, y + 50000, label, fontsize=10, color='darkblue', weight='bold', zorder=6)

# Geoglyphs
if len(gdf_geoglyphs) > 0:
    try:
        gdf_geoglyphs.plot(ax=ax, color='crimson', markersize=30, zorder=7, alpha=0.8)
        print(f"Plotted {len(gdf_geoglyphs)} geoglyphs")
    except Exception as e:
        print(f"Error plotting geoglyphs: {e}")
else:
    print("No valid geoglyphs to plot")

# Final touches
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Amazon Sites, Geoglyphs (Excel), and State Boundaries")

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', markersize=10, label='Archaeological Sites'),
    Line2D([0], [0], color='darkblue', linewidth=2, label='100km Buffer Zones'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', markersize=8, label='Geoglyphs'),
    Line2D([0], [0], color='black', linewidth=1, label='State Boundaries')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
# Save the figure to a file before showing it
plt.savefig("amazon_sites_geoglyphs_map.png", dpi=300)

plt.show()
print("Map completed successfully!")



import pandas as pd
from geopy.distance import geodesic
import numpy as np

# Sites dictionary
sites_dict = {
    "MarajÃ³ Island": (-0.5, -50.5),
    "Geoglyphs of Acre": (-8.0, -68.0),
    "CayalÃ³n": (-8.5, -75.0),
    "Pano Culture Sites": (-9.0, -68.0),
    "Ankau": (-8.1, -74.0),
    "Sierra de la Capaceta": (-10.0, -74.5)
}

# Create a distance matrix
site_names = list(sites_dict.keys())
distance_matrix = pd.DataFrame(index=site_names, columns=site_names)

# Calculate distances
for site1 in site_names:
    for site2 in site_names:
        coord1 = sites_dict[site1]
        coord2 = sites_dict[site2]
        distance_km = geodesic(coord1, coord2).kilometers
        if distance_km <= 500:
            distance_matrix.loc[site1, site2] = round(distance_km, 2)
        else:
            distance_matrix.loc[site1, site2] = np.nan  # or "" if preferred

print(distance_matrix)



client = OpenAI(
  api_key=openai_key
)

prompt = """
Here is a distance matrix between different archaeological sites:

                     MarajÃ³ Island  Geoglyphs of Acre  CayalÃ³n  Pano Culture Sites  Ankau  Sierra de la Capaceta
MarajÃ³ Island               0.0                NaN       NaN               NaN    NaN                    NaN
Geoglyphs of Acre          NaN                0.0       NaN             110.6    NaN                    NaN
CayalÃ³n                     NaN                NaN       0.0               NaN  118.71                 174.76
Pano Culture Sites          NaN              110.6       NaN               0.0    NaN                    NaN
Ankau                       NaN                NaN   118.71               NaN    0.0                  217.21
Sierra de la Capaceta       NaN                NaN   174.76               NaN  217.21                    0.0

Question: Based on these distances, can you provide any insights or hypotheses about the relationships or connectivity between these archaeological sites? Also suggest if water bodies affect the distance between the sites?
"""

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message.content)


client = OpenAI(
  api_key=openai_key
)

prompt = "suggest some method to understand spatial relationship between archaleogical site to the nearest water body"

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message.content);


import ee
import pandas as pd
from geopy.distance import geodesic
import time

ee.Authenticate()
ee.Initialize(project='ee-molly0124')

# 1. Define sites
sites_dict = {
    "MarajÃ³ Island": (-0.5, -50.5),
    "Geoglyphs of Acre": (-8.0, -68.0),
    "CayalÃ³n": (-8.5, -75.0),
    "Pano Culture Sites": (-9.0, -68.0),
    "Ankau": (-8.1, -74.0),
    "Sierra de la Capaceta": (-10.0, -74.5)
}

def find_nearest_water_distance(site_name, lat, lon):
    """Find distance to nearest water for a single site using sampling approach"""
    
    print(f"Processing {site_name}...")
    
    # Create multiple distance rings around the site
    site_point = ee.Geometry.Point([lon, lat])
    
    # Check at different distances: 1km, 5km, 10km, 25km, 50km, 100km, 200km
    search_distances = [1000, 5000, 10000, 25000, 50000, 100000, 200000]
    
    jrc = ee.Image("JRC/GSW1_3/GlobalSurfaceWater").select('occurrence')
    water_permanent = jrc.gte(75)  # Water present â‰¥75% of the time
    
    for search_dist in search_distances:
        print(f"  Searching within {search_dist/1000:.0f} km...")
        
        # Create search area
        search_area = site_point.buffer(search_dist)
        
        # Sample water pixels in this area
        try:
            # Use a coarser scale for sampling
            sample_scale = max(90, search_dist // 100)  # Scale based on distance
            
            # Sample water pixels
            water_sample = water_permanent.sample(
                region=search_area,
                scale=sample_scale,
                numPixels=1000,  # Limit sample size
                geometries=True
            )
            
            # Get sample info
            sample_info = water_sample.getInfo()
            
            if not sample_info['features']:
                continue  # No water found at this distance, try next
            
            # Find water pixels (occurrence > 0)
            water_pixels = [f for f in sample_info['features'] if f['properties']['occurrence'] > 0]
            
            if not water_pixels:
                continue  # No water pixels found
            
            print(f"  Found {len(water_pixels)} water pixels")
            
            # Calculate distances to all water pixels
            min_distance = float('inf')
            site_coords = (lat, lon)
            
            for pixel in water_pixels:
                pixel_coords = pixel['geometry']['coordinates']
                pixel_point = (pixel_coords[1], pixel_coords[0])  # lat, lon
                distance = geodesic(site_coords, pixel_point).kilometers
                min_distance = min(min_distance, distance)
            
            print(f"  Minimum distance found: {min_distance:.2f} km")
            return min_distance
            
        except Exception as e:
            print(f"  Error at {search_dist/1000:.0f} km: {e}")
            continue
    
    print(f"  No water found within {max(search_distances)/1000:.0f} km")
    return None

def find_nearest_water_alternative(site_name, lat, lon):
    """Alternative method using distance transform"""
    
    print(f"Processing {site_name} with distance transform method...")
    
    try:
        # Load water data
        jrc = ee.Image("JRC/GSW1_3/GlobalSurfaceWater").select('occurrence')
        water_mask = jrc.gte(50)  # More lenient threshold
        
        # Create region around site
        site_point = ee.Geometry.Point([lon, lat])
        region = site_point.buffer(10)  # 10km buffer
        
        # Calculate distance to water using Earth Engine's distance function
        # This is more efficient than downloading vectors
        water_distance = water_mask.Not().cumulativeCost(
            source=water_mask,
            maxDistance=100000  # Max 100km
        )
        
        # Sample the distance at the site location
        distance_at_site = water_distance.sample(
            region=site_point,
            scale=90,
            numPixels=1
        ).first()
        
        # Get the distance value
        distance_info = distance_at_site.getInfo()
        
        if distance_info and distance_info['properties']:
            # Convert from meters to kilometers
            distance_m = distance_info['properties'].get('occurrence', None)
            if distance_m is not None:
                distance_km = distance_m / 1000.0
                print(f"  Distance to water: {distance_km:.2f} km")
                return distance_km
        
        return None
        
    except Exception as e:
        print(f"  Error with distance transform: {e}")
        return None

# Process each site
results = []

for site_name, (lat, lon) in sites_dict.items():
    print(f"\n{'='*50}")
    print(f"PROCESSING: {site_name}")
    print(f"{'='*50}")
    
    # Try the sampling method first
    distance = find_nearest_water_distance(site_name, lat, lon)
    
    # If sampling fails, try the distance transform method
    if distance is None:
        print("Sampling method failed, trying distance transform...")
        distance = find_nearest_water_alternative(site_name, lat, lon)
    
    # If both methods fail, try a very simple approach
    if distance is None:
        print("Trying simple grid search...")
        try:
            # Create a simple grid search
            site_point = ee.Geometry.Point([lon, lat])
            jrc = ee.Image("JRC/GSW1_3/GlobalSurfaceWater").select('occurrence')
            water_mask = jrc.gte(25)  # Very lenient threshold
            
            # Check at specific distances
            for test_dist in [1, 5, 10, 25, 50, 100]:
                test_region = site_point.buffer(test_dist * 1000)
                
                # Check if any water exists in this region
                water_in_region = water_mask.reduceRegion(
                    reducer=ee.Reducer.max(),
                    geometry=test_region,
                    scale=300,  # Coarse scale
                    maxPixels=1e8
                )
                
                water_exists = water_in_region.getInfo().get('occurrence', 0)
                
                if water_exists > 0:
                    distance = test_dist
                    print(f"  Water found within {test_dist} km")
                    break
            
            if distance is None:
                distance = 100  # Default if no water found within 100km
                print(f"  No water found within 100 km, using default")
                
        except Exception as e:
            print(f"  All methods failed: {e}")
            distance = None
    
    results.append({
        'Site': site_name,
        'Lat': lat,
        'Lon': lon,
        'MinDistToWater_km': distance
    })
    
    # Delay between sites to avoid rate limiting
    time.sleep(2)

# Create DataFrame and display results
df_distances = pd.DataFrame(results)
print("\n" + "="*60)
print("FINAL RESULTS:")
print("="*60)
print(df_distances.to_string(index=False, float_format='%.2f'))

# Save to CSV
df_distances.to_csv('site_water_distances.csv', index=False)
print(f"\nResults saved to 'site_water_distances.csv'")


client = OpenAI(
  api_key=openai_key
)

prompt = """
Here is the minimum distance of water for each sites

   Site    Lat    Lon  MinDistToWater_km
        MarajÃ³ Island  -0.50 -50.50               1.99
    Geoglyphs of Acre  -8.00 -68.00              77.40
              CayalÃ³n  -8.50 -75.00              15.64
   Pano Culture Sites  -9.00 -68.00               0.16
                Ankau  -8.10 -74.00              45.01
Sierra de la Capaceta -10.00 -74.50              45.82

Question: Based on these distances, can you provide any insights or hypotheses about the relationships? Also provide analysis on how water bodies play the part of the archaelogical sites?
"""

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message.content)


import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- CONFIG ---
data_dir = "/kaggle/input/amzbr-lidar/"  # Folder with files
excluded_columns = ['patch', 'time', 'year', 'id', 'age']  # Columns to ignore
desired_extension = "pss"

sites_dict = {
    "MarajÃ³ Island": (-0.5, -50.5),
    "Geoglyphs of Acre": (-8.0, -68.0),
    "CayalÃ³n": (-8.5, -75.0),
    "Pano Culture Sites": (-9.0, -68.0),
    "Ankau": (-8.1, -74.0),
    "Sierra de la Capaceta": (-10.0, -74.5)
}

# --- HELPERS ---

def format_latlon(val):
    return f"{val:.1f}" if val % 1 else f"{int(val)}.0"

def extract_lat_lon_from_filename(filename):
    match = re.search(r"lat(-?\d+(?:\.\d+)?)lon(-?\d+(?:\.\d+)?)", filename)
    return (float(match.group(1)), float(match.group(2))) if match else (None, None)

def find_matching_file(data_dir, lat, lon, ext="pss", lat_tolerance=5, lon_tolerance=5):
    lat_str = format_latlon(lat)
    lon_str = format_latlon(lon)
    exact_pattern = re.compile(rf"amzbr_\d{{4}}\.lat{re.escape(lat_str)}lon{re.escape(lon_str)}\.{ext}$")

    closest_file = None
    min_lat_diff = float('inf')
    min_lon_diff = float('inf')

    for fname in os.listdir(data_dir):
        if not fname.endswith(f".{ext}"):
            continue
        full_path = os.path.join(data_dir, fname)

        if exact_pattern.match(fname):
            return full_path

        file_lat, file_lon = extract_lat_lon_from_filename(fname)
        if file_lat is None or file_lon is None:
            continue

        lat_diff = abs(file_lat - lat)
        lon_diff = abs(file_lon - lon)

        if lat_diff <= lat_tolerance and lon_diff <= lon_tolerance:
            if (lat_diff + lon_diff) < (min_lat_diff + min_lon_diff):
                min_lat_diff = lat_diff
                min_lon_diff = lon_diff
                closest_file = full_path

    if closest_file:
        print(f"  âš ï¸� No exact match. Using closest file within tolerance: {closest_file}")
        return closest_file

    print(f"  âš ï¸� No file found within lat_tol={lat_tolerance} and lon_tol={lon_tolerance}")
    return None

def read_pss_file_with_header(file_path, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []

    try:
        df = pd.read_csv(file_path, delim_whitespace=True, comment="#")
        print(f"  ğŸ“‹ Columns in file: {df.columns.tolist()}")

        df_numeric = df.select_dtypes(include='number')
        df_numeric = df_numeric.drop(columns=[col for col in exclude_columns if col in df_numeric.columns], errors='ignore')

        if df_numeric.empty:
            print("  âš ï¸� No usable numeric data after filtering.")
            return None

        print(f"  âœ… Using columns: {df_numeric.columns.tolist()}")
        return df_numeric
    except Exception as e:
        print(f"  â�Œ Error reading file: {e}")
        return None

def preprocess_with_pca(df, site_name=None, lat=None, lon=None, n_components=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['Site'] = site_name
    pca_df['Latitude'] = lat
    pca_df['Longitude'] = lon

    print(f"  Explained variance ratio by PCA components: {pca.explained_variance_ratio_}")

    return pca_df, pca

# --- MAIN WORKFLOW ---

all_sites_pca = []

for site_name, (lat, lon) in sites_dict.items():
    print(f"\nğŸ”� Processing site: {site_name} (lat={lat}, lon={lon})")
    file_path = find_matching_file(data_dir, lat, lon, ext=desired_extension)

    if not file_path:
        continue

    df = read_pss_file_with_header(file_path, exclude_columns=excluded_columns)
    if df is None:
        continue

    pca_df, pca_model = preprocess_with_pca(df, site_name=site_name, lat=lat, lon=lon)
    all_sites_pca.append(pca_df)

# Combine all sites PCA results
if all_sites_pca:
    combined_pca_df = pd.concat(all_sites_pca, ignore_index=True)

    # Plot PCA results, colored by site
    plt.figure(figsize=(10, 8))
    for site_name in combined_pca_df['Site'].unique():
        subset = combined_pca_df[combined_pca_df['Site'] == site_name]
        plt.scatter(subset['PC1'], subset['PC2'], label=site_name, alpha=0.7, edgecolors='w', s=80)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Soil Properties by Archaeological Site')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No PCA results to plot.")

print("\nâœ… Finished processing all sites.")



import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap

def reduce_with_tsne_or_umap(df, method='umap', site_name=None, lat=None, lon=None):
    """
    Reduce data to 2D using t-SNE or UMAP and return a DataFrame with coordinates and metadata.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    embedding = reducer.fit_transform(X_scaled)

    embed_df = pd.DataFrame(embedding, columns=['Dim1', 'Dim2'])
    embed_df['Site'] = site_name
    embed_df['Latitude'] = lat
    embed_df['Longitude'] = lon

    return embed_df

all_sites_results = []

for site_name, (lat, lon) in sites_dict.items():
    print(f"\nğŸ”� Processing site: {site_name} (lat={lat}, lon={lon})")
    file_path = find_matching_file(data_dir, lat, lon, ext=desired_extension)

    if not file_path:
        continue

    df = read_pss_file_with_header(file_path, exclude_columns=excluded_columns)
    if df is None:
        continue

    reduced_df = reduce_with_tsne_or_umap(df, method='umap', site_name=site_name, lat=lat, lon=lon)
    all_sites_results.append(reduced_df)

print("\nâœ… Dimensionality reduction completed for all sites.")

# Combine all results
combined_df = pd.concat(all_sites_results, ignore_index=True)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(data=combined_df, x='Dim1', y='Dim2', hue='Site', palette='tab10', s=40)
plt.title("UMAP Projection of Site Data")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title='Site')
plt.grid(True)
plt.show()



import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd

# Define your site coordinates (lon, lat)
sites_dict = {
    "MarajÃ³ Island": (-0.5, -50.5),
    "Geoglyphs of Acre": (-8.0, -68.0),
    "CayalÃ³n": (-8.5, -75.0),
    "Pano Culture Sites": (-9.0, -68.0),
    "Ankau": (-8.1, -74.0),
    "Sierra de la Capaceta": (-10.0, -74.5)
}

# Open raster
tif_path = "/kaggle/input/amazondem/SA_srtm_mosaic_30arcsec_reg_hgt.tif"
with rasterio.open(tif_path) as src:
    raster_crs = src.crs
    elevation = src.read(1, masked=True)

    # Create GeoDataFrame in EPSG:4326 and reproject to raster CRS
    gdf = gpd.GeoDataFrame({
        'Site': list(sites_dict.keys()),
        'geometry': [Point(xy[::-1]) for xy in sites_dict.values()]
    }, crs="EPSG:4326").to_crs(raster_crs)

    # Plot elevation with colorbar and site points
    fig, ax = plt.subplots(figsize=(10, 8))

    # Show the elevation with imshow to get colorbar
    img = show(src, ax=ax, cmap='terrain', title='Elevation with Site Locations')

    # Add colorbar manually
    cbar = plt.colorbar(img.get_images()[0], ax=ax, shrink=0.7)
    cbar.set_label("Elevation (m)")

    # Plot site locations
    gdf.plot(ax=ax, color='red', markersize=50, edgecolor='black')

    # Optional: Add labels
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf['Site']):
        ax.text(x, y, label, fontsize=9, ha='right')

    plt.show()




import os
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

# Add UMAP import
import umap

# --- Helper functions ---

def format_latlon(value):
    return f"{value:+06.2f}".replace('.', '_')

def extract_lat_lon_from_filename(fname):
    lat_match = re.search(r"lat([-+]?\d+\.\d+)", fname)
    lon_match = re.search(r"lon([-+]?\d+\.\d+)", fname)
    if lat_match and lon_match:
        try:
            lat = float(lat_match.group(1))
            lon = float(lon_match.group(1))
            return lat, lon
        except ValueError:
            return None, None
    return None, None

def read_pss_file_with_header(file_path, exclude_columns=None, sample_frac=0.5):
    """Modified to sample data for faster processing"""
    if exclude_columns is None:
        exclude_columns = []
    try:
        df = pd.read_csv(file_path, delim_whitespace=True, comment="#")
        
        # Sample data if file is large
        if len(df) > 100:
            df = df.sample(frac=sample_frac, random_state=42)
        
        df_numeric = df.select_dtypes(include='number')
        df_numeric = df_numeric.drop(columns=[col for col in exclude_columns if col in df_numeric.columns], errors='ignore')
        
        if df_numeric.empty:
            return None
        return df_numeric
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def preprocess_features(df, variance_threshold=0.01, max_features=50, common_features=None):
    """Remove low-variance features and limit feature count"""
    if common_features is not None:
        # Use predefined common features
        available_features = [col for col in common_features if col in df.columns]
        df_filtered = df[available_features]
    else:
        # Remove low-variance features
        selector = VarianceThreshold(threshold=variance_threshold)
        df_filtered = pd.DataFrame(
            selector.fit_transform(df),
            columns=df.columns[selector.get_support()]
        )
        
        # Limit number of features if still too many
        if len(df_filtered.columns) > max_features:
            # Keep features with highest variance
            variances = df_filtered.var().sort_values(ascending=False)
            top_features = variances.head(max_features).index
            df_filtered = df_filtered[top_features]
    
    return df_filtered

def compute_file_statistics(soil_df):
    """Compute summary statistics instead of using all data points"""
    stats = []
    for col in soil_df.columns:
        stats.extend([
            soil_df[col].mean(),
            soil_df[col].std(),
            soil_df[col].median(),
            soil_df[col].quantile(0.25),
            soil_df[col].quantile(0.75)
        ])
    return np.array(stats)

# --- Optimized Parameters ---
data_dir = "/kaggle/input/amzbr-lidar"  
desired_extension = "pss"
excluded_columns = []
min_points_per_file = 5
sample_fraction = 0.5  
use_statistics = True  # Use summary statistics instead of all points
max_files = 1000  # Limit number of files processed

# --- Step 1: Find common features across all files ---

print("ğŸ“‚ Finding common features across files...")

# First pass: find common features
all_columns = set()
valid_files = []

for fname in os.listdir(data_dir):
    if not fname.endswith(f".{desired_extension}"):
        continue
        
    full_path = os.path.join(data_dir, fname)
    soil_df = read_pss_file_with_header(full_path, exclude_columns=excluded_columns, 
                                      sample_frac=sample_fraction)
    
    if soil_df is None or len(soil_df) < min_points_per_file:
        continue
        
    lat, lon = extract_lat_lon_from_filename(fname)
    if lat is None or lon is None:
        continue
    
    all_columns.update(soil_df.columns)
    valid_files.append((fname, full_path, lat, lon))
    
    if len(valid_files) >= max_files:
        break

if not valid_files:
    raise RuntimeError("No valid soil data files found.")

# Find most common features (present in at least 50% of files)
print(f"Found {len(valid_files)} valid files")
print("Finding common features...")

feature_counts = {}
for fname, full_path, lat, lon in valid_files[:min(100, len(valid_files))]:  # Sample for feature detection
    soil_df = read_pss_file_with_header(full_path, exclude_columns=excluded_columns, 
                                      sample_frac=sample_fraction)
    if soil_df is not None:
        for col in soil_df.columns:
            feature_counts[col] = feature_counts.get(col, 0) + 1

min_occurrence = max(1, len(valid_files) * 0.3)  # Present in at least 30% of files
common_features = [col for col, count in feature_counts.items() if count >= min_occurrence]
common_features = sorted(common_features)[:50]  # Limit to top 50 features

print(f"Using {len(common_features)} common features")

# --- Step 2: Load and preprocess data with common features ---

file_features = []
all_files_meta = []

print("Loading and processing files with common features...")

for fname, full_path, lat, lon in valid_files:
    soil_df = read_pss_file_with_header(full_path, exclude_columns=excluded_columns, 
                                      sample_frac=sample_fraction)
    
    if soil_df is None:
        continue

    # Use only common features
    available_features = [col for col in common_features if col in soil_df.columns]
    if len(available_features) < 3:  # Need at least 3 features
        continue
        
    soil_df_filtered = soil_df[available_features]
    
    if use_statistics:
        # Compute statistics for available features, pad with zeros for missing ones
        stats_dict = {}
        for col in common_features:
            if col in soil_df_filtered.columns:
                col_data = soil_df_filtered[col]
                stats_dict[f"{col}_mean"] = col_data.mean()
                stats_dict[f"{col}_std"] = col_data.std()
                stats_dict[f"{col}_median"] = col_data.median()
                stats_dict[f"{col}_q25"] = col_data.quantile(0.25)
                stats_dict[f"{col}_q75"] = col_data.quantile(0.75)
            else:
                # Fill missing features with zeros
                stats_dict[f"{col}_mean"] = 0.0
                stats_dict[f"{col}_std"] = 0.0
                stats_dict[f"{col}_median"] = 0.0
                stats_dict[f"{col}_q25"] = 0.0
                stats_dict[f"{col}_q75"] = 0.0
        
        file_feature_vector = np.array(list(stats_dict.values()))
    else:
        # Use mean values, pad with zeros for missing features
        mean_dict = {}
        for col in common_features:
            if col in soil_df_filtered.columns:
                mean_dict[col] = soil_df_filtered[col].mean()
            else:
                mean_dict[col] = 0.0
        file_feature_vector = np.array(list(mean_dict.values()))
    
    file_features.append(file_feature_vector)
    all_files_meta.append({
        "filename": fname,
        "full_path": full_path,
        "Latitude": lat,
        "Longitude": lon
    })
    
    if len(file_features) % 100 == 0:
        print(f"Processed {len(file_features)} files...")

if not file_features:
    raise RuntimeError("No valid soil data files found.")

print(f"Total files processed: {len(file_features)}")

# --- Step 3: Dimensionality reduction with UMAP ---

# Convert to numpy array
X = np.array(file_features)

# Handle NaN values
X = np.nan_to_num(X, nan=0.0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use UMAP for dimensionality reduction
umap_reducer = umap.UMAP(n_components=5, random_state=42, n_neighbors=15, min_dist=0.1)
embedding = umap_reducer.fit_transform(X_scaled)

print(f"Dimensionality reduced to {embedding.shape[1]} components")

# --- Step 4: Clustering with K-means ---

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embedding)

# --- Step 5: Report results ---

print(f"\nğŸ—‚ï¸� Soil pattern clusters found (using {embedding.shape[1]}D UMAP embedding):\n")

for cluster_id in range(num_clusters):
    cluster_mask = cluster_labels == cluster_id
    cluster_files = np.where(cluster_mask)[0]
    
    print(f"Cluster {cluster_id}: {len(cluster_files)} files")
    
    # Find files closest to cluster center
    cluster_center = kmeans.cluster_centers_[cluster_id]
    cluster_points = embedding[cluster_mask]
    
    if len(cluster_points) > 0:
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        closest_indices = np.argsort(distances)[:min(5, len(distances))]
        
        print("Closest files to cluster center:")
        for idx in closest_indices:
            file_idx = cluster_files[idx]
            meta = all_files_meta[file_idx]
            print(f" - {meta['filename']} (Lat: {meta['Latitude']:.2f}, Lon: {meta['Longitude']:.2f})")
    print()

# --- Step 6: Site matching ---

sites_dict = {
    "MarajÃ³ Island": (-0.5, -50.5),
    "Geoglyphs of Acre": (-8.0, -68.0),
    "CayalÃ³n": (-8.5, -75.0),
    "Pano Culture Sites": (-9.0, -68.0),
    "Ankau": (-8.1, -74.0),
    "Sierra de la Capaceta": (-10.0, -74.5)
}

print("\nğŸ”� Finding clusters closest to archaeological sites:\n")

for site_name, (site_lat, site_lon) in sites_dict.items():
    # Find closest file by geographic distance
    dists = []
    for meta in all_files_meta:
        lat_lon_dist = np.sqrt((meta['Latitude'] - site_lat)**2 + (meta['Longitude'] - site_lon)**2)
        dists.append(lat_lon_dist)
    
    closest_file_idx = np.argmin(dists)
    cluster_id = cluster_labels[closest_file_idx]
    
    print(f"Site '{site_name}' â†’ Cluster {cluster_id}")
    
    # Show a few representative files from this cluster
    cluster_files = np.where(cluster_labels == cluster_id)[0][:5]
    for file_idx in cluster_files:
        meta = all_files_meta[file_idx]
        print(f"  - {meta['filename']} (Lat: {meta['Latitude']:.2f}, Lon: {meta['Longitude']:.2f})")
    print()




client = OpenAI(
  api_key=openai_key
)

prompt = """
Cluster 0: 40 files
Closest files to cluster center:
 - amzbr_0005.lat-12.5lon-61.5.pss (Lat: -12.50, Lon: -61.50)
 - amzbr_0336.lat-11.5lon-59.5.pss (Lat: -11.50, Lon: -59.50)
 - amzbr_0084.lat-8.5lon-54.5.pss (Lat: -8.50, Lon: -54.50)
 - amzbr_0080.lat-8.5lon-58.5.pss (Lat: -8.50, Lon: -58.50)
 - amzbr_0341.lat-12.5lon-59.5.pss (Lat: -12.50, Lon: -59.50)

Cluster 1: 45 files
Closest files to cluster center:
 - amzbr_0095.lat-7.5lon-68.5.pss (Lat: -7.50, Lon: -68.50)
 - amzbr_0141.lat-5.5lon-71.5.pss (Lat: -5.50, Lon: -71.50)
 - amzbr_0118.lat-6.5lon-69.5.pss (Lat: -6.50, Lon: -69.50)
 - amzbr_0174.lat-4.5lon-62.5.pss (Lat: -4.50, Lon: -62.50)
 - amzbr_0097.lat-7.5lon-66.5.pss (Lat: -7.50, Lon: -66.50)

Cluster 2: 41 files
Closest files to cluster center:
 - amzbr_0131.lat-6.5lon-56.5.pss (Lat: -6.50, Lon: -56.50)
 - amzbr_0152.lat-5.5lon-60.5.pss (Lat: -5.50, Lon: -60.50)
 - amzbr_0228.lat-2.5lon-58.5.pss (Lat: -2.50, Lon: -58.50)
 - amzbr_0282.lat-0.5lon-54.5.pss (Lat: -0.50, Lon: -54.50)
 - amzbr_0227.lat-2.5lon-59.5.pss (Lat: -2.50, Lon: -59.50)

Cluster 3: 23 files
Closest files to cluster center:
 - amzbr_0047.lat-9.5lon-68.5.pss (Lat: -9.50, Lon: -68.50)
 - amzbr_0006.lat-12.5lon-60.5.pss (Lat: -12.50, Lon: -60.50)
 - amzbr_0021.lat-11.5lon-55.5.pss (Lat: -11.50, Lon: -55.50)
 - amzbr_0086.lat-8.5lon-52.5.pss (Lat: -8.50, Lon: -52.50)
 - amzbr_0231.lat-2.5lon-55.5.pss (Lat: -2.50, Lon: -55.50)

Cluster 4: 26 files
Closest files to cluster center:
 - amzbr_0182.lat-4.5lon-54.5.pss (Lat: -4.50, Lon: -54.50)
 - amzbr_0208.lat-3.5lon-53.5.pss (Lat: -3.50, Lon: -53.50)
 - amzbr_0184.lat-4.5lon-52.5.pss (Lat: -4.50, Lon: -52.50)
 - amzbr_0145.lat-5.5lon-67.5.pss (Lat: -5.50, Lon: -67.50)
 - amzbr_0020.lat-11.5lon-56.5.pss (Lat: -11.50, Lon: -56.50)

Question: Find add pattern in this soil structure for the cluster for archaelogical sites for amazon forest  
"""

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message.content)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider, CheckButtons
from typing import Dict, Optional, List, Tuple
import random

class ArchaeologicalProbabilityCalculator:
    def __init__(self):
        # Define cluster characteristics based on archaeological potential
        self.cluster_profiles = {
            0: {  # Cluster 0
                'name': 'Cluster 0',
                'base_probability': 0.25,
                'description': 'This cluster is located in the southern part of the Amazon, possibly indicative of historical human settlements near waterways or other resource-rich areas',
                'archaeological_indicators': ['resources', 'refuse deposits', 'agricultural areas'],
                'color': '#8B4513'
            },
            1: {  # Cluster 1
                'name': 'Cluster 1',
                'base_probability': 0.20,
                'description': 'This cluster shows a trend toward slightly more northern latitudes compared to Cluster 0, which may suggest a different type of archaeological activity, possibly linked to trade routes through the Andes or coastal regions.',
                'archaeological_indicators': ['habitation sites', 'burial grounds'],
                'color': '#F4A460'
            },
            2: {  # Cluster 2
                'name': 'Cluster 2',
                'base_probability': 0.30,
                'description': 'The inclusion of latitudes closer to the equator might suggest areas of more fertile lands or areas with historically significant populations.',
                'archaeological_indicators': ['fertile'],
                'color': '#CD853F'
            },
            3: {  # Cluster 3
                'name': 'Cluster 3',
                'base_probability': 0.15,
                'description': 'This cluster is possibly less populated and may reflect areas that were less influenced by major civilizations or more difficult terrains.',
                'archaeological_indicators': ['transitional zones', 'resource extraction'],
                'color': '#DEB887'
            },
            4: {  # Cluster 4
                'name': 'Cluster 4',
                'base_probability': 0.18,
                'description': 'This cluster shows variability; with more southern longitudes, it may indicate areas less explored or potentially significant due to archaeological evidence of cultural diversity',
                'archaeological_indicators': ['cultural'],
                'color': '#D2691E'
            }
        }
        
        # Known archaeological site associations
        self.known_site_clusters = {
            'MarajÃ³ Island': 0,
            'Geoglyphs of Acre': 1,
            'CayalÃ³n': 1,
            'Pano Culture Sites': 3,
            'Ankau': 1,
            'Sierra de la Capaceta': 4
        }
        
        # Calculate cluster weights based on known archaeological associations
        self.cluster_weights = self._calculate_cluster_weights()
        
        # Initialize interactive map data
        self.archaeological_sites = []
        self.water_sources = []
        self.selected_site = None
        
        # Amazon region bounds (longitude, latitude)
        self.amazon_bounds = {
            'lon_min': -73.0,
            'lon_max': -44.0,
            'lat_min': -18.0,
            'lat_max': 5.0
        }
        
    def _calculate_cluster_weights(self):
        """Calculate weights for each cluster based on known archaeological sites"""
        cluster_counts = {}
        for site, cluster in self.known_site_clusters.items():
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        
        total_sites = len(self.known_site_clusters)
        weights = {}
        for cluster_id in range(5):
            # Base weight + bonus for known archaeological associations
            base_weight = 1.0
            site_bonus = cluster_counts.get(cluster_id, 0) / total_sites
            weights[cluster_id] = base_weight + (site_bonus * 0.5)  # Max 50% bonus
            
        return weights
    
    def calculate_archaeological_probability(self, feature: Dict, distance_to_water: float, 
                                          soil_cluster: Optional[int] = None,
                                          cluster_confidence: float = 1.0) -> Dict:
        """
        Calculate probability that a feature is archaeological based on various factors
        
        Args:
            feature (dict): Feature information with keys like 'length', 'angle', 'area', etc.
            distance_to_water (float): Distance to nearest water source in km
            soil_cluster (int, optional): Soil cluster ID (0-4)
            cluster_confidence (float): Confidence in cluster assignment (0-1)
            
        Returns:
            dict: Detailed probability analysis including breakdown by factor
        """
        score_breakdown = {}
        total_score = 0.0
        
        # 1. Distance to water factor (closer = higher probability)
        water_score = 0.0
        if distance_to_water <= 1.0:
            water_score = 0.35
        elif distance_to_water <= 5.0:
            water_score = 0.25
        elif distance_to_water <= 10.0:
            water_score = 0.20
        elif distance_to_water <= 25.0:
            water_score = 0.15
        elif distance_to_water <= 50.0:
            water_score = 0.10
        
        score_breakdown['water_proximity'] = water_score
        total_score += water_score
        
        # 2. Morphological factors
        morphology_score = 0.0
        
        # Length factor (moderate length lines more likely to be archaeological)
        length = feature.get('length', 0)
        if 5 <= length <= 50:
            morphology_score += 0.20
        elif 50 <= length <= 150:
            morphology_score += 0.25
        elif 150 <= length <= 500:
            morphology_score += 0.15
        elif length > 500:  # Very long features might be natural
            morphology_score += 0.05
            
        # Area factor (if available)
        area = feature.get('area', 0)
        if area > 0:
            if 100 <= area <= 10000:  # Moderate sized features
                morphology_score += 0.10
            elif 10000 <= area <= 100000:  # Large features
                morphology_score += 0.15
        
        # Angle factor (cardinal directions more common for human-made features)
        angle = feature.get('angle', 0)
        if angle is not None:
            normalized_angle = abs(angle) % 90
            if normalized_angle <= 15 or normalized_angle >= 75:  # Close to N-S or E-W
                morphology_score += 0.15
            elif 30 <= normalized_angle <= 60:  # Diagonal orientations
                morphology_score += 0.10
        
        score_breakdown['morphology'] = min(morphology_score, 0.35)
        total_score += score_breakdown['morphology']
        
        # 3. Soil cluster factor
        cluster_score = 0.0
        cluster_info = None
        
        if soil_cluster is not None and soil_cluster in self.cluster_profiles:
            cluster_info = self.cluster_profiles[soil_cluster]
            
            # Base probability from cluster characteristics
            base_cluster_prob = cluster_info['base_probability']
            
            # Weight by known archaeological associations
            cluster_weight = self.cluster_weights.get(soil_cluster, 1.0)
            
            # Adjust by confidence in cluster assignment
            cluster_score = base_cluster_prob * cluster_weight * cluster_confidence
            
            score_breakdown['soil_cluster'] = {
                'score': cluster_score,
                'cluster_id': soil_cluster,
                'cluster_name': cluster_info['name'],
                'base_probability': base_cluster_prob,
                'archaeological_weight': cluster_weight,
                'confidence': cluster_confidence
            }
        else:
            # Default cluster score if no cluster information
            cluster_score = 0.10
            score_breakdown['soil_cluster'] = {
                'score': cluster_score,
                'cluster_id': None,
                'note': 'No soil cluster information available'
            }
        
        total_score += cluster_score
        
        # 4. Contextual factors
        context_score = 0.0
        
        # Elevation factor (if available)
        elevation = feature.get('elevation', None)
        if elevation is not None:
            if 100 <= elevation <= 500:  # Moderate elevation, good for settlements
                context_score += 0.10
            elif elevation <= 100:  # Low elevation, near rivers
                context_score += 0.15
        
        # Slope factor (if available)
        slope = feature.get('slope', None)
        if slope is not None:
            if slope <= 5:  # Gentle slopes preferred for settlements
                context_score += 0.10
            elif slope <= 15:
                context_score += 0.05
        
        # Drainage factor (if available)
        drainage = feature.get('drainage', None)
        if drainage is not None:
            if drainage in ['well_drained', 'moderately_drained']:
                context_score += 0.10
        
        score_breakdown['context'] = context_score
        total_score += context_score
        
        # 5. Baseline probability
        baseline_score = 0.05
        score_breakdown['baseline'] = baseline_score
        total_score += baseline_score
        
        # Normalize and apply final adjustments
        final_probability = min(total_score, 1.0)
        
        # Boost probability if multiple positive factors align
        positive_factors = sum(1 for score in [water_score, morphology_score, cluster_score, context_score] 
                             if score > 0.15)
        if positive_factors >= 3:
            final_probability = min(final_probability * 1.2, 1.0)
        
        # Create detailed response
        result = {
            'probability': final_probability,
            'confidence_level': self._get_confidence_level(final_probability),
            'score_breakdown': score_breakdown,
            'total_raw_score': total_score,
            'key_factors': self._identify_key_factors(score_breakdown),
            'recommendations': self._generate_recommendations(score_breakdown, feature, soil_cluster)
        }
        
        if cluster_info:
            result['soil_analysis'] = {
                'cluster_name': cluster_info['name'],
                'description': cluster_info['description'],
                'typical_indicators': cluster_info['archaeological_indicators']
            }
        
        return result
    
    def _get_confidence_level(self, probability: float) -> str:
        """Convert probability to confidence level"""
        if probability >= 0.7:
            return "High"
        elif probability >= 0.5:
            return "Moderate-High"
        elif probability >= 0.3:
            return "Moderate"
        elif probability >= 0.15:
            return "Low-Moderate"
        else:
            return "Low"
    
    def _identify_key_factors(self, breakdown: Dict) -> list:
        """Identify the most important contributing factors"""
        factors = []
        
        if breakdown['water_proximity'] >= 0.2:
            factors.append("Close proximity to water")
        
        if isinstance(breakdown['soil_cluster'], dict) and breakdown['soil_cluster']['score'] >= 0.2:
            cluster_name = breakdown['soil_cluster'].get('cluster_name', 'Unknown')
            factors.append(f"Favorable soil type ({cluster_name})")
        
        if breakdown['morphology'] >= 0.2:
            factors.append("Archaeological morphology")
        
        if breakdown['context'] >= 0.15:
            factors.append("Favorable environmental context")
        
        return factors
    
    def _generate_recommendations(self, breakdown: Dict, feature: Dict, soil_cluster: Optional[int]) -> list:
        """Generate recommendations for further investigation"""
        recommendations = []
        
        prob_score = sum(score if isinstance(score, (int, float)) else score.get('score', 0) 
                        for score in breakdown.values() if score != 'baseline')
        
        if prob_score >= 0.6:
            recommendations.append("High priority for ground survey")
            recommendations.append("Consider remote sensing analysis")
        
        if breakdown['water_proximity'] >= 0.25:
            recommendations.append("Investigate riparian archaeological context")
        
        if soil_cluster is not None and soil_cluster in [0, 2]:  # High organic or clay-rich
            recommendations.append("Good preservation potential - consider subsurface investigation")
        
        if breakdown['morphology'] >= 0.2:
            recommendations.append("Document geometric characteristics in detail")
        
        if not recommendations:
            recommendations.append("Monitor for additional supporting evidence")
        
        return recommendations
    
    def generate_sample_sites(self, num_sites: int = 15) -> List[Dict]:
        """Generate sample archaeological sites for demonstration using real Amazon coordinates"""
        sites = []
        
        # Known sites from the database with real Amazon coordinates
        known_sites_data = [
            {'name': 'MarajÃ³ Island', 'lon': -49.5, 'lat': -0.7, 'cluster': 2, 'type': 'known'},
            {'name': 'Geoglyphs of Acre', 'lon': -67.8, 'lat': -9.0, 'cluster': 1, 'type': 'known'},
            {'name': 'CayalÃ³n', 'lon': -58.2, 'lat': -3.8, 'cluster': 0, 'type': 'known'},
            {'name': 'Pano Culture Sites', 'lon': -65.4, 'lat': -7.2, 'cluster': 3, 'type': 'known'},
            {'name': 'Ankau', 'lon': -60.1, 'lat': -12.5, 'cluster': 4, 'type': 'known'},
            {'name': 'Sierra de la Capaceta', 'lon': -71.2, 'lat': -8.9, 'cluster': 1, 'type': 'known'}
        ]
        
        # Add known sites
        for i, site_data in enumerate(known_sites_data):
            feature = {
                'length': random.uniform(50, 200),
                'angle': random.uniform(0, 90),
                'area': random.uniform(1000, 15000),
                'elevation': random.uniform(20, 300),
                'slope': random.uniform(1, 10)
            }
            
            site = {
                'id': i,
                'name': site_data['name'],
                'lon': site_data['lon'],
                'lat': site_data['lat'],
                'type': site_data['type'],
                'cluster': site_data['cluster'],
                'features': feature,
                'distance_to_water': random.uniform(0.2, 8.0)
            }
            sites.append(site)
        
        # Generate additional potential sites within Amazon bounds
        for i in range(len(known_sites_data), num_sites):
            feature = {
                'length': random.uniform(20, 300),
                'angle': random.uniform(0, 90),
                'area': random.uniform(500, 20000),
                'elevation': random.uniform(10, 400),
                'slope': random.uniform(0, 15)
            }
            
            site = {
                'id': i,
                'name': f'Site {chr(65 + i - len(known_sites_data))}',
                'lon': random.uniform(self.amazon_bounds['lon_min'], self.amazon_bounds['lon_max']),
                'lat': random.uniform(self.amazon_bounds['lat_min'], self.amazon_bounds['lat_max']),
                'type': random.choice(['potential', 'anomaly']),
                'cluster': random.randint(0, 4),
                'features': feature,
                'distance_to_water': random.uniform(0.1, 12.0)
            }
            sites.append(site)
        
        return sites
    
    def generate_water_sources(self, num_points: int = 20) -> List[Tuple[float, float]]:
        """
        Generate water source points within the Amazon region.
        Returns a list of (longitude, latitude) tuples.
        """
        water_points = []
        
        # Major Amazon rivers (approximate coordinates)
        major_rivers = [
            # Amazon River main stem
            (-60.0, -3.1), (-58.5, -3.0), (-56.0, -2.5), (-54.0, -1.5), (-52.0, -1.0),
            (-50.0, -0.5), (-48.5, -1.2), (-47.0, -2.0),
            
            # Tributaries
            (-62.0, -5.5), (-64.5, -7.0), (-66.0, -8.5),  # Purus River
            (-58.0, -6.0), (-59.5, -7.5), (-61.0, -9.0),  # Madeira River
            (-54.5, -5.0), (-56.0, -6.5), (-57.5, -8.0),  # TapajÃ³s River
            (-52.0, -3.5), (-53.5, -5.0), (-55.0, -6.5),  # Xingu River
            (-48.0, -3.0), (-49.5, -4.5), (-51.0, -6.0),  # Tocantins River
        ]
        
        # Add major rivers
        water_points.extend(major_rivers[:num_points])
        
        # Fill remaining points with random water sources
        while len(water_points) < num_points:
            lon = random.uniform(self.amazon_bounds['lon_min'], self.amazon_bounds['lon_max'])
            lat = random.uniform(self.amazon_bounds['lat_min'], self.amazon_bounds['lat_max'])
            water_points.append((lon, lat))
        
        return water_points[:num_points]
    
    def create_interactive_map(self):
        """Create an interactive map visualization"""
        # Generate sample data
        self.archaeological_sites = self.generate_sample_sites()
        self.water_sources = self.generate_water_sources()
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9)
        
        # Initialize map
        self.update_map()
        
        # Create interactive elements
        self.create_controls()
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.title('Interactive Archaeological Site Probability Map - Amazon Basin', fontsize=16, fontweight='bold')
        plt.show()
    
    def update_map(self, show_probabilities=True, show_clusters=True, show_water=True):
        """Update the map display"""
        self.ax.clear()
        
        # Set map bounds to Amazon region
        self.ax.set_xlim(self.amazon_bounds['lon_min'], self.amazon_bounds['lon_max'])
        self.ax.set_ylim(self.amazon_bounds['lat_min'], self.amazon_bounds['lat_max'])
        self.ax.set_aspect('equal')
        
        # Background terrain
        self.ax.set_facecolor('#e8f4f8')
        
        # Add terrain features representing Amazon ecosystem zones
        terrain_patches = [
            # Southern Amazon (Cerrado transition)
            patches.Rectangle((self.amazon_bounds['lon_min'], self.amazon_bounds['lat_min']), 
                            29, 8, facecolor='#d4c5a9', alpha=0.3),
            # Central Amazon (Dense forest)
            patches.Rectangle((self.amazon_bounds['lon_min'], self.amazon_bounds['lat_min'] + 8), 
                            29, 10, facecolor='#228B22', alpha=0.3),
            # Northern Amazon (Guiana Shield)
            patches.Rectangle((self.amazon_bounds['lon_min'], self.amazon_bounds['lat_min'] + 18), 
                            29, 5, facecolor='#8fbc8f', alpha=0.3),
        ]
        
        for patch in terrain_patches:
            self.ax.add_patch(patch)
        
        # Show water sources
        if show_water:
            for wlon, wlat in self.water_sources:
                self.ax.scatter(wlon, wlat, c='blue', s=100, marker='o', alpha=0.7, label='Water')
                # Add water influence circles (1 degree radius â‰ˆ 111 km)
                circle = plt.Circle((wlon, wlat), 1.0, fill=False, color='blue', alpha=0.3, linestyle='--')
                self.ax.add_patch(circle)
        
        # Plot archaeological sites
        for site in self.archaeological_sites:
            # Calculate probability
            analysis = self.calculate_archaeological_probability(
                site['features'], site['distance_to_water'], site['cluster']
            )
            
            # Determine colors and sizes
            if show_probabilities:
                color = self.get_probability_color(analysis['probability'])
                size = 50 + analysis['probability'] * 200
            elif show_clusters:
                color = self.cluster_profiles[site['cluster']]['color']
                size = 100
            else:
                color = self.get_type_color(site['type'])
                size = 100
            
            # Plot site
            marker = self.get_type_marker(site['type'])
            self.ax.scatter(site['lon'], site['lat'], c=color, s=size, marker=marker, 
                           alpha=0.8, edgecolors='black', linewidth=1)
            
            # Add labels
            self.ax.annotate(site['name'], (site['lon'], site['lat']), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
            
            # Add probability text if showing probabilities
            if show_probabilities:
                prob_text = f"{analysis['probability']:.2f}"
                self.ax.annotate(prob_text, (site['lon'], site['lat']), 
                               xytext=(5, -15), textcoords='offset points', 
                               fontsize=7, alpha=0.7, color='darkred')
        
        self.ax.set_xlabel('Longitude (Â°W)', fontsize=12)
        self.ax.set_ylabel('Latitude (Â°N/S)', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        # Add legend
        self.add_legend(show_probabilities, show_clusters)
        
        plt.draw()
    
    def get_probability_color(self, probability):
        """Get color based on probability value"""
        if probability >= 0.7:
            return '#ff0000'  # Red - High
        elif probability >= 0.5:
            return '#ff8800'  # Orange - Moderate-High
        elif probability >= 0.3:
            return '#ffbb00'  # Yellow - Moderate
        elif probability >= 0.15:
            return '#88bb00'  # Yellow-Green - Low-Moderate
        else:
            return '#00bb00'  # Green - Low
    
    def get_type_color(self, site_type):
        """Get color based on site type"""
        colors = {
            'known': '#ff4444',
            'potential': '#ff8844',
            'anomaly': '#ffbb44'
        }
        return colors.get(site_type, '#888888')
    
    def get_type_marker(self, site_type):
        """Get marker based on site type"""
        markers = {
            'known': 's',      # Square
            'potential': 'o',  # Circle
            'anomaly': '^'     # Triangle
        }
        return markers.get(site_type, 'o')
    
    def add_legend(self, show_probabilities, show_clusters):
        """Add legend to the map"""
        legend_elements = []
        
        if show_probabilities:
            prob_colors = ['#ff0000', '#ff8800', '#ffbb00', '#88bb00', '#00bb00']
            prob_labels = ['High (â‰¥0.7)', 'Mod-High (â‰¥0.5)', 'Moderate (â‰¥0.3)', 'Low-Mod (â‰¥0.15)', 'Low (<0.15)']
            for color, label in zip(prob_colors, prob_labels):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=8, label=label))
        
        if show_clusters and not show_probabilities:
            for cluster_id, profile in self.cluster_profiles.items():
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=profile['color'], 
                                                markersize=8, label=profile['name']))
        
        # Site type legend
        type_markers = [('Known Sites', 's', '#ff4444'), ('Potential Sites', 'o', '#ff8844'), 
                       ('Anomalies', '^', '#ffbb44')]
        for label, marker, color in type_markers:
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w',
                                            markerfacecolor=color, markersize=8, label=label))
        
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    def create_controls(self):
        """Create interactive controls"""
        # Checkboxes for display options
        rax = plt.axes([0.02, 0.02, 0.15, 0.15])
        self.check = CheckButtons(rax, ('Probabilities', 'Clusters', 'Water'), (True, False, True))
        self.check.on_clicked(self.on_checkbox_clicked)
        
        # Add analysis button
        ax_button = plt.axes([0.2, 0.02, 0.1, 0.05])
        self.button = Button(ax_button, 'Analyze All')
        self.button.on_clicked(self.analyze_all_sites)
    
    def on_checkbox_clicked(self, label):
        """Handle checkbox clicks"""
        show_prob = self.check.get_status()[0]
        show_clusters = self.check.get_status()[1]
        show_water = self.check.get_status()[2]
        self.update_map(show_prob, show_clusters, show_water)
    
    def on_click(self, event):
        """Handle mouse clicks on the map"""
        if event.inaxes != self.ax:
            return
        
        # Find closest site
        min_dist = float('inf')
        closest_site = None
        
        for site in self.archaeological_sites:
            dist = np.sqrt((site['lon'] - event.xdata)**2 + (site['lat'] - event.ydata)**2)
            if dist < min_dist and dist < 1.0:  # Click tolerance adjusted for lat/lon
                min_dist = dist
                closest_site = site
        
        if closest_site:
            self.show_site_details(closest_site)
    
    def show_site_details(self, site):
        """Display detailed analysis for a selected site"""
        analysis = self.calculate_archaeological_probability(
            site['features'], site['distance_to_water'], site['cluster']
        )
        
        details = f"""
SITE ANALYSIS: {site['name']}
{'='*50}
Type: {site['type'].title()}
Location: {site['lat']:.2f}Â°N, {site['lon']:.2f}Â°W
Soil Cluster: {self.cluster_profiles[site['cluster']]['name']}

PROBABILITY ANALYSIS:
Overall Probability: {analysis['probability']:.3f} ({analysis['confidence_level']})

FACTOR BREAKDOWN:
- Water Proximity: {analysis['score_breakdown']['water_proximity']:.3f}
- Morphology: {analysis['score_breakdown']['morphology']:.3f}
- Soil Cluster: {analysis['score_breakdown']['soil_cluster']['score']:.3f}
- Context: {analysis['score_breakdown']['context']:.3f}

SITE CHARACTERISTICS:
- Distance to Water: {site['distance_to_water']:.1f} km
- Feature Length: {site['features']['length']:.1f} m
- Feature Area: {site['features']['area']:.0f} mÂ²
- Elevation: {site['features']['elevation']:.1f} m
- Slope: {site['features']['slope']:.1f}Â°

KEY FACTORS:
{chr(10).join('- ' + factor for factor in analysis['key_factors'])}

RECOMMENDATIONS:
{chr(10).join('- ' + rec for rec in analysis['recommendations'])}
        """
        
        print(details)
        
        # Highlight selected site on map
        self.ax.scatter(site['lon'], site['lat'], s=300, facecolors='none', 
                       edgecolors='red', linewidth=3, alpha=0.8)
        plt.draw()
    
    
    def analyze_all_sites(self, event):
        """Analyze all sites and print summary"""
        print("\nCOMPREHENSIVE SITE ANALYSIS")
        print("="*60)
        
        high_priority = []
        moderate_priority = []
        low_priority = []
        
        for site in self.archaeological_sites:
            analysis = self.calculate_archaeological_probability(
                site['features'], site['distance_to_water'], site['cluster']
            )
            
            if analysis['probability'] >= 0.6:
                high_priority.append((site['name'], analysis['probability']))
            elif analysis['probability'] >= 0.3:
                moderate_priority.append((site['name'], analysis['probability']))
            else:
                low_priority.append((site['name'], analysis['probability']))
        
        print(f"\nHIGH PRIORITY SITES ({len(high_priority)}):")
        for name, prob in sorted(high_priority, key=lambda x: x[1], reverse=True):
            print(f"- {name}: {prob:.3f}")
        
        print(f"\nMODERATE PRIORITY SITES ({len(moderate_priority)}):")
        for name, prob in sorted(moderate_priority, key=lambda x: x[1], reverse=True):
            print(f"- {name}: {prob:.3f}")
        
        print(f"\nLOW PRIORITY SITES ({len(low_priority)}):")
        for name, prob in sorted(low_priority, key=lambda x: x[1], reverse=True):
            print(f"- {name}: {prob:.3f}")
        
        # Calculate cluster statistics
        cluster_stats = {}
        for site in self.archaeological_sites:
            cluster = site['cluster']
            if cluster not in cluster_stats:
                cluster_stats[cluster] = []
            
            analysis = self.calculate_archaeological_probability(
                site['features'], site['distance_to_water'], site['cluster']
            )
            cluster_stats[cluster].append(analysis['probability'])
        
        print(f"\nCLUSTER PERFORMANCE:")
        for cluster_id, probabilities in cluster_stats.items():
            avg_prob = np.mean(probabilities)
            cluster_name = self.cluster_profiles[cluster_id]['name']
            print(f"- {cluster_name}: {avg_prob:.3f} (n={len(probabilities)})")

# Example usage and interactive demonstration
if __name__ == "__main__":
    print("Archaeological Probability Calculator with Interactive Map")
    print("="*60)
    print()
    
    # Create calculator instance
    calc = ArchaeologicalProbabilityCalculator()
    
    # Create interactive map
    calc.create_interactive_map()
    
    # Run some example calculations
    print("\nExample Calculations:")
    print("-" * 30)
    
    test_features = [
        {
            'name': 'Test Site 1',
            'features': {
                'length': 75,
                'angle': 10,  # Close to N-S
                'area': 5000,
                'elevation': 150,
                'slope': 3
            },
            'distance_to_water': 2.5,
            'cluster': 2
        },
        {
            'name': 'Test Site 2',
            'features': {
                'length': 200,
                'angle': 45,
                'area': 5000,
                'elevation': 50
            },
            'distance_to_water': 0.8,
            'cluster': 0
        }
    ]
    
    for test_site in test_features:
        result = calc.calculate_archaeological_probability(
            test_site['features'], 
            test_site['distance_to_water'], 
            test_site['cluster']
        )
        
        print(f"\n{test_site['name']}:")
        print(f"Probability: {result['probability']:.3f} ({result['confidence_level']})")
        print(f"Key Factors: {', '.join(result['key_factors'])}")
        if 'soil_analysis' in result:
            print(f"Soil Type: {result['soil_analysis']['cluster_name']}")
        print(f"Recommendations: {'; '.join(result['recommendations'])}")


#!/usr/bin/env python3
"""
Archaeological Site Detection System
Analyzes satellite imagery to detect potential archaeological sites,
focusing on linear features near water sources.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.warp import transform_bounds, reproject, Resampling
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, box
import requests
import folium
from folium import plugins
import os
import json
from datetime import datetime
import geemap
import ee
from PIL import Image
from io import BytesIO
import warnings
from skimage.measure import label, regionprops

warnings.filterwarnings('ignore')

# Try to import additional libraries (install if needed)
try:
    from skimage import filters, feature, morphology, measure
    from skimage.transform import hough_line, hough_line_peaks
    from scipy import ndimage
    ADVANCED_PROCESSING = True
except ImportError:
    print("Advanced image processing libraries not available. Install scikit-image and scipy for full functionality.")
    ADVANCED_PROCESSING = False

class ArchaeologicalDetector:
    """
    Main class for archaeological site detection using satellite imagery
    """
    
    def __init__(self, lat, lon, search_radius_km=50):
        """
        Initialize the detector with target coordinates
        
        Args:
            lat (float): Target latitude
            lon (float): Target longitude
            search_radius_km (float): Search radius in kilometers
        """
        self.lat = -12.50 #set as one point for cluster 0
        self.lon = -61.50
        self.search_radius_km = search_radius_km
        self.imagery_data = {}
        self.water_sources = []
        self.detected_features = []
        
        # Create bounding box for area of interest
        self.create_bounding_box()
        
    def create_bounding_box(self):
        """Create bounding box around target coordinates"""
        # Approximate conversion: 1 degree â‰ˆ 111 km
        deg_offset = self.search_radius_km / 111.0
        
        self.bbox = {
            'min_lat': self.lat - deg_offset,
            'max_lat': self.lat + deg_offset,
            'min_lon': self.lon - deg_offset,
            'max_lon': self.lon + deg_offset
        }
        
        # Create shapely polygon for the area
        self.area_polygon = box(
            self.bbox['min_lon'], self.bbox['min_lat'],
            self.bbox['max_lon'], self.bbox['max_lat']
        )
        
   
   
    def download_satellite_imagery(self, imagery_source='sentinel', preview=True):
        """
        Download satellite imagery for the entire Amazon basin with buffer using Google Earth Engine.
        
        Args:
            imagery_source (str): Source of imagery ('sentinel', 'landsat')
            preview (bool): If True, download a quick preview image instead of full-resolution export
        """
    
        print(f"ğŸŒ� Downloading {imagery_source} imagery from Google Earth Engine (Full Amazon Basin)...")
    
        # Authenticate and initialize Earth Engine
        try:
            ee.Initialize(project='ee-molly0124')
        except Exception:
            ee.Authenticate()
            ee.Initialize(project='ee-molly0124')
    
        # Define expanded Amazon region with generous buffer
        # This covers the entire Amazon basin plus surrounding areas
        roi = ee.Geometry.Polygon([
            [-85.0, 10.0],    # Northwest (extended into Central America)
            [-85.0, -25.0],   # Southwest (extended into Argentina)
            [-45.0, -25.0],   # Southeast (extended into Atlantic)
            [-45.0, 10.0],    # Northeast (extended into Caribbean)
            [-85.0, 10.0]     # Close the loop
        ])
    
        # Choose satellite imagery collection
        if imagery_source.lower() == 'sentinel':
            bands = ['B4', 'B3', 'B2']  # Red, Green, Blue
            scale = 10
            collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                .filterBounds(roi) \
                .filterDate('2023-06-01', '2023-06-30') \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
                .select(bands)
        elif imagery_source.lower() == 'landsat':
            bands = ['SR_B4', 'SR_B3', 'SR_B2']
            scale = 30
            collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterBounds(roi) \
                .filterDate('2023-06-01', '2023-06-30') \
                .filter(ee.Filter.lt('CLOUD_COVER', 10)) \
                .select(bands)
        else:
            raise ValueError("Unsupported imagery source. Use 'sentinel' or 'landsat'.")
    
        # Get median composite image
        image = collection.median().clip(roi)
    
        if preview:
            print("ğŸ“¥ Downloading preview image (1024px for full Amazon coverage)...")
            try:
                url = image.getThumbURL({
                    'bands': bands,
                    'region': roi,
                    'dimensions': 1024,  # Increased resolution for better coverage
                    'format': 'png',
                    'min': 0,
                    'max': 3000
                })
    
                print("ğŸ“¡ Thumb URL:", url)
    
                response = requests.get(url)
                response.raise_for_status()
    
                img = Image.open(BytesIO(response.content)).convert('RGB')
                self.imagery_data['rgb'] = np.array(img)
    
                # Update bbox to match the full Amazon region
                self.bbox = {
                    'min_lat': -25.0,
                    'max_lat': 10.0,
                    'min_lon': -85.0,
                    'max_lon': -45.0
                }
    
                print("âœ… Full Amazon basin preview image downloaded and loaded into self.imagery_data.")
    
            except Exception as e:
                print("â�Œ Failed to download or load preview image.")
                print(str(e))
    
        else:
            print("ğŸš€ Starting full-resolution export to Google Drive...")
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=f"{imagery_source.capitalize()}_Amazon_Full_June2023",
                folder='GEE_Exports',
                fileNamePrefix=f"{imagery_source}_amazon_full",
                region=roi,
                scale=scale,
                maxPixels=1e13
            )
            task.start()
            print("âœ… Export task started. Monitor progress in Google Earth Engine Task Manager or your Google Drive.")        

        
    def detect_water_sources(self):
        """
        Detect water sources in the imagery
        Uses NDWI (Normalized Difference Water Index) approach
        """
        print("Detecting water sources...")
        
        if 'rgb' not in self.imagery_data:
            print("No imagery data available. Please download imagery first.")
            return
            
        # Convert to grayscale for water detection
        gray = cv2.cvtColor(self.imagery_data['rgb'], cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to find dark areas (potential water)
        water_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours of water bodies
        contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to remove noise
        min_area = 100
        water_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Convert pixel coordinates to geographic coordinates
        self.water_sources = []
        height, width = gray.shape
        
        for contour in water_contours:
            # Get centroid of water body
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Convert to lat/lon (simplified conversion)
                lat = self.bbox['max_lat'] - (cy / height) * (self.bbox['max_lat'] - self.bbox['min_lat'])
                lon = self.bbox['min_lon'] + (cx / width) * (self.bbox['max_lon'] - self.bbox['min_lon'])
                
                self.water_sources.append({
                    'lat': lat,
                    'lon': lon,
                    'area': cv2.contourArea(contour),
                    'contour': contour
                })
                
        print(f"Detected {len(self.water_sources)} water sources")
        
    def detect_linear_features(self):
        """
        Detect linear features that could indicate archaeological sites by road
        """
        print("Detecting linear archaeological features...")
        
        if 'rgb' not in self.imagery_data:
            print("No imagery data available. Please download imagery first.")
            return
            
        # Convert to grayscale
        gray = cv2.cvtColor(self.imagery_data['rgb'], cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Use Hough Line Transform to detect straight lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        detected_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Convert pixel coordinates to geographic coordinates
                height, width = gray.shape
                
                lat1 = self.bbox['max_lat'] - (y1 / height) * (self.bbox['max_lat'] - self.bbox['min_lat'])
                lon1 = self.bbox['min_lon'] + (x1 / width) * (self.bbox['max_lon'] - self.bbox['min_lon'])
                lat2 = self.bbox['max_lat'] - (y2 / height) * (self.bbox['max_lat'] - self.bbox['min_lat'])
                lon2 = self.bbox['min_lon'] + (x2 / width) * (self.bbox['max_lon'] - self.bbox['min_lon'])
                
                # Calculate line length and angle
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                detected_lines.append({
                    'start': {'lat': lat1, 'lon': lon1},
                    'end': {'lat': lat2, 'lon': lon2},
                    'length': length,
                    'angle': angle,
                    'pixel_coords': [(x1, y1), (x2, y2)],
                    'feature_type': 'hough_line'
                })
        
        # Additional feature detection using advanced methods
        if ADVANCED_PROCESSING:
            detected_lines.extend(self.advanced_feature_detection(gray))
            
        self.detected_features = detected_lines
        print(f"Detected {len(detected_lines)} linear features")

    def advanced_feature_detection(self, gray_image):
        """
        Advanced detection of linear features using ridge filters, local maxima,
        morphological operations, and region analysis.
        
        Args:
            gray_image (np.ndarray): Grayscale image (uint8 or float).
        
        Returns:
            List[Dict]: List of detected linear features with location info.
        """
        print("ğŸ§  Running advanced feature detection...")
    
        # Normalize grayscale image
        gray_norm = gray_image / 255.0 if gray_image.max() > 1 else gray_image
    
        # 1. Ridge detection (Sato filter is good for tubular/linear structures)
        ridges = filters.sato(gray_norm, sigmas=range(1, 5))
    
        # 2. Morphological filtering to clean up noise
        ridges_bin = ridges > 0.1  # Threshold may need tuning
        ridges_clean = morphology.remove_small_objects(ridges_bin, min_size=50)
    
        # 3. Label connected components
        labeled = label(ridges_clean)
        regions = regionprops(labeled)
    
        detected_features = []
        height, width = gray_image.shape
    
        for region in regions:
            # Skip regions that are too small or not elongated
            if region.major_axis_length < 30 or region.eccentricity < 0.8:
                continue
    
            # Get centroid and orientation
            cy, cx = region.centroid
            angle = region.orientation * 180 / np.pi
            length = region.major_axis_length
    
            # Convert centroid pixel to geographic coordinates
            lat = self.bbox['max_lat'] - (cy / height) * (self.bbox['max_lat'] - self.bbox['min_lat'])
            lon = self.bbox['min_lon'] + (cx / width) * (self.bbox['max_lon'] - self.bbox['min_lon'])
    
            detected_features.append({
                'centroid': {'lat': lat, 'lon': lon},
                'length': length,
                'angle': angle,
                'region_bbox': region.bbox,
                'feature_type': 'ridge_detection'
            })
    
        print(f"ğŸ”� Advanced detection found {len(detected_features)} ridge features.")
        return detected_features

   
        
    def analyze_proximity_to_water(self, max_distance_km=2.0):
        """
        Analyze which linear features are near water sources
        
        Args:
            max_distance_km (float): Maximum distance to water in kilometers
        """
        print(f"Analyzing features within {max_distance_km}km of water sources...")
        
        archaeological_candidates = []
        
        for feature in self.detected_features:
            min_distance_to_water = float('inf')
            nearest_water = None
            
            # Calculate distance to each water source
            for water in self.water_sources:
                # Get feature coordinates based on feature type
                if 'start' in feature and 'end' in feature:
                    # Hough line feature - use midpoint
                    feature_lat = (feature['start']['lat'] + feature['end']['lat']) / 2
                    feature_lon = (feature['start']['lon'] + feature['end']['lon']) / 2
                elif 'centroid' in feature:
                    # Ridge detection feature - use centroid
                    feature_lat = feature['centroid']['lat']
                    feature_lon = feature['centroid']['lon']
                else:
                    # Skip features without recognizable coordinate format
                    continue
                
                # Calculate distance using haversine formula
                distance = self.haversine_distance(
                    feature_lat, feature_lon,
                    water['lat'], water['lon']
                )
                
                if distance < min_distance_to_water:
                    min_distance_to_water = distance
                    nearest_water = water
                    
            # If feature is within specified distance of water, mark as candidate
            if min_distance_to_water <= max_distance_km:
                candidate = feature.copy()
                candidate['distance_to_water_km'] = min_distance_to_water
                candidate['nearest_water'] = nearest_water
                candidate['archaeological_probability'] = self.calculate_archaeological_probability(
                    feature, min_distance_to_water
                )
                archaeological_candidates.append(candidate)
                
        # Sort by archaeological probability
        archaeological_candidates.sort(key=lambda x: x['archaeological_probability'], reverse=True)
        
        print(f"Found {len(archaeological_candidates)} potential archaeological features near water")
        return archaeological_candidates
        
    def calculate_archaeological_probability(self, feature, distance_to_water):
        """
        Calculate probability that a feature is archaeological based on various factors
        
        Args:
            feature (dict): Feature information
            distance_to_water (float): Distance to nearest water source in km
            
        Returns:
            float: Probability score (0-1)
        """
        score = 0.0
        
        # Distance to water factor (closer = higher probability)
        if distance_to_water <= 1.0:
            score += 0.4
        elif distance_to_water <= 10.0:
            score += 0.3
        elif distance_to_water <= 50.0:
            score += 0.2
            
        # Length factor (moderate length lines more likely to be archaeological)
        length = feature.get('length', 0)
        if 5 <= length <= 100:
            score += 0.3
        elif 100 <= length <= 300:
            score += 0.2
            
        # Angle factor (cardinal directions more common for human-made features)
        angle = abs(feature.get('angle', 0)) % 90
        if angle <= 15 or angle >= 75:  # Close to N-S or E-W
            score += 0.2
            
        # Random baseline probability
        score += 0.1
        
        return min(score, 1.0)
        
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points on Earth
        
        Returns:
            float: Distance in kilometers
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r
        
    def visualize_results(self, show_all_features=False):
        """
        Create visualizations of the detected features
        
        Args:
            show_all_features (bool): Whether to show all features or just archaeological candidates
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Original imagery
        axes[0, 0].imshow(self.imagery_data['rgb'])
        axes[0, 0].set_title('Original Satellite Imagery (Full Amazon Basin)')
        axes[0, 0].axis('off')
        
        # Plot 2: Detected linear features (basic + advanced)
        img_with_features = self.imagery_data['rgb'].copy()
        for feature in self.detected_features:
            if 'pixel_coords' in feature:
                # Hough line features
                cv2.line(img_with_features, feature['pixel_coords'][0], feature['pixel_coords'][1], (255, 0, 0), 2)
            elif 'centroid' in feature:
                # Ridge detection features
                height, width = img_with_features.shape[:2]
                lat = feature['centroid']['lat']
                lon = feature['centroid']['lon']
                y = int((self.bbox['max_lat'] - lat) / (self.bbox['max_lat'] - self.bbox['min_lat']) * height)
                x = int((lon - self.bbox['min_lon']) / (self.bbox['max_lon'] - self.bbox['min_lon']) * width)
                # Make sure coordinates are within bounds
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(img_with_features, (x, y), 3, (0, 255, 255), -1)
        
        axes[0, 1].imshow(img_with_features)
        axes[0, 1].set_title(f'Detected Linear Features ({len(self.detected_features)})')
        axes[0, 1].axis('off')
        
        # Plot 3: Water sources
        img_with_water = self.imagery_data['rgb'].copy()
        for water in self.water_sources:
            if 'contour' in water:
                cv2.drawContours(img_with_water, [water['contour']], -1, (0, 255, 255), 2)
        axes[1, 0].imshow(img_with_water)
        axes[1, 0].set_title(f'Detected Water Sources ({len(self.water_sources)})')
        axes[1, 0].axis('off')
        
        # Plot 4: Archaeological candidates
        candidates = self.analyze_proximity_to_water()
        img_with_candidates = self.imagery_data['rgb'].copy()
        height, width = img_with_candidates.shape[:2]
        
        for candidate in candidates[:10]:  # Show top 10
            prob = candidate.get('archaeological_probability', 0)
            if prob > 0.6:
                color = (255, 0, 0)  # Red
            elif prob > 0.4:
                color = (255, 255, 0)  # Yellow
            else:
                color = (0, 255, 0)  # Green
                
            if 'pixel_coords' in candidate:
                # Hough line candidates
                coords = candidate['pixel_coords']
                cv2.line(img_with_candidates, coords[0], coords[1], color, 3)
            elif 'centroid' in candidate:
                # Ridge detection candidates
                lat = candidate['centroid']['lat']
                lon = candidate['centroid']['lon']
                y = int((self.bbox['max_lat'] - lat) / (self.bbox['max_lat'] - self.bbox['min_lat']) * height)
                x = int((lon - self.bbox['min_lon']) / (self.bbox['max_lon'] - self.bbox['min_lon']) * width)
                # Make sure coordinates are within bounds
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(img_with_candidates, (x, y), 5, color, -1)
        
        axes[1, 1].imshow(img_with_candidates)
        axes[1, 1].set_title(f'Archaeological Candidates (Top 10)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print top 5 candidates with metadata
        print("\nTop Archaeological Candidates:")
        print("-" * 80)
        for i, candidate in enumerate(candidates[:5]):
            print(f"{i+1}. Probability: {candidate['archaeological_probability']:.2f}")
            print(f"   Feature Type: {candidate.get('feature_type', 'unknown')}")
            print(f"   Length: {candidate.get('length', 0):.1f} pixels")
            print(f"   Distance to water: {candidate.get('distance_to_water_km', 0):.2f} km")
            print(f"   Angle: {candidate.get('angle', 0):.1f}Â°")
            
            if 'start' in candidate and 'end' in candidate:
                print(f"   Coordinates: ({candidate['start']['lat']:.4f}, {candidate['start']['lon']:.4f}) to "
                      f"({candidate['end']['lat']:.4f}, {candidate['end']['lon']:.4f})")
            elif 'centroid' in candidate:
                print(f"   Centroid: ({candidate['centroid']['lat']:.4f}, {candidate['centroid']['lon']:.4f})")
            print()

    def create_interactive_map(self):
        """
        Create an interactive map showing all detected features
        """
        # Create base map centered on Amazon basin
        center_lat = (self.bbox['min_lat'] + self.bbox['max_lat']) / 2
        center_lon = (self.bbox['min_lon'] + self.bbox['max_lon']) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
        
        # Add search area boundary
        folium.Rectangle(
            bounds=[[self.bbox['min_lat'], self.bbox['min_lon']], 
                   [self.bbox['max_lat'], self.bbox['max_lon']]],
            color='blue',
            fill=False,
            popup='Amazon Basin Search Area'
        ).add_to(m)
        
        # Add water sources
        for i, water in enumerate(self.water_sources):
            folium.CircleMarker(
                location=[water['lat'], water['lon']],
                radius=5,
                popup=f'Water Source {i+1}<br>Area: {water["area"]:.0f} pixels',
                color='blue',
                fill=True,
                fillColor='lightblue'
            ).add_to(m)
            
        # Add archaeological candidates
        candidates = self.analyze_proximity_to_water()
        for i, candidate in enumerate(candidates):
            # Color based on probability
            prob = candidate.get('archaeological_probability', 0)
            color = 'red' if prob > 0.7 else 'orange' if prob > 0.4 else 'green'
            
            if 'start' in candidate and 'end' in candidate:
                # Hough line features
                start_latlon = [candidate['start']['lat'], candidate['start']['lon']]
                end_latlon = [candidate['end']['lat'], candidate['end']['lon']]
                
                folium.PolyLine(
                    locations=[start_latlon, end_latlon],
                    color=color,
                    weight=3,
                    popup=(
                        f'Archaeological Candidate {i+1}<br>'
                        f'Type: {candidate.get("feature_type", "unknown")}<br>'
                        f'Probability: {prob:.2f}<br>'
                        f'Distance to water: {candidate.get("distance_to_water_km", 0):.2f} km<br>'
                        f'Length: {candidate.get("length", 0):.1f} pixels'
                    )
                ).add_to(m)
            elif 'centroid' in candidate:
                # Ridge detection features
                centroid_latlon = [candidate['centroid']['lat'], candidate['centroid']['lon']]
                
                folium.CircleMarker(
                    location=centroid_latlon,
                    radius=8,
                    color=color,
                    fill=True,
                    fillColor=color,
                    popup=(
                        f'Archaeological Candidate {i+1}<br>'
                        f'Type: {candidate.get("feature_type", "unknown")}<br>'
                        f'Probability: {prob:.2f}<br>'
                        f'Distance to water: {candidate.get("distance_to_water_km", 0):.2f} km<br>'
                        f'Length: {candidate.get("length", 0):.1f} pixels'
                    )
                ).add_to(m)
            
        return m

    def export_results(self, filename='archaeological_survey_results.json'):
        results = {
            'survey_info': {
                'target_coordinates': {'lat': self.lat, 'lon': self.lon},
                'search_radius_km': self.search_radius_km,
                'bounding_box': self.bbox,
                'survey_date': datetime.now().isoformat(),
                'coverage_area': 'Full Amazon Basin with Buffer'
            },
            'water_sources': self.water_sources,
            'linear_features': self.detected_features,
            'archaeological_candidates': self.analyze_proximity_to_water()
        }
    
        # Strip non-serializable items
        for candidate in results['archaeological_candidates']:
            if 'nearest_water' in candidate and 'contour' in candidate['nearest_water']:
                del candidate['nearest_water']['contour']
    
        for water in results['water_sources']:
            if 'contour' in water:
                del water['contour']
    
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
    
        print(f"Results exported to {filename}")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()  # Convert scalar numpy types to native Python
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert arrays to lists
        return super().default(obj)

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj
        
def main():
    """
    Main function to run the archaeological detection system
    """

    # Initialize detector with larger search radius for Amazon scale
    detector = ArchaeologicalDetector(lat, lon, search_radius_km=100)
    
    # Run detection pipeline
    detector.download_satellite_imagery()
    detector.detect_water_sources()
    detector.detect_linear_features()
    # detector.advanced_feature_detection()
    
    # Visualize results
    detector.visualize_results()
    
    # Create interactive map
    interactive_map = detector.create_interactive_map()
    interactive_map.save('amazon_archaeological_survey_map.html')
    print("Interactive map saved as 'amazon_archaeological_survey_map.html'")
    
    # Export results
    detector.export_results('amazon_archaeological_survey_results.json')
    
    print("\nAmazon Basin Archaeological Survey completed successfully!")

if __name__ == "__main__":
    main()




