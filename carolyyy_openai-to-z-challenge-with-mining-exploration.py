!pip install geopandas --no-index --find-links=/kaggle/input/kaggle-packages
!pip install shapely --no-index --find-links=/kaggle/input/kaggle-packages
!pip install rasterio 
!pip install pdfplumber
!pip install contextily
!pip install folium
!pip install osmnx
!pip install osmium
!pip install geodatasets


import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import seaborn as sns
import rasterio
import re
import pdfplumber
import os
import contextily as ctx
import osmnx as ox
from matplotlib.colors import LinearSegmentedColormap
import folium
from folium.plugins import MarkerCluster, HeatMap
import geodatasets as gds


# get centroid latitude and longitude coordinates for each state by gpt(from gpt-chat-records in input)
brazil_state_centroids = {   
    "AC": (-8.77, -70.55),   # Acre
    "AM": (-3.47, -65.10),   # Amazonas
    "RO": (-11.22, -62.80),  # Rondônia
    "AL": (-9.62, -36.82),   # Alagoas
    "SE": (-10.57, -37.45),  # Sergipe
    "RR": (1.89, -61.22),    # Roraima
    "PA": (-5.53, -52.29),   # Pará
    "MT": (-12.64, -55.42),  # Mato Grosso
    "AP": (1.41, -51.77),    # Amapá
    "BA": (-12.96, -38.51),  # Bahia
    "MG": (-18.10, -44.38),  # Minas Gerais
    "PE": (-8.28, -35.07),   # Pernambuco
    "PI": (-7.28, -42.79),   # Piauí
    "ES": (-19.19, -40.34),  # Espírito Santo
    "CE": (-5.20, -39.53),   # Ceará
    "RN": (-5.22, -36.52),   # Rio Grande do Norte
    "PB": (-7.06, -35.55),   # Paraíba
    "DF": (-15.83, -47.86),  # Distrito Federal
    "GO": (-16.64, -49.31),  # Goiás
    "RJ": (-22.25, -42.66),  # Rio de Janeiro
    "TO": (-10.25, -48.25),  # Tocantins
    "MS": (-20.51, -54.54),  # Mato Grosso do Sul
    "MA": (-5.42, -45.44),   # Maranhão
    "SP": (-22.19, -48.79),  # São Paulo
    "PR": (-24.89, -51.55),  # Paraná
    "SC": (-27.33, -49.44),  # Santa Catarina
    "RS": (-30.03, -51.22)   # Rio Grande do Sul
}
mining_dir = '/kaggle/input/amazonian-related-open-data/Brazil_mining_concessions.csv'
mining_dfa = pd.read_csv(mining_dir, encoding='latin1')
mining_df = mining_dfa[mining_dfa.uf.str.upper().isin(brazil_state_centroids.keys())]
mining_df['latitude'] = mining_df['uf'].map(lambda x: brazil_state_centroids[x.upper()][0])
mining_df['longitude'] = mining_df['uf'].map(lambda x: brazil_state_centroids[x.upper()][1])
mining_df['area_density'] = mining_df['area_ha'] / mining_df['shape_Area']
mining_df['log_area'] = np.log10(mining_df['area_ha'] + 1)


# process PDF file, get the coordinate positions of the current discovered works
pdf_path = '/kaggle/input/amazonian-related-open-data/List of Southwestern Amazonian Earthworks 25.08.2024b.pdf'
coordinates = []
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        pattern = r'(-?\d+\.\d+)\s+(-?\d+\.\d+)'
        matches = re.findall(pattern, text)

        for match in matches:
            longitude = float(match[0])
            latitude = float(match[1])
            coordinates.append({'latitude': latitude,'longitude': longitude})

df = pd.DataFrame(coordinates)


import json
trans_path = '/kaggle/input/translate-pt2en/translate.json'
with open(trans_path, 'r', encoding='utf-8') as file:  
    trans_dict = json.load(file)
mining_df['subs_en'] = mining_df['subs'].map(trans_dict)


mining_df['geometry'] = mining_df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
mining_gdf = gpd.GeoDataFrame(mining_df, geometry='geometry', crs="EPSG:4326")

def plot_site_mining(site_lon, site_lat, mining_gdf, buffer_km):
    site_point = Point(site_lon, site_lat)
    buffer_zone = site_point.buffer(buffer_km / 111)  # 1°≈111km
    mining_in_buffer = mining_gdf[mining_gdf.intersects(buffer_zone)].copy()

    if not mining_in_buffer.empty:
        print(f"Mining activities were found within a {buffer_km}km radius around {site_lon, site_lat}.")
        
for k in coordinates:
    plot_site_mining(k['longitude'], k['latitude'], mining_gdf, buffer_km=250)


# Create virtual mining area locations (randomly distributed within the state)
def create_virtual_locations(row, state_centroids):
    centroid = state_centroids[row['uf'].upper()]
    # Generate random points within a 1-degree range around the centroid
    jitter_lon = centroid[1] + np.random.uniform(-1.0, 1.0)
    jitter_lat = centroid[0] + np.random.uniform(-1.0, 1.0)
    return Point(jitter_lon, jitter_lat)

mining_df['geometry'] = mining_df.apply(lambda row: create_virtual_locations(row, brazil_state_centroids), axis=1)
mining_gdf = gpd.GeoDataFrame(mining_df, geometry='geometry', crs="EPSG:4326")

# load the List of Southwestern Amazonian Earthworks
ancient_sites = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.longitude, df.latitude),crs="EPSG:4326")


# group by states
state_mining = mining_gdf.groupby('uf').agg(total_area=('area_ha', 'sum'),
        avg_area=('area_ha', 'mean'),
        count=('id', 'count'),
        avg_density=('area_density', 'mean'),
        minerals=('subs_en', lambda x: x.value_counts().to_dict())).reset_index()

state_mining['latitude'] = state_mining['uf'].map(lambda x: brazil_state_centroids[x.upper()][0])
state_mining['longitude'] = state_mining['uf'].map(lambda x: brazil_state_centroids[x.upper()][1])

world = gpd.read_file(gds.get_path("naturalearth.land"))
brazil_lon_min, brazil_lon_max = -74.05, -34.47  
brazil_lat_min, brazil_lat_max = -33.45, 5.16    
brazil = world.cx[brazil_lon_min:brazil_lon_max, brazil_lat_min:brazil_lat_max]

def plot_distance_miningworks(state_mining,ancient_sites):
# plot1 show the distance of mining sites and ancient works
    state_gdf = gpd.GeoDataFrame(state_mining,
        geometry=gpd.points_from_xy(state_mining.longitude, state_mining.latitude),
        crs="EPSG:4326")
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    sc = ax.scatter(
        state_gdf.geometry.x,
        state_gdf.geometry.y,
        s=state_gdf['total_area'] / 1000,  
        c=state_gdf['count'],
        cmap='viridis',
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5,
        zorder=10
    )
    
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('count of mining site')
    
    for idx, row in state_gdf.iterrows():
        ax.annotate(row['uf'],
                    (row.geometry.x, row.geometry.y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, weight='bold')
    ancient_sites.plot(ax=ax, color='red', markersize=20, marker='*',edgecolor='gold', label='ancient works', zorder=20)

    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldTerrain, alpha=0.5)
    # brazil.plot(ax=ax, facecolor='lightgray', edgecolor='gray', alpha=0.5, zorder=1)
    
    sizes = [100, 1000,5000, 10000,100000]  # different mining area levels
    labels = ['100 ha', '1000 ha', '5000 ha','10000 ha','100000 ha']
    log_sizes = np.log10(np.array(sizes))
    scaled_sizes = 100 * (log_sizes - min(log_sizes)) / (max(log_sizes) - min(log_sizes)) + 20
    handles = []
    for s, size_val, label in zip(sizes, scaled_sizes, labels):
        handle = plt.scatter([], [], 
                         s=size_val,  
                         edgecolor='black', 
                         facecolor='none', 
                         linewidth=1.5,  
                         label=label)
        handles.append(handle)
    ax.legend(handles=handles, title='sum of mining area', loc='upper left')
    plt.title('Distribution of earth works and mining sites', fontsize=16)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.grid(alpha=0.3)
    
    plt.savefig('state_mining_intensity_with_sites.png', dpi=300, bbox_inches='tight')
    plt.show()
    return state_gdf


plot_distance_miningworks(state_mining,ancient_sites)


# Visualize the intensity of mining activities and the distribution of ancient civilization sites
def plot_state_mining_intensity(mining_gdf, ancient_sites,top_n=5):
    # Add major mineral types for each state
    for i, row in state_mining.iterrows():
        minerals = row['minerals']
        if minerals:
            # get topn
            sorted_minerals = sorted(minerals.items(), key=lambda x: x[1], reverse=True)[:top_n]
            state_mining.at[i, 'main_minerals'] = ", ".join([m[0] for m in sorted_minerals])
            state_mining.at[i, 'top_mineral'] = sorted_minerals[0][0]
            state_mining.at[i, 'mineral_count'] = sum(minerals.values())
        else:
            state_mining.at[i, 'main_minerals'] = "NoData"
            state_mining.at[i, 'top_mineral'] = "NoData"

    state_mining['site_count'] = 0
    for key,site in ancient_sites.items():
        # find the closest state
        min_dist = float('inf')
        closest_state = None
        site_point = Point(site[0],site[1])

        for state, coord in brazil_state_centroids.items():
            state_point = Point(coord[1], coord[0])  # (lon, lat)
            dist = site_point.distance(state_point)
            if dist < min_dist:
                min_dist = dist
                closest_state = state

        if closest_state in state_mining['uf'].values:
            state_mining.loc[state_mining['uf'] == closest_state, 'site_count'] += 1   
        
        fig2 = plt.figure(figsize=(20, 18))
        gs = GridSpec(3, 2, figure=fig2, height_ratios=[1.5, 1, 1])

# plot2 Mineral Types and Site Distribution
        ax_map = fig2.add_subplot(gs[0, :])

        unique_minerals = mining_gdf['subs_en'].unique()
        mineral_colors_ = plt.cm.gist_rainbow(np.linspace(0, 1, len(unique_minerals)))
        mineral_colors = np.array([np.clip(c * 1.5, 0, 1) for c in mineral_colors_])
        mineral_color_map = dict(zip(unique_minerals, mineral_colors))

        for idx_x, row in state_mining.iterrows():
            if row['top_mineral'] in mineral_color_map:
                color = mineral_color_map[row['top_mineral']]
                size = np.sqrt(row['total_area']) / 50  # area scale

                ax_map.scatter(row['longitude'],row['latitude'],
                    s=size,c=[color],edgecolor='black',alpha=0.8,zorder=10)

                mineral_abbr = ''.join([word[0] for word in row['top_mineral'].split()])[:4]
                label = f"{row['uf']}\n{mineral_abbr}"
                ax_map.annotate(
                    label,
                    (row['longitude'], row['latitude']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle="round",pad=0.3, fc="white", alpha=0.7)
                )
        
        ax_map.scatter(
        site[0], site[1],
        s=150,  
        c='gold',  
        marker='*',  
        edgecolor='black',  
        linewidth=1,  
        alpha=0.9,  
        zorder=20,  
        label=key  
        )

        ctx.add_basemap(ax_map, crs="EPSG:4326", source=ctx.providers.Esri.WorldTerrain, alpha=0.5)
        # brazil.plot(ax=ax_map, facecolor='lightgray', edgecolor='gray', alpha=0.5, zorder=1)
        ax_map.set_title(f'Mineral Types and {key} Archaeological Site in Brazilian States', fontsize=16, pad=20)
        ax_map.set_xlabel('Longitude')
        ax_map.set_ylabel('Latitude')

        legend_elements = []
        for mineral, color in mineral_color_map.items():
            legend_elements.append(
                Patch(facecolor=color, edgecolor='black', label=mineral)
            )

        ax_map.legend(
            handles=legend_elements[:15], 
            loc='upper left',
            bbox_to_anchor=(1.01, 1),
            title="Mineral Types",
            ncol=2
        )

# plot3 Mineral Type Distribution in States with Abundant Archaeological Sites
        ax_stacked = fig2.add_subplot(gs[1, 1])

        top_states = state_mining.nlargest(3, 'site_count')['uf']

        stack_data = []
        for state in top_states:
            minerals = state_mining[state_mining['uf'] == state]['minerals'].iloc[0]
            if isinstance(minerals, dict):
                sorted_minerals = sorted(minerals.items(), key=lambda x: x[1], reverse=True)[:5]
                for mineral, count in sorted_minerals:
                    stack_data.append({
                        'State': state,
                        'Mineral': mineral,
                        'Count': count
                    })

        stack_df = pd.DataFrame(stack_data)

        pivot_df = stack_df.pivot(index='State', columns='Mineral', values='Count').fillna(0)

        pivot_df.plot(kind='bar', stacked=True, ax=ax_stacked,
                      color=[mineral_color_map.get(m, 'gray') for m in pivot_df.columns])

        ax_stacked.set_ylabel('Number of Mines')
        ax_stacked.set_title(f'Mineral Distribution in States with {key} Archaeological Site', fontsize=14)
        ax_stacked.legend(title='Mineral Types', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_stacked.grid(axis='y', alpha=0.3)
        fig2.savefig(f'Mineral Distribution in States with {key} Archaeological Site.png', 
                     dpi=300,  
                     bbox_inches='tight', 
                     facecolor='white')  


points = np.array([[p['longitude'], p['latitude']] for p in coordinates])
ancient_sites_most = {
        "northernmost": points[np.argmax(points[:, 1])],  
        "southernmost": points[np.argmin(points[:, 1])],  
        "easternmost": points[np.argmax(points[:, 0])],   
        "westernmost": points[np.argmin(points[:, 0])],   
        "centroid": np.around(np.mean(points, axis=0), decimals=6).tolist(),     
    }
print(ancient_sites_most)
plot_state_mining_intensity(mining_gdf, ancient_sites_most)  


from rasterio.crs import CRS

# get the Lidar data around the most of the ancient_sites
data_dirs = [
    '/kaggle/input/amazonian-related-open-data/S1A_IW_SLC__1SSV_20141122T225310_20141122T225326_003400_003F6A_2981/S1A_IW_SLC__1SSV_20141122T225310_20141122T225326_003400_003F6A_2981.SAFE',
    '/kaggle/input/amazonian-related-open-data/S1A_IW_SLC__1SSV_20141021T100541_20141021T100611_002926_00351B_E372/S1A_IW_SLC__1SSV_20141021T100541_20141021T100611_002926_00351B_E372.SAFE'
]

subswath_data = {}

for p,data_dir in enumerate(data_dirs):
    measurement_dir = os.path.join(data_dir, "measurement")
        
    tiff_files = []
    for file in os.listdir(measurement_dir):
        if file.endswith(".tiff"):
            tiff_files.append(os.path.join(measurement_dir, file))
    
    for s,file in enumerate(tiff_files):
        with rasterio.open(file) as src:
            filename = os.path.basename(file)
            parts = filename.split('-')
            if len(parts) < 4:
                print(f"filename error: {filename}")
                continue
                
            subswath = parts[1]
            polarization = parts[3]
            band = src.read(1)

            abs_band = np.abs(band)
            abs_band[abs_band == 0] = 1e-10
            db_band = 10 * np.log10(abs_band)

            key = f"lidar{p+1}_{subswath}_{polarization}"
            subswath_data[key] = {
                "data": band,
                "db_data": db_band,
                "transform": src.transform,
                "crs": src.crs,
                "bounds": src.bounds,
                "shape": db_band.shape  
            }
            print(f"load Lidar{p+1} file{s+1} subswath:{subswath} polarization:{polarization} | shape: {band.shape}")
            print('Coordinate System:', src.crs)    
            
def plot_four_corners_in_one(data_dict, subswath_pol):
    if subswath_pol not in data_dict:
        print(f"data not found: {subswath_pol}")
        return
        
    data = data_dict[subswath_pol]
    db_data = data['db_data']
    transform = data['transform']
    crs = data['crs']
    height, width = data['shape']
    
    k = 1 / np.sqrt(6)  
    w = int(width * k)  
    h = int(height * k) 
    
    # define the corners
    regions = [
        ("Upper Left", (0, 0, w, h)),
        ("Upper Right", (width - w, 0, width, h)),
        ("Lower Left", (0, height - h, w, height)),
        ("Lower Right", (width - w, height - h, width, height))
    ]
    
    # 2x2grid
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Corner Regions of {subswath_pol} Radar Image", fontsize=16, y=0.95)
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    global_vmin = np.nanpercentile(db_data, 5)
    global_vmax = np.nanpercentile(db_data, 95)
    
    for i, (name, (col_start, row_start, col_end, row_end)) in enumerate(regions):
        ax = fig.add_subplot(gs[i])
        roi = db_data[row_start:row_end, col_start:col_end]
        x_ul, y_ul = transform * (col_start, row_start)
        x_br, y_br = transform * (col_end, row_end)
        
        im = ax.imshow(
            roi, 
            cmap='gray',
            extent=(x_ul, x_br, y_br, y_ul),  # (left, right, bottom, top)
            vmin=global_vmin,
            vmax=global_vmax
        )
        ax.set_title(f'{name} Corner\nSize: {roi.shape[1]}x{roi.shape[0]}', fontsize=12)
        
        if i in [0, 2]:  
            if crs and crs.is_geographic:
                ax.set_ylabel('Latitude', fontsize=10)
            else:
                ax.set_ylabel('Y Coord (m)', fontsize=10)
        
        if i in [2, 3]:  
            if crs and crs.is_geographic:
                ax.set_xlabel('Longitude', fontsize=10)
            else:
                ax.set_xlabel('X Coord (m)', fontsize=10)
        
        ax.grid(color='white', alpha=0.3, linestyle='--')
        
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Backscattering Coefficient(dB)', fontsize=12)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])  
    
    plt.savefig(f"corner_regions_{subswath_pol}.png", dpi=300, bbox_inches='tight')
    plt.show()

for subswath_pol in subswath_data.keys():
    print(f"\n Process corners of {subswath_pol}...")
    plot_four_corners_in_one(subswath_data, subswath_pol)

