# Standard library imports and helpers
import json
import os
import folium
import io, contextlib
from folium.plugins import HeatMap
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from haversine import haversine, Unit
import numpy as np
warnings.filterwarnings('ignore')

OUTPUT_DIR = "/kaggle/working/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


geojson_file_path = '/kaggle/input/archaeoblog-amazon-geoglyphs/geoglyph_subset.geojson'
latitudes = []
longitudes = []
extracted_placemarks = [] 
try:
    with open(geojson_file_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
except FileNotFoundError:
    print(f"Error: GeoJSON file not found at {geojson_file_path}.")
    print("Please ensure the dataset 'archaeoblog-amazon-geoglyphs' is added to your Kaggle notebook.")
    geojson_data = {"type": "FeatureCollection", "features": []}
except json.JSONDecodeError as e:
    print(f"Error: Could not decode GeoJSON file. Invalid JSON format: {e}")
    geojson_data = {"type": "FeatureCollection", "features": []}
except Exception as e:
    print(f"An unexpected error occurred while reading the GeoJSON file: {e}")
    geojson_data = {"type": "FeatureCollection", "features": []}


if geojson_data and "features" in geojson_data:
    for feature in geojson_data["features"]:
        # Ensure the feature has a geometry and it's a Point type.
        if "geometry" in feature and feature["geometry"]["type"] == "Point":
            lon, lat = feature["geometry"]["coordinates"][:2]
            latitudes.append(lat)
            longitudes.append(lon)

            properties = feature.get("properties", {})
            name = properties.get("Name", "Unnamed Feature")
            description = properties.get("description", "No description provided.")

            extracted_placemarks.append({
                'name': name,
                'description': description,
                'latitude': lat,
                'longitude': lon
            })
            
if latitudes and longitudes:
    avg_lat = sum(latitudes) / len(latitudes)
    avg_lon = sum(longitudes) / len(longitudes)
    zoom_level = 8
else:
    avg_lat, avg_lon = -5.0, -60.0
    zoom_level = 6 

m = folium.Map(location=[avg_lat, avg_lon], zoom_start=zoom_level)

for placemark in extracted_placemarks:
    popup_text = f"<b>{placemark['name']}</b><br>{placemark['description']}"
    folium.Marker(
        location=[placemark['latitude'], placemark['longitude']],
        popup=popup_text,
        tooltip=placemark['name']
    ).add_to(m)

folium.GeoJson(
    geojson_data,
    name="Original GeoJSON Data",
    tooltip=folium.GeoJsonTooltip(fields=["Name", "description"]),
    popup=folium.GeoJsonPopup(fields=["Name", "description"])
).add_to(m)

folium.LayerControl().add_to(m)
m



heat_data = []
if extracted_placemarks:
    for placemark in extracted_placemarks:
        heat_data.append([placemark['latitude'], placemark['longitude'], 1])
else:
    print("No point data available from GeoJSON to create a heatmap.")

if latitudes and longitudes:
    avg_lat = sum(latitudes) / len(latitudes)
    avg_lon = sum(longitudes) / len(longitudes)
    zoom_level = 8
else:
    avg_lat, avg_lon = -5.0, -60.0
    zoom_level = 6

m_heatmap_only = folium.Map(location=[avg_lat, avg_lon], zoom_start=zoom_level)

if heat_data:
    HeatMap(heat_data, radius=15, blur=10).add_to(m_heatmap_only)
    folium.LayerControl(collapsed=False).add_to(m_heatmap_only)

legend_html = """
     <div style="position: fixed;
                 bottom: 50px; left: 50px; width: 220px; height: 110px;
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:white; opacity:0.9;
                 border-radius: 8px; padding: 10px;">
       &nbsp; <b>Geoglyph Density Heatmap</b> <br>
       &nbsp; <i style="background:linear-gradient(to right, #00FF00, #FFFF00, #FF0000);
                        width:20px; height:20px; display:inline-block; vertical-align: middle;
                        border-radius: 4px;"></i> &nbsp; Low Density &ndash; High Density <br>
       &nbsp; (More Geoglyphs = Redder Regions) <br>
       <hr style="margin: 5px 0;">
       <small>Toggle layers with the icon on top-right.</small>
     </div>
     """

legend_iframe = folium.Html(legend_html, script=True)
m_heatmap_only.get_root().html.add_child(legend_iframe)
m_heatmap_only


sns.set_theme(style="whitegrid")
processed_geoglyphs = []
size_pattern = re.compile(r'(\d+)m')
shape_pattern = re.compile(r'\b(Circle|Line|Square|Figure|Complex|Geoglyph|Oval|Rectangle)\b', re.IGNORECASE)

altitude_desc_pattern = re.compile(r'\n(-?\d+(\.\d+)?)\s*$')
if geojson_data and "features" in geojson_data:
    for feature in geojson_data["features"]:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})

        name = properties.get("Name", "Unnamed")
        description = str(properties.get("description") or "")
        
        coordinates = geometry.get("coordinates", [None, None, None])
        longitude = coordinates[0]
        latitude = coordinates[1]
        altitude_geom = coordinates[2] 
        geoglyph_size_m = None
        size_match = size_pattern.search(description)
        if size_match:
            try:
                geoglyph_size_m = int(size_match.group(1))
            except ValueError:
                pass

        geoglyph_shape_type = "Unknown"
        shape_match = shape_pattern.search(description)
        if shape_match:
            geoglyph_shape_type = shape_match.group(1).title()
        altitude_from_desc = None
        alt_desc_match = altitude_desc_pattern.search(description)
        if alt_desc_match:
            try:
                altitude_from_desc = float(alt_desc_match.group(1))
            except ValueError:
                pass
        final_altitude = altitude_geom
        if altitude_from_desc is not None and (altitude_geom is None or altitude_geom == 0):
            final_altitude = altitude_from_desc
        elif altitude_from_desc is not None and altitude_geom is not None and altitude_geom != 0 and abs(altitude_from_desc - altitude_geom) < 5: # Small difference, trust geometry more
            final_altitude = altitude_geom
        elif altitude_from_desc is not None:
            final_altitude = altitude_from_desc


        processed_geoglyphs.append({
            "Name": name,
            "Description_Raw": description,
            "Latitude": latitude,
            "Longitude": longitude,
            "Altitude_m": final_altitude,
            "Geoglyph_Size_m": geoglyph_size_m,
            "Geoglyph_Shape_Type": geoglyph_shape_type
        })

df = pd.DataFrame(processed_geoglyphs)


print("--- DataFrame Head ---")
print(df.head())
print("\n--- DataFrame Info ---")
df.info()
print("\n--- Descriptive Statistics for Numerical Columns ---")
print(df.describe())
print("\n--- Value Counts for Categorical Columns ---")
print(df['Geoglyph_Shape_Type'].value_counts())



txt_path = os.path.join(OUTPUT_DIR, "geoglyph_dataframe_summary.txt")

buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    print("--- DataFrame Head ---")
    print(df.head())
    print("\n--- DataFrame Info ---")
    df.info()
    print("\n--- Descriptive Statistics for Numerical Columns ---")
    print(df.describe())
    print("\n--- Value Counts for Categorical Columns ---")
    print(df["Geoglyph_Shape_Type"].value_counts())

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(buffer.getvalue())

print(f"\nâœ… Summary saved to {txt_path}")


plt.figure(figsize=(10, 6))
sns.histplot(df['Geoglyph_Size_m'].dropna(), kde=True, bins=20)
plt.title('Distribution of Geoglyph Sizes (m)')
plt.xlabel('Size (m)')
plt.ylabel('Count')
plt.savefig("Distribution of Geoglyph Sizes.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df['Altitude_m'].dropna(), kde=True, bins=20)
plt.title('Distribution of Geoglyph Altitudes (m)')
plt.xlabel('Altitude (m)')
plt.ylabel('Count')
plt.savefig("Distribution of Geoglyph Altitudes.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='Geoglyph_Shape_Type', order=df['Geoglyph_Shape_Type'].value_counts().index)
plt.title('Count of Geoglyph Shapes/Types')
plt.xlabel('Count')
plt.ylabel('Shape Type')
plt.savefig("Count of Geoglyph Shapes Types.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Geoglyph_Size_m', y='Altitude_m', hue='Geoglyph_Shape_Type', s=100, alpha=0.7)
plt.title('Geoglyph Size vs. Altitude (Colored by Shape Type)')
plt.xlabel('Geoglyph Size (m)')
plt.ylabel('Altitude (m)')
plt.legend(title='Shape Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("Geoglyph Size vs. Altitude_Colored by Shape Type.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure(figsize=(12, 7))
sns.boxplot(data=df, x='Geoglyph_Shape_Type', y='Geoglyph_Size_m')
plt.title('Geoglyph Size Distribution by Shape Type')
plt.xlabel('Shape Type')
plt.ylabel('Size (m)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("Geoglyph Size Distribution by Shape Type.png", dpi=300, bbox_inches="tight")
plt.show()


sns.pairplot(df[['Geoglyph_Size_m', 'Altitude_m', 'Geoglyph_Shape_Type']].dropna(), hue='Geoglyph_Shape_Type')
plt.suptitle('Pair Plot of Numerical Features by Shape Type', y=1.02)
plt.savefig("Pair Plot of Numerical Features by Shape Type.png", dpi=300, bbox_inches="tight")
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import folium
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

features_for_clustering = ['Latitude', 'Longitude', 'Altitude_m', 'Geoglyph_Size_m']
df_cluster = df[features_for_clustering].dropna().copy()

if df_cluster.empty:
    print("No complete data points for clustering after dropping NaNs. Cannot perform clustering.")
else:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cluster)

    df_scaled = pd.DataFrame(scaled_features, columns=features_for_clustering, index=df_cluster.index)


n_clusters = 5
print(f"\n--- Applying KMeans with {n_clusters} clusters ---")

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_cluster['Cluster'] = kmeans.fit_predict(scaled_features)

df.loc[df_cluster.index, 'Cluster'] = df_cluster['Cluster']
df['Cluster'] = df['Cluster'].fillna(-1).astype(int)

print("\n--- Cluster Distribution ---")
print(df['Cluster'].value_counts())


valid_latitudes = df['Latitude'].dropna().tolist()
valid_longitudes = df['Longitude'].dropna().tolist()

if valid_latitudes and valid_longitudes:
    avg_lat = sum(valid_latitudes) / len(valid_latitudes)
    avg_lon = sum(valid_longitudes) / len(valid_longitudes)
    zoom_level = 6
else:
    avg_lat, avg_lon = -5.0, -60.0 # default Amazon
    zoom_level = 4

m_clusters = folium.Map(location=[avg_lat, avg_lon], zoom_start=zoom_level)

colors = plt.cm.get_cmap('tab10', n_clusters + 1)
cluster_colors = {}
for i in range(n_clusters):
    cluster_colors[i] = mcolors.to_hex(colors(i))
cluster_colors[-1] = '#808080' # grey for unclustered points

cluster_names = {
    0: "Low Altitude, Small Features",
    1: "Mid-Range, Varied Sizes",
    2: "High Altitude, Larger Structures",
    3: "Coastal/Riverine Clusters",
    4: "Dense Central Grouping",
    -1: "Unclustered (Missing Data)"
}

for index, row in df.iterrows():
    if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
        cluster_id = row['Cluster']
        color = cluster_colors.get(cluster_id, '#000000')
        
        display_cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

        popup_text = f"<b>Name:</b> {row['Name']}<br>" \
                     f"<b>Shape:</b> {row['Geoglyph_Shape_Type']}<br>" \
                     f"<b>Size:</b> {row['Geoglyph_Size_m']}m<br>" \
                     f"<b>Altitude:</b> {row['Altitude_m']}m<br>" \
                     f"<b>Cluster:</b> {display_cluster_name}"

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=popup_text,
            tooltip=f"{row['Name']} ({display_cluster_name})"
        ).add_to(m_clusters)

legend_html = """
     <div style="position: fixed;
                 bottom: 50px; left: 50px; width: 220px; height: auto;
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:white; opacity:0.9;
                 border-radius: 8px; padding: 10px;">
       &nbsp; <b>Geoglyph Clusters (K-Means)</b> <br>
       <hr style="margin: 5px 0;">
       """
for cluster_id_key in sorted(cluster_colors.keys()):
    if cluster_id_key != -1:
        legend_html += f"""
               &nbsp; <i style="background:{cluster_colors[cluster_id_key]}; width:18px; height:18px;
                               display:inline-block; vertical-align: middle; border-radius: 50%;"></i>
               &nbsp; {cluster_names.get(cluster_id_key, f"Cluster {cluster_id_key}")} <br>
               """
if -1 in df['Cluster'].unique():
    legend_html += f"""
           &nbsp; <i style="background:{cluster_colors[-1]}; width:18px; height:18px;
                           display:inline-block; vertical-align: middle; border-radius: 50%;"></i>
           &nbsp; {cluster_names.get(-1, 'Unclustered (NaN Data)')} <br>
           """
legend_html += """
     </div>
     """

legend_iframe = folium.Html(legend_html, script=True)
m_clusters.get_root().html.add_child(legend_iframe)
m_clusters


df_num = df[['Geoglyph_Size_m', 'Altitude_m']].dropna()
scaled_num = scaler.fit_transform(df_num)

pca = PCA(n_components=2, random_state=42)
pca_proj = pca.fit_transform(scaled_num)
df_num['PC1'], df_num['PC2'] = pca_proj[:, 0], pca_proj[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_num.join(df[['Geoglyph_Shape_Type']]),
    x='PC1', y='PC2', hue='Geoglyph_Shape_Type', alpha=0.7, s=90
)
plt.title('PCA Projection of Geoglyph Size & Altitude')
plt.tight_layout()
plt.savefig("PCA Projection of Geoglyph Size & Altitude.png", dpi=300, bbox_inches="tight")
plt.show()

tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', random_state=42)
tsne_proj = tsne.fit_transform(scaled_num)
df_num['tSNE1'], df_num['tSNE2'] = tsne_proj[:, 0], tsne_proj[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_num.join(df[['Geoglyph_Shape_Type']]),
    x='tSNE1', y='tSNE2', hue='Geoglyph_Shape_Type', alpha=0.7, s=90
)
plt.title('tâ€‘SNE Projection of Geoglyph Size & Altitude')
plt.tight_layout()
plt.savefig("tâ€‘SNE Projection of Geoglyph Size & Altitude.png", dpi=300, bbox_inches="tight")
plt.show()


coords = df[['Latitude', 'Longitude']].dropna().to_numpy()
coords_rad = np.radians(coords)

kms_per_radian = 6371.0088
eps_km = 30
eps_rad = eps_km / kms_per_radian

db = DBSCAN(eps=eps_rad, min_samples=10, metric='haversine')
labels = db.fit_predict(coords_rad)

df.loc[df[['Latitude', 'Longitude']].dropna().index, 'SpatialCluster'] = labels
df['SpatialCluster'] = df['SpatialCluster'].fillna(-1).astype(int)

print(df['SpatialCluster'].value_counts().sort_index())

spatial_colors = plt.cm.get_cmap('Paired', len(set(labels)) + 1)
spatial_color_hex = {lab: mcolors.to_hex(spatial_colors(lab if lab >= 0 else -1))
                     for lab in set(labels)}
spatial_color_hex[-1] = "#808080"  # grey for noise

m_spatial = folium.Map(
    location=[df['Latitude'].mean(), df['Longitude'].mean()],
    zoom_start=6,
    tiles="OpenStreetMap"
)


for _, row in df.dropna(subset=['Latitude', 'Longitude']).iterrows():
    lab = row['SpatialCluster']
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=6,
        color=spatial_color_hex[lab],
        fill=True, fill_opacity=0.8,
        tooltip=f"{row['Name']} (SC {lab})"
    ).add_to(m_spatial)

# Minimal legend
legend = """<div style='position: fixed; bottom: 50px; left: 50px; 
             border:1px solid grey; padding:8px; background:white; z-index:9999;'>
             <b>Spatial Clusters</b><hr>"""
for lab, col in spatial_color_hex.items():
    legend += f"<i style='background:{col};width:12px;height:12px;display:inline-block;'></i>&nbsp;SC {lab}<br>"
legend += "</div>"
m_spatial.get_root().html.add_child(folium.Element(legend))
m_spatial


for name, fmap in {
    "geoglyph_pip_map": m,
    "geoglyph_heatmap": m_heatmap_only,
    "geoglyph_clusters": m_clusters,
    "geoglyph_spacial_clusters": m_spatial
}.items():
    try:
        fmap.save(os.path.join(OUTPUT_DIR, f"{name}.html"))
        print(f"âœ… Saved {name}.html")
    except Exception as e:
        print(f"âš ï¸�  Map save error ({name}): {e}")

