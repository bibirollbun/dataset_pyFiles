!pip install odfpy folium geopy networkx


import pandas as pd
import folium
import re
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np


def dms_to_dd(dms):
    if pd.isnull(dms): return None
    match = re.match(r"(\d+)째([EWNS]) (\d+)' (\d+)", str(dms))
    if not match: return None
    deg, dir_, min_, sec = match.groups()
    dd = float(deg) + float(min_)/60 + float(sec)/3600
    if dir_ in ['S', 'W']:
        dd *= -1
    return dd

file_path = "/kaggle/input/brazilian-zooarch-database-zooarchbr/Fossile et al. Table 3 (Rev.SAB) - Archaeological Sites.ods"
df = pd.read_excel(file_path, engine="odf")

sites = []
for idx, row in df.iterrows():
    name = row['Archaeological site (*approximate coordinate)']
    lon_str = row['Datum SIRGAS2000 (*approximate coordinate)']
    lat_str = row['Unnamed: 6']
    lon = dms_to_dd(lon_str)
    lat = dms_to_dd(lat_str)
    if lon is not None and lat is not None:
        sites.append({'name': name, 'lat': lat, 'lon': lon})

m = folium.Map(location=[-14.2, -51.9], zoom_start=4)
for site in sites:
    folium.Marker(
        [site['lat'], site['lon']],
        popup=site['name']
    ).add_to(m)

m.save('brazil_sites_map.html')
m


file_path = "/kaggle/input/brazilian-zooarch-database-zooarchbr/Fossile et al. Table 3 (Rev.SAB) - Archaeological Sites.ods"
df = pd.read_excel(file_path, engine="odf")

def dms_to_dd(dms):
    import re
    if pd.isnull(dms): return None
    match = re.match(r"(\d+)째([EWNS]) (\d+)' (\d+)", str(dms))
    if not match: return None
    deg, dir_, min_, sec = match.groups()
    dd = float(deg) + float(min_)/60 + float(sec)/3600
    if dir_ in ['S', 'W']:
        dd *= -1
    return dd

coords = []
names = []
for idx, row in df.iterrows():
    name = row['Archaeological site (*approximate coordinate)']
    lon_str = row['Datum SIRGAS2000 (*approximate coordinate)']
    lat_str = row['Unnamed: 6']
    lon = dms_to_dd(lon_str)
    lat = dms_to_dd(lat_str)
    if lon is not None and lat is not None:
        coords.append([lat, lon])
        names.append(name)

coords = np.array(coords)

clustering = DBSCAN(eps=0.2, min_samples=2).fit(coords)

plt.figure(figsize=(10,8))
plt.scatter(coords[:,1], coords[:,0], c=clustering.labels_, cmap='tab10', s=100, label='Sites')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Archaeological Sites Clustering')
plt.colorbar(label='Cluster')
plt.show()

G = nx.Graph()
for i, name in enumerate(names):
    G.add_node(i, label=name, pos=(coords[i,1], coords[i,0]))

for i in range(len(coords)):
    for j in range(i+1, len(coords)):
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist < 0.5:  
            G.add_edge(i, j, weight=dist)

pos = {i: (coords[i,1], coords[i,0]) for i in range(len(coords))}
plt.figure(figsize=(12,10))
nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue')
plt.title('Graph of Archaeological Sites by Proximity')
plt.show()


unique_labels = set(clustering.labels_)
for label in unique_labels:
    if label == -1:
        continue  
    cluster_coords = coords[clustering.labels_ == label]
    centroid = cluster_coords.mean(axis=0)
    print(f"Possible new point near the cluster {label}: {centroid}")

