# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster

# Load the data (SAMPLED)
file_path = '/kaggle/input/historical-and-archeological-sites-in-brazil/translated_sites.csv'
df = pd.read_csv(file_path).sample(n=1000)

# Create a base map centered on Brazil
m = folium.Map(location=[-14.235004, -51.92528], zoom_start=4)

# Create a marker cluster to handle multiple markers
marker_cluster = MarkerCluster().add_to(m)

# Add markers for each site
for idx, row in df.iterrows():
    if pd.notna(row['latitude']) and pd.notna(row['longitude']):
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['site_name']}\n{row['site_summary']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)

# Save the map to an HTML file
m.save('brazil_sites_map.html')

# Display the map
m


