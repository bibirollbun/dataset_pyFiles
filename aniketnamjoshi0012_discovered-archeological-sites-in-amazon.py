pip install pandas folium


import pandas as pd
import folium
import numpy as np

# csv Loading
df = pd.read_csv('/kaggle/input/arch-sites/submit.csv')

# Creating a folium map centered in the Amazon
m = folium.Map(location=[-5, -63], zoom_start=5, tiles='CartoDB positron')

# Plotting all the points
for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row['y'], row['x']],
        radius=3,
        color='red',
        fill=True,
        fill_opacity=0.7,
        popup=f"Type: {row['type']}<br>Lat: {row['y']:.4f}, Lon: {row['x']:.4f}"
    ).add_to(m)


m.save('amazon_archaeo_sites_map.html')


m

