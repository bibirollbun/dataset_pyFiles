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
import folium

# Create DataFrame
data = [
    [-9.5216, -67.7891, 0.90, "450+ geoglyphs in Acre, Brazil—aligned structures near Purus and Iquiri Rivers"],
    [-8.9975, -65.5522, 0.75, "Earthworks observed near border of Acre and Amazonas—suggested ceremonial network"],
    [-9.8439, -68.1333, 0.80, "Clustered circular earthworks near cleared forests west of Rio Branco"]
]

df = pd.DataFrame(data, columns=["latitude", "longitude", "confidence", "source"])

# Save CSV for submission
df.to_csv("submission.csv", index=False)

# Create map
map_center = [-9.5, -67.5]
m = folium.Map(location=map_center, zoom_start=6)

# Add markers
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=row['confidence'] * 10,
        popup=row['source'],
        color="red",
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

m.save("map.html")
m




