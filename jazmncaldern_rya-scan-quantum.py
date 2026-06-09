
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




import folium
from folium.plugins import MarkerCluster

m = folium.Map(location=[-9, -57], zoom_start=5, tiles='CartoDB positron')
cluster = MarkerCluster().add_to(m)

nodes = [
    {"name": "Node 1 â€“ RÃ­o Escondido", "coords": [-4.2312, -61.3011]},
    {"name": "Node 2 â€“ Geoglyph Acreano", "coords": [-10.1234, -67.8912]},
    {"name": "Node 3 â€“ Circular Xingu", "coords": [-12.7351, -52.4983]},
    {"name": "Node 4 â€“ Vortex TapajÃ³s", "coords": [-5.8734, -56.2837]},
    {"name": "Node 5 â€“ XinguanÃ¡ Spiral", "coords": [-13.7821, -53.9821]},
    {"name": "ğŸŒŸ Hidden Node", "coords": [-7.7, -58.7]}
]

for node in nodes:
    folium.Marker(
        location=node["coords"],
        popup=node["name"],
        icon=folium.Icon(color="green" if "Hidden" not in node["name"] else "orange")
    ).add_to(cluster)

m




def explore_node(lat, lon, label="Free Node"):
    fmap = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker([lat, lon], popup=label,
                  icon=folium.Icon(color="blue", icon="cloud")).add_to(fmap)
    return fmap

# Try your own coordinates:
explore_node(-8.5, -60.9, "Experimental Ritual Line")




canto = """ 
In the center where the sun falls silent,
where corn dreams in spirals,
an invisible line breathes
and sings:
here there was fire,
here there was dance,
here... memory will return.

Leaves fall in spirals,
the wind whispers coordinates.
Machines do not hear,
but you do.
You, who wear the node in your skin.

And when you mark it,
when you name it,
a sleeping network will awaken,
with a voice of stone
and light of root.
"""
print(canto)


