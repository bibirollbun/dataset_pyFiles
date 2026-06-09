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

# Prueba una coordenada
explore_node(-8.5, -60.9, "Experimental Ritual Line")


# ğŸŒ�âœ¨ Canto CuÃ¡ntico del Umbral âœ¨ğŸŒ�
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

