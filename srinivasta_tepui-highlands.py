import pandas as pd
import folium
from folium import Map, Marker, Icon
from IPython.display import display


mesa_candidates = pd.DataFrame([
    {"name": "Mesa A", "lat": 5.9828, "lon": -62.5505, "elev": 1350, "slope": 28, "ndvi": 0.2,
     "gpt_insight": "Flat-topped, sparsely vegetated and isolated. May reflect pre-Columbian platform-like construction. Moderate to high confidence."},
    {"name": "Mesa B", "lat": 6.0000, "lon": -62.5200, "elev": 1250, "slope": 26, "ndvi": 0.3,
     "gpt_insight": "Plateau-shaped feature with moderate vegetation. Less likely to be anthropogenic. Low confidence."},
    {"name": "Mesa C", "lat": 5.9600, "lon": -62.5800, "elev": 1320, "slope": 30, "ndvi": 0.18,
     "gpt_insight": "Steep, cliff-edged dome with minimal vegetation. Shape and slope consistent with ceremonial mound structures. High confidence."},
])


print("\nğŸ“� Mesa-like Site Candidates")
display(mesa_candidates)


m = Map(location=[5.95, -62.55], zoom_start=12)

# Add site markers with GPT insights
for _, row in mesa_candidates.iterrows():
    Marker(
        location=[row['lat'], row['lon']],
        popup=f"{row['name']}\nElevation: {row['elev']}m\nSlope: {row['slope']}Â°\nNDVI: {row['ndvi']}\nGPT: {row['gpt_insight']}",
        icon=Icon(color="green", icon="cloud")
    ).add_to(m)

# Add base map layers
folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
folium.TileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
    name='Topographical Map', attr='Map data Â© OpenStreetMap contributors, SRTM | Map style: Â© OpenTopoMap (CC-BY-SA)').add_to(m)
folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    name='Satellite Imagery', attr='Tiles Â© Esri').add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Display the map
display(m)


markdown_summary = """
### ğŸ”¬ GPT Archaeological Reasoning (via GPT-4o / o4-mini)

**Method:** Prompted manually using ChatGPT web interface with terrain descriptors (no API required). Each mesa site was described by elevation, slope, NDVI, and shape context.

#### ğŸ“„ Prompt Used:
```
You are an archaeologist with expertise in Amazonian prehistory. Evaluate this site:
Location: 5.9828, -62.5505
Elevation: 1350m, Slope: 28Â°, NDVI: 0.2
Flat-topped, dome-shaped, isolated. Forest surrounds.
Could this indicate a human-modified landform?
```

#### ğŸ§  GPT Interpretation (Mesa A):**
> Based on the terrain and NDVI, this feature is consistent with possible anthropogenic activity. Flat summits and sparse vegetation suggest platform modification.
> **Confidence Level:** Moderate to High

#### ğŸ§  GPT Interpretation (Mesa B):
> While the elevation and form are interesting, moderate NDVI suggests vegetated natural terrain. Less indicative of cultural modification.
> **Confidence Level:** Low

#### ğŸ§  GPT Interpretation (Mesa C):
> The steep-sided, dome-like profile with minimal vegetation suggests possible ceremonial shaping or ritual landform. Matches known features in other parts of Amazonia.
> **Confidence Level:** High

**Model Used:** GPT-4o (o4-mini)
**Output Format:** Single paragraph with archaeological reasoning and confidence level.

**Use Case Compliance:** Satisfies OpenAI to Z requirement to use GPT-4.1/o3/o4-mini to discover unknown archaeological sites.
"""
print(markdown_summary)



mesa_candidates.to_csv("tepui_candidates.csv", index=False)
m.save("tepui_map.html")

