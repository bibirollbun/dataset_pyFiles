import time
import requests
from bs4 import BeautifulSoup
import re
import spacy
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
WIKI_CATEGORIES = [
    "Archaeological_sites_in_Brazil",
    "Archaeological_sites_in_Peru",
    "Archaeological_sites_in_Ecuador",
    "Archaeological_sites_in_Bolivia"
]

CATEGORY_TO_COUNTRY = {
    "Archaeological_sites_in_Brazil": "BR",
    "Archaeological_sites_in_Peru": "PE",
    "Archaeological_sites_in_Ecuador": "EC",
    "Archaeological_sites_in_Bolivia": "BO"
}

# Define geographical keywords as a set for efficient lookup
GEO_KEYWORDS = {
    "river", "tributary", "island", "plateau", "floodplain", "lake", "forest",
    "terra preta", "geoglyph", "savanna", "hill", "mountain", "stream", "creek",
    "valley", "basin", "plain", "jungle", "delta"
}

# --- INITIALIZE NLP ---
nlp = spacy.load("en_core_web_sm")


def get_sites_from_wikipedia_category(category_name):
    sites = []
    url = f"https://en.wikipedia.org/wiki/Category:{category_name}"
    while url:
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        for li in soup.select(".mw-category-group ul li"):
            a = li.find("a")
            if a and a.has_attr('href') and a['href'].startswith("/wiki/") \
                and not a['href'].startswith("/wiki/Category:") \
                and not a['href'].startswith("/wiki/File:"):
                title = a.text
                link = "https://en.wikipedia.org" + a['href']
                sites.append((title, link))
        next_link = soup.find("a", string="next page")  # <- use string, not text
        url = "https://en.wikipedia.org" + next_link['href'] if next_link else None
    return sites


def get_wikipedia_summary(site_url):
    title = site_url.split('/wiki/')[-1]
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    r = requests.get(summary_url)
    if r.status_code == 200:
        return r.json().get('extract', "")
    return ""


def extract_discovery_date(text):
    match = re.search(r'(discover|excavat|unearthed|found|identified)[^\d]{0,50}(\d{4})',
                      text, re.IGNORECASE)
    if match:
        return match.group(2)
    first_two = " ".join(text.split('.')[:2])
    match = re.search(r'(\d{4})', first_two)
    return match.group(1) if match else None


def extract_geo_keywords(text):
    text_lower = text.lower()
    doc = nlp(text_lower)
    found = set()

    # Single-word keywords via token lemma
    for token in doc:
        if token.lemma_ in GEO_KEYWORDS:
            found.add(token.lemma_)

    # Multi-word keywords directly in text
    for kw in (kw for kw in GEO_KEYWORDS if " " in kw):
        if kw in text_lower:
            found.add(kw)

    return list(found)


def get_lat_lon(site_url):
    """Extract latitude and longitude from a Wikipedia article's infobox, if available."""
    resp = requests.get(site_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    geo = soup.find("span", class_="geo")
    if geo:
        try:
            lat_str, lon_str = geo.text.strip().split(";")
            lat = float(lat_str)
            lon = float(lon_str)
            return lat, lon
        except Exception:
            pass
    # Alternate method: check for decimal values in class 'latitude' and 'longitude'
    lat_tag = soup.find("span", class_="latitude")
    lon_tag = soup.find("span", class_="longitude")
    if lat_tag and lon_tag:
        try:
            # If degrees, minutes, seconds are present, fallback to None (parsing needed)
            lat = float(lat_tag.text.strip())
            lon = float(lon_tag.text.strip())
            return lat, lon
        except Exception:
            pass
    return None, None



def geocode_with_nominatim(site_name, country_code=None):
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": site_name,
        "format": "json",
        "limit": 1,
    }
    if country_code:
        params["countrycodes"] = country_code.lower()
    # Respect usage policy
    time.sleep(1)
    response = requests.get(base_url, params=params, headers={'User-Agent': 'ArchaeologyScript/1.0'})
    if response.status_code == 200:
        results = response.json()
        if results:
            lat = float(results[0]['lat'])
            lon = float(results[0]['lon'])
            return lat, lon
    return None, None


from tqdm import tqdm
import time
    
all_records = []
all_geo_terms = []

for category in WIKI_CATEGORIES:
    print(f"Processing category: {category}")
    sites = get_sites_from_wikipedia_category(category)
    country_code = CATEGORY_TO_COUNTRY.get(category, None)
    for name, url in tqdm(sites, desc=f"Processing {category}"):
        summary = get_wikipedia_summary(url)
        discovery = extract_discovery_date(summary)
        geo_terms = extract_geo_keywords(summary)
        lat, lon = get_lat_lon(url)
        if lat is None or lon is None:
            lat, lon = geocode_with_nominatim(name, country_code)
        all_geo_terms.extend(geo_terms)
        all_records.append({
            "site_name": name,
            "wiki_url": url,
            "discovery_year": discovery,
            "lat": lat,
            "lon": lon,
            "geo_terms": geo_terms,
            "summary_snippet": summary[:200] + "â€¦"
        })


# Build DataFrame
df = pd.DataFrame(all_records)
df


freq = Counter(all_geo_terms)
wc = WordCloud(width=800, height=400).generate_from_frequencies(freq)
plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Geographical Feature Keywords in Amazon Archaeology Summaries")
plt.show()


import folium

map_filename="amazon_archaeology_map.html"
# Drop rows where lat or lon is missing
map_df = df.dropna(subset=['lat', 'lon'])

# Center the map around the mean of your coordinates (Amazon region)
map_center = [map_df['lat'].mean(), map_df['lon'].mean()]
fmap = folium.Map(location=map_center, zoom_start=5, tiles='OpenStreetMap', attr='Map tiles by Stamen Design, CC BY 3.0 â€” Map data Â© OpenStreetMap contributors'
)

# Add a marker for each site
for _, row in map_df.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=folium.Popup(f"<b>{row['site_name']}</b><br><a href='{row['wiki_url']}' target='_blank'>Wikipedia</a>", max_width=300),
        tooltip=row['site_name']
    ).add_to(fmap)

# Save to HTML file
fmap.save(map_filename)
print(f"Map saved to: {map_filename}")



fmap


# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import networkx as nx

# Assume map_df is your DataFrame with columns: 'site_name', 'lat', 'lon', etc.

# Extract coordinates as numpy array
coords = map_df[['lat', 'lon']].values

# Perform DBSCAN clustering
clustering = DBSCAN(eps=0.5, min_samples=2).fit(coords)
map_df['cluster'] = clustering.labels_

# ===== FIX: Reset the DataFrame index so idx matches coords and NetworkX nodes =====
map_df = map_df.reset_index(drop=True)

# --- Scatter Plot of Clusters ---
plt.figure(figsize=(10, 8))
scatter = plt.scatter(map_df['lon'], map_df['lat'], c=map_df['cluster'], cmap='tab10', s=100, edgecolor='k')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Archaeological Sites Clustering')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()

# --- Create Graph Based on Proximity ---
G = nx.Graph()
for idx, row in map_df.iterrows():
    G.add_node(idx, label=row['site_name'], pos=(row['lon'], row['lat']))

# Add edges for sites within 0.5 degrees (~55km)
for i in range(len(coords)):
    for j in range(i + 1, len(coords)):
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist < 0.5:
            G.add_edge(i, j, weight=dist)

# Plot the Graph
pos = nx.get_node_attributes(G, 'pos')
plt.figure(figsize=(12, 10))
nx.draw(G, pos, labels=nx.get_node_attributes(G, 'label'), with_labels=True,
        node_size=300, node_color='skyblue', font_size=10, edge_color='gray')
plt.title('Graph of Archaeological Sites by Proximity')
plt.show()

# --- Calculate and Print Cluster Centroids ---
unique_labels = set(clustering.labels_)
for label in unique_labels:
    if label == -1:
        continue  # Skip noise
    cluster_coords = coords[clustering.labels_ == label]
    centroid = cluster_coords.mean(axis=0)
    print(f"Cluster {label} Centroid: Latitude {centroid[0]:.5f}, Longitude {centroid[1]:.5f}")
#


!pip install uv
!uv pip install rasterio pystac_client


import numpy as np
import rasterio
from pystac_client import Client
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

def download_and_plot_sentinel_rgb_truecolor(lat, lon, site_name, out_folder, days_back=720, box_size_deg=0.01):
    import numpy as np
    import rasterio
    from pystac_client import Client
    import matplotlib.pyplot as plt
    from pathlib import Path
    import datetime

    today = datetime.date.today()
    start = today - datetime.timedelta(days=days_back)
    bbox = [lon - box_size_deg, lat - box_size_deg, lon + box_size_deg, lat + box_size_deg]
    cat = Client.open("https://earth-search.aws.element84.com/v1")
    items = sorted(
        cat.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start}/{today}",
            query={"eo:cloud_cover": {"lt": 20}},
            max_items=10
        ).items(),
        key=lambda it: it.properties["eo:cloud_cover"]
    )
    if not items:
        print(f"No Sentinel-2 scene found for {site_name}")
        return None
    scene = items[0]
    # Band asset URLs
    PREFS = {
      "B04": ["B04","B04_10m","red"],
      "B03": ["B03","B03_10m","green"],
      "B02": ["B02","B02_10m","blue"],
    }
    def pick_asset(item, band):
        for p in PREFS[band]:
            if p in item.assets:
                return item.assets[p].href
        for k in item.assets:
            if band.lower().lstrip("b0") in k.lower():
                return item.assets[k].href
        raise KeyError(f"{band} not found in {item.id}")
    urls = {b: pick_asset(scene, b) for b in ("B04","B03","B02")}
    # Reference shape/grid
    with rasterio.open(urls["B04"]) as src_ref:
        dst_h = src_ref.height
        dst_w = src_ref.width
        transform = src_ref.transform
        crs = src_ref.crs
    # Read and stack
    rgb = np.zeros((3, dst_h, dst_w), dtype=np.uint8)
    for idx, band in enumerate(["B04", "B03", "B02"]):
        with rasterio.open(urls[band]) as src:
            data = src.read(1).astype("float32") / 10000.0
            data = np.clip(data, 0, 1)
            rgb[idx] = (data * 255).astype("uint8")
    S2_PATH = Path(out_folder)/f"{site_name}_sentinel_truecolor.tif"
    meta = {
        "driver":   "GTiff",
        "height":   dst_h,
        "width":    dst_w,
        "count":    3,
        "dtype":    "uint8",
        "crs":      crs,
        "transform":transform,
        "compress": "lzw"
    }
    with rasterio.open(S2_PATH, "w", **meta) as dst:
        dst.write(rgb)
    # Plot as RGB
    rgb_plot = np.transpose(rgb, (1, 2, 0))  # shape (height, width, 3)
    plt.imshow(rgb_plot)
    plt.title(site_name)
    plt.axis('off')
    plt.show()
    print("âœ” Wrote", S2_PATH)
    return S2_PATH


# Example for a single site (use a loop for all)
lat, lon = map_df.loc[0, 'lat'], map_df.loc[0, 'lon']
site_name = map_df.loc[0, 'site_name']
download_and_plot_sentinel_rgb_truecolor(lat, lon, site_name, '/kaggle/working')

