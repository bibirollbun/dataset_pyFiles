import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster


known_sites = pd.DataFrame({
    "lat": [-11.5247, -2.4383, -13.2500],
    "lon": [-53.1719, -54.6995, -65.1167],
    "elevation_anomaly": [1, 1, 1],
    "vegetation_disruption": ["square clearing", "mounds", "ditch system"],
    "near_water": [1, 1, 1]
})


lats = np.linspace(-13.5, -2.0, 10)
lons = np.linspace(-66.0, -53.0, 10)
grid_points = [(lat, lon) for lat in lats for lon in lons]


def simulate_tile_score(lat, lon):
    np.random.seed(int((lat + lon) * 1000) % 10000)
    return {
        "lat": lat,
        "lon": lon,
        "elevation_anomaly": np.random.choice([0, 1], p=[0.8, 0.2]),
        "vegetation_disruption": np.random.choice(["none", "square clearing", "linear patch"], p=[0.6, 0.3, 0.1]),
        "near_water": np.random.choice([0, 1], p=[0.5, 0.5])
    }



scored_tiles = pd.DataFrame([simulate_tile_score(lat, lon) for lat, lon in grid_points])

def vegetation_score(v):
    return {"none": 0, "linear patch": 0.5, "square clearing": 1.0}.get(v, 0)

scored_tiles["vegetation_score"] = scored_tiles["vegetation_disruption"].apply(vegetation_score)
scored_tiles["site_score"] = (
    0.4 * scored_tiles["elevation_anomaly"] +
    0.4 * scored_tiles["vegetation_score"] +
    0.2 * scored_tiles["near_water"]
)



top_predictions = scored_tiles.sort_values(by="site_score", ascending=False).head(5)



m = folium.Map(location=[-9.5, -58], zoom_start=5)
marker_cluster = MarkerCluster().add_to(m)

for _, row in top_predictions.iterrows():
    folium.Marker(
        location=[row["lat"], row["lon"]],
        popup=(f"Score: {row['site_score']:.2f}<br>"
               f"Vegetation: {row['vegetation_disruption']}<br>"
               f"Near Water: {bool(row['near_water'])}")
    ).add_to(marker_cluster)


output_html = "predicted_sites_map.html"
m.save(output_html)


print(top_predictions[["lat", "lon", "site_score", "vegetation_disruption", "near_water"]])



## ğŸ“¦ Load Data

import pandas as pd
sample_df = pd.read_csv('/kaggle/input/amazon-rainforest-anomaly-detection/amazon_anomalies_sample.csv')
sample_df.head()


import pandas as pd
import geopandas as gpd
from shapely import wkt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import MarkerCluster

# Load CSV
df = pd.read_csv('/kaggle/input/amazon-rainforest-anomaly-detection/amazon_anomalies_sample.csv')

# Convert 'geometry' from WKT to shapely
df['geometry'] = df['geometry'].apply(wkt.loads)
sample_df = gpd.GeoDataFrame(df, geometry='geometry')

# Run Isolation Forest
features = sample_df[['NDVI', 'NDWI', 'BAIS2', 'elevation']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

clf = IsolationForest(contamination=0.05, random_state=42)
sample_df['anomaly'] = clf.fit_predict(scaled)

# Create Folium map
m = folium.Map(location=[-4.0, -65.5], zoom_start=7)
marker_cluster = MarkerCluster().add_to(m)

for _, row in sample_df.iterrows():
    color = 'red' if row['anomaly'] == -1 else 'blue'
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=5,
        color=color,
        fill=True,
        fill_opacity=0.6,
        popup=f"NDVI: {row['NDVI']:.2f}, Elev: {row['elevation']}"
    ).add_to(marker_cluster)

# Save and display
m.save('amazon_anomalies_map.html')
m



import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt



# CSV with numerical and coordinate data
sample_df = pd.read_csv('/kaggle/input/amazon-rainforest-anomaly-detection/amazon_anomalies_sample.csv')

# GeoJSON for spatial analysis
geojson = gpd.read_file('/kaggle/input/amazon-rainforest-anomaly-detection/amazon_anomalies.geojson')



plt.figure(figsize=(8,6))
plt.scatter(sample_df['NDVI'], sample_df['elevation'], alpha=0.6)
plt.xlabel('NDVI')
plt.ylabel('Elevation')
plt.title('NDVI vs Elevation of Sampled Points')
plt.grid(True)
plt.show()




import folium
from folium.plugins import MarkerCluster
from shapely import wkt

# Convert 'geometry' column to Shapely objects
sample_df['geometry'] = sample_df['geometry'].apply(wkt.loads)

# Create base map
m = folium.Map(location=[-4.0, -65.5], zoom_start=7)
marker_cluster = MarkerCluster().add_to(m)

# Add points
for _, row in sample_df.iterrows():
    folium.CircleMarker(
        location=[row['geometry'].y, row['geometry'].x],  # correct order: [lat, lon]
        radius=4,
        color='red',
        fill=True,
        fill_opacity=0.7,
        popup=f"NDVI: {row['NDVI']:.3f}, Elev: {row['elevation']}"
    ).add_to(marker_cluster)

m.save("map_with_sample_df.html")
m




import folium
from folium.plugins import MarkerCluster
from shapely import wkt
import pandas as pd

# Load your CSV
df = pd.read_csv("/kaggle/input/amazon-rainforest-anomaly-detection/amazon_anomalies_sample.csv")

# Parse geometry into lat/lon
df['geometry'] = df['geometry'].apply(wkt.loads)

# Sort and get top 3 by NDVI
top3 = df.sort_values(by="NDVI", ascending=False).head(3)

# Create a folium map centered on average location
avg_lat = top3['geometry'].apply(lambda x: x.y).mean()
avg_lon = top3['geometry'].apply(lambda x: x.x).mean()
m = folium.Map(location=[avg_lat, avg_lon], zoom_start=9)

# Add top 3 markers
marker_cluster = MarkerCluster().add_to(m)

for i, row in top3.iterrows():
    folium.Marker(
        location=[row['geometry'].y, row['geometry'].x],
        popup=(
            f"<b>Top {i+1}</b><br>"
            f"NDVI: {row['NDVI']:.3f}<br>"
            f"Elevation: {row['elevation']}m"
        ),
        icon=folium.Icon(color='darkgreen', icon='leaf')
    ).add_to(marker_cluster)

# Save to HTML for interactive viewing
m.save("top3_ndvi_map.html")

# Display inline if in notebook
m



from openai import OpenAI
import pandas as pd

# Load your sample
sample_df = pd.read_csv('/kaggle/input/amazon-rainforest-anomaly-detection/amazon_anomalies_sample.csv')

# Extract coordinates
sample_df[['lon', 'lat']] = sample_df['geometry'].str.extract(r'POINT \(([-\d\.]+) ([-\d\.]+)\)').astype(float)

# ğŸ”� Your actual API key
client = OpenAI(api_key="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")  # Replace with your actual key

# Function to ask GPT-4o
def describe_anomaly(lat, lon, ndvi, elev):
    prompt = f"""
You are an expert in Amazonian archaeology and remote sensing. Analyze this geospatial point:
- Latitude: {lat}
- Longitude: {lon}
- NDVI: {ndvi}
- Elevation: {elev} meters

Is it likely to be a pre-Columbian site based on these features? Answer in 2â€“3 sentences.
"""
    try:
        chat = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return chat.choices[0].message.content
    except Exception as e:
        print(f"â�Œ GPT call failed at ({lat}, {lon}): {e}")
        return "Error"

# ğŸ›°ï¸� Top 3 points
top3 = sample_df.head(3).copy()
descriptions = []

for i, row in top3.iterrows():
    desc = describe_anomaly(row['lat'], row['lon'], row['NDVI'], row['elevation'])
    print(f"\nğŸ›°ï¸� Site {i+1}:\n{desc}\n")
    descriptions.append(desc)

# Save it
top3['gpt_description'] = descriptions
top3.to_csv('top3_gpt_site_descriptions.csv', index=False)
print("âœ… Descriptions saved to 'top3_gpt_site_descriptions.csv'")




pip install openai beautifulsoup4 PyPDF2 tiktoken



import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from io import BytesIO
import pandas as pd
import openai

# âœ… Set up OpenAI client
client = openai.OpenAI(api_key="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")  # use your actual key

# List of URLs
urls = [
    "https://books.google.com/books?hl=en&lr=&id=B4DSEAAAQBAJ&oi=fnd&pg=PP1&dq=amazon+lidar+archaeology&ots=oK0FItet27&sig=oFAGRog0cFkX9MooeDiaoRvVWzs#v=onepage&q=amazon%20lidar%20archaeology&f=false",
    "https://journal.caa-international.org/articles/10.5334/jcaa.48",
    "https://www.nature.com/articles/s41467-018-03510-7",
    "https://www.tandfonline.com/doi/full/10.1080/00934690.2025.2466877",
    "https://journal.caa-international.org/articles/10.5334/jcaa.45",
    "https://www.tandfonline.com/doi/abs/10.1080/01431161.2017.1295486",
    "https://www.nature.com/articles/s41586-022-04780-4",
    "https://www.science.org/doi/10.1126/science.ade2541",
    "https://www.tandfonline.com/doi/full/10.1080/2150704X.2022.2109942",
    "https://peerj.com/articles/15137/",
    "https://www.tandfonline.com/doi/full/10.1080/00934690.2017.1417198"
]

def extract_text_from_url(url):
    try:
        if url.endswith(".pdf"):
            response = requests.get(url)
            reader = PdfReader(BytesIO(response.content))
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        else:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"â�Œ Failed to extract from {url}: {e}")
        return ""

def summarize_text(text, url):
    try:
        prompt = (
            f"You are an expert in Amazonian archaeology. Summarize the key archaeological themes from this text (URL: {url}). "
            "Extract themes such as LiDAR use, settlement patterns, landscape transformation, or geoglyphs. "
            "Return 3â€“5 bullet points."
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You summarize archaeological research articles."},
                {"role": "user", "content": prompt + "\n\n" + text[:3000]}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"â�Œ GPT summarization failed: {e}"

results = []
for url in urls:
    print(f"\nğŸ“– Processing: {url}")
    text = extract_text_from_url(url)
    if text:
        summary = summarize_text(text, url)
        print(summary)
        results.append({'url': url, 'summary': summary})
    else:
        print("âš ï¸� No text extracted.")

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("archaeological_article_summaries.csv", index=False)






