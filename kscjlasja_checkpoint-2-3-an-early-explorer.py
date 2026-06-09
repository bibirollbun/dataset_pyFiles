import pandas as pd
import re
import base64
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import plotly.express as px
import plotly.graph_objects as go


#Display our top 5 anomalies based on deforestation
top5_anomalies= pd.read_csv("/kaggle/input/anomaly-points/anomaly_points.csv")
top5_anomalies


import openai

# Securely retrieve the OpenAI API key stored as a Kaggle secret
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("OPENAI_API_KEY")

# Create the OpenAI client using the retrieved key
client = openai.OpenAI(api_key=secret_value_0)


all_responses = []

for idx, row in top5_anomalies.iterrows():
    img_path = f"/kaggle/input/anomalies/Anomaly{idx+1}.png"
    
    try:
        with open(img_path, "rb") as img_file:
            base64_img = base64.b64encode(img_file.read()).decode()

        base_prompt = (
           "Analyze the deforested area shown in this Sentinel-2 image. "
           "Look for visible geometric patterns, ditches, circular clearings, or signs of human-modified landscape. "
           "Estimate a confidence score (0 to 1) that this could be a pre-Columbian anthropogenic feature."
        )
        
        full_prompt = (
            f"Anomaly {idx+1} — Latitude: {row['lat']:.5f}, Longitude: {row['lon']:.5f}, Radius: {row['radius_m']} m\n\n"
            + base_prompt
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]}
            ],
            max_tokens=500,
            temperature=0  # deterministic model for reproducibility
        )
        
        # Extract confidence score
        match = re.search(r'confidence.*?(\d*\.?\d+)', response.choices[0].message.content, re.IGNORECASE)
        if match:
            score = float(match.group(1))
        else:
            score = None
        
        top5_anomalies.loc[idx, 'gpt_confidence'] = score
        
        # Store full GPT text and image path
        all_responses.append((idx+1, row['lat'], row['lon'], response.choices[0].message.content, img_path))
    
    except FileNotFoundError:
        print(f"Image not found: {img_path} — skipping.")
    except Exception as e:
        print(f"Error on anomaly {idx+1}: {e}")



for anomaly_id, lat, lon, text, img_path in all_responses:
    print(f"Anomaly {anomaly_id} ({lat:.5f}, {lon:.5f})")
    print(text)
    print("="*80)
    
    img = Image.open(img_path)
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


top5_anomalies


# Build a location list from top 5 anomalies
location_lines = []
for idx, row in top5_anomalies.iterrows():
    location_lines.append(
        f"{idx+1}. Latitude: {row['lat']:.5f}, Longitude: {row['lon']:.5f}"
    )

# Combine into a single prompt
locations_text = "\n".join(location_lines)
question_prompt = (
    f"Here are five geographic locations in the southwestern Amazon with recent deforestation and detected anomalies:\n\n"
    f"{locations_text}\n\n"
    "Can you check if any of these locations are known in archaeological literature or records for pre-Columbian anthropogenic features "
    "such as geoglyphs, ring ditches, causeways, or anthropogenic soils? "
    "Please provide citations or known site names if available."
)

print(question_prompt)



response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": question_prompt}
    ],
    max_tokens=500,
    temperature=0
)

print(response.choices[0].message.content)



import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic

#Load the geoglyph dataset
geoglyphs1 = gpd.read_file("/kaggle/input/amazon-geoglyphs/amazon_geoglyphs.geojson")

#Load your top 5 anomaly points
anomalies = gpd.GeoDataFrame(
    top5_anomalies,
    geometry=gpd.points_from_xy(top5_anomalies.lon, top5_anomalies.lat),
    crs="EPSG:4326"
)

#Project anomalies and geoglyphs to EPSG:4326 (WGS 84) for distance calculation
anomalies_wgs84 = anomalies.to_crs(epsg=4326)
geoglyphs1_wgs84 = geoglyphs1.to_crs(epsg=4326)

# Extract lat/lon from geometry for anomalies and geoglyphs
anomalies_wgs84['lat'] = anomalies_wgs84.geometry.apply(lambda x: x.y if isinstance(x, Point) else x.centroid.y)
anomalies_wgs84['lon'] = anomalies_wgs84.geometry.apply(lambda x: x.x if isinstance(x, Point) else x.centroid.x)

geoglyphs1_wgs84['lat'] = geoglyphs1_wgs84.geometry.apply(lambda x: x.y if isinstance(x, Point) else x.centroid.y)
geoglyphs1_wgs84['lon'] = geoglyphs1_wgs84.geometry.apply(lambda x: x.x if isinstance(x, Point) else x.centroid.x)

# Loop through anomalies and calculate distance to the nearest geoglyph
for idx, anomaly in anomalies_wgs84.iterrows():
    anomaly_coords = (anomaly['lat'], anomaly['lon'])  # Anomaly coordinates (lat, lon)
    
    # Calculate distances to all geoglyphs
    distances = geoglyphs1_wgs84.apply(lambda row: geodesic(anomaly_coords, (row['lat'], row['lon'])).km, axis=1)
    
    # Find the minimum distance and the nearest geoglyph
    min_dist = distances.min()
    nearest_site = geoglyphs1_wgs84.iloc[distances.idxmin()]
    
    print(f"Anomaly {idx+1} is {min_dist:.2f} km from nearest geoglyph: {nearest_site['Name']}")



import cv2
import matplotlib.pyplot as plt
import glob

# Find all your NDVI overlay files and Canny edge files
overlay_paths = sorted(glob.glob('/kaggle/input/anomalies/*_NDVI.png'))
canny_paths = sorted(glob.glob('/kaggle/input/anomalies/*_CE.png'))

for ov_path, ce_path in zip(overlay_paths, canny_paths):
    # Load the images
    overlay_img = cv2.imread(ov_path)
    overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    canny_img = cv2.imread(ce_path, cv2.IMREAD_GRAYSCALE)

    # Plot side by side
    fig, ax = plt.subplots(1, 2, figsize=(14,7))

    ax[0].imshow(overlay_rgb)
    ax[0].set_title(f"{ov_path.split('/')[-1]} – NDVI overlay")
    ax[0].axis('off')

    ax[1].imshow(canny_img, cmap='gray')
    ax[1].set_title(f"{ce_path.split('/')[-1]} – Canny edges")
    ax[1].axis('off')

    plt.show()



top5_anomalies['gpt_confidence_filters'] = None

ce_paths = sorted(glob.glob("/kaggle/input/anomalies/*_CE.png"))

responses = []

for idx, img_path in enumerate(ce_paths):
    with open(img_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    prompt_text = (
        f"Analyze this Canny edge detection image (from anomaly {idx+1}). "
        "Focus on visible geometric patterns such as circles, squares, ditches, or lines "
        "that could suggest pre-Columbian earthworks or anthropogenic features. "
        "Estimate a confidence score (0 to 1) that this represents an anthropogenic feature."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
            ]}
        ],
        max_tokens=500
    )

    responses.append(response.choices[0].message.content)
    print(f"Model response for {img_path.split('/')[-1]}:\n{response.choices[0].message.content}")
    print("=" * 80)



# Loop through each stored response and apply regex to extract confidence score
for idx, text_response in enumerate(responses):
    # Attempt to extract confidence score using a more general pattern
    match = re.search(r'confidence.*?(\d*\.?\d+)', text_response, re.IGNORECASE)
    
    if match:
        score = float(match.group(1))
        
        # Store the score in the dataframe
        top5_anomalies.loc[idx, 'gpt_confidence_filters'] = score
    else:
        print(f"Could not extract confidence score from response for Anomaly {idx+1}")




top5_anomalies


pc= "/kaggle/input/anomalies/PC_-13.6370_-63.6492.png"
img = cv2.imread(pc)

# Convert BGR to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display
plt.figure(figsize=(10,8))
plt.imshow(img_rgb)
plt.title("PC_-13.6370_-63.6492.png")
plt.axis('off')
plt.show()


img_path = "/kaggle/input/anomalies/PC_-13.6370_-63.6492.png"
with open(img_path, "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

prompt_text = (
    "This image contains three panels of the same location:\n"
    "- The left panel is an RGB optical image.\n"
    "- The center panel is an NDVI vegetation index highlighting vegetation health.\n"
    "- The right panel is a Sobel edge-detection filter emphasizing geometric edges.\n\n"
    "Carefully analyze all three panels together. Look for geometric patterns such as squares, rectangles, "
    "circular clearings, ditches, or possible ancient causeways that could indicate a pre-Columbian "
    "anthropogenic earthwork. Estimate a confidence score from 0 to 1 that this location represents such "
    "an anthropogenic feature."
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
        ]}
    ],
    max_tokens=500
)

print(response.choices[0].message.content)



import base64

# Load your composite image (3 panels: RGB, NDVI, Sobel)
img_path = "/kaggle/input/anomalies/PC_-13.6370_-63.6492.png"
with open(img_path, "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# Build the explicit positive control prompt
prompt_text = (
    "This image shows three panels of the SAME geographic location, which is a known, documented "
    "pre-Columbian archaeological site used here as a positive control:\n"
    "- The LEFT panel is an RGB optical image.\n"
    "- The CENTER panel is an NDVI vegetation index highlighting vegetation health.\n"
    "- The RIGHT panel is a Sobel edge-detection filter emphasizing geometric edges.\n\n"
    "Because we already know this location contains anthropogenic earthworks, please analyze how clearly "
    "these features are visible across the three panels. Then provide an estimated confidence score from 0 to 1 "
    "representing how strongly this imagery supports the presence of pre-Columbian geometric structures."
)

# Send to GPT
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
        ]}
    ],
    max_tokens=500
)

print(response.choices[0].message.content)



top5_anomalies['gpt_confidence_sobel'] = None

analysis_paths = sorted(glob.glob("/kaggle/input/anomalies/analysis_anomaly*.png"))

print(f"Found {len(analysis_paths)} images for analysis.")

responses = []

for idx, img_path in enumerate(analysis_paths):
    with open(img_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    prompt_text = (
        "This image contains three panels of the same location:\n"
        "- The left panel is an RGB optical image.\n"
        "- The center panel is an NDVI vegetation index highlighting vegetation health.\n"
        "- The right panel is a Sobel edge-detection filter emphasizing geometric edges.\n\n"
        "Carefully analyze all three panels together. Look for geometric patterns such as squares, rectangles, "
        "circular clearings, ditches, or possible ancient causeways that could indicate a pre-Columbian "
        "anthropogenic earthwork. Estimate a confidence score from 0 to 1 that this location represents such "
        "an anthropogenic feature."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
            ]}
        ],
        max_tokens=500
    )

    text_response = response.choices[0].message.content
    responses.append(text_response)

    print(f"Analysis for {img_path.split('/')[-1]}")
    print(text_response)
    print("="*80)



for idx, text_response in enumerate(responses):
    match = re.search(
        r'confidence.*?(?:score)?\s.*?(?:estimated|approximately)?\s.*?(\d*\.?\d+)',
        text_response, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        top5_anomalies.loc[idx, "gpt_confidence_sobel"] = score
    else:
        print(f"Could not extract confidence score from response for Anomaly {idx+1}")
        print(f"Response text: {text_response}")



top5_anomalies


top5_anomalies.to_csv("top5_anomalies_scores.csv")


fig = go.Figure()

fig.add_trace(go.Bar(
    x=top5_anomalies["id"],
    y=top5_anomalies["gpt_confidence"],
    name="GPT Confidence"
))

fig.add_trace(go.Bar(
    x=top5_anomalies["id"],
    y=top5_anomalies["gpt_confidence_filters"],
    name="GPT Confidence Filters"
))

fig.add_trace(go.Bar(
    x=top5_anomalies["id"],
    y=top5_anomalies["gpt_confidence_sobel"],
    name="GPT Confidence Sobel"
))

fig.update_layout(
    title="Grouped Bar Chart of GPT Confidence Scores by Anomaly",
    xaxis_title="Anomaly ID",
    yaxis_title="Confidence Score",
    barmode="group",  # groups them side by side
    yaxis=dict(range=[0, 1])  # optional: keep scale consistent for confidence
)

fig.show(renderer='iframe')



top5_anomalies = top5_anomalies.dropna(subset=['gpt_confidence_sobel'])
# Heatmap using Density Mapbox
fig = px.density_mapbox(top5_anomalies, 
                        lat='lat', 
                        lon='lon', 
                        z='gpt_confidence_sobel',
                        radius=10, 
                        center={"lat": top5_anomalies['lat'].mean(), "lon": top5_anomalies['lon'].mean()},
                        zoom=5, 
                        mapbox_style="carto-positron",
                        title="Heatmap of Confidence Scores Across Anomalies (Sobel Filter)"
                       )
fig.show(renderer='iframe')

