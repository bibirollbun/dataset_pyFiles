!pip install rasterio pydeck


import os
import requests
import math
from openai import OpenAI
import numpy as np
import pandas as pd
import polars as pl
import rasterio
import ee
import matplotlib.pyplot as plt


IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ


if IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    OPENAI_API_KEY = user_secrets.get_secret("OPENAI_API_KEY")
    OPENTOPO_API_KEY = user_secrets.get_secret("OPENTOPO_API_KEY")
    IAM_SERVICE_ACCOUNT = user_secrets.get_secret("IAM_SERVICE_ACCOUNT")
    EE_CREDENTIAL_PATH = user_secrets.get_secret("EE_CREDENTIAL_PATH")
    MAPBOX_API_KEY = user_secrets.get_secret("MAPBOX_API_KEY")
    os.environ["MAPBOX_API_KEY"] = MAPBOX_API_KEY
else:    
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENTOPO_API_KEY = os.environ["OPENTOPO_API_KEY"]
    IAM_SERVICE_ACCOUNT = os.environ["IAM_SERVICE_ACCOUNT"]
    EE_CREDENTIAL_PATH = os.environ["EE_CREDENTIAL_PATH"]

ee_creds = ee.ServiceAccountCredentials(IAM_SERVICE_ACCOUNT, EE_CREDENTIAL_PATH) # fetch your service account credentials
ee.Initialize(ee_creds) # initialize earth engine using your service account credentials


all_point_df = pd.read_csv("../input/archaeoblog/all_points.csv")
point_df = all_point_df.query("-10 <= latitude and latitude <= 3 and -75 <= longitude and longitude <= -55")


import pydeck as pdk

def plot_points(df: pd.DataFrame, is_satellite: bool = False) -> pdk.Deck:
    layer = pdk.Layer(
        'ScatterplotLayer',
        df,
        get_position=['longitude', 'latitude'],
        auto_highlight=True,
        get_radius=1000,  # Radius is given in meters
    get_fill_color=[240, 0, 200, 30], 
    pickable=True
    )

    view_state = pdk.ViewState(
        longitude=-70, 
        latitude=-10, 
        zoom=2,
    )
    # Render (I already set the MAPBOX_API_KEY in the environment variable)
    return pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state,
        map_provider="mapbox",
        map_style=pdk.map_styles.MAPBOX_SATELLITE if is_satellite else pdk.map_styles.MAPBOX_ROAD,
        height=800,
    )


plot_points(point_df, is_satellite=True)


sample_size = 5
sample_df = point_df.sample(sample_size, random_state=2025)
sample_df


BUFFER_METERS = 3000


import urllib
import io
from PIL import Image

def download_sentinel_satellite_image(coordinates: list[float], buffer_meters: int, output_file_name: str):
    """
    coordinates: list[float]  (longitude, latitude)
    buffer_meters: int  (meters)
    """
    if os.path.exists(output_file_name):
        print(f"File {output_file_name} already exists. Skipping download.")

    sugarloaf = ee.Geometry.Point(coordinates)
    region = sugarloaf.buffer(buffer_meters).bounds()

    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(sugarloaf) \
        .filterDate('2024-01-01', '2024-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .sort('CLOUDY_PIXEL_PERCENTAGE')

    sentinel = collection.first()

    if sentinel is None:
        raise ValueError("No suitable Sentinel-2 image found.")

    vis_params = {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0,
        'max': 3000,
        'gamma': 1.3
    }

    url = sentinel.getThumbURL({
        'region': region, 
        'dimensions': '800', 
        'format': 'jpg',
        'bands': vis_params['bands'],
        'min': vis_params['min'],
        'max': vis_params['max']
    })

    response = urllib.request.urlopen(url)
    img_data = response.read()
    img = Image.open(io.BytesIO(img_data))
    img = img.convert("RGB")  # Ensure the image is in RGB format
    img.save(output_file_name, format='JPEG')
    return img 


points = sample_df[['longitude', 'latitude']].to_numpy()
ids = sample_df["id"].to_numpy()


sentinel_imgs = []
for point, id in zip(points, ids):
    sentinel_imgs.append(download_sentinel_satellite_image(list(point), BUFFER_METERS, f"sentinel_{id}.png"))


fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, img in enumerate(sentinel_imgs):
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"{id=}")

plt.tight_layout()
plt.show()


import base64

def download_dem_tile(demtype:str, lat:float, lon:float, buffer_meters: float, output_file_name: str, api_key:str=OPENTOPO_API_KEY) -> None:
    """
    Downloads a DEM tile from OpenTopography using the specified parameters.
    """
    if os.path.exists(output_file_name):
        print(f"DEM data already exists: {output_file_name}")
        return

    # Define the OpenTopography API endpoint
    BASE_URL = "https://portal.opentopography.org/API/globaldem"

    # Parameters for the API request
    south, north, west, east = get_bbox_from_point(lat, lon, buffer_meters)
    params = {
        "demtype": demtype,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key
    }

    # Make the API request
    response = requests.get(BASE_URL, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        with open(output_file_name, "wb") as file:
            file.write(response.content)
        print(f"DEM data downloaded successfully: {output_file_name}")
    else:
        print(f"Failed to download data. Status code: {response.status_code}")
        print("Response:", response.text)


def get_bbox_from_point(lat:float, lon:float, buffer_m: float) -> tuple[float, float, float, float]:
    """
    Returns south, north, west, east coordinates from a center point (lat, lon) and buffer distance.

    Parameters:
    -----------
    lat : float
        Center latitude (degrees)
    lon : float
        Center longitude (degrees)
    buffer_m : float
        Buffer distance (meters)

    Returns:
    --------
    south, north, west, east : tuple of float
        Boundary coordinates (degrees)
    """
    R = 6378137.0  # Earth's equatorial radius in meters
    delta_lat = (buffer_m / R) * (180.0 / math.pi)
    lat_rad = math.radians(lat)
    delta_lon = (buffer_m / (R * math.cos(lat_rad))) * (180.0 / math.pi)

    south = lat - delta_lat
    north = lat + delta_lat
    west  = lon - delta_lon
    east  = lon + delta_lon
    return south, north, west, east


# 画像ファイルをbase64エンコード
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


DEM_TYPE = "COP30"


for point, id in zip(points, ids):
    lon, lat = point
    output_file_name = f"dem_{DEM_TYPE}_{id}.tif"
    download_dem_tile(
        demtype=DEM_TYPE,
        lat=lat,
        lon=lon,
        buffer_meters=BUFFER_METERS,
        output_file_name=output_file_name,
    )


fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, id in enumerate(ids):
    with rasterio.open(f"dem_{DEM_TYPE}_{id}.tif") as src:
        img = src.read(1)
    axes[i].imshow(img, cmap='terrain')
    axes[i].axis('off')
    axes[i].set_title(f"{id=}")
    img_pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    img_pil.save(f"dem_{DEM_TYPE}_{id}.png")
plt.tight_layout()
plt.show()


OPENAI_MODEL = "gpt-4.1-mini"


prompt = """
Provided data: 
  - Satellite images
  - Digital Surface Model (DSM) images
  - There is a known archaeological site located at the center of the image. 
Your Task: 
  - Analyze these images and detecting the anomaly features.
"""
client = OpenAI(
  api_key=OPENAI_API_KEY
)

contents = []
for id in ids:
    satellite_img_base64 = encode_image(f"sentinel_{id}.png")
    dem_tile_base64 = encode_image(f"dem_{DEM_TYPE}_{id}.png")
    contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{satellite_img_base64}"}})
    contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{dem_tile_base64}"}})

completion = client.chat.completions.create(
    model=OPENAI_MODEL,
    store=True,
    messages=[
      {"role": "user", "content": prompt},
      {"role": "user", "content": contents},
    ]
)
result = completion.choices[0].message.content


print(result)


prompt2 = f"""
How can these data be used to discover new archaeological sites from the following findings?
The user's thoughts are also attached, but you do not have to agree, just tell us what policy to take from an objective point of view.

## Findings
{result}

## User's thoughts.
Based on these data, I think the following steps in the future.
- Create a machine learning model for binary classification of the presence or absence of archaeological sites based on the features obtained from the images and DEM data near the sites.
- Areas predicted by the machine learning model to have a high probability of having archaeological sites are extracted, and information on literature and archaeological sites discovered in the past is given to the LLM to estimate the location of new archaeological sites.
"""


completion = client.chat.completions.create(
    model="gpt-4.1-mini",
    store=True,
    messages=[
      {"role": "user", "content": prompt2},
    ]
)
result2 = completion.choices[0].message.content


 print(result2)




