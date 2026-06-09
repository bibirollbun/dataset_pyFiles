import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import openai
import requests
from PIL import Image
from kaggle_secrets import UserSecretsClient
from openai import OpenAI
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session


COPERNICUS_CLIENT_ID = UserSecretsClient().get_secret("COPERNICUS_CLIENT_ID")
COPERNICUS_CLIENT_SECRET = UserSecretsClient().get_secret("COPERNICUS_CLIENT_SECRET")


@dataclass
class Site:
    name: str
    lat: float
    lon: float
    path_google_earth_raw: str = ""
    path_google_earth_annotation: str = ""
    path_sentinel: str = ""


sites = (
    Site(
        name="assis",
        lat=-10.910099,
        lon=-69.493066,
        path_google_earth_raw="../input/google-earth/assis_raw.jpg",
        path_google_earth_annotation="../input/google-earth/assis.jpg"
    ),
    Site(
        name="assiv & mouv11",
        lat=-10.911896,
        lon=-69.532982,
        path_google_earth_raw="../input/google-earth/mouv11_assiv_raw.jpg",
        path_google_earth_annotation="../input/google-earth/mouv11_assiv.jpg"
    ),
    Site(
        name="acds2 & acrq58",
        lat=-9.758053,
        lon=-67.194505,
        path_google_earth_raw="../input/google-earth/acds2_acrq58_raw.jpg",
        path_google_earth_annotation="../input/google-earth/acds2_acrq58.jpg"
    ),
    Site(
        name="random1",
        lat=-9.8132874,
        lon=-67.4394589,
        path_google_earth_raw="../input/google-earth/random1.jpg",
    ),
    Site(
        name="random2",
        lat=-9.8108195,
        lon=-67.8182464,
        path_google_earth_raw="../input/google-earth/random2.jpg",
    ),
    Site(
        name="clearing",
        lat=-9.8467201,
        lon=-68.1594539,
        path_google_earth_raw="../input/google-earth/clearing.jpg",
    )
)


def get_bbox(lat, lon):    
    size_km = 0.5
    
    delta_lat = size_km / 111
    delta_lon = size_km / (111 * math.cos(math.radians(lat)))
    
    min_lat = lat - delta_lat / 2
    max_lat = lat + delta_lat / 2
    min_lon = lon - delta_lon / 2
    max_lon = lon + delta_lon / 2
    
    return [min_lon, min_lat, max_lon, max_lat]


def write_sentinel_image_org(lat, lon):
    client = BackendApplicationClient(client_id=COPERNICUS_CLIENT_ID)
    oauth = OAuth2Session(client=client)

    process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    token = oauth.fetch_token(
        token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        client_secret=COPERNICUS_CLIENT_SECRET,
        include_client_id=True
    )
    access_token = token['access_token']
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "bounds": {
                "bbox": get_bbox(lat, lon),
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": "2024-01-01T00:00:00Z",
                        "to": "2025-05-20T00:00:00Z"
                    },
                    "maxCloudCoverage": 30
                },
            }]
        },
        "output": {
            "width": 1024,  
            "height": 1024,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
        },
        "evalscript": """
            //VERSION=3
            function setup() {
                return {
                    input: ["B04", "B03", "B02"],
                    output: { bands: 3 }
                };
            }
            function evaluatePixel(sample) {
                return [sample.B04, sample.B03, sample.B02];
            }
        """
    }
    
    r = requests.post(process_url, headers=headers, json=payload)
    if r.status_code != 200:
        raise RuntimeError("Something went wrong")
    path = f"sentinel2_rgb_{abs(lat)}_{abs(lon)}.png"
    with open(path, "wb") as f:
        f.write(r.content)
    print(f"{path} created")
    return path


def write_sentinel_image(lat, lon):
    client = BackendApplicationClient(client_id=COPERNICUS_CLIENT_ID)
    oauth = OAuth2Session(client=client)

    process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    token = oauth.fetch_token(
        token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        client_secret=COPERNICUS_CLIENT_SECRET,
        include_client_id=True
    )
    access_token = token['access_token']
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "bounds": {
                "bbox": get_bbox(lat, lon),
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": "2024-01-01T00:00:00Z",
                        "to": "2025-05-20T00:00:00Z"
                    },
                    "maxCloudCoverage": 10
                },
                "processing": {
                    "mosaicking": "leastCC"
                }
            }]
        },
        "output": {
            "width": 50,  
            "height": 50,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
        },
        "evalscript": """
            //VERSION=3
            function setup() {
                return {
                    input: ["B04", "B03", "B02"],
                    output: { bands: 3 }
                };
            }
            function evaluatePixel(sample) {
                return [sample.B04 * 3, sample.B03 * 3, sample.B02 * 3];
            }
        """
    }
    
    r = requests.post(process_url, headers=headers, json=payload)
    if r.status_code != 200:
        raise RuntimeError("Something went wrong")
    path = f"sentinel2_rgb_{abs(lat)}_{abs(lon)}.png"
    with open(path, "wb") as f:
        f.write(r.content)
    print(f"{path} created")
    return path
        


def create_file(path, client):
    with open(path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id

def get_openai_response(path):
    client = OpenAI(api_key=UserSecretsClient().get_secret("OPENAI_API_KEY"))
    file_id = create_file(path, client)
    system_message = """You are an expert in archaeological image analysis.

When given a satellite image:
- Analyze it carefully.
- Describe **only** features that are clearly visible and could reasonably indicate past human activity, such as:
  - Geometric clearings
  - Raised fields
  - Ditches
  - Other signs of land modification

Guidelines:
- Do not fabricate details or make speculative assumptions.
- Base your response **solely on observable evidence** in the image.
- If no such features are evident, **clearly state that**.
- Be **objective**, **precise**, and **concise** in your analysis."""

    user_message = """Please analyze the uploaded satellite image.

Focus on identifying any visible features that may suggest past human activity, such as:
- Geometric clearings
- Raised fields
- Ditches
- Other land modifications

Instructions:
- Base your observations **strictly on the visual evidence** in the image.
- **Do not use or assume** any prior knowledge about the location.
- Avoid speculation — describe **only what is clearly observable**.

Finally, output the probability that the image shows an archaeological site.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_message
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_message
                    },
                {
                  "type": "input_image",
                  "file_id": file_id
                }
              ]
            }
        ],
        text={
            "format": {
              "type": "text"
            }
        },
        reasoning={},
        tools=[],
        temperature=0.3,
        max_output_tokens=2048,
        top_p=1,
        store=True
    )
    return response


def go(site, func):
    print("-" * 80)
    print()
    print(site)

    site.path_sentinel = func(site.lat, site.lon)
    img = Image.open(site.path_sentinel)
    plt.imshow(img)
    plt.title("Sentinel")
    plt.show()

    openai_response = get_openai_response(site.path_sentinel)
    print("OpenAI response to Sentinel:", openai_response.output_text)

    if site.path_google_earth_raw:
        img = Image.open(site.path_google_earth_raw)
        plt.imshow(img)
        plt.title("Google Earth Raw")
        plt.show()
        openai_response = get_openai_response(site.path_google_earth_raw)
        print("OpenAI response to Google Earth:", openai_response.output_text)

    if site.path_google_earth_annotation:
        img = Image.open(site.path_google_earth_annotation)
        plt.imshow(img)
        plt.title("Google Earth Annotation")
        plt.show()


for site in sites:
    go(site, write_sentinel_image_org)


for site in sites:
    go(site, write_sentinel_image)

