import requests
import uuid
import json
import math
from openai import OpenAI
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

OpenAI_key = user_secrets.get_secret("OpenAI_key")

# Registered on copernicus.eu and get keys  https://documentation.dataspace.copernicus.eu/Registration.html
# OpenAI and Copernicus  kays saved on  Kaggle Secrets 

CLIENT_ID = user_secrets.get_secret("Copernicus_CLIENT_ID")
CLIENT_SECRET = user_secrets.get_secret("Copernicus_CLIENT_SECRET")

# My own class to use  OpenAI models directly, without SDK and using retries with idempotancy key
# timeout 900  just need for using service_tier="flex" on some models

class OpenAIClient:
    def __init__(self, api_key, base_url="https://api.openai.com/v1", timeout=900, retries=3):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def _post(self, endpoint, data, idempotency_key=None):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key or str(uuid.uuid4()),
        }
        last_exception = None
        for attempt in range(self.retries):
            try:
                resp = requests.post(
                    url,
                    json=data,
                    headers=headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_exception = e
        raise last_exception

    def chat_completion(self, **kwargs):
        return self._post("chat/completions", kwargs)

    def embedding(self, **kwargs):
        return self._post("embeddings", kwargs)

    def responses(self, **kwargs):
        return self._post("responses", kwargs)

llm_client = OpenAIClient(api_key=OpenAI_key)


resp = llm_client.chat_completion(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Hello, world!"},
    ]
)
content = resp["choices"][0]["message"]["content"]
print(content)


emb = llm_client.embedding(
    model="text-embedding-3-small",
    input="Kaggle is awesome!"
)
print(list(emb["data"][0]["embedding"])[:5])


messages=[
    {
        "role": "user",
        "content": (
            "Please respond ONLY in strict JSON format, with no extra text. "
            "Create an object with the following three keys:\n"
            "1. \"city\" (string): any city name in the world,\n"
            "2. \"population\" (integer): an approximate population for that city,\n"
            "3. \"fun_fact\" (string): a fun fact about this city.\n"
            "Example format:\n"
            "{\n"
            "  \"city\": \"...\",\n"
            "  \"population\": ...,\n"
            "  \"fun_fact\": \"...\"\n"
            "}\n"
            "Return ONLY the JSON."
        )
    }
]

result = llm_client.responses(
        model="o4-mini-2025-04-16",
        input=messages,
        reasoning={"effort": "low"},
        #service_tier="flex",
        text={"format": {"type": "json_object"}},
)
content_raw = result["output"][1]["content"][0]["text"]
json.loads(content_raw)


from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# Your client credentials
client_id = CLIENT_ID
client_secret = CLIENT_SECRET

# Create a session
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                          client_secret=client_secret, include_client_id=True)

# All requests using this session will have an access token automatically added
resp = oauth.get("https://sh.dataspace.copernicus.eu/configuration/v1/wms/instances")
#print(resp.content)


lat = -5.693857905360785
lon = -61.230715571205344

size_km = 0.5  # 500 m

delta_lat = size_km / 111  # ~0.009 degrees
delta_lon = size_km / (111 * math.cos(math.radians(lat)))  # ~0.0125 degrees

min_lat = lat - delta_lat / 2
max_lat = lat + delta_lat / 2
min_lon = lon - delta_lon / 2
max_lon = lon + delta_lon / 2

bbox = [min_lon, min_lat, max_lon, max_lat]
print("bbox:", bbox)


access_token = token['access_token'] 

process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

payload = {
    "input": {
        "bounds": {
            "bbox":  bbox,  
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
            }
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
print(r.status_code)
with open("sentinel2_rgb.png", "wb") as f:
    f.write(r.content)



img = Image.open("sentinel2_rgb.png")
arr = np.array(img).astype(np.float32)

arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
arr = arr.astype(np.uint8)

plt.imshow(arr)
plt.axis('off')
plt.show()


payload = {
    "input": {
        "bounds": {
            "bbox": bbox,
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
            }
        }]
    },
    "output": {
        "width": 512,
        "height": 512,
        "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
    },
    "evalscript": """
//VERSION=3
function setup() {
  return {
    input: ["B08", "B04"],
    output: { bands: 1 }
  };
}
function evaluatePixel(sample) {
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  return [ndvi];
}
"""
}

r = requests.post(process_url, headers=headers, json=payload)
print(r.status_code)

with open("ndvi.png", "wb") as f:
    f.write(r.content)



img = Image.open("ndvi.png")
arr = np.array(img).astype(np.float32)

arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
arr = arr.astype(np.uint8)

plt.imshow(arr)
plt.axis('off')
plt.show()

