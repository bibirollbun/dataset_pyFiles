import ee
from openai import OpenAI

import base64
import io
import os
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image


def load_secret(name):
    """Loads secret from Colab/Kaggle."""

    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(name)
        except Exception:
            pass 
    else:
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception: 
            pass

    return 'Secret not found'


iam_service_account = load_secret('iam_service_account') # the address of your project's IAM service account
ee_credentials_json = load_secret('ee_credentials') # the file path for the JSON file containing the relevant credentials
ee_creds = ee.ServiceAccountCredentials(iam_service_account, ee_credentials_json) # fetch your service account credentials
ee.Initialize(ee_creds) # initialize earth engine using your service account credentials


openai_key = load_secret('openai_key_2025')
client = OpenAI(
  api_key=openai_key
)


coordinates = [-43.1566, -22.9486]
sugarloaf = ee.Geometry.Point(coordinates)
region = sugarloaf.buffer(3000).bounds()

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

plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.title('Sugarloaf Mountain - Sentinel-2 Image')
plt.axis('off')
plt.annotate('Sugarloaf Mountain', xy=(400, 400), xytext=(500, 350),
             arrowprops=dict(facecolor='red', shrink=0.05))
plt.show()

print("Image coordinates: ", coordinates)


prompt = "Describe the surface features of this satellite image using plain english. Make a guess as to the location and time period."
buffered = io.BytesIO()
img.save(buffered, format="JPEG")
img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant skilled at analyzing satellite images."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]
        }
    ],
    max_tokens=500
)

print(completion.choices[0].message.content);







