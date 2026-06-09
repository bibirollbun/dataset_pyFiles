import ee
import io
import os
import matplotlib.pyplot as plt
import urllib.request
from kaggle_secrets import UserSecretsClient
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


coordinates = [-43.1566, -22.9486]
dem = ee.Image('USGS/SRTMGL1_003')
xy = ee.Geometry.Point(coordinates)
elev = dem.sample(xy, 30).first().get('elevation').getInfo()
print('Sugarloaf Mountain elevation (m):', elev)


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





