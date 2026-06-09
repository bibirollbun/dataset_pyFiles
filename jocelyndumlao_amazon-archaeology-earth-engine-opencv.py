!pip install geemap
!pip install folium
!pip install rasterio
!pip install earthengine-api
!pip install pyproj



import ee
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image  # To display images
import requests
from io import BytesIO
from PIL import Image as PILImage
import cv2

import warnings
warnings.filterwarnings("ignore")


import ee

# Define the path to the service account key
KEY_FILE = '/kaggle/input/json-file/civil-honor-461007-f0-106709184203.json'

# Use the service account email from your JSON file (update this if different)
SERVICE_ACCOUNT = 'your-service-account@civil-honor-461007-f0.iam.gserviceaccount.com'

# Authenticate and initialize Earth Engine
credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
ee.Initialize(credentials)
print("Earth Engine initialized successfully!")



# ROI near potential site
roi = ee.Geometry.Point([-65.34210, -12.56740]).buffer(5000)  # 5km buffer


sentinel = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(roi) \
    .filterDate('2023-01-01', '2023-12-31') \
    .sort('CLOUDY_PIXEL_PERCENTAGE') \
    .first()

rgb = sentinel.select(['B4', 'B3', 'B2'])  # Red, Green, Blue

url = rgb.visualize(min=0, max=3000).getThumbURL({'region': roi, 'dimensions': 512})
display(Image(url=url))



srtm = ee.Image('USGS/SRTMGL1_003')
elevation = srtm.clip(roi)

elev_viz = elevation.visualize(min=100, max=300, palette=['white', 'green', 'brown'])
url = elev_viz.getThumbURL({'region': roi, 'dimensions': 512})
display(Image(url=url))



ndvi = sentinel.normalizedDifference(['B8', 'B4']).rename('NDVI')  # (NIR - Red)/(NIR + Red)

ndvi_viz = ndvi.visualize(min=0, max=1, palette=['white', 'yellow', 'green'])
url = ndvi_viz.getThumbURL({'region': roi, 'dimensions': 512})
display(Image(url=url))



landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filterBounds(roi) \
    .filterDate('2020-01-01', '2020-12-31') \
    .sort('CLOUD_COVER') \
    .first()

landsat_rgb = landsat.select(['SR_B4', 'SR_B3', 'SR_B2']).multiply(0.0000275).add(-0.2)

landsat_viz = landsat_rgb.visualize(min=0, max=0.3)
url = landsat_viz.getThumbURL({'region': roi, 'dimensions': 512})
display(Image(url=url))



# Analyze SRTM Image for Geometric Shapes using OpenCV

# Download the SRTM image as a NumPy array
url = elev_viz.getThumbURL({'region': roi, 'dimensions': 512})
response = requests.get(url)
img = PILImage.open(BytesIO(response.content))
img_array = np.array(img)

# Convert the image to grayscale
gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

# Apply edge detection (Canny)
edges = cv2.Canny(gray, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Analyze contours for geometric shapes
min_size = 80  # Minimum size in pixels (adjust based on the actual size in meters and image resolution)
potential_sites = []

for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the shape is a rectangle or circle-like
    if len(approx) == 4:  # Rectangle
        x, y, w, h = cv2.boundingRect(approx)
        if w * h >= min_size:
            center_x = x + w // 2
            center_y = y + h // 2
            potential_sites.append((center_x, center_y, "rectangle"))
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
    else:  # Check for circle-like shapes (more sides)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        if radius >= min_size / 2:  # Radius should be at least half of min_size for circles
            potential_sites.append((int(x), int(y), "circle"))
            cv2.circle(img_array, center, radius, (0, 0, 255), 2)  # Draw circle

# Display the image with detected shapes
plt.imshow(img_array)
plt.title("Detected Geometric Shapes (SRTM)")
plt.show()

print("Potential Archaeological Sites (Approximate Center Coordinates) from SRTM:")
for site in potential_sites:
    print(f"Shape: {site[2]}, Center: {site[0]}, {site[1]}")

# Analyze NDVI for Canopy Dips

# Download the NDVI image as a NumPy array
url = ndvi_viz.getThumbURL({'region': roi, 'dimensions': 512})
response = requests.get(url)
img = PILImage.open(BytesIO(response.content))
img_array = np.array(img)

# Convert the image to grayscale
gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

# Use a threshold to identify low NDVI areas (potential canopy dips)
# Adjust the threshold based on the NDVI visualization and expected values
threshold = 100  # Example threshold - adjust as needed
canopy_dips = gray < threshold

# Find coordinates of canopy dips
dip_coords = np.where(canopy_dips)
dip_pixels = list(zip(dip_coords[1], dip_coords[0])) # (x, y)

# Display the NDVI image with potential canopy dips highlighted
plt.imshow(img_array)
plt.title("Potential Canopy Dips (NDVI)")
plt.scatter([x for x, y in dip_pixels], [y for x, y in dip_pixels], color='red', marker='x', label='Dip')
plt.legend()
plt.show()

print("Potential Canopy Dip Coordinates (x, y) from NDVI:")
for coord in dip_pixels:
    print(coord)

# Analyze SRTM for Terrain Models (Elevation Anomalies) 

# Enhance elevation differences (e.g., using a Sobel filter for edge detection)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
gradient_magnitude = cv2.normalize(gradient_magnitude, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # Scale to 0-255

# Display the gradient magnitude image (highlights terrain changes)
plt.imshow(gradient_magnitude, cmap='gray') # Use grayscale colormap
plt.title("Terrain Model (Elevation Anomalies - SRTM)")
plt.show()


# Analyze Landsat for Spectral Scars (Bare Soil, Clearing)

# Download the Landsat image as a NumPy array (using the visualization)
url = landsat_viz.getThumbURL({'region': roi, 'dimensions': 512})
response = requests.get(url)
img = PILImage.open(BytesIO(response.content))
img_array = np.array(img)

# Convert to grayscale
gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

# Thresholding to detect brighter areas (potential bare soil/clearing)
# Adjust threshold based on Landsat visualization
threshold = 200 # Example value - adjust based on the specific image
bare_soil = gray > threshold

# Find coordinates of potential bare soil
bare_soil_coords = np.where(bare_soil)
bare_soil_pixels = list(zip(bare_soil_coords[1], bare_soil_coords[0])) # (x, y)

# Display the Landsat image with potential bare soil highlighted
plt.imshow(img_array)
plt.title("Potential Spectral Scars (Bare Soil/Clearing - Landsat)")
plt.scatter([x for x, y in bare_soil_pixels], [y for x, y in bare_soil_pixels], color='yellow', marker='.', label='Bare Soil')
plt.legend()
plt.show()

print("Potential Spectral Scar Coordinates (x, y) from Landsat:")
for coord in bare_soil_pixels:
    print(coord)


submission = {
    "lat": -12.56740,
    "lng": -65.34210,
    "rationale": (
        "SRTM data shows potential geometric features (rectangles/circles). "
        "NDVI analysis indicates possible canopy dips indicative of past human activity. "
        "Further investigation with higher resolution LiDAR is recommended."
    )
}

print("\nSubmission Entry:")
print(submission)




