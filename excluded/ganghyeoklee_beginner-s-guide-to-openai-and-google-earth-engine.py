from openai import OpenAI
from kaggle_secrets import UserSecretsClient
import ee
import io
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image


# Initialize OpenAI client
secret_label = "OPENAI_API_KEY"
OPENAI_API_KEY = UserSecretsClient().get_secret(secret_label)
client = OpenAI(api_key=OPENAI_API_KEY)


prompt = "Say hello"
response = client.responses.create(
  model="gpt-4o-mini",
  input=[
    {"role": "user", "content": prompt}
  ]
)

print(response.output_text)


# This might take a while; be patient :)
ee.Authenticate()


# Initialize the library.
ee.Initialize(project="custom-rigging-363211") # replace this with your Cloud Project name


# Helper functions
def get_least_cloudy_s2_image(point_coords, start_date, end_date, cloud_filter_percentage, collection_id):
    """
    Filters the Sentinel-2 image collection for the least cloudy image
    over a given point and date range.

    Args:
        point_coords (list): A list of [longitude, latitude].
        start_date (str): The start date for filtering (YYYY-MM-DD).
        end_date (str): The end date for filtering (YYYY-MM-DD).
        cloud_filter_percentage (float): Maximum cloud pixel percentage.
        collection_id (str): The Earth Engine image collection ID.

    Returns:
        ee.Image: The least cloudy Sentinel-2 image or None if no image is found.
        ee.Geometry: The bounds of the buffered region around the point.
    """
    target_point = ee.Geometry.Point(point_coords)
    # Define the region of interest for the thumbnail by buffering and getting bounds
    region_bounds = target_point.buffer(BUFFER_RADIUS_METERS).bounds()

    collection = ee.ImageCollection(collection_id) \
        .filterBounds(target_point) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter_percentage)) \
        .sort('CLOUDY_PIXEL_PERCENTAGE') # Sort by cloud cover, ascending

    least_cloudy_image = collection.first()
    return least_cloudy_image, region_bounds

def get_image_thumbnail_url(image, region, dimensions, img_format, vis_params):
    """
    Generates a thumbnail URL for an Earth Engine image.

    Args:
        image (ee.Image): The Earth Engine image.
        region (ee.Geometry): The region for the thumbnail.
        dimensions (str): The dimensions (e.g., '800' for 800xN or Nx800).
        img_format (str): The image format (e.g., 'jpg', 'png').
        vis_params (dict): Visualization parameters including bands, min, max, gamma.

    Returns:
        str: The thumbnail URL.
    """
    # Apply visualization parameters to create an 8-bit RGB image
    # This is where gamma correction is applied before generating the thumbnail
    visualized_image = image.visualize(
        bands=vis_params['bands'],
        min=vis_params['min'],
        max=vis_params['max'],
        gamma=vis_params['gamma']
    )

    thumbnail_params = {
        'region': region,
        'dimensions': dimensions,
        'format': img_format
    }
    return visualized_image.getThumbURL(thumbnail_params)

def download_and_display_image(url, title):
    """
    Downloads an image from a URL and displays it with Matplotlib.

    Args:
        url (str): The URL of the image.
        title (str): The title for the plot.
    """
    try:
        response = urllib.request.urlopen(url)
        img_data = response.read()
        img = Image.open(io.BytesIO(img_data))

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(title, fontsize=15)
        plt.axis('off')
        plt.show()

        return img_data

    except urllib.error.URLError as e:
        print(f"Error downloading image: {e}")
    except Exception as e:
        print(f"An error occurred while processing the image: {e}")


# Configuration
# Target location (Latitude, Longitude)
COORDINATES = [-6.981918953145955, -58.36623265568083] # Lat, Lon for GEE Point
POINT_OF_INTEREST_NAME = 'Amazon Rainforest'
BUFFER_RADIUS_METERS = 3000  # Buffer around the point to define the region of interest

# Image collection and filtering parameters
IMAGE_COLLECTION_ID = 'COPERNICUS/S2_SR_HARMONIZED'
START_DATE = '2024-05-01'
END_DATE = '2025-05-01'
MAX_CLOUD_COVERAGE = 20  # Maximum cloud pixel percentage

# Visualization parameters for an RGB image
VIS_PARAMS = {
    'bands': ['B4', 'B3', 'B2'],  # Red, Green, Blue bands for true color
    'min': 0,
    'max': 3000, # Adjusted for Sentinel-2 SR typical reflectance values (scaled by 10000)
                # Max can be tuned (e.g., 0.3 * 10000 for reflectance, but 3000 is common for viz)
    'gamma': 1.3
}

# Thumbnail parameters
THUMBNAIL_DIMENSIONS = '800' # Width/height in pixels
THUMBNAIL_FORMAT = 'jpg'


# Note: GEE Point uses (longitude, latitude) order
gee_point_coords = [COORDINATES[1], COORDINATES[0]]

least_cloudy_image, region_for_thumbnail = get_least_cloudy_s2_image(
    gee_point_coords,
    START_DATE,
    END_DATE,
    MAX_CLOUD_COVERAGE,
    IMAGE_COLLECTION_ID
)

image_id = least_cloudy_image.id().getInfo()
print(f"Found image: {image_id}")


thumbnail_url = get_image_thumbnail_url(
    least_cloudy_image,
    region_for_thumbnail,
    THUMBNAIL_DIMENSIONS,
    THUMBNAIL_FORMAT,
    VIS_PARAMS
)

plot_title = f'{POINT_OF_INTEREST_NAME} - Sentinel-2 ({VIS_PARAMS["bands"][0]}/{VIS_PARAMS["bands"][1]}/{VIS_PARAMS["bands"][2]})\n' \
             f'Least Cloudy Image ({START_DATE} to {END_DATE}, <{MAX_CLOUD_COVERAGE}% cloud)'

image = download_and_display_image(
    thumbnail_url,
    plot_title,
)

print(f"Original coordinates for {POINT_OF_INTEREST_NAME}: {COORDINATES} (Lat, Lon)")


# Initialize OpenAI
secret_label = "OPENAI_API_KEY"
OPENAI_API_KEY = UserSecretsClient().get_secret(secret_label)
client = OpenAI(api_key=OPENAI_API_KEY)
CHOSEN_MODEL = "gpt-4o-mini"

prompt = "Analyze the satelite image and describe surface features in plain English"
response = client.responses.create(
    model=CHOSEN_MODEL,
    input=[
        {"role": "user", "content": prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": thumbnail_url
                }
            ]
        }
    ]
)

print(response.output_text)

