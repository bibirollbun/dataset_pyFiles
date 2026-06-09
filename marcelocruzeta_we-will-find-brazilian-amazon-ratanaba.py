# ========================= CELL 1: Package Installations =========================

# Upgrade pip to ensure the latest package management features
!pip install --upgrade pip
# Installs or updates rasterio, a library for reading and writing raster data formats like GeoTIFF
!pip install --upgrade rasterio
# Installs or updates geopandas, a library for working with geospatial dataframes
!pip install --upgrade geopandas
# Installs or updates seaborn, a statistical data visualization library built on top of matplotlib
!pip install --upgrade seaborn
# Installs or updates the Earth Engine API for Python, used to access Google Earth Engine data and tools
!pip install earthengine-api --upgrade

# Optional: Add other installations if needed, e.g.,
# !pip install folium  # For interactive map visualization


# ========================= CELL 2: Imports and Initial Configurations =========================

# Standard library imports
import os  # To perform operating system dependent functionalities, like file management
import json  # Standard library for working with JSON (JavaScript Object Notation) data.
import ee  # Earth Engine API for accessing and manipulating satellite and terrestrial data.
import base64  # Standard library for encoding and decoding data in Base64 format.
import geemap.core as geemap  # Library to facilitate the use of Google Earth Engine with interactive maps and visualizations.
import requests  # Library for making HTTP requests easily and quickly.
import gdown  # Library to download files from Google Drive directly using the file ID.
import math  # Standard library for mathematical functions, such as trigonometry and basic operations.
import cv2  # OpenCV library to facilitate computer vision tasks, including image processing, object detection, and video analysis.
import numpy as np  # For numerical computations and array manipulations, essential for data processing
import pandas as pd  # For structured data handling, dataframes, and data analysis
import rasterio  # To read, write, and process GeoTIFF raster data
from rasterio.transform import Affine
from rasterio.transform import xy
import matplotlib  # Library for creating static, animated, and interactive visualizations in Python.
import pprint  # Standard library for "pretty-printing" data structures to make them more readable.
import matplotlib.pyplot as plt  # To create static, animated, and interactive visualizations
import matplotlib.colors as colors  # Submodule of Matplotlib for handling colors.
import matplotlib.patches as patches  # Submodule of Matplotlib for creating shapes (patches) in plots.
from IPython.display import display, HTML  # Functions to display HTML elements within Jupyter notebooks.
from io import BytesIO  # In-memory binary streams for input and output data manipulation.
import geopandas as gpd  # Extends pandas for spatial data reading, writing, and analysis
import seaborn as sns  # For statistical data visualization, built on top of matplotlib
from PIL import Image  # Provides image processing capabilities (loading, manipulating images)
from openai import OpenAI  # To interact with the OpenAI API for language model queries
from kaggle_secrets import UserSecretsClient  # To securely access API keys stored in Kaggle Secrets
from scipy.ndimage import gaussian_filter  # For image filtering and smoothing operations
import folium  # To create interactive maps for visualization of spatial data
from folium import Rectangle  # Class from Folium for creating rectangles on interactive maps.
from tabulate import tabulate  # Library for formatting tables in a readable way in text.
# Optional: set Seaborn style for aesthetic plots
sns.set(style="whitegrid")


def load_raster(file_path):

    """
    Load a GeoTIFF raster file and return the data.
    
    Parameters:
    file_path (str): The path to the GeoTIFF file.

    Returns:
    geoTiff_data: The raster data read from the file.
    profile: The metadata profile of the raster.
    """
    try:
        with rasterio.open(file_path) as src:
            geoTiff_data = src.read()  # Read the raster data
            profile = src.profile  # Get the metadata profile
        return geoTiff_data, profile  # Return the loaded data and profile
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None, None  # Return None on error
    except rasterio.errors.RasterioError as e:
        print(f"Rasterio error occurred: {e}")
        return None, None  # Return None on error
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
        return None, None  # Return None on error

def inspect_geotiff(file_path):
    """Inspects a GeoTIFF file and prints relevant information."""
    try:
        with rasterio.open(file_path) as src:
            print("File:", file_path)
            print("Number of Bands:", src.count)
            print("Dimensions (width, height):", src.width, "x", src.height)
            print("Transformation:", src.transform)
            print("Coordinate Reference System (CRS):", src.crs)
            print("Min and Max Values:", src.read(1).min(), "to", src.read(1).max())
            print("Bounding Box (BBOX):", src.bounds)
            
            # Calculate statistics for slope calculation
            data = src.read(1)
            mean_value = np.mean(data)
            median_value = np.median(data)
            std_dev = np.std(data)
            print("Mean Elevation:", mean_value)
            print("Median Elevation:", median_value)
            print("Standard Deviation:", std_dev)

            # Size of pixels in geographic coordinates
            pixel_size_x = abs(src.transform[0])  # Width of pixel
            pixel_size_y = abs(src.transform[4])  # Height of pixel
            print("Pixel Size (in geographic units):", pixel_size_x, "by", pixel_size_y)

            # You can add more information as needed
    except Exception as e:
        print(f"An error occurred while inspecting the GeoTIFF: {e}")

def descriptive_statistics(geoTiff_data, profile):
    """
    Calculate and return descriptive statistics of the raster data.
    
    Parameters:
    geoTiff_data: The raster geoTiff_data (3D numpy array).
    profile: The metadata profile of the raster (for geotransform and dimensions).
    
    Returns:
    results: A dictionary containing descriptive statistics including area covered,
             minimum and maximum altitudes, valid data ratio, mean and standard deviation.
    """
    results = {}
    
    # 1. Dimensions
    results['width'] = profile['width']  # Width of the raster in pixels
    results['height'] = profile['height']  # Height of the raster in pixels
    
    # 2. Area Covered - Calculate the total area in square meters
    geotransform = profile['transform']  # Transformation array providing scaling info
    pixel_area = abs(geotransform[0] * geotransform[4])  # Area of a single pixel in square meters
    area_covered = results['width'] * results['height'] * pixel_area  # Total area in square meters
    results['area_covered'] = area_covered  # Store the total area
    
    # 3. Minimum and Maximum Altitudes
    data_2d = geoTiff_data[0]  # Assuming the data is in the first band
    results['altitude_min'] = np.nanmin(data_2d)  # Minimum altitude value, ignoring NaNs
    results['altitude_max'] = np.nanmax(data_2d)  # Maximum altitude value, ignoring NaNs
    
    # 4. Valid Data Statistics
    valid_data_count = np.count_nonzero(~np.isnan(data_2d))  # Count valid (non-null) entries
    total_data_count = data_2d.size  # Total number of pixels in the raster
    results['valid_data_ratio'] = valid_data_count / total_data_count  # Ratio of valid data to total pixels
    
    # 5. Mean and Standard Deviation
    results['mean_altitude'] = np.nanmean(data_2d)  # Mean altitude value, ignoring NaNs
    results['std_dev_altitude'] = np.nanstd(data_2d)  # Standard deviation of altitude values, ignoring NaNs

    return results

def create_numbered_icon(number):
    # Cria um Ã­cone numerado usando HTML para um marcador
    icon_html = f"""
    <div style="font-size: 14px; color: white; background-color: red; 
                text-align: center; width: 30px; border-radius: 50%; padding: 5px;">
        {number}
    </div>
    """
    return folium.DivIcon(html=icon_html)

def generate_dtm_from_elevation(elevation_data, output_path):
    """
    Save the elevation data as a DTM in a GeoTIFF file.
    
    Parameters:
    elevation_data: The 2D array of elevation data.
    output_path: The path where the DTM will be saved.
    """
    try:
        # Get information to create the GeoTIFF
        with rasterio.open(file_path_0) as src:
            profile = src.profile  # Get the profile of data
            profile.update({
                'count': 1,  # Only one band
                'dtype': elevation_data.dtype  # Data type
            })
        
        # Save as GeoTIFF
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(elevation_data, 1)  # Write the first band
        print(f"DTM saved at {output_path}")
    
    except Exception as e:
        print(f"An error occurred while generating the DTM: {e}")

def visualize_raster(file_path):
    """
    Visualize a raster image from a GeoTIFF file.

    Parameters:
    file_path (str): The path to the GeoTIFF file containing elevation data.
    """
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Read the first band
            plt.figure(figsize=(10, 8))  # Set the figure size
            plt.imshow(data, cmap='terrain')  # Use terrain colormap for better visualization
            plt.colorbar(label='Elevation (m)')  # Add a colorbar to indicate elevation scale
            plt.title('Digital Terrain Model (DTM)')  # Set the title of the plot
            plt.xlabel('Columns (pixels)')  # X-axis label
            plt.ylabel('Rows (pixels)')  # Y-axis label
            plt.show()  # Show the plot
    except Exception as e:
        print(f"An error occurred while visualizing the raster: {e}")

def visualize_raster_dtm(file_path):
    """
    Visualize a raster image from a GeoTIFF file.

    Parameters:
    file_path (str): The path to the GeoTIFF file containing elevation data.
    """
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Read the first band
            plt.figure(figsize=(10, 8))  # Set the figure size
            plt.imshow(data, cmap='terrain')  # Use terrain colormap for better visualization
            plt.colorbar(label='Elevation (m)')  # Add a colorbar to indicate elevation scale
            plt.title('Digital Terrain Model (DTM)')  # Set the title of the plot
            plt.xlabel('Columns (pixels)')  # X-axis label
            plt.ylabel('Rows (pixels)')  # Y-axis label
            plt.show()  # Show the plot
    except Exception as e:
        print(f"An error occurred while visualizing the raster: {e}")

def visualize_raster_dtm_square(file_path, center_x, center_y, size):
    """
    Visualize a raster image from a GeoTIFF file.
    Parameters:
    file_path (str): The path to the GeoTIFF file containing elevation data.
    center_x (int): Center x coordinate in pixels.
    center_y (int): Center y coordinate in pixels.
    size (int): Size of the square in pixels.
    """
    try:
        with rasterio.open(file_path) as src:  # Corrigido aqui
            data = src.read(1)  # Read the first band
            transform = src.transform
            
            plt.figure(figsize=(10, 8))
            plt.imshow(data, cmap='terrain')
            plt.colorbar(label='Elevation (m)')
            plt.title('Digital Terrain Model (DTM)')
            plt.xlabel('Columns (pixels)')
            plt.ylabel('Rows (pixels)')
            
            # Parameters for the square
            lower_left_x = center_x - (size / 2)
            lower_left_y = center_y - (size / 2)
            
            # Draw the square on the plot
            square = patches.Rectangle((lower_left_x, lower_left_y), size, size,
                                       linewidth=2, edgecolor='red', facecolor='none')
            plt.gca().add_patch(square)
            
            # Convert pixel coordinates to geographic coordinates using the profile's transform
            center_geo_x = transform[2] + transform[0] * center_x
            center_geo_y = transform[5] + transform[4] * center_y
            
            # Show the plot
            plt.tight_layout(pad=0)
            plt.show()
            
            # Print coordinates below the plot
            print(f'Center Coordinates (Geographic): ({center_geo_x:.4f}, {center_geo_y:.4f})')
    except Exception as e:
        print(f"An error occurred while visualizing the raster: {e}")


def calculate_slope(dtm_data):
    """
    Calculate the slope of the Digital Terrain Model (DTM) using gradients.

    Parameters:
    dtm_data (numpy array): The 2D array of elevation data from the DTM.

    Returns:
    numpy array: A 2D array representing the slope in degrees.
    """
    # Smooth the DTM data to reduce noise
    dtm_filtered = gaussian_filter(dtm_data, sigma=1)  # Apply Gaussian filter for smoothing

    # Calculate gradients (slope)
    dx, dy = np.gradient(dtm_filtered)  # Compute gradients in the x and y directions
    slope = np.arctan(np.sqrt(dx**2 + dy**2))  # Calculate slope in radians
    slope_degrees = np.degrees(slope)  # Convert radians to degrees
    return slope_degrees  # Return the slope in degrees

def slope_calculation(file_path, sigma_value, center_x, center_y, size):
    geoTiff_data, profile = load_raster(file_path)

    # Cell: Slope Calculation and Plot with Sigma Control
    sigma_value = sigma_value

    # Calculate slope data
    dtm_data = geoTiff_data[0]  # Extract elevation data from the first band
    slope_data = calculate_slope(dtm_data)  # Calculate slope

    # Calculate mean and std deviation
    mean_val = np.nanmean(slope_data)
    std_val = np.nanstd(slope_data)

    # Ensure vmin is not negative since slope values are physically non-negative
    vmin = max(0, mean_val - sigma_value * std_val)
    vmax = mean_val + sigma_value * std_val

    # Guarantee vmax > vmin, adjust if necessary (edge case)
    if vmax <= vmin:
        vmax = vmin + 1e-6  # minimal positive delta

    masked_slope = np.ma.masked_invalid(slope_data)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(10, 8))
    plt.imshow(masked_slope, cmap='terrain', norm=norm)
    plt.colorbar(label='Slope (degrees)')
    plt.title('Slope Calculation from Digital Terrain Model (DTM) Using Variable Sigma')
    plt.xlabel('Columns (pixels)')
    plt.ylabel('Rows (pixels)')

    # Parameters for the square
    center_x_in_pixels = center_x  # Define the center x in pixels
    center_y_in_pixels = center_y   # Define the center y in pixels
    size_in_pixels = size        # Size of the square in pixels

    # Calculate the square's lower left corner based on center and size
    lower_left_x = center_x_in_pixels - (size_in_pixels / 2)
    lower_left_y = center_y_in_pixels - (size_in_pixels / 2)

    # Draw the square on the plot
    square = patches.Rectangle((lower_left_x, lower_left_y), size_in_pixels, size_in_pixels,
                            linewidth=2, edgecolor='red', facecolor='none')
    plt.gca().add_patch(square)

    # Convert pixel coordinates to geographic coordinates using the profile's transform
    transform = profile['transform']
    center_geo_x = transform[0] * center_x_in_pixels + transform[2]  # Calculate geographic x
    center_geo_y = transform[4] * center_y_in_pixels + transform[5]  # Calculate geographic y

    # Show the plot
    plt.tight_layout(pad=0)
    plt.show()

    # Print coordinates below the plot
    print(f'Center Coordinates (Geographic): ({center_geo_x:.4f}, {center_geo_y:.4f})')

def crop_circular_region(image, center, radius):
    """
    Crops a circular region from the given image.

    Parameters:
    - image: 2D or 3D array of the original image
    - center: tuple (x_center, y_center) of the circle's center
    - radius: radius of the circular crop

    Returns:
    - the cropped region with the mask applied
    """
    # Get the dimensions of the image
    rows, cols = image.shape[:2]
    y_indices, x_indices = np.ogrid[:rows, :cols]
    
    # Calculate the distance of each pixel from the center
    dist = np.sqrt((x_indices - center[0])**2 + (y_indices - center[1])**2)
    
    # Create the circular mask
    mask = dist <= radius
    
    # Apply the mask to the image
    if image.ndim == 3:
        # For color images with channels
        mask = mask[:, :, np.newaxis]
        cropped_region = np.where(mask, image, np.nan)  # Replace outside with NaN
    else:
        cropped_region = np.where(mask, image, np.nan)
        
    return cropped_region

def save_region(region_array, filename):
    """
    Save the already cropped region to an image file.
    """
    plt.imsave(filename, region_array, cmap='terrain')  # Use colormap padrÃ£o que usou

def selecionar_registros(csv_path, col, col_list, show_table=False):
    """
    Loads records from a CSV, filters by col list and displays the table.
    
    Parameters:
    - csv_path (str): Path of the CSV file.
    - col: column name
    - col_list (list): List of col_list names to filter.
    - show_table (bool): If True, displays the filtered table.
    
    Returns:
    - filtered_df (pd.DataFrame): DataFrame with the filtered records.
    """
    # Leitura do CSV
    df = pd.read_csv(csv_path)

    # Filtra por KLM
    filtered_df = df[df[col].isin(col_list)].copy()

        # Opcional: mostra a tabela
    if show_table:
        headers = ["KLM", "Site Name", "Latitude", "Longitude"]
        print("List of selected records:")
        print(tabulate(
            filtered_df,
            headers=headers,
            tablefmt='fancy_grid',
            numalign='center',
            stralign='center',
            showindex=False
        ))

    # Retorna o dataframe filtrado
    return filtered_df

def get_s2_sr_cld_col(aoi, start_date, end_date):
    # ColeÃ§Ã£o harmonizada do Sentinel-2
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))
    # ColeÃ§Ã£o de probabilidade de nuvem
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))
    # Join para associar a probabilidade de nuvem
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))
    
def add_cld_mask(img):
    # Obter a banda de probabilidade de nuvem
    cloud_prob = ee.Image(img.get('s2cloudless')).select('probability')
    # Mask de nuvens com limiar
    clouds = cloud_prob.gt(CLD_PRB_THRESH).rename('clouds')
    return img.addBands(clouds)

def add_shadows_mask(img):
    # Bits de classificaÃ§Ã£o na banda 'SCL' indicam sombras, Ã¡gua, etc.
    scl = img.select('SCL')
    not_water = scl.neq(6)
    # Pixels escuros na banda B8 (NIR) como indicativo de sombras
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH * SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')
    # DireÃ§Ã£o de projeÃ§Ã£o de sombras (ajuste para sua regiÃ£o ou solar)
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
    # Projetar sombras de nuvens
    cld_proj = (img.select('clouds')
        .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST * 10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 20})
        .select('distance')
        .mask()
        .rename('cloud_transform'))
    # Pessoas que sÃ£o pixels escuros intersectando projeÃ§Ãµes de sombra
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

# Para visualizar na interface interativa
def add_ee_layer(self, ee_image, vis_params, name):
    map_id_dict = ee_image.getMapId(vis_params)
    tiles_url = map_id_dict['tile_fetcher'].url_format
    folium.TileLayer(
        tiles=tiles_url,
        attr='Map Data &copy; Google Earth Engine',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

def add_cld_shdw_mask(img):
    # Bands adicionais de nuvem e sombra
    img_cloud = add_cld_mask(img)
    img_cloud_shadow = add_shadows_mask(img_cloud)
    # Combinar mÃ¡scaras de nuvem e sombra (1= nuvem/shadow, 0=no)
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)
    # Refinar mÃ¡scara com focalMin e focalMax para eliminar pequenas Ã¡reas
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER / 20)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 20})
        .rename('cloudmask'))
    return img_cloud_shadow.addBands(is_cld_shdw)

# Function to download a topography tile
def download_topo_tile(lat_center, lon_center, area_km2=10, save_dir='/kaggle/working/', filename='tile.tif'):
    delta_deg = calculate_delta_deg(area_km2)
    south = lat_center - delta_deg / 2
    north = lat_center + delta_deg / 2
    west = lon_center - delta_deg / 2
    east = lon_center + delta_deg / 2

    url = (
        'https://portal.opentopography.org/API/globaldem'
        f'?demtype=SRTMGL1'
        f'&south={south}'
        f'&north={north}'
        f'&west={west}'
        f'&east={east}'
        f'&outputFormat=GTiff'
        f'&API_Key={API_Key}'
    )

    response = requests.get(url)
    if response.status_code == 200:
        full_path = os.path.join(save_dir, filename)
        with open(full_path, 'wb') as f:
            f.write(response.content)
        print(f'{filename} saved.')
    else:
        print(f"Error downloading {filename}: {response.status_code}")

# Function to encode an image to Base64
def encode_image(image_path):
    """
    Encodes an image to a Base64 string.
    Parameters:
    - image_path (str): Path to the image file to encode.
    Returns:
    - str: Base64 encoded string of the image.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image '{image_path}': {e}")
        return None
        
# Step 1: Select TIFFs from a directory
def get_tiff_files(directory):
    """Retrieve a list of TIFF files from the given directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.tif')]

# Step 2: Analyze TIFF and log basic data
def analyze_tiff(tiff_path):
    """Extract geographic information from the TIFF file."""
    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
        altitude = src.read(1).mean()  # Average altitude from the first band
        lat = (bounds.top + bounds.bottom) / 2
        lon = (bounds.left + bounds.right) / 2
    return {
        'file_name': os.path.basename(tiff_path),
        'latitude': lat,
        'longitude': lon,
        'altitude': altitude,
    }
    
# Step 3: Convert TIFF to PNG
def convert_tif_to_png(tif_path, png_path):
    """Convert a TIFF file to PNG format."""
    with rasterio.open(tif_path) as src:
        data = src.read(1)  # Read the first band
        data = (data - data.min()) / (data.max() - data.min()) * 255  # Normalize
        data = data.astype('uint8')
        with rasterio.open(
            png_path,
            'w',
            driver='PNG',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype='uint8'
        ) as dst:
            dst.write(data, 1)

def analyze_image(client, image_path):
    """Send the PNG image to OpenAI for analysis."""
    base64_image = encode_image(image_path)
    prompt = (
        "You are being shown a COPERNICUS/S2_SR_HARMONIZED satelite image, "
        "look for distinct linear and curving ridges and depressions, "
        "straight and intersecting raised lines or embankments,"
        "possibly indicating man-made structures such as ancient roads,"
        "agricultural terraces, or boundaries,"
        "elongated depressions running roughly parallel to these ridges,"
        "creating a geometric or grid-like pattern"
        "Small mounds and subtle rises in the terrain visible between linear features"
        "If you find some of these characteristics in the analyzed image,"
        "write the word FOUND in capital letters in the first position of your description"
        "Do not use the word FOUND in your description if you do not find any of the anomalies described above"
        "Return the coordinates of your findings in a simple format like: [{x1, y1}, {x2, y2}, ...], where x and y are pixel values."
        "Describe surface features in plain English."
    )
    # Chamada Ã  API da OpenAI
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}",
                    },
                ],
            }
        ]
    )
    # Acesso ao texto resultante a partir da estrutura fornecida
    output_text = response.output[0].content[0].text  # Ajustando para acessar o texto gerado
    model_version = response.model  # Acessa a versÃ£o do modelo
    print(f"Analise completed for: {image_path}")
    return {
        'output_text': output_text,
        'model_version': model_version,
        'dataset_id': 'https://doi.org/10.5069/G9028PQB'  # Substituir pelo ID real
    }

def analyze_single_png(prompt, png_file):
    """Analyzes a PNG image and generates a description using OpenAI's API.

    Args:
        prompt (str): The prompt to instruct the AI on how to describe the image.
        png_file (str): The path to the PNG image file to be analyzed.

    Returns:
        dict: A dictionary containing the dataset ID, model version, and generated output text.
    """
    # Retrieve user secrets for OpenAI API
    user_secrets = UserSecretsClient()
    openaikey = user_secrets.get_secret("Z_Challenge_Key")

    # Initialize OpenAI client
    client = OpenAI(api_key=openaikey)

    # Get the Base64 string of the image
    base64_image = encode_image(png_file)

    # Create a response using the OpenAI client
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )

    # Define the dataset ID
    dataset_id = 'marcelocruzeta/z-challenge-files'  # Replace with your actual dataset ID

    # Prepare result summary
    result_summary = {
        "Dataset ID": dataset_id,
        "Model Version": response.model,
        "Generated Output Text": response.output_text
    }

    return result_summary

# Step 5: Save logs to JSON
def save_logs_to_json(logs, output_file):
    """Save the logs of analysis and metadata to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(logs, f, indent=4)

# Main processing function
def process_images(input_directory, output_directory, log_file):
    """Process all TIFF files in the input directory, converting them to PNG and analyzing them."""
    tiff_files = get_tiff_files(input_directory)
    logs = []
    # Instantiate the OpenAI client
    api_key = load_secret('Z_Challenge_Key')  # Load your API key
    client = OpenAI(api_key=api_key)
    for tif_file in tiff_files:
        # Analyze TIFF
        metadata = analyze_tiff(tif_file)
        
        # Convert to PNG
        png_file = os.path.join(output_directory, f"{os.path.splitext(os.path.basename(tif_file))[0]}.png")
        convert_tif_to_png(tif_file, png_file)
        
        # Analyze PNG
        analysis_result = analyze_image(client, png_file)
        
        # Combine logs
        log_entry = {
            'metadata': metadata,
            'analysis': analysis_result
        }
        logs.append(log_entry)
    # Save all logs to JSON
    save_logs_to_json(logs, log_file)

def process_range_images(csv_path, start_tile, end_tile, save_dir, output_dir, log_file):
    # Step 1: Download the images from OpenTopography
    download_range_from_csv(csv_path, start_tile, end_tile, save_dir)
    
    # Step 2: Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Step 3: Process the downloaded images
    process_images(input_directory=save_dir, output_directory=output_dir, log_file=log_file)

# Function to retrieve and download tiles based on list of names
def download_multiple_tiles(klm_list, df, area_km2=10, save_dir='/kaggle/working/SRTMGL1/'):
    for klm_name in klm_list:
        record = df[df['KLM'] == klm_name]
        if record.empty:
            print(f"Record not found for KLM: {klm_name}")
            continue
        lat = float(record.iloc[0]['lat'])
        lon = float(record.iloc[0]['lon'])
        filename = klm_name + ".tif"
        download_topo_tile(lat, lon, area_km2, save_dir, filename)
        print(f"Download completed for: {klm_name}")

def filter_tiles_by_range(csv_path, start_tile, end_tile):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter the data to include only rows whose tiles are within the specified range
    filtered_df = df[(df['Tile'] >= start_tile) & (df['Tile'] <= end_tile)]
    
    # Print the selected data
    print("Selected data:")
    print(filtered_df)

    return filtered_df

def download_topo_tile_coord(lat_nw, lon_nw, lat_se, lon_se, save_dir, filename):
    url = (
        'https://portal.opentopography.org/API/globaldem'
        f'?demtype=SRTMGL1'
        f'&south={lat_se}'    # Latitude mais ao sul
        f'&north={lat_nw}'    # Latitude mais ao norte
        f'&west={lon_nw}'     # Longitude mais ao oeste
        f'&east={lon_se}'     # Longitude mais ao leste
        f'&outputFormat=GTiff'
        f'&API_Key={API_Key}' # Substitua por sua chave de API
    )
    
    response = requests.get(url)
    if response.status_code == 200:
        full_path = os.path.join(save_dir, filename)
        with open(full_path, 'wb') as f:
            f.write(response.content)
        print(f'{filename} saved.')
    else:
        print(f"Error downloading {filename}: {response.status_code}")

def download_tiles_from_csv(csv_path, tile, save_dir):
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Filter the DataFrame to only include rows with the specified tile numbers
    filtered_df = df[df['Tile'].isin(tile_numbers)]

    for index, row in filtered_df.iterrows():
        # Extract coordinates for the corners
        nw_lat = row['NW_Latitude']
        nw_lon = row['NW_Longitude']
        se_lat = row['SE_Latitude']
        se_lon = row['SE_Longitude']

        # Create a filename based on the tile number
        tile_number = row['Tile']
        filename = f"tile_{tile_number}.tif"

        # Download the tile using the corner coordinates
        download_topo_tile_coord(nw_lat, nw_lon, se_lat, se_lon, save_dir=save_dir, filename=filename)

def download_range_from_csv(csv_path, start_tile, end_tile, save_dir):
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Get the filtered tiles within the specified range
    filtered_df = filter_tiles_by_range(csv_path, start_tile, end_tile)

    # Loop over the filtered DataFrame and download each tile
    for index, row in filtered_df.iterrows():
        # Extract the coordinates for the corners
        nw_lat = row['NW_Latitude']
        nw_lon = row['NW_Longitude']
        se_lat = row['SE_Latitude']
        se_lon = row['SE_Longitude']
        
        # Create a filename based on the tile number
        tile_number = row['Tile']
        filename = f"tile_{tile_number}.tif"
        
        # Download the tile using the corner coordinates
        download_topo_tile_coord(nw_lat, nw_lon, se_lat, se_lon, save_dir=save_dir, filename=filename)
        print(f"Download completed for: {filename}")

def format_value(val):
    if isinstance(val, float):
        return f"{val:.4f}"
    elif isinstance(val, tuple):
        return "(" + ", ".join(f"{v:.4f}" for v in val) + ")"
    else:
        return val

# Aqui uma funÃ§Ã£o que faz toda a rotina para um registro
def processar_registro(lon, lat, nome, buffer_km=2):
    # Define a regiÃ£o de interesse
    aoi = ee.Geometry.Point(lon, lat)
    region = aoi.buffer(buffer_km*1000).getInfo()['coordinates']
    
    # Busca coleÃ§Ã£o
    col = get_s2_sr_cld_col(aoi, START_DATE, END_DATE)
    # Aplica as mÃ¡scaras
    col_masked = col.map(add_cld_shdw_mask)
    # Gera a mediana
    median_img = col_masked.median()
    # VisualizaÃ§Ã£o (nÃ£o obrigatÃ³rio na rotina, mas se desejar pode exibir)
    # imagem_visualizada = median_img.visualize(**{'bands': ['B4', 'B3', 'B2'], 'min':0, 'max':3000, 'gamma':1.1})

    # Exporta usando sua lÃ³gica
    imagem_visualizada = median_img.visualize(**{'bands': ['B4', 'B3', 'B2'], 'min':0, 'max':3000, 'gamma':1.1})

    # Configura a tarefa
    task = ee.batch.Export.image.toDrive(
        image=imagem_visualizada,
        description='Export_'+nome,
        folder='GEEE',
        fileNamePrefix=nome,
        region=region,
        scale=10,
        maxPixels=1e10
    )
    # Inicia a exportaÃ§Ã£o
    task.start()
    print(f'Exportando {nome}...')

def load_secret(secret_label):
    """Load secret from Kaggle."""
    try:
        return UserSecretsClient().get_secret(secret_label)
    except Exception as e:
        print("Error loading secret from Kaggle:", e)
        return None
        
# Function to convert image to base64
def pil_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Function to display one image with one text
def image_text(image_path, text):

    if os.path.exists(image_path):  # Check if the image file exists
        png_img = Image.open(image_path)  # Open the image
        png_base64 = pil_to_base64(png_img)  # Convert the image to base64

        # Define the text below the image
        text = text

        # Create HTML content for displaying the image and text
        html_content = f"""
        <div style='text-align: center;'>
            <img src='{png_base64}' style='width:250px; height:auto;'>
            <h3>{text}</h3>
        </div>
        """

        # Display the HTML content
        display(HTML(html_content))
    else:
        print(f"Image file '{image_path}' does not exist.")

# Display images from a directory with the given text
def img_dir_text(directory, descriptions):
    # Create a DataFrame for easier visualization
    data = []
    for file_name, description in descriptions.items():
        tile_png = os.path.join(directory, file_name)
        if os.path.exists(tile_png):  # Only add existing images
            data.append((tile_png, description))

    df = pd.DataFrame(data, columns=['PNG', 'Description'])

    # Function to convert image to base64
    def pil_to_base64(img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    # Collect HTML for all images and descriptions
    html_content = ""

    for index, row in df.iterrows():
        png_img = Image.open(row['PNG'])
        png_base64 = pil_to_base64(png_img)

        # Extract just the file names for display
        png_file_name = os.path.basename(row['PNG'])

        # Create HTML for each image and its description
        html_content += f"""
        <table style="min-width: 50px; margin-bottom: 20px;">
            <colgroup>
                <col style="min-width: 250px;">  <!-- Adjusted for image column size -->
                <col style="min-width: 250px;">  <!-- Keep the smaller description column -->
            </colgroup>
            <tbody>
                <tr>
                    <td colspan="1" rowspan="1" style="text-align: left;">
                        <strong>Image: {png_file_name}</strong><br>
                        <img src="{png_base64}" width="250" height="auto">  <!-- Increased image width -->
                    </td>
                    <td colspan="1" rowspan="2" style="text-align: left; padding-left: 20px;">
                        <strong>Description</strong><br>
                        {row['Description']}
                    </td>
                </tr>
            </tbody>
        </table>
        """

    # Display all the collected HTML content
    display(HTML(html_content))

# Function to display the table
def display_images(row):
    png_img = Image.open(row['PNG'])
    with rasterio.open(row['TIFF']) as src:
        tif_img = src.read(1)  # Read the first band of the TIFF
        plt.imshow(tif_img, cmap='terrain')  # Use the terrain colormap
        plt.axis('off')

        # Save the figure to a buffer for display as PNG
        plt.savefig('tif_image.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    png_base64 = pil_to_base64(png_img)
    with open('tif_image.png', 'rb') as f:
        tif_base64 = base64.b64encode(f.read()).decode()

    # Extract just the file names for display
    png_file_name = os.path.basename(row['PNG'])
    tif_file_name = os.path.basename(row['TIFF'])

    html_content = f"""
    <table style="min-width: 50px">
        <colgroup>
            <col style="min-width: 250px;">  <!-- Adjusted for image column size -->
            <col style="min-width: 150px;">  <!-- Keep the smaller analysis column -->
        </colgroup>
        <tbody>
            <tr>
                <td colspan="1" rowspan="1" style="text-align: left;">
                    <strong>Image: {png_file_name}</strong><br>
                    <img src="{png_base64}" width="150" height="auto">  <!-- Increased image width -->
                </td>
                <td colspan="1" rowspan="2" style="text-align: left; padding-left: 20px;">
                    <strong>JSON Analysis</strong><br>
                    {row['Analysis']}
                </td>
            </tr>
            <tr>
                <td colspan="1" rowspan="1" style="text-align: left;">
                    <strong>Tile: {tif_file_name}</strong><br>
                    <img src="data:image/png;base64,{tif_base64}" width="150" height="auto">  <!-- Increased image width -->
                </td>
            </tr>
        </tbody>
    </table>
    """
    return HTML(html_content)

# Function to display the table
def display_analysis(directory, json_file):
    # Lendo o arquivo JSON
    with open(os.path.join(directory, json_file)) as f:
        json_data = json.load(f)

    if isinstance(json_data, dict):
        tiles = json_data.get('tiles', [])
    elif isinstance(json_data, list):
        tiles = json_data
    else:
        tiles = []

    if tiles:
        data = []
        for tile in tiles:
            if isinstance(tile, dict) and 'metadata' in tile:
                tile_png = os.path.join(directory, tile['metadata']['file_name'].replace('.tif', '.png'))
                tile_tif = os.path.join(directory, tile['metadata']['file_name'])
                # VerificaÃ§Ã£o da existÃªncia dos arquivos
                if os.path.exists(tile_png) and os.path.exists(tile_tif):
                    data.append({
                        'PNG': tile_png,
                        'TIFF': tile_tif,
                        'analysis': tile.get('analysis', {}),
                        'metadata': tile.get('metadata', {})
                    })
        
        # Extraindo identificadores numÃ©ricos para classificar
        for tile in data:
            file_name = os.path.basename(tile['PNG'])
            tile['Num'] = float(file_name.split('.')[0].split('tile_')[-1])  # Supondo que o formato do nome seja algo como "tile_1.png"
            
        # Ordenando os dados com base nos identificadores numÃ©ricos
        data.sort(key=lambda x: x['Num'])
    
    # Iterando sobre cada um dos tiles para exibir
    for tile_data in data:
        file_name = os.path.basename(tile['PNG'])
        tile['Num'] = float(file_name.split('.')[0].split('tile_')[-1])  # Supondo que o formato do nome seja algo como "tile_1.png"
        
    # Ordenando os dados com base nos identificadores numÃ©ricos
    data.sort(key=lambda x: x['Num'])
    
    # Iterando sobre cada um dos tiles para exibir
    for tile_data in data:
            png_img = Image.open(tile_data['PNG'])
            with rasterio.open(tile_data['TIFF']) as src:
                tif_img = src.read(1)  # Ler a primeira banda do TIFF
                plt.imshow(tif_img, cmap='terrain')  # Usar o colormap 'terrain'
                plt.axis('off')
                # Salvar a figura em um buffer para visualizaÃ§Ã£o como PNG
                plt.savefig('tif_image.png', bbox_inches='tight', pad_inches=0)
                plt.close()
            
            png_base64 = pil_to_base64(png_img)
            with open('tif_image.png', 'rb') as f:
                tif_base64 = base64.b64encode(f.read()).decode()

            # Extraindo dados da metadata
            metadata = tile_data['metadata']
            file_name = os.path.basename(tile_data['PNG'])
            tif_file_name = os.path.basename(tile_data['TIFF'])

            # ConteÃºdo HTML, incluindo as novas informaÃ§Ãµes
            html_content = f"""
            <table style="min-width: 50px">
                <colgroup>
                    <col style="min-width: 250px;">  <!-- Ajustado para tamanho da coluna da imagem -->
                    <col style="min-width: 150px;">  <!-- Manter a coluna de anÃ¡lise menor -->
                </colgroup>
                <tbody>
                    <tr>
                        <td colspan="1" rowspan="1" style="text-align: left;">
                            <strong>Imagem: {file_name}</strong><br>
                            <img src="{png_base64}" width="150" height="auto">  <!-- Aumento da largura da imagem -->
                        </td>
                        <td colspan="1" rowspan="2" style="text-align: left; padding-left: 20px;">
                            <strong>JSON AnÃ¡lise</strong><br>
                            {tile_data['analysis'].get('output_text', "No analysis available")}<br><br>
                            
                            <strong>File Name:</strong> {metadata['file_name']}<br>
                            <strong>Latitude:</strong> {metadata['latitude']}<br>
                            <strong>Longitude:</strong> {metadata['longitude']}<br>
                            <strong>Altitude:</strong> {metadata['altitude']}<br>
                            <strong>Model Version:</strong> {tile_data['analysis'].get('model_version', "N/A")}<br>
                            <strong>Dataset ID:</strong> {tile_data['analysis'].get('dataset_id', "N/A")}
                        </td>
                    </tr>
                    <tr>
                        <td colspan="1" rowspan="1" style="text-align: left;">
                            <strong>Tile: {tif_file_name}</strong><br>
                            <img src="data:image/png;base64,{tif_base64}" width="150" height="auto">  <!-- Aumento da largura da imagem -->
                        </td>
                    </tr>
                </tbody>
            </table>
            """
            display(HTML(html_content))
    else:
        print("Nenhum tile encontrado no JSON.")

def hough_transform_apply(file_path, x_start, y_start, width, height, low_threshold=50, high_threshold=200, hough_threshold=30):
    """
    Detects lines in a geotiff image using Hough Transform.
    
    Parameters:
        file_path (str): The path to the input image file (GeoTIFF or PNG).
        x_start (int): The starting x-coordinate for the region of interest.
        y_start (int): The starting y-coordinate for the region of interest.
        width (int): The width of the region of interest.
        height (int): The height of the region of interest.
        low_threshold (int): Lower threshold for Canny edge detection.
        high_threshold (int): Upper threshold for Canny edge detection.
        hough_threshold (int): Threshold for Hough Transform line detection.
    
    Returns:
        output_image (numpy.ndarray): Image with detected lines drawn.
    """
    
    # Load the GeoTIFF image
    with rasterio.open(file_path) as src:
        image = src.read(1)

    # Normalize the image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Select the Region of Interest (ROI)
    roi = image[y_start:y_start + height, x_start:x_start + width]

    # Apply Gaussian Blur to the ROI
    roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # Canny Edge Detection on the ROI
    edges = cv2.Canny(roi_blurred, low_threshold, high_threshold)

    # Apply Hough Transform to the Edges
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)

    # Create an output image to draw the lines
    output_image = np.zeros_like(roi)

    # Draw the lines on the output image if any are detected
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Edges')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Detected Lines')
    plt.imshow(output_image, cmap='gray')
    plt.axis('off')
    
    plt.show()

    return output_image

def detect_and_overlay_lines(file_path, x_start, y_start, width, height, low_threshold=50, high_threshold=200, hough_threshold=30, center_x=None, center_y=None, size=None):
    """
    Detects lines in a GeoTIFF image and overlays them on the original image.
    
    Parameters:
        file_path (str): The path to the input image file (GeoTIFF).
        x_start (int): The starting x-coordinate for the region of interest.
        y_start (int): The starting y-coordinate for the region of interest.
        width (int): The width of the region of interest.
        height (int): The height of the region of interest.
        low_threshold (int): Lower threshold for Canny edge detection.
        high_threshold (int): Upper threshold for Canny edge detection.
        hough_threshold (int): Threshold for Hough Transform line detection.
        center_x (int): X-coordinate for the center of the square (optional).
        center_y (int): Y-coordinate for the center of the square (optional).
        size (int): Size of the square in pixels (optional).
    
    Returns:
        output_image (numpy.ndarray): Image with detected lines drawn over the original.
    """
    
    # Load the GeoTIFF image
    with rasterio.open(file_path) as src:
        original_image = src.read(1)  # Read the first band
        profile = src.profile  # Get the profile for transformation later

    # Normalize the image for processing
    original_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    # Select the Region of Interest (ROI)
    roi = original_image[y_start:y_start + height, x_start:x_start + width]
    
    # Apply Gaussian Blur to the ROI
    roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    
    # Canny Edge Detection on the ROI
    edges = cv2.Canny(roi_blurred, low_threshold, high_threshold)
    
    # Apply Hough Transform to the edges
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)

    # Create a copy of the ROI to draw the lines
    output_image = np.copy(roi)
    
    # Draw the lines on the output image if any are detected
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            # Draw lines on the output image in white color
            cv2.line(output_image, (x1, y1), (x2, y2), 255, 1)
    
    # Overlay the lines on the original image
    overlay_image = original_image.copy()
    
    # Overlay the lines (adjusting the section to draw lines in the original image context)
    overlay_image[y_start:y_start + height, x_start:x_start + width] = output_image
    
    # Visualize the overlay image with axes and the square
    plt.figure(figsize=(8, 6))
    plt.title('Overlay of Detected Lines')
    plt.imshow(overlay_image, cmap='gray')

    # Plot the square if center_x, center_y, and size are provided
    if center_x is not None and center_y is not None and size is not None:
        lower_left_x = center_x - size / 2
        lower_left_y = center_y - size / 2
        square = patches.Rectangle((lower_left_x, lower_left_y), size, size,
                                   linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(square)

        # Convert pixel coordinates to geographic coordinates using the profile's transform
        transform = profile['transform']
        center_geo_x = transform[0] * center_x + transform[2]  # Calculate geographic x
        center_geo_y = transform[4] * center_y + transform[5]  # Calculate geographic y

    plt.axis('on')  # Show axes
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout(pad=0)
    plt.show()  # Ensure the plot is shown
    
    # Print geographic coordinates
    print(f'Center Coordinates (Geographic): ({center_geo_x:.4f}, {center_geo_y:.4f})')
    
    return overlay_image


# Select the database and records to be used in the process
csv_path='/kaggle/input/archaeological-geodesy/minus5toMinus13.csv'
col = 'KLM'
col_list=['acrq58', 'acds2', 'acrs35', 'acrp11', 'acrg44']
#klm_list=['acrq58']

to_folium = selecionar_registros(csv_path, col, col_list, show_table=True)

# Compute the map center ignoring masked/invalid lat/lon
valid_lat = to_folium['lat'].data if hasattr(to_folium['lat'], 'data') else to_folium['lat']
valid_lon = to_folium['lon'].data if hasattr(to_folium['lon'], 'data') else to_folium['lon']

# Filter out masked/NaN values
valid_lat = valid_lat[~np.ma.getmaskarray(to_folium['lat'])]
valid_lon = valid_lon[~np.ma.getmaskarray(to_folium['lon'])]

center_lat = np.mean(valid_lat)
center_lon = np.mean(valid_lon)

# Create folium Map centered on average coordinates
m = folium.Map(location=['-8.81', '-62.34'], zoom_start=6)

# Iterate through rows and add markers for valid points
for idx, row in to_folium.iterrows():
    # Skip rows with masked lat or lon
    if np.ma.is_masked(row['lat']) or np.ma.is_masked(row['lon']):
        continue

    popup_text = (f"KLM: {row.get('KLM', 'N/A')}<br>"
                  f"Site Name: {row.get('Site Name', 'N/A')}<br>"
                  f"Latitude: {row['lat']:.5f}<br>"
                  f"Longitude: {row['lon']:.5f}")

    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

folium.Marker(
    location=[-8.128366, -57.898488],
    popup="Reference Image of RatanabÃ¡",
    icon=folium.Icon(color="red") 
).add_to(m)

# Display the map
m


file_path = '/kaggle/input/geotiffs/apiacas_geotiff_hh.tif' 
inspect_geotiff(file_path)


# Cell: Visualize Raster
file_path = '/kaggle/input/geotiffs/apiacas_geotiff_hh.tif'
    # First visualization in grayscale (left)
visualize_raster(file_path)


file_path = '/kaggle/input/geotiffs/apiacas_geotiff_hh.tif'
hough_transform_apply(file_path, 
                      x_start=150, 
                      y_start=180, 
                      width=50, 
                      height=50, 
                      low_threshold=5, 
                      high_threshold=20, 
                      hough_threshold=30)


file_path = '/kaggle/input/geotiffs/apiacas_geotiff_hh.tif'
geoTiff_data, profile = load_raster(file_path)
# Cell: Slope Calculation and Plot with Sigma Control
sigma_value = 1.5  # Adjustable sigma value

dtm_data = geoTiff_data[0]  # Extract elevation data from first band
slope_data = calculate_slope(dtm_data)  # Your slope calculation function

mean_val = np.nanmean(slope_data)
std_val = np.nanstd(slope_data)

# Ensure vmin is not negative since slope values are physically non-negative
vmin = max(0, mean_val - sigma_value * std_val)
vmax = mean_val + sigma_value * std_val

# Guarantee vmax > vmin, adjust if necessary (edge case)
if vmax <= vmin:
    vmax = vmin + 1e-6  # minimal positive delta

masked_slope = np.ma.masked_invalid(slope_data)
norm = colors.Normalize(vmin=vmin, vmax=vmax)

plt.figure(figsize=(10, 8))
plt.imshow(masked_slope, cmap='terrain', norm=norm)
plt.colorbar(label='Slope (degrees)')
plt.title('Slope Calculation from Digital Terrain Model (DTM) Using Variable Sigma')
plt.xlabel('Columns (pixels)')
plt.ylabel('Rows (pixels)')
plt.tight_layout(pad=0)
plt.show()


# Configurable parameters
x_center = 180
y_center = 210
radius = 50

# Compute the circular cropped region (assuming your function is defined)
cropped_region = crop_circular_region(slope_data, (x_center, y_center), radius)

# Create mask for valid (non-NaN) values
if cropped_region.ndim == 3:
    mask_valid = ~np.isnan(cropped_region).all(axis=2)
else:
    mask_valid = ~np.isnan(cropped_region)
rows_non_nan, cols_non_nan = np.where(mask_valid)

x_min, x_max = cols_non_nan.min(), cols_non_nan.max()
y_min, y_max = rows_non_nan.min(), rows_non_nan.max()

# Crop the image according to bounds
cropped_box = cropped_region[y_min:y_max+1, x_min:x_max+1]

# Replace infinities with NaN for safety
cropped_box = np.where(np.isfinite(cropped_box), cropped_box, np.nan)

# Mask invalid (NaN) values
masked_cropped_box = np.ma.masked_invalid(cropped_box)

# Obtain a copy of the 'terrain' colormap using the new matplotlib API
cmap = matplotlib.colormaps['terrain'].copy()
cmap.set_bad(color='white')  # Set color for masked (NaN) values

# Define color scaling limits ignoring NaNs
vmin = np.nanmin(cropped_box)
vmax = np.nanmax(cropped_box)

# (Optional) If slope should be non-negative, enforce vmin >= 0 here:
vmin = max(0, vmin)

# Plot the masked cropped region
plt.imshow(masked_cropped_box, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
plt.title('Full Fill Cropped Region')
plt.colorbar(label='Slope (degrees)')
plt.tight_layout(pad=0)
plt.show()


# Cell: Save file to working 
plt.imsave('slope.png', slope_data, cmap='terrain')
# Optional: confirmation message
print("Image saved as 'slope.png' in the current working directory.")
masked_data = np.ma.masked_invalid(cropped_box)
# Save the masked array as PNG, avoiding warnings
plt.imsave('slope_c.png', masked_data, cmap='terrain')
print("Image saved as 'slope_c.png'")


# Retrieve user secrets for OpenAI API
user_secrets = UserSecretsClient()
openaikey = user_secrets.get_secret("Z_Challenge_Key")

# Initialize OpenAI client
client = OpenAI(api_key=openaikey)

# Path to your image
image_path = "/kaggle/input/z-challenge-files/slope_c.png"

# Get the Base64 string of the image
base64_image = encode_image(image_path)

# Create a response using the OpenAI client
response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "You are being shown a slope image derived from LIDAR elevation data, "
                             "it reveals terrain features such as raised mounds, geometric depressions, "
                             "terracing, or other patterns that may not be visible in standard imagery. "
                             "Describe surface features in plain English."
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }
    ],
)

# Define the dataset ID
dataset_id = 'marcelocruzeta/z-challenge-files'  # Replace with your actual dataset ID

# Print dataset ID and model version in a formatted way
print("\n--- Summary of Results ---")
print(f"Dataset ID: {dataset_id}")
print(f"Model Version: {response.model}\n")  # Accessing the model version directly from response

# Print output text with a clear label
print("Generated Output Text:")
print("-------------------------")
print(response.output_text)  # Output the main text response


# Load your OpenTopography API key (adjust secret label as needed)

# Select the database and records to be used in the process
csv_path='/kaggle/input/archaeological-geodesy/minus5toMinus13.csv'
col = 'KLM'
# List of records to be extracted from the database 
col_list=['acrq58', 'acds2', 'acrs35', 'acrp11', 'acrg44']
# col_list=['acrq58']

API_Key = load_secret('OpenTopography')  # Replace with your secret label
if API_Key is None:
    raise ValueError("API key for OpenTopography not loaded.")


df = selecionar_registros(csv_path, col, col_list, show_table=True)


# Execute the process
folder_path = "/kaggle/working/SRTMGL1"
os.makedirs(folder_path, exist_ok=True)

"""
The line below has been commented out to avoid unnecessary 
use of the OpenTopography system. 
The free service serves thousands of users. 
The fewer unnecessary requests the better.
"""

# download_multiple_tiles(klm_list, df, area_km2=10, save_dir='/kaggle/working/SRTMGL1')


# Cell: GEE connection
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)
ee.Authenticate()
ee.Initialize(project='openai-to-z-challenge-461613')
print(ee.String('Hello from the Earth Engine servers!').getInfo())


# Cell: Show filtered image without clouds
# Parameters and area of interest
AOI = ee.Geometry.Point(-57.898488,  -8.128366)
START_DATE = '2018-01-01'
END_DATE = '2025-05-31'
CLOUD_FILTER = 60
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 2
BUFFER = 100

# Get the collection
collection = get_s2_sr_cld_col(AOI, START_DATE, END_DATE)

# Add Earth Engine layer method to folium
folium.Map.add_ee_layer = add_ee_layer

# Create the final collection with cloud and shadow mask applied
masked_collection = collection.map(add_cld_shdw_mask)

# Take the median to reduce remaining clouds
median_image = masked_collection.median()

# Create the visual map
center = AOI.centroid().coordinates().getInfo()
m = folium.Map(location=[center[1], center[0]], zoom_start=14)

# Display the final cloud-free image
m.add_ee_layer(median_image,
               {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1.1},
               'Sentinel-2 cloud-free image with enhanced mask')
folium.Marker(
    location=[-8.125366, -57.898488],
    popup="Reference Image of Supposed RatanabÃ¡",
    icon=folium.Icon(color="red") 
).add_to(m)

# Show map
m


# Parameters and area of interest
AOI = ee.Geometry.Point(-67.123219, -9.716453)
START_DATE = '2018-01-01'
END_DATE = '2025-05-31'
CLOUD_FILTER = 60
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 2
BUFFER = 100

# Get the collection
col = get_s2_sr_cld_col(AOI, START_DATE, END_DATE)

# Create the final collection with cloud and shadow mask applied
col_masked = col.map(add_cld_shdw_mask)

# Take the median to reduce remaining clouds
median_img = col_masked.median()

# Select the database and records to be used in the process
csv_path='/kaggle/input/archaeological-geodesy/minus5toMinus13.csv'
col = 'KLM'
# List of records to be extracted from the database 
col_list=['acrq58', 'acds2', 'acrs35', 'acrp11', 'acrg44']
# col_list=['acrq58']
selected_records = selecionar_registros(csv_path, col, col_list, show_table=True)


for idx, row in selected_records.iterrows():
    lon = row['lon']
    lat = row['lat']
    nome = row['KLM']
    # Commented not to process files in the testing phase
    # processar_registro(lon, lat, nome)


user_secrets = UserSecretsClient()
secret_value_1 = user_secrets.get_secret("GEEE Folder")

url = f'https://drive.google.com/drive/folders/{secret_value_1}'
 # Commented not to process files in the testing phase
# gdown.download_folder(url, quiet=False)


# Path to the directory with files and json file
directory = '/kaggle/input/z-challenge-files/GEEE'
json_file = '/kaggle/input/z-challenge-files/analysis_logs.json'
#display_analysis(directory, json_file)


input_directory = '/kaggle/input/z-challenge-files/GEEE'
output_directory = '/kaggle/working/GEEE'
log_file = os.path.join(output_directory, 'analysis_logs.json')

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Process all images
# Commented not to process files in the testing phase
# process_images(input_directory, output_directory, log_file)


# Path to the directory with files
directory = '/kaggle/input/z-challenge-files/GEEE'
json_directory = '/kaggle/input/z-challenge-files'

# Read the JSON file
with open(os.path.join(json_directory, 'analysis_logs.json')) as f:
    json_data = json.load(f)

# Inspect the type of json_data
if isinstance(json_data, dict):
    tiles = json_data.get('tiles', [])
elif isinstance(json_data, list):
    tiles = json_data  # Assuming the entire list represents tiles
else:
    tiles = []

# If tiles are found, we can proceed
if tiles:
    data = []
    for tile in tiles:
        if isinstance(tile, dict) and 'metadata' in tile:
            file_name = tile['metadata'].get('file_name', 'unknown.tif')
            analysis_text = tile.get('analysis', {}).get('output_text', "No analysis available")

            tile_png = os.path.join(directory, file_name.replace('.tif', '.png'))
            tile_tif = os.path.join(directory, file_name)

            # Verify file existence
            if os.path.exists(tile_png) and os.path.exists(tile_tif):
                data.append((tile_png, tile_tif, analysis_text, file_name))

    # Create a DataFrame for easier visualization
    df = pd.DataFrame(data, columns=['PNG', 'TIFF', 'Analysis', 'FileName'])

    # Extract numerical identifiers to sort
    df['Num'] = df['FileName'].str.extract(r'(\d+\.\d+)').astype(float)
    df = df.sort_values('Num')  # Sort by the numeric identifier
    df = df.drop(columns=['Num', 'FileName'])  # Drop the helper column

    # Display each row of the DataFrame
    for index, row in df.iterrows():
        display(display_images(row))
else:
    print("No tiles found in the JSON.")


# Load the CSV file with UTF-7 encoding
# Adjust 'data.csv' to the name of your file
df = pd.read_csv('/kaggle/input/archaeological-geodesy/minus5toMinus10.csv', encoding='utf-7')

# Filter the points where longitude is between -78 and -68
df_filtered = df[(df['lon'] >= -78) & (df['lon'] <= -68)]

# Check if there are points after filtering
# print(df_filtered)

# Create a map centered at the average coordinates of the filtered points
if not df_filtered.empty:
    latitude_mean = df_filtered['lat'].mean()
    longitude_mean = df_filtered['lon'].mean()

    mapa = folium.Map(location=['-8.0', '-71.50'], zoom_start=6)

    # Add markers for each filtered location
    for index, row in df_filtered.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=row['Site Name'],
        ).add_to(mapa)

    # Display the map
    mapa
else:
    print("No points within the specified longitude range.")

mapa


# Define the southwest and northeast corners
southwest = [-9.25, -73.00]  # Southwest corner (SW)
northeast = [-7.35, -70.35]  # Northeast corner (NE)
zoom_start = 7

# Constants
km_per_degree_latitude = 111.32  # km per degree of latitude

# Calculate the approximate side length for a square of approximately 300 kmÂ²
side_length_km = 17.32  # Side length (in km)
lat_length = side_length_km / km_per_degree_latitude  # in degrees for latitude

# Function to calculate km per degree of longitude based on latitude
def km_per_longitude(latitude):
    return 111.32 * abs(math.cos(math.radians(latitude)))

# Prepare the map centered at the midpoint of the defined area
map_center = [(-9.25 + -7.35) / 2, (-73.00 + -70.35) / 2]
m = folium.Map(location=map_center, zoom_start=zoom_start)

# Initialize a counter for square numbering
square_counter = 1

# Generate squares and add them to the map, starting from NW to SE
lat = northeast[0]  # Start from the northernmost latitude
while lat > southwest[0]:  # Go until the southernmost latitude
    long = southwest[1]  # Reset longitude to the southwest longitude for each new row
    while long < northeast[1]:  # Go until the easternmost longitude
        # Calculate the length of longitude in degrees based on the current latitude
        long_length = side_length_km / km_per_longitude(lat)  # in degrees for longitude
        
        # Define bounds for the rectangle
        bounds = [[lat, long], [lat - lat_length, long + long_length]]  # Adjust lat for downwards
        Rectangle(bounds=bounds, color='blue', fill=True, fill_opacity=0.2).add_to(m)

        # Calculate the center of the square for numbering
        center_lat = lat - (lat_length / 2)  # Adjust for downwards
        center_long = long + (long_length / 2)

        # Add a marker with the square number at the center
        folium.Marker(
            location=[center_lat, center_long],
            icon=folium.DivIcon(html=f'<div>{square_counter}</div>')
        ).add_to(m)

        long += long_length  # Move to the next square
        square_counter += 1  # Increment the square number
    lat -= lat_length  # Move to the next row (downwards)

# Display the map inline
m


# Define the southwest and northeast corners
southwest = [-9.25, -73.00]  # Southwest corner (SW)
northeast = [-7.35, -70.35]  # Northeast corner (NE)

# Constants
km_per_degree_latitude = 111.32  # km per degree of latitude

# Calculate the approximate side length for a square of approximately 300 kmÂ²
side_length_km = 17.32  # Side length (in km)
lat_length = side_length_km / km_per_degree_latitude  # in degrees for latitude

# Function to calculate km per degree of longitude based on latitude
def km_per_longitude(latitude):
    return 111.32 * abs(math.cos(math.radians(latitude)))

# Initialize a list to hold tile data with corner coordinates
tile_data = []  # List to hold tile information

# Generate tiles, starting from the northernmost latitude
lat = northeast[0]
while lat > southwest[0]:  # Go until the southernmost latitude
    long = southwest[1]  # Reset longitude to southwest longitude for each new row
    while long < northeast[1]:  # Go until the easternmost longitude
        # Calculate the length of longitude in degrees based on the current latitude
        long_length = side_length_km / km_per_longitude(lat)  # in degrees for longitude
        
        # Calculate all four corner coordinates
        nw = [lat, long]                             # Northwest corner
        ne = [lat, long + long_length]             # Northeast corner
        se = [lat - lat_length, long + long_length] # Southeast corner
        sw = [lat - lat_length, long]               # Southwest corner

        # Append tile information to the list
        tile_data.append({
            'Tile': len(tile_data) + 1,
            'NW_Latitude': nw[0],
            'NW_Longitude': nw[1],
            'NE_Latitude': ne[0],
            'NE_Longitude': ne[1],
            'SE_Latitude': se[0],
            'SE_Longitude': se[1],
            'SW_Latitude': sw[0],
            'SW_Longitude': sw[1],
            'Processed': ''  # Initialize as unprocessed
        })

        long += long_length  # Move to the next tile
    lat -= lat_length  # Move to the next row (downwards)

# Convert tile data to a DataFrame for better visualization and manipulation
tile_df = pd.DataFrame(tile_data)

# Display the DataFrame
print(tile_df.head())  # Show the first few rows for validation

# Save the DataFrame to a CSV file
tile_df.to_csv('tile_data.csv', index=False)  # Do not include row indices


csv_path = '/kaggle/input/archaeological-geodesy/tile_data_01.csv'
tile_numbers = [1]  # Replace with the desired tile numbers
save_dir = '/kaggle/working/'

# commented not to use OpenTopography site in vain
# download_tiles_from_csv(csv_path, tile_numbers, save_dir)


file_path = '/kaggle/input/images-of-tiles-1-to-17-with-analysis/1-17/tile_15.0.tif'
visualize_raster(file_path)


# Here we only test the selection process
csv_path = '/kaggle/input/archaeological-geodesy/tile_data_01.csv'
start_tile = 2
end_tile = 3
selected_tiles = filter_tiles_by_range(csv_path, start_tile, end_tile)


csv_path = '/kaggle/input/archaeological-geodesy/tile_data_01.csv'
start_tile = 1 # First tile to be donwnloaded
end_tile = 17 # Last tile to be donwnloaded

# Create directory names based on the start and end tiles
base_dir = f"{start_tile}-{end_tile}"
save_dir = base_dir
output_dir = base_dir
log_file = f"{base_dir}/{base_dir}.json"

# Commented not to process files in the testing phase
# process_range_images(csv_path, start_tile, end_tile, save_dir, output_dir, log_file)


# Path to the directory with files and json file
directory = '/kaggle/input/images-of-tiles-1-to-17-with-analysis/1-17'
json_file = '/kaggle/input/images-of-tiles-1-to-17-with-analysis/1-17/1-17.json'
display_analysis(directory, json_file)


# Apply Hough Transform and overlay image
file_path = '/kaggle/input/images-of-tiles-1-to-17-with-analysis/1-17/tile_15.0.tif'  # (str): The path to the input image file (GeoTIFF).
x_start = 78 # (int): The starting x-coordinate for the region of interest.
y_start = 4 # (int): The starting y-coordinate for the region of interest.
width = 40 # (int): The width of the region of interest.
height = 40 # (int): The height of the region of interest.
low_threshold = 20 # (int): Lower threshold for Canny edge detection (default=50).
high_threshold = 20 # (int): Upper threshold for Canny edge detection (default=200).
hough_threshold = 26 # (int): Threshold for Hough Transform line detection (default=30).
center_x = 97 # (int, optional): X-coordinate for the center of the square (default=None).
center_y = 20 # (int, optional): Y-coordinate for the center of the square (default=None).
size = 20 # (int, optional): Size of the square in pixels (default=None).

detect_and_overlay_lines(file_path, x_start, y_start, width, height, low_threshold, high_threshold, hough_threshold, center_x, center_y, size)


file_path = '/kaggle/input/images-of-tiles-1-to-17-with-analysis/1-17/tile_15.0.tif'
sigma_value = 0.8

# Coordinates for the first square
center_x = 97
center_y = 19
size = 20
# Coordinates for the second square
#center_x = 100
#center_y = 70
#size = 30

slope_calculation(file_path, sigma_value, center_x, center_y, size)


# Apply Hough Transform and overlay image
file_path = '/kaggle/input/images-of-tiles-1-to-17-with-analysis/1-17/tile_15.0.tif'  # (str): The path to the input image file (GeoTIFF).
x_start = 1 # (int): The starting x-coordinate for the region of interest.
y_start = 1 # (int): The starting y-coordinate for the region of interest.
width = 200 # (int): The width of the region of interest.
height = 200 # (int): The height of the region of interest.
low_threshold = 10 # (int): Lower threshold for Canny edge detection (default=50).
high_threshold = 30 # (int): Upper threshold for Canny edge detection (default=200).
hough_threshold = 115 # (int): Threshold for Hough Transform line detection (default=30).
center_x = 100 # (int, optional): X-coordinate for the center of the square (default=None).
center_y = 90 # (int, optional): Y-coordinate for the center of the square (default=None).
size = 30 # (int, optional): Size of the square in pixels (default=None).

detect_and_overlay_lines(file_path, x_start, y_start, width, height, low_threshold, high_threshold, hough_threshold, center_x, center_y, size)


# Define the path to the directory with files
directory = '/kaggle/input/images-of-tiles-1-to-17-with-analysis/discoveries'

# Define a dictionary for image filenames and their descriptions
descriptions = {
    "disc.png": "The image of our square in the the right place.</br> Try using these coordinates in your GEP and tell me if what you see in the images when you zoom in on the location isn't surprising. (-70.72, -7.37)",
    "disc_6png.png": "The image of the place as we see in GEP.</br> The objective here is just to compare the definition of the image extracted from GEP with what we currently have from GEE, the quality is incomparable.",
    "disc_7png.png": "The image of the place as we see in GEP, more zoom.</br> The objective here is just to compare the definition of the image extracted from GEP with what we currently have from GEE, the quality is incomparable.",
    "disc_8png.png": "The image of the place as we see in GEP, more zoom.</br> The objective here is just to compare the definition of the image extracted from GEP with what we currently have from GEE, the quality is incomparable.",
    "disc_3.png": "This 3D image from GEP is very intriguing.</br> It looks like there was a landslide, perhaps due to a lot of rain, which is showing what appears to be a stone slab or something like that. What do you think? Or has someone already been there digging to find the lost treasures in the middle of the jungle?",
    "disc_5.png": "I have seen incredible things while flying over these forests. However, for obvious reasons, I could not stop and take a look.</br>On this route, Cruzeiro do Sul - TarauacÃ¡, it is a flight of more than an hour in a small plane flying over the forest, there is nothing but forest to see.</br>I found these images so unusual that I even sent an email to IPHAN notifying them of the Finding.</br>The first thing we do when we find an archaeological site is to notify the responsible government agency.",
    
    # Add more images and descriptions as needed
}

img_dir_text(directory, descriptions)


# Send the image for analysis with our improved function analyze_single_png
# How to use it:
prompt = ("We show you a png image derived from LIDAR elevation data before, and you"
          "concluded it suggested possible archaeological or anthropogenic modification of the terrain."
          "Now we are bringing an aerial image with deail of the same location."
          "Can you describe this image in plain English based on the information I have given you?")
png_file = "/kaggle/input/images-of-tiles-1-to-17-with-analysis/discoveries/disc_3.png"

# Analyze the PNG file
result = analyze_single_png(prompt, png_file)

# Print results
print("\n--- Summary of Results ---")
print(f"Dataset ID: {result['Dataset ID']}")
print(f"Model Version: {result['Model Version']}\n")
print("Generated Output Text:")
print("-------------------------")
print(result['Generated Output Text'])


# Define the path to the directory with files
directory = '/kaggle/input/images-of-tiles-1-to-17-with-analysis/findings_2'

# Define a dictionary for image filenames and their descriptions
descriptions = {
    "__results___61_1.png": "Just south of the previously found site we can see a large square.",
    "big_square.png": "This is a picture of the GEP, if you have a little perseverence the square becomes visible before your eyes!",
    "tile_15.png": "This is an image of tile 15 that we are currently analyzing.",
    
    # Add more images and descriptions as needed
}

img_dir_text(directory, descriptions)


# Apply Hough Transform and overlay image
file_path = '/kaggle/input/images-of-tiles-1-to-17-with-analysis/1-17/tile_16.0.tif'  # (str): The path to the input image file (GeoTIFF).
x_start = 1 # (int): The starting x-coordinate for the region of interest.
y_start = 1 # (int): The starting y-coordinate for the region of interest.
width = 200 # (int): The width of the region of interest.
height = 200 # (int): The height of the region of interest.
low_threshold = 10 # (int): Lower threshold for Canny edge detection (default=50).
high_threshold = 40 # (int): Upper threshold for Canny edge detection (default=200).
hough_threshold = 117 # (int): Threshold for Hough Transform line detection (default=30).
center_x = 92 # (int, optional): X-coordinate for the center of the square (default=None).
center_y = 96 # (int, optional): Y-coordinate for the center of the square (default=None).
size = 40 # (int, optional): Size of the square in pixels (default=None).

detect_and_overlay_lines(file_path, x_start, y_start, width, height, low_threshold, high_threshold, hough_threshold, center_x, center_y, size)



# Define the path to your image file
image_path = '/kaggle/input/images-of-tiles-1-to-17-with-analysis/disc_9.png'  # Adjust the path as necessary
# Define the text below the image
text =("Here we can see a rectangle with a side measurement of "+
           "approximately 240x300 meters. It is covered by vegetation "+
           "but is clearly visible. It can be seen on our map in item 26 "+
           "at marker number 4, but not as clearly as here.")

image_text(image_path, text)


file_path = '/kaggle/input/images-of-tiles-1-to-17-with-analysis/1-17/tile_16.0.tif'

center_x = 90
center_y = 100
size = 30

visualize_raster_dtm_square(file_path, center_x, center_y, size)


# Apply Hough Transform and overlay image
file_path = '/kaggle/input/images-of-tiles-1-to-17-with-analysis/1-17/tile_13.0.tif'  # (str): The path to the input image file (GeoTIFF).
x_start = 1 # (int): The starting x-coordinate for the region of interest.
y_start = 1 # (int): The starting y-coordinate for the region of interest.
width = 200 # (int): The width of the region of interest.
height = 200 # (int): The height of the region of interest.
low_threshold = 10 # (int): Lower threshold for Canny edge detection (default=50).
high_threshold = 40 # (int): Upper threshold for Canny edge detection (default=200).
hough_threshold = 117 # (int): Threshold for Hough Transform line detection (default=30).
center_x = 105 # (int, optional): X-coordinate for the center of the square (default=None).
center_y = 87 # (int, optional): Y-coordinate for the center of the square (default=None).
size = 30 # (int, optional): Size of the square in pixels (default=None).

detect_and_overlay_lines(file_path, x_start, y_start, width, height, low_threshold, high_threshold, hough_threshold, center_x, center_y, size)


file_path = '/kaggle/input/images-of-tiles-1-to-17-with-analysis/1-17/tile_13.0.tif'
sigma_value = 0.8

# Coordinates for the first square
center_x = 105
center_y = 87
size = 30
# Coordinates for the second square
#center_x = 100
#center_y = 70
#size = 30

slope_calculation(file_path, sigma_value, center_x, center_y, size)


# Define the path to your image file
image_path = '/kaggle/input/images-of-tiles-1-to-17-with-analysis/disc_10.png'  # Adjust the path as necessary
# Define the text below the image
text =("Here we can see a square with a side measurement of "+
           "approximately 350 meters. It is not clearly visible. "
           )

image_text(image_path, text)


# Cell: Show filtered image without clouds
# Parameters and area of interest
AOI = ee.Geometry.Point(-70.78,  -7.43)
START_DATE = '2018-01-01'
END_DATE = '2025-05-31'
CLOUD_FILTER = 60
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 2
BUFFER = 100

# Get the collection
collection = get_s2_sr_cld_col(AOI, START_DATE, END_DATE)

# Add Earth Engine layer method to folium
folium.Map.add_ee_layer = add_ee_layer

# Create the final collection with cloud and shadow mask applied
masked_collection = collection.map(add_cld_shdw_mask)

# Take the median to reduce remaining clouds
median_image = masked_collection.median()

# Create the visual map
center = AOI.centroid().coordinates().getInfo()
m = folium.Map(location=[center[1], center[0]], zoom_start=8)

# Display the final cloud-free image
m.add_ee_layer(median_image,
               {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1.1},
               'Sentinel-2 cloud-free image with enhanced mask')
icon_1=create_numbered_icon(1)
icon_2=create_numbered_icon(2)
icon_3=create_numbered_icon(3)
icon_4=create_numbered_icon(4)
folium.Marker(
    location=[-7.3662, -70.7229], 
    popup="Possible Ancient Ruin Found by Cruzeta",
    icon=icon_1
).add_to(m)

folium.Marker(
    location=[-7.4246, -70.7204],
    popup="Possible geoglyph",
    icon=icon_2 
).add_to(m)

folium.Marker(
    location=[-7.4221, -71.0304],
    popup="Possible Possible geoglyph",
    icon=icon_3 
).add_to(m) 

folium.Marker(
    location=[-7.4355, -70.5743],
    popup="Visible Big Rectangle",
    icon=icon_4 
).add_to(m) 

# Show map
m


# Define the southwest and northeast corners for the new area
southwest = [-7.55, -71.30]  # Southwest corner (SW)
northeast = [-7.00, -70.20]  # Northeast corner (NE)
zoom_start = 9

# Constants
km_per_degree_latitude = 111.32  # km per degree of latitude

# Calculate the approximate side length for a square of approximately 300 kmÂ²
side_length_km = 17.32  # Side length (in km)
lat_length = side_length_km / km_per_degree_latitude  # in degrees for latitude

# Function to calculate km per degree of longitude based on latitude
def km_per_longitude(latitude):
    return 111.32 * abs(math.cos(math.radians(latitude)))

# Prepare the map centered at the midpoint of the defined area
map_center = [(-7.55 + -7.00) / 2, (-71.30 + -70.20) / 2]
m = folium.Map(location=map_center, zoom_start=zoom_start)

# Initialize a counter for square numbering
square_counter = 1

# Generate squares and add them to the map, starting from NW to SE
lat = northeast[0]  # Start from the northernmost latitude
while lat > southwest[0]:  # Go until the southernmost latitude
    long = southwest[1]  # Reset longitude to the southwest longitude for each new row
    while long < northeast[1]:  # Go until the easternmost longitude
        # Calculate the length of longitude in degrees based on the current latitude
        long_length = side_length_km / km_per_longitude(lat)  # in degrees for longitude
        
        # Define bounds for the rectangle
        bounds = [[lat, long], [lat - lat_length, long + long_length]]  # Adjust lat for downward movement
        
        # Create the rectangle
        folium.Rectangle(bounds=bounds, color='blue', fill=True, fill_opacity=0.2).add_to(m)
        
        # Calculate the center of the square for numbering
        center_lat = lat - (lat_length / 2)  # Adjust for downward placement
        center_long = long + (long_length / 2)

        # The location of the anomalies we found until now
        icon_1=create_numbered_icon(1)
        icon_2=create_numbered_icon(2)
        icon_3=create_numbered_icon(3)
        icon_4=create_numbered_icon(4)
        folium.Marker(
            location=[-7.3662, -70.7229], 
            popup="Possible Ancient Ruin Found by Cruzeta",
            icon=icon_1
        ).add_to(m)
        
        folium.Marker(
            location=[-7.4246, -70.7204],
            popup="Possible geoglyph",
            icon=icon_2 
        ).add_to(m)
        
        folium.Marker(
            location=[-7.4221, -71.0304],
            popup="Possible Possible geoglyph",
            icon=icon_3 
        ).add_to(m) 
        
        folium.Marker(
            location=[-7.4355, -70.5743],
            popup="Visible Big Rectangle",
            icon=icon_4 
        ).add_to(m)
        
        # Add a marker with the square number at the center
        folium.Marker(
            location=[center_lat, center_long],
            icon=folium.DivIcon(html=f'<div>{square_counter}</div>')
        ).add_to(m)
        
        long += long_length  # Move to the next square
        square_counter += 1  # Increment the square number
    lat -= lat_length  # Move to the next row (downward)

# Display the map inline
m

