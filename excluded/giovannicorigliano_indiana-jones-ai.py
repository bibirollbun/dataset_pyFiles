from openai import OpenAI
from kaggle_secrets import UserSecretsClient

# === SETUP KEYS ===
user_secrets = UserSecretsClient()
OPENAI_API_KEY = user_secrets.get_secret("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# === STUDY AREA PARAMETERS  ===
CENTER_LAT_STUDIO     = -12.9997   # decimali
CENTER_LON_STUDIO     = -64.5      # decimali
SIDE_LENGTH_KM = 5          

# === 1) FUNZIONE: SEARCH REFERENCES ===
def search_archaeological_sources(lat, lon, side_km):
    query = f"""
    You are an expert in Amazonian archaeology.

    Given an NDVI + LIDAR anomaly around lat={lat}, lon={lon} (Â±{side_km/2} km),
    return a list of historical or academic sources that describe:
      - pre-Columbian earthworks in this region
      - raised fields, mounds, canals or causeways
      - any nearby settlements mentioned in academic literature

    For each source include:
      â€¢ title and authors 
      â€¢ year 
      â€¢ short summary (1â€“2 sentences) 
      â€¢ bibliographic references in APA 
      â€¢ if available, approximate coordinates (lat, lon)
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional Amazon archaeologist."},
            {"role": "user", "content": query}
        ],
        temperature=0.3
    )
    return resp.choices[0].message.content

# === 2) FUNZIONE: ESTRAI COORDINATE & METADATI ===
# Definiamo lo â€œschemaâ€� della funzione che GPT dovrÃ  restituire
functions = [
    {
        "name": "extract_coordinates",
        "description": "Extract latitude, longitude and metadata from an archaeological reference text",
        "parameters": {
            "type": "object",
            "properties": {
                "sites": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "latitude":   {"type": "number"},
                            "longitude":  {"type": "number"},
                            "site_name":  {"type": "string"},
                            "description":{"type": "string"},
                            "reference":  {"type": "string"}
                        },
                        "required": ["latitude","longitude","site_name"]
                    }
                }
            },
            "required": ["sites"]
        }
    }
]

def parse_to_structured(raw_text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract geo-metadata from archaeological reference text."},
            {"role": "user",   "content": raw_text}
        ],
        functions=functions,
        function_call={"name": "extract_coordinates"}
    )
    # la risposta sarÃ  nel campo `function_call.arguments` come JSON
    import json
    args = resp.choices[0].message.function_call.arguments
    data = json.loads(args)
    return data["sites"]

# === 3) EXECUTE PIPELINE ===
if __name__ == "__main__":
    # 1) ottengo il testo con i riferimenti
    raw_refs = search_archaeological_sources(CENTER_LAT_STUDIO, CENTER_LON_STUDIO, SIDE_LENGTH_KM)
    print("ğŸ—’ï¸� Riferimenti grezzi:\n", raw_refs, "\n")

    # 2) estraggo la lista strutturata
    sites = parse_to_structured(raw_refs)
    print("ğŸ“� Lista di siti con coordinate e metadati:")
    for s in sites:
        print(f" - {s['site_name']} @ ({s['latitude']}, {s['longitude']}): {s.get('description','')} [{s['reference']}]")

sites[0].get('latitude')


pip install -qq rasterio numpy matplotlib geopandas sentinelhub


#SEARCH
CENTER_LAT     = sites[0].get('latitude')   # decimal degrees 
CENTER_LON     = sites[0].get('longitude')  # decimal degrees
SIDE_LENGTH_KM = 5     # kilometers


#SRTMGL3 new

import math
import requests
from kaggle_secrets import UserSecretsClient

# Carica la tua API key da Kaggle Secrets
user_secrets = UserSecretsClient()
API_KEY = user_secrets.get_secret("OpenTopography")

def download_dem_by_center(
    center_lat: float,
    center_lon: float,
    side_length_km: float,
    demtype: str,
    output_format: str,
    api_key: str,
    output_filename: str
):
    """
    Purpose:
      Download a global Digital Elevation Model (DEM) GeoTIFF from the
      OpenTopography API for a square area centered on a given lat/lon,
      with sides of specified length (km), and save it locally.

    Endpoint:
      GET https://portal.opentopography.org/API/globaldem

    Parameters:
      center_lat     : Latitude of square center (decimal degrees)
      center_lon     : Longitude of square center (decimal degrees)
      side_length_km : Side length of square (kilometers)
      demtype        : DEM product ID (e.g. 'SRTMGL3')
      output_format  : Format (e.g. 'GTiff')
      api_key        : Your OpenTopography API key
      output_filename: Local filename for the GeoTIFF
    """
    # Earthâ€™s radius [km]
    R = 6371.0
    half = side_length_km / 2.0

    # Convert halfâ€�side from km to degrees lat/lon
    delta_lat = (half / R) * (180.0 / math.pi)
    delta_lon = (half / (R * math.cos(math.radians(center_lat)))) * (180.0 / math.pi)

    # Bounding box
    south = center_lat - delta_lat
    north = center_lat + delta_lat
    west  = center_lon - delta_lon
    east  = center_lon + delta_lon

    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": demtype,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": output_format,
        "API_Key": api_key
    }

    # Stream download to avoid high memory use
    resp = requests.get(url, params=params, stream=True)
    resp.raise_for_status()

    # Save file
    with open(output_filename, "wb") as f:
        for chunk in resp.iter_content(8192):
            if chunk:
                f.write(chunk)

    print(f"Saved DEM ({demtype}) for {side_length_km} km square around "
          f"({center_lat}, {center_lon}) â†’ {output_filename}")


if __name__ == "__main__":
 
    download_dem_by_center(
        center_lat     = CENTER_LAT,
        center_lon     = CENTER_LON,
        side_length_km = SIDE_LENGTH_KM,
        demtype        = "SRTMGL1", #SRTMGL3 
        output_format  = "GTiff",
        api_key        = API_KEY,
        output_filename= "dem_out.tiff"
    )



import os
import rasterio
import matplotlib.pyplot as plt

# Base directory dove si trovano i GeoTIFF
BASE_DIR = "/kaggle/working"

# Genera la lista completa dei percorsi per i 12 file
band_files = [
    os.path.join(BASE_DIR, f"dem_out.tiff")
]

# Visualizza ogni banda in una figura separata
for filepath in band_files:
    if not os.path.exists(filepath):
        print(f"File non trovato: {filepath}")
        continue

    with rasterio.open(filepath) as src:
        data = src.read(1)
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='gray')
    plt.title(os.path.basename(filepath))
    plt.axis('off')

plt.tight_layout()
plt.show()


##Sentinel-2 L2A (No athmosphere) TOKEN
import requests
import json
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
sentinel_id = user_secrets.get_secret("SENTINEL_ID")
sentinel_secret = user_secrets.get_secret("SENTINEL_SECRET")

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
sentinel_id = user_secrets.get_secret("SENTINEL_ID")
sentinel_secret = user_secrets.get_secret("SENTINEL_SECRET")

def get_sentinel_token(
    token_url: str,
    client_id: str,
    client_secret: str
) -> str:
    """
    Purpose:
      Obtain an OAuth2 Bearer token from Sentinel Hub using the
      Client Credentials grant.

    Parameters:
      token_url : str
        The Sentinel Hub token endpoint.
      client_id : str
        Your OAuth2 client ID.
      client_secret : str
        Your OAuth2 client secret.

    Returns:
      Access token string for use in Authorization headers.
    """
    # Form fields for OAuth2 Client Credentials grant
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    # Send POST to token endpoint
    resp = requests.post(token_url, data=data, headers=headers)
    resp.raise_for_status()
    token_response = resp.json()

    # Extract access_token from JSON response
    return token_response["access_token"]


#Sentinel-2 L2A (No athmosphere)
import requests
import json
import rasterio
import math

def create_square_coords_km(
    center_lat: float,
    center_lon: float,
    side_length_km: float
) -> list[list[float]]:
    """
    Purpose:
      Generate the coordinates of a square polygon centered on a given
      lat/lon point, with sides of a specified length in kilometers.

    Parameters:
      center_lat      : float
        Latitude of the squareâ€™s center (decimal degrees).
      center_lon      : float
        Longitude of the squareâ€™s center (decimal degrees).
      side_length_km  : float
        Length of each side of the square, in kilometers.

    Returns:
      List of five [lon, lat] pairs defining the squareâ€™s corners
      (closed ring: first == last), in counter-clockwise order.
    """
    R = 6371.0  # Earth radius in kilometers
    half = side_length_km / 2.0

    # Convert halfâ€�side to decimal degrees latitude:
    delta_lat = (half / R) * (180.0 / math.pi)
    # Convert halfâ€�side to decimal degrees longitude at this latitude:
    delta_lon = (half / (R * math.cos(math.radians(center_lat)))) * (180.0 / math.pi)

    bl = [center_lon - delta_lon, center_lat - delta_lat]  # bottom-left
    tl = [center_lon - delta_lon, center_lat + delta_lat]  # top-left
    tr = [center_lon + delta_lon, center_lat + delta_lat]  # top-right
    br = [center_lon + delta_lon, center_lat - delta_lat]  # bottom-right

    return [bl, tl, tr, br, bl]  # closed ring


def process_sentinel_data(
    request_payload: dict,
    evalscript: str,
    bearer_token: str,
    multi_band_filename: str = "sentinel_multiband.tiff"
):
    """
    Purpose:
      Send a POST to Sentinel Hubâ€™s Process API to run a custom evalscript
      over Sentinel-2 L2A imagery for a specified area and time range,
      and save the returned GeoTIFF locally.

    API Endpoint:
      POST https://services.sentinel-hub.com/api/v1/process

    Parameters:
      request_payload : dict
        The JSON body defining input bounds, data filters, and output specs.
      evalscript : str
        A JavaScript snippet (Evalscript v3) that selects which bands to return.
      bearer_token : str
        OAuth2 Bearer token for Authorization header.
      output_filename : str
        Local filename to save the GeoTIFF.
    """
    url = "https://services.sentinel-hub.com/api/v1/process"

    # HTTP headers include OAuth2 Bearer token
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }

    # Form fields: 'request' as JSON string, 'evalscript' as JS code
    files = {
        "request": (None, json.dumps(request_payload), "application/json"),
        "evalscript": (None, evalscript, "application/javascript")
    }

  

    # 1) Download the multi-band GeoTIFF
    resp = requests.post(url, headers=headers, files=files, stream=True)
    resp.raise_for_status()
    with open(multi_band_filename, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Multi-band TIFF saved as '{multi_band_filename}'")

    # 2) Open with rasterio and write each band out separately
    with rasterio.open(multi_band_filename) as src:
        band_count = src.count
        meta = src.meta.copy()
        for i in range(1, band_count + 1):
            band_meta = meta.copy()
            band_meta.update({
                "count": 1
            })
            band_name = f"B{str(i).zfill(2)}"  # e.g. B01, B02, â€¦ B12
            out_fname = f"sentinel_{band_name}.tiff"
            with rasterio.open(out_fname, "w", **band_meta) as dst:
                dst.write(src.read(i), 1)
            print(f"  â†’ Wrote band {i} to '{out_fname}'")

if __name__ == "__main__":
    # OAuth2 token endpoint for Sentinel Hub
    TOKEN_URL    = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
    CLIENT_ID    = sentinel_id      # Replace with your client ID
    CLIENT_SECRET= sentinel_secret  # Replace with your client secret

    # Fetch Bearer token
    token = get_sentinel_token(
        token_url=TOKEN_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    # Generate square polygon coordinates
    square_coords = create_square_coords_km(
        center_lat=CENTER_LAT,
        center_lon=CENTER_LON,
        side_length_km=SIDE_LENGTH_KM
    )
    print('LAT: ' + str(CENTER_LAT) + ' - LON: ' + str(CENTER_LON))
    # Define the JSON payload for the processing request
    #"http://www.opengis.net/def/crs/EPSG/0/4326"
    #"http://www.opengis.net/def/crs/OGC/1.3/CRS84"
    REQUEST_PAYLOAD = {
        "input": {
            "bounds": {
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [square_coords]
                }
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": "2022-05-01T00:00:00Z",
                            "to":   "2022-05-31T00:00:00Z"
                        },
					"mosaickingOrder": "mostRecent",
					"previewMode": "EXTENDED_PREVIEW",
					"maxCloudCoverage": 1
                    },
                    "processing": {
                        "harmonizeValues": "false",
                        "upsampling": "NEAREST",
    					"downsampling": "NEAREST"
                    },
                    "type": "S2L2A"

                }
            ]
        },
        "output": {
            "width": 2500,
            "height": 2500,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    }
                }
            ]
        }
    }

    # Define the Evalscript (v3) selecting 12 Sentinel-2 bands, unsigned 16-bit
    EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"],
      units: "DN"
    }],
    output: {
      id: "default",
      bands: 12,
      sampleType: SampleType.FLOAT32
    }
  };
}

function evaluatePixel(sample) {
  // Return array of band values in the same order as 'bands'
  return [
    sample.B01, sample.B02, sample.B03, sample.B04,
    sample.B05, sample.B06, sample.B07, sample.B08,
    sample.B8A, sample.B09, sample.B11, sample.B12
  ];
}"""

    OUTPUT_FILE = "sentinel2_202210_oct.tiff"  # Desired local filename

    # Process data using the freshly obtained token
    process_sentinel_data(
        request_payload=REQUEST_PAYLOAD,
        evalscript=EVALSCRIPT,
        bearer_token=token,
        multi_band_filename=OUTPUT_FILE
    )



import os
import rasterio
import matplotlib.pyplot as plt

# Base directory dove si trovano i GeoTIFF
BASE_DIR = "/kaggle/working"

# Genera la lista completa dei percorsi per i 12 file
band_files = [
    os.path.join(BASE_DIR, f"sentinel_B{str(i).zfill(2)}.tiff")
    for i in range(1, 13)
]

# Visualizza ogni banda in una figura separata
for filepath in band_files:
    if not os.path.exists(filepath):
        print(f"File non trovato: {filepath}")
        continue

    with rasterio.open(filepath) as src:
        data = src.read(1)
        print(data.shape)
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='gray')
    plt.title(os.path.basename(filepath))
    plt.axis('off')

plt.tight_layout()
plt.show()



import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
import os

#RASTERIO Geographic information systems use GeoTIFF and other formats to organize 
#and store gridded raster datasets such as satellite imagery and terrain models. 
#Rasterio reads and writes these formats and provides a Python API based on 
#Numpy N-dimensional arrays and GeoJSON.

# === CONFIGURAZIONE DEI PERCORSI ===
# Inserisci i percorsi ai file tif scaricati manualmente.
# Questi file devono trovarsi nella stessa cartella dello script.
DTM_PATH = "/kaggle/working/dem_out.tiff"  # File DTM LIDAR scaricato da OpenTopography
B04_PATH = "/kaggle/working/sentinel_B04.tiff"    # Banda rossa di Sentinel-2 (Red)
B08_PATH = "/kaggle/working/sentinel_B08.tiff"    # Banda NIR (Near Infrared)

# === FUNZIONE PER CALCOLARE NDVI ===
def calculate_ndvi(nir_path, red_path):
    # Apriamo le due bande usando rasterio
    with rasterio.open(nir_path) as nir_src, rasterio.open(red_path) as red_src:
        # Leggiamo il contenuto dei raster come array float32
        nir = nir_src.read(1).astype('float32')
        red = red_src.read(1).astype('float32')

        # Calcolo NDVI: (NIR - RED) / (NIR + RED)
        ndvi = (nir - red) / (nir + red + 1e-6)  # aggiungiamo piccolo valore per evitare divisione per zero

        # Copiamo il profilo del raster per eventuali salvataggi futuri
        profile = nir_src.profile
        profile.update(dtype=rasterio.float32)

    return ndvi, profile

# === CALCOLO DELL'NDVI ===
ndvi, ndvi_profile = calculate_ndvi(B08_PATH, B04_PATH)

# === VISUALIZZAZIONE: NDVI E DTM ===
# Creiamo una figura con due sottopannelli (uno per NDVI e uno per DTM)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# NDVI: colorazione da verde (vegetazione sana) a rosso (bassa)
ndvi_img = ax1.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
ax1.set_title('NDVI Sentinel-2 (Beni)')
fig.colorbar(ndvi_img, ax=ax1, orientation='vertical')

# DTM: visualizzazione dellâ€™elevazione (da LIDAR)
with rasterio.open(DTM_PATH) as dtm_src:
    dtm = dtm_src.read(1)  # carichiamo il raster del terreno
    dtm_img = ax2.imshow(dtm, cmap='terrain')  # mappa colore altimetrica
    ax2.set_title('DTM LIDAR (OpenTopography)')
    fig.colorbar(dtm_img, ax=ax2, orientation='vertical')

# Ottimizziamo il layout
plt.tight_layout()
plt.show()



import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIGURAZIONE ===
B04_PATH = "/kaggle/working/sentinel_B04.tiff"    # Banda rossa di Sentinel-2 (Red)
B08_PATH = "/kaggle/working/sentinel_B08.tiff" 

NDVI_OUTPUT = "ndvi_output.tif"

# NDVI THRESHOLD â€” used to detect unusual low-vegetation areas
# These can correspond to anthropogenic features like raised fields, ditches, causeways, or mound surfaces
NDVI_THRESHOLD = 0.4    

# === CALCOLO NDVI ===
def calculate_ndvi(nir_path, red_path):
    with rasterio.open(nir_path) as nir_src, rasterio.open(red_path) as red_src:
        nir = nir_src.read(1).astype('float32')
        red = red_src.read(1).astype('float32')
        ndvi = (nir - red) / (nir + red + 1e-6)
        profile = nir_src.profile
        profile.update(dtype=rasterio.float32, count=1)
    return ndvi, profile

# === CALCOLO DELL'NDVI ===
ndvi, ndvi_profile = calculate_ndvi(B08_PATH, B04_PATH)

# === VISUALIZZAZIONE: NDVI E DTM ===
# Creiamo una figura con due sottopannelli (uno per NDVI e uno per DTM)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# NDVI: colorazione da verde (vegetazione sana) a rosso (bassa)
ndvi_img = ax1.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
ax1.set_title('NDVI Sentinel-2 (Beni)')
fig.colorbar(ndvi_img, ax=ax1, orientation='vertical')

# DTM: visualizzazione dellâ€™elevazione (da LIDAR)
with rasterio.open(DTM_PATH) as dtm_src:
    dtm = dtm_src.read(1)  # carichiamo il raster del terreno
    dtm_img = ax2.imshow(dtm, cmap='terrain')  # mappa colore altimetrica
    ax2.set_title('DTM LIDAR (OpenTopography)')
    fig.colorbar(dtm_img, ax=ax2, orientation='vertical')

# Ottimizziamo il layout
plt.tight_layout()
plt.show()

# === SALVATAGGIO NDVI COME GeoTIFF ===
with rasterio.open(NDVI_OUTPUT, 'w', **ndvi_profile) as dst:
    dst.write(ndvi, 1)
print(f"âœ… NDVI salvato come: {NDVI_OUTPUT}")

# === DETECT POTENTIAL ARCHAEOLOGICAL FEATURES USING NDVI THRESHOLD ===
# We flag pixels with NDVI below a threshold as potentially altered (e.g., human-made clearings, structures)
# TUNING GUIDE:
# - NDVI < 0.3 â†’ very bare or disturbed zones (aggressive detection)
# - NDVI < 0.4 â†’ good balance for raised mounds with thinner vegetation
# - NDVI < 0.5 â†’ broad detection, may include natural variation
anomalies_mask = ndvi < NDVI_THRESHOLD

# === PLOT NDVI E ANOMALIE ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# NDVI
ndvi_plot = ax1.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
ax1.set_title("NDVI Sentinel-2")
fig.colorbar(ndvi_plot, ax=ax1)

# Aree con valori NDVI anomali
anomaly_plot = ax2.imshow(anomalies_mask, cmap='gray')
ax2.set_title(f"Aree NDVI < {NDVI_THRESHOLD} (potenziali anomalie)")
fig.colorbar(anomaly_plot, ax=ax2)

plt.tight_layout()
plt.show()



import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

"""
This script identifies and exports low-NDVI pixels (potential archaeological anomalies)
from a georeferenced NDVI raster image.

Purpose:
- Convert NDVI-based anomaly detection into a spatial GeoJSON format.
- Each point represents a location where NDVI is below a specified threshold,
  which may correspond to anthropogenic features like mounds, raised fields,
  or cleared areas inside the Amazon forest.
- Output is suitable for mapping, spatial analysis, and integration in QGIS or web GIS tools.
"""

# === CONFIGURATION ===
NDVI_PATH = "ndvi_output.tif"          # NDVI raster GeoTIFF
NDVI_THRESHOLD = 0.4                   # NDVI anomaly threshold
GEOJSON_OUTPUT = "ndvi_anomalies.geojson"  # Output file path

# === LOAD NDVI RASTER ===
with rasterio.open(NDVI_PATH) as src:
    ndvi = src.read(1)                # Read first band
    transform = src.transform         # Affine transform to map pixel to coordinates
    crs = src.crs                     # Coordinate Reference System

# === FIND PIXELS BELOW THRESHOLD ===
# This will return a binary mask of pixels considered anomalous
anomaly_mask = ndvi < NDVI_THRESHOLD
rows, cols = np.where(anomaly_mask)

# === CONVERT PIXEL LOCATIONS TO GEO COORDINATES ===
# We'll create a list of shapely Point geometries (lat/lon or easting/northing)
points = []
ndvi_values = []

for row, col in zip(rows, cols):
    x, y = rasterio.transform.xy(transform, row, col, offset='center')
    points.append(Point(x, y))             # Create a Point geometry
    ndvi_values.append(ndvi[row, col])     # Store the original NDVI value for context

# === CREATE A GeoDataFrame WITH ATTRIBUTES ===
gdf = gpd.GeoDataFrame({
    "ndvi": ndvi_values
}, geometry=points, crs=crs)

# === EXPORT TO GEOJSON ===
gdf.to_file(GEOJSON_OUTPUT, driver="GeoJSON")

print(f"âœ… Exported {len(gdf)} anomalies to: {GEOJSON_OUTPUT}")


