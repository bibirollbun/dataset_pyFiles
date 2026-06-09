# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install requests rasterio bmi-topography geopandas rioxarray sentinelhub shapely matplotlib faiss-cpu  ftfy regex tqdm git+https://github.com/openai/CLIP.git -q


'''
author : Krutarth Parmar
reach me @ kayparmar[dot]com [https://kayparmar.com]
'''


# Outline
### diving in head-first
### integrating FAISS store, CLIP + RAG (in-progress)


'''
Let's get two verifiable public sources set-up. I am using sentinel-2 and open-topography (provided below)
| Technology                                          | Missions / Instruments                                                                   | DEM products you see in the API ( `demtype` )                                                                                      | Nominal grid-spacing | Coverage                                                                        |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------- |
| **Synthetic-aperture-radar interferometry (InSAR)** | *C-band* SRTM (Shuttle Radar Topography Mission, STS-99), *X-band* TanDEM-X / TerraSAR-X | `SRTMGL1` / `SRTMGL3` (v3 “void-filled”), `NASADEM` (re-processed SRTM), `COP30` / `COP90` (Copernicus DEM from TanDEM-X WorldDEM) | 30 m & 90 m          | 56° S – 60° N (SRTM), global 90° S – 90° N (Copernicus)                         |
| **Optical stereo / tri-stereo photogrammetry**      | ASTER VNIR on NASA *Terra*; PRISM on JAXA *ALOS / DAICHI*                                | `ASTERGDEM3` (ASTER GDEM v3), `AW3D30` (ALOS World 3D v3.2)                                                                        | 30 m                 | 83° S – 83° N (ASTER); global except polar holes (AW3D30) ([OpenTopography][1]) |
| **Laser / altimetry**                               | GEDI full-waveform LiDAR on the ISS; (sister site *OpenAltimetry* hosts ICESat-2)        | `GEDI_L3` 1 km DTM & canopy-height grid                                                                                            | 1 km                 | Between 51.6° S and 51.6° N (ISS ground track)                                  |
| **(Bonus) Radar-altimetry bathymetry blend**        | Multiple satellite altimeters + ship soundings                                           | `SRTM15PLUS` (SRTM15+ V2.1 global topo-bathymetry)                                                                                 | 15 arc-sec (\~500 m) | global ocean & land                                                             |

[1]: https://opentopography.org/blog/comparison-aster-gdem-srtm?utm_source=chatgpt.com "Comparison of ASTER GDEM to SRTM - OpenTopography"

Thank you to everyone who got actual freaking satellites into orbit! Kudos! 
'''

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
OTK = user_secrets.get_secret("OPEN_TOPOGRAPHY_API")
OAIK = user_secrets.get_secret("OPENAI_API_KEY")
SID = user_secrets.get_secret("SENTINEL_CLIENT_ID")
SIDS = user_secrets.get_secret("SENTINEL_CLIENT_SECRET")

#Setinel-2 set-up
from sentinelhub import SHConfig
import openai

config = SHConfig()
config.sh_client_id = SID
config.sh_client_secret = SIDS
config.save()
from IPython.display import Markdown, display

def openAICompelete(prompt : str):
    completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
    )

client = openai.OpenAI(api_key = OAIK)
prompt = "Tell me more about Amazon Rainforest and it's developement ecological and otherwise, provide detailed and comprehensive outlook"
completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)
content = completion.choices[0].message.content
display(Markdown(content))


import numpy as np


def tile_amazon_region(min_lat=-20, max_lat=5,
                       min_lon=-80, max_lon=-45,
                       step=0.2,
                       order="latlon"):
    """
    Create a dict of 0.2° tiles covering the Amazon Basin.
    
    Parameters
    ----------
    order : str
        'latlon'  → (min_lat, max_lat, min_lon, max_lon) 
        'lonlat'  → (min_lon, min_lat, lon_max, max_lat)
    """
    tiles, index = {}, 0
    for lat in np.arange(min_lat, max_lat, step):
        for lon in np.arange(min_lon, max_lon, step):
            if order == "latlon":
                bbox = (lat, lat + step, lon, lon + step)
            else:                        # 'lonlat'
                bbox = (lon, lat, lon + step, lat + step)
            tiles[f"tile_{index}"] = {"bbox": bbox, "timestamp": None}
            index += 1
    return tiles

import numpy as np

def get_image_from_bbox(bbox_coords, time_interval):
    bbox = BBox(bbox = bbox_coords, crs= CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution = 10)

    request = SentinelHubRequest(
        data_folder = "amazon_tiles",
        evalscript = evalscript,
        input_data = [
            SentinelHubRequest.input_data(            
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval_amazon,
            mosaicking_order=MosaickingOrder.LEAST_CC,)
        ],
        responses = [SentinelHubRequest.output_response('default', MimeType.TIFF)],
        bbox = bbox, 
        size = size, 
        config = config,
    )
    return request.get_data(save_data = True, redownload= True)
    
def latlon_to_lonlat(box_latlon):
    lat_min, lat_max, lon_min, lon_max = box_latlon
    return (lon_min, lat_min, lon_max, lat_max)


sites = {
    "Pedra Pintada": (-5.467, -5.367, -52.400, -52.300),
    "Acre Geoglyph": (-8.895, -8.795, -67.305, -67.205),
    "Kuhikugu":      (-12.608, -12.508, -53.161, -53.061),
    "Casarabe":      (-14.920, -14.820, -64.531, -64.431),
    "Teso dos Bichos": (-0.200, -0.100, -50.000, -49.900)
}

def find_tile(lat, lon, tiles):
    for name, info in tiles.items():
        lat_min, lat_max, lon_min, lon_max = info["bbox"]
        if lat_min <= lat < lat_max and lon_min <= lon < lon_max:
            return name
    return None

site_to_tile = {}
amazon_tiles = tile_amazon_region(step=0.2, order="latlon")
for site, (lat_min, lat_max, lon_min, lon_max) in sites.items():
    centre = ((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
    tile = find_tile(*centre, tiles=amazon_tiles)
    site_to_tile[site] = tile
    print(f"{site:18s} → {tile}")



#very important to get this right!! 
evalscript= '''
/**
 * Sentinel-2 L2A “Amazon-Archaeo” stack
 *  ─ visible, red-edge, NIR, SWIR reflectances
 *  ─ NDVI & NDMI indices
 *  ─ QA band showing 1 = clear land / 0 = cloud, nodata, or water-vapor opaque
 */
function setup() {
  return {
    input: [{
      // ↑ add B01/B09 if you want atmospheric or coastal info
      bands: [
        'B02','B03','B04',          // RGB
        'B05','B06','B07',          // red-edge trio (good for subtle veg stress)
        'B08','B8A',                // broad & narrow NIR
        'B11','B12',                // SWIR 1 & 2
        'CLP','dataMask'
      ],
      units: 'DN'         
    }],
    output: { bands: 13, sampleType: 'FLOAT32' }
  };
}

/* --- helper to compute spectral indices ---------------------------------- */
function safeDiv(num, den) { return den === 0 ? 0 : num / den; }

function evaluatePixel(s) {

  /* ----- cloud & nodata filtering */
  const cloudProb = s.CLP / 255.0;          // CLP is 0-255
  const isClear   = cloudProb <= 0.50 &&    // ≈ 30 % threshold – tweak if needed
                    s.dataMask === 1;       // valid pixel

  if (!isClear) {
    // return all-zeros + 0 in the QA slot so you can ignore later in training (this is ai added i don't understand it yet)
    return Array(12).fill(0).concat([0]);
  }

  /* ----- indices that expose vegetation / soil changes -------------------- */
  const ndvi = safeDiv(s.B08  - s.B04,  s.B08  + s.B04);   // classic (‘greenness’)
  const ndmi = safeDiv(s.B08  - s.B11,  s.B08  + s.B11);   // moisture - canopy gaps

  /* ----- output ----------------------------------------------------------- */
  return [
    s.B02, s.B03, s.B04,          // 1-3  Blue, Green, Red
    s.B05, s.B06, s.B07,          // 4-6  Red-edge 1-3
    s.B08, s.B8A,                 // 7-8  NIR, narrow-NIR
    s.B11, s.B12,                 // 9-10 SWIR 1-2
    ndvi,                         // 11   NDVI
    ndmi,                         // 12   NDMI
    1                             // 13   QA flag = clear
  ];
}
'''



from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)
import rasterio
import matplotlib.pyplot as plt

resolution = 10                       # metres
time_interval_amazon = ("2020-07-01", "2020-07-15")

for site, tile_id in site_to_tile.items():
    # ---- fetch bbox from your master grid ---------------------------
    raw_box = amazon_tiles[tile_id]["bbox"]          # (latmin, latmax, lonmin, lonmax)
    sh_box  = latlon_to_lonlat(raw_box)              # convert for Sentinel-Hub
    
    print(f"\n=== {site}  ({tile_id}) ===")
    print(f"Sentinel-Hub bbox (lon/lat order): {sh_box}")

    # ---- size check --------------------------------------------------
    bbox_obj   = BBox(bbox=sh_box, crs=CRS.WGS84)
    img_size   = bbox_to_dimensions(bbox_obj, resolution)
    print(f"Image size at {resolution} m: {img_size}  (pixels)")

    # ---- call helper ----------------------------------
    images = get_image_from_bbox(sh_box, time_interval_amazon)
    print(f"Request returned {len(images)} band-stack(s)")

    # ---- quicklook: first RGB TIFF in amazon_tiles/ -----------------
    def find_first_tiff(base_dir="amazon_tiles"):
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith(".tiff"):
                    return os.path.join(root, f)
        return None

    tiff_path = find_first_tiff()

    if images:                                   # SentinelHubRequest.get_data()
        rgb = images[0][..., [3-1, 2-1, 1-1]]    # bands 3-2-1 → R-G-B
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        plt.figure(figsize=(6, 6))
        plt.imshow(rgb)
        plt.title(f"{site} — RGB composite (from memory)")
        plt.axis("off")
        plt.show()
    else:
        print("⚠️  Request returned no data.")


import requests
import rasterio
from rasterio import MemoryFile
from rasterio.plot import show
import matplotlib.pyplot as plt 

API_KEY = OTK
dataset  = "SRTMGL1"               # 30 m DEM
out_root = "amazon_tiles"          # parent folder for DEMs
os.makedirs(out_root, exist_ok=True)

for site, tile_id in site_to_tile.items():
    raw_box = amazon_tiles[tile_id]["bbox"]        # (latmin, latmax, lonmin, lonmax)
    west, south, east, north = latlon_to_lonlat(raw_box)

    params = {
        "demtype"     : dataset,
        "south"       : south,
        "north"       : north,
        "west"        : west,
        "east"        : east,
        "outputFormat": "GTiff",
        "API_Key"     : API_KEY,
    }

    print(f"\n=== {site} ({tile_id}) ===")
    print("OpenTopography bbox (W,S,E,N):", (west, south, east, north))
    url = "https://portal.opentopography.org/API/globaldem"

    # -- request DEM --------------------------------------------------
    print("Requesting DEM …")
    r = requests.get(url, params=params, timeout=300)
    r.raise_for_status()

    # -- preview & save ----------------------------------------------
    with MemoryFile(r.content) as memfile:
        with memfile.open() as ds:
            plt.figure(figsize=(5, 5))
            show(ds, cmap="terrain")
            plt.title(site)
            plt.axis("off")
          

            out_path = os.path.join(
                out_root,
                f"{site.replace(' ', '_').lower()}_{dataset.lower()}.tif"
            )
            print("Saving ➜", out_path)
            with rasterio.open(out_path, "w", **ds.profile) as dest:
                dest.write(ds.read(1), 1)   # DEM has a single band

