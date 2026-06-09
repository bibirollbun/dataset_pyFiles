!pip install --upgrade pip
!pip install uv
!uv pip install pystac-client stackstac rioxarray rasterio elevation folium geopandas shapely pillow unsloth[torch] bitsandbytes --no-cache-dir
!uv pip install transformers bitsandbytes accelerate


from pathlib import Path
import datetime
import os

AOI_BBOX = {
    "west":  -72.5,
    "south": -11.0,
    "east":  -66.0,
    "north":  -7.5,
}
TODAY = datetime.date.today()
START = TODAY.replace(year=TODAY.year - 2)
TILE_METERS = 1_000

WORK = Path('/kaggle/working')
RAW = WORK/'data'/'raw'
TILE = WORK/'data'/'tiles'
OUT = WORK/'outputs'

for d in (RAW, TILE, OUT):
    d.mkdir(parents=True, exist_ok=True)


import numpy as np
import rasterio
from pystac_client import Client
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

# 1. Find the clearest scene
cat   = Client.open("https://earth-search.aws.element84.com/v1")
items = sorted(
    cat.search(
        collections=["sentinel-2-l2a"],
        bbox=[AOI_BBOX[k] for k in ("west","south","east","north")],
        datetime=f"{START}/{TODAY}",
        query={"eo:cloud_cover":{"lt":20}},
        max_items=20
    ).items(),
    key=lambda it: it.properties["eo:cloud_cover"]
)
if not items:
    raise RuntimeError("No Sentinel-2 scenes found!")
scene = items[0]
print(f"Using scene {scene.id} (cloud {scene.properties['eo:cloud_cover']}%)")

# 2. Map bands â†’ asset keys
PREFS = {
  "B08": ["B08","B08_10m","nir","nir08"],
  "B04": ["B04","B04_10m","red"],
  "B03": ["B03","B03_10m","green"],
}
def pick_asset(item, band):
    for p in PREFS[band]:
        if p in item.assets:
            return item.assets[p].href
    # fallback substring match
    for k in item.assets:
        if band.lower().lstrip("b0") in k.lower():
            return item.assets[k].href
    raise KeyError(f"{band} not found in {item.id}")

urls = {b: pick_asset(scene, b) for b in ("B08","B04","B03")}

# 3. Use the native grid of the first band asset for output TIFF
with rasterio.open(urls["B08"]) as src_ref:
    dst_h = src_ref.height
    dst_w = src_ref.width
    transform = src_ref.transform
    crs = src_ref.crs

# 4. Read & scale (no reproject needed, just read & stack in order)
rgb = np.zeros((3, dst_h, dst_w), dtype=np.uint8)
for idx, band in enumerate(["B08", "B04", "B03"]):
    with rasterio.open(urls[band]) as src:
        data = src.read(1).astype("float32") / 10000.0   # reflectance [0â€“1]
        data = np.clip(data, 0, 1)
        rgb[idx] = (data * 255).astype("uint8")

# 5. Write out uint8 GeoTIFF
meta = {
    "driver":   "GTiff",
    "height":   dst_h,
    "width":    dst_w,
    "count":    3,
    "dtype":    "uint8",
    "crs":      crs,
    "transform":transform,
    "compress": "lzw"
}

S2_PATH = RAW/'sentinel_falsecolor.tif'
    
with rasterio.open(S2_PATH, "w", **meta) as dst:
    dst.write(rgb)
print("âœ” False-colour composite written to", S2_PATH)


import rasterio, matplotlib.pyplot as plt, numpy as np

with rasterio.open(S2_PATH) as src:
    img = src.read([1,2,3])            # bands: NIRâ†’R, Redâ†’G, Greenâ†’B
    img = np.transpose(img, (1,2,0))   # (h,w,3)

plt.figure(figsize=(8,8))
plt.imshow(img)
plt.title("Sentinel-2 False-Colour Composite (NIR-Red-Green)")
plt.axis("off")
plt.show()


import requests

API_KEY = "your_opentopo_api_key_here"
def download_opentopo_srtm_api(aoi, out_path, key=API_KEY):
    w, s, e, n = aoi['west'], aoi['south'], aoi['east'], aoi['north']
    url = (f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1"
           f"&south={s}&north={n}&west={w}&east={e}&outputFormat=GTiff&API_Key={key}")
    print("Requesting:", url)
    r = requests.get(url, stream=True)
    if not r.ok:
        raise RuntimeError("OpenTopography request failed: " + r.text)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("DEM downloaded to", out_path)



with rasterio.open(DEM_PATH) as src:
    dem_img = src.read(1)
plt.figure(figsize=(8,6))
plt.imshow(dem_img, cmap='terrain')
plt.title('SRTM DEM')
plt.axis('off')
plt.colorbar(label='Elevation (m)')
plt.show()


import rasterio
import numpy as np
HILLSHADE_PATH = RAW/'hillshade.tif'
with rasterio.open(DEM_PATH) as src:
    elev = src.read(1).astype('float32')
    x, y = np.gradient(elev, src.res[0], src.res[1])
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    az, alt = np.deg2rad(315), np.deg2rad(45)
    hs = np.sin(alt)*np.sin(slope) + np.cos(alt)*np.cos(slope)*np.cos(az - aspect)
    hs = ((hs - hs.min())/(hs.max()-hs.min())*255).astype('uint8')
    meta = src.meta.copy(); meta.update(dtype='uint8', count=1)
    with rasterio.open(HILLSHADE_PATH,'w',**meta) as dst: dst.write(hs,1)
print('Hillshade saved:', HILLSHADE_PATH)



with rasterio.open(HILLSHADE_PATH) as src:
    hs_img = src.read(1)
plt.figure(figsize=(8,6))
plt.imshow(hs_img, cmap='gray')
plt.title('Hillshade')
plt.axis('off')
plt.show()


NDVI_PATH = RAW/'ndvi.tif'
with rasterio.open(S2_PATH) as src:
    nir = src.read(1).astype('float32')
    red = src.read(2).astype('float32')
    ndvi = (nir-red)/(nir+red+1e-6)
    meta = src.meta.copy(); meta.update(dtype='float32', count=1)
    with rasterio.open(NDVI_PATH,'w',**meta) as dst: dst.write(ndvi.astype('float32'),1)
print('NDVI saved:', NDVI_PATH)


with rasterio.open(NDVI_PATH) as src:
    ndvi_img = src.read(1)
plt.figure(figsize=(8,6))
plt.imshow(ndvi_img, cmap='RdYlGn')
plt.title('NDVI')
plt.axis('off')
plt.colorbar(label='NDVI')
plt.show()



from rasterio.windows import Window
from tqdm import tqdm
from PIL import Image
def raster_to_tiles(raster_path, out_dir, tile_meters):
    tiles = []
    with rasterio.open(raster_path) as src:
        px_size = src.res[0]
        tile_px = int(tile_meters/px_size)
        for row in tqdm(range(0, src.height, tile_px)):
            for col in range(0, src.width, tile_px):
                if row+tile_px > src.height or col+tile_px > src.width:
                    continue
                win = Window(col, row, tile_px, tile_px)
                data = src.read(window=win)
                if data.shape[0]==1:
                    data = np.repeat(data,3,axis=0)
                img = np.transpose(data, (1,2,0)).astype('uint8')
                out_path = out_dir/f"tile_{row}_{col}.png"
                Image.fromarray(img).save(out_path, optimize=True)
                cx, cy = src.xy(row+tile_px/2, col+tile_px/2)
                tiles.append(dict(path=str(out_path), lon=cx, lat=cy))
    return tiles

tile_index = raster_to_tiles(S2_PATH, TILE, TILE_METERS)
import pandas as pd
tiles_df = pd.DataFrame(tile_index)
tiles_df.to_csv(WORK/'tiles_index.csv', index=False)
print(f'Generated {len(tiles_df)} tiles')



import matplotlib.pyplot as plt
from PIL import Image
samples = tiles_df.sample(min(4, len(tiles_df)), random_state=42)
plt.figure(figsize=(12,6))
for i, row in enumerate(samples.itertuples()):
    plt.subplot(1,4,i+1)
    img = Image.open(row.path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Tile {i+1}")
plt.suptitle('Sample 1kmÂ² Tiles (Sentinel-2)')
plt.show()


import rasterio
import numpy as np
import pandas as pd
import re
from rasterio.windows import Window

ndvi_threshold = 0.2
black_threshold = 0.98  # e.g., 98% or more pixels are zero â†’ considered "black/empty"

flags = []
black_flags = []

with rasterio.open(NDVI_PATH) as src:
    px_size = src.res[0]
    tile_px = int(TILE_METERS / px_size)
    for _, row in tiles_df.iterrows():
        m = re.search(r'tile_(\d+)_(\d+)', row['path'])
        r, c = map(int, m.groups())
        win = Window(c, r, tile_px, tile_px)
        nd = src.read(1, window=win)
        
        # Low NDVI flag
        low_ndvi = float(np.nanmean(nd)) < ndvi_threshold
        # "Mostly black" flag
        frac_black = np.sum(nd == 0) / nd.size
        not_black = frac_black < black_threshold
        
        # Store both flags if you want
        flags.append(low_ndvi and not_black)
        black_flags.append(not_black)  # Optional: for analysis

tiles_df['low_ndvi_and_not_black'] = flags
tiles_df['not_black'] = black_flags   # Optional: for diagnostics

# Only keep the good tiles
cand_df = tiles_df[tiles_df['low_ndvi_and_not_black']].reset_index(drop=True)
cand_df.to_csv(WORK/'candidates.csv', index=False)
print(f'{len(cand_df)} candidate tiles flagged for Gemma analysis')


import torch
from transformers import pipeline

pipe = pipeline(
    "image-text-to-text",
    model="unsloth/gemma-3-4b-it-bnb-4bit",
    torch_dtype=torch.bfloat16
)


import requests
from io import BytesIO
from PIL import Image
from IPython.display import display

# 1) load & show the Machu Picchu test image
url = "https://www.boletomachupicchu.com/gutblt/wp-content/images/satelital-machu-picchu.jpg"
resp = requests.get(url)
test_img = Image.open(BytesIO(resp.content)).convert("RGB")
display(test_img)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url},
            {"type": "text", "text": "Describe any signs of man-made or geometric earthworks "
                      "larger than 80 m in this image. Do you see patterns, "
                      "ditches, mounds, or shapes that could indicate "
                      "ancient structures?"}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])


url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTcFCcoobR7zHu-uBHVT5uUSydnO-fcJwjOCA&s"
resp = requests.get(url)
test_img = Image.open(BytesIO(resp.content)).convert("RGB")
display(test_img)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": test_img},
            {"type": "text", "text": "Describe any signs of man-made or geometric earthworks "
                      "larger than 80 m in this image. Do you see patterns, "
                      "ditches, mounds, or shapes that could indicate "
                      "ancient structures?"}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])


import pandas as pd
from datasets import Dataset
from PIL import Image

MAX_AOI = 12

# Build dataset from image paths
records = []
for rec in cand_df[:MAX_AOI].itertuples():
    records.append({
        "path": rec.path,
        "prompt": (
            "Describe any signs of man-made or geometric earthworks in the Sentinel-2 False-Colour Composite (NIR-Red-Green) images larger than 80 m in this image. "
            "Do you see patterns, ditches, mounds, or shapes that could indicate ancient structures?"
        )
    })
dataset = Dataset.from_list(records)

def extract_answer(out):
    if isinstance(out, dict):
        gen = out.get("generated_text", None)
        if isinstance(gen, list) and len(gen) > 0:
            last_item = gen[-1]
            if isinstance(last_item, dict) and "content" in last_item:
                return last_item["content"]
            return str(last_item)
        elif isinstance(gen, str):
            return gen
        else:
            return str(gen)
    elif isinstance(out, list):
        return extract_answer(out[-1])
    else:
        return str(out)

def infer(batch):
    images = [Image.open(p).convert("RGB") for p in batch["path"]]
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        for img, prompt in zip(images, batch["prompt"])
    ]
    outputs = pipe(text=messages, max_new_tokens=200)
    answers = [extract_answer(out) for out in outputs]
    return {"gemma_answer": answers}

# Batched inference for max GPU throughput!
results = dataset.map(
    infer,
    batched=True,
    batch_size=4,  # adjust for your GPU
)

res_df = pd.DataFrame(results)
res_df.to_csv(OUT / "gemma_outputs.csv", index=False)
print("Results saved!")



import folium
import pandas as pd
from pyproj import Transformer

# Read your CSV
df = pd.read_csv(OUT/'gemma_outputs.csv')
df = pd.merge(cand_df, df, on='path', how='inner')

print("Merged columns:", df.columns.tolist())
print("Merged shape:", df.shape)
print(df.head())

# ---- Fix coordinates: Convert UTM to lat/lon ----
utm_zone = 19  # Check your AOI's UTM zone!
utm_crs = f"+proj=utm +zone={utm_zone} +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
transformer = Transformer.from_crs(utm_crs, "epsg:4326", always_xy=True)

# Convert columns: lon/lat (meters) to lon_deg/lat_deg (degrees)
df['lon_deg'], df['lat_deg'] = transformer.transform(df['lon'].values, df['lat'].values)

# ---- Folium map, using corrected coordinates ----
m = folium.Map(
    location=[(AOI_BBOX['south']+AOI_BBOX['north'])/2,
              (AOI_BBOX['west']+AOI_BBOX['east'])/2],
    zoom_start=7
)

for _, r in df.iterrows():
    folium.Marker(
        location=[r['lat_deg'], r['lon_deg']],
        popup=(str(r['gemma_answer'])[:250] + '...'),
        tooltip=str(r['path']).split('/')[-1]
    ).add_to(m)

MAP_HTML = OUT/'flagged_tiles_map.html'
m.save(MAP_HTML)
print('Map saved:', MAP_HTML)



from IPython.display import IFrame

# Use the relative path from /kaggle/working/
IFrame("outputs/flagged_tiles_map.html", width=800, height=600)


