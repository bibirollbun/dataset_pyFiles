import shutil
import os

# ✅ Correct source folder
source_folder = "/kaggle/input/ai-lidar-dataset/Dataset"

# Output directory
output_dir = "/kaggle/working/outputs"
os.makedirs(output_dir, exist_ok=True)

# Output ZIP path
output_zip_path = os.path.join(output_dir, "tile_dataset_bundle.zip")

# Create ZIP
shutil.make_archive(base_name=output_zip_path.replace(".zip", ""), format='zip', root_dir=source_folder)

print("✅ Zipped dataset ready at:", output_zip_path)



!pip install earthaccess==0.13.0 s3fs==2023.4.0 fsspec==2023.4.0 aiobotocore==2.5.4 botocore==1.31.17


!pip list | grep -E "earthaccess|s3fs|fsspec|botocore|aiobotocore"


# ======================================
# 🔧 TorchVision ViT – Clean One-Cell Setup
# ======================================

# STEP 0: Clean uninstall of broken/partial libraries
!pip uninstall -y torch torchvision torchaudio fastai transformers accelerate peft sentence-transformers torchao -q

# STEP 1: Install matching versions of Torch ecosystem (known stable)
!pip install -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# STEP 2: Install geospatial + utility libraries
!pip install -q rasterio rioxarray shapely pystac-client planetary-computer geopandas folium h5py pillow requests tqdm

# STEP 3: Suppress TF/CUDA log spam (if relevant)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# STEP 4: Core imports
import json, gc, zipfile, logging
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import geopandas as gpd
from shapely.geometry import box, Point
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from rasterio.windows import Window
from rasterio.enums import Resampling
from pyproj import Transformer
import folium
from PIL import Image
from io import BytesIO
import requests
from tqdm.notebook import tqdm

# STEP 5: TorchVision ViT – Load model and weights
import torch
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_V1
model   = vit_b_16(weights=weights).eval()
prep    = weights.transforms()

# STEP 6: Satellite-ready clients
from pystac_client import Client
from planetary_computer import sign

# STEP 7: 🔍 Sanity test – Run on known image (dog)
img_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
img     = Image.open(BytesIO(requests.get(img_url).content)).convert("RGB")

# Preprocess & inference
x = prep(img).unsqueeze(0)
with torch.no_grad():
    logits = model(x)
probs, idxs = logits.softmax(-1).topk(5)

# Decode results
labels  = [weights.meta["categories"][i] for i in idxs[0]]
results = list(zip(labels, probs[0].tolist()))
print("✅ Top-5 Predictions:", results)



# --- STEP 1: Define Paths & AOI ---
BASE_DIR = Path("/kaggle/working")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
for d in [DATA_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

AOI_BBOX = {"west": -62.501, "south": -10.168, "east": -62.411, "north": -10.078}
GEE_POINT = [-62.456, -10.123]


# --- STEP 2: Sentinel-2 Download & RGB/NDVI Gen ---
stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
START_DATE = (date.today() - timedelta(days=180)).isoformat()
END_DATE = date.today().isoformat()
items = stac.search(
    collections=["sentinel-2-l2a"],
    bbox=list(AOI_BBOX.values()),
    datetime=f"{START_DATE}/{END_DATE}",
    query={"eo:cloud_cover": {"lt": 10}}
).item_collection()
item = items[0]
bands = {"B04": "red", "B03": "green", "B02": "blue", "B08": "nir"}
arrays, thumb_arrays = {}, []
for b, name in bands.items():
    with rasterio.open(sign(item.assets[b].href)) as src:
        if b in ["B04", "B08"]:
            arrays[name] = src.read(1)
            if b == "B04": profile = src.profile
        if b in ["B04", "B03", "B02"]:
            lowres = src.read(1, out_shape=(int(src.height*0.1), int(src.width*0.1)), resampling=Resampling.bilinear)
            thumb_arrays.append(lowres)

rgb_thumb = np.stack(thumb_arrays, axis=-1)
rgb_thumb = np.clip(rgb_thumb / 3000, 0, 1)
plt.imsave(OUTPUT_DIR / "s2_rgb_composite.png", rgb_thumb)

ndvi = (arrays["nir"] - arrays["red"]) / (arrays["nir"] + arrays["red"] + 1e-5)
ndvi_path = DATA_DIR / "s2_ndvi.tif"
profile.update(count=1, dtype=rasterio.float32)
with rasterio.open(ndvi_path, "w", **profile) as dst:
    dst.write(ndvi.astype(rasterio.float32), 1)
del arrays, rgb_thumb



# --- STEP 3: Landsat Historical Comparison ---
landsat_items = stac.search(
    collections=["landsat-8-c2-l2"],
    bbox=list(AOI_BBOX.values()),
    datetime="2015-01-01/2016-01-01",
    query={"eo:cloud_cover": {"lt": 15}}
).item_collection()
l8 = landsat_items[0]
with rasterio.open(sign(l8.assets["SR_B4"].href)) as r: red = r.read(1)
with rasterio.open(sign(l8.assets["SR_B3"].href)) as g: green = g.read(1)
with rasterio.open(sign(l8.assets["SR_B2"].href)) as b: blue = b.read(1)
l8_vis = np.clip(np.stack([red, green, blue], axis=0).transpose(1,2,0) / 3000, 0, 1)
plt.imsave(OUTPUT_DIR / "landsat_rgb_historical.png", l8_vis)


# --- STEP 4: Generate Random NDVI Tiles ---
tile_size, num_tiles = 100, 100
coords = []
with rasterio.open(ndvi_path) as src:
    for _ in range(num_tiles):
        x, y = np.random.randint(0, src.width-tile_size), np.random.randint(0, src.height-tile_size)
        coords.append({"pixel_x": x, "pixel_y": y})
candidates_df = pd.DataFrame(coords)

with rasterio.open(ndvi_path) as src:
    transform = src.transform
    for i, row in candidates_df.iterrows():
        px, py = row['pixel_x'], row['pixel_y']
        lon, lat = rasterio.transform.xy(transform, py+tile_size//2, px+tile_size//2)
        candidates_df.loc[i, "lon"] = lon
        candidates_df.loc[i, "lat"] = lat


# --- STEP 5: AI Model Classification (torchvision ViT) ---

import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm.notebook import tqdm

# Assume these are already defined elsewhere in your notebook:
# candidates_df: DataFrame with columns ['lon','lat','pixel_x','pixel_y']
# ndvi_path: Path to your NDVI GeoTIFF
# OUTPUT_DIR: Path object where outputs go

# 1) Prepare model + transforms
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ViT_B_16_Weights.IMAGENET1K_V1
model   = vit_b_16(weights=weights).to(device).eval()
prep    = weights.transforms()  # resize→224², center crop, to tensor, normalize

# 2) Extract 100×100 pixel patches as PIL Images
images = []
rows_to_keep = []
with rasterio.open(ndvi_path) as src:
    for idx, row in tqdm(candidates_df.iterrows(), total=len(candidates_df)):
        win = Window(int(row.pixel_x), int(row.pixel_y), 100, 100)
        arr = src.read(1, window=win)  # NDVI values
        if np.isnan(arr).all():
            continue  # skip empty windows

        # Save a quick PNG (optional—only if you need to inspect)
        tile_path = OUTPUT_DIR / f"tile_{idx}.png"
        plt.imsave(tile_path, arr, cmap='gray')

        # Convert to 3‐channel PIL
        img = Image.open(tile_path).convert("RGB")
        images.append(img)
        rows_to_keep.append(row)

# 3) Batch-infer with ViT
batch_size = 16
results = []

for i in range(0, len(images), batch_size):
    batch = images[i : i + batch_size]
    # preprocess + stack
    tensor_batch = torch.stack([prep(img) for img in batch]).to(device)

    with torch.no_grad():
        logits = model(tensor_batch)
        probs, preds = torch.softmax(logits, dim=1).max(dim=1)

    # collect per‐image result
    for j, (prob, pred) in enumerate(zip(probs, preds)):
        row   = rows_to_keep[i + j]
        label = weights.meta["categories"][pred.item()]
        score = prob.item()
        results.append({
            "lon":   row.lon,
            "lat":   row.lat,
            "label": label,
            "score": score
        })

# 4) Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "hf_tile_analysis.csv", index=False)



# --- STEP 6: Export Top 10 Tiles ---
top_tiles = results_df.sort_values("score", ascending=False).head(10).reset_index(drop=True)
with zipfile.ZipFile(OUTPUT_DIR / "top10_tiles.zip", 'w') as z:
    for idx, row in top_tiles.iterrows():
        path = OUTPUT_DIR / f"tile_{idx}.png"
        if path.exists():
            z.write(path, arcname=f"tile_{idx+1}.png")


from kaggle_secrets import UserSecretsClient
import earthaccess
import h5py
import pandas as pd
import os
from pathlib import Path

# 📂 Paths
DATA_DIR = Path("/kaggle/working/data")
GEDI_DIR = DATA_DIR / "gedi"
DATA_DIR.mkdir(exist_ok=True)
GEDI_DIR.mkdir(exist_ok=True)

# 🌍 AOI BBox (Rondônia region)
AOI_BBOX = {"west": -62.9, "south": -10.4, "east": -62.0, "north": -9.8}

# 🔐 Earthdata credentials
secrets = UserSecretsClient()
USER = secrets.get_secret("EARTHDATA_USERNAME")
PWD = secrets.get_secret("EARTHDATA_PASSWORD")

# 🗝️ Write .netrc
netrc_path = Path.home() / ".netrc"
netrc_path.write_text(f"""machine urs.earthdata.nasa.gov\nlogin {USER}\npassword {PWD}""")
os.chmod(netrc_path, 0o600)

# 🌐 Login
earthaccess.login(persist=True)

# 🔍 Search GEDI granules
results = earthaccess.search_data(
    short_name="GEDI02_B",
    bounding_box=(AOI_BBOX["west"], AOI_BBOX["south"], AOI_BBOX["east"], AOI_BBOX["north"]),
    temporal=("2023-01-01", "2025-06-25"),
    cloud_hosted=True
)

# 💾 Download
gedi_files = earthaccess.download(results, GEDI_DIR)

# 🧬 Parse canopy heights
records = []
for fp in gedi_files:
    with h5py.File(fp, "r") as f:
        for beam in [b for b in f if b.startswith("BEAM")]:
            try:
                lats = f[f"{beam}/geolocation/lat_lowestmode"][:]
                lons = f[f"{beam}/geolocation/lon_lowestmode"][:]
                rh100 = f[f"{beam}/rh100"][:]
                for lat, lon, h in zip(lats, lons, rh100):
                    if AOI_BBOX["west"] <= lon <= AOI_BBOX["east"] and AOI_BBOX["south"] <= lat <= AOI_BBOX["north"]:
                        records.append({"Latitude": float(lat), "Longitude": float(lon), "CanopyHeight": float(h)})
            except KeyError:
                continue

# 🧾 Save CSV
gedi_df = pd.DataFrame(records)
gedi_df.to_csv(DATA_DIR / "gedi_canopy_filtered.csv", index=False)



import pandas as pd

# Load HuggingFace tile prediction scores
df = pd.read_csv("outputs/hf_tile_analysis.csv")

# Pick top 10 tiles by HuggingFace score
top_tiles = df.sort_values("score", ascending=False).head(10).reset_index(drop=True)

# Example structure should be: ["tile_id", "lat", "lon", "score"]
print(top_tiles.head())



from shapely.geometry import Point
import pandas as pd
import folium
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Build geometry for GEDI points
gedi_df["geometry"] = gedi_df.apply(lambda row: Point(row["Longitude"], row["Latitude"]), axis=1)

# Analyze top tiles
tile_results = []
for i, row in top_tiles.iterrows():
    tile_center = Point(row["lon"], row["lat"])
    buffer = tile_center.buffer(0.0009)
    matched = gedi_df[gedi_df["geometry"].apply(lambda pt: pt.within(buffer))]
    heights = matched["CanopyHeight"].dropna()
    tile_results.append({
        "tile_id": i + 1,
        "tile_lat": row["lat"],
        "tile_lon": row["lon"],
        "tile_score": row["score"],
        "min_canopy_height": heights.min(),
        "max_canopy_height": heights.max(),
        "mean_canopy_height": heights.mean(),
        "std_canopy_height": heights.std(),
        "num_gedi_points": len(matched)
    })

# Save summary CSV
summary_df = pd.DataFrame(tile_results)
summary_df.to_csv(OUTPUT_DIR / "tile_gedi_summary.csv", index=False)



# --- STEP 9: Change Detection Image ---

from PIL import Image, ImageChops

# Load your two composites
s2 = Image.open(OUTPUT_DIR / "s2_rgb_composite.png").convert("RGB")
l8 = Image.open(OUTPUT_DIR / "landsat_rgb_historical.png").convert("RGB")

# Compute per-pixel absolute difference
diff = ImageChops.difference(s2, l8)

# Save the change map
diff.save(OUTPUT_DIR / "historical_change_map.png")

print("🎯 DONE: All outputs ready in /outputs — time to win the OpenAI to Z hackathon!")




from matplotlib.pyplot import imshow
from PIL import Image
img = Image.open("outputs/historical_change_map.png")
imshow(img)



import pandas as pd
scores_df = pd.read_csv("outputs/hf_tile_analysis.csv")
scores_df.head(10)



import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from pyproj import Transformer
import numpy as np

# ------------------------------------------
# 🔄 STEP 1: Convert UTM (EPSG:32720) to WGS84 (EPSG:4326)
# ------------------------------------------
transformer = Transformer.from_crs("EPSG:32720", "EPSG:4326", always_xy=True)

# Replace these lines with your actual top_tiles DataFrame
# Example fallback if you're reloading from CSV:
# top_tiles = pd.read_csv("outputs/hf_tile_analysis.csv")  

# 🔁 Transform UTM to Lat/Lon
top_tiles[["tile_lon", "tile_lat"]] = top_tiles.apply(
    lambda row: pd.Series(transformer.transform(row["lon"], row["lat"])),
    axis=1
)

# ------------------------------------------
# 📌 STEP 2: Build GeoDataFrames
# ------------------------------------------
# GEDI Points (already in EPSG:4326)
gdf_gedi = gpd.GeoDataFrame(
    gedi_df,
    geometry=gpd.points_from_xy(gedi_df["Longitude"], gedi_df["Latitude"]),
    crs="EPSG:4326"
)

# Top Tile Centers (converted to EPSG:4326)
gdf_tiles = gpd.GeoDataFrame(
    top_tiles,
    geometry=gpd.points_from_xy(top_tiles["tile_lon"], top_tiles["tile_lat"]),
    crs="EPSG:4326"
)

# ✅ Sanity check
assert gdf_tiles["geometry"].x.between(-180, 180).all(), "Invalid longitudes"
assert gdf_tiles["geometry"].y.between(-90, 90).all(), "Invalid latitudes"

# ------------------------------------------
# 🗺️ STEP 3: Plot GEDI + Tile Centers
# ------------------------------------------
ax = gdf_gedi.plot(marker='.', figsize=(10, 8), alpha=0.4, color='green', label='GEDI')
gdf_tiles.plot(ax=ax, color='red', markersize=80, label='Top Tiles')

# ✅ Set proper aspect ratio
bounds = gdf_gedi.total_bounds
mid_lat = (bounds[1] + bounds[3]) / 2
ax.set_aspect(1 / np.cos(mid_lat * np.pi / 180))

# 📝 Annotate
plt.legend()
plt.title("GEDI Footprints vs Top Tile Centers")
plt.text(-62.4, -10.25, "High score\nno GEDI", color="red")
plt.grid(True)
plt.show()


import pandas as pd
import folium
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import Point

# =====================
# 📁 Load tile summary CSV
# =====================
summary_df = pd.read_csv("outputs/tile_gedi_summary.csv")

# =====================
# 🔁 Convert UTM to Lat/Lon (EPSG:32720 → EPSG:4326)
# =====================
transformer = Transformer.from_crs("EPSG:32720", "EPSG:4326", always_xy=True)

summary_df[["tile_lon", "tile_lat"]] = summary_df.apply(
    lambda row: pd.Series(transformer.transform(row["tile_lon"], row["tile_lat"])),
    axis=1
)

# =====================
# 🗺️ Create Folium Map
# =====================
m = folium.Map(
    location=[summary_df["tile_lat"].mean(), summary_df["tile_lon"].mean()],
    zoom_start=9
)

# =====================
# 📍 Add Tile Markers
# =====================
for _, row in summary_df.iterrows():
    popup_text = (
        f"<b>Tile {row['tile_id']}</b><br>"
        f"Score: {row['tile_score']:.3f}<br>"
      
    )
    
    folium.Marker(
        location=[row["tile_lat"], row["tile_lon"]],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color="green" if row['num_gedi_points'] > 0 else "red", icon="tree-conifer")
    ).add_to(m)

# =====================
# 💾 Save and Display
# =====================
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
m.save(str(OUTPUT_DIR / "interactive_map.html"))

from IPython.display import IFrame
IFrame("outputs/interactive_map.html", width=950, height=500)



!cp /kaggle/input/tilecollage/archaeology_tile_collage_labeled.png .



# ✅ Copy top 3 tile collage to the working directory
!cp /kaggle/input/top3tiles/top3.png .



# 🔍 Final check for all critical imports
try:
    import earthaccess, s3fs, fsspec, aiobotocore, botocore
    print("✅ All dependencies successfully imported.")
except ImportError as e:
    print("🚫 ImportError detected:", e)
    raise





