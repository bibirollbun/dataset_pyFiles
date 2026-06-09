pip install rasterio


# ────────────────────────────────────────────────────────────────────────────────
# Full Pipeline with Regional Hotspots + Model‐Agreement Check (using o3, GPU-accelerated)
#
# • Checkpoint 1: DEM → sample → stats  
# • Checkpoint 1e: 3-model GPT terrain analysis (o3, gpt-4o-mini & gpt-4.1)
# • NEW Step: Regional-scale hotspot selection from South-America DEM  
# • Checkpoint 2: Fine-scale anomaly detection at those hotspots (GPU via CuPy if available)  
# • Checkpoint 3: 3-model LLM anomaly interpretation  
# • Checkpoint 4: Compare the three LLMs’ picks; if ≥2 agree, save consensus
# ────────────────────────────────────────────────────────────────────────────────

import os
import json
import re
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy
from openai import OpenAI
from kaggle_secrets import UserSecretsClient
from pyproj import Transformer

# GPU acceleration for array ops
try:
    import cupy as cp
    xp = cp
    GPU_ENABLED = True
    print("GPU detected: using CuPy for array operations.")
except ImportError:
    import numpy as cp  # alias cp for compatibility
    xp = cp
    GPU_ENABLED = False
    print("CuPy not found: falling back to NumPy.")

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def load_secret(name):
    try:
        return UserSecretsClient().get_secret(name)
    except:
        return os.environ.get(name, "")

DATA_DIR          = "data"
CHECK_DEM_PATH    = "/kaggle/input/incapuquio-fault-south-peru-santa-elena-zone-2018/Inkpkio_SE.tif"
REGIONAL_DEM_PATH = "/kaggle/input/south-america-dem/sa_dem_3s.tif"
os.makedirs(DATA_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# 1) Load small DEM & compute 100×100 sample stats
# ────────────────────────────────────────────────────────────────────────────────
with rasterio.open(CHECK_DEM_PATH) as src:
    dem_small, transform_small, nodata_small, dem_small_crs = (
        src.read(1), src.transform, src.nodata, src.crs
    )

# find a valid 100×100 window
WINDOW = 100
with rasterio.open(CHECK_DEM_PATH) as src:
    for _, win in src.block_windows(1):
        blk = src.read(1, window=win)
        blk_gpu = xp.asarray(blk)
        mask_gpu = blk_gpu != nodata_small
        if mask_gpu.any():
            i0 = int(win.row_off + int(xp.where(mask_gpu)[0][0]))
            j0 = int(win.col_off + int(xp.where(mask_gpu)[1][0]))
            break
    r0 = max(0, i0 - WINDOW//2)
    c0 = max(0, j0 - WINDOW//2)
    sample = src.read(1, window=Window(c0, r0, WINDOW, WINDOW))

# compute stats on sample, using GPU arrays if enabled
arr_sample = xp.asarray(sample)
mask = arr_sample != nodata_small
vals = arr_sample[mask]
stats = {
    "min_elev": float(vals.min()),
    "max_elev": float(vals.max()),
    "mean_elev": float(vals.mean()),
    "std_dev":  float(vals.std())
}
print("Sample stats:", stats)

# ────────────────────────────────────────────────────────────────────────────────
# 1e) 3-model terrain analysis
# ────────────────────────────────────────────────────────────────────────────────
openai_key = load_secret("openai key")
if not openai_key:
    raise RuntimeError("OpenAI key not found")
client = OpenAI(api_key=openai_key)

terrain_prompt = (
    f"Dataset: Inkpkio_SE (DEM)\n"
    f"Stats:\n"
    f"- Min:  {stats['min_elev']:.2f} m\n"
    f"- Max:  {stats['max_elev']:.2f} m\n"
    f"- Mean: {stats['mean_elev']:.2f} m\n"
    f"- Std:  {stats['std_dev']:.2f} m\n\n"
    "Describe the surface features and terrain characteristics."
)

models = ["o3", "gpt-4o-mini", "gpt-4.1"]
for m in models:
    resp = client.chat.completions.create(
        model=m,
        messages=[
            {"role":"system","content":"You are a geospatial analysis assistant."},
            {"role":"user",  "content":terrain_prompt}
        ]
    )
    out = resp.choices[0].message.content
    with open(f"{DATA_DIR}/terrain_{m}.txt","w") as f:
        f.write(out)
    print(f"Saved terrain_{m}.txt")




# ────────────────────────────────────────────────────────────────────────────────
# 2) Regional-scale hotspot selection (streaming/windowed, low‐mem)
# ────────────────────────────────────────────────────────────────────────────────
REG_WIN  = 200
REG_STEP = 100
topn_reg = 10
hotspots = []

with rasterio.open(REGIONAL_DEM_PATH) as src:
    nod = src.nodata
    tform = src.transform
    h, w = src.height, src.width

    for i in range(0, h - REG_WIN + 1, REG_STEP):
        for j in range(0, w - REG_WIN + 1, REG_STEP):
            window = Window(j, i, REG_WIN, REG_WIN)
            # read only this window into a small array
            blk = src.read(1, window=window)
            # minimal valid fraction check
            valid = blk != nod
            if valid.sum() < REG_WIN * REG_WIN * 0.5:
                continue

            # cast to GPU array only for this small block
            blk_gpu = xp.asarray(blk)
            mask_gpu = blk_gpu != nod
            std_dev = float(xp.std(blk_gpu[mask_gpu]))
            hotspots.append((std_dev, i, j))

# pick top‐N by std deviation
hotspots.sort(key=lambda x: x[0], reverse=True)
top_hot = hotspots[:topn_reg]

# convert pixel centers to lon/lat
transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
regional_candidates = []
for _, i, j in top_hot:
    row = i + REG_WIN//2
    col = j + REG_WIN//2
    x, y = xy(tform, row, col)
    lon, lat = transformer.transform(x, y)
    regional_candidates.append({"lon": lon, "lat": lat})

# save & inspect
with open(f"{DATA_DIR}/regional_candidates.json", "w") as f:
    json.dump({"regional_candidates": regional_candidates}, f, indent=2)

print("Regional candidates:", regional_candidates)



# ────────────────────────────────────────────────────────────────────────────────
# 3) Fine-scale anomaly detection (GPU if enabled)
# ────────────────────────────────────────────────────────────────────────────────
def detect_anomalies(arr, nodata, win=100, step=50, topn=5):
    data_gpu = xp.asarray(arr)
    cands = []
    h, w = arr.shape
    for i in range(0, h-win, step):
        for j in range(0, w-win, step):
            blk_gpu = data_gpu[i:i+win, j:j+win]
            mask_blk = blk_gpu != nodata
            if mask_blk.sum() < win*win*0.5:
                continue
            std_dev = float(xp.std(blk_gpu[mask_blk]))
            cands.append((std_dev, i, j))
    cands.sort(reverse=True, key=lambda x: x[0])
    return [
        {"lat": lat, "lon": lon}
        for _, i, j in cands[:topn]
        for lon, lat in [xy(transform_small, i+win//2, j+win//2)]
    ]

all_anomalies = []
for cand in regional_candidates:
    all_anomalies.extend(detect_anomalies(dem_small, nodata_small))
unique = {(a['lon'],a['lat']) for a in all_anomalies}
anomalies = [{"lon":lon,"lat":lat} for lon,lat in unique]
with open(f"{DATA_DIR}/anomalies.json","w") as f:
    json.dump({"anomalies": anomalies}, f, indent=2)
print("Anomalies:", anomalies)



# ────────────────────────────────────────────────────────────────────────────────
# 4) 3-model anomaly interpretation
# ────────────────────────────────────────────────────────────────────────────────
geo_anoms = regional_candidates
anomaly_prompt = (
    "Here are candidate anomaly coordinates (lon, lat):\n"
    + "\n".join(f"- ({pt['lon']:.6f}, {pt['lat']:.6f})" for pt in geo_anoms)
    + "\n\nBased on topography and typical archaeological signatures, list your top 5 picks with coordinates and a one-sentence rationale each."
)

anomaly_analyses = {}
for m in models:
    resp = client.chat.completions.create(
        model=m,
        messages=[
            {"role":"system","content":"You are an archaeological terrain expert."},
            {"role":"user",  "content":anomaly_prompt}
        ]
    )
    text = resp.choices[0].message.content
    with open(f"{DATA_DIR}/anomaly_analysis_{m}.txt","w") as f:
        f.write(text)
    anomaly_analyses[m] = text
    print(f"Saved anomaly_analysis_{m}.txt")



# 5) Compare all three models’ picks (require ≥2 agreement)

import re
import json

# models and anomaly_analyses should already be defined, e.g.:
# models = ["o3", "gpt-4o-mini", "gpt-4.1"]
# anomaly_analyses = {
#     "o3": text_o3,
#     "gpt-4o-mini": text_4om,
#     "gpt-4.1": text_41
# }
# DATA_DIR also defined

def parse_picks(text):
    # Normalize any Unicode en-dash or em-dash to ASCII hyphen
    normalized = text.replace("–", "-").replace("—", "-")
    # Extract lines like "(lon, lat)"
    matches = re.findall(r"\(\s*([-]?\d+\.\d+)\s*,\s*([-]?\d+\.\d+)\s*\)", normalized)
    return {(float(lon), float(lat)) for lon, lat in matches}

# Parse each model's picks
picks = {m: parse_picks(anomaly_analyses[m]) for m in models}

# Tally how many models picked each coordinate
tally = {}
for pick_set in picks.values():
    for coord in pick_set:
        tally[coord] = tally.get(coord, 0) + 1

# Keep those picked by at least two models
consensus = [
    {"lon": lon, "lat": lat}
    for (lon, lat), count in tally.items()
    if count >= 2
]

if consensus:
    print("✅ Consensus (>=2 models agree):", consensus)
    with open(f"{DATA_DIR}/anomaly_consensus.json", "w") as f:
        json.dump({"consensus": consensus}, f, indent=2)
    print("Saved anomaly_consensus.json")
else:
    print("❌ No consensus among models.")



