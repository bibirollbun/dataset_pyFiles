# =======================================================
# Amazon Archaeology Submission - Setup, Data, Helpers
# =======================================================

# Core pip installs (run ONCE per notebook)
!pip install --quiet pystac-client planetary-computer rasterstats rasterio geopandas PyPDF2 beautifulsoup4 ipywidgets

# =======================================================
# Universal Setup & Imports
# =======================================================

# --- Standard Library ---
import os, glob, gc, time, warnings, requests, re
from pathlib import Path

# --- Scientific/Data ---
import numpy as np
import pandas as pd

# --- Plotting ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Geospatial ---
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds
import geopandas as gpd
from shapely.geometry import Point, LineString
from pyproj import Transformer

# --- Image & Spatial Analysis ---
from scipy.spatial import cKDTree
from scipy.ndimage import label
from skimage.filters import sobel
from sklearn.cluster import DBSCAN

# --- Remote Sensing/Cloud ---
from pystac_client import Client
from planetary_computer import sign

# --- Text & Web ---
from bs4 import BeautifulSoup
import PyPDF2
import openai

warnings.filterwarnings('ignore')

# If running on Kaggle and using OpenAI API:
try:
    from kaggle_secrets import UserSecretsClient
    _HAS_KAGGLE_SECRETS = True
except Exception:
    _HAS_KAGGLE_SECRETS = False

# ---- Constants ----
CRS_UTM20S = "EPSG:32620"
AMAZON_BOUNDS = {
    "xmin": -67.66403852431348, "ymin": -11.231272611581758,
    "xmax": -59.99531469980342, "ymax": -6.68742807903692,
}

# =======================================================
# Kaggle + Local Path Resolver (Git-friendly)
# =======================================================

def on_kaggle() -> bool:
    return Path("/kaggle").exists()

# Map your Kaggle dataset locations (match the names in the right "Input" panel)
K_DATASETS = {
    "anomaly_mask": Path("/kaggle/input/anomaly-mask-and-reprojected-data/anomaly_mask.tif"),
    "modis":        Path("/kaggle/input/modis-landcover-amazon-2001-2018/modis_landcover_amazon_2001_2018/LC_Type1_h12v09_2001.tif"),
    "srtm":         Path("/kaggle/input/amazon-data-set/output_SRTMGL1.tif"),
    "hydro":        Path("/kaggle/input/amazon-data-set/HydroRIVERS_v10_sa/HydroRIVERS_v10_sa.gdb"),
    "landsat":      Path("/kaggle/input/landsat8/"),
    "text_data":    Path("/kaggle/input/text-data"),
}

def get_path(key: str) -> Path:
    """
    Return a Path for the given dataset key.
    - On Kaggle: points to /kaggle/input/... 
    - Locally:   looks under ./data/<key>/ (or ./data/<key>.tif for files)
    """
    if on_kaggle():
        return K_DATASETS[key]
    # Local mirrors / small samples for testing
    base = Path("./data") / key
    if key in {"anomaly_mask", "modis", "srtm"} and not base.suffix:
        tif = base.with_suffix(".tif")
        return tif if tif.exists() else base
    return base

print("Running on Kaggle:", on_kaggle())
for k in ["anomaly_mask", "modis", "srtm", "hydro"]:
    print(f"{k}: {get_path(k)}")

# =======================================================
# OpenAI key setup (optional, for literature mining)
# =======================================================
if _HAS_KAGGLE_SECRETS:
    try:
        openai.api_key = UserSecretsClient().get_secret("OpenAI")
    except Exception:
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
else:
    openai.api_key = os.getenv("OPENAI_API_KEY", "")

# =======================================================
# Helper Functions: Text, Geo, NDVI, Topo, Rivers
# =======================================================

# ---- Text Chunking and Q&A for Literature Analysis ----
def split_text(text, max_words=1200):
    """Split text into ~max_words chunks for GPT calls."""
    paras = text.split("\n\n")
    chunks, chunk, count = [], [], 0
    for para in paras:
        nwords = len(para.split())
        if count + nwords > max_words:
            chunks.append("\n\n".join(chunk))
            chunk, count = [], 0
        chunk.append(para)
        count += nwords
    if chunk:
        chunks.append("\n\n".join(chunk))
    return chunks

def gpt_answer(question, context, system_prompt="You are an expert Amazon archaeologist."):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def citation_from_filename(filename):
    """Returns a simple citation label for a file (customize as needed)."""
    base = os.path.basename(filename)
    if "peerj" in base.lower():
        if "15137" in base:
            return "Walker et al. 2023 (PeerJ 15137)"
        if "3863" in base:
            return "de Souza et al. 2017 (PeerJ 3863)"
    return base

# ---- Raster Pixel to Lat/Lon ----
def pixel_to_coords(transform, rows_cols):
    coords = []
    for row, col in rows_cols:
        lon, lat = rasterio.transform.xy(transform, row, col)
        coords.append((lon, lat))
    return np.array(coords)

# ---- NDVI Calculation (optional, for Landsat bands) ----
def calc_ndvi(red_band_path, nir_band_path):
    with rasterio.open(red_band_path) as red_src, rasterio.open(nir_band_path) as nir_src:
        red = red_src.read(1).astype(float)
        nir = nir_src.read(1).astype(float)
        ndvi = (nir - red) / (nir + red)
        ndvi[np.isinf(ndvi)] = np.nan
    return ndvi

# ---- MODIS Landcover Lookup ----
def modis_class_at_point(lon, lat, modis_raster, modis_transform):
    row, col = ~modis_transform * (lon, lat)
    row, col = int(round(row)), int(round(col))
    if 0 <= row < modis_raster.shape[0] and 0 <= col < modis_raster.shape[1]:
        return modis_raster[row, col]
    return np.nan

# ---- Slope Calculation from SRTM ----
def slope_from_srtm(elevation):
    dy = sobel(elevation, axis=0, mode='constant')
    dx = sobel(elevation, axis=1, mode='constant')
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2) / 8)
    slope_deg = np.degrees(slope_rad)
    return slope_deg

# ---- River Sampling for Distance to Nearest River ----
def sample_river_points(geoms, step=100):
    """Given a list of LineString or MultiLineString, return sample points along each line."""
    points = []
    for geom in geoms:
        if geom.geom_type == 'LineString':
            n = int(geom.length // step) + 2
            for dist in np.linspace(0, geom.length, n):
                pt = geom.interpolate(dist)
                points.append((pt.x, pt.y))
        elif geom.geom_type == 'MultiLineString':
            for line in geom:
                n = int(line.length // step) + 2
                for dist in np.linspace(0, line.length, n):
                    pt = line.interpolate(dist)
                    points.append((pt.x, pt.y))
    return np.array(points)

# ---- DBSCAN Clustering Helper ----
def run_dbscan(coords, eps=1000, min_samples=5):
    """Run DBSCAN and return cluster labels."""
    if len(coords) == 0:
        return np.array([])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    return clustering.labels_

# =======================================================
# Data Loading and Preprocessing
# =======================================================

# --- Load anomaly mask and transform ---
with rasterio.open(get_path("anomaly_mask")) as src:
    anomaly_mask = src.read(1)
    mask_transform = src.transform
    mask_crs = src.crs

# --- Load MODIS Landcover ---
with rasterio.open(get_path("modis")) as src:
    modis = src.read(1)
    modis_transform = src.transform
    modis_crs = src.crs

# --- Load SRTM elevation ---
with rasterio.open(get_path("srtm")) as src:
    elevation = src.read(1)
    srtm_transform = src.transform
    srtm_crs = src.crs

# --- Load HydroRIVERS ---
hydro_gdf = gpd.read_file(get_path("hydro"), driver="OpenFileGDB")

# --- (Optional) NDVI/landsat loading if needed downstream ---
landsat_folder = get_path("landsat")
# Example: landsat_ndvi_files = glob.glob(str(landsat_folder / "landsat8_ndvi_*.tif"))


# Check CRS and spatial alignment for all rasters before analysis
# --- Helper: Print Raster Info ---
def print_raster_info(path, label=""):
    with rasterio.open(path) as src:
        print(f"{label} CRS: {src.crs}")
        print(f"{label} Bounds: {src.bounds}")
        print(f"{label} Shape: {src.shape}")

# --- Print NDVI, SRTM, MODIS, Sentinel-1 File Info ---
Xmin, Ymin = -67.66403, -11.23127
Xmax, Ymax = -59.99531, -6.68743

ndvi_dir = "/kaggle/input/landsat8/"
pattern = re.compile(r"landsat8_ndvi_(-?\d+\.\d+)_(-?\d+\.\d+)\.tif", re.IGNORECASE)
all_ndvi_files = glob.glob(ndvi_dir + "landsat8_ndvi_*.tif")

print("NDVI tiles in AOI:")
for ndvi_path in all_ndvi_files:
    basename = os.path.basename(ndvi_path)
    match = pattern.match(basename)
    if match:
        lon, lat = float(match.group(1)), float(match.group(2))
        if (Xmin <= lon <= Xmax) and (Ymin <= lat <= Ymax):
            print_raster_info(ndvi_path, f"Landsat8 NDVI {basename}")

print_raster_info(srtm_path, 'SRTM')
print_raster_info(modis_path, 'MODIS Landcover')
print_raster_info('/kaggle/input/sentinel1-2/s1_vv_-67.66403_-10.23127.tif', 'Sentinel-1')


# ---- Landcover Analysis & Cluster Summaries ----

# 1. Map MODIS class codes to labels for readability
modis_labels = {
    0: 'Water',
    1: 'Evergreen Needleleaf forest',
    2: 'Evergreen Broadleaf forest',
    3: 'Deciduous Needleleaf forest',
    4: 'Deciduous Broadleaf forest',
    5: 'Mixed forest',
    6: 'Closed shrublands',
    7: 'Open shrublands',
    8: 'Woody savannas',
    9: 'Savannas',
    10: 'Grasslands',
    11: 'Permanent wetlands',
    12: 'Croplands',
    13: 'Urban and built-up',
    14: 'Cropland/Nat. veg. mosaic',
    15: 'Snow/ice',
    16: 'Barren/sparse',
    17: 'Unknown',  # not standard, appears in your data
    254: 'Unclassified',
    255: 'Fill'
}

anomalies_df['modis_lc2001_label'] = anomalies_df['modis_lc2001'].map(modis_labels)

# 2. Table: Anomalies per MODIS landcover (count, percent)
modis_counts = (
    anomalies_df['modis_lc2001_label']
    .value_counts(dropna=False)
    .to_frame('count')
)
modis_counts['percent'] = 100 * modis_counts['count'] / modis_counts['count'].sum()
print("Anomaly Points by MODIS Landcover (Count, %):")
print(modis_counts)
display(modis_counts.head(10))

# 3. Table: Cluster x MODIS class (pivot, like a heatmap table)
cluster_modis_table = (
    anomalies_df
    .groupby(['cluster_id', 'modis_lc2001_label'])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)
print("\nAnomaly Points per Cluster by MODIS Landcover:")
print(cluster_modis_table.head(10))

# 4. Optional: Aggregates by landcover or cluster
# Mean elevation and slope by MODIS class
agg_by_modis = (
    anomalies_df
    .groupby('modis_lc2001_label')[['elevation_m', 'slope_deg']]
    .describe()
)
print("\nElevation/Slope Stats by MODIS Landcover:")
print(agg_by_modis)

# Mean elevation and slope by cluster
agg_by_cluster = (
    anomalies_df
    .groupby('cluster_id')[['elevation_m', 'slope_deg']]
    .describe()
)
print("\nElevation/Slope Stats by Cluster:")
print(agg_by_cluster.head(10))

# 5. (Optional) Export for further use
modis_counts.to_csv('/kaggle/working/modis_landcover_summary.csv')
cluster_modis_table.to_csv('/kaggle/working/cluster_modis_summary.csv')
agg_by_modis.to_csv('/kaggle/working/agg_by_modis.csv')
agg_by_cluster.to_csv('/kaggle/working/agg_by_cluster.csv')


# ===============================================
# Anomaly Elevation Cleaning, Visualization & Export
# ===============================================

# Remove SRTM 'nodata' values from analysis
elev_nodata = -32768  # SRTM nodata value
anomalies_filtered = anomalies_df[anomalies_df['elevation_m'] != elev_nodata].copy()

# ---- Visualization: Elevation by Landcover ----
plt.figure(figsize=(12, 5))
sns.boxplot(
    data=anomalies_filtered[~anomalies_filtered['modis_lc2001_label'].isna()],
    x='modis_lc2001_label', y='elevation_m')
plt.xticks(rotation=45, ha='right')
plt.title("Elevation Distribution by MODIS Landcover (Nodata removed)")
plt.tight_layout()
plt.show()

# ---- Bar Plot: Anomaly Count by MODIS Class with Percent Labels ----
ax = sns.barplot(
    data=modis_counts.reset_index().sort_values('count', ascending=False),
    x='count', y='modis_lc2001_label', palette='viridis')
for i, (count, pct) in enumerate(zip(modis_counts['count'], modis_counts['percent'])):
    ax.text(count, i, f"{pct:.1f}%", va='center', ha='left')
plt.title("Anomaly Counts by MODIS Landcover")
plt.tight_layout()
plt.show()

# ---- Export Filtered Anomaly Data ----
# Save as CSV (attributes only)
anomalies_filtered.to_csv('/kaggle/working/anomaly_points_filtered.csv', index=False)
print("Filtered anomaly points (CSV) saved.")

# Save as GeoJSON (for mapping)
gdf = gpd.GeoDataFrame(
    anomalies_filtered,
    geometry=gpd.points_from_xy(anomalies_filtered['longitude'], anomalies_filtered['latitude']),
    crs="EPSG:4326"
)
gdf.to_file('/kaggle/working/anomaly_points_filtered.geojson', driver='GeoJSON')
print("Filtered anomaly points (GeoJSON) saved.")


def extract_pdf_text(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Failed to parse {pdf_path}: {e}")
    return text

# Scan ALL input datasets for PDFs
input_dirs = [
    "/kaggle/input/text-data",
    "/kaggle/input/peerj-walker-2023",
    "/kaggle/input/peerj-3863",  # Add all relevant PDF folders here
    "/kaggle/working/text-data"
]
pdf_files = []
for folder in input_dirs:
    pdf_files += glob.glob(os.path.join(folder, "*.pdf"))

print(f"Found {len(pdf_files)} PDFs.")

output_dir = "/kaggle/working/text-data"
os.makedirs(output_dir, exist_ok=True)
for pdf_path in pdf_files:
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_path = os.path.join(output_dir, basename + ".txt")
    if not os.path.exists(txt_path):  # Avoid re-extraction if file already exists
        text = extract_pdf_text(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extracted text from {pdf_path} to {txt_path} ({len(text)} chars)")
    else:
        print(f"Text for {pdf_path} already extracted.")


out_txt = []

# Cluster summary
df_cluster = pd.read_csv('/kaggle/working/agg_by_cluster.csv')
out_txt.append("=== Cluster Summary ===\n")
out_txt.append(df_cluster.to_string(index=False))

# MODIS summary
df_modis = pd.read_csv('/kaggle/working/agg_by_modis.csv')
out_txt.append("\n\n=== MODIS Landcover Summary ===\n")
out_txt.append(df_modis.to_string(index=False))

# (Optional) Full anomaly points — just sample
df_points = pd.read_csv('/kaggle/working/anomaly_points_filtered.csv')
out_txt.append("\n\n=== Sample Anomaly Points ===\n")
out_txt.append(df_points.head(10).to_string(index=False))

out_txt.append("\n\n=== Notable Outlier Points (if any) ===\n")
out_txt.append(df_points.tail(5).to_string(index=False))

# Save the combined file
summary_txt_path = '/kaggle/working/text-data/project_analysis_summary.txt'
with open(summary_txt_path, 'w') as f:
    f.write('\n'.join(out_txt))

print(f"Project summary exported to {summary_txt_path}")


# -- Setup: OpenAI Key and Parameters --
user_secrets = UserSecretsClient()
openai.api_key = user_secrets.get_secret("OpenAI")
output_dir = "/kaggle/working/text-data"
txt_files = glob.glob(os.path.join(output_dir, "*.txt"))
# Exclude irrelevant/non-archaeology docs
txt_files = [f for f in txt_files if "peerj-3863" not in f and not f.endswith("_final_summary.txt")]

# --- Questions List (add as many as you want) ---
questions = [
    "Are there any references to pre-Columbian earthworks or geoglyphs in this document?",
    "Does the text mention indigenous land use or ancient settlement patterns?",
    "Are specific archaeological sites, features, or artifacts described?",
    "Is there evidence of ancient river or water management systems?",
    "Are there clues about possible areas with buried structures or anthropogenic soils?",
    "Summarize any discussion about the impact of deforestation or modern activity on archaeological preservation.",
    "Are there references to LiDAR or remote sensing discoveries in the Amazon?",
    "What does the document say about the age, dating methods, or chronological sequence of archaeological sites or features?",
    "Is there discussion of social organization, complexity, or evidence of centralized authority in pre-Columbian Amazonian societies?",
    "Are there insights about ancient diets, agriculture, or the use of plants and animals by indigenous peoples?",
    "Which remote sensing, geophysical, or excavation techniques are described or recommended in this work?",
    "Does the document connect archaeological findings to current indigenous practices or oral histories?",
    "Are there criticisms or limitations mentioned about the methods or interpretations used in the document?",
    "Are interdisciplinary studies (e.g., with ecology, soil science, genetics) highlighted as important in this research?",
    "Does the author identify gaps, unanswered questions, or suggest priorities for future research?",
]

def split_text(text, max_words=1200):
    # Breaks text into ~1200-word chunks (safe for GPT-4o context limits)
    paras = text.split("\n\n")
    chunks, chunk, count = [], [], 0
    for para in paras:
        nwords = len(para.split())
        if count + nwords > max_words:
            chunks.append("\n\n".join(chunk))
            chunk, count = [], 0
        chunk.append(para)
        count += nwords
    if chunk:
        chunks.append("\n\n".join(chunk))
    return chunks

def gpt_answer(question, context, system_prompt="You are an expert Amazon archaeologist."):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.2,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- Q&A Per Document and Question ---
for txt_path in txt_files:
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = split_text(text, max_words=1200)
    doc_basename = os.path.splitext(os.path.basename(txt_path))[0]
    for q_num, question in enumerate(questions):
        all_answers = []
        for i, chunk in enumerate(chunks):
            print(f"Doc: {doc_basename} | Q{q_num+1}/{len(questions)} | Chunk {i+1}/{len(chunks)}")
            answer = gpt_answer(question, chunk)
            all_answers.append(f"Chunk {i+1}: {answer}")
        answer_path = os.path.join(output_dir, f"{doc_basename}_q{q_num+1}.txt")
        with open(answer_path, "w", encoding="utf-8") as f:
            f.write(f"Question: {question}\n\n" + "\n\n---\n\n".join(all_answers))
        print(f"Saved Q{q_num+1} answers to {answer_path}")

# --- Chunked, Multi-step Executive Summary ---
def pyramid_summarize(big_text, question, system_prompt):
    """Chunk input if too large, summarize each, then summarize the summaries."""
    MAX_INPUT_CHARS = 10000
    if len(big_text) <= MAX_INPUT_CHARS:
        return gpt_answer(question, big_text, system_prompt=system_prompt)
    # Otherwise, break into ~MAX_INPUT_CHARS chunks and summarize each, then summarize those.
    n_chunks = (len(big_text) // MAX_INPUT_CHARS) + 1
    part_size = len(big_text) // n_chunks + 1
    part_summaries = []
    for i in range(n_chunks):
        part = big_text[i * part_size : (i + 1) * part_size]
        part_summary = gpt_answer(question, part, system_prompt=system_prompt)
        part_summaries.append(f"PART {i+1}:\n{part_summary}")
    # Now synthesize those summaries
    final_input = "\n\n".join(part_summaries)
    return gpt_answer(
        question="Synthesize and summarize these partial executive summaries into a concise, structured document with bullet points. Include citations and coverage of all key research questions.",
        context=final_input,
        system_prompt=system_prompt,
    )

for txt_path in txt_files:
    doc_basename = os.path.splitext(os.path.basename(txt_path))[0]
    all_summaries = []
    for q_num, question in enumerate(questions):
        answer_path = os.path.join(output_dir, f"{doc_basename}_q{q_num+1}.txt")
        if os.path.exists(answer_path):
            with open(answer_path, "r", encoding="utf-8") as f:
                all_summaries.append(f.read())
    super_summary_input = "\n\n".join(all_summaries)
    # Multi-chunk summary for long docs!
    final_summary = pyramid_summarize(
        super_summary_input,
        question="Synthesize the main findings, recurring themes, and unique insights about Amazonian archaeology in this document. Highlight any evidence for anthropogenic features, landscape modification, indigenous land use, or new discoveries. Note any limitations, controversies, or open research questions. Use bullet points and cite chunks as needed.",
        system_prompt="You are an expert Amazon archaeologist."
    )
    summary_path = os.path.join(output_dir, f"{doc_basename}_final_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(final_summary)
    print(f"Saved final summary to {summary_path}")


# Helper: Extract citation from file name (customize as needed)
def citation_from_filename(filename):
    # Simple stub: you may want to parse authors/year/title if possible
    base = os.path.basename(filename)
    if "peerj" in base.lower():
        if "15137" in base:
            return "Walker et al. 2023 (PeerJ 15137)"
        if "3863" in base:
            return "de Souza et al. 2017 (PeerJ 3863)"
    return base

# --- For theme matrix: ---
theme_matrix = []
theme_labels = [q.split('?')[0][:60] + "..." for q in questions]  # Shorten for table

for txt_path in txt_files:
    doc_basename = os.path.splitext(os.path.basename(txt_path))[0]
    all_summaries = []
    theme_presence = []

    # Collect all answers for this doc, all questions
    for q_num, question in enumerate(questions):
        answer_path = os.path.join(output_dir, f"{doc_basename}_q{q_num+1}.txt")
        if os.path.exists(answer_path):
            with open(answer_path, "r", encoding="utf-8") as f:
                answer_text = f.read()
                all_summaries.append(f"**Q{q_num+1}: {question}**\n{answer_text}\n")
                # Theme detection: if answer is long, contains 'yes', or mentions details, mark as covered
                presence = "YES" if any(s in answer_text.lower() for s in ["yes", "evidence", "describ", "discuss", "summar", "present", "highlight", "report"]) and len(answer_text) > 50 else "no"
                theme_presence.append(presence)
        else:
            all_summaries.append(f"**Q{q_num+1}: {question}**\n(No answer found)\n")
            theme_presence.append("no")

    # Compose all answers into a single string for summarization
    super_summary_input = "\n\n".join(all_summaries)

    # Compose GPT prompt for summary, include traceable bullet points and citation
    citation = citation_from_filename(txt_path)
    system_prompt = (
        f"You are an expert Amazon archaeologist. When summarizing, include the citation [{citation}] "
        "at the top. Present the main findings in bullet points, and organize by key research questions/themes. "
        "Reference which findings come from which specific question/section. "
        "Highlight evidence for anthropogenic features, landscape modification, indigenous land use, "
        "or new discoveries. Note any limitations, controversies, or open research questions."
    )

    final_summary = gpt_answer(
        question="Provide a detailed, bullet-point executive summary of the document. Organize by research theme. Include citation and reference which answers correspond to which research question.",
        context=super_summary_input,
        system_prompt=system_prompt
    )

    # Save to file
    summary_path = os.path.join(output_dir, f"{doc_basename}_final_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(final_summary)
    print(f"Saved final summary to {summary_path}")

    # Collect for theme matrix
    theme_matrix.append([citation] + theme_presence)

# --- Output a cross-document theme matrix table (CSV and print) ---
theme_df = pd.DataFrame(theme_matrix, columns=["Document"] + theme_labels)
matrix_path = os.path.join(output_dir, "cross_document_theme_matrix.csv")
theme_df.to_csv(matrix_path, index=False)
print("\nCross-document theme coverage matrix:\n")
print(theme_df)
print(f"\nSaved theme matrix to: {matrix_path}")


# Load your theme matrix CSV
matrix_path = "/kaggle/working/text-data/cross_document_theme_matrix.csv"
theme_df = pd.read_csv(matrix_path)

# Reformat for heatmap (YES=1, no=0)
hm_data = (theme_df.iloc[:, 1:] == "YES").astype(int)
doc_labels = theme_df["Document"].tolist()

plt.figure(figsize=(min(18, 2+hm_data.shape[1]), 1+hm_data.shape[0]))
sns.heatmap(
    hm_data,
    annot=True, fmt='d', cmap='YlGnBu',
    yticklabels=doc_labels, xticklabels=theme_df.columns[1:], 
    cbar=False, linewidths=0.5, linecolor='gray'
)
plt.title("Amazonian Archaeology Theme Coverage by Document", fontsize=16, pad=20)
plt.xlabel("Research Theme", fontsize=13)
plt.ylabel("Document", fontsize=13)
plt.tight_layout()
plt.show()


user_secrets = UserSecretsClient()
openai.api_key = user_secrets.get_secret("OpenAI")

# Collect all final summaries
output_dir = "/kaggle/working/text-data"
summary_files = glob.glob(os.path.join(output_dir, "*_final_summary.txt"))

all_summaries = []
for f in summary_files:
    doc_name = os.path.basename(f).replace("_final_summary.txt", "")
    with open(f, "r", encoding="utf-8") as file:
        summary = file.read()
    all_summaries.append(f"--- {doc_name} ---\n{summary}")

meta_input = "\n\n".join(all_summaries)

def gpt_meta_summary(big_context):
    # Use chunking for long input if needed
    max_tokens = 8000  # Adjust if needed
    if len(big_context.split()) < max_tokens:
        return openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert Amazon archaeologist. Produce a comparative meta-summary of these research documents, noting overlap, unique findings, gaps, and patterns. Organize with bullet points, mention documents by name, and cite themes."},
                {"role": "user", "content": f"Context:\n{big_context}\n\nQuestion: What are the key comparative findings, patterns, and open questions across these Amazonian archaeology research documents?"}
            ],
            temperature=0.2,
            max_tokens=800
        ).choices[0].message.content
    # If too big, split into chunks and summarize each, then synthesize.
    print("Input too large, chunking meta-summary...")
    parts = []
    words = big_context.split()
    for i in range(0, len(words), max_tokens):
        part = " ".join(words[i:i+max_tokens])
        part_summary = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert Amazon archaeologist. Summarize the following Amazonian archaeology research findings."},
                {"role": "user", "content": f"{part}"}
            ],
            temperature=0.2,
            max_tokens=400
        ).choices[0].message.content
        parts.append(part_summary)
    # Synthesize all part summaries
    final_input = "\n\n".join(parts)
    final_meta = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert Amazon archaeologist. Synthesize these findings into a comparative meta-summary."},
            {"role": "user", "content": final_input}
        ],
        temperature=0.2,
        max_tokens=800
    ).choices[0].message.content
    return final_meta

meta_summary = gpt_meta_summary(meta_input)

meta_summary_path = os.path.join(output_dir, "project_meta_summary.txt")
with open(meta_summary_path, "w", encoding="utf-8") as f:
    f.write(meta_summary)
print("Meta-summary saved to", meta_summary_path)
print("\n=== META-SUMMARY ===\n")
print(meta_summary)




