# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing
import os

file_paths = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file_paths.append(os.path.join(dirname, filename))

# Print only summary + a few examples
print(f"Total files found: {len(file_paths)}")
print("Example files:")
for path in file_paths[:5]:  
    print("-", path)


# Install and import all required packages
!pip install -q geopandas rasterio matplotlib faiss-cpu torch folium textblob transformers sentence_transformers rouge-score evaluate


# # Install and import all required packages
import os
import pickle
import faiss
import torch
import folium
import geopandas as gpd
import numpy as np
from textblob import TextBlob
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from IPython.display import display, Markdown


# Load embeddings, indices, and metadata for both modalities
CLIP_INDEX_PATH = "/kaggle/input/embeddings/clip_tile_index.faiss"
CLIP_META_PATH = "/kaggle/input/embeddings/clip_tile_metadata.pkl"
TEXT_INDEX_PATH = "/kaggle/input/embeddings/text_index.faiss"
TEXT_META_PATH = "/kaggle/input/embeddings/text_metadata.pkl"
TILE_BOUNDS_PATH = "/kaggle/input/tile-bounds/tile_bounds.pkl"
TILE_DIR = "/kaggle/input/tiles/tiles"
LIDAR_SHP_DIR = "/kaggle/input/lidar-deforestation"


# ---------- LOAD MODELS ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_model = SentenceTransformer("all-MiniLM-L6-v2")


# Load FAISS indices
def load_faiss_index(index_path):
    return faiss.read_index(index_path)

# Load metadata
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Load everything
clip_index = load_faiss_index(CLIP_INDEX_PATH)
clip_meta = load_pickle(CLIP_META_PATH)

text_index = load_faiss_index(TEXT_INDEX_PATH)
text_meta = load_pickle(TEXT_META_PATH)

tile_bounds = load_pickle(TILE_BOUNDS_PATH)

# Confirm loading
print(f"CLIP Index size: {clip_index.ntotal}")
print(f"Text Index size: {text_index.ntotal}")
print(f"Number of tile bounds: {len(tile_bounds)}")


# ---------- SPELL CORRECT ----------
def correct_query(query):
    return str(TextBlob(query).correct())

# ---------- EMBED QUERY ----------
def embed_query_clip(query):
    inputs = clip_processor(text=query, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        vector = clip_model.get_text_features(**inputs).cpu().numpy().astype("float32")
    return vector

def embed_query_text(query):
    return text_model.encode([query]).astype("float32")


# Load SentenceTransformer model
text_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Function to query text index
def retrieve_text_chunks(query, k=10):
    query_embed = text_embedder.encode([query])
    D, I = text_index.search(query_embed, k)

    text_chunks = text_meta["texts"]
    results = [text_chunks[i] if i < len(text_chunks) else f"[Missing chunk {i}]" for i in I[0]]
    return results


# Load CLIP Text Encoder
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to encode query with CLIP
def encode_clip_text(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        return clip_model.get_text_features(**inputs).cpu().numpy()

# Function to query image tile index
def retrieve_tile_ids(query, k=10):
    query_embed = encode_clip_text(query)
    D, I = clip_index.search(query_embed, k)
    results = [clip_meta[i] for i in I[0]]
    return results


from openai import OpenAI
import os
import openai

# If using Kaggle secrets
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
os.environ["OPENAI_API_KEY"] = user_secrets.get_secret("OPENAI_API_KEY")

# Initialize the client using your Kaggle environment variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def run_geographic_expansion_agent(query, retrieved_texts):
    expansion_prompt = f"""
You are an archaeological reasoning assistant. The user asked:

"{query}"

You have access to historical text excerpts and satellite imagery tile references that may contain clues about river proximity, terrain, and prior settlements.

Your task is to:
- Analyze the following text chunks and satellite tile IDs
- Identify signs of multiple **distinct river systems or regions** mentioned or implied
- Propose 2â€“3 **alternative Amazon river regions** that could also be promising areas for human settlement based on the data retrieved or archaeological significance.

Only suggest rivers or regions if they are either:
- mentioned in the text, OR
- clearly inferred from the distribution or features in the imagery

Historical Texts:
{chr(10).join(['- ' + t for t in retrieved_texts])}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": expansion_prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()


import openai
import os

# Load OpenAI API key from Kaggle secrets or environment
openai.api_key = os.environ.get("OPENAI_API_KEY")  # Ensure this is set in Kaggle secrets

# Hypothesis generator with structured response
def generate_hypothesis(query, text_chunks, top_tiles=None, expanded_regions=None):
    tile_references = ""
    if top_tiles:
        tile_references = "\n".join([f"- {tile}" for tile in top_tiles[:5]])

    prompt = f"""
You are an expert archaeological assistant with access to both historical texts and satellite imagery.

User query: {query}

You are given the following historical text excerpts:
{text_chunks}

You are also provided with satellite image tiles that exhibit notable landscape features:
{tile_references}
"""
    if expanded_regions:
        prompt += f"""

Additionally, you are encouraged to consider these other potentially relevant regions based on known Amazonian settlement patterns:
{expanded_regions}

Based on both the historical text and image references, provide a detailed and academically grounded hypothesis about what these features may represent. Be specific in your analysis, quote the exact references wherever needed, referencing past known civilizations, agricultural or ceremonial patterns, or any anthropogenic shapes like rectangles, grids, or radial structures that are visible.

Structure your output like this:
**Hypothesis**
...

**Justification**
...

Use clear language that could be understood by a research reviewer or geospatial archaeologist.
"""

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    
    return response.choices[0].message.content.strip()


import folium
from folium import IFrame
import base64
import os
import geopandas as gpd
import branca.colormap as cm

def render_tile_map(tile_ids, tile_bounds, hypothesis=None, confidence_scores=None):
    default_location = [-3.4653, -62.2159]
    fmap = folium.Map(location=default_location, zoom_start=5)

    for i, tile_id in enumerate(tile_ids):
        if tile_id not in tile_bounds:
            continue
        lat_min, lon_min, lat_max, lon_max = tile_bounds[tile_id]
        bounds = [[lat_min, lon_min], [lat_max, lon_max]]

        # Load image and encode as base64
        image_path = os.path.join(TILE_DIR, tile_id)
        encoded_img = ""
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                encoded_img = base64.b64encode(f.read()).decode()
        
        # Optional confidence for color
        score = confidence_scores[i] if confidence_scores else 0.5
        color = "green" if score >= 0.75 else "orange" if score >= 0.5 else "red"

        html = f"<b>{tile_id}</b><br>"
        if encoded_img:
            html += f'<img src="data:image/png;base64,{encoded_img}" width="200" />'

        popup = folium.Popup(IFrame(html, width=210, height=220), max_width=250)

        folium.Rectangle(
            bounds=bounds,
            color=color,
            fill=True,
            fill_opacity=0.4,
            popup=popup
        ).add_to(fmap)
        

    if hypothesis:
        folium.Marker(
            location=default_location,
            icon=folium.Icon(color='green'),
            popup=hypothesis[:300]
        ).add_to(fmap)

    return fmap
    
def add_shapefile_overlay(fmap, shp_dir, max_features=500):
    for fname in os.listdir(shp_dir):
        if fname.endswith(".shp"):
            path = os.path.join(shp_dir, fname)
            gdf = gpd.read_file(path)
            if len(gdf) > max_features:
                gdf = gdf.sample(max_features)  # subsample
            folium.GeoJson(gdf, name=fname).add_to(fmap)
    return fmap


def run_justification_agent(query, text_chunks, tile_ids, llm_hypothesis):
    justification_prompt = f"""
You are a scientific reasoning agent. Your task is to verify whether the following hypothesis is well-supported by the provided data.

User Query:
{query}

Hypothesis:
{llm_hypothesis}

Supporting Text Chunks:
{chr(10).join(['- ' + chunk for chunk in text_chunks])}

Referenced Image Tile IDs:
{tile_ids}

Please return a structured justification. Highlight which parts of the hypothesis are strongly supported, weakly supported, or unsupported based on the data.
"""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": justification_prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


def run_evaluation_agent(hypothesis, justification):
    evaluation_prompt = f"""
You are an evaluator AI assessing the scientific credibility of a hypothesis and its justification.

Hypothesis:
{hypothesis}

Justification:
{justification}

Please evaluate the overall credibility and clarity of the hypothesis. Return:

### Evaluation
- Confidence Score (1â€“10): X/10
- Major Strengths:
- Limitations or Uncertainty Factors:
- Should this be pursued further? (Yes/No and Why)
"""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": evaluation_prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


#change your query here
query = "Where might ancient settlements exist near major rivers in the Amazon?"

# Step 1: Retrieve
text_results = retrieve_text_chunks(query, k=10)
image_results = retrieve_tile_ids("circular mound structures", k=10)

# Step 2: Expand regionally
expanded_regions = run_geographic_expansion_agent(query, text_results)

# Step 3: Generate hypothesis
hypothesis_output = generate_hypothesis(
    query=query,
    text_chunks="\n\n".join(text_results),
    top_tiles=image_results,
    expanded_regions=expanded_regions
)

# Step 4: Justify + Evaluate
justification = run_justification_agent(query, text_results, image_results, hypothesis_output)
evaluation = run_evaluation_agent(hypothesis_output, justification)


from IPython.display import Markdown, display

# Wrap the output in Markdown
display(Markdown(hypothesis_output))


display(Markdown(justification))


display(Markdown(evaluation))


# Display top-k image tile matches as thumbnails for visual reference
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt
import os

def show_top_tiles(tile_ids, tile_dir=TILE_DIR, size=(150, 150)):
    """
    Display top image tiles used in RAG with resized thumbnails.
    """
    num_tiles = min(len(tile_ids), 10)
    fig, axes = plt.subplots(1, num_tiles, figsize=(num_tiles * 2, 2.5))
    
    if num_tiles == 1:
        axes = [axes]  # ensure iterable if only one tile
    
    for i in range(num_tiles):
        tile_path = os.path.join(tile_dir, tile_ids[i])
        try:
            img = Image.open(tile_path).resize(size)
            axes[i].imshow(img)
            axes[i].set_title(tile_ids[i][:20], fontsize=8)
            axes[i].axis("off")
        except Exception as e:
            axes[i].set_title("Missing", fontsize=8)
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()


show_top_tiles(image_results)


from IPython.display import display

tile_ids = image_results

map_out = render_tile_map(tile_ids, tile_bounds, hypothesis=hypothesis_output)

# Add .shp overlays 
map_out = add_shapefile_overlay(map_out, LIDAR_SHP_DIR)

# Display
display(map_out)


from evaluate import load

rouge = load("rouge")

# Duplicate the hypothesis for each reference
predictions = [hypothesis_output] * len(text_results)
references = text_results  # list of retrieved chunks

# Compute ROUGE
results = rouge.compute(predictions=predictions, references=references)
print(results)


# ðŸ”½ Set up paths
output_path = "/kaggle/working/submission.txt"

# ðŸ”½ Prepare all relevant parts
header = "===== Amazon Archaeological Site Hypothesis Report =====\n\n"

# Hypothesis section
hypothesis_section = "### Hypothesis:\n" + hypothesis_output + "\n\n"

# Justification section
justification_section = "### Justification:\n" + justification + "\n\n"

# Evaluation score summary
evaluation_section = ""
if "rouge_scores" in globals():
    evaluation_section = "### Evaluation (ROUGE):\n"
    for metric, score in rouge_scores.items():
        evaluation_section += f"{metric}: {score:.4f}\n"
    evaluation_section += "\n"

# Map tile matches
tile_section = "### Top Satellite Tile Matches Used:\n"
tile_section += "\n".join(image_results[:10]) + "\n\n"

# Save to submission.txt
with open(output_path, "w", encoding="utf-8") as f:
    f.write(header)
    f.write(hypothesis_section)
    f.write(justification_section)
    f.write(evaluation_section)
    f.write(tile_section)

print(f"âœ… submission.txt saved at: {output_path}")

