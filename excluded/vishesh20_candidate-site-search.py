!pip install huggingface_hub open_clip_torch -q
# !git clone https://github.com/ChenDelong1999/RemoteCLIP/ #Uncomment when running the first time


from huggingface_hub import hf_hub_download
import torch, open_clip
from PIL import Image
from IPython.display import display

for model_name in ['ViT-B-32']: #, 'ViT-B-32', 'ViT-L-14']: #faster loading
# for model_name in ['RN50', 'ViT-B-32', 'ViT-L-14']: #all models
    checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", cache_dir='checkpoints')
    print(f'{model_name} is downloaded to {checkpoint_path}.')


!pip install osmnx geopandas shapely -q
!pip install rasterio -q
!pip install elevation -q
!sudo apt-get install osmium-tool -q


model_name = 'ViT-B-32' # options ['RN50', 'ViT-B-32', 'ViT-L-14']
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

path_to_your_checkpoints = 'checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38'

ckpt = torch.load(f"{path_to_your_checkpoints}/RemoteCLIP-{model_name}.pt", map_location="cpu")
message = model.load_state_dict(ckpt)
print(message)
model = model.cuda().eval()


import ee
import geemap
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime # For potential client-side date parsing if needed

ee.Authenticate()
ee.Initialize(project='openaitoz-460313')


import json
# Load the grid cells from the JSON file
output_file = '/kaggle/input/amazon-geoglyphs/interesting_grid_cells.json'  # Nitai to change to your local address
try:
    with open(output_file, 'r') as f:
        grid_cells = json.load(f)
    print(f"Successfully loaded {len(grid_cells)} grid cells from {output_file}")
except FileNotFoundError:
    print(f"Error: Grid cells file not found at {output_file}")
    grid_cells = []
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {output_file}")
    grid_cells = []


#@title remoteclip config
import os
from PIL import Image
import torch
from torchvision import transforms
from glob import glob

# Define your text queries
text_queries = [
    'A large circular pattern carved into bare earth, partially faded and surrounded by vegetation.',
    'Ancient ring or earth circle visible from overhead satellite view.',
    'Concentric circular traces on open land, possibly man-made or symbolic.',
    'A weathered circle structure embedded in deforested terrain.',
    'Round or ring-shaped feature cutting through natural textures.',
    'Large faint circle visible in forest clearing, likely a geoglyph.',    
    'Square or trapezoid structure with straight lines visible in open terrain.',
    'Man-made angular outline on the ground, possibly an ancient enclosure.',
    'A faint geometric shape resembling ruins or foundations.',
    'Polygonal earthwork partially hidden by regrowth.',
    'Geometric pattern on bare land, inconsistent with natural forms.',
    'Straight-edged rectangular structure from aerial perspective.',    
    'Several geometric shapes joined or overlapping in a clearing.',
    'Interconnected patterns forming partial rings and squares.',
    'Complex shape visible from above with partial occlusion by trees.',    
    'No visible man-made structure; only natural terrain and shadows.',
    'Flat land without any clear shapes or outlines.',
    'Forest canopy and textures show no patterns or symbols.',
    'Vegetation and ground cover with no angular or circular features.',
    'No sign of ancient construction or regular geometry in the image.'
]
text = tokenizer(text_queries)



import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import ee
import geemap
import cv2
import rasterio
from rasterio.plot import show
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from skimage.measure import approximate_polygon, subdivide_polygon
from scipy.spatial import distance
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from PIL import Image
import torch.nn.functional as F
import contextlib
import cupy as cp # Import CuPy
import cupyx.scipy.ndimage # For GPU-accelerated image processing tasks if needed
from cupyx.scipy.signal import convolve2d # Example of a cuPy signal function
import os
import pandas as pd
import numpy as np
import rasterio
from PIL import Image
from tqdm.notebook import tqdm # Use tqdm.notebook for Colab progress bars
import math

# Define batch size for RemoteCLIP processing (adjust based on GPU memory)
N = 500000  # Number of grid cells to process for Vishesh 
BATCH_SIZE = 64 # Increased batch size, adjust based on GPU memory
DOWNLOAD_WORKERS = 16 # More concurrent downloads
CANOPY_WORKERS = os.cpu_count() # Use all CPU cores for canopy calculation

# Create DataFrame for results
results_df = pd.DataFrame(columns=['grid_id', 'geoglyph_prob', 'canopy_cover'])
# output_csv_path = '/content/drive/MyDrive/EarthEngineExports/openai2z/branched_strategy_results/interesting_geoglyph_canopy_results_nitai.csv'  # for Nitai
output_csv_path = '/kaggle/working/interesting_geoglyph_canopy_results_vishesh_kaggle.csv' # for Vishesh

# Resume functionality (same as before)
start_index = int(N*0.75)        # starting from midpoint for Vishesh

if os.path.exists(output_csv_path):
    try:
        existing_df = pd.read_csv(output_csv_path)
        results_df = existing_df
        if not existing_df.empty:
            last_processed_grid_id = existing_df['grid_id'].iloc[-1]
            try:
                last_processed_index = next(i for i, cell in enumerate(grid_cells)
                    if f"{cell['minx']},{cell['miny']},{cell['maxx']},{cell['maxy']}" == last_processed_grid_id)
                start_index = last_processed_index + 1
                print(f"Resuming from index {start_index} (grid_id: {last_processed_grid_id})")
            except StopIteration:
                print(f"Last grid_id not found. Starting from index 0.")
                start_index = 0
    except Exception as e:
        print(f"Error loading CSV: {e}. Starting from index 0.")

# --- Optimized Functions ---

# Check for empty canopy image (can remain largely the same)
def is_tif_empty(tif_path):
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            if src.nodata is not None:
                if np.amax(data)==255 and np.amin(data)==255:
                  return True
                nodata_mask = (data == src.nodata)
                non_nodata_count = np.sum(~nodata_mask)
                return non_nodata_count == 0
            else:
                return False
    except rasterio.errors.RasterioIOError:
        print(f"Error reading raster file: {tif_path}. Treating as empty.")
        return True

# Download functions (can remain largely the same)
def download_tms_geotiff(bbox, output_path):
    try:
        geemap.tms_to_geotiff(
            output=output_path,
            bbox=bbox,
            zoom=17,
            source='Satellite',
            quiet=True
        )
        return True
    except Exception as e:
        # print(f"Error downloading TMS: {e}") # Comment out for cleaner output during batch processing
        return False

def download_canopy_height(bbox, output_path):
    try:
        canopy_ht = ee.Image('users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1')
        roi = ee.Geometry.Rectangle(bbox)
        canopy_ht_clipped = canopy_ht.clip(roi)

        with contextlib.redirect_stdout(None):
            geemap.ee_export_image(
                canopy_ht_clipped,
                filename=output_path,
                scale=1,
                region=roi,
                file_per_band=False,
                timeout=300 # Increased timeout
            )
        return True
    except Exception as e:
        # print(f"Error downloading canopy: {e}") # Comment out for cleaner output
        return False

def batch_remoteclip_prediction(image_paths):
    """Process images in batches using GPU"""
    if not image_paths:
        return []

    # Load and preprocess images on CPU first, then transfer to GPU
    # This is often more efficient than opening/processing on GPU directly
    valid_paths = []
    image_tensors_cpu = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            # Ensure images have consistent size if needed for batching, or handle resizing.
            # RemoteCLIP preprocess handles resizing, but if you have other batch steps, be aware.
            img_tensor = preprocess(img) # preprocess returns a tensor
            image_tensors_cpu.append(img_tensor)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error preprocessing {path}: {e}")

    if not image_tensors_cpu:
        return []

    # Stack tensors and move to GPU
    batch = torch.stack(image_tensors_cpu, 0).cuda()

    # Batch processing on GPU
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(batch)
        text_features = model.encode_text(text.cuda())
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        logits = (image_features @ text_features.T) * 100
        probs = logits.softmax(dim=-1).cpu().numpy() # Move results back to CPU

    # Calculate geoglyph probabilities
    results = {}
    for i, path in enumerate(valid_paths):
        # Assuming the last 5 text queries are for non-geoglyph classes
        # geoglyph_prob = sum(probs[i][:-5]) # Sum of probabilities for positive classes
        geoglyph_prob = 1 - np.sum(probs[i][-5:]) # 1 - sum of probabilities for negative classes

        results[path] = geoglyph_prob

    # Return probabilities in the same order as input paths
    return [results.get(path, -1.0) for path in image_paths]


# Function to calculate canopy cover (can remain on CPU or explore CuPy for intermediate steps)
def calculate_canopy_cover(canopy_height_path, threshold=15.0):
    try:
        with rasterio.open(canopy_height_path) as src:
            canopy_data = src.read(1)
            if src.nodata is not None:
                nodata_mask = (canopy_data == src.nodata)
                canopy_data = np.ma.masked_array(canopy_data, mask=nodata_mask)

        if isinstance(canopy_data, np.ma.MaskedArray):
            
             # Use CuPy for mask operations if data fits in GPU memory
             try:
                 canopy_data_cp = cp.asarray(canopy_data.filled(src.nodata)) # Move filled data to GPU
                 nodata_mask_cp = cp.asarray(canopy_data.mask)
                 vegetation_mask_cp = (canopy_data_cp > threshold) & (canopy_data_cp < 255) & (~nodata_mask_cp)
                 total_pixels_cp = cp.sum(~nodata_mask_cp)
                 vegetated_pixels_cp = cp.sum(vegetation_mask_cp)

                 total_pixels = cp.asnumpy(total_pixels_cp).item()
                 vegetated_pixels = cp.asnumpy(vegetated_pixels_cp).item()

                 del canopy_data_cp, nodata_mask_cp, vegetation_mask_cp, total_pixels_cp, vegetated_pixels_cp
                 cp._default_memory_pool.free_all_blocks() # Free GPU memory

             except cp.cuda.memory.OutOfMemoryError:
                 # Fallback to NumPy if data is too large for GPU
                 # print("CuPy OOM for canopy calculation, falling back to NumPy.")
                 vegetation_mask = (canopy_data > threshold)&(canopy_data < 255) & (~canopy_data.mask)
                 total_pixels = canopy_data.count()
                 vegetated_pixels = np.sum(vegetation_mask)
        else:
            # Use CuPy for mask operations if data fits in GPU memory
            try:
                canopy_data_cp = cp.asarray(canopy_data)
                vegetation_mask_cp = (canopy_data_cp > threshold) & (canopy_data_cp < 255)
                total_pixels_cp = cp.array(canopy_data_cp.size)
                vegetated_pixels_cp = cp.sum(vegetation_mask_cp)
    
                total_pixels = cp.asnumpy(total_pixels_cp).item()
                vegetated_pixels = cp.asnumpy(vegetated_pixels_cp).item()
    
                del canopy_data_cp, vegetation_mask_cp, total_pixels_cp, vegetated_pixels_cp
                cp._default_memory_pool.free_all_blocks()
    
            except cp.cuda.memory.OutOfMemoryError:
                 # Fallback to NumPy
                 # print("CuPy OOM for canopy calculation, falling back to NumPy.")
                 vegetation_mask = (canopy_data > threshold)&(canopy_data < 255)
                 total_pixels = canopy_data.size
                 vegetated_pixels = np.sum(vegetation_mask)

        if total_pixels == 0:
            return 0.0

        canopy_cover = vegetated_pixels / total_pixels
        return canopy_cover
    except rasterio.errors.RasterioIOError:
        # print(f"Error reading raster file: {canopy_height_path}")
        return -1.0
    except Exception as e:
        # print(f"Error calculating canopy cover from {canopy_height_path}: {e}")
        return -1.0

def parallel_canopy_cover(canopy_paths, max_workers):
    """Calculate canopy cover in parallel"""
    results = {}
    if not canopy_paths:
        return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(calculate_canopy_cover, path): path for path in canopy_paths}

        # Collect results as they complete
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                results[path] = future.result()
            except Exception as exc:
                print(f'{path} generated an exception: {exc}')
                results[path] = -1.0 # Indicate failure

    # Return results in the original order of canopy_paths
    return [results.get(path, -1.0) for path in canopy_paths]

# --- Main Processing Loop ---
print(f"Starting processing from index {start_index} up to {N}")

# Determine the number of grid cells to process in this run
num_to_process = min(N, len(grid_cells)) - start_index
print(f"Processing {num_to_process} grid cells in total from index {start_index}.")

# Process in batches
for batch_start in tqdm(range(start_index, min(N, len(grid_cells)), BATCH_SIZE), desc="Processing Batches"):
    batch_end = min(batch_start + BATCH_SIZE, min(N, len(grid_cells)))
    current_batch_indices = list(range(batch_start, batch_end))
    # print(f"\nProcessing batch from index {batch_start} to {batch_end-1}")

    batch_data_futures = []
    # Download Phase (parallel)
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        download_tasks = []
        for idx in current_batch_indices:
            cell = grid_cells[idx]
            bbox = [cell['minx'], cell['miny'], cell['maxx'], cell['maxy']]
            tms_path = f'/content/tms_image_{idx}.tif'
            canopy_path = f'/content/canopy_height_{idx}.tif'
            download_tasks.append(executor.submit(download_tms_geotiff, bbox, tms_path))
            download_tasks.append(executor.submit(download_canopy_height, bbox, canopy_path))

        # Collect download results for the current batch
        # We need to map the results back to the original index
        download_results = [future.result() for future in as_completed(download_tasks)]

        # Organize downloaded file paths and success status
        batch_files_info = []
        for i, idx in enumerate(current_batch_indices):
            tms_path = f'/content/tms_image_{idx}.tif'
            canopy_path = f'/content/canopy_height_{idx}.tif'
            # We need a way to know which download result corresponds to which file/index.
            # A better approach is to submit tasks that return their result along with the index.
            # Let's restructure the download phase slightly.

    # Restructured Download Phase
    batch_download_info = [] # Store (idx, tms_path, canopy_path, tms_success, canopy_success)
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        future_to_task = {}
        for idx in current_batch_indices:
             cell = grid_cells[idx]
             bbox = [cell['minx'], cell['miny'], cell['maxx'], cell['maxy']]
             tms_path = f'/kaggle/working/tms_image_{idx}.tif'
             canopy_path = f'/kaggle/working/canopy_height_{idx}.tif'
             future_to_task[executor.submit(download_tms_geotiff, bbox, tms_path)] = ('tms', idx, tms_path)
             future_to_task[executor.submit(download_canopy_height, bbox, canopy_path)] = ('canopy', idx, canopy_path)

        # Initialize a structure to hold download status for the batch
        temp_batch_info = {idx: {'tms_path': f'/kaggle/working/tms_image_{idx}.tif',
                                 'canopy_path': f'/kaggle/working/canopy_height_{idx}.tif',
                                 'tms_success': False,
                                 'canopy_success': False}
                           for idx in current_batch_indices}

        for future in as_completed(future_to_task):
            task_type, idx, path = future_to_task[future]
            try:
                success = future.result()
                if task_type == 'tms':
                    temp_batch_info[idx]['tms_success'] = success
                elif task_type == 'canopy':
                    temp_batch_info[idx]['canopy_success'] = success
            except Exception as exc:
                print(f'Download task for {path} generated an exception: {exc}')
                # Success remains False

    batch_download_info = list(temp_batch_info.values())

    # Prepare data for CLIP and Canopy processing based on download success
    tms_paths_for_clip = []
    canopy_paths_for_canopy = []
    batch_data_for_processing = [] # Store info for joining results later

    for item in batch_download_info:
        if item['tms_success']:
            tms_paths_for_clip.append(item['tms_path'])
            batch_data_for_processing.append({
                'idx': current_batch_indices[batch_download_info.index(item)], # Need original index
                'tms_path': item['tms_path'],
                'canopy_path': item['canopy_path'],
                'canopy_success_flag': item['canopy_success'] # Flag for later
            })
            if item['canopy_success']:
                canopy_paths_for_canopy.append(item['canopy_path'])
        else:
             # Cleanup failed TMS downloads immediately
            if os.path.exists(item['tms_path']):
                os.remove(item['tms_path'])
            if os.path.exists(item['canopy_path']): # Also cleanup canopy if TMS failed for that cell
                os.remove(item['canopy_path'])


    # RemoteCLIP Batch Processing
    clip_results = batch_remoteclip_prediction(tms_paths_for_clip)

    # Canopy Cover Processing
    canopy_results = parallel_canopy_cover(canopy_paths_for_canopy, max_workers=CANOPY_WORKERS)

    # Map results back to original grid cell indices
    clip_results_map = {path: prob for path, prob in zip(tms_paths_for_clip, clip_results)}
    canopy_results_map = {path: cover for path, cover in zip(canopy_paths_for_canopy, canopy_results)}

    # Process results and update DataFrame
    new_rows = []
    for data in batch_data_for_processing:
        idx = data['idx']
        tms_path = data['tms_path']
        canopy_path = data['canopy_path']
        canopy_success_flag = data['canopy_success_flag']

        geoglyph_prob = clip_results_map.get(tms_path, -1.0)
        canopy_cover = canopy_results_map.get(canopy_path, -1.0) if canopy_success_flag else -1.0


        # Check conditions and add to new_rows
        # Include the check for is_tif_empty(canopy_path) only if canopy_success_flag is True and canopy_cover is valid
        canopy_valid_and_not_empty = (
            canopy_success_flag and
            canopy_cover != -1.0 and
            0.05 < canopy_cover <= 0.2
        )

        if geoglyph_prob > 0.8 and canopy_valid_and_not_empty:
             # Double check is_tif_empty right before adding
             if not is_tif_empty(canopy_path):
                cell = grid_cells[idx]
                grid_id = f"{cell['minx']},{cell['miny']},{cell['maxx']},{cell['maxy']}"
                new_rows.append({
                    'grid_id': grid_id,
                    'geoglyph_prob': geoglyph_prob,
                    'canopy_cover': canopy_cover
                })
             else:
                 # print(f"Skipping grid {idx}: Canopy file empty after check.")
                 pass # Do not add this row

        # Cleanup files for the current item
        if os.path.exists(tms_path):
            os.remove(tms_path)
        if os.path.exists(canopy_path):
             os.remove(canopy_path)


    # Update results DataFrame and save
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        results_df = pd.concat([results_df, new_df], ignore_index=True)
        try:
            # Use 'a' mode to append if file exists, 'w' mode if not (pandas handles this)
            # For robustness, it's better to write the full dataframe each time in case of crashes
            results_df.to_csv(output_csv_path, index=False)
            # print(f"Saved {len(new_rows)} results for batch {batch_start}-{batch_end-1}")
        except Exception as e:
            print(f"CSV save error for batch {batch_start}-{batch_end-1}: {e}")


print("Processing complete.")



# Remove useless tif downloads if needed
# !rm *.tif


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
csv_path = '/kaggle/input/amazon-geoglyphs/merged_data_candidate_grids.csv'
try:
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded data from {csv_path}")
except FileNotFoundError:
    print(f"Error: File not found at {csv_path}")
    df = pd.DataFrame() # Create an empty DataFrame
except Exception as e:
    print(f"Error reading CSV file: {e}")
    df = pd.DataFrame() # Create an empty DataFrame

# Check if the DataFrame is not empty and has the necessary columns
if not df.empty and 'grid_id' in df.columns:
    # Extract centroid coordinates
    centroids = []
    for grid_id in df['grid_id']:
        try:
            # Split the grid_id string into minx, miny, maxx, maxy
            minx, miny, maxx, maxy = map(float, grid_id.split(','))
            # Calculate the center of the bounding box
            center_lon = (minx + maxx) / 2
            center_lat = (miny + maxy) / 2
            centroids.append((center_lon, center_lat))
        except ValueError:
            print(f"Skipping invalid grid_id format: {grid_id}")
            continue # Skip this row if format is invalid

    # Separate latitudes and longitudes
    if centroids:
        lons, lats = zip(*centroids)

        # Create a simple scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(lons, lats, s=5, alpha=0.5) # s is marker size, alpha is transparency
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Centroids of Grid Cells')
        plt.grid(True)
        plt.show()
    else:
        print("No valid grid centroids found to plot.")
else:
    print("DataFrame is empty or 'grid_id' column is missing. Cannot plot.")


!pip -q install --upgrade langchain langchain-openai tiktoken openai tqdm 


import os, json, glob, base64, time, mimetypes, pathlib
import pandas as pd
from IPython.display import display, HTML
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import ee
import geemap
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime # For potential client-side date parsing if needed

from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64


data_path = "/kaggle/input/amazon-geoglyphs/merged_data_candidate_grids_part1.csv"
all_files = glob.glob(data_path)
print(all_files)
all_df = []

for f in all_files:
    df = pd.read_csv(f)
    all_df.append(df)

combined_df = pd.concat(all_df, ignore_index=True)
df = combined_df.copy()


!rm -rf /kaggle/working/images/*.tif


#image download
def download_tms_geotiff(bbox, output_path):
    """Download TMS image with timeout handling"""
    try:
        geemap.tms_to_geotiff(
            output=output_path,
            bbox=bbox,
            zoom=17,
            source='Satellite',
            quiet=True
        )
        return True
    except Exception as e:
        print(f"Error downloading TMS: {e}")
        return False

from tqdm import tqdm
import os
image_path = "/kaggle/working/images/"
os.makedirs(image_path, exist_ok=True)

bbox_filename_mapping = []  # List to store the mapping
idx_start=5000
idx_end=10000
count_rows=idx_end-idx_start

for index, row in tqdm(df.iterrows()):
    bbox_m = row['grid_id'] # Get the bounding box from the DataFrame row
    bbox = [float(num) for num in bbox_m.split(',')]

    counter = index + 1 # Start counter from 1
    output_filename = os.path.join(image_path, f"image_{counter}.tif")
    # print(f"Downloading image for bbox: {bbox} to {output_filename}")
    success = download_tms_geotiff(bbox, output_filename)

    if success:
        bbox_filename_mapping.append({'bbox': bbox_m, 'filename': output_filename})
    else:
        print(f"Skipping mapping for {bbox} due to download error.")
    if counter == count_rows:
      break

import csv
output_mapping_file = f'/kaggle/working/bbox_filename_mapping_{idx_start}_{idx_end}.csv'
with open(output_mapping_file, 'w', newline='') as csvfile:
    fieldnames = ['bbox', 'filename']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for mapping in bbox_filename_mapping:
        writer.writerow(mapping)

print(f"Bounding box to filename mapping saved to: {output_mapping_file}")


# image function
def encode_image_b64(image_path):
    # Open and convert image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size

    # Resize to 512x512 (OpenAI-recommended size)
    img = img.resize((1024, 1024))

    # Convert to PNG in-memory
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    
    size_kb = buffer.tell() / 1024

    # print(f"âœ“ {os.path.basename(image_path)} â€” original: {original_size}, resized: (1024, 1024), PNG size: {size_kb:.1f} KB")

    # Base64 encode PNG
    return base64.b64encode(buffer.read()).decode("utf-8")


import getpass
import os

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")


output_folder = "/kaggle/working/qa-results/"
os.makedirs(output_folder, exist_ok=True)


MODEL_NAME_FT = "ft:gpt-4.1-2025-04-14:chandrabhraman:geoglyphs101:Bi0lIPZN" # Replace with your actual fine-tuned model ID
chat_ft = ChatOpenAI(model_name = MODEL_NAME_FT)

image_folder_path = "/kaggle/working/images" # Folder containing the images
image_files = glob.glob(os.path.join(image_folder_path, "*.tif")) # Assuming images are TIFF files

results_list = [] # Initialize an empty list to store results

if not image_files:
  print(f"No image files found in: {image_folder_path}")
else:
  for image_to_process_path in tqdm(image_files, desc="Processing Images"):
    if not os.path.exists(image_to_process_path):
      print(f"Image not found at: {image_to_process_path}. Skipping.")
      continue

    image_b64_inference = encode_image_b64(image_to_process_path)

    messages_inference = [
        HumanMessage(content=[
            {"type": "text", "text": "Describe the geospatial properties of this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64_inference}"}}
        ])
    ]

    raw_json_ft = "" # Initialize raw_json_ft for each image
    try:
        response_ft = chat_ft.invoke(messages_inference)
        raw_json_ft = response_ft.content.strip()
    except Exception as e:
        print(f"Error processing {os.path.basename(image_to_process_path)}: {e}")
        continue

    # Append the filename and raw output to the results list
    results_list.append({'filename': os.path.basename(image_to_process_path), 'raw_output': raw_json_ft})

# Save the results to a CSV file

output_csv_file = f"/kaggle/working/qa-results/image_model_output_{idx_start}_{idx_end}.csv"
results_df = pd.DataFrame(results_list)
results_df.to_csv(output_csv_file, index=False)
    
print(f"Image filenames and raw model outputs saved to: {output_csv_file}")


idx_start=5000
idx_end=10000
output_csv_file = f"/kaggle/working/qa-results/image_model_output_{idx_start}_{idx_end}.csv"
results_df = pd.DataFrame(results_list)
results_df.to_csv(output_csv_file, index=False)
    
print(f"Image filenames and raw model outputs saved to: {output_csv_file}")


output_mapping_file='/kaggle/working/bbox_filename_mapping_5000_10000.csv'
output_csv_file='/kaggle/working/qa-results/image_model_output_5000_10000.csv'

geoglyph_files_zip = "/kaggle/working/geoglyph_images.zip"
import zipfile
import pandas as pd

# Load the output_mapping_file and output_csv_file into pandas DataFrames
try:
    mapping_df = pd.read_csv(output_mapping_file)
    output_df = pd.read_csv(output_csv_file)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    # Handle the error, e.g., exit or return
    exit() # or return []

# Initialize an empty list to store the 'bbox' values for geoglyphs
geoglyph_bboxes = []
geoglyph_image_paths=[]
# Iterate through the rows of the output_csv_file DataFrame
for index, row in output_df.iterrows():
    raw_output = row['raw_output']
    filename = '/kaggle/working/images/'+row['filename']

    # Check if the word 'geoglyph' is in the 'raw_output'
    if 'geoglyph' in str(raw_output).lower():
        # Find the corresponding 'bbox' from the mapping_df using the filename
        matching_row = mapping_df[mapping_df['filename'] == filename]
        geoglyph_image_paths.append(filename)

        if not matching_row.empty:
            # Add the 'bbox' to the geoglyph_bboxes list
            geoglyph_bboxes.append(matching_row.iloc[0]['bbox'])
        else:
            print(f"Warning: Filename '{filename}' not found in {output_mapping_file}")

# Create a new DataFrame with the collected 'bbox' values
geoglyph_bbox_df = pd.DataFrame({'geoglyph_bbox': geoglyph_bboxes})

# Define the path for the new CSV file
publish_csv_file = "/kaggle/working/qa-results/geoglyph_bboxes_for_publish.csv"

# Save the new DataFrame to a CSV file
geoglyph_bbox_df.to_csv(publish_csv_file, index=False)

# Zip the geoglyph image files
with zipfile.ZipFile(geoglyph_files_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for image_path in geoglyph_image_paths:
        if os.path.exists(image_path):
            zipf.write(image_path, os.path.basename(image_path))
        else:
            print(f"Warning: Image file not found for zipping: {image_path}")

print(f"Geoglyph image files zipped to: {geoglyph_files_zip}")

print(f"Geoglyph bounding boxes saved to: {publish_csv_file}")
print(f"Number of geoglyph bounding boxes found: {len(geoglyph_bboxes)}")


import pandas as pd
import os

idx_start=5000
idx_end=10000
output_csv_path = "/kaggle/working/qa-results/geoglyph_bboxes_for_publish.csv"

# Read the input CSV file
try:
    results_df = pd.read_csv(output_csv_path)
    print(f"Successfully loaded results from {output_csv_path}")
except FileNotFoundError:
    print(f"Error: Input results file not found at {output_csv_path}")
    results_df = pd.DataFrame(columns=['geoglyph_bbox']) # Create an empty DataFrame
except Exception as e:
    print(f"Error reading input results file: {e}")
    results_df = pd.DataFrame(columns=['geoglyph_bbox']) # Create an empty DataFrame

maps_df=results_df
# Create a list to store the Google Maps URLs
maps_urls = []

# Extract the centers of the bboxes from the maps_df
pin_coordinates = []
for index, row in maps_df.iterrows():
    grid_id = row['geoglyph_bbox']
    # print(grid_id)
    try:
        minx, miny, maxx, maxy = map(float, grid_id.split(','))
        # print(map(float, grid_id.split(',')))        
        center_lon = (minx + maxx) / 2
        center_lat = (miny + maxy) / 2
        pin_coordinates.append((center_lat, center_lon))
        # print(pin_coordinates)
        # exit()
    except ValueError:
        print(f"Skipping invalid grid_id format: {grid_id}")
        continue

import folium

# Create a base Folium map centered roughly on the area of interest (adjust as needed)
# You can use the average of your pin coordinates or a known central point
if pin_coordinates:
    avg_lat = sum([p[0] for p in pin_coordinates]) / len(pin_coordinates)
    avg_lon = sum([p[1] for p in pin_coordinates]) / len(pin_coordinates)
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12, tiles='OpenStreetMap') # Start with OpenStreetMap
else:
    # Default center if no pins
    m = folium.Map(location=[-5.0, -70.0], zoom_start=6, tiles='OpenStreetMap')

google_satellite_url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
folium.TileLayer(
    tiles=google_satellite_url,
    attr='Google Satellite',
    name='Google Satellite',
    overlay=True,
    control=True
).add_to(m)


# Option 2: Using Esri World Imagery (a good alternative if Google Satellite tiles are an issue)
folium.TileLayer(
    tiles='Esri.WorldImagery',
    attr='Esri World Imagery',
    name='Esri World Imagery',
    overlay=True,
    control=True
).add_to(m)


# Add the pins to the map
# Add a LayerControl so users can switch tile layers
folium.LayerControl().add_to(m)

# Add the pins with popups (optional, you can customize popups)
for i, (lat, lon) in enumerate(pin_coordinates):
    folium.Marker(
        location=[lat, lon],
        popup=f'Pin {i+1}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}'
    ).add_to(m)

# Save the map to an HTML file
output_html_path = f'/kaggle/working/qa-results/geoglyph_canopy_pins_map_{idx_start}_{idx_end}.html'
m.save(output_html_path)

print(f"Folium map with pins and satellite layers saved to {output_html_path}")

