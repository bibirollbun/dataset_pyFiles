%%capture
!pip install -q geopandas rasterio folium opencv-python-headless scikit-image openai google-api-python-client earthengine-api tqdm


# Core Python libraries
import os
import json
import time
import base64
import warnings
from pathlib import Path
from tqdm.notebook import tqdm

# Geospatial and Data Handling
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
import ee # Google Earth Engine
import folium
from folium.plugins import MarkerCluster

# Image Processing and Visualization
import cv2
import matplotlib.pyplot as plt
import folium # For interactive maps

# APIs and Integration
import openai

# Environment setup
IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
warnings.filterwarnings('ignore')

print(f"Setup Complete. Running in {'Kaggle' if IS_KAGGLE else 'Local'} environment.")


# --- 1. Load Secrets ---
if IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    try:
        OPENAI_API_KEY = user_secrets.get_secret("OPENAI_API_KEY")
        print("Kaggle secrets loaded successfully.")
    except Exception as e:
        print(f"Could not load Kaggle secrets: {e}. Please ensure OPENAI_API_KEY is set.")
        OPENAI_API_KEY = "DUMMY_KEY"
else:
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai.api_key = OPENAI_API_KEY
    print("Local .env secrets loaded.")

# --- 2. Define Core Configuration ---
CONFIG = {
    'USE_MOCK_DATA': True,
    
    # Area of Interest: A region in Acre, Brazil known for geoglyphs
    'AOI_BOUNDS': {'west':-70.8, 'south':-10.5, 'east':-69.8, 'north':-9.5}, 
    
    # Tiling parameters
    'TILE_SIZE_PX': 512, 
    'TILE_OVERLAP_PX': 64,
    
    # GEE parameters
    'DATE_RANGE': ('2023-06-01', '2023-09-30'),
    'CLOUDY_PIXEL_PERCENTAGE': 10,

    # CV filter threshold
    'GEOMETRY_SCORE_THRESHOLD': 5,
    'VISUALIZATION_SAMPLES': 3,
    'LLM_ANALYSIS_COUNT': 5,

    # Paths
    'BASE_DIR': Path('./amazonia_ai'),
    'SOURCE_DATA_DIR': Path('./amazonia_ai/01_source_data'),
    'TILES_DIR': Path('./amazonia_ai/02_tiles'),
    'RESULTS_DIR': Path('./amazonia_ai/03_results'),
    'CACHE_DIR': Path('./amazonia_ai/cache')
}

for p in ['BASE_DIR', 'SOURCE_DATA_DIR', 'TILES_DIR', 'RESULTS_DIR', 'CACHE_DIR']:
    CONFIG[p].mkdir(exist_ok=True)
print("Project directory structure verified.")


# === MOCK DATA GENERATION ===
def create_mock_raster(config):
    """Creates a mock raster file with more 'natural' simulated anomalies."""
    print("--- Creating MOCK Data File ---")
    mock_path = config['SOURCE_DATA_DIR'] / 'MOCK_DATA.tif'
    shape = (2048, 2048)

    background = np.random.normal(80, 20, (shape[0], shape[1])).astype(np.uint8)
    background = cv2.GaussianBlur(background, (9, 9), 0)

    anomalies_layer = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(anomalies_layer, (200, 200), (500, 500), 200, -1)
    # Круг
    cv2.circle(anomalies_layer, (1300, 1300), 150, 180, -1)
    cv2.line(anomalies_layer, (800, 100), (800, 600), 190, 8)
    cv2.line(anomalies_layer, (700, 400), (950, 400), 190, 8)
    
    anomalies_layer = cv2.GaussianBlur(anomalies_layer, (11, 11), 0)
    
    final_raster = cv2.addWeighted(background, 0.7, anomalies_layer, 0.9, 0)
    
    aoi_bounds = config['AOI_BOUNDS']
    transform = from_origin(aoi_bounds['west'], aoi_bounds['north'], 0.0001, 0.0001)
    profile = {
        'driver': 'GTiff', 'count': 5, 'dtype': 'uint8',
        'width': shape[1], 'height': shape[0],
        'crs': 'EPSG:4326', 'transform': transform
    }
    with rasterio.open(mock_path, 'w', **profile) as dst:
        for i in range(1, 6):
            dst.write(final_raster, i)
            
    print(f"Mock data saved to {mock_path}")
    return mock_path

# === TILING LOGIC ===
def process_and_tile_data(source_path, config):
    """Tiles a source raster into smaller overlapping chips for analysis."""
    print(f"--- Tiling data from {source_path} ---")
    metadata, tile_coords = [], []
    tile_size = config['TILE_SIZE_PX']
    step = config['TILE_SIZE_PX'] - config['TILE_OVERLAP_PX']
    
    with rasterio.open(source_path) as src:
        for y in tqdm(range(0, src.height - tile_size, step), desc="Tiling Progress"):
            for x in range(0, src.width - tile_size, step):
                window = Window(x, y, tile_size, tile_size)
                tile_dir = config['TILES_DIR'] / f"tile_{x}_{y}"
                tile_dir.mkdir(exist_ok=True)
                
                rgb_path = tile_dir / "rgb.png"
                ndvi_path = tile_dir / "ndvi.tif"
                slope_path = tile_dir / "slope.tif"
                
                bands = src.read(window=window)
                
                rgb_bands = bands[:3]
                v_min, v_max = np.percentile(rgb_bands, [2, 98])
                rgb_stretched = np.clip((rgb_bands - v_min) * 255.0 / (v_max - v_min), 0, 255).astype(np.uint8)
                rgb_img = np.dstack(rgb_stretched)
                cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

                base_profile = src.profile.copy()
                base_profile.update(width=tile_size, height=tile_size, transform=src.window_transform(window), count=1, dtype='uint8')
                
                with rasterio.open(ndvi_path, 'w', **base_profile) as dst: dst.write(bands[3], 1)
                with rasterio.open(slope_path, 'w', **base_profile) as dst: dst.write(bands[4], 1)
                
                coords = src.xy(y + tile_size // 2, x + tile_size // 2)
                metadata.append({
                    'tile_id': f"tile_{x}_{y}", 'rgb_path': str(rgb_path), 'ndvi_path': str(ndvi_path), 'slope_path': str(slope_path),
                    'lon': coords[0], 'lat': coords[1]
                })
    
    df = pd.DataFrame(metadata)
    print(f"Tiling complete. {len(df)} tiles created.")
    return df


# === DATA PIPELINE EXECUTION ===
df_tiles = pd.DataFrame()

if CONFIG['USE_MOCK_DATA']:
    print("INFO: Using MOCK data pipeline.")
    mock_file_path = create_mock_raster(CONFIG)
    df_tiles = process_and_tile_data(mock_file_path, CONFIG)
else:
    print("INFO: REAL data pipeline would run here.")
    print("This requires GEE authentication and Google Drive setup.")
    print("To run, implement the GEE download logic and set USE_MOCK_DATA to False.")

if not df_tiles.empty:
    print("\n--- Sample Generated Tiles ---")
    sample_rows = df_tiles.sample(min(CONFIG['VISUALIZATION_SAMPLES'], len(df_tiles)))
    for _, row in sample_rows.iterrows():
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        img = cv2.imread(row['rgb_path'])
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Tile: {row['tile_id']}")
        ax.axis('off')
        plt.show()


# === CV FILTERING FUNCTIONS ===
def detect_geometric_shapes(image_path):
    """Applies Canny edge detection and Hough line transform to find straight lines."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None: return 0, None, None
        
        edges = cv2.Canny(img, threshold1=30, threshold2=150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=30, maxLineGap=10)
        
        viz_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        num_lines = 0
        if lines is not None:
            num_lines = len(lines)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        return num_lines, edges, viz_img
    except Exception as e:
        print(f"CV processing error for {image_path}: {e}")
        return 0, None, None


# === CV ANALYSIS EXECUTION ===
df_analysis = pd.DataFrame()
df_candidates = pd.DataFrame()

if not df_tiles.empty:
    print("--- Filtering All Tiles with Classical CV ---")
    analysis_results = []
    for _, row in tqdm(df_tiles.iterrows(), total=len(df_tiles), desc="Analyzing Tiles"):
        geometry_score, viz_canny, viz_geometry = detect_geometric_shapes(row['rgb_path'])
        
        result_row = row.to_dict()
        result_row.update({
            'geometry_score': geometry_score,
            'is_candidate': geometry_score >= CONFIG['GEOMETRY_SCORE_THRESHOLD']
        })
        analysis_results.append(result_row)
        
    df_analysis = pd.DataFrame(analysis_results)
    df_candidates = df_analysis[df_analysis['is_candidate']].copy()
    
    print(f"\nFiltering complete. Found {len(df_candidates)} high-potential candidates out of {len(df_analysis)} total tiles.")
else:
    print("Skipping CV filtering as no tiles were loaded.")


# === VISUALIZATION OF CV FILTERING RESULTS ===
def visualize_cv_steps(df_analysis, num_to_show=2):
    """Displays a detailed, stylish comparison of tiles that passed and failed the CV filter."""
    plt.style.use('dark_background')
    for state, color, df_subset in [('CANDIDATE (PASSED)', '#4CAF50', df_analysis[df_analysis['is_candidate']]), 
                                ('REJECTED', '#f44336', df_analysis[~df_analysis['is_candidate']])]:
        if not df_subset.empty:
            print(f"\n--- Showing {min(num_to_show, len(df_subset))} examples for: {state} ---")
            for _, row in df_subset.head(num_to_show).iterrows():
                _, viz_canny, viz_geometry = detect_geometric_shapes(row['rgb_path'])
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                fig.patch.set_facecolor('#1a1a1a')
                
                axes[0].imshow(cv2.cvtColor(cv2.imread(row['rgb_path']), cv2.COLOR_BGR2RGB))
                axes[0].set_title("Original RGB Tile")
                axes[1].imshow(viz_canny, cmap='hot')
                axes[1].set_title("Step 1: Canny Edges")
                axes[2].imshow(cv2.cvtColor(viz_geometry, cv2.COLOR_BGR2RGB))
                axes[2].set_title(f"Step 2: Hough Lines (Score: {row['geometry_score']})")
                
                fig.suptitle(f"CV Analysis of Tile: {row['tile_id']} -> {state}", fontsize=16, weight='bold', color=color)
                for ax in axes: ax.axis('off')
                plt.tight_layout(rect=[0, 0, 1, 0.94])
                plt.show()

if not df_analysis.empty:
    candidates_path = CONFIG['RESULTS_DIR'] / 'cv_candidates.csv'
    df_candidates.to_csv(candidates_path, index=False)
    print(f"Candidate list saved to: {candidates_path}")
    visualize_cv_steps(df_analysis, num_to_show=CONFIG['VISUALIZATION_SAMPLES'])
else:
    print("No CV analysis results to visualize.")


# === LLM ANALYSIS FUNCTIONS WITH CACHING ===
def create_composite_image_for_llm(row):
    """Creates a composite image (RGB, NDVI, Slope) for the LLM."""
    rgb_img = cv2.imread(row['rgb_path'])
    
    with rasterio.open(row['ndvi_path']) as src: ndvi_raw = src.read(1)
    with rasterio.open(row['slope_path']) as src: slope_raw = src.read(1)

    ndvi_color = (plt.cm.viridis(cv2.normalize(ndvi_raw, None, 0, 255, cv2.NORM_MINMAX))[:, :, :3] * 255).astype(np.uint8)
    slope_color = (plt.cm.magma(cv2.normalize(slope_raw, None, 0, 255, cv2.NORM_MINMAX))[:, :, :3] * 255).astype(np.uint8)

    composite = np.hstack([cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), ndvi_color, slope_color])

    cv2.putText(composite, 'RGB', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(composite, 'NDVI', (522, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(composite, 'Slope', (1034, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return composite

def encode_image_to_base64(image_np):
    """Encodes a numpy image array into a base64 string."""
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpeg', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')


def analyze_candidate_with_gpt4o(row, config):
    """
    Sends a candidate to GPT-4o with retries and caching.
    """
    cache_file = config['CACHE_DIR'] / f"{row['tile_id']}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            composite_img = create_composite_image_for_llm(row)
            base64_image = encode_image_to_base64(composite_img)

            expert_prompt = (
                "You are an expert remote sensing archaeologist specializing in the Amazon Basin. "
                "Analyze the following composite image which contains three panels: 1. True-color RGB, 2. NDVI (vegetation index), 3. Slope (topography). "
                "Your task is to identify potential anthropogenic features such as geoglyphs, earthworks, or ancient settlements. "
                "Look for unnatural geometric patterns (straight lines, right angles, circles), unusual vegetation patterns, or subtle earthworks. "
                "Respond ONLY with a valid JSON object: {\"contains_anthropogenic_features\": boolean, \"confidence_score\": float (0.0-1.0), "
                "\"rationale\": string, \"feature_type_guess\": string}."
            )

            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": expert_prompt},
                         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                     ]}
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("API returned None content")

            result = json.loads(content)
            
            cache_file.write_text(json.dumps(result))
            return result
        
        except (ValueError, json.JSONDecodeError, openai.APIError) as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for tile {row['tile_id']}: {e}. Retrying in {retry_delay}s...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"All retries failed for tile {row['tile_id']}.")
                return {"rationale": f"API call failed after {max_retries} attempts", "confidence_score": 0.0, "contains_anthropogenic_features": False, "feature_type_guess": "error"}

    return {"rationale": "API call failed after all retries", "confidence_score": 0.0, "contains_anthropogenic_features": False, "feature_type_guess": "error"}


# === LLM ANALYSIS EXECUTION ===
df_final_results = pd.DataFrame()

if not df_candidates.empty:
    num_to_analyze = min(CONFIG['LLM_ANALYSIS_COUNT'], len(df_candidates))
    print(f"--- Sending {num_to_analyze} High-Potential Candidates to GPT-4o ---")
    
    llm_results = []
    for _, row in tqdm(df_candidates.head(num_to_analyze).iterrows(), total=num_to_analyze, desc="LLM Analysis"):
        analysis = analyze_candidate_with_gpt4o(row, CONFIG)
        llm_results.append({**row.to_dict(), **analysis})
    
    df_final_results = pd.DataFrame(llm_results)
    final_path = CONFIG['RESULTS_DIR'] / 'final_llm_analysis.csv'
    df_final_results.to_csv(final_path, index=False)
    
    print(f"\nLLM analysis complete. Final results saved to {final_path}")
    print("\n--- GPT-4o Analysis Results ---")
    display(df_final_results[['tile_id', 'contains_anthropogenic_features', 'confidence_score', 'feature_type_guess', 'rationale']])
else:
    print("Skipping LLM analysis as no candidate tiles were generated.")


# === INTERACTIVE MAP VISUALIZATION ===
def create_results_map(df, config):
    """
    Generates a Folium map with switchable base layers (Positron and Satellite) 
    and markers for positive LLM results.
    """
    aoi = config['AOI_BOUNDS']
    map_center = [(aoi['south'] + aoi['north']) / 2, (aoi['west'] + aoi['east']) / 2]
    
    m = folium.Map(location=map_center, zoom_start=9, tiles='CartoDB positron', attr='CartoDB positron')

    folium.TileLayer(
        'Esri_WorldImagery',
        attr='Esri | Earthstar Geographics, CNES/Airbus DS, P-VUE, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, swisstopo, and the GIS User Community'
    ).add_to(m)
    
    positive_results = df[df['contains_anthropogenic_features'] == True]
    if positive_results.empty:
        print("No positive features identified by the LLM to display on the map.")
        return m
        
    print(f"Adding {len(positive_results)} positive sites to the map...")

    marker_cluster = MarkerCluster().add_to(m)

    for _, row in positive_results.iterrows():
        composite_img = create_composite_image_for_llm(row)
        encoded = encode_image_to_base64(cv2.resize(composite_img, (0,0), fx=0.4, fy=0.4))
        
        html = f'''
        <h4>Tile: {row['tile_id']}</h4>
        <b>LLM Guess:</b> <span style="color: #4CAF50; font-weight: bold;">{row['feature_type_guess'].upper()}</span><br>
        <b>Confidence:</b> {row['confidence_score']:.2f}<br>
        <b>Rationale:</b> <em>{row['rationale']}</em><br>
        <img src="data:image/jpeg;base64,{encoded}" width="600px" alt="Composite Image">
        '''
        
        iframe = folium.IFrame(html, width=640, height=310)
        popup = folium.Popup(iframe, max_width=640)
        
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=popup,
            tooltip=f"Candidate: {row['tile_id']} (Click to see details)",
            icon=folium.Icon(color='green', icon='search', prefix='fa')
        ).add_to(marker_cluster)
        
    folium.LayerControl().add_to(m)
        
    return m

if not df_final_results.empty:
    results_map = create_results_map(df_final_results, CONFIG)
    map_path = CONFIG['RESULTS_DIR'] / 'interactive_results_map.html'
    results_map.save(map_path)
    print(f"Interactive map saved to {map_path}")
    display(results_map)
else:
    print("No final LLM results to create a map from.")




