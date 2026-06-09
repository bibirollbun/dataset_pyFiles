%pip install -q --upgrade contextily rasterio scikit-learn category-encoders mermaid-py


import mermaid as mmd
flowchart = """ 
flowchart TD

    %% === Node Style Definitions ===

    classDef startEnd fill:#d4edda,stroke:#155724,stroke-width:2px,color:#155724
    classDef stage fill:#e2e3e5,stroke:#383d41,stroke-width:2px
    classDef substage fill:#fff,stroke:#6c757d
    classDef data fill:#d1ecf1,stroke:#0c5460,stroke-width:2px,stroke-dasharray: 5 5,color:#0c5460
    classDef decision fill:#fff3cd,stroke:#856404,stroke-width:2px,color:#856404
    classDef parallel fill:#f8d7da,stroke:#721c24,stroke-width:2px,color:#721c24

    %% === Workflow Definition ===

    %% --- Start ---
    A(Start) --> B(["Input: Test Locations CSV"]);
    class A startEnd;
    class B data;

    %% --- Stage 1 ---
    B --> C["<b>Stage 1: Initialize</b><br>Load Test Data & Pre-trained Models"];
    class C stage;

    %% --- Stage 2 ---
    C --> E["<b>Stage 2: Anomaly Detection Agent</b><br><i>(For each Test Location)</i>"];
    class E stage;

    subgraph E [Details]
        direction TB
        E1["A. Feature Extraction<br>via Google Earth Engine"] --> E2["B. Isolation Forest Analysis"];
        class E1,E2 substage;
    end

    E --> E_out(["Output: Top ROI Candidates"]);
    class E_out data;

    %% --- Stage 3 ---
    E_out --> F["<b>Stage 3: Async AI Analysis & Decision Pipeline</b><br><i>(For each ROI Candidate)</i>"];
    class F parallel;

    subgraph F [Details]
        direction TB
        F1["Parallel Image Acquisition<br>(DEM & Sentinel-2)"] --> F2["Parallel AI(LLM) Analysis"];
        
        subgraph F2 [ ]
            direction TB
            F2_dem["DEM Analysis<br><i>(GPT-4.1 â†’ GPT-4o)</i>"]
            F2_s2["Sentinel-2 Analysis<br><i>(GPT-4.1 â†’ GPT-4o)</i>"]
            F2_web["Web Search for Context<br><i>(GPT-4.1 + Search Tool)</i>"]
        end

        F2 --> F3{"Decision Agent<br><i>(o4-mini)</i>"};
        class F1,F2,F2_dem,F2_s2,F2_web substage;
        class F3 decision;
    end

    F --> F_out(["Output: Classification & Confidence"]);
    class F_out data;

    %% --- Stage 4 & 5 ---
    F_out --> G["<b>Stage 4 & 5: Aggregate Results & Visualize</b>"];
    class G stage;
    
    G --> H(["Final Report, Maps & Found Sites JSON"]);
    class H data;

    %% --- End ---
    H --> I(End);
    class I startEnd;
"""
mmd.Mermaid(flowchart,position=mmd.Position.CENTER)


# Core imports
import pandas as pd
import numpy as np
import ee
import time
import os
import joblib
import urllib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from dataclasses import dataclass
import json
import warnings
import textwrap
from IPython.display import display, HTML
from openai import OpenAI, AsyncOpenAI
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from scipy.spatial.distance import cdist
from plotly.subplots import make_subplots
import textwrap
warnings.filterwarnings('ignore')

# For async in Jupyter
import nest_asyncio
nest_asyncio.apply()


# Initialize OpenAI clients

OT_API_KEY = os.getenv("OT_api_key", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

if OT_API_KEY and OPENAI_API_KEY:
	client = OpenAI(api_key=OPENAI_API_KEY)
	async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
else:
	from kaggle_secrets import UserSecretsClient
	OPENAI_API_KEY = UserSecretsClient().get_secret("OPENAI_API_KEY")
	client = OpenAI(api_key=OPENAI_API_KEY)
	async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
	OT_API_KEY = UserSecretsClient().get_secret("OT_API_KEY")

# Initialize Earth Engine
def load_secret(name):
	"""Loads secret from Colab/Kaggle."""
	if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
		try:
			from kaggle_secrets import UserSecretsClient
			return UserSecretsClient().get_secret(name)
		except Exception:
			raise
	else:
		try:
			return os.getenv(name)
		except Exception:
			pass
	return "Secret not found"

iam_service_account = load_secret(
	"iam_service_account"
)	# the address of your project's IAM service account
ee_credentials_json = load_secret(
	"ee_credentials"
)	# the file path for the JSON file containing the relevant credentials
ee_creds = ee.ServiceAccountCredentials(
	iam_service_account, ee_credentials_json
)	# fetch your service account credentials
ee.Initialize(
	ee_creds
)	# initialize earth engine using your service account credentials


# Configuration parameters
BUFFER_KM = 1.0	 # Buffer around known sites
GRID_SIZE_M = 300	 # Smaller grid size for testing (was 500)
ZONE_SIZE_KM = 3.0	 # Smaller search area for testing (was 5.0)
TARGET_SCALE_METERS = 250
BATCH_SIZE = 1000
MODELS_DIR = "/kaggle/input/openaitoz_isolation_forest/scikitlearn/default/1"	 # Pre-trained models directory
DATA_DIR = "/kaggle/input/dataset"	 # Data directory
OUTPUT_DIR = "/kaggle/working/"
BUFFER_RADIUS_M_S2 = 5000	 # 5km radius for S2
BUFFER_RADIUS_M_DEM = 50000	 # 50km radius for DEM

print("ğŸŒ� Multi-Agent Archaeological Discovery System")
print("=" * 65)


# Load test data - first 10 rows
test_data_path = os.path.join(DATA_DIR, "testset.csv")
test_df = pd.read_csv(test_data_path)


## Get hte name of the site back to test_df
full_df = pd.read_csv('/kaggle/input/dataset/amazon_geoglyphs_sites_cleaned.csv').drop_duplicates()
test_df	 = test_df.merge(full_df,on=["latitude","longitude"])


import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
train_df = pd.read_csv(os.path.join(DATA_DIR, "trainset.csv"))
# Create GeoDataFrames
train_gdf = gpd.GeoDataFrame(
	 train_df, geometry=gpd.points_from_xy(train_df.longitude, train_df.latitude), crs="EPSG:4326"
)
test_gdf = gpd.GeoDataFrame(
	 test_df, geometry=gpd.points_from_xy(test_df.longitude, test_df.latitude), crs="EPSG:4326"
)

# --- Plotting ---
# Create a figure and axes for your plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the training data
train_gdf.plot(
	ax=ax,
	color='blue',
	alpha=0.7,
	markersize=50,	# CORRECTED: Use 'markersize' instead of 's'
	label='Training Data'
)

# Plot the testing data on the same axes
test_gdf.plot(
	ax=ax,
	color='red',
	alpha=0.7,
	markersize=50,	# CORRECTED: Use 'markersize' instead of 's'
	label='Test Data'
)

# --- Add the Basemap ---
ctx.add_basemap(ax, crs=train_gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)

# --- Final Touches ---
ax.legend()
plt.title("Train and Test Data on Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Display the plot
plt.show()


print(f"ğŸ“� Loaded {len(test_df)} test locations")
print("\nTest locations:")

# Load known sites (for reference, not filtering)
known_sites = pd.read_csv(os.path.join(DATA_DIR, "amazon_geoglyphs_sites_cleaned.csv"))
print(f"\nâœ… Loaded {len(known_sites)} known archaeological sites (for reference)")


import sys
sys.path.insert(1, '/kaggle/input/openaitoz-prompts')
from prompts import (
				DEM_1stlook_prompt, DEM_2ndlook_prompt,
				S2_1stlook_prompt, S2_2ndlook_prompt,
				SEARCH_PROMPT, DECISION_PROMPT
	)
from get_images import download_and_process_dem, get_least_cloudy_s2_image, numpy_image_to_base64


# Load pre-trained models
def load_trained_model(models_dir: str) -> Dict[str, Any]:
	"""
	Load pre-trained isolation forest and associated components.
	"""
	paths = {
		"model": os.path.join(models_dir, "iforest_dbs_model.pkl"),
		"scaler": os.path.join(models_dir, "iforest_dbs_scaler.pkl"),
		"features": os.path.join(models_dir, "iforest_dbs_features.pkl"),
		"clustering": os.path.join(models_dir, "dbs_clustering_model.pkl")
	}
	
	missing_files = []
	for component, path in paths.items():
		if not os.path.exists(path):
			missing_files.append(f"{component}: {path}")
	
	if missing_files:
		print(f"â�Œ Missing model files:")
		for missing in missing_files:
			print(f"	- {missing}")
		return None
	
	try:
		model = joblib.load(paths["model"])
		scaler = joblib.load(paths["scaler"])
		features = joblib.load(paths["features"])
		dbscan_model = joblib.load(paths["clustering"])
		
		print(f"âœ… Successfully loaded trained models")
		print(f"	- Model contamination: {model.contamination}")
		print(f"	- Feature count: {len(features)}")
		
		return {
			"model": model,
			"scaler": scaler,
			"features": features,
			"dbscan": dbscan_model
		}
	except Exception as e:
		print(f"â�Œ Error loading models: {e}")
		return None

# Load the models
model_components = load_trained_model(MODELS_DIR)
if model_components is None:
	print("âš ï¸� Continuing without pre-trained models")


from ml_train import train, AnomalyDetectionConfig
kaggle_config = AnomalyDetectionConfig()
kaggle_config.output_csv_filename = "/kaggle/input/dataset/extracted_covariates_for_sites.csv"
class AnomalyDetectionConfig:
    """Configuration class for anomaly detection parameters"""
    def __init__(self):
        self.output_csv_filename = "data/extracted_covariates_for_sites.csv"
        self.manual_eps = 0.5
        self.min_samples = 5
        self.contamination_levels = [0.01, 0.05, 0.1, 0.15, 0.2]
        self.best_contamination = 0.05
        self.random_state = 42
        self.n_estimators = 100
        self.models_dir = "/kaggle/working/models"
        
        # Model paths
        self.model_path = "/kaggle/working/models/iforest_dbs_model.pkl"
        self.scaler_path = "/kaggle/working/models/iforest_dbs_scaler.pkl"
        self.feature_names_path = "/kaggle/working/models/iforest_dbs_features.pkl"
        self.clustering_model_path = "/kaggle/working/models/dbs_clustering_model.pkl"

isolation_forest_training = train(config = kaggle_config)


def get_covariates_image():
	"""Load key covariate datasets from Google Earth Engine."""
	# print("Assembling covariate image layers from Google Earth Engine...")
	
	try:
		# 1. Topography (Elevation & Slope) from SRTM
		elevation = ee.Image("CGIAR/SRTM90_V4").rename("elevation")
		slope = ee.Terrain.slope(elevation).rename("slope")
		
		# 2. Climate from WorldClim
		worldclim = ee.Image("WORLDCLIM/V1/BIO")
		climate = worldclim.select(
			["bio01", "bio04", "bio12", "bio15"],
			["mean_temp", "temp_seasonality", "annual_precip", "precip_seasonality"],
		)
		
		# 3. Soil pH
		soil_ph = (
			ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02")
			.select("b0")
			.rename("soil_ph")
		)
		soil_clay = (
			ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02")
			.select("b0")
			.rename("soil_clay")
		)
		# Remove problematic CEC layer for now - using organic carbon instead
		soil_oc = (
			ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02")
			.select("b0")
			.rename("soil_organic_carbon")
		)
		# 4. Vegetation (NDVI)
		ndvi = (
			ee.ImageCollection("MODIS/061/MOD13Q1")
			.filterDate("2020-01-01", "2020-12-31")
			.select("NDVI")
			.max()
			.multiply(0.0001)
			.rename("max_ndvi")
		)
		
		# 5. Water occurrence
		# water_occurrence = (
		#	 ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
		#	 .select("occurrence")
		#	 .rename("water_occurrence")
		# )
		
		# Combine layers
		covariates_image = ee.Image.cat([
			elevation, slope, climate, soil_ph,soil_clay,soil_oc, ndvi 
		])
		
		# print(f"âœ… Assembled {covariates_image.bandNames().size().getInfo()} covariate bands")
		return covariates_image
		
	except Exception as e:
		print(f"â�Œ Error assembling covariate image: {e}")
		raise

def extract_features_for_points(points: List[Tuple[float, float]], 
							scale_meters: int = TARGET_SCALE_METERS) -> pd.DataFrame:
	"""
	Extract geographic features for a batch of points.
	"""
	if not points:
		return pd.DataFrame()
	
	# print(f"\nExtracting features for {len(points)} points...")
	
	# Process in batches to avoid Earth Engine limits
	all_results = []
	
	for i in range(0, len(points), BATCH_SIZE):
		batch_points = points[i:i+BATCH_SIZE]
		# print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(points)-1)//BATCH_SIZE + 1}...")
		
		# Create Earth Engine features
		points_features = []
		for j, (lat, lon) in enumerate(batch_points):
			point = ee.Feature(
				ee.Geometry.Point([lon, lat]), 
				{"point_id": i + j, "latitude": lat, "longitude": lon}
			)
			points_features.append(point)
		
		points_fc = ee.FeatureCollection(points_features)
		
		try:
			covariates = get_covariates_image()
			
			# Extract features using reduceRegions with buffer
			extracted_features = covariates.reduceRegions(
				collection=points_fc.map(lambda f: f.buffer(scale_meters / 2)),
				reducer=ee.Reducer.mean(),
				scale=scale_meters,
				crs="EPSG:4326",
			)
			
			# Get results
			extracted_info = extracted_features.getInfo()
			
			if not extracted_info.get("features"):
				print("â�Œ No features extracted from Earth Engine")
				continue
			
			# Process results
			for k, f in enumerate(extracted_info["features"]):
				properties = f["properties"].copy()
				
				# Get coordinates
				if k < len(batch_points):
					lat, lng = batch_points[k]
					properties['latitude'] = lat
					properties['longitude'] = lng
				
				# Clean properties
				cleaned_props = {}
				for key, val in properties.items():
					if key in ['point_id']:
						continue
					elif key in ['latitude', 'longitude']:
						cleaned_props[key] = val
					else:
						# Convert None/null to 0
						if val is None or (isinstance(val, str) and val == 'null'):
							cleaned_props[key] = 0.0
						elif isinstance(val, (int, float)):
							cleaned_props[key] = float(val)
						else:
							cleaned_props[key] = 0.0
				
				all_results.append(cleaned_props)
			
		except Exception as e:
			print(f"â�Œ Feature extraction failed for batch: {e}")
			continue
	
	# Convert to DataFrame
	if all_results:
		df = pd.DataFrame(all_results)
		
		# Fill NaN values
		numeric_columns = [col for col in df.columns if col not in ['latitude', 'longitude']]
		df[numeric_columns] = df[numeric_columns].fillna(0)
		
		# print(f"âœ… Successfully extracted features for {len(df)} points with {len(df.columns)} features")

		return df
	else:
		print("â�Œ No features extracted")
		return pd.DataFrame()


def isolation_forest_agent(coordinates: List[Tuple[float, float]], 
						model_components: Optional[Dict] = None,
						max_rois: int = 15) -> Dict[str, Any]:
	"""
	Apply Isolation Forest to identify anomalous locations.
	
	Args:
		coordinates: List of (lat, lng) tuples
		model_components: Pre-trained model components
		max_rois: Maximum number of ROI candidates to return
	"""
	# print("\nğŸ”· ISOLATION FOREST AGENT")
	# print("-" * 40)
	
	start_time = time.time()
	
	# Validate input
	if not coordinates:
		return {"error": "No coordinates provided"}
	
	# Extract features for all coordinates
	feature_df = extract_features_for_points(coordinates)
	
	if feature_df.empty:
		return {"error": "Failed to extract features"}
	
	extraction_time = time.time() - start_time
	# print(f"Feature extraction completed in {extraction_time:.1f}s")
	
	# Prepare features
	feature_columns = [col for col in feature_df.columns if col not in ['latitude', 'longitude']]
	# print(f"Available features ({len(feature_columns)}): {feature_columns}")
	
	# Handle missing values
	feature_df[feature_columns] = feature_df[feature_columns].fillna(feature_df[feature_columns].median())
	
	# Apply Isolation Forest
	if model_components and 'model' in model_components:
		# Use pre-trained model
		model = model_components['model']
		scaler = model_components['scaler']
		expected_features = model_components['features']
		
		# Ensure feature alignment
		if set(feature_columns) != set(expected_features):
			print(f"âš ï¸� Feature mismatch: expected {expected_features}, got {feature_columns}")
			# Add missing features with default values
			for feat in expected_features:
				if feat not in feature_df.columns:
					feature_df[feat] = 0.0
			# Reorder columns to match training
			feature_df = feature_df[['latitude', 'longitude'] + expected_features]
			feature_columns = expected_features
		
		X_scaled = scaler.transform(feature_df[feature_columns])
	else:
		# Train new model
		print("Training new Isolation Forest model...")
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(feature_df[feature_columns])
		model = IsolationForest(contamination=0.1, random_state=42)
		model.fit(X_scaled)
	
	# Get anomaly scores
	anomaly_scores = model.score_samples(X_scaled)
	predictions = model.predict(X_scaled)
	
	# Add results to dataframe
	feature_df['isolation_score'] = anomaly_scores
	feature_df['is_anomaly'] = predictions == -1
	
	# Calculate percentiles
	feature_df['score_percentile'] = feature_df['isolation_score'].rank(pct=True) * 100
	
	# Select ROI candidates (all anomalous sites) - limit to max_rois
	roi_candidates = feature_df[feature_df['is_anomaly'] == True].copy()
	roi_candidates = roi_candidates.sort_values('isolation_score').head(max_rois)
	roi_candidates['roi_id'] = [f"ROI_{i+1:03d}" for i in range(len(roi_candidates))]
	
	# print(f"\nâœ… Analysis complete:")
	# print(f"	- Total points: {len(feature_df)}")
	# print(f"	- Anomalous points: {sum(feature_df['is_anomaly'])}")
	# print(f"	- ROI candidates: {len(roi_candidates)}")
	
	return {
		"feature_matrix": feature_df,
		"roi_candidates": roi_candidates,
		"isolation_metrics": {
			"total_points": len(feature_df),
			"anomaly_count": sum(feature_df['is_anomaly']),
			"roi_count": len(roi_candidates),
			"anomaly_rate": sum(feature_df['is_anomaly']) / len(feature_df),
			"feature_columns": feature_columns,
			"score_range": [anomaly_scores.min(), anomaly_scores.max()]
		}
	}


# Async Look-Twice analysis function
async def LookTwice(firstprompt: str, secondprompt: str, base64_image: str):
	"""
	Perform two-stage analysis on an image using OpenAI's response API.
	"""
	# First look
	first_response = await async_client.responses.create(
		model="gpt-4.1",
		input=[
			{
				"role": "system",
				"content": [
					{
						"type": "input_text",
						"text": firstprompt
					}
				]
			},
			{
				"role": "user",
				"content": [
					{
						"type": "input_image",
						"image_url": f"data:image/jpeg;base64,{base64_image}",
					},
				],
			}
		],
		text={
			"format": {
				"type": "json_object"
			}
		},
		store=True
	)
	
	first_look_findings = first_response.output_text
	
	# Second look with context
	second_response = await async_client.responses.create(
		model="gpt-4o",
		input=[
			{
				"role": "system",
				"content": [
					{
						"type": "input_text",
						"text": secondprompt.replace("{first_look_findings}", first_look_findings)
					}
				]
			},
			{
				"role": "user",
				"content": [
					{
						"type": "input_image",
						"image_url": f"data:image/jpeg;base64,{base64_image}",
					},
				],
			}
		],
		text={
			"format": {
				"type": "json_object"
			}
		},
		store=True
	)
	
	return first_response, second_response


# Async web search function
async def websearch_agent(prompt: str, dem_image: str, s2_image: str):
	"""
	Perform web search for historical context using OpenAI's web search tool.
	"""
	response = await async_client.responses.create(
		model="gpt-4.1",
		instructions=prompt,
		input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You must include All the academia paper you found in the output string"
                    }
                ]
            },
			{
				"role": "user",
				"content": [
					{
						"type": "input_text",
						"text": "Below is DEM image:",
					},
					{
						"type": "input_image",
						"image_url": f"data:image/jpeg;base64,{dem_image}",
					},
					{
						"type": "input_text",
						"text": "Below is least cloudy Sentinel-2 image:",
					},
					{
						"type": "input_image",
						"image_url": f"data:image/jpeg;base64,{s2_image}",
					},
				]
			}
		],
		text={
			"format": {
				"type": "text"
			}
		},
		tools=[
			{
				"type": "web_search_preview",
				"user_location": {
					"type": "approximate"
				},
				"search_context_size": "medium",
			}
		],
		tool_choice={
			"type": "web_search_preview"
		},
		top_p=1,
		store=True
	)
	return response


# Async decision agent
async def decision_agent(prompt: str, finding_summary: str):
	"""
	Make final decision based on all analyses.
	"""
	response = await async_client.responses.create(
		model="o4-mini",
		input=[
			{
				"role": "system",
				"content": [
                    {
                        "type": "input_text",
                        "text": "You must include All the academia paper you found in the output string"
                    },
                    {
                        "type": "input_text",
                        "text": "If Isolation forest didn't identify it as anomaly, the site is unlikely to be an archaeological site."
                    },
					{
						"type": "input_text",
						"text": prompt,
					}
				]
			},
			{
				"role": "user",
				"content": [
					{
						"type": "input_text",
						"text": finding_summary,
					}
				]
			}
		],
		text={
			"format": {
				"type": "json_object"
			}
		},
		reasoning={"effort": "low"},
		top_p=1,
		store=True
	)
	return response


# Main async pipeline for processing ROI locations
async def process_single_location(latitude: float, longitude: float, isolation_score: float):
	"""
	Process a single location with parallel image downloads and analyses.
	"""
	# print(f"\nğŸ“� Processing location: ({latitude:.6f}, {longitude:.6f})")
	
	# Parallel download of DEM and S2 images
	dem_task = asyncio.create_task(asyncio.to_thread(
		download_and_process_dem, latitude, longitude, BUFFER_RADIUS_M_DEM, OT_API_KEY, False
	))
	s2_task = asyncio.create_task(asyncio.to_thread(
		get_least_cloudy_s2_image, latitude, longitude, BUFFER_RADIUS_M_S2, False
	))
	
	# Wait for both downloads
	(dem, dem_id), (s2, s2_id) = await asyncio.gather(dem_task, s2_task)
	
	# print(f"	âœ“ Images downloaded: DEM={dem_id}, S2={s2_id}")
	
	# Convert images to base64
	dem_base64 = numpy_image_to_base64(dem, image_format='jpeg')
	s2_base64 = numpy_image_to_base64(s2, image_format='jpeg')
	
	# Parallel analysis: DEM, S2, and Web Search
	dem_analysis_task = LookTwice(DEM_1stlook_prompt, DEM_2ndlook_prompt, dem_base64)
	s2_analysis_task = LookTwice(S2_1stlook_prompt, S2_2ndlook_prompt, s2_base64)
	websearch_task = websearch_agent(SEARCH_PROMPT, dem_base64, s2_base64)
	
	# Run all analyses in parallel
	(
		(first_dem_response, second_dem_response),
		(first_s2_response, second_s2_response),
		websearch_result
	) = await asyncio.gather(dem_analysis_task, s2_analysis_task, websearch_task)
	
	print(f"	âœ“ Analyses completed")
	
	# Compile findings
	image_analysis_summary = f"""
	DEM First Analysis: {first_dem_response.output_text}
	DEM Second Analysis: {second_dem_response.output_text}
	S2 First Analysis: {first_s2_response.output_text}
	S2 Second Analysis: {second_s2_response.output_text}
	Historical Analysis: {websearch_result.output_text}
	"""
	
	isolation_summary = f"Location: latitude {latitude}, longitude {longitude}, isolation_score: {isolation_score}"
	finding_summary = isolation_summary + "\n\n" + image_analysis_summary
	
	# Make final decision
	decision = await decision_agent(DECISION_PROMPT, finding_summary)
	decision_json = json.loads(decision.output_text)
	
	# print(f"	âœ“ Decision: {decision_json.get('classification', 'Unknown')}")
	
	return {
		'location': {'latitude': latitude, 'longitude': longitude},
		'isolation_score': isolation_score,
		'images': {'dem_id': dem_id, 's2_id': s2_id, 'dem': dem, 's2': s2},
		'analyses': {
			'dem_1st': first_dem_response.output_text,
			'dem_2nd': second_dem_response.output_text,
			's2_1st': first_s2_response.output_text,
			's2_2nd': second_s2_response.output_text,
			'websearch': websearch_result.output_text,
			'decision': decision.output_text
		},
		'decision': decision_json,
        'openai_response_id':{
            'dem_1st': first_dem_response.id,
			'dem_2nd': second_dem_response.id,
			's2_1st': first_s2_response.id,
			's2_2nd': second_s2_response.id,
			'websearch': websearch_result.id,
			'decision': decision.id
            
        }
        
	}


# Batch processing with parallel execution
async def DecisionPipeline(roi_locations_withscore: List[Tuple[float, float, float]]):
	"""
	Process multiple locations in parallel.
	"""
	# print(f"\nğŸš€ Starting parallel analysis of {len(roi_locations_withscore)} locations")
	
	# Create tasks for all locations
	tasks = [
		process_single_location(lat, lng, score)
		for lat, lng, score in roi_locations_withscore
	]
	
	# Process all locations in parallel
	results = await asyncio.gather(*tasks, return_exceptions=True)
	
	# Separate successful results from errors
	decisions = []
	found_sites = []
	errors = []
	
	for i, result in enumerate(results):
		if isinstance(result, Exception):
			errors.append((roi_locations_withscore[i], str(result)))
		else:
			decisions.append(result['decision'])
			if result['decision'].get('site_found', False):
				found_sites.append(result)
	
	# print(f"\nâœ… Pipeline complete:")
	# print(f"	- Successful analyses: {len(decisions)}")
	# print(f"	- Sites found: {len(found_sites)}")
	# print(f"	- Errors: {len(errors)}")
	
	return decisions, found_sites, errors


def create_interactive_map(center_lat: float, center_lng: float,
							grid_points: List, roi_candidates: pd.DataFrame):
	"""
	Create an improved interactive Plotly map showing grid points and ROI candidates.
	
	Args:
		center_lat: The latitude of the map's center point.
		center_lng: The longitude of the map's center point.
		grid_points: A list of (latitude, longitude) tuples for the analysis grid.
		roi_candidates: DataFrame with ROI data, including latitude, longitude,
						roi_id, and isolation_score.
	"""
	fig = go.Figure()

	# Add grid points for context
	if grid_points:
		grid_lats = [p[0] for p in grid_points]
		grid_lngs = [p[1] for p in grid_points]
		fig.add_trace(go.Scattermapbox(
			mode='markers',
			lon=grid_lngs,
			lat=grid_lats,
			marker={'size': 5, 'color': 'lightblue', 'opacity': 0.5},
			name='Grid Points',
			hovertemplate='Lat: %{lat}<br>Lng: %{lon}<extra></extra>'
		))

	# Add ROI candidates with improved styling
	if not roi_candidates.empty:
		fig.add_trace(go.Scattermapbox(
			mode='markers',
			lon=roi_candidates['longitude'],
			lat=roi_candidates['latitude'],
			marker={
				'size': 15,
				'color': roi_candidates['isolation_score'],
				'colorscale': 'YlOrRd_r',	# Vibrant, reversed Yellow-Orange-Red scale
				'showscale': True,
				'colorbar': {'title': 'Anomaly Score'}
			},
			text=roi_candidates['roi_id'], # Text is used in the hover template
			name='ROI Candidates',
			hovertemplate=(
				"<b>ROI ID:</b> %{text}<br>"
				"<b>Anomaly Score:</b> %{marker.color:.4f}<br>"
				"<b>Lat:</b> %{lat:.4f}<br>"
				"<b>Lon:</b> %{lon:.4f}"
				"<extra></extra>"
			)
		))

	# Add a distinct center point marker
	fig.add_trace(go.Scattermapbox(
		mode='markers',
		lon=[center_lng],
		lat=[center_lat],
		marker={'size': 15, 'color': 'green', 'symbol': 'star'},
		name='Analysis Center',
		hovertemplate=(
			"<b>Analysis Center</b><br>"
			"<b>Lat:</b> %{lat:.4f}<br>"
			"<b>Lon:</b> %{lon:.4f}"
			"<extra></extra>"
		)
	))

	# Update layout for better presentation
	fig.update_layout(
		title=f"Archaeological Anomaly Analysis - Center: ({center_lat:.4f}, {center_lng:.4f})",
		mapbox={
			'style': 'open-street-map',
			'center': {'lat': center_lat, 'lon': center_lng},
			'zoom': 11
		},
		legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.7)'),
		margin={"r":0,"t":40,"l":0,"b":0}
	)

	return fig


def create_analysis_plots(feature_df: pd.DataFrame, roi_candidates: pd.DataFrame):
	"""
	Create a comprehensive and improved analysis dashboard.
	This function dynamically handles features present in the DataFrame.

	Args:
		feature_df: DataFrame containing all data points and their features.
		roi_candidates: DataFrame containing only the filtered ROI candidates.
	"""
	fig = make_subplots(
		rows=2, cols=2,
		subplot_titles=(
			'<b>Isolation Score Distribution</b>',
			'<b>Feature Correlation Matrix</b>',
			'<b>Anomalous Candidate Locations</b>',
			'<b>Feature Importance (Mean Difference)</b>'
		),
		specs=[
			[{'type': 'histogram'}, {'type': 'heatmap'}],
			[{'type': 'scatter'}, {'type': 'bar'}]
		]
	)

	# ==========================================================================
	# 1. Score Distribution Histogram (Top-Left)
	# ==========================================================================
	fig.add_trace(go.Histogram(
		x=feature_df['isolation_score'],
		name='All Points',
		marker_color='#3399FF',	# A distinct blue
		opacity=0.7,
		nbinsx=40
	), row=1, col=1)

	if not roi_candidates.empty:
		fig.add_trace(go.Histogram(
			x=roi_candidates['isolation_score'],
			name='ROI Candidates',
			marker_color='#FF5733',	# A strong orange/red
			nbinsx=20
		), row=1, col=1)
		
		# Add a line for the anomaly threshold (e.g., the max score in candidates)
		threshold = roi_candidates['isolation_score'].max()
		fig.add_vline(x=threshold, line_width=2, line_dash="dash", line_color="black", 
						annotation_text="Anomaly Cutoff", annotation_position="top right", row=1, col=1)

	# ==========================================================================
	# 2. Feature Correlation Heatmap (Top-Right)
	# ==========================================================================
	feature_cols = ['elevation', 'slope', "mean_temp", "temp_seasonality", "annual_precip", "precip_seasonality", "soil_ph","soil_clay","soil_organic_carbon","max_ndvi"]
	if len(feature_cols) > 1:
		corr_matrix = feature_df[feature_cols].corr()
		fig.add_trace(go.Heatmap(
			z=corr_matrix.values,
			x=corr_matrix.columns,
			y=corr_matrix.columns,
			colorscale='RdBu_r', # Red-Blue diverging scale, good for correlations
			zmid=0,
			# Add text annotations to each cell
			text=corr_matrix.round(2).values,
			texttemplate="%{text}",
			textfont={"size":10}
		), row=1, col=2)

	# ==========================================================================
	# 3. ROI Locations Scatter Plot (Bottom-Left)
	# ==========================================================================
	if not roi_candidates.empty:
		fig.add_trace(go.Scatter(
			x=roi_candidates['longitude'],
			y=roi_candidates['latitude'],
			mode='markers',
			marker=dict(
				size=10,
				color=roi_candidates['isolation_score'],
				colorscale='YlOrRd_r',
				showscale=True,
				colorbar=dict(title='Score', x=0.46, len=0.4) # Position colorbar
			),
			text=roi_candidates['roi_id'],
			name='ROI Candidates',
			hovertemplate=( # Cleaner hover info
				"<b>ROI ID:</b> %{text}<br>"
				"<b>Score:</b> %{marker.color:.4f}<br>"
				"<b>Lat:</b> %{y:.4f}, <b>Lon:</b> %{x:.4f}"
				"<extra></extra>"
			)
		), row=2, col=1)
		fig.update_xaxes(title_text="Longitude", row=2, col=1)
		fig.update_yaxes(title_text="Latitude", row=2, col=1)
	
	# ==========================================================================
	# 4. Feature Importance Bar Chart (Bottom-Right)
	# ==========================================================================
	if len(feature_cols) > 0 and not roi_candidates.empty:
		all_means = feature_df[feature_cols].mean()
		roi_means = roi_candidates[feature_cols].mean()
		diff = (roi_means - all_means).abs().sort_values(ascending=False)
		
		fig.add_trace(go.Bar(
			x=diff.index,
			y=diff.values,
			text=diff.round(3).values, # Add values as text on bars
			textposition='outside',
			name='Feature Differences',
			marker_color='#FFA500' # Bright Orange
		), row=2, col=2)
		fig.update_yaxes(title_text="Absolute Mean Difference", row=2, col=2)

	# ==========================================================================
	# Final Layout Updates
	# ==========================================================================
	fig.update_layout(
		height=800,
		width=900,
		showlegend=True,
		title_text="<b>Comprehensive Archaeological Analysis Dashboard</b>",
		title_x=0.5,
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
	)
	# Use barmode 'overlay' for the histogram
	fig.update_layout(barmode='overlay')
	# Make histogram bars semi-transparent
	fig.update_traces(opacity=0.75, selector=dict(type='histogram'))

	return fig


async def analyze_individual_archaeological_site(lat: float, lng: float, site_name: str, 
										isolation_results: Dict, site_index: int):
	"""
	Analyze a single archaeological site using pre-computed isolation forest results.
	
	Args:
		lat: Site latitude
		lng: Site longitude 
		site_name: Name of the archaeological site
		isolation_results: Results from batch isolation forest analysis
		site_index: Index of this site in the original dataset
		
	Returns:
		Dictionary containing complete site analysis results
	"""
	# print(f"\nğŸ�›ï¸�  ANALYZING SITE: {site_name}")
	# print(f"ğŸ“� Coordinates: ({lat:.6f}, {lng:.6f})")
	# print("-" * 60)
	
	site_start_time = time.time()
	
	results = {
		'site_info': {
			'name': site_name, 
			'latitude': lat, 
			'longitude': lng,
			'site_index': site_index,
			'analysis_timestamp': time.time()
		},
		'isolation_analysis': {},
		'decision_pipeline_results': [],
		'archaeological_features': [],
		'site_classification': {},
		'processing_stats': {}
	}
	try:
		# Extract isolation results for this specific site
		if 'error' in isolation_results:
			print(f"â�Œ Isolation analysis failed: {isolation_results['error']}")
			results['isolation_analysis'] = {'error': isolation_results['error']}
			return results
		
		# Get site-specific results from batch analysis
		feature_matrix = isolation_results.get('feature_matrix', pd.DataFrame())
		roi_candidates = isolation_results.get('roi_candidates', pd.DataFrame())
		
		# Find this site's data in the results
		site_data = None
		site_roi_count = 0
		
		if not feature_matrix.empty:
			# Find closest match to this site's coordinates
			distances = ((feature_matrix['latitude'] - lat)**2 + 
						(feature_matrix['longitude'] - lng)**2)**0.5
			closest_idx = distances.idxmin()
			
			if distances[closest_idx] < 0.001:	# Within ~100m
				site_data = feature_matrix.loc[closest_idx]
				
				# Check if this site is in ROI candidates
				if not roi_candidates.empty:
					roi_distances = ((roi_candidates['latitude'] - lat)**2 + 
									(roi_candidates['longitude'] - lng)**2)**0.5
					site_roi_count = (roi_distances < 0.001).sum()
		
		results['isolation_analysis'] = {
			'site_found_in_analysis': site_data is not None,
			'isolation_score': site_data['isolation_score'] if site_data is not None else None,
			'is_anomalous': site_roi_count > 0,
			'roi_candidate': site_roi_count > 0
		}
		
		# print(f"ğŸ”� Isolation Analysis for {site_name}:")
		# print(f"	- Found in feature matrix: {'Yes' if site_data is not None else 'No'}")
		# print(f"	- Isolation score: {site_data['isolation_score']:.4f}" if site_data is not None else "	- Isolation score: N/A")
		# print(f"	- Anomalous/ROI candidate: {'Yes' if site_roi_count > 0 else 'No'}")
		
		# Step 2: Advanced Analysis for Anomalous Sites
		if site_roi_count > 0:
			# print(f"\nğŸ”¬ Running advanced analysis for anomalous site: {site_name}")
			decision_start = time.time()
			
			# Prepare site for decision pipeline
			site_location_data = [[lat, lng, site_data['isolation_score']]]
			
			# Run archaeological decision pipeline
			decisions, archaeological_features, analysis_errors = await DecisionPipeline(site_location_data)
			
			decision_duration = time.time() - decision_start
			results['processing_stats']['decision_pipeline_time'] = decision_duration
			
			results['decision_pipeline_results'] = decisions
			results['archaeological_features'] = archaeological_features
			
			# Classify site based on findings
			if archaeological_features and len(archaeological_features) > 0:
				feature_count = len(archaeological_features)
				confidence_scores = [f.get('decision', {}).get('confidence_level', 'Unknown') 
								   for f in archaeological_features]
				
				results['site_classification'] = {
					'status': 'SIGNIFICANT_ARCHAEOLOGICAL_POTENTIAL',
					'feature_count': feature_count,
					'confidence_levels': confidence_scores,
					'recommendation': 'HIGH_PRIORITY_FOR_FIELD_INVESTIGATION'
				}
				
				# print(f"\nğŸ�›ï¸�  ARCHAEOLOGICAL ASSESSMENT for {site_name}:")
				# print(f"	âœ… Status: SIGNIFICANT POTENTIAL")
				# print(f"	ğŸ”� Features detected: {feature_count}")
				# print(f"	ğŸ“Š Confidence levels: {confidence_scores}")
				# print(f"	ğŸ�¯ Recommendation: HIGH PRIORITY for field investigation")
				
			else:
				results['site_classification'] = {
					'status': 'ANOMALOUS_BUT_UNCLEAR',
					'feature_count': 0,
					'recommendation': 'MODERATE_PRIORITY_FOR_INVESTIGATION'
				}
				# print(f"\nğŸ”� ARCHAEOLOGICAL ASSESSMENT for {site_name}:")
				# print(f"	âš ï¸�  Status: Anomalous but unclear archaeological significance")
				# print(f"	ğŸ�¯ Recommendation: MODERATE PRIORITY for further investigation")
		
		else:
			results['site_classification'] = {
				'status': 'NORMAL_PROFILE',
				'feature_count': 0,
				'recommendation': 'LOW_PRIORITY_FOR_INVESTIGATION'
			}
			# print(f"\nğŸ“Š ARCHAEOLOGICAL ASSESSMENT for {site_name}:")
			# print(f"	âœ… Status: Normal environmental profile")
			# print(f"	ğŸ�¯ Recommendation: LOW PRIORITY for archaeological investigation")
		
		# Final processing statistics
		total_duration = time.time() - site_start_time
		results['processing_stats']['total_analysis_time'] = total_duration
		
	except Exception as e:
		print(f"\nâ�Œ Error analyzing {site_name}: {str(e)}")
		import traceback
		traceback.print_exc()
		results['error'] = {
			'message': str(e),
			'traceback': traceback.format_exc()
		}
	
	return results


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert NumPy data types to Python native types for JSON serialization.
    
    Args:
        obj: Any object that may contain NumPy types
        
    Returns:
        Object with NumPy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def safe_json_dump(data: Any, filepath: str, indent: int = 2) -> bool:
    """
    Safely dump data to JSON file with NumPy type conversion.
    
    Args:
        data: Data to save
        filepath: Path to save file
        indent: JSON indentation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert NumPy types to Python native types
        json_safe_data = convert_numpy_types(data)
        
        with open(filepath, 'w') as f:
            json.dump(json_safe_data, f, indent=indent)
        return True
        
    except Exception as e:
        # print(f"â�Œ Failed to save JSON file {filepath}: {str(e)}")
        return False
# Improved parallel processing for archaeological analysis

async def process_archaeological_sites_batch(batch_sites: List[Tuple[int, pd.Series]], 
                                             batch_number: int,
                                             model_components: Optional[Dict] = None,
                                             parallel_batch_size: int = 10) -> List[Dict]:
    """
    Process a batch of archaeological sites with TRUE PARALLEL processing within each batch.
    
    Args:
        batch_sites: List of (index, site_row) tuples from sites DataFrame
        batch_number: Batch identifier for tracking
        model_components: Pre-trained model components
        parallel_batch_size: Number of sites to process in parallel (default: 10)
        
    Returns:
        List of analysis results for all sites in the batch
    """
    # print(f"\nğŸš€ PROCESSING ARCHAEOLOGICAL BATCH {batch_number}")
    # print(f"ğŸ“¦ Batch contains {len(batch_sites)} sites")
    # print("==" * 30)
    
    batch_start_time = time.time()
    
    # Step 1: Extract all site coordinates for batch analysis
    batch_coordinates = []
    site_info_list = []
    
    for site_idx, site_row in batch_sites:
        lat, lng, name = site_row['latitude'], site_row['longitude'], site_row['name']
        batch_coordinates.append((lat, lng))
        site_info_list.append({'index': site_idx, 'name': name, 'lat': lat, 'lng': lng})
    
    # print(f"\nğŸ§  Running batch isolation forest analysis on {len(batch_coordinates)} sites...")
    
    # Step 2: Run isolation forest on all sites in batch
    try:
        isolation_results = isolation_forest_agent(batch_coordinates, model_components)
        
        if 'error' in isolation_results:
            print(f"â�Œ Batch isolation analysis failed: {isolation_results['error']}")
            # Create error results for all sites
            batch_results = []
            for site_info in site_info_list:
                error_result = {
                    'site_info': {
                        'name': site_info['name'],
                        'latitude': site_info['lat'],
                        'longitude': site_info['lng'],
                        'site_index': site_info['index']
                    },
                    'error': f"Batch isolation analysis failed: {isolation_results['error']}",
                    'batch_number': batch_number
                }
                batch_results.append(error_result)
            return batch_results
        
    except Exception as isolation_error:
        print(f"â�Œ Critical error in batch isolation analysis: {str(isolation_error)}")
        import traceback
        traceback.print_exc()
        
        batch_results = []
        for site_info in site_info_list:
            error_result = {
                'site_info': {
                    'name': site_info['name'],
                    'latitude': site_info['lat'],
                    'longitude': site_info['lng'],
                    'site_index': site_info['index']
                },
                'error': f"Isolation analysis failed: {str(isolation_error)}",
                'batch_number': batch_number
            }
            batch_results.append(error_result)
        return batch_results
    
    # Step 3: Process sites in TRULY PARALLEL batches using asyncio.gather()
    batch_results = []
    
    # Create parallel processing batches
    parallel_batches = []
    for i in range(0, len(batch_sites), parallel_batch_size):
        parallel_batch = batch_sites[i:i+parallel_batch_size]
        parallel_batches.append((parallel_batch, i))
    
    # print(f"\nâš¡ Processing {len(batch_sites)} sites in {len(parallel_batches)} parallel batches (max {parallel_batch_size} sites per parallel batch)")
    
    for parallel_batch, batch_start_idx in parallel_batches:
        # print(f"\nğŸ”¥ Starting TRULY PARALLEL analysis of {len(parallel_batch)} locations")
        
        # Create tasks for SIMULTANEOUS execution
        analysis_tasks = []
        task_metadata = []
        
        for relative_idx, (site_idx, site_row) in enumerate(parallel_batch):
            site_name = site_row['name']
            site_lat = site_row['latitude']
            site_lng = site_row['longitude']
            
            # print(f"ğŸ�›ï¸�  Queuing site {batch_start_idx + relative_idx + 1}/{len(batch_sites)}: {site_name}")
            
            # Create async task for site analysis
            task = analyze_individual_archaeological_site(
                site_lat, site_lng, site_name, isolation_results, site_idx
            )
            analysis_tasks.append(task)
            task_metadata.append({
                'site_idx': site_idx,
                'site_row': site_row,
                'global_idx': batch_start_idx + relative_idx,
                'site_name': site_name,
                'site_lat': site_lat,
                'site_lng': site_lng
            })
        
        # ğŸš€ EXECUTE ALL TASKS IN PARALLEL using asyncio.gather()
        # print(f"âš¡ Running {len(analysis_tasks)} tasks in PARALLEL...")
        parallel_start_time = time.time()
        
        try:
            # This is where the MAGIC happens - all tasks run simultaneously
            task_results = await asyncio.gather(
                *analysis_tasks,
                return_exceptions=True
            )
            
            parallel_duration = time.time() - parallel_start_time
            # print(f"âœ… Parallel execution completed in {parallel_duration:.2f}s")
            # print(f"âš¡ Speed improvement: {len(analysis_tasks)/parallel_duration:.2f} sites/second")
            
            # Process results
            for i, (result, metadata) in enumerate(zip(task_results, task_metadata)):
                if isinstance(result, Exception):
                    print(f"â�Œ Error analyzing site {metadata['site_name']}: {str(result)}")
                    
                    error_result = {
                        'site_info': {
                            'name': metadata['site_name'],
                            'latitude': metadata['site_lat'],
                            'longitude': metadata['site_lng'],
                            'site_index': metadata['site_idx']
                        },
                        'batch_metadata': {
                            'batch_number': batch_number,
                            'site_index_in_dataset': metadata['site_idx'],
                            'site_index_in_batch': metadata['global_idx'],
                            'batch_size': len(batch_sites),
                            'parallel_batch_size': parallel_batch_size
                        },
                        'error': str(result),
                        'isolation_analysis': None,
                        'decision_pipeline_results': None,
                        'archaeological_features': None,
                        'site_classification': None,
                        'processing_stats': {}
                    }
                    batch_results.append(error_result)
                    
                else:
                    # print(f"âœ… Completed analysis for site {metadata['site_name']} in parallel batch")
                    
                    # Add batch metadata to successful result
                    result['batch_metadata'] = {
                        'batch_number': batch_number,
                        'site_index_in_dataset': metadata['site_idx'],
                        'site_index_in_batch': metadata['global_idx'],
                        'batch_size': len(batch_sites),
                        'parallel_batch_size': parallel_batch_size,
                        'parallel_execution_time': parallel_duration
                    }
                    
                    batch_results.append(result)
                    
                    # Save individual site results with NumPy type conversion
                    if result.get('decision_pipeline_results'):
                        site_results_file = f"/kaggle/working/site_{metadata['site_idx']+1}_{metadata['site_name'].replace(' ', '_')}_batch_{batch_number}.json"
                        
                        site_summary = {
                            'site_info': result['site_info'],
                            'isolation_analysis': result['isolation_analysis'],
                            'site_classification': result['site_classification'],
                            'features_found': len(result['archaeological_features']) if result['archaeological_features'] else 0,
                            'processing_time': result['processing_stats'].get('total_analysis_time', 0),
                            'batch_info': result['batch_metadata'],
                            'detailed_results': {
                                'decisions': result['decision_pipeline_results'],
                                'archaeological_features': result['archaeological_features']
                            }
                        }
                        
                        # Use safe JSON dump with NumPy conversion
                        if safe_json_dump(site_summary, site_results_file):
                            print(f"ğŸ’¾ Site analysis saved: {site_results_file}")
                        else:
                            print(f"â�Œ Failed to save site analysis: {site_results_file}")
        
        except Exception as parallel_error:
            print(f"â�Œ Critical error in parallel processing: {str(parallel_error)}")
            import traceback
            traceback.print_exc()
            
            # Add error results for all sites in this parallel batch
            for metadata in task_metadata:
                error_result = {
                    'site_info': {
                        'name': metadata['site_name'],
                        'latitude': metadata['site_lat'],
                        'longitude': metadata['site_lng'],
                        'site_index': metadata['site_idx']
                    },
                    'batch_metadata': {
                        'batch_number': batch_number,
                        'site_index_in_dataset': metadata['site_idx'],
                        'site_index_in_batch': metadata['global_idx'],
                        'batch_size': len(batch_sites),
                        'parallel_batch_size': parallel_batch_size
                    },
                    'error': f"Parallel processing failed: {str(parallel_error)}",
                    'isolation_analysis': None,
                    'decision_pipeline_results': None,
                    'archaeological_features': None,
                    'site_classification': None,
                    'processing_stats': {}
                }
                batch_results.append(error_result)
    
    batch_duration = time.time() - batch_start_time
    avg_time_per_site = batch_duration / len(batch_sites) if len(batch_sites) > 0 else 0
    
    # print(f"\nâœ… Batch {batch_number} completed in {batch_duration:.1f}s")
    # print(f"ğŸ“Š Analyzed {len(batch_results)} sites")
    # print(f"âš¡ Average time per site: {avg_time_per_site:.2f}s")
    # print(f"ğŸš€ Processing speed: {len(batch_sites)/batch_duration:.2f} sites/second")
    
    return batch_results


async def analyze_archaeological_sites_dataset(sites_df: pd.DataFrame, 
                                                 batch_size: int = 10,
                                                 parallel_batch_size: int = 10,
                                                 model_components: Optional[Dict] = None):
    """
    Analyze all archaeological sites in the dataset using IMPROVED PARALLEL batch processing.

    Args:
        sites_df: DataFrame with columns ['latitude', 'longitude', 'name']
        batch_size: Number of sites to process per batch (default: 10)
        parallel_batch_size: Number of sites to analyze in PARALLEL within each batch (default: 10)
        model_components: Pre-trained model components
        
    Returns:
        Complete analysis results for all archaeological sites
    """
    # print("ğŸ�›ï¸�  IMPROVED PARALLEL ARCHAEOLOGICAL SITES ANALYSIS PIPELINE")
    # print("==" * 40)
    # print(f"ğŸ“Š Dataset: {len(sites_df)} archaeological sites")
    # print(f"ğŸ“¦ Batch size: {batch_size} sites per batch")
    # print(f"âš¡ PARALLEL processing: {parallel_batch_size} sites analyzed SIMULTANEOUSLY")
    # print(f"ğŸš€ Speed improvement: Up to {parallel_batch_size}x faster per batch")
    # print("ğŸ”¬ Analysis mode: Exact site coordinates with TRUE parallel processing")
    # print("==" * 40)
    
    # Validate dataset structure
    required_columns = ['latitude', 'longitude', 'name']
    missing_columns = [col for col in required_columns if col not in sites_df.columns]
    
    if missing_columns:
        raise ValueError(f"Archaeological sites dataset missing required columns: {missing_columns}")
    
    # Display dataset overview
    # print(f"\nğŸ“‹ DATASET OVERVIEW:")
    # print(f"   ğŸ“� Archaeological sites: {len(sites_df)}")
    # print(f"   ğŸŒ� Latitude range: {sites_df['latitude'].min():.6f} to {sites_df['latitude'].max():.6f}")
    # print(f"   ğŸŒ� Longitude range: {sites_df['longitude'].min():.6f} to {sites_df['longitude'].max():.6f}")
    
    # Create processing batches
    site_batches = []
    for i in range(0, len(sites_df), batch_size):
        batch_data = list(sites_df.iloc[i:i+batch_size].iterrows())
        batch_number = i//batch_size + 1
        site_batches.append((batch_data, batch_number))
    
    estimated_speedup = min(parallel_batch_size, batch_size)
    # print(f"\nğŸ“¦ Created {len(site_batches)} processing batches")
    # print(f"âš¡ Expected speedup: Up to {estimated_speedup}x faster with parallel processing")
    
    # Process all batches
    all_analysis_results = []
    pipeline_start_time = time.time()
    
    for batch_data, batch_num in site_batches:
        try:
            # print(f"\nâ�³ Starting batch {batch_num}/{len(site_batches)}")
            
            # Process archaeological site batch with IMPROVED parallel processing
            batch_results = await process_archaeological_sites_batch(
                batch_data, batch_num, model_components, parallel_batch_size
            )
            all_analysis_results.extend(batch_results)
            
            # Save batch summary with NumPy type conversion
            successful_sites = [r for r in batch_results if 'error' not in r]
            failed_sites = [r for r in batch_results if 'error' in r]
            anomalous_sites = [r for r in successful_sites 
                                 if r.get('isolation_analysis', {}).get('is_anomalous', False)]
            
            batch_summary = {
                'batch_number': batch_num,
                'sites_analyzed': len(batch_results),
                'successful_analyses': len(successful_sites),
                'failed_analyses': len(failed_sites),
                'anomalous_sites_detected': len(anomalous_sites),
                'total_archaeological_features': sum(
                    len(site['archaeological_features']) if site['archaeological_features'] else 0 
                    for site in successful_sites
                ),
                'site_names': [site['site_info']['name'] for site in batch_results],
                'parallel_processing_used': True,
                'parallel_batch_size': parallel_batch_size
            }
            
            batch_summary_file = f"/kaggle/working/archaeological_batch_{batch_num}_summary.json"
            if safe_json_dump(batch_summary, batch_summary_file):
                print(f"ğŸ“Š Batch {batch_num}: {len(successful_sites)} analyzed, {len(anomalous_sites)} anomalous, {len(failed_sites)} failed")
            else:
                print(f"â�Œ Failed to save batch {batch_num} summary")
            
        except Exception as batch_error:
            print(f"â�Œ Critical error in batch {batch_num}: {str(batch_error)}")
            import traceback
            traceback.print_exc()
    
    # Generate comprehensive final report
    total_duration = time.time() - pipeline_start_time
    successful_analyses = [r for r in all_analysis_results if 'error' not in r]
    failed_analyses = [r for r in all_analysis_results if 'error' in r]
    anomalous_sites = [r for r in successful_analyses 
                       if r.get('isolation_analysis', {}).get('is_anomalous', False)]
    
    total_features_discovered = sum(
        len(site['archaeological_features']) if site['archaeological_features'] else 0 
        for site in successful_analyses
    )
    
    high_priority_sites = [r for r in successful_analyses 
                           if r.get('site_classification', {}).get('recommendation') == 'HIGH_PRIORITY_FOR_FIELD_INVESTIGATION']
    
    # Calculate speed improvements
    theoretical_sequential_time = total_duration * parallel_batch_size
    actual_speedup = theoretical_sequential_time / total_duration if total_duration > 0 else 1
    
    # Save comprehensive final results with NumPy type conversion
    final_report = {
        'analysis_metadata': {
            'total_sites': len(sites_df),
            'successful_analyses': len(successful_analyses),
            'failed_analyses': len(failed_analyses),
            'success_rate_percent': len(successful_analyses)/len(sites_df)*100,
            'total_processing_time_seconds': total_duration,
            'average_time_per_site_seconds': total_duration/len(sites_df),
            'batch_size_used': batch_size,
            'parallel_batch_size_used': parallel_batch_size,
            'total_batches_processed': len(site_batches),
            'analysis_mode': 'improved_parallel_processing',
            'estimated_speedup_factor': actual_speedup,
            'parallel_processing_enabled': True
        },
        'archaeological_assessment': {
            'anomalous_sites_detected': len(anomalous_sites),
            'total_features_discovered': total_features_discovered,
            'high_priority_sites': len(high_priority_sites),
            'average_features_per_site': total_features_discovered/len(successful_analyses) if successful_analyses else 0,
            'anomaly_detection_rate_percent': len(anomalous_sites)/len(successful_analyses)*100 if successful_analyses else 0
        },
        'performance_metrics': {
            'sites_per_second': len(sites_df)/total_duration if total_duration > 0 else 0,
            'parallel_efficiency': actual_speedup/parallel_batch_size if parallel_batch_size > 0 else 0,
            'theoretical_sequential_time': theoretical_sequential_time,
            'actual_parallel_time': total_duration,
            'time_saved_seconds': theoretical_sequential_time - total_duration
        },
        'detailed_site_analyses': all_analysis_results
    }
    
    final_report_file = "/kaggle/working/improved_parallel_archaeological_analysis_report.json"
    if safe_json_dump(final_report, final_report_file):
        print(f"\nğŸ’¾ Complete analysis report saved: {final_report_file}")
        print(f"âš¡ Processing speed: {len(sites_df)/total_duration:.2f} sites/second")
        print(f"ğŸš€ Estimated speedup: {actual_speedup:.2f}x faster than sequential processing")
        print(f"â�±ï¸�  Time saved: {(theoretical_sequential_time - total_duration)/60:.1f} minutes")
    else:
        print(f"\nâ�Œ Failed to save complete analysis report")
    
    return all_analysis_results


# Safe execution wrapper
async def run_archaeological_sites_analysis(sites_df: pd.DataFrame, 
                                           batch_size: int = 10,
                                           parallel_batch_size: int = 10,
                                           model_components: Optional[Dict] = None):
    """
    Safe wrapper for IMPROVED PARALLEL archaeological sites analysis.
    
    Args:
        sites_df: DataFrame containing archaeological sites data
        batch_size: Batch processing size (default: 10)
        parallel_batch_size: Number of sites to process in PARALLEL (default: 10)
        model_components: Pre-trained model components
        
    Returns:
        Analysis results or empty list on critical failure
    """
    try:
        # print("ğŸ›¡ï¸�  Starting IMPROVED PARALLEL archaeological sites analysis...")
        # print(f"âš¡ Parallel processing: {parallel_batch_size} sites will run simultaneously")
        # print(f"ğŸš€ Expected performance improvement: Up to {parallel_batch_size}x faster")
        
        return await analyze_archaeological_sites_dataset(
            sites_df, batch_size, parallel_batch_size, model_components
        )
    except Exception as critical_error:
        # print(f"\nğŸ’¥ CRITICAL PIPELINE FAILURE: {str(critical_error)}")
        import traceback
        traceback.print_exc()
        return []


# Fixed execution with proper model_components handling
async def run_fixed_archaeological_analysis():
    """
    Run the complete archaeological analysis pipeline with fixes.
    """
    print("ğŸ�›ï¸�  ARCHAEOLOGICAL SITES ANALYSIS PIPELINE")
    print("=" * 80)
    
    # Load your test data (first 90 sites for testing)
    test_subset = test_df.head(90)
    
    try:
        # Ensure model_components is globally accessible
        global model_components
        if 'model_components' not in globals() or model_components is None:
            print("âš ï¸�  Loading pre-trained models...")
            model_components = load_trained_model(MODELS_DIR)
        
        # Run analysis with proper model_components passing
        results = await run_archaeological_sites_analysis(
            sites_df=test_subset, 
            batch_size=10,  # Smaller batch size for testing
            model_components=model_components
        )
        
        # print(f"\nâœ… Analysis completed for {len(test_subset)} sites")
        return results
        
    except Exception as e:
        print(f"â�Œ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

# Run the fixed analysis
results = await run_fixed_archaeological_analysis()


class ArchaeologicalEvaluationMetrics:
    """
    CORRECTED: Comprehensive evaluation metrics for archaeological site detection.
    
    CORRECTED Detection Logic:
    1. Use 'detected' column (True only if classification == 'LIKELY')
    2. Sites filtered out by isolation forest count as missed detections
    3. All test samples are known archaeological sites
    
    Accuracy = detected_sites / total_sites
    """
    
    def __init__(self, known_sites_df: pd.DataFrame, predictions_df: pd.DataFrame):
        """
        Initialize with known archaeological sites and model predictions.
        
        Args:
            known_sites_df: DataFrame with columns ['latitude', 'longitude', 'name'] 
            predictions_df: DataFrame with corrected predictions including 'detected' column
        """
        self.known_sites = known_sites_df
        self.predictions = predictions_df
        self.evaluation_results = {}
        
        # Validate that we're evaluating the same sites
        if len(known_sites_df) != len(predictions_df):
            print(f"âš ï¸�  Warning: Known sites ({len(known_sites_df)}) != Predictions ({len(predictions_df)})")
        
    def calculate_detection_metrics(self, distance_threshold_km: float = 1.0) -> Dict[str, Any]:
        """
        CORRECTED: Calculate detection metrics using corrected AI classification logic.
        
        Detection Logic:
        1. Use 'detected' column (True only if classification_result == 'LIKELY')
        2. Count sites filtered out by isolation forest as missed detections
        3. All test samples are archaeological sites (ground truth = True)
        """
        print(f"ğŸ�¯ Calculating CORRECTED detection metrics using AI classification")
        print("ğŸ“‹ Detection Logic: Only 'detected' = True counts as successful detection")
        print("ğŸ“‹ Framework: ALL test samples are known archaeological sites")
        
        # Basic counts using corrected detection logic
        total_archaeological_sites = len(self.predictions)
        
        # Count detections based on corrected 'detected' column
        detected_sites = int(self.predictions['detected'].sum())
        missed_sites = total_archaeological_sites - detected_sites    # False Negatives
        
        # Breakdown by detection method
        went_through_ai = int(self.predictions['has_decision_pipeline'].sum())
        filtered_by_isolation = total_archaeological_sites - went_through_ai
        
        print(f"ğŸ“Š Total Archaeological Sites: {total_archaeological_sites}")
        print(f"ğŸ“Š Sites that went through AI Pipeline: {went_through_ai}")
        print(f"ğŸ“Š Sites filtered out by Isolation Forest: {filtered_by_isolation}")
        print(f"ğŸ“Š Sites Classified as LIKELY: {detected_sites}")
        print(f"ğŸ“Š Sites Missed (False Negatives): {missed_sites}")
        
        # Calculate metrics
        detection_rate = detected_sites / total_archaeological_sites if total_archaeological_sites > 0 else 0
        
        # In single-class archaeological detection:
        # - Precision = 1.0 (all detections are correct - no false positives)
        # - Recall = detection_rate (TP / (TP + FN))
        # - F1 = 2 * (precision * recall) / (precision + recall)
        
        precision = 1.0  # All predictions are correct (no false positives possible)
        recall = detection_rate
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Score analysis
        isolation_scores = self.predictions['isolation_score'].values
        
        # Separate scores by detection result (using corrected 'detected' column)
        detected_mask = self.predictions['detected'] == True
        
        detected_scores = isolation_scores[detected_mask]
        missed_scores = isolation_scores[~detected_mask]
        
        # Classification breakdown
        classification_counts = self.predictions['classification_result'].value_counts().to_dict()
        
        metrics = {
            'total_known_sites': total_archaeological_sites,
            'total_predictions': total_archaeological_sites,
            'sites_through_ai_pipeline': went_through_ai,
            'sites_filtered_by_isolation': filtered_by_isolation,
            'detected_sites': detected_sites,         # True Positives (LIKELY classifications)
            'missed_sites': missed_sites,            # False Negatives
            'detection_rate': detection_rate,        # TP / (TP + FN)
            'precision': precision,                  # Always 1.0 (no FP possible)
            'recall': recall,                       # Same as detection_rate
            'f1_score': f1,
            'accuracy': detection_rate,             # Same as detection_rate
            'distance_threshold_km': distance_threshold_km,  # Kept for compatibility
            
            # Classification breakdown
            'classification_counts': classification_counts,
            'likely_classifications': classification_counts.get('LIKELY', 0),
            'filtered_out_count': classification_counts.get('FILTERED_OUT', 0),
            'other_classifications': sum(v for k, v in classification_counts.items() 
                                       if k not in ['LIKELY', 'FILTERED_OUT']),
            
            # Score analysis (using corrected detection results)
            'mean_isolation_score_all': np.mean(isolation_scores),
            'median_isolation_score_all': np.median(isolation_scores),
            'std_isolation_score_all': np.std(isolation_scores),
            'mean_isolation_score_detected': np.mean(detected_scores) if len(detected_scores) > 0 else 0.0,
            'mean_isolation_score_missed': np.mean(missed_scores) if len(missed_scores) > 0 else 0.0,
            
            # For backward compatibility
            'mean_distance_to_nearest_km': 0.0,
            'median_distance_to_nearest_km': 0.0,
            'min_distances_km': np.zeros(total_archaeological_sites),
            'detected_site_indices': self.predictions[detected_mask].index.tolist()
        }
        
        self.evaluation_results['detection_metrics'] = metrics
        return metrics
    
    def calculate_anomaly_performance(self) -> Dict[str, Any]:
        """
        CORRECTED: Calculate anomaly detection performance using corrected detection logic.
        """
        # print("ğŸ”� Calculating CORRECTED anomaly detection performance")
        
        total_sites = len(self.predictions)
        isolation_scores = self.predictions['isolation_score'].values
        
        # Use corrected 'detected' column (True only for LIKELY classifications)
        detection_flags = self.predictions['detected'].values
        
        # Calculate metrics
        known_sites_flagged_detected = int(np.sum(detection_flags))
        detection_rate = known_sites_flagged_detected / total_sites if total_sites > 0 else 0
        
        # Score statistics
        detected_scores = isolation_scores[detection_flags]
        not_detected_scores = isolation_scores[~detection_flags]
        
        anomaly_metrics = {
            'known_sites_flagged_detected': known_sites_flagged_detected,  # CORRECTED name
            'anomaly_detection_rate': detection_rate,
            'detection_method_used': 'AI_Classification_Pipeline',
            'mean_isolation_score_known_sites': np.mean(isolation_scores),
            'median_isolation_score_known_sites': np.median(isolation_scores),
            'mean_isolation_score_flagged': np.mean(detected_scores) if len(detected_scores) > 0 else 0.0,
            'mean_isolation_score_not_flagged': np.mean(not_detected_scores) if len(not_detected_scores) > 0 else 0.0,
            'isolation_scores_all': isolation_scores.tolist(),
            'detection_flags_all': detection_flags.tolist()  # CORRECTED: actual detection flags
        }
        
        self.evaluation_results['anomaly_metrics'] = anomaly_metrics
        return anomaly_metrics
    
    def calculate_precision_at_k(self, k_values: List[int] = [5, 10, 20, 50]) -> Dict[str, Any]:
        """
        CORRECTED: Calculate Precision@K using corrected detection results.
        """
        print(f"ğŸ“Š Calculating CORRECTED Precision@K using AI classification for K={k_values}")
        
        # Sort predictions by isolation score (most anomalous first - lowest scores)
        sorted_predictions = self.predictions.sort_values('isolation_score', ascending=True)
        
        precision_at_k = {}
        
        for k in k_values:
            if k > len(sorted_predictions):
                k = len(sorted_predictions)
            
            top_k_preds = sorted_predictions.head(k)
            
            # Count how many of top-k were successfully detected (using corrected 'detected' column)
            detected_count = int(top_k_preds['detected'].sum())
            
            precision_at_k[f'P@{k}'] = detected_count / k if k > 0 else 0.0
        
        self.evaluation_results['precision_at_k'] = precision_at_k
        return precision_at_k
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report using corrected detection logic.
        """
        print("ğŸ“‹ Generating CORRECTED comprehensive evaluation report")
        
        # Calculate all metrics
        detection_metrics = self.calculate_detection_metrics()
        anomaly_metrics = self.calculate_anomaly_performance()
        precision_k_metrics = self.calculate_precision_at_k()
        
        # Summary statistics
        summary = {
            'dataset_statistics': {
                'total_known_sites': len(self.known_sites),
                'total_predictions': len(self.predictions),
                'sites_through_ai_pipeline': detection_metrics['sites_through_ai_pipeline'],
                'sites_filtered_by_isolation': detection_metrics['sites_filtered_by_isolation'],
                'successfully_detected_sites': detection_metrics['detected_sites'],  # LIKELY classifications
                'detection_method': 'AI_Classification_Pipeline_With_Isolation_Filter',
                'geographic_span_lat': self.known_sites['latitude'].max() - self.known_sites['latitude'].min(),
                'geographic_span_lng': self.known_sites['longitude'].max() - self.known_sites['longitude'].min()
            },
            'performance_summary': {
                'detection_accuracy': detection_metrics['accuracy'],
                'detection_rate': detection_metrics['detection_rate'],
                'precision': detection_metrics['precision'],
                'recall': detection_metrics['recall'],
                'f1_score': detection_metrics['f1_score'],
                'isolation_filter_rate': detection_metrics['sites_through_ai_pipeline'] / len(self.predictions),
                'ai_classification_success_rate': detection_metrics['detected_sites'] / detection_metrics['sites_through_ai_pipeline'] if detection_metrics['sites_through_ai_pipeline'] > 0 else 0,
                'precision_at_10': precision_k_metrics.get('P@10', 0.0),
            },
            'classification_breakdown': detection_metrics.get('classification_counts', {})
        }
        
        comprehensive_report = {
            'summary': summary,
            'detection_metrics': detection_metrics,
            'anomaly_metrics': anomaly_metrics,
            'precision_at_k': precision_k_metrics,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'evaluation_type': 'CORRECTED_AI_CLASSIFICATION_WITH_ISOLATION_FILTER'
        }
        
        self.evaluation_results['comprehensive_report'] = comprehensive_report
        return comprehensive_report
    
    def print_evaluation_summary(self):
        """
        Print a comprehensive evaluation summary with corrected metrics.
        """
        if not self.evaluation_results:
            self.generate_comprehensive_report()
        
        report = self.evaluation_results['comprehensive_report']
        
        print("\n" + "="*80)
        print("ğŸ�›ï¸�  CORRECTED ARCHAEOLOGICAL SITE DETECTION EVALUATION REPORT")
        print("="*80)
        print("ğŸ“‹ Evaluation Type: Single-Class Archaeological Detection")
        print("ğŸ�¯ Framework: ALL test samples are known archaeological sites")
        print("ğŸ”� Detection Method: AI Classification Pipeline with Isolation Forest Filter")
        
        # Dataset Statistics
        stats = report['summary']['dataset_statistics']
        print(f"\nğŸ“Š DATASET STATISTICS:")
        print(f"   Total Archaeological Sites Tested: {stats['total_known_sites']}")
        print(f"   Sites that passed Isolation Forest: {stats['sites_through_ai_pipeline']}")
        print(f"   Sites filtered out by Isolation Forest: {stats['sites_filtered_by_isolation']}")
        print(f"   Sites Successfully Detected (LIKELY): {stats['successfully_detected_sites']}")
        print(f"   Sites Missed: {stats['total_known_sites'] - stats['successfully_detected_sites']}")
        
        # Classification breakdown
        if 'classification_breakdown' in report['summary']:
            breakdown = report['summary']['classification_breakdown']
            if breakdown:
                print(f"\nğŸ“‹ CLASSIFICATION BREAKDOWN:")
                for classification, count in breakdown.items():
                    if classification == "LIKELY":
                        symbol = "âœ…"
                    elif classification == "FILTERED_OUT":
                        symbol = "ğŸš«"
                    else:
                        symbol = "â�Œ"
                    print(f"   {symbol} {classification}: {count}")
        
        # Performance Metrics
        perf = report['summary']['performance_summary']
        print(f"\nğŸ�¯ DETECTION PERFORMANCE:")
        print(f"   ğŸ�† OVERALL DETECTION ACCURACY: {perf['detection_accuracy']:.3f} ({perf['detection_accuracy']*100:.1f}%)")
        print(f"   ğŸ“ˆ Detection Rate (Recall): {perf['detection_rate']:.3f} ({perf['detection_rate']*100:.1f}%)")
        print(f"   ğŸ�¯ Precision: {perf['precision']:.3f} (Always 1.0 - no false positives possible)")
        print(f"   ğŸ“Š F1-Score: {perf['f1_score']:.3f}")
        print(f"   ğŸ”� Isolation Filter Pass Rate: {perf['isolation_filter_rate']:.3f} ({perf['isolation_filter_rate']*100:.1f}%)")
        print(f"   ğŸ¤– AI Classification Success Rate: {perf['ai_classification_success_rate']:.3f} ({perf['ai_classification_success_rate']*100:.1f}%)")
        
        # Performance interpretation
        accuracy = perf['detection_accuracy']
        if accuracy >= 0.9:
            performance = "ğŸŒŸ EXCELLENT"
        elif accuracy >= 0.8:
            performance = "âœ… GOOD"
        elif accuracy >= 0.7:
            performance = "âš ï¸�  FAIR"
        elif accuracy >= 0.5:
            performance = "ğŸ”¸ POOR"
        else:
            performance = "â�Œ VERY POOR"
        
        print(f"   ğŸ“Š Performance Level: {performance}")
        
        # Precision@K
        precision_k = report['precision_at_k']
        print(f"\nğŸ“ˆ PRECISION@K ANALYSIS:")
        print(f"   (What % of top-K most anomalous sites were successfully detected)")
        for k, p in precision_k.items():
            print(f"   {k}: {p:.3f} ({p*100:.1f}%)")
        
        # Score Analysis
        detection_metrics = report['detection_metrics']
        print(f"\nğŸ“Š ISOLATION SCORE ANALYSIS:")
        print(f"   Mean Score (All Sites): {detection_metrics['mean_isolation_score_all']:.4f}")
        if detection_metrics['mean_isolation_score_detected'] > 0:
            print(f"   Mean Score (Successfully Detected): {detection_metrics['mean_isolation_score_detected']:.4f}")
        if detection_metrics['mean_isolation_score_missed'] != 0:
            print(f"   Mean Score (Missed Sites): {detection_metrics['mean_isolation_score_missed']:.4f}")
        
        # Detailed Analysis
        filtered_count = stats['sites_filtered_by_isolation']
        missed_by_ai = stats['sites_through_ai_pipeline'] - stats['successfully_detected_sites']
        
        print(f"\nğŸ”� DETAILED FAILURE ANALYSIS:")
        print(f"   ğŸš« Sites filtered out by Isolation Forest: {filtered_count}")
        print(f"   â�Œ Sites that went through AI but classified as non-LIKELY: {missed_by_ai}")
        print(f"   ğŸ“Š Total missed detections: {filtered_count + missed_by_ai}")


# Extract predictions for evaluation
def extract_predictions_for_evaluation(analysis_results: List[Dict], 
                                     original_sites_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract predictions for CORRECTED evaluation using AI decision pipeline results.
    
    CORRECTED Detection Logic:
    1. If decision_pipeline_results exists and classification == "LIKELY" â†’ detected = True
    2. If decision_pipeline_results exists and classification != "LIKELY" â†’ detected = False  
    3. If NO decision_pipeline_results (filtered by isolation forest) â†’ detected = False
    
    Args:
        analysis_results: List of analysis results from the pipeline
        original_sites_df: Original sites DataFrame for reference
        
    Returns:
        DataFrame with corrected prediction data
    """
    
    # print("ğŸ”§ CORRECTED: Extracting predictions using AI classification logic...")
    print("ğŸ“‹ Detection Rule: classification == 'LIKELY' = True Positive")
    print("ğŸ“‹ Fallback Rule: No decision_pipeline_results = False (filtered out by isolation forest)")
    
    predictions_data = []
    
    for i, result in enumerate(analysis_results):
        if 'error' not in result and result.get('site_info'):
            site_info = result['site_info']
            isolation_analysis = result.get('isolation_analysis', {})
            
            # Extract basic site info
            site_name = site_info.get('name', 'Unknown')
            latitude = site_info.get('latitude', 0.0)
            longitude = site_info.get('longitude', 0.0)
            isolation_score = isolation_analysis.get('isolation_score', 0.0)
            is_anomalous = isolation_analysis.get('is_anomalous', False)
            
            # CORRECTED DETECTION LOGIC
            decision_pipeline_results = result.get('decision_pipeline_results', [])
            
            if decision_pipeline_results and len(decision_pipeline_results) > 0:
                # Site went through AI pipeline - check classification
                classification = decision_pipeline_results[0].get('classification', 'UNKNOWN')
                detected = (classification == 'LIKELY')
                detection_method = 'AI_Classification'
                
                # print(f"   Site {i+1} ({site_name}): AI Classification = '{classification}' â†’ {'âœ… DETECTED' if detected else 'â�Œ MISSED'}")
                
            else:
                # Site was filtered out by isolation forest - automatic miss
                classification = 'FILTERED_OUT'
                detected = False  # Always False when filtered out
                detection_method = 'Filtered_By_Isolation_Forest'
                
                # print(f"   Site {i+1} ({site_name}): Filtered by isolation forest â†’ â�Œ MISSED")
            
            # Extract additional details
            confidence_level = 'Unknown'
            site_found = False
            
            if decision_pipeline_results and len(decision_pipeline_results) > 0:
                decision_data = decision_pipeline_results[0]
                confidence_level = decision_data.get('confidence_level', 'Unknown')
                site_found = decision_data.get('site_found', False)
            # print(site_name)
            pred_data = {
                'latitude': latitude,
                'longitude': longitude,
                'site_name': site_name,
                'isolation_score': isolation_score,
                'is_anomaly': is_anomalous,  # Keep for reference
                'classification_result': classification,  # AI classification or FILTERED_OUT
                'detected': detected,  # True only if LIKELY, False otherwise
                'detection_method': detection_method,
                'confidence_level': confidence_level,
                'site_found': site_found,
                'has_decision_pipeline': len(decision_pipeline_results) > 0
            }
            predictions_data.append(pred_data)
        
        else:
            # Handle error cases
            site_info = result.get('site_info', {})
            site_name = site_info.get('name', f'Site_{i+1}')
            
            # print(f"   Site {i+1} ({site_name}): ERROR â†’ â�Œ MISSED")
            
            pred_data = {
                'latitude': site_info.get('latitude', 0.0),
                'longitude': site_info.get('longitude', 0.0),
                'site_name': site_name,
                'isolation_score': 0.0,
                'is_anomaly': False,
                'classification_result': 'ERROR',
                'detected': False,
                'detection_method': 'Error',
                'confidence_level': 'N/A',
                'site_found': False,
                'has_decision_pipeline': False
            }
            predictions_data.append(pred_data)
    
    df = pd.DataFrame(predictions_data)
    
    # Summary statistics
    total_sites = len(df)
    detected_sites = df['detected'].sum()
    went_through_ai = df['has_decision_pipeline'].sum()
    likely_classifications = (df['classification_result'] == 'LIKELY').sum()
    filtered_out = (df['classification_result'] == 'FILTERED_OUT').sum()
    
    print(f"\nğŸ“Š SUMMARY:")
    print(f"   Total Sites: {total_sites}")
    print(f"   Sites that went through AI Pipeline: {went_through_ai}")
    print(f"   Sites filtered out by Isolation Forest: {filtered_out}")
    print(f"   Sites Classified as LIKELY: {likely_classifications}")
    print(f"   Sites Detected (Final): {detected_sites}")
    print(f"   Detection Rate: {detected_sites/total_sites*100:.1f}% ({detected_sites}/{total_sites})")
    
    # Breakdown by classification
    classification_counts = df['classification_result'].value_counts()
    print(f"\nğŸ“‹ CLASSIFICATION BREAKDOWN:")
    for classification, count in classification_counts.items():
        if classification == "LIKELY":
            symbol = "âœ…"
        elif classification == "FILTERED_OUT":
            symbol = "ğŸš«"
        else:
            symbol = "â�Œ"
        print(f"   {symbol} {classification}: {count}")
    
    return df

# Evaluation function
def evaluate_archaeological_detection(analysis_results: List[Dict], 
                                    test_df: pd.DataFrame,
                                    save_results: bool = True) -> Dict[str, Any]:
    """
    Run CORRECTED evaluation using AI classification results.
    
    CORRECTED Detection Logic:
    1. Sites with decision_pipeline_results and classification == 'LIKELY' â†’ detected = True
    2. Sites with decision_pipeline_results and classification != 'LIKELY' â†’ detected = False  
    3. Sites with NO decision_pipeline_results (filtered by isolation forest) â†’ detected = False
    
    Args:
        analysis_results: List of analysis results from the pipeline
        test_df: Original test dataset
        save_results: Whether to save results to files
        
    Returns:
        Dictionary with evaluation results
    """
    
    print("ğŸ”§ RUNNING CORRECTED AI CLASSIFICATION EVALUATION")
    print("=" * 60)
    print("ğŸ�¯ Detection Rule: classification == 'LIKELY' â†’ True Positive")
    print("ğŸ�¯ Filter Rule: No decision_pipeline_results â†’ False Negative")
    print("ğŸ�¯ Framework: ALL test samples are archaeological sites")
    
    if not analysis_results:
        print("â�Œ No analysis results provided")
        return {}
    
    # Extract predictions using CORRECTED logic
    predictions_df = extract_predictions_for_evaluation(analysis_results, test_df)
    
    if len(predictions_df) == 0:
        print("â�Œ No predictions extracted")
        return {}
    
    # Get corresponding known sites
    known_sites_subset = test_df.head(len(predictions_df)).copy()
    
    # Initialize evaluator with corrected predictions
    evaluator = ArchaeologicalEvaluationMetrics(known_sites_subset, predictions_df)
    
    # Generate comprehensive report
    report = evaluator.generate_comprehensive_report()
    
    # Print detailed summary
    evaluator.print_evaluation_summary()
    
    if save_results:
        # Save detailed report
        report_file = '/kaggle/working/corrected_archaeological_evaluation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save predictions with evaluation details
        predictions_file = '/kaggle/working/corrected_predictions_with_evaluation.csv'
        predictions_df.to_csv(predictions_file, index=False)
        
        print(f"\nğŸ’¾ Files Saved:")
        print(f"   - {report_file}")
        print(f"   - {predictions_file}")
    
    # Quick summary
    accuracy = report['summary']['performance_summary']['detection_accuracy']
    detected_count = report['summary']['dataset_statistics']['successfully_detected_sites']
    total_count = report['summary']['dataset_statistics']['total_known_sites']
    filtered_count = report['summary']['dataset_statistics']['sites_filtered_by_isolation']
    
    print(f"\nğŸ�¯ CORRECTED QUICK SUMMARY:")
    print(f"   Overall Detection Accuracy: {accuracy:.1%}")
    print(f"   Sites Successfully Detected: {detected_count}/{total_count}")
    print(f"   Sites Filtered by Isolation Forest: {filtered_count}")
    print(f"   Sites Classified as LIKELY: {detected_count}")
    print(f"   Total Missed: {total_count - detected_count}")
    
    return {
        'evaluator': evaluator,
        'report': report,
        'predictions_df': predictions_df,
        'accuracy': accuracy,
        'detected_count': detected_count,
        'total_count': total_count,
        'filtered_count': filtered_count
    }


# Run evaluation if we have results
if results and len(results) > 0:
    print("\nğŸ”¬ RUNNING EVALUATION")
    print("-" * 40)
    
    # Extract predictions
    predictions_df = extract_predictions_for_evaluation(results, test_df)
    
    if len(predictions_df) > 0:
        # Known sites (ground truth) - these are the actual archaeological sites
        known_sites_subset = test_df.head(len(predictions_df))
        
        # Run comprehensive evaluation
        evaluation_results = evaluate_archaeological_detection(
            analysis_results=results,
            test_df=test_df,
            save_results=True
        )
        
        print("\nğŸ�¯ EVALUATION COMPLETE!")
        print("Check the saved files:")
        print("- archaeological_evaluation_report.json")
        print("- predictions_with_evaluation.csv")
        
    else:
        print("â�Œ No predictions to evaluate")
else:
    print("â�Œ No results to evaluate")


# Archaeological-specific metrics
def calculate_archaeological_specific_metrics(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate performance metrics for archaeological site detection, including
    sensitivity, precision, and recall.

    Args:
        predictions_df: DataFrame with model predictions. Must include:
                        - 'classification_result': The model's prediction (e.g., "LIKELY").
                        - 'is_known_site': The ground truth (True if it is a real site).

    Returns:
        A dictionary containing the calculated metrics.
    """
    if len(predictions_df) == 0:
        return {}

    # Define conditions for TP, FP, FN
    true_positives = (predictions_df['classification_result'] == "LIKELY").sum()
    total_sites = len(predictions_df)
    false_negatives = total_sites - true_positives

    # Sensitivity (Recall) = TP / (TP + FN) = TP / total_sites
    # This is the proportion of known sites that were successfully detected.
    sensitivity_recall = true_positives / total_sites if total_sites > 0 else 0.0
    
    # Precision = TP / (TP + FP). Since there are no non-sites in the data,
    # False Positives (FP) are impossible, so FP = 0.
    # Precision = TP / TP, which will be 1.0 or 100%.
    precision = 1.0 if true_positives > 0 else 0.0

    arch_metrics = {
        'sensitivity_recall': sensitivity_recall,
        'precision': precision,
        
        # Other metrics from your original function
        'high_confidence_rate': len(predictions_df[predictions_df['confidence_level'].str.lower().str.contains('high', na=False)]) / len(predictions_df),
        'moderate_confidence_rate': len(predictions_df[predictions_df['confidence_level'].str.lower().str.contains('moderate', na=False)]) / len(predictions_df),
        'low_confidence_rate': len(predictions_df[predictions_df['confidence_level'].str.lower().str.contains('low', na=False)]) / len(predictions_df),
        
        'mean_isolation_score': predictions_df['isolation_score'].mean(),
        'median_isolation_score': predictions_df['isolation_score'].median(),
        'isolation_score_std': predictions_df['isolation_score'].std(),
    }
    
    return arch_metrics

# --- UPDATED SCRIPT EXECUTION ---
if 'predictions_df' in locals() and len(predictions_df) > 0:
    arch_metrics = calculate_archaeological_specific_metrics(predictions_df)
    
    print("\nğŸ�›ï¸�  ARCHAEOLOGICAL PERFORMANCE METRICS (for known sites dataset)")
    print("=" * 60)
    # Note: Sensitivity and Recall are the same metric. This is your key performance indicator.
    print(f"Sensitivity (Recall / Detection Rate): {arch_metrics['sensitivity_recall']:.2%}")
    print(f"Precision: {arch_metrics['precision']:.2%}")
    print("   (Note: Precision is 100% because the dataset only contains true sites)")
    print("-" * 60)
    print("Confidence & Score Distribution:")
    print(f"  High Confidence Rate: {arch_metrics['high_confidence_rate']:.2%}")
    print(f"  Moderate Confidence Rate: {arch_metrics['moderate_confidence_rate']:.2%}")
    print(f"  Low Confidence Rate: {arch_metrics['low_confidence_rate']:.2%}")
    print(f"  Mean Isolation Score: {arch_metrics['mean_isolation_score']:.4f}")

else:
    print("\nâš ï¸�  No predictions available for archaeological metrics")



def create_clean_evaluation_dashboard(analysis_results: List[Dict]) -> Dict[str, go.Figure]:
    """
    Create clean, separated visualizations for evaluation results.
    
    Args:
        analysis_results: List of analysis results from the pipeline
        
    Returns:
        Dictionary of Plotly figures for different aspects of evaluation
    """
    
    # Extract clean data
    valid_results = [r for r in analysis_results if 'error' not in r and r.get('site_info')]
    
    if not valid_results:
        print("â�Œ No valid results for visualization")
        return {}
    
    # Prepare data
    viz_data = []
    for result in valid_results:
        site_info = result.get('site_info', {})
        isolation = result.get('isolation_analysis', {})
        classification = result.get('site_classification', {})
        
        viz_data.append({
            'site_name': site_info.get('name', 'Unknown'),
            'latitude': site_info.get('latitude', 0.0),
            'longitude': site_info.get('longitude', 0.0),
            'isolation_score': isolation.get('isolation_score', 0.0),
            'is_anomalous': isolation.get('is_anomalous', False),
            'status': classification.get('status', 'UNKNOWN'),
            'recommendation': classification.get('recommendation', 'UNKNOWN'),
            'feature_count': classification.get('feature_count', 0)
        })
    
    df = pd.DataFrame(viz_data)
    
    figures = {}  
    # 1. DETECTION PERFORMANCE SUMMARY
    figures['detection_summary'] = create_detection_summary_chart(df)
    
    # 2. GEOGRAPHIC DISTRIBUTION MAP
    figures['geographic_map'] = create_geographic_distribution_map(df)
    
    # 3. ISOLATION SCORE ANALYSIS
    figures['score_analysis'] = create_score_analysis_chart(df)
    
    # 4. CLASSIFICATION BREAKDOWN
    figures['classification_breakdown'] = create_classification_breakdown(df)
    
    return figures

def create_detection_summary_chart(df: pd.DataFrame) -> go.Figure:
    """Create a clean detection summary chart."""
    
    # Calculate metrics
    total_sites = len(df)
    detected_sites = df['is_anomalous'].sum()
    missed_sites = total_sites - detected_sites
    detection_rate = detected_sites / total_sites * 100
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=['Correctly Detected', 'Missed Detection'],
        y=[detected_sites, missed_sites],
        marker_color=['#28a745', '#dc3545'],  # Green and red
        text=[f'{detected_sites}<br>({detected_sites/total_sites*100:.1f}%)', 
              f'{missed_sites}<br>({missed_sites/total_sites*100:.1f}%)'],
        textposition='auto',
        textfont=dict(size=14, color='white'),
        name='Detection Results'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>Archaeological Site Detection Performance</b><br>' +
                 f'<span style=\"font-size:14px\">Overall Accuracy: {detection_rate:.1f}% ({detected_sites}/{total_sites} sites)</span>',
            x=0.5,
            font=dict(size=18)
        ),
        xaxis_title='Detection Category',
        yaxis_title='Number of Sites',
        showlegend=False,
        plot_bgcolor='white',
        height=400,
        margin=dict(t=80, b=60, l=60, r=60)
    )
    
    # Add grid
    fig.update_yaxes(gridcolor='lightgray', gridwidth=1)
    fig.update_xaxes(linecolor='black', linewidth=1)
    
    return fig

def create_geographic_distribution_map(df: pd.DataFrame) -> go.Figure:
    """Create a clean geographic distribution map."""
    
    fig = go.Figure()
    
    # Detected sites
    detected_df = df[df['is_anomalous'] == True]
    if len(detected_df) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=detected_df['latitude'],
            lon=detected_df['longitude'],
            mode='markers',
            marker=dict(
                size=12,
                color='green',
                opacity=0.8
            ),
            text=detected_df['site_name'],
            name='Correctly Detected Sites',
            hovertemplate='<b>%{text}</b><br>' +
                         'Lat: %{lat:.4f}<br>' +
                         'Lon: %{lon:.4f}<br>' +
                         'Status: Correctly Detected<extra></extra>'
        ))
    
    # Missed sites
    missed_df = df[df['is_anomalous'] == False]
    if len(missed_df) > 0:
        fig.add_trace(go.Scattermapbox(
            lat=missed_df['latitude'],
            lon=missed_df['longitude'],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                opacity=0.8,
                symbol='x'
            ),
            text=missed_df['site_name'],
            name='Missed Detections',
            hovertemplate='<b>%{text}</b><br>' +
                         'Lat: %{lat:.4f}<br>' +
                         'Lon: %{lon:.4f}<br>' +
                         'Status: Missed Detection<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Geographic Distribution of Detection Results</b>',
            x=0.5,
            font=dict(size=18)
        ),
        mapbox=dict(
            style='open-street-map',
            center=dict(
                lat=df['latitude'].mean(),
                lon=df['longitude'].mean()
            ),
            zoom=6
        ),
        height=500,
        margin=dict(t=60, b=20, l=20, r=20),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig

def create_score_analysis_chart(df: pd.DataFrame) -> go.Figure:
    """Create isolation score analysis chart."""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Score Distribution by Detection Status', 'Score Statistics'),
        specs=[[{'type': 'histogram'}, {'type': 'box'}]]
    )
    
    # Histogram
    detected_scores = df[df['is_anomalous'] == True]['isolation_score']
    missed_scores = df[df['is_anomalous'] == False]['isolation_score']
    
    if len(detected_scores) > 0:
        fig.add_trace(
            go.Histogram(
                x=detected_scores,
                name='Detected Sites',
                marker_color='green',
                opacity=0.7,
                nbinsx=15
            ),
            row=1, col=1
        )
    
    if len(missed_scores) > 0:
        fig.add_trace(
            go.Histogram(
                x=missed_scores,
                name='Missed Sites',
                marker_color='red',
                opacity=0.7,
                nbinsx=15
            ),
            row=1, col=1
        )
    
    # Box plot
    box_data = []
    box_labels = []
    
    if len(missed_scores) > 0:
        fig.add_trace(
            go.Box(
                y=missed_scores,
                name='Missed Sites',
                marker_color='red',
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ),
            row=1, col=2
        )
    
    if len(detected_scores) > 0:
        fig.add_trace(
            go.Box(
                y=detected_scores,
                name='Detected Sites',
                marker_color='green',
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Isolation Score Analysis</b>',
            x=0.5,
            font=dict(size=18)
        ),
        height=400,
        showlegend=True,
        plot_bgcolor='white'
    )
    
    fig.update_xaxes(title_text='Isolation Score', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text='Isolation Score', row=1, col=2)
    
    return fig

def create_classification_breakdown(df: pd.DataFrame) -> go.Figure:
    """Create classification breakdown charts."""
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'pie'}, {'type': 'bar'}]],
        subplot_titles=('Site Status Distribution', 'Priority Recommendations')
    )
    
    # Status pie chart
    status_counts = df['status'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            name='Status',
            textinfo='label+percent',
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Recommendation bar chart
    rec_counts = df['recommendation'].value_counts()
    
    # Color mapping for recommendations
    color_map = {
        'HIGH_PRIORITY_FOR_FIELD_INVESTIGATION': '#dc3545',
        'MODERATE_PRIORITY_FOR_INVESTIGATION': '#ffc107',
        'LOW_PRIORITY_FOR_INVESTIGATION': '#28a745',
        'UNKNOWN': '#6c757d'
    }
    
    colors = [color_map.get(rec, '#6c757d') for rec in rec_counts.index]
    
    fig.add_trace(
        go.Bar(
            x=rec_counts.values,
            y=rec_counts.index,
            orientation='h',
            marker_color=colors,
            text=rec_counts.values,
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Site Classification Analysis</b>',
            x=0.5,
            font=dict(size=18)
        ),
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(title_text='Number of Sites', row=1, col=2)
    
    return fig

def display_evaluation_dashboard(analysis_results: List[Dict]):
    """
    Display the complete evaluation dashboard with clean visualizations.
    
    Args:
        analysis_results: List of analysis results from the pipeline
    """
    
    print("ğŸ“Š CREATING CLEAN EVALUATION DASHBOARD")
    print("=" * 50)
    
    figures = create_clean_evaluation_dashboard(analysis_results)
    
    if not figures:
        print("â�Œ No figures generated")
        return
    
    print(f"âœ… Generated {len(figures)} visualization panels:")
    for name in figures.keys():
        print(f"   - {name}")
    
    print("\nğŸ�¯ Displaying evaluation dashboard...")
    
    # Display each figure
    for name, fig in figures.items():
        print(f"\nğŸ“ˆ Showing: {name.replace('_', ' ').title()}")
        fig.show(rednerer='iframe')
    
    # Save figures
    for name, fig in figures.items():
        filename = f'/kaggle/working/eval_viz_{name}.html'
        fig.write_html(filename)
        print(f"ğŸ’¾ Saved: {filename}")
    
    print("\nâœ… Evaluation dashboard complete!")


# Usage example and main function
def run_fixed_evaluation_visualization(analysis_results: List[Dict]):
    """
    Run the complete fixed evaluation visualization pipeline.
    
    Args:
        analysis_results: List of analysis results from the pipeline
    """
    
    print("ğŸ”§ RUNNING EVALUATION VISUALIZATION PIPELINE")
    print("=" * 60)
    print("ğŸ�¯ This creates clean, separated visualizations instead of mixed charts")
    
    if not analysis_results:
        print("â�Œ No analysis results provided")
        return
    
    # Create and display dashboard
    display_evaluation_dashboard(analysis_results)

    print(f"\nâœ… evaluation visualization complete!")
    print(f"ğŸ“� Check /kaggle/working/ for saved HTML files and CSV summary")

# Usage
if 'results' in locals() and results:
    print("ğŸš€ Running fixed evaluation visualization...")
    run_fixed_evaluation_visualization(results)
else:
    print("â�Œ No results available. Please run the main analysis pipeline first.")


def display_section_header(title: str, level: int = 1):
    """Create attractive section headers for Kaggle"""
    if level == 1:
        display(HTML(f'''
        <div style="background: linear-gradient(90deg, #1e3c72, #2a5298); 
                    color: white; padding: 15px; border-radius: 10px; margin: 20px 0;">
            <h2 style="margin:0; text-align:center;">ğŸ�›ï¸� {title}</h2>
        </div>
        '''))
    elif level == 2:
        display(HTML(f'''
        <div style="background: linear-gradient(90deg, #667eea, #764ba2); 
                    color: white; padding: 12px; border-radius: 8px; margin: 15px 0;">
            <h3 style="margin:0;">{title}</h3>
        </div>
        '''))
    elif level == 3:
        display(HTML(f'''
        <div style="background: #f8f9fa; border-left: 4px solid #007bff; 
                    padding: 10px; margin: 10px 0;">
            <h4 style="margin:0; color: #007bff;">{title}</h4>
        </div>
        '''))

def display_info_box(content: str, box_type: str = "info"):
    """Display information in colored boxes"""
    colors = {
        "success": "#d4edda", "error": "#f8d7da", "warning": "#fff3cd", "info": "#d1ecf1"
    }
    border_colors = {
        "success": "#28a745", "error": "#dc3545", "warning": "#ffc107", "info": "#17a2b8"
    }
    
    display(HTML(f'''
    <div style="background-color: {colors.get(box_type, colors['info'])}; 
                border: 1px solid {border_colors.get(box_type, border_colors['info'])}; 
                padding: 10px; border-radius: 5px; margin: 10px 0;">
        {content}
    </div>
    '''))

def get_site_detection_status(result: Dict) -> tuple:
    """
    Determine if a site is detected based on decision pipeline classification
    Returns: (is_detected: bool, classification: str, confidence: str)
    """
    try:
        decision_results = result.get('decision_pipeline_results', [])
        if not decision_results:
            return False, "No Decision", "Unknown"
        
        decision = decision_results[0]
        classification = decision.get('classification', '').upper()
        confidence = decision.get('confidence_level', 'Unknown')
        
        # Define positive classifications (indicating detection)
        positive_classifications = [
            'LIKELY', 'PROBABLE', 'POSSIBLE', 'POTENTIAL',
            'ARCHAEOLOGICAL_SITE', 'SIGNIFICANT_ARCHAEOLOGICAL_POTENTIAL', 
            'HIGH_ARCHAEOLOGICAL_VALUE', 'POTENTIAL_SITE', 'SITE_DETECTED',
            'ANOMALOUS', 'ANOMALY', 'INTERESTING', 'CANDIDATE'
        ]
        
        # Define negative classifications (indicating no detection)  
        negative_indicators = [
            'UNLIKELY', 'NO_SITE', 'NORMAL', 'BACKGROUND', 'NEGATIVE',
            'NOT_ARCHAEOLOGICAL', 'NATURAL', 'GEOLOGICAL_ONLY', 'CLEAR_NEGATIVE'
        ]
        
        # First check for explicit negative indicators
        if any(neg in classification for neg in negative_indicators):
            is_detected = False
        # Then check for positive indicators
        elif any(pos_class in classification for pos_class in positive_classifications):
            is_detected = True
        # Default case - if unclear, consider as not detected
        else:
            is_detected = False
            
        return is_detected, classification, confidence
        
    except Exception as e:
        print(f"Error determining detection status: {e}")
        return False, "Error", "Unknown"

def show_site_overview(result: Dict, location_index: int):
    """Show compact site overview with key information"""
    site_info = result.get('site_info', {})
    site_name = site_info.get('name', f'Site {location_index}')
    lat = site_info.get('latitude', 0.0)
    lng = site_info.get('longitude', 0.0)
    
    is_detected, classification, confidence = get_site_detection_status(result)
    
    # Status styling
    status_color = "#28a745" if is_detected else "#dc3545"
    status_text = "âœ… DETECTED" if is_detected else "â�Œ NOT DETECTED"
    
    display(HTML(f'''
    <div style="border: 2px solid {status_color}; border-radius: 10px; padding: 15px; margin: 15px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; color: #333;">ğŸ“� {site_name}</h3>
                <p style="margin: 5px 0; color: #666;">
                    <strong>Coordinates:</strong> {lat:.6f}, {lng:.6f}<br>
                    <strong>Classification:</strong> {classification}<br>
                    <strong>Confidence:</strong> {confidence}
                </p>
            </div>
            <div style="text-align: center;">
                <div style="background: {status_color}; color: white; padding: 10px; border-radius: 5px;">
                    <strong>{status_text}</strong>
                </div>
            </div>
        </div>
    </div>
    '''))

def create_satellite_imagery_plot(result: Dict, site_name: str, lat: float, lng: float):
    """Create improved satellite imagery visualization"""
    archaeological_features = result.get('archaeological_features', [])
    
    if not archaeological_features or len(archaeological_features) == 0:
        display_info_box("â�Œ No satellite imagery data available", "error")
        return
    
    feature_data = archaeological_features[0]
    images_data = feature_data.get('images', {})
    
    dem_array = images_data.get('dem', None)
    s2_array = images_data.get('s2', None)
    
    # Create figure with improved styling
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'ğŸ›°ï¸� Satellite Imagery Analysis: {site_name}\nğŸ“� {lat:.6f}, {lng:.6f}', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Plot DEM
    if dem_array is not None and isinstance(dem_array, np.ndarray):
        im1 = axes[0].imshow(dem_array, cmap='terrain', interpolation='bilinear')
        axes[0].set_title('ğŸ�”ï¸� Digital Elevation Model (DEM)', fontweight='bold', pad=20)
        axes[0].axis('off')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label('Elevation (m)', rotation=270, labelpad=15)
        
        # Add statistics box
        dem_stats = f"Shape: {dem_array.shape}\nRange: {dem_array.min():.1f} - {dem_array.max():.1f}m"
        axes[0].text(0.02, 0.98, dem_stats, transform=axes[0].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top', fontfamily='monospace', fontsize=9)
    else:
        axes[0].text(0.5, 0.5, 'â�Œ DEM Data\nNot Available', ha='center', va='center', 
                    transform=axes[0].transAxes, fontsize=16, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        axes[0].set_title('ğŸ�”ï¸� DEM (Not Available)', color='red')
    
    # Plot Sentinel-2
    if s2_array is not None and isinstance(s2_array, np.ndarray):
        if len(s2_array.shape) == 3 and s2_array.shape[2] >= 3:
            # RGB composite
            s2_rgb = s2_array[:, :, :3].astype(np.float32)
            # Normalize and enhance
            if s2_rgb.max() > 1:
                s2_rgb = s2_rgb / 255.0
            s2_rgb = np.clip(s2_rgb * 1.3, 0, 1)
            
            axes[1].imshow(s2_rgb, interpolation='bilinear')
        else:
            axes[1].imshow(s2_array, cmap='viridis', interpolation='bilinear')
        
        axes[1].set_title('ğŸ›°ï¸� Sentinel-2 RGB Composite', fontweight='bold', pad=20)
        axes[1].axis('off')
        
        # Add statistics box
        s2_stats = f"Shape: {s2_array.shape}\nRange: {s2_array.min()} - {s2_array.max()}"
        axes[1].text(0.02, 0.98, s2_stats, transform=axes[1].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top', fontfamily='monospace', fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'â�Œ Sentinel-2 Data\nNot Available', ha='center', va='center', 
                    transform=axes[1].transAxes, fontsize=16,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        axes[1].set_title('ğŸ›°ï¸� Sentinel-2 (Not Available)', color='red')
    
    plt.tight_layout()
    plt.show()

def show_isolation_analysis(result: Dict):
    """Show isolation forest analysis results"""
    isolation_analysis = result.get('isolation_analysis', {})
    
    if not isolation_analysis:
        display_info_box("â�Œ No isolation forest analysis data available", "error")
        return
    
    # Extract isolation data
    site_found = isolation_analysis.get('site_found_in_analysis', False)
    isolation_score = isolation_analysis.get('isolation_score', 0.0)
    is_anomalous = isolation_analysis.get('is_anomalous', False)
    roi_candidate = isolation_analysis.get('roi_candidate', False)
    
    # Determine isolation forest status
    iso_status_color = "#28a745" if is_anomalous else "#dc3545"
    iso_status_text = "ANOMALOUS" if is_anomalous else "NORMAL"
    
    # Create isolation analysis display
    isolation_html = f'''
    <div style="background: #f8f9fa; border: 2px solid {iso_status_color}; border-radius: 10px; padding: 20px; margin: 15px 0;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div style="text-align: center; padding: 15px; background: {iso_status_color}20; border-radius: 8px;">
                <h4 style="margin: 0; color: {iso_status_color};">ğŸ�¯ Isolation Status</h4>
                <p style="margin: 10px 0; font-size: 1.3em; font-weight: bold; color: {iso_status_color};">{iso_status_text}</p>
            </div>
            <div style="text-align: center; padding: 15px; background: #e3f2fd; border-radius: 8px;">
                <h4 style="margin: 0; color: #1976d2;">ğŸ“Š Isolation Score</h4>
                <p style="margin: 10px 0; font-size: 1.3em; font-weight: bold; color: #1976d2;">{isolation_score:.6f}</p>
            </div>
            <div style="text-align: center; padding: 15px; background: #e8f5e8; border-radius: 8px;">
                <h4 style="margin: 0; color: #388e3c;">ğŸ”� Site Found</h4>
                <p style="margin: 10px 0; font-size: 1.3em; font-weight: bold; color: #388e3c;">{'âœ… YES' if site_found else 'â�Œ NO'}</p>
            </div>
            <div style="text-align: center; padding: 15px; background: #fff3e0; border-radius: 8px;">
                <h4 style="margin: 0; color: #f57c00;">ğŸ�† ROI Candidate</h4>
                <p style="margin: 10px 0; font-size: 1.3em; font-weight: bold; color: #f57c00;">{'âœ… YES' if roi_candidate else 'â�Œ NO'}</p>
            </div>
        </div>
        
        <div style="margin-top: 15px; padding: 15px; background: white; border-radius: 8px;">
            <h4 style="margin: 0 0 10px 0;">ğŸ“‹ Interpretation:</h4>
    '''
    
    # Add interpretation based on results
    if is_anomalous and roi_candidate:
        isolation_html += '''
            <p style="margin: 5px 0; color: #28a745;"><strong>âœ… PASSED:</strong> Site successfully flagged by isolation forest as anomalous</p>
            <p style="margin: 5px 0; color: #28a745;"><strong>ğŸ�¯ STATUS:</strong> Qualified for detailed AI analysis pipeline</p>
            <p style="margin: 5px 0; color: #666;"><strong>ğŸ“Š RESULT:</strong> Contributes to TRUE POSITIVE detection rate</p>
        '''
    elif is_anomalous and not roi_candidate:
        isolation_html += '''
            <p style="margin: 5px 0; color: #ffc107;"><strong>âš ï¸� PARTIAL:</strong> Flagged as anomalous but not ROI candidate</p>
            <p style="margin: 5px 0; color: #ffc107;"><strong>ğŸ�¯ STATUS:</strong> May proceed to analysis with lower priority</p>
        '''
    else:
        isolation_html += '''
            <p style="margin: 5px 0; color: #dc3545;"><strong>â�Œ FILTERED OUT:</strong> Not flagged as anomalous by isolation forest</p>
            <p style="margin: 5px 0; color: #dc3545;"><strong>ğŸ�¯ STATUS:</strong> Would be excluded from detailed analysis</p>
            <p style="margin: 5px 0; color: #666;"><strong>âš ï¸� RISK:</strong> Potential FALSE NEGATIVE if this is an archaeological site</p>
        '''
    
    isolation_html += '''
        </div>
    </div>
    '''
    
    display(HTML(isolation_html))

def show_decision_results(result: Dict):
    """Show decision pipeline results"""
    decision_results = result.get('decision_pipeline_results', [])
    
    if not decision_results:
        display_info_box("â�Œ No decision pipeline results available", "error")
        return
    
    decision = decision_results[0]
    is_detected, classification, confidence = get_site_detection_status(result)
    
    # Create decision summary
    decision_html = f'''
    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h4>ğŸ“Š Classification Results:</h4>
        <ul>
            <li><strong>Classification:</strong> {classification}</li>
            <li><strong>Confidence Level:</strong> {confidence}</li>
            <li><strong>Site Detected:</strong> {'âœ… YES' if is_detected else 'â�Œ NO'}</li>
        </ul>
    '''
    
    # Add evidence if available
    evidence = decision.get('primary_evidence', '')
    if evidence:
        wrapped_evidence = '<br>'.join(textwrap.wrap(evidence, width=80))
        decision_html += f'<h4>ğŸ”� Primary Evidence:</h4><p>{wrapped_evidence}</p>'
    
    # Add next steps if available
    next_steps = decision.get('recommended_next_steps', '')
    if next_steps:
        wrapped_steps = '<br>'.join(textwrap.wrap(next_steps, width=80))
        decision_html += f'<h4>ğŸ“‹ Recommended Next Steps:</h4><p>{wrapped_steps}</p>'
    
    decision_html += '</div>'
    display(HTML(decision_html))

def show_academic_references(result: Dict):
    """Show academic references from decision pipeline"""
    try:
        decision_results = result.get('decision_pipeline_results', [])
        if not decision_results:
            display_info_box("â�Œ No decision pipeline results available for references", "error")
            return
        
        decision = decision_results[0]
        academic_refs = decision.get('academic_references', [])
        
        if not academic_refs:
            display_info_box("â�Œ No academic references available", "warning")
            return
        
        refs_html = '''
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 10px 0;">
            <h4>ğŸ“š Academic References Used in Analysis:</h4>
            <div style="margin-top: 15px;">
        '''
        
        for i, ref in enumerate(academic_refs, 1):
            # Clean up the reference text
            ref_text = ref.strip().replace('\\"', '"')
            refs_html += f'''
            <div style="background: white; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; border-radius: 4px;">
                <p style="margin: 0; font-family: 'Georgia', serif; line-height: 1.6;">
                    <strong>[{i}]</strong> {ref_text}
                </p>
            </div>
            '''
        
        refs_html += '''
            </div>
        </div>
        '''
        
        display(HTML(refs_html))
        
    except Exception as e:
        display_info_box(f"â�Œ Error displaying academic references: {str(e)}", "error")

def show_intermediate_analyses(result: Dict):
    """Show detailed intermediate analyses from each stage"""
    try:
        archaeological_features = result.get('archaeological_features', [])
        if not archaeological_features:
            display_info_box("â�Œ No archaeological features data available", "error")
            return
        
        analyses = archaeological_features[0].get('analyses', {})
        if not analyses:
            display_info_box("â�Œ No intermediate analyses available", "warning")
            return
        
        # Define analysis stages with descriptions
        stages = [
            ('dem_1st', 'ğŸ�”ï¸� DEM First Look', 'Initial analysis of Digital Elevation Model'),
            ('dem_2nd', 'ğŸ�”ï¸� DEM Second Look', 'Detailed re-analysis of DEM features'),
            ('s2_1st', 'ğŸ›°ï¸� Sentinel-2 First Look', 'Initial Sentinel-2 imagery analysis'),
            ('s2_2nd', 'ğŸ›°ï¸� Sentinel-2 Second Look', 'Detailed Sentinel-2 re-analysis'),
            ('websearch', 'ğŸ”� Web Search & Literature Review', 'Academic literature and context search'),
            ('decision', 'ğŸ�¯ Final Decision', 'Final classification and recommendations')
        ]
        
        analyses_html = '''
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 10px 0;">
        '''
        
        for stage_key, stage_name, stage_desc in stages:
            if stage_key in analyses:
                analysis_data = analyses[stage_key]
                
                # Parse JSON if it's a string
                if isinstance(analysis_data, str):
                    try:
                        import json
                        parsed_data = json.loads(analysis_data)
                        analysis_content = format_analysis_content(parsed_data, stage_key)
                    except:
                        analysis_content = f"<pre style='white-space: pre-wrap; font-size: 0.9em;'>{analysis_data}</pre>"
                else:
                    analysis_content = f"<pre style='white-space: pre-wrap; font-size: 0.9em;'>{str(analysis_data)}</pre>"
                
                analyses_html += f'''
                <div style="margin-bottom: 25px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #007bff, #0056b3); color: white; padding: 15px;">
                        <h4 style="margin: 0;">{stage_name}</h4>
                        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">{stage_desc}</p>
                    </div>
                    <div style="background: white; padding: 20px; max-height: 400px; overflow-y: auto;">
                        {analysis_content}
                    </div>
                </div>
                '''
            else:
                analyses_html += f'''
                <div style="margin-bottom: 25px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
                    <div style="background: #6c757d; color: white; padding: 15px;">
                        <h4 style="margin: 0;">{stage_name}</h4>
                        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;">{stage_desc}</p>
                    </div>
                    <div style="background: #f8f9fa; padding: 20px; text-align: center; color: #6c757d;">
                        <p style="margin: 0;"><em>â�Œ Analysis not completed or data not available</em></p>
                    </div>
                </div>
                '''
        
        analyses_html += '</div>'
        display(HTML(analyses_html))
        
    except Exception as e:
        display_info_box(f"â�Œ Error displaying intermediate analyses: {str(e)}", "error")

def format_analysis_content(data: dict, stage_key: str) -> str:
    """Format analysis content based on stage type"""
    try:
        if stage_key in ['dem_1st', 'dem_2nd', 's2_1st', 's2_2nd']:
            # Handle DEM and S2 analyses
            content_html = ''
            
            if 'identified_anomalies' in data:
                anomalies = data['identified_anomalies']
                if anomalies:
                    content_html += '<h5>ğŸ”� Identified Anomalies:</h5>'
                    for i, anomaly in enumerate(anomalies, 1):
                        feature_id = anomaly.get('feature_id', f'Feature {i}')
                        classification = anomaly.get('classification', 'Unknown')
                        confidence = anomaly.get('confidence_score', 'N/A')
                        description = anomaly.get('description', 'No description')
                        
                        content_html += f'''
                        <div style="background: #e7f3ff; border-left: 4px solid #007bff; padding: 15px; margin: 10px 0; border-radius: 4px;">
                            <h6 style="margin: 0 0 10px 0; color: #007bff;">ğŸ�¯ {feature_id}</h6>
                            <p style="margin: 5px 0;"><strong>Classification:</strong> {classification}</p>
                            <p style="margin: 5px 0;"><strong>Confidence:</strong> {confidence}</p>
                            <p style="margin: 10px 0 0 0; line-height: 1.5;"><strong>Description:</strong> {description}</p>
                        </div>
                        '''
                else:
                    content_html += '<p style="color: #666; font-style: italic;">No anomalies identified in this analysis.</p>'
            
            if 'reason' in data:
                content_html += f'''
                <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px;">
                    <h6 style="margin: 0 0 10px 0; color: #856404;">ğŸ“� Analysis Reasoning:</h6>
                    <p style="margin: 0; line-height: 1.5;">{data['reason']}</p>
                </div>
                '''
            
            if 'anomaly_found' in data:
                status = "âœ… ANOMALIES DETECTED" if data['anomaly_found'] else "â�Œ NO ANOMALIES"
                color = "#28a745" if data['anomaly_found'] else "#dc3545"
                content_html += f'''
                <div style="background: {color}20; border: 1px solid {color}; padding: 10px; border-radius: 4px; text-align: center; margin: 15px 0;">
                    <strong style="color: {color};">{status}</strong>
                </div>
                '''
            
            return content_html if content_html else '<p style="color: #666;">No structured data available for this analysis.</p>'
        
        elif stage_key == 'websearch':
            # Handle websearch content (usually a long text)
            if isinstance(data, str):
                # Format the long text with proper line breaks
                formatted_text = data.replace('\n\n', '</p><p>').replace('\n', '<br>')
                return f'<div style="line-height: 1.6;"><p>{formatted_text}</p></div>'
            else:
                return f'<pre style="white-space: pre-wrap; line-height: 1.5;">{str(data)}</pre>'
        
        elif stage_key == 'decision':
            # Handle decision content
            content_html = ''
            
            if 'site_found' in data:
                status = "âœ… SITE FOUND" if data['site_found'] else "â�Œ NO SITE"
                color = "#28a745" if data['site_found'] else "#dc3545"
                content_html += f'''
                <div style="background: {color}20; border: 1px solid {color}; padding: 15px; border-radius: 4px; text-align: center; margin: 15px 0;">
                    <strong style="color: {color}; font-size: 1.2em;">{status}</strong>
                </div>
                '''
            
            for key, label in [('classification', 'ğŸ�›ï¸� Classification'), ('confidence_level', 'ğŸ“Š Confidence Level')]:
                if key in data:
                    content_html += f'<p><strong>{label}:</strong> {data[key]}</p>'
            
            return content_html if content_html else '<p style="color: #666;">No structured decision data available.</p>'
        
        else:
            # Default formatting
            return f'<pre style="white-space: pre-wrap; line-height: 1.5;">{str(data)}</pre>'
    
    except Exception as e:
        return f'<p style="color: #dc3545;">Error formatting analysis content: {str(e)}</p>'

def show_analysis_pipeline_status(result: Dict):
    """Show analysis pipeline completion status in a clean format"""
    archaeological_features = result.get('archaeological_features', [])
    
    if not archaeological_features:
        display_info_box("â�Œ No analysis pipeline data available", "error")
        return
    
    analyses = archaeological_features[0].get('analyses', {})
    
    # Define pipeline stages
    stages = [
        ('dem_1st', 'ğŸ�”ï¸� DEM First Look'),
        ('dem_2nd', 'ğŸ�”ï¸� DEM Second Look'),
        ('s2_1st', 'ğŸ›°ï¸� S2 First Look'),
        ('s2_2nd', 'ğŸ›°ï¸� S2 Second Look'),
        ('websearch', 'ğŸ”� Web Search'),
        ('decision', 'ğŸ�¯ Final Decision')
    ]
    
    # Create status display
    status_html = '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">'
    
    for stage_key, stage_name in stages:
        completed = stage_key in analyses
        color = "#28a745" if completed else "#dc3545"
        symbol = "âœ…" if completed else "â�Œ"
        
        status_html += f'''
        <div style="background: {color}20; border: 1px solid {color}; 
                    border-radius: 5px; padding: 10px; text-align: center;">
            <strong>{symbol} {stage_name}</strong>
        </div>
        '''
    
    status_html += '</div>'
    
    display_section_header("ğŸ¤– AI Analysis Pipeline Status", 3)
    display(HTML(status_html))

def show_detailed_results(result: Dict, location_index: int):
    """Show detailed results for a single location with improved formatting"""
    
    display_section_header(f"Location {location_index} - Detailed Analysis", 1)
    
    # Site overview
    show_site_overview(result, location_index)
    
    # Get basic info
    site_info = result.get('site_info', {})
    site_name = site_info.get('name', f'Site {location_index}')
    lat = site_info.get('latitude', 0.0)
    lng = site_info.get('longitude', 0.0)
    
    # Satellite imagery
    display_section_header("ğŸ“· Satellite Imagery Analysis", 2)
    create_satellite_imagery_plot(result, site_name, lat, lng)
    
    # Isolation Forest Analysis
    display_section_header("ğŸ”� Isolation Forest Analysis", 2)
    show_isolation_analysis(result)
    
    # Analysis pipeline status
    show_analysis_pipeline_status(result)
    
    # Decision results
    display_section_header("ğŸ�¯ Decision Pipeline Results", 2)
    show_decision_results(result)
    
    # Academic references
    display_section_header("ğŸ“š Academic References", 2)
    show_academic_references(result)
    
    # Detailed intermediate analyses
    display_section_header("ğŸ”¬ Detailed Intermediate Analyses", 2)
    show_intermediate_analyses(result)
    
    # Performance metrics
    display_section_header("â�±ï¸� Processing Performance", 3)
    processing_stats = result.get('processing_stats', {})
    
    if processing_stats:
        total_time = processing_stats.get('total_analysis_time', 0)
        decision_time = processing_stats.get('decision_pipeline_time', 0)
        
        perf_html = f'''
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 10px 0;">
            <div style="text-align: center; padding: 10px; background: #e3f2fd; border-radius: 5px;">
                <h4 style="margin: 0;">ğŸ•’ Total Time</h4>
                <p style="margin: 5px 0; font-size: 1.2em;"><strong>{total_time:.2f}s</strong></p>
            </div>
            <div style="text-align: center; padding: 10px; background: #f3e5f5; border-radius: 5px;">
                <h4 style="margin: 0;">ğŸ¤– Decision Time</h4>
                <p style="margin: 5px 0; font-size: 1.2em;"><strong>{decision_time:.2f}s</strong></p>
            </div>
            <div style="text-align: center; padding: 10px; background: #e8f5e8; border-radius: 5px;">
                <h4 style="margin: 0;">âš¡ Efficiency</h4>
                <p style="margin: 5px 0; font-size: 1.2em;"><strong>{1/total_time:.3f} sites/s</strong></p>
            </div>
        </div>
        '''
        display(HTML(perf_html))
    else:
        display_info_box("â�Œ No performance metrics available", "warning")

def analyze_multiple_locations_improved(results: List[Dict], n_locations: int = 3):
    """
    Improved analysis function with better Kaggle layout and correct detection logic
    """
    
    display_section_header("ğŸ”¬ Archaeological Site Analysis Dashboard", 1)

    display_info_box(f"ğŸ“Š <strong>Analysis Configuration:</strong> Processing up to {n_locations} locations from {len(results)} available results", "info")
    
    # Filter valid results
    valid_results = [r for r in results if 'error' not in r and r.get('site_info')]
    
    if not valid_results:
        display_info_box("â�Œ No valid results available for analysis", "error")
        return
    
    # Select locations
    selected_results = valid_results[:min(n_locations, len(valid_results))]

    display_info_box(f"âœ… <strong>Selected for Analysis:</strong> {len(selected_results)} out of {len(valid_results)} valid locations", "success")
    
    # Show overview
    display_section_header(f"ğŸ“‹ Analysis Overview - {len(selected_results)} Locations", 2)
    
    overview_html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">'
    
    for i, result in enumerate(selected_results):
        site_info = result.get('site_info', {})
        site_name = site_info.get('name', f'Site {i+1}')
        lat = site_info.get('latitude', 0.0)
        lng = site_info.get('longitude', 0.0)
        
        is_detected, classification, confidence = get_site_detection_status(result)
        status_color = "#28a745" if is_detected else "#dc3545"
        status_text = "DETECTED" if is_detected else "NOT DETECTED"
        
        # Get isolation forest results
        isolation_analysis = result.get('isolation_analysis', {})
        is_anomalous = isolation_analysis.get('is_anomalous', False)
        isolation_score = isolation_analysis.get('isolation_score', 0.0)
        roi_candidate = isolation_analysis.get('roi_candidate', False)
        
        iso_color = "#28a745" if is_anomalous else "#dc3545"
        iso_symbol = "ğŸŸ¢" if is_anomalous else "ğŸ”´"
        
        overview_html += f'''
        <div style="border: 1px solid {status_color}; border-radius: 8px; padding: 15px;">
            <h4 style="margin: 0 0 10px 0; color: {status_color};">ğŸ�›ï¸� {site_name}</h4>
            <div style="margin-bottom: 10px;">
                <strong>ğŸ�¯ Final Status:</strong> <span style="color: {status_color};">{status_text}</span><br>
                <strong>ğŸ“Š Classification:</strong> {classification}<br>
                <strong>ğŸ”’ Confidence:</strong> {confidence}
            </div>
            <div style="border-top: 1px solid #eee; padding-top: 10px; font-size: 0.85em;">
                <strong>ğŸ”� Isolation Forest:</strong><br>
                {iso_symbol} <strong>Anomalous:</strong> {'Yes' if is_anomalous else 'No'}<br>
                ğŸ“Š <strong>Score:</strong> {isolation_score:.4f}<br>
                ğŸ�† <strong>ROI:</strong> {'Yes' if roi_candidate else 'No'}
            </div>
        </div>
        '''
    
    overview_html += '</div>'
    display(HTML(overview_html))
    
    # Detailed analysis for each location
    for i, result in enumerate(selected_results):
        show_detailed_results(result, i+1)
        
        # Add separator except for last item
        if i < len(selected_results) - 1:
            display(HTML('<hr style="margin: 40px 0; border: 2px solid #007bff;">'))
    
    # Final summary
    display_section_header("ğŸ�� Analysis Summary", 1)
    
    # Calculate statistics
    total_analyzed = len(selected_results)
    detected_count = sum(1 for r in selected_results if get_site_detection_status(r)[0])
    detection_rate = detected_count / total_analyzed * 100 if total_analyzed > 0 else 0
    
    # Isolation forest statistics  
    iso_anomalous_count = sum(1 for r in selected_results 
                             if r.get('isolation_analysis', {}).get('is_anomalous', False))
    iso_roi_count = sum(1 for r in selected_results 
                       if r.get('isolation_analysis', {}).get('roi_candidate', False))
    iso_detection_rate = iso_anomalous_count / total_analyzed * 100 if total_analyzed > 0 else 0
    
    # Summary metrics
    summary_html = f'''
    <div style="margin-bottom: 30px;">
        <h3 style="text-align: center; color: #333; margin-bottom: 20px;">ğŸ�¯ Final Decision Pipeline Results</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; border-radius: 10px;">
                <h4 style="margin: 0;">ğŸ�›ï¸� Total Sites</h4>
                <p style="margin: 10px 0; font-size: 2em; font-weight: bold;">{total_analyzed}</p>
            </div>
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        color: white; border-radius: 10px;">
                <h4 style="margin: 0;">âœ… Detected</h4>
                <p style="margin: 10px 0; font-size: 2em; font-weight: bold;">{detected_count}</p>
            </div>
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        color: white; border-radius: 10px;">
                <h4 style="margin: 0;">ğŸ“Š Detection Rate</h4>
                <p style="margin: 10px 0; font-size: 2em; font-weight: bold;">{detection_rate:.1f}%</p>
            </div>
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        color: white; border-radius: 10px;">
                <h4 style="margin: 0;">â�Œ Missed</h4>
                <p style="margin: 10px 0; font-size: 2em; font-weight: bold;">{total_analyzed - detected_count}</p>
            </div>
        </div>
    </div>
    
    <div style="margin-bottom: 30px;">
        <h3 style="text-align: center; color: #333; margin-bottom: 20px;">ğŸ”� Isolation Forest Performance</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                        color: white; border-radius: 10px;">
                <h4 style="margin: 0;">ğŸ�¯ Anomalous</h4>
                <p style="margin: 10px 0; font-size: 2em; font-weight: bold;">{iso_anomalous_count}</p>
            </div>
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #ee9ca7 0%, #ffdde1 100%); 
                        color: #333; border-radius: 10px;">
                <h4 style="margin: 0;">ğŸ�† ROI Candidates</h4>
                <p style="margin: 10px 0; font-size: 2em; font-weight: bold;">{iso_roi_count}</p>
            </div>
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        color: #333; border-radius: 10px;">
                <h4 style="margin: 0;">ğŸ“ˆ ISO Detection Rate</h4>
                <p style="margin: 10px 0; font-size: 2em; font-weight: bold;">{iso_detection_rate:.1f}%</p>
            </div>
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                        color: #333; border-radius: 10px;">
                <h4 style="margin: 0;">ğŸ“Š Normal Profile</h4>
                <p style="margin: 10px 0; font-size: 2em; font-weight: bold;">{total_analyzed - iso_anomalous_count}</p>
            </div>
        </div>
    </div>
    '''
    
    display(HTML(summary_html))
    
    # Performance insights
    if detection_rate == 100:
        display_info_box("ğŸ�‰ <strong>EXCELLENT:</strong> All archaeological sites were correctly detected!", "success")
    elif detection_rate >= 80:
        display_info_box("âœ… <strong>GOOD:</strong> Most archaeological sites were correctly detected", "success")
    elif detection_rate >= 50:
        display_info_box("âš ï¸� <strong>FAIR:</strong> Detection rate could be improved", "warning")
    else:
        display_info_box("â�Œ <strong>NEEDS IMPROVEMENT:</strong> Many archaeological sites were missed", "error")
    
    # Isolation forest vs final decision comparison
    if iso_detection_rate != detection_rate:
        diff = abs(iso_detection_rate - detection_rate)
        if iso_detection_rate > detection_rate:
            display_info_box(f"ğŸ”� <strong>ISOLATION FOREST INSIGHT:</strong> Isolation forest flagged {iso_detection_rate:.1f}% as anomalous, but final decision only classified {detection_rate:.1f}% as detected. The AI analysis pipeline filtered out {diff:.1f}% of anomalous sites.", "info")
        else:
            display_info_box(f"ğŸ¤– <strong>AI PIPELINE INSIGHT:</strong> Final AI analysis detected {detection_rate:.1f}% while isolation forest only flagged {iso_detection_rate:.1f}% as anomalous. The AI pipeline enhanced detection by {diff:.1f}%.", "info")

# Main execution function
def run_improved_analysis(results: List[Dict], n_locations: int = 3):
    """
    Main function to run the improved archaeological analysis
    """
    if not results:
        display_info_box("â�Œ No results provided for analysis", "error")
        return
    
    try:
        analyze_multiple_locations_improved(results, n_locations)
        
        # display_section_header("ğŸ�‰ Analysis Complete!", 2)
        display_info_box(f"âœ… Successfully analyzed {min(n_locations, len(results))} locations with enhanced visualization", "success")
        
    except Exception as e:
        display_info_box(f"â�Œ Error during analysis: {str(e)}", "error")
        raise

# Execute the analysis
run_improved_analysis(results, n_locations=min(len(results), 5))


print("\nâœ… INTEGRATION COMPLETE!")
print("Your archaeological site detection pipeline is now fixed and evaluated!")
print("\nğŸ“Š Summary of what was accomplished:")
print("1. âœ… Fixed variable scope issues with model_components")
print("2. âœ… Corrected isolation_forest_agent parameter handling")
print("3. âœ… Added comprehensive evaluation metrics for single-class detection")
print("4. âœ… Implemented archaeological-specific performance metrics")
print("5. âœ… Created visualization and reporting pipeline")
print("\nğŸ“� Check the following files for detailed results:")
print("- /kaggle/working/archaeological_evaluation_report.json")
print("- /kaggle/working/predictions_with_evaluation.csv")
print("- /kaggle/working/evaluation_summary_report.csv")


## classification_result == 'LIKELY' is consider as correct prediction
try:
    final_pred = pd.read_csv('/kaggle/working/corrected_predictions_with_evaluation.csv')
    # Calculate accuracy
    accuracy = sum(final_pred["classification_result"] == "LIKELY") / len(final_pred)
    
    # Display with clear formatting
    print(f"Number of Samples: {len(final_pred)}")
    print(f"Prediction Accuracy: {accuracy:.2%}")
except:
    pass

