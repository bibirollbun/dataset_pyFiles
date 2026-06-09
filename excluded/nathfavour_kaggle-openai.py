# Environment Detection and Setup
import os
import sys
from pathlib import Path

# Detect environment
IS_KAGGLE = '/kaggle' in os.getcwd()
IS_LOCAL = not IS_KAGGLE

print(f"Python version: {sys.version}")
print(f"Environment: {'Kaggle â˜�ï¸�' if IS_KAGGLE else 'Local ğŸ� '}")

if IS_KAGGLE:
    # Kaggle Environment Setup
    print(f"Working directory: {os.getcwd()}")
    print(f"Input data: {os.listdir('/kaggle/input') if os.path.exists('/kaggle/input') else 'No input data'}")
    print(f"GPU available: {os.environ.get('KAGGLE_GPU_TYPE', 'None')}")
    
    WORK_DIR = '/kaggle/working'
    INPUT_DIR = '/kaggle/input'
    
    # Install required packages for Kaggle
    import subprocess
    kaggle_packages = ['openai', 'python-dotenv']
    for package in kaggle_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            print(f"âœ“ Installed {package}")
        except:
            print(f"âœ— Failed to install {package}")
            
else:
    # Local Environment Setup
    repo_root = Path(__file__).parent.parent if '__file__' in globals() else Path('../')
    sys.path.append(str(repo_root))
    
    WORK_DIR = str(repo_root / 'notebooks')
    INPUT_DIR = str(repo_root / 'data')
    
    print(f"Repository root: {repo_root}")
    print(f"Work directory: {WORK_DIR}")

print("âœ“ Environment setup complete!")


# Import packages and configurations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import requests
import json

# Environment-specific imports and configurations
if IS_KAGGLE:
    # === KAGGLE CONFIGURATION ===
    print("ğŸ”§ Setting up Kaggle configuration...")
    
    # Kaggle-specific imports
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        OPENAI_API_KEY = user_secrets.get_secret("openai_api_key")
        print("âœ“ OpenAI API key loaded from Kaggle secrets")
    except:
        OPENAI_API_KEY = None
        print("âš  OpenAI API key not found in Kaggle secrets")
    
    # Inline configuration for Kaggle
    class Config:
        # API Keys
        OPENAI_API_KEY = OPENAI_API_KEY
        
        # Project settings
        PROJECT_NAME = 'openai_to_z_challenge'
        DATA_DIR = INPUT_DIR
        OUTPUT_DIR = WORK_DIR
        
        # Study area (Australia - changed from Amazon for this project)
        STUDY_BOUNDS = {
            'west': 110.0,   # Western Australia
            'south': -45.0,  # Southern Australia
            'east': 155.0,   # Eastern Australia
            'north': -10.0   # Northern Australia
        }
        
        # OpenAI model settings
        OPENAI_MODEL = 'gpt-4o-mini'  # Use available model
        MAX_TOKENS = 4000
        TEMPERATURE = 0.3
        
        # Kaggle-specific settings
        KAGGLE_WORK_DIR = WORK_DIR
        KAGGLE_INPUT_DIR = INPUT_DIR
    
    # Inline OpenAI client for Kaggle
    try:
        import openai
        
        class OpenAIAnalyzer:
            def __init__(self):
                if Config.OPENAI_API_KEY:
                    self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
                    print("âœ“ OpenAI client initialized")
                else:
                    self.client = None
                    print("âš  OpenAI client not initialized - no API key")
            
            def test_connection(self):
                if not self.client:
                    return False
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=10
                    )
                    return True
                except:
                    return False
        
        openai_analyzer = OpenAIAnalyzer()
        
    except ImportError:
        print("âš  OpenAI package not available")
        openai_analyzer = None

else:
    # === LOCAL CONFIGURATION ===
    print("ğŸ”§ Setting up local configuration...")
    
    # Import local modules
    try:
        from src.config import Config
        from src.ai_analysis.openai_client import OpenAIAnalyzer
        print("âœ“ Local modules imported successfully")
        
        # Initialize OpenAI analyzer
        openai_analyzer = OpenAIAnalyzer()
        
    except ImportError as e:
        print(f"âš  Could not import local modules: {e}")
        print("ğŸ’¡ Make sure you're running from the repo root or install requirements.txt")
        
        # Fallback config for local development
        class Config:
            STUDY_BOUNDS = {
                'west': 110.0,
                'south': -45.0, 
                'east': 155.0,
                'north': -10.0
            }
            PROJECT_NAME = 'openai_to_z_challenge'
        
        openai_analyzer = None

# Try to import optional geospatial packages
try:
    import geopandas as gpd
    print("âœ“ GeoPandas available")
    geo_available = True
except ImportError:
    print("âš  GeoPandas not available")
    geo_available = False

print(f"\nğŸ“Š Setup Summary:")
print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"Study area bounds: {Config.STUDY_BOUNDS}")
print(f"Geospatial support: {geo_available}")
print(f"OpenAI integration: {'âœ“' if openai_analyzer else 'âœ—'}")


# Create study area map (works in both environments)
center_lat = (Config.STUDY_BOUNDS['north'] + Config.STUDY_BOUNDS['south']) / 2
center_lon = (Config.STUDY_BOUNDS['east'] + Config.STUDY_BOUNDS['west']) / 2

print(f"Creating map centered at: {center_lat:.2f}, {center_lon:.2f}")

# Create map with environment-appropriate settings
m = folium.Map(
    location=[center_lat, center_lon], 
    zoom_start=5,
    tiles='OpenStreetMap'
)

# Add study area rectangle
folium.Rectangle(
    bounds=[
        [Config.STUDY_BOUNDS['south'], Config.STUDY_BOUNDS['west']],
        [Config.STUDY_BOUNDS['north'], Config.STUDY_BOUNDS['east']]
    ],
    color='red',
    fill=True,
    fillOpacity=0.2,
    popup=f"Australian Study Area ({'Kaggle' if IS_KAGGLE else 'Local'} Mode)"
).add_to(m)

# Add markers for major cities
cities = [
    {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
    {"name": "Melbourne", "lat": -37.8136, "lon": 144.9631},
    {"name": "Brisbane", "lat": -27.4698, "lon": 153.0251},
    {"name": "Perth", "lat": -31.9505, "lon": 115.8605},
    {"name": "Adelaide", "lat": -34.9285, "lon": 138.6007},
    {"name": "Darwin", "lat": -12.4634, "lon": 130.8456}
]

for city in cities:
    folium.Marker(
        [city["lat"], city["lon"]], 
        popup=city["name"],
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

# Save map to appropriate location
map_filename = "study_area_map.html"
if IS_KAGGLE:
    map_path = f"{WORK_DIR}/{map_filename}"
else:
    map_path = f"{WORK_DIR}/{map_filename}"

m.save(map_path)
print(f"ğŸ“� Map saved to: {map_path}")

# Display map
m


# Environment validation and testing
print("=== Environment Validation ===")
print(f"ğŸŒ� Environment: {'Kaggle â˜�ï¸�' if IS_KAGGLE else 'Local ğŸ� '}")
print(f"ğŸ“� Working directory: {os.getcwd()}")
print(f"ğŸ�� Python executable: {sys.executable}")

if IS_KAGGLE:
    # Kaggle-specific checks
    try:
        disk_space = os.statvfs('.').f_bavail * os.statvfs('.').f_frsize // (1024**3)
        print(f"ğŸ’¾ Available disk space: {disk_space} GB")
    except:
        print("ğŸ’¾ Disk space: Unable to check")
    
    if os.path.exists('/kaggle/input'):
        input_datasets = os.listdir('/kaggle/input')
        print(f"ğŸ“Š Input datasets: {input_datasets}")
    else:
        print("ğŸ“Š No input datasets found")
        
    print(f"ğŸ–¥ï¸� GPU: {os.environ.get('KAGGLE_GPU_TYPE', 'None')}")

else:
    # Local-specific checks
    repo_structure = {
        'src': os.path.exists('../src'),
        'data': os.path.exists('../data'),
        'notebooks': os.path.exists('../notebooks'),
        'requirements.txt': os.path.exists('../requirements.txt')
    }
    print(f"ğŸ“� Repository structure: {repo_structure}")

# Test internet connectivity
try:
    response = requests.get("https://httpbin.org/get", timeout=5)
    print("ğŸŒ� Internet connectivity: âœ“ Available")
except:
    print("ğŸŒ� Internet connectivity: âœ— Limited")

# Test OpenAI connection if available
if openai_analyzer:
    if hasattr(openai_analyzer, 'test_connection'):
        connection_test = openai_analyzer.test_connection()
        print(f"ğŸ¤– OpenAI API: {'âœ“ Connected' if connection_test else 'âœ— Connection failed'}")
    else:
        print("ğŸ¤– OpenAI API: âœ“ Client initialized")
else:
    print("ğŸ¤– OpenAI API: âœ— Not available")

print("\n=== Setup Complete ===")
print(f"ğŸš€ Ready for {'Kaggle' if IS_KAGGLE else 'local'} development!")

