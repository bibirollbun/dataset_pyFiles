


import pandas as pd
import os
import json
import subprocess
import requests
from tqdm.notebook import tqdm


DATA_DIR = "/fast/Workspace/openai_to_z/data/"
NASA_LIDAR_CSV = "NASA_Lidar/cms_brazil_lidar_tile_inventory.csv"
NASA_LIDAR_DIR = "NASA_Lidar/Lidar_tiles"
NADA_LIDAR_URL = "https://daac.ornl.gov/orders/62322fd434b8bd1f198b1bbc7e32ada0/LiDAR_Forest_Inventory_Brazil/data/"

nasa_lidar_df = pd.read_csv(os.path.join(DATA_DIR, NASA_LIDAR_CSV))





nasa_lidar_df.head()


# Create output directories if they don't exist
output_dir = os.path.join(DATA_DIR, NASA_LIDAR_DIR)
dtm_output_dir = os.path.join(DATA_DIR, "NASA_Lidar/DTM_tiles")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(dtm_output_dir, exist_ok=True)

def create_pdal_pipeline(input_laz, output_tif, srs):
    """Create a PDAL pipeline to convert LAZ to DTM TIFF"""
    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": input_laz,
                "override_srs": srs
            },
            {
                "type": "filters.smrf",
                "scalar": 1.0
            },
            {
                "type": "filters.range",
                "limits": "Classification[2:2]"
            },
            {
                "type": "writers.gdal",
                "filename": output_tif,
                "resolution": 1.0,
                "gdaldriver": "GTiff",
                "output_type": "max",
                "window_size": 5,
                "gdalopts": ["COMPRESS=LZW"]
            }
        ]
    }
    return pipeline

def download_and_process_file(row):
    """Download a LAZ file, convert to DTM, and clean up"""
    filename = row['filename']
    url = NADA_LIDAR_URL + filename
    
    # File paths
    laz_file = os.path.join(output_dir, filename)
    dtm_file = os.path.join(dtm_output_dir, filename.replace('.laz', '.tif'))
    
    # Skip if DTM already exists
    if os.path.exists(dtm_file):
        print(f"DTM file already exists: {dtm_file}")
        return
    
    try:
        # Download LAZ file
        print(f"Downloading {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(laz_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Create and run PDAL pipeline
        pipeline = create_pdal_pipeline(laz_file, dtm_file, row['srs'])
        pipeline_json = json.dumps(pipeline)

        
        # Run PDAL pipeline
        print(f"Converting {filename} to DTM")
        subprocess.run(["pdal", "pipeline", "-s"], input=pipeline_json.encode(), check=True)
        
        # Clean up LAZ file
        os.remove(laz_file)
        print(f"Processed {filename} successfully")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        # Clean up if file was downloaded but processing failed
        if os.path.exists(laz_file):
            os.remove(laz_file)

# Process a small subset for testing (first 5 files)
#test_subset = nasa_lidar_df.head(5)

#for _, row in tqdm(test_subset.iterrows(), total=len(test_subset)):
#    download_and_process_file(row)

# Process all files in the dataframe
# Uncomment when ready to process all files
for _, row in tqdm(nasa_lidar_df.iterrows(), total=len(nasa_lidar_df)):
     download_and_process_file(row)




