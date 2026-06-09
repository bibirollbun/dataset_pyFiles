!pip install geoviews



import re
import os
import h5py
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Point
import geoviews as gv
from geoviews import opts, tile_sources as gvts
import holoviews as hv
gv.extension('bokeh', 'matplotlib')
import shapely
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


import requests

def gedi_finder(product, bbox):
    
    # Define the base CMR granule search url, including LPDAAC provider name and max page size (2000 is the max allowed)
    cmr = "https://cmr.earthdata.nasa.gov/search/granules.json?pretty=true&provider=LPCLOUD&page_size=2000&concept_id="
    
    # Set up dictionary where key is GEDI shortname + version
    concept_ids = {'GEDI02_B.002': 'C2142776747-LPCLOUD', 
                   'GEDI02_A.002': 'C2142771958-LPCLOUD', 
                   'GEDI01_B.002': 'C2142749196-LPCLOUD'}
    
    # CMR uses pagination for queries with more features returned than the page size
    page = 1
    bbox = bbox.replace(' ', '')  # remove any white spaces
    try:
        # Send GET request to CMR granule search endpoint w/ product concept ID, bbox & page number, format return as json
        cmr_response = requests.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox}&pageNum={page}").json()['feed']['entry']
        # If 2000 features are returned, move to the next page and submit another request, and append to the response
        while len(cmr_response) % 2000 == 0:
            page += 1
            cmr_response += requests.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox}&pageNum={page}").json()['feed']['entry']
        # CMR returns more info than just the download links, below use list comprehension to return a list of DP links
        return [c['links'][0]['href'] for c in cmr_response]
    except:
        # If the request did not complete successfully, print out the response from CMR
        print(requests.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox.replace(' ', '')}&pageNum={page}").json())


# User-provided inputs (UPDATE FOR YOUR DESIRED PRODUCT AND BOUNDING BOX REGION OF INTEREST)
product = 'GEDI02_B.002'           # Options include 'GEDI01_B.002', 'GEDI02_A.002', 'GEDI02_B.002'
bbox = '-69.26220683008432,-4.919116467989866,-57.18603347986936,-0.7591338330879864'  # bounding box coordinates in LL Longitude, LL Latitude, UR Longitude, UR Latitude format


granules = gedi_finder(product, bbox)
print(f"{len(granules)} {product} Version 2 granules found.")


from datetime import datetime

# Set up output text file name using the current datetime
outName = f"{product.replace('.', '_')}_GranuleList_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

# Open file and write each granule link on a new line
with open(outName, "w") as gf:
    for g in granules:
        gf.write(f"{g}\n")
print(f"File containing links to intersecting {product} Version 2 data has been saved to:\n {os.getcwd()}\{outName}")


#read the file and extract the first link the outname file
with open(outName, "r") as gf:
    first_link = gf.readline().strip()
print(f"The first link in the file is: {first_link}")



!pip install earthaccess==0.8.2 --quiet


import os
import earthaccess
from pathlib import Path

NETRC_PATH = Path("/root/.netrc")  # Explicit path for Kaggle
USERNAME = '<REDACTED>'
PASSWORD = '<REDACTED>'
def create_netrc(username, password):
    """Create .netrc file with explicit permissions check"""
    try:
        # Create the file content
        netrc_content = f"""machine urs.earthdata.nasa.gov
    login {username}
    password {password}
"""
        # Write to explicit path
        with open(NETRC_PATH, 'w') as f:
            f.write(netrc_content)
        
        # Set strict permissions
        os.chmod(NETRC_PATH, 0o600)
        print(f"Created .netrc at {NETRC_PATH}")
        
        # Verify file exists
        if NETRC_PATH.exists():
            print("Verification: .netrc file exists")
            print("File contents:", NETRC_PATH.read_text())
        else:
            print("Error: File creation failed!")
            
    except Exception as e:
        print(f"Error creating .netrc: {str(e)}")
create_netrc(USERNAME, PASSWORD)


import earthaccess
earthaccess.login(strategy="netrc")


import earthaccess
earthaccess.login()
results = [first_link]
# download 
downloaded_files = earthaccess.download(
    results,
    local_path='.',
)


gediFiles = [g for g in os.listdir('.') if g.startswith('GEDI02_B') and g.endswith('.h5')]  # List GEDI L2A .h5 files in the inDir
gediFiles


# Open the HDF5 file and list its groups/datasets
L2B = f'{gediFiles[0]}'
L2B



gediL2B = h5py.File(L2B, 'r')  # Read file using h5py


list(gediL2B.keys())


terraBrasilis_link="https://terrabrasilis.dpi.inpe.br/download/dataset/amz-prodes/vector/prodes_amazonia_nb.gpkg.zip"
# Download the shapefile
response = requests.get(terraBrasilis_link)
# Save the shapefile to a local file
with open('deforestation.zip', 'wb') as f:
    f.write(response.content)

