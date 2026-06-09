# ==============================================================================
# SECTION: GENERAL LIBRARY IMPORTS & WARNINGS CONFIGURATION
# ==============================================================================
"""
Loads core Python, data handling, and geospatial libraries.
Suppresses non-critical runtime warnings (e.g., NumPy operations on NaNs).
"""

# --------------------------------------------------------------------------
# Suppress common runtime warnings (e.g., invalid value in numpy comparisons)
# --------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------
# Core Data Science Libraries
# --------------------------------------------------------------------------
import numpy as np         # For numerical operations and array manipulation
import pandas as pd        # For tabular data handling and CSV I/O
import geopandas as gpd    # For spatial vector data and shapefiles
import os                  # For file system access and path handling
import re                  # For regular expression matching and parsing
import json                # For reading/writing JSON configuration or metadata



# ==============================================================================
# SECTION: LOAD AND PARSE GEOGEOMETRIC GLYPH DATA
# ==============================================================================
"""
Loads a CSV of Amazon geoglyph point records, extracts geometry and shape information,
cleans and parses description fields, and exports a simplified version for modeling.
"""

# --------------------------------------------------------------------------
# Load original geoglyph dataset and rename key columns
# --------------------------------------------------------------------------
df = pd.read_csv("/kaggle/input/amazon-geoglyphs/geoglyph_points.csv")
df = df.rename(columns={'canonical_name': 'site'})

# --------------------------------------------------------------------------
# Define list of known shape keywords for geoglyph classification
# --------------------------------------------------------------------------
shape_keywords = [
    'circle', 'rectangle', 'square', 'mound', 'mounds', 'quadrangle', 'oval', 'geoglyph',
    'double circle', 'octagon', 'double octagon', 'double square', 'parallelogram',
    'ring', 'double ring', 'enclosure', 'ellipse'
]

# --------------------------------------------------------------------------
# Extract latitude and longitude from WKT-style geometry string
# --------------------------------------------------------------------------
def extract_lat_lon(geom):
    match = re.search(r'POINT Z \(([-\d\.]+) ([-\d\.]+)', geom)
    if match:
        lon, lat = map(float, match.groups())
        return pd.Series({'latitude': lat, 'longitude': lon})
    return pd.Series({'latitude': None, 'longitude': None})

# --------------------------------------------------------------------------
# Parse textual descriptions to extract shape type, size, and notes
# --------------------------------------------------------------------------
def parse_description(desc):
    if pd.isna(desc):
        return pd.Series({'shape': None, 'size': None, 'extra_number': None, 'notes': None})
    
    lines = [line.strip() for line in desc.splitlines() if line.strip()]
    shape, size, extra_number = None, None, None

    # First line: attempt to extract shape keyword and size
    if lines:
        first_line = lines[0].lower()
        for word in shape_keywords:
            if word in first_line:
                shape = word
                break
        size_match = re.search(r'(\d+)\s?m', first_line)
        if size_match:
            size = int(size_match.group(1))

    # Skip lines that contain only coordinates or numeric values
    remaining = [line for line in lines[1:] if not re.match(r"^-?\d+\.\d+$", line.strip())]

    # Extract optional numeric value (e.g., "2") from second line
    if remaining and re.match(r"^\d+$", remaining[0]):
        extra_number = int(remaining[0])
        remaining = remaining[1:]

    # Anything left becomes "notes"
    notes = "\n".join(remaining) if remaining else None

    return pd.Series({
        'shape': shape,
        'size': size,
        'extra_number': extra_number,
        'notes': notes
    })

# --------------------------------------------------------------------------
# Apply parsing to geometry and description columns
# --------------------------------------------------------------------------
df[['latitude', 'longitude']] = df['geometry'].apply(extract_lat_lon)
df[['shape', 'size', 'extra_number', 'notes']] = df['description'].apply(parse_description)

# --------------------------------------------------------------------------
# Drop original raw columns that are no longer needed
# --------------------------------------------------------------------------
df.drop(columns=['description', 'geometry'], inplace=True)

# --------------------------------------------------------------------------
# Save cleaned dataset to disk
# --------------------------------------------------------------------------
df.to_csv("geoglyphs_parsed.csv", index=False)




def process_llanos_de_mojos():
    """
    Loads and processes Llanos de Mojos archaeological site data.

    - Reads a CSV of site names, UTM coordinates, and metadata
    - Converts UTM Zone 20S (EPSG:32720) coordinates to WGS84 lat/lon (EPSG:4326)
    - Combines 'Tier' and 'Observations' into a unified 'notes' column
    - Returns a cleaned DataFrame with key fields for mapping or modeling

    Returns:
    -------
    DataFrame
        Columns: ['site', 'Name', 'latitude', 'longitude', 'notes']
    """
    # ----------------------------------------------------------------------
    # Load raw data and rename columns for consistency
    # ----------------------------------------------------------------------
    df = pd.read_csv("/kaggle/input/llanos-do-mojos-sites/Llanos de Mojos sites.csv")
    df.columns = ['No.', 'site', 'Name', 'x', 'y', 'Tier', 'Observations']

    # Ensure x/y are numeric (in case they're read as strings)
    df['x'] = pd.to_numeric(df['x'])
    df['y'] = pd.to_numeric(df['y'])

    # Cast tier and observations to object (to allow mixed NaN + str)
    df['Tier'] = df['Tier'].astype(object) 
    df['Observations'] = df['Observations'].astype(object)

    from pyproj import Transformer  # Coordinate transformer (UTM → WGS84)

    # ----------------------------------------------------------------------
    # 1. Convert UTM Zone 20S (EPSG:32720) to WGS84 Latitude/Longitude
    # ----------------------------------------------------------------------
    transformer = Transformer.from_crs("EPSG:32720", "EPSG:4326", always_xy=False)

    # Create placeholder lat/lon columns
    df['latitude'] = np.nan
    df['longitude'] = np.nan

    # Identify rows with valid coordinate values
    valid_coords_mask = df['x'].notna() & df['y'].notna()
    x_coords = df.loc[valid_coords_mask, 'x'].values
    y_coords = df.loc[valid_coords_mask, 'y'].values

    # Perform coordinate transformation if possible
    if len(x_coords) > 0:
        latitudes, longitudes = transformer.transform(x_coords, y_coords)
        df.loc[valid_coords_mask, 'latitude'] = latitudes
        df.loc[valid_coords_mask, 'longitude'] = longitudes
    else:
        print("No valid x, y coordinates found to transform.")

    # ----------------------------------------------------------------------
    # 2. Merge 'Tier' and 'Observations' into a single 'notes' column
    # ----------------------------------------------------------------------
    def create_notes_column(row):
        notes_parts = []
        tier_val = row['Tier']
        obs_val = row['Observations']

        if pd.notna(tier_val) and str(tier_val).lower() != 'nan':
            notes_parts.append(f"Tier: {tier_val}")
        if pd.notna(obs_val) and str(obs_val).lower() != 'nan':
            notes_parts.append(f"Observations: {obs_val}")

        return "; ".join(notes_parts) if notes_parts else np.nan

    df['notes'] = df.apply(create_notes_column, axis=1)

    df['source'] = "Llanos de Mojos"
    
    # Return only relevant cleaned columns
    df = df[['site', 'Name', 'latitude', 'longitude', 'notes', 'source', 'Tier']]
    return df




# ==============================================================================
# STEP: Merge and Normalize Archaeological Site Records from Multiple Sources
# ==============================================================================
"""
Loads and standardizes multiple datasets related to pre-Columbian archaeological sites
in the Amazon Basin, including geoglyphs, earthworks, zooarchaeological records, 
peer-reviewed publications, and megalithic structures.

Each dataset is assigned a source label and optional weight (if relevant).
All sources are concatenated into a unified DataFrame for downstream mapping or modeling.
"""

# --------------------------------------------------------------------------
# Load parsed geoglyphs dataset (with lat/lon, shape, and size)
# --------------------------------------------------------------------------
geoglyphs = pd.read_csv("/kaggle/working/geoglyphs_parsed.csv")
geoglyphs['source'] = 'amazon_geoglyphs'
geoglyphs['weight'] = 5  # Assign higher confidence weight

# --------------------------------------------------------------------------
# Load Earthworks Catalog (pre-filtered by region or relevance)
# --------------------------------------------------------------------------
ec_filtered = pd.read_csv("/kaggle/input/earthworks-catalog-filtered/earthworks_catalog_filtered.csv")
ec_filtered = ec_filtered.rename(columns={'canonical_name': 'site'})
ec_filtered['source'] = 'earthworks_catalog'

# --------------------------------------------------------------------------
# Load peer-reviewed science site data (Science Advances, etc.)
# --------------------------------------------------------------------------
sc = pd.read_csv("/kaggle/input/science-data/science.ade2541_data_s2.csv")
sc = sc[['Site', 'Latitude', 'Longitude']]
sc.columns = ['site', 'latitude', 'longitude']
sc['source'] = 'science_data'

# --------------------------------------------------------------------------
# Load processed Llanos de Mojos data (pre-transformed to WGS84)
# --------------------------------------------------------------------------
llanos = process_llanos_de_mojos()

# --------------------------------------------------------------------------
# Load Upper Tapajós Basin site data
# --------------------------------------------------------------------------
utb = pd.read_csv("/kaggle/input/upper-tapajs-basin-sites/Upper Tapajos Basin.csv")
utb['source'] = 'Pre-Columbian earth-builders'

# --------------------------------------------------------------------------
# Load Amazon Megalithic site dataset from ChatGPT Scholar reference
# --------------------------------------------------------------------------
megaliths = pd.read_csv("/kaggle/input/amazon-megalithic-sites-from-chatgpt-scholar/amazon_megalithic_sites_precise.csv")
megaliths.columns = ['Name', 'latitude', 'longitude']
megaliths['source'] = 'ChatGPT Scholar'
megaliths['site'] = megaliths['Name']
megaliths['weight'] = 10  # Assign high weight due to significance

# --------------------------------------------------------------------------
# Combine all site records into a single DataFrame
# --------------------------------------------------------------------------
dfp = pd.concat((geoglyphs, ec_filtered, sc, llanos, utb, megaliths), ignore_index=True)

# Fill missing 'Name' fields with 'site' value for display purposes
dfp.loc[pd.isnull(dfp['Name']), 'Name'] = dfp['site']

# Output the combined dataset
dfp



# ==============================================================================
# STEP: Assign Site Weights Based on Shape, Size, Tier, and Area for Modeling
# ==============================================================================
"""
Weights (scale: 1–10+) represent how archaeologically 'interesting' a site is for
modeling purposes — especially in identifying similar high-impact or rare site types.
"""

# --------------------------------------------------------------------------
# Default all unassigned weights to 1 (baseline interest)
# --------------------------------------------------------------------------
dfp.loc[pd.isnull(dfp['weight']), 'weight'] = 1

# --------------------------------------------------------------------------
# Adjust weight based on known shape categories
# --------------------------------------------------------------------------
dfp.loc[dfp['shape'].isin(['mound']), 'weight'] = 1
dfp.loc[dfp['shape'].isin(['mounded village', 'circle', 'oval', 'ring']), 'weight'] = 2
dfp.loc[dfp['shape'].isin(['square', 'rectangle']), 'weight'] = 3
dfp.loc[dfp['shape'].isin(['parallelogram', 'enclosure', 'circular enclosure']), 'weight'] = 5
dfp.loc[dfp['shape'].isin(['geoglyph']), 'weight'] = 5
dfp.loc[dfp['shape'].isin(['causeway', 'hexagonal enclosure']), 'weight'] = 6
dfp.loc[dfp['shape'].isin(['circular enclosure hexagonal']), 'weight'] = 7
dfp.loc[dfp['shape'].isin(['octagon']), 'weight'] = 8

# Optional: print distribution
dfp['weight'].value_counts()

# --------------------------------------------------------------------------
# Adjust weight based on size (if provided, in meters)
# --------------------------------------------------------------------------
dfp.loc[dfp['size'].between(100, 500), 'weight'] += 1
dfp.loc[dfp['size'].between(500, 1000), 'weight'] += 3
dfp.loc[dfp['size'] > 1000, 'weight'] += 5

# --------------------------------------------------------------------------
# Adjust weight based on estimated site area (if available, in hectares)
# --------------------------------------------------------------------------
dfp.loc[dfp['Area_ha'] < 0.1, 'weight'] = 1
dfp.loc[dfp['Area_ha'].between(0.1, 1), 'weight'] += 1
dfp.loc[dfp['Area_ha'].between(1, 5), 'weight'] += 2
dfp.loc[dfp['Area_ha'].between(5, 10), 'weight'] += 3
dfp.loc[dfp['Area_ha'] > 10, 'weight'] += 5

# --------------------------------------------------------------------------
# Adjust weight based on site Tier ranking (from Llanos de Mojos dataset)
# --------------------------------------------------------------------------
dfp.loc[dfp['Tier'] == 2, 'weight'] += 1
dfp.loc[dfp['Tier'] == 3, 'weight'] += 2
dfp.loc[dfp['Tier'] == 4, 'weight'] += 4

# --------------------------------------------------------------------------
# Final check: view distribution of weight values
# --------------------------------------------------------------------------
dfp['weight'].value_counts()



# ==============================================================================
# STEP: Sort Sites by Weight and Export Final Ranked Dataset
# ==============================================================================
"""
Sorts the unified site dataset (`dfp`) by descending weight (importance).
Exports the result to a CSV file for use in modeling, analysis, or mapping.
"""

# --------------------------------------------------------------------------
# Sort by descending weight to prioritize most significant sites
# --------------------------------------------------------------------------
dfp = dfp.sort_values('weight', ascending=False)

# --------------------------------------------------------------------------
# Save the full ranked dataset to disk
# --------------------------------------------------------------------------
dfp.to_csv("geoglyphs.csv", index=False)

# --------------------------------------------------------------------------
# Preview the top 10 highest-weighted sites
# --------------------------------------------------------------------------
dfp.head(10)





