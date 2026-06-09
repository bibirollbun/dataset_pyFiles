# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install google-cloud-bigquery[bqstorage]


import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
print("Step 1 complete. Now RESTART the kernel (Runtime -> Restart Runtime)")
print("Then run Cell 2 below.")


# !pip install --upgrade google-cloud-bigquery==3.31.0
# !pip install rich==13.7.1



# ===================================================================
# DUAL-PATH BIGQUERY CONNECTION FOR KAGGLE NOTEBOOKS
# -------------------------------------------------------------------
# This cell creates a robust connection that works in two modes:
# 1. EDITOR MODE: If Kaggle Secrets are available, it uses them for
#    private, authenticated access.
# 2. PUBLIC MODE: If secrets are NOT found, it falls back to public
#    authentication, which works for judges and public viewers if the
#    dataset is public.
# ===================================================================

# --- Step 1: Necessary Imports ---
import os
from google.colab import auth
from google.cloud import bigquery
import bigframes.pandas as bf
from kaggle_secrets import UserSecretsClient

# --- Step 2: Define Project Constants ---
project_id = "bqhackathonautoianalysis"
location = "us-central1"

# --- Step 3: Clean up any previous BigFrames session ---
# This prevents hanging issues on re-runs.
try:
    bf.close_session()
except Exception:
    pass

# --- Step 4: Attempt Authentication ---
print("Attempting to connect to BigQuery...")
client = None
auth_method = "Unknown"

try:
    # --- PATH 1: EDITOR MODE (using Kaggle Secrets) ---
    print("  -> Trying Editor Mode (using Kaggle Secrets)...")
    
    user_secrets = UserSecretsClient()
    service_account_info = user_secrets.get_secret("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Write the secret to a temporary file
    with open('/tmp/service-account-key.json', 'w') as f:
        f.write(service_account_info)
    
    # Set the environment variable for automatic authentication
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/tmp/service-account-key.json'
    
    # Initialize clients (they will automatically use the environment variable)
    client = bigquery.Client(project=project_id)
    bf.options.bigquery.project = project_id
    bf.options.bigquery.location = location
    
    auth_method = "Kaggle Secrets"
    print("âœ… Successfully connected using Editor Mode (Kaggle Secrets).")

except Exception:
    # --- PATH 2: PUBLIC MODE (using default authentication) ---
    print("  -> Kaggle Secrets not found. Falling back to Public Mode...")
    
    try:
        # Authenticate the user for the session (triggers pop-up for owner, automatic for public)
        auth.authenticate_user()
        
        # Explicitly get the credentials to pass to the clients
        credentials, _ = auth.default()

        client = bigquery.Client(project=project_id, credentials=credentials)
        bf.options.bigquery.project = project_id
        bf.options.bigquery.location = location
        bf.options.bigquery.credentials = credentials
        
        auth_method = "Public Authentication"
        print("âœ… Successfully connected using Public Mode.")
        
    except Exception as e:
        print(f"â�Œ Public authentication also failed. Error: {e}")
        auth_method = "Failed"

# --- Step 5: Final Connection Test ---
if client:
    try:
        test_query = f"SELECT COUNT(*) as total_rows FROM `{project_id}.autoAnalysis_Dataset.Claims`"
        result = client.query(test_query).to_dataframe()
        print("\n---------------------------------------------------------")
        print(f"ğŸ�‰ SUCCESS! Connection established via '{auth_method}'.")
        print(f"Total rows in the Claims table: {result['total_rows'].iloc[0]}")
        print("---------------------------------------------------------")
    except Exception as e:
        print(f"â�Œ Connection test failed after authenticating via '{auth_method}'.")
        print(f"   Error: {e}")
else:
    print("\nâ�Œ Could not establish a BigQuery client.")
    print("\nTroubleshooting:")
    print("  - For Editors: Ensure your Kaggle secret is named 'GOOGLE_APPLICATION_CREDENTIALS'.")
    print("  - For Public Users/Judges: The dataset 'bqhackathonautoianalysis.autoAnalysis_Dataset' must be made public.")

# ===================================================================


sql = """
SELECT * FROM `bqhackathonautoianalysis.autoAnalysis_Dataset.Policies` 
"""
df = client.query(sql).to_dataframe()
df


# Query 2: List all tables in the dataset
query_tables = """
SELECT table_name
FROM `bqhackathonautoianalysis.autoAnalysis_Dataset.INFORMATION_SCHEMA.TABLES`
ORDER BY table_name
"""
df_tables = client.query(query_tables).to_dataframe()
print(df_tables)


from google.cloud import bigquery
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize BigQuery client
project_id = "bqhackathonautoianalysis"
bq_client = bigquery.Client(project=project_id)

# Define table names
tables = {
    "Image Claims": "Image_Claims_Analysis",
    "Vehicles": "Vehicles", 
    "Claims": "Claims",
    "Policies": "Policies",
    "Policyholders": "Policyholders"
}

def clean_dataframe_simple(df):
    """Simple cleaning without fillna - just handle infinities"""
    # Replace infinite values with NaN (but don't fill)
    df.replace([float('inf'), float('-inf')], np.nan, inplace=True)
    return df

# Alternative: Query with explicit handling in SQL
def get_table_with_coalesce(table_name):
    """Get table data with SQL-level null handling"""
    full_table = f"{project_id}.autoAnalysis_Dataset.{table_name}"
    
    # First, get column info
    info_query = f"""
    SELECT column_name, data_type 
    FROM `{project_id}.autoAnalysis_Dataset.INFORMATION_SCHEMA.COLUMNS` 
    WHERE table_name = '{table_name}'
    """
    
    try:
        col_info = bq_client.query(info_query).to_dataframe()
        
        # Build select statement with appropriate defaults
        select_parts = []
        for _, row in col_info.iterrows():
            col_name = row['column_name']
            data_type = row['data_type'].upper()
            
            if 'BOOL' in data_type:
                select_parts.append(f"COALESCE({col_name}, FALSE) as {col_name}")
            elif any(x in data_type for x in ['INT', 'FLOAT', 'NUMERIC']):
                select_parts.append(f"COALESCE({col_name}, 0) as {col_name}")
            elif 'STRING' in data_type:
                select_parts.append(f"COALESCE({col_name}, '') as {col_name}")
            else:
                select_parts.append(col_name)  # Leave as is
        
        query = f"SELECT {', '.join(select_parts)} FROM `{full_table}`"
        return bq_client.query(query).to_dataframe()
        
    except Exception as e:
        print(f"Could not get column info for {table_name}: {e}")
        # Fallback to simple query
        query = f"SELECT * FROM `{full_table}`"
        return bq_client.query(query).to_dataframe()

# Query and display each table
for name, table in tables.items():
    print(f"\n=== {name.upper()} ===")
    
    try:
        # Try the SQL-level approach first
        df = get_table_with_coalesce(table)
        
        # Simple cleaning
        df = clean_dataframe_simple(df)
        
        print(f"Successfully loaded {name}")
        display(df.head())
        print(f"Shape: {df.shape}")
        print(f"Data types:")
        print(df.dtypes)
        print(f"Null values per column:")
        print(df.isnull().sum())
        
    except Exception as e:
        print(f"Error loading {name}: {e}")
        
        # Try simple fallback approach
        try:
            full_table = f"{project_id}.autoAnalysis_Dataset.{table}"
            query = f"SELECT * FROM `{full_table}`"
            df = bq_client.query(query).to_dataframe()
            
            print(f"Fallback successful for {name} (with null values)")
            display(df.head())
            print(f"Shape: {df.shape}")
            print(f"Null values per column:")
            print(df.isnull().sum())
            
        except Exception as e2:
            print(f"Complete failure for {name}: {e2}")
            continue


from google.cloud import bigquery
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

project_id = "bqhackathonautoianalysis"
bq_client = bigquery.Client(project=project_id)
# List of tables and expected join keys
tables = {
    "Claims": "Claims",
    "Image_Claims_Analysis": "Image_Claims_Analysis",
    "Vehicles": "Vehicles",
    "Policies": "Policies",
    "Policyholders": "Policyholders"
}

# Display schema for each table
for name, table in tables.items():
    query = f"""
    SELECT column_name, data_type, is_nullable
    FROM `{project_id}.autoAnalysis_Dataset.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table}'
    ORDER BY column_name
    """
    schema_df = bq_client.query(query).to_dataframe()
    print(f"\n=== {name.upper()} SCHEMA ===")
    display(schema_df)



# Load each table into a DataFrame
dfs = {}
for name, table in tables.items():
    query = f"SELECT * FROM `{project_id}.autoAnalysis_Dataset.{table}`"
    df = bq_client.query(query).to_dataframe()
    dfs[name] = df

# Profile each table
for name, df in dfs.items():
    print(f"\n=== {name.upper()} ===")
    display(df.head())
    print(df.info())
    print("Missing values:\n", df.isnull().sum())
    print("Unique values:\n", df.nunique())



# ==============================================================================
# --- Distribution Analysis (Presentation Ready) ---
# This cell provides a polished visual analysis of key data distributions
# for the final submission notebook.
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set a professional plot style for all subsequent charts in the notebook
sns.set_theme(style="whitegrid", palette="viridis")

# --- Analysis 1: Accident Type Distribution ---

# Get the data and prepare it for display
df_claims = dfs["Claims"]
accident_counts = df_claims["accident_type"].value_counts()

# -- Output 1.1: Beautified Table --
print("Accident Type Distribution (Table View)")
display(accident_counts.to_frame(name="Number of Claims").style
    .set_caption("Frequency of Each Accident Type")
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#440154'), ('color', 'white'), ('font-weight', 'bold')]},
        {'selector': 'caption', 'props': [('color', '#440154'), ('font-size', '1.1em'), ('font-weight', 'bold')]}
    ])
    .bar(color='#40B7AD', vmin=0)
)

# -- Output 1.2: Beautified Bar Chart --
plt.figure(figsize=(12, 8))
ax = sns.countplot(
    data=df_claims,
    y='accident_type',
    order=accident_counts.index  # Sort bars from most to least frequent
)

# Add clear data labels to the end of each bar
ax.bar_label(ax.containers[0], padding=5, fontsize=11, fontweight='bold')

# Set well-defined titles and labels
ax.set_title('Distribution of Accident Types', fontsize=16, fontweight='bold')
ax.set_xlabel('Number of Claims', fontsize=12)
ax.set_ylabel('Accident Type', fontsize=12)
ax.set_xlim(right=accident_counts.max() * 1.15) # Add padding for labels

print("\n\nAccident Type Distribution (Chart View)")
plt.tight_layout()
plt.show()


# --- Analysis 2: Vehicle Age Distribution ---

# Get the data
df_vehicles = dfs["Vehicles"]

# -- Output 2.1: Beautified Histogram --
plt.figure(figsize=(10, 6))
ax = sns.histplot(
    data=df_vehicles,
    x='vehicle_age',
    bins=25,          # Adjusted for better granularity
    kde=True,         # Adds a smooth density curve
    color="#21918c"
)

# Add a vertical line for the mean age for context
mean_age = df_vehicles['vehicle_age'].mean()
ax.axvline(mean_age, color='#fde725', linestyle='--', linewidth=2.5, label=f'Mean Age: {mean_age:.1f} years')

# Set well-defined titles and labels
ax.set_title('Distribution of Vehicle Ages', fontsize=16, fontweight='bold')
ax.set_xlabel('Vehicle Age (Years)', fontsize=12)
ax.set_ylabel('Frequency (Number of Vehicles)', fontsize=12)
ax.legend() # Display the label for the mean line

print("\n\nVehicle Age Distribution")
plt.tight_layout()
plt.show()


# Reload Claims table to get the updated schema with vin column
query = "SELECT * FROM `bqhackathonautoianalysis.autoAnalysis_Dataset.Claims`"
df_claims = client.query(query).to_dataframe()

# Verify the vin column is now available
print("Updated Claims table columns:")
print(df_claims.columns.tolist())

# Check if vin column exists
if 'vin' in df_claims.columns:
    print("âœ… vin column now exists in Claims DataFrame")
    print("Sample vin values:", df_claims['vin'].head())
else:
    print("â�Œ vin column still missing")


# Join Claims with Image_Claims_Analysis
claims_images = pd.merge(df_claims, dfs["Image_Claims_Analysis"], on="claim_id", how="left")
print("Claims with image data:", claims_images["image_uri"].notnull().sum())

# Join Claims with Policies and Vehicles
# Method 1: Use left_on and right_on for explicit column mapping
# Method 1: Use suffixes to handle conflicts
claims_policies = pd.merge(
    df_claims, 
    dfs["Policies"], 
    on=["policy_id", "vin"], 
    how="left"
)
print("After merge with suffixes:", claims_policies.columns.tolist())

# Now try the second merge
claims_full = pd.merge(
    claims_policies, 
    dfs["Vehicles"], 
    on="vin", 
    how="left"
)
print("Final columns:", claims_full.columns.tolist())
print("Claims with full policy and vehicle info:", claims_full.dropna(subset=["vin", "policyholder_id"]).shape[0])


target_cols = [
    "airbag_deployed",
    "drivable_post_accident",
    "predicted_damage_severity",
    "predicted_quote",
    "damage_location"
]

print("\nTarget Column Missingness:")
print(df_claims[target_cols].isnull().sum())



df_images = dfs["Image_Claims_Analysis"]
print("\nIdentified Make Distribution:")
display(df_images["identified_make"].value_counts())


# Derived features
df_claims["claim_delay"] = pd.to_datetime(df_claims["claim_filed_date"]) - pd.to_datetime(df_claims["accident_date"])
df_policies = dfs["Policies"]
df_policies["policy_duration"] = pd.to_datetime(df_policies["end_date"]) - pd.to_datetime(df_policies["start_date"])



# Use BigFrames instead of creating intermediate tables
import bigframes.pandas as bf

# Load all data into BigFrames
claims = bf.read_gbq("bqhackathonautoianalysis.autoAnalysis_Dataset.Claims")
images = bf.read_gbq("bqhackathonautoianalysis.autoAnalysis_Dataset.Image_Claims_Analysis")
vehicles = bf.read_gbq("bqhackathonautoianalysis.autoAnalysis_Dataset.Vehicles")

# Join without creating tables
master_df = claims.merge(images, on="claim_id").merge(vehicles, on="vin")


import bigframes.pandas as bf
from bigframes.ml.llm import GeminiTextGenerator

# ==============================================================================
# --- Session Management (CRITICAL FIX) ---
# ==============================================================================
# Close any previously active session to allow settings to be changed.
# This makes the entire cell re-runnable.
try:
    bf.close_session()
except Exception:
    pass # Fails silently if no session is active, which is fine.

# Now, set the project for the new session. This is the correct way.
bf.options.bigquery.project = "bqhackathonautoianalysis"

# bf.options.bigquery.project = "bqhackathonautoianalysis"

# 1. Load the raw images table
raw_images = bf.read_gbq("bqhackathonautoianalysis.autoAnalysis_Dataset.claim_images_raw")

# 2. Create analysis prompt
raw_images["analysis_prompt"] = (
    "You are an AI claims inspector. Analyze the given vehicle accident image. "
    "Return structured text with:\n"
    "Full response: <your description>\n"
    "Damage type: <scratch/dent/etc>\n"
    "Damage severity: <low/medium/high>\n"
    "Confidence score: <0-100>\n\n"
    "Image URI: " + raw_images["uri"]
)

# 3. Run Gemini model safely
generator = GeminiTextGenerator()

# ğŸ‘‡ This gives a new DataFrame with multiple cols, not a Series
ai_output = generator.predict(raw_images["analysis_prompt"])

# 4. Inspect what Gemini returned
print(ai_output.head())   # check columns

# 5. Assume the text is in 'ml_generate_text_result' column
raw_images["final_response"] = ai_output["ml_generate_text_llm_result"]

# 6. Extract structured fields from final_response
# Instead of .apply(lambda x: f"Extract the damage type from this text: {x}")
raw_images["damage_type_prompt"] = "Extract the damage type from this text: " + raw_images["final_response"]
raw_images["damage_severity_prompt"] = "Extract the damage severity from this text: " + raw_images["final_response"]
raw_images["confidence_prompt"] = "Provide a confidence score (0-1) for this analysis: " + raw_images["final_response"]

# Now pass these directly into Gemini
extractor = GeminiTextGenerator()
# Run Gemini and select only the text output
damage_type_output = extractor.predict(raw_images["damage_type_prompt"])
raw_images["identified_damage_type"] = damage_type_output["ml_generate_text_llm_result"]

severity_output = extractor.predict(raw_images["damage_severity_prompt"])
raw_images["identified_damage_severity"] = severity_output["ml_generate_text_llm_result"]

confidence_output = extractor.predict(raw_images["confidence_prompt"])
raw_images["confidence_score"] = confidence_output["ml_generate_text_llm_result"]

print("--- Starting Enrichment: Adding Vehicle Make and Model ---")

# 11. Load the Image_Claims_Analysis table
# We only need the key ('image_uri') and the data we want to add.
# This is more efficient than loading the whole table.
columns_to_fetch = [ 'claim_id', 'image_uri', 'identified_make', 'identified_model']
image_analysis_data = bf.read_gbq(
    "bqhackathonautoianalysis.autoAnalysis_Dataset.Image_Claims_Analysis",
    columns=columns_to_fetch
)
print("\nSuccessfully loaded make/model data. Columns:", image_analysis_data.columns)

# 12. Merge the two DataFrames
# This performs a LEFT JOIN, keeping all records from `raw_images` and adding
# matching make/model data from `image_analysis_data`.
enriched_images = raw_images.merge(
    image_analysis_data,
    left_on='uri',         # Key in the original DataFrame
    right_on='image_uri',    # Key in the new data
    how='left'             # Keep all rows from the left DataFrame
)

# 13. Clean up the final DataFrame
# The merge leaves a redundant 'image_uri' column, which we can drop.
if 'image_uri' in enriched_images.columns:
    enriched_images = enriched_images.drop(columns=['image_uri'])


# --- Verification ---
print("\n--- Final DataFrame Columns After Merge ---")
print(enriched_images.columns)

print("\n--- Sample of Final Enriched Data ---")
# Displaying the key columns plus the newly added ones to verify the merge
print(enriched_images[[
    'uri', 
    'identified_make', 
    'identified_model',
    'identified_damage_type', 
    'identified_damage_severity', 
    'confidence_score'
]].head())



import pandas as pd

# Select the key columns and take 5 random rows
preview_df = enriched_images[
    ["identified_make", "identified_model", "identified_damage_type", "identified_damage_severity"
    ]].to_pandas().sample(5, random_state=42)

# Display as a styled HTML table
preview_df.style.set_table_styles(
    [
        {"selector": "th", "props": [("background-color", "#f4f4f4"), ("font-weight", "bold")]},
        {"selector": "td", "props": [("padding", "8px"), ("border", "1px solid #ddd")]}
    ]
).set_caption("Sample of Predicted Car Damage Analysis")



# ==============================================================================
# --- Phase 2: Predicting Key Claim Metrics (Presentation Ready v3) ---
# This version fixes the KeyError and produces a polished output.
# ==============================================================================
import bigframes.pandas as bf
import warnings
import logging
import bigframes # Import the top-level package to access exceptions

# Suppress PreviewWarning and other informational logs for a clean output
warnings.filterwarnings("ignore", category=bigframes.exceptions.PreviewWarning)
logging.basicConfig(level=logging.WARNING)

print("--- Phase 2: Starting Prediction of Key Claim Metrics ---")

# Initialize the GeminiTextGenerator model
extractor = GeminiTextGenerator()

# --- Step 1: Predict Damage Location ---
enriched_images["damage_location_prompt"] = "You are an expert vehicle claims adjuster. Based on the following damage description, identify the primary location of the damage (e.g., 'Front Bumper'). Return ONLY the location as a short phrase.\n\nDescription: " + enriched_images["final_response"]
damage_location_output = extractor.predict(enriched_images["damage_location_prompt"])
enriched_images["damage_location"] = damage_location_output["ml_generate_text_llm_result"]
print("âœ… Predicted Damage Location.")

# --- Step 2: Predict Airbag & Drivability ---
enriched_images["airbag_prompt"] = "Given the accident description for a " + enriched_images["identified_make"] + " " + enriched_images["identified_model"] + ", is it likely airbags deployed? Answer ONLY with 'True' or 'False'.\n\nDescription: " + enriched_images["final_response"]
enriched_images["drivable_prompt"] = "Given the accident description, is the vehicle likely drivable? Answer ONLY with 'True' or 'False'.\n\nDescription: " + enriched_images["final_response"]
airbag_output = extractor.predict(enriched_images["airbag_prompt"])
enriched_images["airbag_deployed"] = airbag_output["ml_generate_text_llm_result"].str.contains("True", case=False)
drivable_output = extractor.predict(enriched_images["drivable_prompt"])
enriched_images["drivable_post_accident"] = drivable_output["ml_generate_text_llm_result"].str.contains("True", case=False)
print("âœ… Predicted Airbag Deployment and Drivability.")

# --- Step 3: Improved Severity Prediction and Parsing ---
enriched_images["refined_severity_prompt"] = (
    "Analyze the damage description below. Classify the severity as ONLY one of the following words: 'Low', 'Medium', or 'High'.\n\n"
    "Description: " + enriched_images["final_response"]
)
severity_output = extractor.predict(enriched_images["refined_severity_prompt"])
severity_text = severity_output["ml_generate_text_llm_result"]

# Robustly parse the output to get clean labels
regex_pattern = r'(?i)(Low|Medium|High)'
extracted_severity_df = severity_text.str.extract(regex_pattern)
# CORRECTED: Access the column using the string '0', not the integer 0.
clean_severity = extracted_severity_df['0'].str.capitalize().fillna('Medium')
enriched_images["predicted_damage_severity"] = clean_severity
print("âœ… Refined and Parsed Predicted Damage Severity.")

# --- Step 4: Predict Repair Quote ---
enriched_images["quote_prompt"] = "You are an expert auto repair estimator. For a '" + enriched_images["identified_make"] + " " + enriched_images["identified_model"] + "' with '" + enriched_images["predicted_damage_severity"] + "' damage, estimate the repair cost in USD. Return ONLY a single number.\n\nDescription: " + enriched_images["final_response"]
quote_output = extractor.predict(enriched_images["quote_prompt"])
enriched_images["predicted_quote_text"] = quote_output["ml_generate_text_llm_result"]

# Parse the numeric quote value
regex_pattern_quote = r'(\d*\.?\d+)'
extracted_df_quote = enriched_images["predicted_quote_text"].str.extract(regex_pattern_quote)
# CORRECTED: Access the column using the string '0', not the integer 0.
extracted_quote_str = extracted_df_quote['0'].fillna('0')
numeric_quote = extracted_quote_str.astype(float)
enriched_images["predicted_quote"] = numeric_quote
print("âœ… Predicted and Parsed Repair Quote.")

# --- Step 5: Create the Final DataFrame for Presentation ---
final_preview_df = enriched_images[[
    'claim_id',
    'identified_make',
    'identified_model',
    'damage_location',
    'identified_damage_type',
    'predicted_damage_severity',
    'airbag_deployed',
    'drivable_post_accident',
    'predicted_quote'
]]

# --- FINAL, CLEAN OUTPUTS FOR JUDGES ---

# Output 1: Display the final, parsed severity distribution
print("\n--- Final AI-Predicted Severity Distribution ---")
severity_counts = enriched_images["predicted_damage_severity"].value_counts().to_pandas()
print(severity_counts)

# Output 2: Display a sample of the final, enriched table
print("\n--- Sample of Final Enriched Claims Data ---")
display(final_preview_df.head(5))


# ==============================================================================
# --- Final Presentation: Polished Executive Triage Table (Top 10 Entries) ---
# This cell takes the final predictions and formats them into a clean,
# professional HTML table exactly as requested.
# ==============================================================================
import pandas as pd # Ensure pandas is imported for styling

# Fetch the top 10 rows from BigFrames into a standard pandas DataFrame.
# Using .head(10).to_pandas() is efficient as it only brings a small sample into memory.
final_pandas_df = final_preview_df.head(10).to_pandas()

# Define the color-coding function for the severity column.
def style_severity(severity):
    if severity is None:
        return ''
    # The .strip() and .lower() ensure robust matching (e.g., " High " becomes "high")
    severity = str(severity).strip().lower()
    if 'high' in severity:
        return 'background-color: #ffcccc; color: #a60000; font-weight: bold;' # Red
    elif 'medium' in severity:
        return 'background-color: #ffebcc; color: #b85c00;' # Orange
    elif 'low' in severity:
        return 'background-color: #d6f5d6; color: #006400;' # Green
    else:
        return ''

# Apply the full set of styles to create the final presentation table.
styled_df = final_pandas_df.style \
    .set_caption("Live Triage: AI-Predicted Claim Attributes (Top 10 Sample)") \
    .set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#f2f2f2'),
            ('font-weight', 'bold'),
            ('text-align', 'left'),
            ('padding', '8px'),
            ('border', '1px solid #ddd')
        ]},
        {'selector': 'td', 'props': [
            ('padding', '8px'),
            ('border', '1px solid #ddd'),
            ('text-align', 'left') # Default alignment for text
        ]},
        {'selector': 'caption', 'props': [
            ('color', 'black'),
            ('font-size', '1.3em'),
            ('font-weight', 'bold'),
            ('margin-bottom', '10px')
        ]}
    ]) \
    .format({
        "predicted_quote": "${:,.2f}",  # Format as currency
        "airbag_deployed": lambda x: 'âœ”ï¸� Yes' if x else 'â�Œ No',
        "drivable_post_accident": lambda x: 'âœ”ï¸� Yes' if x else 'â�Œ No'
    }) \
    .apply(lambda s: s.map(style_severity), subset=['predicted_damage_severity']) \
    .set_properties(**{'text-align': 'center'}, subset=['airbag_deployed', 'drivable_post_accident', 'predicted_damage_severity']) \
    .set_properties(**{'text-align': 'right'}, subset=['predicted_quote']) \
    .hide(axis='index')

# Display the final, styled table.
display(styled_df)



# ==============================================================================
# --- Validation & Summary: Final Severity Distribution (CORRECTED) ---
# This cell provides a high-level overview of the final prediction results,
# proving the success of our data refinement.
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ensure a professional plot style
sns.set_style("whitegrid")

# --- ROBUST FIX 1: NORMALIZE THE DATA ---
# Convert the generated severity to lowercase and strip any extra spaces.
# This makes the data consistent and prevents errors from minor AI variations.
final_preview_df["predicted_damage_severity"] = final_preview_df["predicted_damage_severity"].str.lower().str.strip()

# Get the final counts of each severity level from the BigFrames DataFrame
final_severity_counts = final_preview_df["predicted_damage_severity"].value_counts().to_pandas()

# --- Part 1: The Raw Numbers (For the Report) ---
print("--- Final Distribution of Predicted Damage Severity ---")
print("After refining the prompts and robustly parsing the output, the final counts are:")
print(final_severity_counts)
print("\n" + "="*50 + "\n") # Visual separator

# --- Part 2: The Executive Visualization ---
# Create a bar chart to visually represent the distribution
plt.figure(figsize=(10, 6))

# --- ROBUST FIX 2: UPDATE INDEX AND PALETTE TO MATCH NORMALIZED DATA ---
# Use lowercase for the index and palette keys to match the cleaned data.
ordered_index = pd.Index(['low', 'medium', 'high']).intersection(final_severity_counts.index)

# Ensure the palette keys are also lowercase
palette = {'low': '#90ee90', 'medium': '#ffcc99', 'high': '#ff9999'}

# Check if there is anything to plot to avoid the error completely
if not ordered_index.empty:
    sns.barplot(
        x=ordered_index,
        y=final_severity_counts.loc[ordered_index],
        palette=palette,
        edgecolor='black'
    )
    plt.title('Final Count of Claims by AI-Predicted Severity', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Damage Severity', fontsize=12)
    plt.ylabel('Number of Claims', fontsize=12)
    
    # Add data labels on top of the bars for clarity
    for index, value in enumerate(final_severity_counts.loc[ordered_index]):
        plt.text(index, value + 0.5, str(value), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.show()
else:
    print("âš ï¸� No data available to plot for severity distribution. The `ordered_index` is empty.")



# ==============================================================================
# --- Stakeholder View 1: Financial Impact Analysis (Presentation Ready v2) ---
# This version corrects the AttributeError and enhances readability.
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

# Set a professional plot style
sns.set_style("whitegrid")

# --- Data Cleaning for Consistent Plotting ---
# CORRECTED: Use .str.capitalize() which is supported by BigFrames.
# For single words like 'Low' or 'High', it works the same as .title().
enriched_images["predicted_damage_severity"] = enriched_images["predicted_damage_severity"].str.strip().str.capitalize()
enriched_images["identified_make"] = enriched_images["identified_make"].str.strip().str.capitalize()


# --- Analysis 1: Average Quote by Damage Severity ---
# Group by the now-clean severity column
avg_quote_by_severity = enriched_images.groupby("predicted_damage_severity")["predicted_quote"].mean().to_pandas().sort_values()

# Create the plot using matplotlib's axes for more control
fig, ax = plt.subplots(figsize=(10, 6))
# Ensure the order uses the same capitalization
sns.barplot(x=avg_quote_by_severity.index, y=avg_quote_by_severity.values, palette="Reds", ax=ax, order=['Low', 'Medium', 'High'])

# Add clear data labels on top of each bar
ax.bar_label(ax.containers[0], fmt='${:,.0f}', fontsize=12, fontweight='bold', padding=3)

# Set well-defined titles and labels
ax.set_title('Average Predicted Repair Quote by Damage Severity', fontsize=16, fontweight='bold')
ax.set_xlabel('AI-Predicted Damage Severity', fontsize=12)
ax.set_ylabel('Average Quote (USD)', fontsize=12)
ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # Format y-axis as currency

# Adjust layout to prevent any overlap
plt.tight_layout()
plt.show()


# --- Analysis 2: Average Quote by Vehicle Make (Top 10) ---
# Group by the now-clean make column
avg_quote_by_make = enriched_images.groupby("identified_make")["predicted_quote"].mean().to_pandas().nlargest(10).sort_values()

# Create the horizontal bar plot
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x=avg_quote_by_make.values, y=avg_quote_by_make.index, palette="Blues_r", orient='h', ax=ax)

# Add clear data labels to the right of each bar
ax.bar_label(ax.containers[0], fmt=' ${:,.0f}', fontsize=11, fontweight='bold', padding=5)

# Set well-defined titles and labels
ax.set_title('Top 10 Most Expensive Vehicle Makes to Repair', fontsize=16, fontweight='bold')
ax.set_xlabel('Average Quote (USD)', fontsize=12)
ax.set_ylabel('AI-Identified Vehicle Make', fontsize=12)
ax.xaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # Format x-axis as currency

# Ensure the x-axis has some padding so labels don't get cut off
ax.set_xlim(right=avg_quote_by_make.max() * 1.15) 

# Adjust layout to prevent any overlap
plt.tight_layout()
plt.show()


# ==============================================================================
# --- Stakeholder View 2: Operational Triage - Damage Hotspots ---
# ==============================================================================

# Calculate the frequency of each damage location
damage_location_counts = enriched_images["damage_location"].value_counts().to_pandas().nlargest(10)

plt.figure(figsize=(12, 7))
location_plot = sns.barplot(x=damage_location_counts.values, y=damage_location_counts.index, palette="viridis", orient='h')
plt.title('Top 10 Most Common Damage Locations', fontsize=16, fontweight='bold')
plt.xlabel('Number of Claims', fontsize=12)
plt.ylabel('Damage Location', fontsize=12)
plt.show()


# ==============================================================================
# --- Step 6: Save Final Output Files for Submission (Robust Version) ---
# This cell safely saves a representative sample of the key resulting DataFrames
# as CSV files, with clear error handling.
# ==============================================================================
import bigframes.pandas as bf

print("--- Saving final output files to /kaggle/working/ ---")

# Define the output file paths
enriched_claims_output_path = "/kaggle/working/FINAL_Enriched_Claims_Data_Sample.csv"
high_priority_output_path = "/kaggle/working/FINAL_High_Priority_Claims_Sample.csv"

# --- Save the main enriched DataFrame ---
try:
    # First, check if the DataFrame exists in the notebook's memory
    if 'final_preview_df' in locals():
        print("DataFrame 'final_preview_df' found. Saving a sample of up to 1000 rows...")
        
        # Use .head(1000).to_pandas() to get a sample and prevent memory errors
        final_preview_df.head(1000).to_pandas().to_csv(enriched_claims_output_path, index=False)
        
        print(f"âœ… Successfully saved enriched claims data sample to: {enriched_claims_output_path}")
    else:
        print("â�Œ Error: DataFrame 'final_preview_df' was not found. Please re-run the preceding 'Phase 2' cell.")

except Exception as e:
    print(f"\nâ�Œ An unexpected error occurred while saving 'final_preview_df'.")
    print(f"   Error details: {e}")

# --- Save the high-priority claims DataFrame ---
try:
    # Check if the high-priority DataFrame exists
    if 'high_priority_claims' in locals():
        print("\nDataFrame 'high_priority_claims' found. Saving a sample of up to 1000 rows...")
        
        # Save a sample of the high-priority claims
        high_priority_claims.head(1000).to_pandas().to_csv(high_priority_output_path, index=False)
        
        print(f"âœ… Successfully saved high-priority claims sample to: {high_priority_output_path}")
    else:
        print("\nâ�Œ Error: DataFrame 'high_priority_claims' was not found. Please re-run the preceding 'Phase 2' cell.")

except Exception as e:
    print(f"\nâ�Œ An unexpected error occurred while saving 'high_priority_claims'.")
    print(f"   Error details: {e}")

# ==============================================================================
# You can now commit the notebook. The generated CSV files will be available
# for download from the notebook's output tab.
# ==============================================================================




