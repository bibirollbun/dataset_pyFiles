import geopandas as gpd

# Load built-in dataset and extract Brazil
amazonia = gpd.read_file("/kaggle/input/geographical-boundaries-of-amazonia-by-eva-et-al/amazonia_polygons.shp")
amazonia = amazonia.to_crs("EPSG:4326")  # Ensure lat/lon

# Optional: Save for later use
amazonia.to_file("amazonia.geojson", driver="GeoJSON")


repo = 'Major-TOM'
dataset = 'Core-S2L1C-SSL4EO'



from huggingface_hub import list_repo_files

# Replace with your dataset repo name
repo_id = f"{repo}/{dataset}"

# List all files in the repo
files = list_repo_files(repo_id, repo_type="dataset")

# Filter by directory or file extension
parquet_files = [f for f in files if f.startswith("embeddings/") and f.endswith(".parquet")]

print(parquet_files[:5])  # Preview
print(f"Total Parquet files: {len(parquet_files)}")


from datasets import load_dataset
dataset = load_dataset(repo_id, split="train", streaming=True)
# Select only latitude and longitude
dataset = dataset.with_format("python")  # Ensure you can index examples


import duckdb

con = duckdb.connect()
con.execute("INSTALL spatial; LOAD spatial;")


con.execute("""
CREATE TABLE brazil_geom AS
SELECT * FROM ST_Read('amazonia.geojson');
""")


# Use a list of full URLs to all Parquet files
parquet_urls = [
    f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file}"
    for file in parquet_files
]


# Track files with rows in Brazil
matching_files = []

for url in parquet_urls:
    query = f"""
    SELECT COUNT(*) > 0 AS has_rows
    FROM read_parquet('{url}') AS p, brazil_geom AS b
    WHERE ST_Within(ST_Point(p.centre_lon, p.centre_lat), b.geom)
    """
    try:
        has_rows = con.execute(query).fetchone()[0]
        if has_rows:
            matching_files.append(url)
    except Exception as e:
        print(f"Error reading {url}: {e}")


len(matching_files)



for i, sub_file in enumerate(matching_files):

    print(i)
    
    
    query = f"""
    SELECT *
    FROM read_parquet({[sub_file]}) AS p,
         brazil_geom AS b
    WHERE ST_Within(ST_Point(p.centre_lon, p.centre_lat), b.geom)
    """
    
    # Run query and fetch results
    df = con.execute(query).fetchdf()

    df.to_parquet(f"Major-TOM-Core-S2L1C-SSL4EO-Amazonia-Filtered_Subset_{i}.parquet", index=False)


