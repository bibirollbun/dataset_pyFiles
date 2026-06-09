import numpy as np
!pip install polars
!pip install geopandas
!pip install duckdb
import glob
import polars as plrs
import geopandas as gpd
import duckdb
import pandas as pd


EMBEDDING_PATH = "/kaggle/input/major-tom-core-s2l1c-ssl4eo-amazonia-embeddings/*parquet"
parquet_files = glob.glob(EMBEDDING_PATH)
embeddings_ids_df = plrs.read_parquet(parquet_files, columns=['centre_lat', 'centre_lon', 'unique_id'])


embeddings_ids_df.head()


len(embeddings_ids_df)


geoglyph_gdf = gpd.read_file("/kaggle/input/archaeoblog-amazon-geoglyphs/geoglyph_subset.geojson")
print(len(geoglyph_gdf))


from sklearn.neighbors import BallTree

# 1. Load geoglyphs
geo_points = np.radians(np.vstack([
    geoglyph_gdf.geometry.y.values,
    geoglyph_gdf.geometry.x.values
]).T)



embedding_points = np.radians(np.vstack([
    embeddings_ids_df["centre_lat"].to_numpy(),
    embeddings_ids_df["centre_lon"].to_numpy()
]).T)

# 3. Build BallTree
tree = BallTree(embedding_points, metric='haversine')

# 4. Set radius (convert meters to radians)
radius_m = 1120
EARTH_RADIUS_M = 6_371_000
radius_rad = radius_m / EARTH_RADIUS_M

# 5. Query all neighbors within radius
indices_within_radius = tree.query_radius(geo_points, r=radius_rad, return_distance=True)


results = []

for i, (ind, dist_rad) in enumerate(zip(*indices_within_radius)):
    if len(ind) == 0:
        continue  # no embeddings within 1120m for this geoglyph

    min_idx = ind[np.argmin(dist_rad)]
    dist_m = dist_rad[np.argmin(dist_rad)] * EARTH_RADIUS_M

    results.append({
        "geoglyph_name": geoglyph_gdf.iloc[i]["Name"],
        "geoglyph_index": i,
        "embedding_id": embeddings_ids_df[int(min_idx)]["unique_id"][0],
        "distance_m": dist_m
    })

matched_df = pd.DataFrame(results)


matched_df


matched_ids = matched_df["embedding_id"].unique().tolist()
matched_ids = np.array(matched_ids)


# Prepare IDs as a DuckDB-compatible tuple list
id_list = "(" + ", ".join(f"'{uid}'" for uid in matched_ids) + ")"


# Run query
query = f"""
SELECT *
FROM read_parquet('{EMBEDDING_PATH}')
WHERE unique_id IN {id_list}
"""

con = duckdb.connect()
glyph_embeddings_df = con.execute(query).fetchdf()
glyph_embeddings = np.stack(glyph_embeddings_df["embedding"])


# Load positive IDs as a DuckDB temp table
positive_ids_df = pd.DataFrame({"unique_id": matched_ids})
con.register("positives", positive_ids_df)

# Sample 1k negatives from all Parquet files
query = f"""
    SELECT *
    FROM parquet_scan('{EMBEDDING_PATH}')
    WHERE unique_id NOT IN (SELECT unique_id FROM positives)
    USING SAMPLE RESERVOIR(1000 ROWS)
"""
neg_df = con.execute(query).fetchdf()
neg_embeddings = np.stack(neg_df["embedding"])


!pip install xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


pos_ids = glyph_embeddings_df['unique_id'].values.reshape(-1,1)
neg_ids = neg_df['unique_id'].values.reshape(-1,1)


pos_labels = np.ones(glyph_embeddings.shape[0]).reshape(-1,1)
neg_labels = np.zeros(neg_embeddings.shape[0]).reshape(-1,1)


full_labels = np.vstack([pos_labels, neg_labels])
full_embeddings = np.vstack([glyph_embeddings, neg_embeddings])
full_ids = np.vstack([pos_ids, neg_ids])


glyph_embeddings.shape


pos_ids.shape


pos_weight =  len(neg_labels) / len(pos_labels) 
print(pos_weight)


# Split into train and validation
train_embeddings, val_embeddings, train_labels, val_labels, train_ids, val_ids = train_test_split(full_embeddings, full_labels, full_ids, test_size=0.2, random_state=42)


clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.01,
    random_state=42,
    scale_pos_weight=pos_weight,
    verbosity=1
)

clf.fit(train_embeddings, train_labels)


y_pred_prob = clf.predict_proba(val_embeddings)[:, 1]
y_pred = clf.predict(val_embeddings)

print("ROC AUC:", roc_auc_score(val_labels, y_pred_prob))
print(classification_report(val_labels, y_pred))


import os
from pathlib import Path

# ---- Parameters ---- #
embedding_folder = Path(EMBEDDING_PATH)  # Folder with .parquet files
top_k_per_file = 1000  # Optional: top-N to retain per file

# ---- Store partial results ---- #
results = []

# ---- Iterate over all parquet files ---- #
for parquet_path in parquet_files:
    print(f"Processing {parquet_path}...")
    df = plrs.read_parquet(parquet_path)
    
    embeddings = np.stack(df["embedding"])

    # Predict
    glyph_probs = clf.predict_proba(embeddings)[:, 1]

    # Append score
    df = df.with_columns(plrs.Series("glyph_score", glyph_probs))
    
    # Keep top-k for efficiency (optional)
    top_df = df.sort("glyph_score", descending=True)#.head(top_k_per_file)

    # Append without large embedding column
    results.append(top_df[['centre_lat', 'centre_lon', 'unique_id', 'glyph_score']])

# ---- Combine all and sort globally ---- #
final_df = plrs.concat(results).sort("glyph_score", descending=True)

# Add ranking column
final_df = final_df.with_columns(
    plrs.Series("rank", np.arange(1, len(final_df) + 1))
)

# ---- Save to disk ---- #
final_df.write_parquet("top_similar_embeddings.parquet")


final_df


pos_val_ids = val_ids[val_labels == 1]
pos_train_ids = train_ids[train_labels == 1]


train_ids_set = set(pos_train_ids)  # Faster lookups

train_ids_ranks_df = final_df.filter(plrs.col("unique_id").is_in(train_ids_set))


val_ids_set = set(pos_val_ids)  # Faster lookups

val_ids_ranks_df = final_df.filter(plrs.col("unique_id").is_in(val_ids_set))


train_ids_ranks_df.head()


train_ids_ranks_df['rank'].mean()
val_ids_ranks_df.head()
val_ids_ranks_df['rank'].mean()



# Remove training and validation points
final_df = final_df.filter(
    ~plrs.col("unique_id").is_in(matched_ids)
)


from shapely.geometry import Point

# Select top 500 rows
top_500_df = final_df.sort("glyph_score", descending=True).head(500)

# Convert to Pandas (if you're using Polars)
top_500_pd = top_500_df.to_pandas()

# Create Point geometry from lat/lon
top_500_pd["geometry"] = top_500_pd.apply(
    lambda row: Point(row["centre_lon"], row["centre_lat"]), axis=1
)

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(top_500_pd, geometry="geometry", crs="EPSG:4326")

# Save to GeoJSON
gdf.to_file("top_500_similar.geojson", driver="GeoJSON")

