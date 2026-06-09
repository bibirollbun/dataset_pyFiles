!pip install contextily


import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from kaggle_secrets import UserSecretsClient

# Load your GeoJSON files
known_sites = gpd.read_file("/kaggle/input/ade-geoglyphs-others-robertswalker/ancient_amazonia_archaeology.geojson")

# Load boundary data (e.g., Amazon biome, country outlines, etc.)
boundaries = gpd.read_file("/kaggle/input/geographical-boundaries-of-amazonia-by-eva-et-al/amazonia_polygons.shp")


# Reproject both to Web Mercator for contextily
known_sites_3857 = known_sites.to_crs(epsg=3857)
boundaries = boundaries.to_crs(epsg=3857)


earthwork_df = known_sites_3857[known_sites_3857['type'] == 'earthwork']
ade_df = known_sites_3857[known_sites_3857['type'] == 'ADE']
other_df = known_sites_3857[known_sites_3857['type'] == 'other']
# Plot the map with contextily basemap
fig, ax = plt.subplots(figsize=(12, 10))

# Plot boundary layer (e.g., Amazon region or political boundaries)
boundaries.plot(ax=ax, facecolor='none', edgecolor='purple', linewidth=2, label="Amazon Boundary")

# Plot your points/polygons
ade_df.plot(ax=ax, color='blue', markersize=10, alpha=0.7, label="Amazon Dark Earth")
earthwork_df.plot(ax=ax, color='red', markersize=10, alpha=0.7, label="Earthwork")
other_df.plot(ax=ax, color='gray', markersize=10, alpha=0.7, label="Other")

# Add a basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Create custom legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Amazon Dark Earth',
           markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Earthwork',
           markerfacecolor='red', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Other',
           markerfacecolor='gray', markersize=8),
    Patch(facecolor='none', edgecolor='purple', label='Amazon Boundary'),

]

ax.legend(handles=legend_elements, loc='upper right')

# Clean up
ax.set_axis_off()
plt.title("Pre-Columbian Sites in the Amazon Basin", fontsize=14)
plt.tight_layout()
plt.show()




import numpy as np
import glob
import polars as plrs
import duckdb
import pandas as pd


EMBEDDING_PATH = "/kaggle/input/major-tom-core-s2l1c-ssl4eo-amazonia-embeddings/*parquet"
parquet_files = glob.glob(EMBEDDING_PATH)
embeddings_ids_df = plrs.read_parquet(parquet_files, columns=['centre_lat', 'centre_lon', 'unique_id'])



embeddings_ids_df.head()


len(embeddings_ids_df)


from sklearn.neighbors import BallTree


# 1. Load geoglyphs
geo_points = np.radians(np.vstack([
    known_sites.geometry.y.values,
    known_sites.geometry.x.values
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
        "object_type": known_sites.iloc[i]["type"],
        "object_index": i,
        "embedding_id": embeddings_ids_df[int(min_idx)]["unique_id"][0],
        "distance_m": dist_m
    })

matched_df = pd.DataFrame(results)



# Step 1: Group by embedding_id and count unique object types
conflict_counts = matched_df.groupby("embedding_id")["object_type"].nunique()

# Step 2: Identify embedding_ids that appear with >1 class
conflicting_ids = conflict_counts[conflict_counts > 1].index

# Step 3: Filter out conflicting embedding_ids
filtered_df = matched_df[~matched_df["embedding_id"].isin(conflicting_ids)]


matched_df = filtered_df


# Step 1: Extract matched IDs
matched_ids = matched_df["embedding_id"].unique().tolist()

# Step 2: Prepare query
id_list = "(" + ", ".join(f"'{uid}'" for uid in matched_ids) + ")"

query = f"""
SELECT *
FROM read_parquet('{EMBEDDING_PATH}')
WHERE unique_id IN {id_list}
"""

# Step 3: Run DuckDB query
con = duckdb.connect()
object_embeddings_df = con.execute(query).fetchdf()

# Step 4: Merge back object_type
object_embeddings_df = object_embeddings_df.merge(
    matched_df[["embedding_id", "object_type"]],
    left_on="unique_id",
    right_on="embedding_id",
    how="left"
)

# Step 5: Get embeddings as NumPy array
object_embeddings = np.stack(object_embeddings_df["embedding"])


ade_emb_df =  object_embeddings_df[object_embeddings_df["object_type"] == "ADE"]
earthwork_emb_df =  object_embeddings_df[object_embeddings_df["object_type"] == "earthwork"]
other_emb_df =  object_embeddings_df[object_embeddings_df["object_type"] == "other"]


ade_emb = np.stack(ade_emb_df['embedding'].values)
earthwork_emb = np.stack(earthwork_emb_df['embedding'].values)
other_emb = np.stack(other_emb_df['embedding'].values)


other_labels = np.zeros(other_emb.shape[0]).reshape(-1,1).astype(int)
ade_labels = np.ones(ade_emb.shape[0]).reshape(-1,1).astype(int)
earthwork_labels = np.ones(earthwork_emb.shape[0]).reshape(-1,1).astype(int)*2


other_ids = other_emb_df['unique_id'].values.reshape(-1,1)
ade_ids = ade_emb_df['unique_id'].values.reshape(-1,1)
earthwork_ids = earthwork_emb_df['unique_id'].values.reshape(-1,1)



print(ade_emb.shape)
print(ade_labels.shape)
print(ade_ids.shape)


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


full_labels = np.vstack([other_labels, ade_labels, earthwork_labels])
full_embeddings = np.vstack([other_emb, ade_emb, earthwork_emb])
full_ids = np.vstack([other_ids, ade_ids, earthwork_ids])


# Split into train and validation
train_embeddings, val_embeddings, train_labels, val_labels, train_ids, val_ids = train_test_split(full_embeddings, full_labels, full_ids, test_size=0.2, random_state=42)



from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight(class_weight='balanced', y=train_labels)






clf = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",  # Monitor multiclass log loss
    use_label_encoder=False,
    n_estimators=100,        # Set high, early stopping will cut it off
    max_depth=3,
    learning_rate=0.01,
    random_state=42,
    verbosity=1
)

clf.fit(
    train_embeddings,
    train_labels,
    #sample_weight=sample_weights,
    eval_set=[
        (train_embeddings, train_labels),   # ← this logs training loss
        (val_embeddings, val_labels)
    ],
    early_stopping_rounds=20,
    verbose=True
)


# Get evaluation results from the underlying Booster
evals_result = clf.evals_result()



# Extract training and validation losses
train_loss = evals_result['validation_0']['mlogloss']
val_loss = evals_result['validation_1']['mlogloss']

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Boosting Round')
plt.ylabel('Log Loss')
plt.title('Training vs. Validation Log Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("Best iteration:", clf.best_iteration)
preds = clf.predict(val_embeddings, iteration_range=(0, clf.best_iteration + 1))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict class probabilities
probs = clf.predict_proba(val_embeddings)

# Convert to class labels by taking the argmax
val_preds = probs.argmax(axis=1)

cm = confusion_matrix(val_labels, val_preds)

val_labels = np.array(val_labels).ravel().astype(int)


disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42)
proj = tsne.fit_transform(train_embeddings)

plt.scatter(proj[:, 0], proj[:, 1], c=train_labels, cmap='tab5', alpha=0.7)
plt.title("t-SNE projection of embeddings")
plt.colorbar()
plt.show()


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
    probs = clf.predict_proba(embeddings)
    other_prob = probs[:, 0]
    ade_prob = probs[:, 1]
    earthwork_prob = probs[:,2]
    

    # Append score
    df = df.with_columns([
        plrs.Series("other_score", other_prob),
        plrs.Series("ade_score", ade_prob),
        plrs.Series("earthwork_score", earthwork_prob),
    ])    
    
    # Keep top-k for efficiency (optional)
    top_df = df.sort("earthwork_score", descending=True)#.head(top_k_per_file)

    # Append without large embedding column
    results.append(top_df[['centre_lat', 'centre_lon', 'unique_id', 'other_score', 'ade_score', 'earthwork_score']])

# ---- Combine all and sort globally ---- #
final_df = plrs.concat(results).sort("earthwork_score", descending=True)

# Add ranking column
final_df = final_df.with_columns(
    plrs.Series("rank", np.arange(1, len(final_df) + 1))
)

# ---- Save to disk ---- #
final_df.write_parquet("top_similar_embeddings.parquet")



final_df


import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

# ---- Load your final DataFrame ---- #
df = final_df

# ---- Extract coordinates and probabilities ---- #
lons = df["centre_lon"].to_numpy()
lats = df["centre_lat"].to_numpy()
ade_probs = df["ade_score"].to_numpy()
earthwork_probs = df["earthwork_score"].to_numpy()
other_probs = df["other_score"].to_numpy()  # Assuming this is other_score

# ---- Define raster properties ---- #
resolution_deg = 0.0201  # ~2240m at equator, adjust as needed

# Compute bounds
min_lon, max_lon = lons.min(), lons.max()
min_lat, max_lat = lats.min(), lats.max()

width = int(np.ceil((max_lon - min_lon) / resolution_deg))
height = int(np.ceil((max_lat - min_lat) / resolution_deg))

# Affine transform
transform = from_origin(min_lon, max_lat, resolution_deg, resolution_deg)




width


# ---- Helper to create raster ---- #
def make_raster(lons, lats, values):
    raster = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.int32)

    xs = ((lons - min_lon) / resolution_deg).astype(int)
    ys = ((max_lat - lats) / resolution_deg).astype(int)

    # Accumulate values
    for x, y, v in zip(xs, ys, values):
        if 0 <= x < width and 0 <= y < height:
            raster[y, x] += v
            count[y, x] += 1

    # Average in case of multiple points per pixel
    count[count == 0] = 1  # avoid division by zero
    return raster / count

# ---- Create rasters ---- #
ade_raster = make_raster(lons, lats, ade_probs)
earthwork_raster = make_raster(lons, lats, earthwork_probs)
other_raster = make_raster(lons, lats, other_probs)

# ---- Show side-by-side ---- #
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ["ADE", "Earthwork", "Other"]
rasters = [ade_raster, earthwork_raster, other_raster]

for ax, title, raster in zip(axes, titles, rasters):
    ax.imshow(raster, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f"{title} Probability")
    ax.axis('off')

plt.tight_layout()
plt.show()


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



!cp /kaggle/input/archaeoblog-amazon-geoglyphs/AmazoniaGeoglyphs_top500_XGBoost.png .

