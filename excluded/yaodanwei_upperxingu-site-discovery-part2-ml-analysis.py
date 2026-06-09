!pip install geopandas rasterio contextily cartopy shapely --quiet
!pip install seaborn --quiet


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
import numpy as np
import contextily as ctx
from shapely.geometry import Point
from rasterio.plot import show
from rasterio.enums import Resampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



train_df = pd.read_csv('/kaggle/input/upperxingu-train-and-test-dataset/danwei_meaned_training_data.csv') #2073 rows
test_df = pd.read_csv('/kaggle/input/upperxingu-train-and-test-dataset/dw_cleaned_test.csv') # 48



X = train_df.drop(columns = ['longitude', 'latitude', 'label','type'])
y = train_df['type']


geometry = [Point(xy) for xy in zip(train_df['longitude'],train_df['latitude'])]
gdf = gpd.GeoDataFrame(train_df, geometry=geometry, crs="EPSG:4326")
bbox = {
    "south": -13.0,
    "north": -12.0,
    "west": -53.5,
    "east": -52.5
}

xingu = train_df[
    (train_df.latitude >= bbox['south']) & (train_df.latitude <= bbox['north']) &
    (train_df.longitude >= bbox['west']) & (train_df.longitude <= bbox['east'])
]
gdf = gpd.GeoDataFrame(xingu, geometry=gpd.points_from_xy(xingu.longitude, xingu.latitude), crs="EPSG:4326")


API_KEY = "f98502a1d20f17dac51e485a21be3d81"
import requests
import os 
bbox = {
    "west": -54.2,
    "south": -13,
    "east": -52.5,
    "north": -10.5,
}

# Construct the URL for Copernicus DEM (30m resolution)
url = (
    "https://portal.opentopography.org/API/globaldem?"
    "demtype=COP30"
    f"&south={bbox['south']}&north={bbox['north']}"
    f"&west={bbox['west']}&east={bbox['east']}"
    "&outputFormat=GTiff"
    f"&API_Key={API_KEY}"
)

# Make the request and save the file
print("ğŸ“¡ Requesting DEM tile from OpenTopography...")
resp = requests.get(url, stream=True)

if resp.status_code != 200:
    raise Exception("DEM request failed: " + resp.text)

from rasterio.io import MemoryFile

with MemoryFile(resp.content) as memfile:
    with memfile.open() as src:
        elevation = src.read(1).astype(float)
        elevation[elevation == src.nodata] = np.nan
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
# Save to Desktop
# output_path = os.path.expanduser("figures/upper_xingu_dem.tif")
# with open(output_path, "wb") as f:
#     for chunk in resp.iter_content(1024):
#         f.write(chunk)

# print(f"DEM saved to: {output_path}")


# dem_path = "/Users/hereagain/Desktop/upper_xingu_dem.tif"
# with rasterio.open(dem_path) as src:
#     elevation = src.read(1).astype(float)
#     elevation[elevation == src.nodata] = np.nan
#     bounds = src.bounds
#     extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

from matplotlib.colors import LightSource
ls = LightSource(azdeg=315, altdeg=45)
shaded = ls.shade(elevation, cmap=plt.cm.terrain, blend_mode='overlay', vert_exag=0.5)

# === Prepare the plot ===
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
ax.imshow(shaded, extent=extent, zorder=0)

# === Overlay archaeological site points ===
colors = {"ADE": "#2171b5",         # deep blue
          "earthwork": "#de2d26",   # strong red
          "other": "#fdae6b"}       # soft orange (better contrast than gray)
labels = {"ADE": "ADE", "earthwork": "Earthwork", "other": "Other"}

for stype in xingu["type"].unique():
    subset = xingu[xingu["type"] == stype]
    ax.scatter(
        subset["longitude"], subset["latitude"],
        color=colors[stype], label=labels[stype],
        s=40, alpha=0.9, edgecolor="white", linewidth=0.5, zorder=3
    )

# === Formatting and labels ===
ax.set_title("Known Upper Xingu Archaeological Sites", fontsize=16)
ax.set_xlabel("Longitude", fontsize=13)
ax.set_ylabel("Latitude", fontsize=13)
ax.tick_params(labelsize=11)

# === Legend ===
legend = ax.legend(title="Site Type", loc="lower left", fontsize=11, title_fontsize=12, frameon=False)
for handle in legend.legendHandles:
    handle.set_sizes([60])  # Increase dot size in legend

output_dir = "/Users/hereagain/Desktop/OpenAItoZ/figures"
plt.tight_layout()
#plt.savefig(f"{output_dir}/upper_xingu_known_sites.pdf", dpi=600, bbox_inches='tight')
#plt.savefig(f"{output_dir}/upper_xingu_known_sites.png", dpi=600, bbox_inches='tight')
plt.show()



from scipy.interpolate import griddata
newbbox = {
    "west": -80.0,
    "east": -45.0,
    "south": -20.0,
    "north": 5.0
}
num_points = 500  # increase for smoother interpolation
grid_lon = np.linspace(newbbox["west"], newbbox["east"], num_points)
grid_lat = np.linspace(newbbox["south"], newbbox["north"], num_points)
grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)


points = train_df[["longitude", "latitude"]].values
values = train_df["distriver1"].values

grid_z = griddata(points, values, (grid_x, grid_y), method="cubic")
fig, ax = plt.subplots(figsize=(10, 8))

# Plot interpolated heatmap
im = ax.imshow(
    grid_z,
    extent=[newbbox["west"], newbbox["east"], newbbox["south"], newbbox["north"]],
    origin="lower",
    cmap="viridis",
    alpha=0.7,
    zorder=1
)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Distance to Major River (km)", fontsize=12)

# Site type styling
colors = {
    "ADE": "#1f78b4",       # bold blue
    "earthwork": "#e31a1c", # vivid red
    "other": "#cccccc"      # soft gray
}

# Overlay site markers
for stype in train_df["type"].unique():
    subset = train_df[train_df["type"] == stype]
    ax.scatter(
        subset["longitude"], subset["latitude"],
        color=colors.get(stype, "gray"),
        label=stype.capitalize(),
        s=55, alpha=0.95,
        edgecolor="black", linewidth=0.4,
        zorder=2
    )

# Axes and labels
ax.set_title("Proximity to Major Rivers Across Amazonian Sites", fontsize=15, weight="bold")
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)

# Legend
ax.legend(title="Site Type", loc="lower left", fontsize=10, title_fontsize=11, frameon=True)

# Ticks and aesthetics
ax.tick_params(labelsize=10)
ax.set_aspect("equal", adjustable="box")

# Save as high-res
plt.tight_layout()
#plt.savefig(f"{output_dir}/amazon_dist_to_river_map.png", dpi=600, bbox_inches="tight")
#plt.savefig(f"{output_dir}/amazon_dist_to_river_map.pdf", dpi=600, bbox_inches="tight")
plt.show()
plt.close(fig)


newbbox = {
    "west": -80.0,
    "east": -45.0,
    "south": -20.0,
    "north": 5.0
}

# Grid for interpolation
num_points = 500
grid_lon = np.linspace(newbbox["west"], newbbox["east"], num_points)
grid_lat = np.linspace(newbbox["south"], newbbox["north"], num_points)
grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

# Points and elevation values
points = train_df[["longitude", "latitude"]].values
values = train_df["lidar_elevation_value"].values 
mask = ~np.isnan(values)
values = values[mask]
points = points[mask]
grid_z = griddata(points, values, (grid_x, grid_y), method="cubic")
vmin = np.nanpercentile(grid_z, 2)
vmax = np.nanpercentile(grid_z, 98)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Heatmap
im = ax.imshow(
    grid_z,
    extent=[newbbox["west"], newbbox["east"], newbbox["south"], newbbox["north"]],
    origin="lower",
    cmap="terrain",  # You can try 'viridis', 'gist_earth', or 'Greens' too
    alpha=0.7,
    zorder=1,
    vmin = vmin,
    vmax = vmax)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Elevation (m)", fontsize=12)

# Colors for site types
colors = {
    "ADE": "#1f78b4",
    "earthwork": "#e31a1c",
    "other": "#cccccc"
}

# Overlay sites
for stype in train_df["type"].unique():
    subset = train_df[train_df["type"] == stype]
    ax.scatter(
        subset["longitude"], subset["latitude"],
        color=colors.get(stype, "gray"),
        label=stype.capitalize(),
        s=55, alpha=0.95,
        edgecolor="black", linewidth=0.4,
        zorder=2
    )

# Labels and title
ax.set_title("LiDAR Elevation Across Amazonian Sites", fontsize=15, weight="bold")
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
ax.legend(title="Site Type", loc="lower left", fontsize=10, title_fontsize=11, frameon=True)

ax.tick_params(labelsize=10)
ax.set_aspect("equal", adjustable="box")
plt.tight_layout()
#plt.savefig("figures/amazon_elev_map.png", dpi=600, bbox_inches="tight")
#plt.savefig(f"figures/amazon_elev_map.pdf", dpi=600, bbox_inches="tight")
plt.show()
plt.close(fig)


sns.set_theme(style="white", font_scale=1.2)


points = train_df[["longitude", "latitude"]].values
values = train_df["bio19"].values


grid_z = griddata(points, values, (grid_x, grid_y), method="cubic")

fig, ax = plt.subplots(figsize=(10, 8))

# Interpolated background
im = ax.imshow(
    grid_z,
    extent=[newbbox["west"], newbbox["east"], newbbox["south"], newbbox["north"]],
    origin="lower",
    cmap="YlGnBu",
    alpha=0.7,
    zorder=1
)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Bio19: Precipitation of Coldest Quarter (mm)", fontsize=12)

# Site type colors
colors = {
    "ADE": "#2171b5",       # dark blue
    "earthwork": "#de2d26", # red
    "other": "#fdae6b"      # light orange
}

# Overlay sites
for stype in train_df["type"].unique():
    subset = train_df[train_df["type"] == stype]
    ax.scatter(
        subset["longitude"], subset["latitude"],
        s=55, alpha=0.95,
        color=colors.get(stype, "gray"),
        label=stype.capitalize(),
        edgecolor="black", linewidth=0.4,
        zorder=2
    )

# Title and labels
ax.set_title("Precipitation in Coldest Quarter (Bio19) Across Amazonian Sites", fontsize=14, weight='bold')
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)

# Legend
ax.legend(title="Site Type", loc="lower left", fontsize=10, title_fontsize=11, frameon=True)

# Final layout
ax.set_aspect("equal", adjustable="box")
plt.tight_layout()

# Save high-resolution
#plt.savefig(f"{output_dir}/bio19_precip_sites_map.png", dpi=600, bbox_inches="tight")
#plt.savefig(f"{output_dir}/bio19_precip_sites_map.pdf", dpi=600, bbox_inches="tight")
plt.show()



# Define the color palette consistent with your maps
palette = {
    "ADE": "#2171b5",       # deep blue
    "earthwork": "#de2d26", # red
    "other": "#fdae6b"      # soft orange
}

# Order for categorical x-axis (optional but improves consistency)
order = ["ADE", "earthwork", "other"]

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(
    x="type", y="bio7", data=train_df,
    palette=palette, order=order,
    width=0.6, linewidth=1.2, fliersize=2, ax=ax
)

# Axis labels and title
ax.set_title("Temperature Annual Range (Bio7) by Site Type", fontsize=14, weight='bold')
ax.set_xlabel("Site Type", fontsize=12)
ax.set_ylabel("Bio7: Temperature Annual Range (Â°C)", fontsize=12)

# Improve layout and save
plt.tight_layout()
#plt.savefig(f"{output_dir}/bio7_by_site_type_boxplot.png", dpi=600, bbox_inches="tight")
#plt.savefig(f"{output_dir}/bio7_by_site_type_boxplot.pdf", dpi=600, bbox_inches="tight")
plt.show()



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


y.value_counts()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)



X_train.isna().sum(), X_test.isna().sum(), y_train.isna().sum(), y_test.isna().sum()


from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


param_grid = {
    'n_estimators': [200, 250,300],
    'max_depth': [10, 15, 20]
}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
# rand_search = RandomizedSearchCV(rf, 
#                                  param_distributions = param_dist, 
#                                  n_iter=5, 
#                                  cv=5)

rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=1,  # use all cores
                           verbose=1)
# Fit the random search object to the data
grid_search.fit(X_train, y_train)


best_rf = grid_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  grid_search.best_params_) # 15,10, 300
# import joblib

# joblib.dump(best_rf, "dwbest_rf_model.pkl")


from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)

# Classification report
print("ğŸ”� Classification Report (Test Set):")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("ğŸ§© Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

class_names = ['ADE', 'earthwork', 'other']  # make sure this matches your actual labels
y_test_bin = label_binarize(y_test, classes=class_names)

auc_macro = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
auc_weighted = roc_auc_score(y_test_bin, y_proba, average='weighted', multi_class='ovr')

print(f"Macro AUC: {auc_macro:.3f}") # 0.978
print(f"Weighted AUC: {auc_weighted:.3f}") #0.980




importances = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    
    'importance': importances
}).sort_values(by='importance', ascending=False)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.xlabel("Importance")
plt.title("Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



import joblib
#best_rf = joblib.load("dwbest_rf_model.pkl")
test_x = test_df.drop(columns=['longitude', 'latitude'])
y_pred = best_rf.predict(test_x)
y_proba = best_rf.predict_proba(test_x)



class_names = best_rf.classes_

proba_df = pd.DataFrame(y_proba, columns=[f"prob_{cls}" for cls in class_names])

result_df = pd.DataFrame({
    "longitude": test_df["latitude"].values,
    "latitude": test_df["longitude"].values,
    "predicted_type": y_pred
})

# Combine
final_df = pd.concat([result_df, proba_df], axis=1)



final_df.head()


final_df['predicted_type'].value_counts()


API_KEY = "f98502a1d20f17dac51e485a21be3d81"
import requests
import os 
bbox = {
    "west": -56,
    "south": -13.5,
    "east": -52.5,
    "north": -10.5,
}

# Construct the URL for Copernicus DEM (30m resolution)
url = (
    "https://portal.opentopography.org/API/globaldem?"
    "demtype=COP30"
    f"&south={bbox['south']}&north={bbox['north']}"
    f"&west={bbox['west']}&east={bbox['east']}"
    "&outputFormat=GTiff"
    f"&API_Key={API_KEY}"
)

# Make the request and save the file
print("ğŸ“¡ Requesting DEM tile from OpenTopography...")
resp = requests.get(url, stream=True)

if resp.status_code != 200:
    raise Exception("DEM request failed: " + resp.text)

from rasterio.io import MemoryFile

with MemoryFile(resp.content) as memfile:
    with memfile.open() as src:
        elevation = src.read(1).astype(float)
        elevation[elevation == src.nodata] = np.nan
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]






from matplotlib.colors import LightSource
ls = LightSource(azdeg=315, altdeg=45)
shaded = ls.shade(elevation, cmap=plt.cm.terrain, blend_mode='overlay', vert_exag=0.5)

# === Create the plot ===
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# Plot shaded elevation background
ax.imshow(shaded, extent=extent, zorder=0)

# Overlay ADE probability predictions
scatter = ax.scatter(
    final_df["longitude"],
    final_df["latitude"],
    c=final_df["prob_ADE"],
    cmap="Blues",
    s=60,
    edgecolor="black",
    linewidth=0.3,
    alpha=0.95,
    zorder=2
)

# Title and labels
ax.set_title("Predicted Probability of ADE Sites", fontsize=16)
ax.set_xlabel("Longitude", fontsize=13)
ax.set_ylabel("Latitude", fontsize=13)
ax.tick_params(labelsize=11)

# Set axis limits to match background extent
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])

# Colorbar
cbar = plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Predicted Probability (ADE)", fontsize=12)

plt.tight_layout()
# Save if needed
# plt.savefig(f"{output_dir}/ADE_probability_map.png", dpi=600, bbox_inches='tight')
plt.show()



# ls = LightSource(azdeg=315, altdeg=45)
# shaded = ls.shade(elevation, cmap=plt.cm.terrain, blend_mode='overlay', vert_exag=0.5)

# # === Create the plot ===
# fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# # Plot shaded elevation background
# ax.imshow(shaded, extent=extent, zorder=0)

# # Overlay ADE probability predictions
# scatter = ax.scatter(
#     final_df["longitude"],
#     final_df["latitude"],
#     c=final_df["prob_earthwork"],
#     cmap="Reds",
#     s=60,
#     edgecolor="black",
#     linewidth=0.3,
#     alpha=0.95,
#     zorder=2
# )

# # Title and labels
# ax.set_title("Predicted Probability of Earthwork Sites", fontsize=16)
# ax.set_xlabel("Longitude", fontsize=13)
# ax.set_ylabel("Latitude", fontsize=13)
# ax.tick_params(labelsize=11)

# # Set axis limits to match background extent
# ax.set_xlim(extent[0], extent[1])
# ax.set_ylim(extent[2], extent[3])

# # Colorbar
# cbar = plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
# cbar.set_label("Predicted Probability (Earthwork)", fontsize=12)

# plt.tight_layout()
# # Save if needed
# #plt.savefig("earthwork_probability_map.png", dpi=600, bbox_inches='tight')
# plt.show()
# plt.close(fig)


# ls = LightSource(azdeg=315, altdeg=45)
# shaded = ls.shade(elevation, cmap=plt.cm.terrain, blend_mode='overlay', vert_exag=0.5)

# # === Create the plot ===
# fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# # Plot shaded elevation background
# ax.imshow(shaded, extent=extent, zorder=0)

# # Overlay ADE probability predictions
# scatter = ax.scatter(
#     final_df["longitude"],
#     final_df["latitude"],
#     c=final_df["prob_other"],
#     cmap="Oranges",
#     s=60,
#     edgecolor="black",
#     linewidth=0.3,
#     alpha=0.95,
#     zorder=2
# )

# # Title and labels
# ax.set_title("Predicted Probability of Other Sites", fontsize=16)
# ax.set_xlabel("Longitude", fontsize=13)
# ax.set_ylabel("Latitude", fontsize=13)
# ax.tick_params(labelsize=11)

# # Set axis limits to match background extent
# ax.set_xlim(extent[0], extent[1])
# ax.set_ylim(extent[2], extent[3])

# # Colorbar
# cbar = plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
# cbar.set_label("Predicted Probability (Other)", fontsize=12)

# plt.tight_layout()
# # Save if needed
# plt.savefig("other_probability_map.png", dpi=600, bbox_inches='tight')
# plt.show()


top_5_ade_sites = final_df[["latitude", "longitude", "prob_ADE"]].sort_values(
    by="prob_ADE", ascending=False).head(5).reset_index(drop=True)

top_5_ade_sites

