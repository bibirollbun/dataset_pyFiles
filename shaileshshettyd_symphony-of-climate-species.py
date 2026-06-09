#Import Libraries
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import torch
import plotly.express as px


train_meta = pd.read_csv("/kaggle/input/geolifeclef-2025/GLC25_PA_metadata_train.csv")
test_meta = pd.read_csv("/kaggle/input/geolifeclef-2025/GLC25_PA_metadata_test.csv")
print("Train shape:", train_meta.shape, " | Test shape:", test_meta.shape)


train_meta.head()


train_meta['geometry'] = train_meta.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
test_meta['geometry'] = test_meta.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

train_gdf = gpd.GeoDataFrame(train_meta, geometry='geometry', crs='EPSG:4326')
test_gdf = gpd.GeoDataFrame(test_meta, geometry='geometry', crs='EPSG:4326')

plt.figure(figsize=(12, 6))
plt.scatter(train_gdf['lon'], train_gdf['lat'], s=1, label='Train', alpha=0.5, color='green')
plt.scatter(test_gdf['lon'], test_gdf['lat'], s=1, label='Test', alpha=0.5, color='red')
plt.legend()
plt.title("Train vs Test Geospatial Distribution")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()


missing_train = train_meta.isnull().sum()
missing_train[missing_train > 0].sort_values(ascending=False)


species_count = train_meta['speciesId'].value_counts()
species_count_cleaned = species_count.replace([np.inf, -np.inf], np.nan).dropna()

plt.figure(figsize=(14, 4))
sns.histplot(species_count_cleaned.values, bins=100, kde=True, color="blue")
plt.title("Species Frequency Distribution")
plt.xlabel("Number of Observations per Species")
plt.ylabel("Species Count")
plt.grid(True)
plt.show()


bioclim = pd.read_csv("/kaggle/input/geolifeclef-2025/EnvironmentalValues/ClimateAverage_1981-2010/GLC25-PA-test-bioclimatic.csv")
print("Bioclim shape:", bioclim.shape)
bioclim.describe().T.style.background_gradient(cmap="YlGnBu")


# Define the sample ID and cube path
sample_id = 1000012
cube_path = "/kaggle/input/geolifeclef-2025/SateliteTimeSeries-Landsat/cubes/PA-train/GLC25-PA-train-landsat-time-series_1000012_cube.pt"

# Safely load the Landsat cube
try:
    landsat_cube = torch.load(cube_path, weights_only=True)
except TypeError:
    landsat_cube = torch.load(cube_path)  # fallback for older PyTorch versions

print("Cube shape (bands, quarters, years):", landsat_cube.shape)

# Flatten and visualize the time series for each band
plt.figure(figsize=(14, 5))
bands = ['Red', 'Green', 'Blue', 'NIR', 'SWIR1', 'SWIR2']
for i in range(6):
    plt.plot(landsat_cube[i].flatten().numpy(), label=bands[i])

plt.title(f"Landsat Time Series for Survey ID: {sample_id}")
plt.xlabel("Time Steps (Quarters Ã— Years)")
plt.ylabel("Reflectance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


bioclim_csv = pd.read_csv("/kaggle/input/geolifeclef-2025/BioclimTimeSeries/values/GLC25-PA-test-bioclimatic_monthly.csv")

# Pick a survey ID
survey_id = 1001615
sample_df = bioclim_csv[bioclim_csv["surveyId"] == survey_id].drop(columns=["surveyId"])

# Reshape to long format
long_df = sample_df.T.reset_index()
long_df.columns = ["feature_time", "value"]

# Extract metadata
long_df[["feature", "month", "year"]] = long_df["feature_time"].str.extract(r'Bio-(\w+)_([0-9]+)_([0-9]+)')
long_df["date"] = pd.to_datetime(dict(year=long_df["year"].astype(int),
                                      month=long_df["month"].astype(int),
                                      day=1))

# Plotly interactive line plot
fig = px.line(
    long_df,
    x="date",
    y="value",
    color="feature",
    title=f"ğŸŒ¿ Bioclimatic Features Over Time (Survey ID: {survey_id})",
    labels={"date": "Date", "value": "Feature Value"},
    template="plotly_dark"
)

fig.update_layout(
    legend=dict(title="Feature", orientation="v", x=1.01, y=1),
    margin=dict(l=60, r=150, t=60, b=60),
    height=600
)

fig.show()


# Load metadatasets
meta = pd.read_csv("/kaggle/input/geolifeclef-2025/GLC25_PA_metadata_train.csv")
elev = pd.read_csv("/kaggle/input/geolifeclef-2025/EnvironmentalValues/Elevation/GLC25-PA-train-elevation.csv")
foot = pd.read_csv("/kaggle/input/geolifeclef-2025/EnvironmentalValues/HumanFootprint/GLC25-PA-train-human_footprint.csv")

# Preprocess metadata: keep unique surveys
meta = meta.drop_duplicates(subset="surveyId")

# Rename elevation column if necessary
if "value" in elev.columns:
    elev = elev.rename(columns={"value": "Elevation"})

# Choose the footprint feature to visualize
selected_footprint = "HumanFootprint-building-residential"  # â†� swap to any other as needed

# Reduce footprint to just selected feature
foot = foot[["surveyId", selected_footprint]].rename(columns={selected_footprint: "FootprintFeature"})

# Merge everything together
df = meta.merge(elev, on="surveyId", how="left").merge(foot, on="surveyId", how="left")

# Drop missing values
df = df.dropna(subset=["Elevation", "FootprintFeature", "lat"])

# Plotly 3D scatter
fig = px.scatter_3d(
    df,
    x="FootprintFeature",
    y="Elevation",
    z="lat",
    color="region",
    hover_data=["country", "year", "lon", "lat", "speciesId"],
    title=f"ğŸŒ� 3D Landscape View: {selected_footprint.split('-')[-1].capitalize()} vs Elevation vs Latitude",
    opacity=0.75,
    height=700
)

fig.update_layout(scene=dict(
    xaxis_title=selected_footprint.split('-')[-1].capitalize() + " (%)",
    yaxis_title="Elevation (m)",
    zaxis_title="Latitude"
))

fig.show()


# Clean column names
elev = elev.rename(columns={'value': 'Elevation'}) if 'value' in elev.columns else elev
foot = foot.rename(columns={'value': 'HumanFootprint'}) if 'value' in foot.columns else foot

# Merge on unique surveyId level
meta_unique = meta.drop_duplicates("surveyId")
df = meta_unique.merge(elev, on="surveyId", how="left").merge(foot, on="surveyId", how="left")


heatmap = px.density_mapbox(
    df, lat="lat", lon="lon", radius=5,
    center={"lat": 47, "lon": 10}, zoom=3,
    mapbox_style="carto-positron",
    title="ğŸŒ� Survey Site Density Heatmap"
)
heatmap.show()


species_rich = meta.groupby("surveyId")["speciesId"].nunique().reset_index()
species_rich = species_rich.merge(df[["surveyId", "lat", "lon"]], on="surveyId")

bubble_map = px.scatter_mapbox(
    species_rich, lat="lat", lon="lon", size="speciesId",
    color="speciesId", zoom=3, size_max=10,
    mapbox_style="open-street-map",
    title="ğŸ§ª Species Richness per Survey Location",
    labels={"speciesId": "Species Count"}
)
bubble_map.show()


lat_div = meta.groupby("lat")["speciesId"].nunique().reset_index()
lat_div.columns = ["Latitude", "UniqueSpecies"]

lat_plot = px.line(
    lat_div, x="Latitude", y="UniqueSpecies",
    title="ğŸ“Š Species Diversity by Latitude",
    labels={"UniqueSpecies": "Unique Species Count"}
)
lat_plot.show()


print("ğŸ”¢ Total unique surveyIds:", meta["surveyId"].nunique())
print("ğŸ§¬ Total unique species:", meta["speciesId"].nunique())
print("ğŸŒ� Geographic Range:")
print(f"    Latitude: {meta['lat'].min():.2f} â†’ {meta['lat'].max():.2f}")
print(f"    Longitude: {meta['lon'].min():.2f} â†’ {meta['lon'].max():.2f}")

