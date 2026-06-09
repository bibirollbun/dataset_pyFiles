import os
import zipfile
import pandas as pd

# --- SMART DATA LOADING ---

# 1. Define the file paths
# Kaggle datasets usually live here:
KAGGLE_INPUT_DIR = '/kaggle/input/nyc-taxi-trip-duration/'
# We must extract to the "Working" directory because Input is read-only
KAGGLE_WORK_DIR = '/kaggle/working/'
LOCAL_DIR = './'

# 2. Check where we are (Kaggle or Local)
if os.path.exists(KAGGLE_INPUT_DIR):
    print("ğŸŒ� Environment: Kaggle Cloud")
    base_path = KAGGLE_INPUT_DIR
    work_path = KAGGLE_WORK_DIR
else:
    print("ğŸ’» Environment: Local Machine")
    base_path = LOCAL_DIR
    work_path = LOCAL_DIR

# 3. Handle Zip Files automatically
target_file = 'train.csv'

# If the csv isn't here, check for a zip and extract it
if not os.path.exists(os.path.join(work_path, target_file)):
    zip_path = os.path.join(base_path, 'train.zip')
    
    if os.path.exists(zip_path):
        print(f"ğŸ“¦ Found zip file: {zip_path}")
        print("   Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(work_path)
        print("   Extraction Complete!")
    else:
        # If no zip, maybe the csv is already in the input folder (unzipped by Kaggle)
        if os.path.exists(os.path.join(base_path, target_file)):
            work_path = base_path

# 4. Load Data
full_path = os.path.join(work_path, target_file)
print(f"ğŸ“‚ Loading data from: {full_path}")
df = pd.read_csv(full_path)
print("âœ… Data Loaded Successfully")


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df.sample(5)


df.columns


df.describe().T


df.info()


df.isnull().sum()


df['trip_duration_minutes'] = df['trip_duration']/60


# calculating the distance between the pickup and dropoff coordination:

def manhattan_distance_vectorized(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    # Convert to radians (works on Series)
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)

    # 1. Calculate North-South distance (change in latitude only)
    dlat = lat2 - lat1
    a1 = np.sin(dlat/2)**2
    c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1-a1))
    dist_ns = R * c1

    # 2. Calculate East-West distance (change in longitude only)
    # Use average latitude for the calculation
    avg_lat = (lat1 + lat2) / 2
    dlon = lon2 - lon1
    a2 = np.cos(avg_lat)**2 * np.sin(dlon/2)**2
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1-a2))
    dist_ew = R * c2

    # Manhattan distance is the sum
    return dist_ns + dist_ew


df['manhattan_dist_km'] = manhattan_distance_vectorized(
    df['pickup_latitude'], 
    df['pickup_longitude'], 
    df['dropoff_latitude'], 
    df['dropoff_longitude']
)


df['avg_velocity_km/hr'] = df['manhattan_dist_km']/(df['trip_duration_minutes']/60)


df.describe().T


plt.figure(figsize=(5,3))
plt.subplot(1,2,1)
sns.boxplot(data=df, y='trip_duration_minutes', color='darkred')

plt.subplot(1,2,2)
sns.boxplot(data=df, y='avg_velocity_km/hr', color='darkred')

plt.tight_layout()
plt.show()


df[df['trip_duration_minutes']>120].sort_values(by='manhattan_dist_km', ascending=False)


df[df['avg_velocity_km/hr']>120].sort_values(by='avg_velocity_km/hr', ascending=False)


df[df['manhattan_dist_km']>120].sort_values(by='manhattan_dist_km', ascending=False)


# just in new york

xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]

df = df[(df.pickup_longitude> xlim[0]) & (df.pickup_longitude < xlim[1])]
df = df[(df.dropoff_longitude> xlim[0]) & (df.dropoff_longitude < xlim[1])]
df = df[(df.pickup_latitude> ylim[0]) & (df.pickup_latitude < ylim[1])]
df = df[(df.dropoff_latitude> ylim[0]) & (df.dropoff_latitude < ylim[1])]


df[df['manhattan_dist_km']>120].sort_values(by='manhattan_dist_km', ascending=False)


plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
sns.histplot(data=df, x='manhattan_dist_km', bins=20, kde=True, color='darkred')

plt.subplot(1,2,2)
sns.boxplot(data=df, y='manhattan_dist_km', color='darkred')

plt.tight_layout()
plt.show()


plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
sns.histplot(data=df, x='avg_velocity_km/hr', bins=20, kde=True, color='darkred')

plt.subplot(1,2,2)
sns.boxplot(data=df, y='avg_velocity_km/hr', color='darkred')

plt.tight_layout()
plt.show()


df[df['avg_velocity_km/hr']==0].sort_values(by='avg_velocity_km/hr', ascending=False)


df = df[df['avg_velocity_km/hr']>0]


df.head()


df.shape


df[df['avg_velocity_km/hr']>100].sort_values(by='avg_velocity_km/hr', ascending=False)


df = df[df['avg_velocity_km/hr'] <= 90]


df.shape


plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
sns.histplot(data=df, x='avg_velocity_km/hr', bins=20, kde=True, color='darkred')

plt.subplot(1,2,2)
sns.boxplot(data=df, y='avg_velocity_km/hr', color='darkred')

plt.tight_layout()
plt.show()


plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
sns.histplot(data=df, x='trip_duration_minutes', bins=20, kde=True, color='darkred')

plt.subplot(1,2,2)
sns.boxplot(data=df, y='trip_duration_minutes', color='darkred')

plt.tight_layout()
plt.show()


df[df['trip_duration_minutes']>150].sort_values(by='avg_velocity_km/hr', ascending=False)


df = df[df['trip_duration_minutes']<=120]


df.shape


plt.figure(figsize=(6,6))

plt.subplot(3,2,1)
sns.histplot(data=df, x='manhattan_dist_km', bins=20, kde=True, color='darkred')

plt.subplot(3,2,2)
sns.boxplot(data=df, y='manhattan_dist_km', color='darkred')

plt.subplot(3,2,3)
sns.histplot(data=df, x='trip_duration_minutes', bins=20, kde=True, color='darkred')

plt.subplot(3,2,4)
sns.boxplot(data=df, y='trip_duration_minutes', color='darkred')

plt.subplot(3,2,5)
sns.histplot(data=df, x='avg_velocity_km/hr', bins=20, kde=True, color='darkred')

plt.subplot(3,2,6)
sns.boxplot(data=df, y='avg_velocity_km/hr', color='darkred')

plt.tight_layout()
plt.show()


plt.figure(figsize = (5,5))
plt.plot(df['pickup_longitude'], df['pickup_latitude'],'.', alpha = 0.5, markersize = 0.05, color='darkred')
plt.show()


df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])


df['pickup_datetime_year'] = df['pickup_datetime'].dt.year


df['pickup_datetime_month'] = df['pickup_datetime'].dt.month


df['pickup_datetime_day'] = df['pickup_datetime'].dt.day


df['pickup_datetime_hour'] = df['pickup_datetime'].dt.hour


df['pickup_datetime_year'].unique()


df['pickup_datetime_month'].unique()


df['pickup_datetime_day'].unique()


group_month = df.groupby('pickup_datetime_month')['id'].count()
group_month_df = group_month.reset_index()

plt.figure(figsize=(4,3))
sns.barplot(data=group_month_df, x='pickup_datetime_month', y='id', color='darkred')
plt.show()

print(group_month_df)


group_day = df.groupby('pickup_datetime_day')['id'].count()
group_day_df = group_day.reset_index()

plt.figure(figsize=(8,3))
sns.barplot(data=group_day_df, x='pickup_datetime_day', y='id', color='darkred')
plt.xticks(rotation=90)
plt.show()

print(group_day_df)


group_hour = df.groupby('pickup_datetime_hour')['id'].count()
group_hour_df = group_hour.reset_index()

plt.figure(figsize=(8,3))
sns.lineplot(data=group_hour_df, x='pickup_datetime_hour', y='id', color='darkred', marker='o')
plt.xticks(group_hour_df.index)
plt.show()

print(group_hour_df)



plt.figure(figsize=(5, 5))


# Create a polar plot
ax = plt.subplot(111, projection='polar')

# Convert hours to radians
hours_in_radians = np.deg2rad((group_hour.index) * 360 / 24)

# Plot values
ax.plot(hours_in_radians, group_hour.values, marker='o', linestyle='-', color='#850000')

# Set radial axis labels to show hours
ax.set_xticks(hours_in_radians)
ax.set_xticklabels([str(hour) + ':00' for hour in group_hour.index])
ax.set_theta_offset(np.pi/2)

# Fill the space between the data points
ax.fill(hours_in_radians, group_hour.values, '#850000', alpha=0.5)  # Adjust alpha for transparency ---> #850000 is dark red.

# Add a title
ax.set_title('Pickups hours vs. Values')

# Show the polar plot
plt.show()


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# --- PHASE 1: CLUSTERING (CORRECTED SECTION) ---

# 1. Prepare Data for Clustering (FIT on Pickup Coordinates)
coords = df[['pickup_latitude', 'pickup_longitude']].copy()
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

# 2. Apply K-Means (K=15 Zones)
K = 15
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(coords_scaled)

# 3. Assign Cluster Labels to DataFrames
# Pickup clusters (This part was correct)
df['pickup_cluster'] = kmeans.predict(coords_scaled)

# Dropoff clusters: CORRECTED STEPS
dropoff_coords = df[['dropoff_latitude', 'dropoff_longitude']].copy()

# RENAME columns to match the features the scaler/model were fitted on (pickup_*)
dropoff_coords.columns = ['pickup_latitude', 'pickup_longitude'] 

# Now transform and predict
dropoff_coords_scaled = scaler.transform(dropoff_coords)
df['dropoff_cluster'] = kmeans.predict(dropoff_coords_scaled)

# --- The rest of your code (adding centroids and aggregating) can follow ---
print(f"Clustering complete. {K} zones defined.")

# --- PHASE 2: AGGREGATE NEW CLUSTER FLOW DATA ---

# 1. Create a DataFrame for flow aggregation
cluster_flow_data = df.groupby(['pickup_cluster', 'dropoff_cluster', 'pickup_datetime_hour']).agg(
    trip_count=('id', 'count')
).reset_index()

# 2. Add Cluster Centroids (The coordinates for your desire lines)
# The centroids are stored in the K-Means object, but they are SCALED. We must inverse transform them.
centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids_unscaled, columns=['centroid_lat', 'centroid_lon'])
centroids_df['cluster_id'] = centroids_df.index

# 3. Merge Centroids into Flow Data (Twice: once for origin, once for destination)
cluster_flow_data = cluster_flow_data.merge(
    centroids_df.rename(columns={'cluster_id': 'pickup_cluster'}), 
    on='pickup_cluster', 
    how='left'
)
cluster_flow_data = cluster_flow_data.merge(
    centroids_df.rename(columns={'cluster_id': 'dropoff_cluster', 
                                 'centroid_lat': 'dropoff_centroid_lat',
                                 'centroid_lon': 'dropoff_centroid_lon'}), 
    on='dropoff_cluster', 
    how='left'
)

# 4. Final Cleanup for Visualization
# Log transform the trip count for line width scaling (crucial for visual clarity)
cluster_flow_data['log_trip_count'] = np.log1p(cluster_flow_data['trip_count'])

cluster_flow_data.to_csv('cluster_flow_data.csv', index=False)
print("\n--- New Cluster Flow Data Head ---")
print(cluster_flow_data.head())
print("\n[File cluster_flow_data.csv created for cluster-based flow analysis.]")


df.head()


plt.figure(figsize = (10,5))

plt.subplot(1,2,1)
sns.scatterplot(
    data=df,
    x='pickup_longitude',
    y='pickup_latitude',
    hue='pickup_cluster', 
    s=1,        
    alpha=0.5,  
    legend=False,
    palette='Spectral'
)

plt.subplot(1,2,2)
sns.scatterplot(
    data=df,
    x='dropoff_longitude',
    y='dropoff_latitude',
    hue='dropoff_cluster', 
    s=1,        
    alpha=0.5,  
    legend=False,
    palette='Spectral'
)

plt.title('Pickup Locations by Geographic Zone')
plt.show()


import contextily as ctx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def lnglat_to_mercator(lng, lat):
    r = 6378137.000
    x = r * np.radians(lng)
    scale = x / lng
    y = 180.0 / np.pi * np.log(np.tan(np.pi / 4.0 + lat * (np.pi / 180.0) / 2.0)) * scale
    return x, y


# --- 4. Prepare Network Data & Convert to Web Mercator ---
# Get Centroids (Lat/Lon)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=['lat', 'lon'])
centroid_df['cluster_id'] = centroid_df.index

# CONVERT Centroids to Web Mercator (The Fix!)
centroid_df['x'], centroid_df['y'] = lnglat_to_mercator(centroid_df['lon'], centroid_df['lat'])

# Aggregate Flow
flow = df.groupby(['pickup_datetime_hour', 'pickup_cluster', 'dropoff_cluster']).size().reset_index(name='trips')
flow = flow.merge(centroid_df, left_on='pickup_cluster', right_on='cluster_id')
flow.rename(columns={'x': 'start_x', 'y': 'start_y'}, inplace=True)
flow = flow.merge(centroid_df, left_on='dropoff_cluster', right_on='cluster_id')
flow.rename(columns={'x': 'end_x', 'y': 'end_y'}, inplace=True)
flow = flow[flow['pickup_cluster'] != flow['dropoff_cluster']]

# Calculate width scaling
flow['width'] = np.log1p(flow['trips'])
max_width = flow['width'].max()

# --- 5. Generate Animation Frames with Map Background ---
print("Generating frames...")

# Define the bounds in Mercator (The Fix!)
nyc_min_x, nyc_min_y = lnglat_to_mercator(-74.03, 40.63)
nyc_max_x, nyc_max_y = lnglat_to_mercator(-73.77, 40.85)

for hour in range(24):
    # Create larger figure for high-quality map
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. Plot the Map Tiles (Geographical Layer)
    # We force the axis limits to the NYC Mercator coordinates calculated above
    ax.set_xlim(nyc_min_x, nyc_max_x)
    ax.set_ylim(nyc_min_y, nyc_max_y)
    
    # Add the map background (CartoDB Positron is clean and professional)
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        print(f"Could not fetch map tiles: {e}. Check internet connection.")

    # 2. Draw Static Nodes (Centroids)
    ax.scatter(centroid_df['x'], centroid_df['y'], s=200, c='red', edgecolors='white', zorder=5, alpha=0.9)

    # 3. Draw Dynamic Flows (Arrows)
    hour_data = flow[flow['pickup_datetime_hour'] == hour]
    threshold = hour_data['trips'].quantile(0.50) 
    top_flows = hour_data[hour_data['trips'] > threshold]

    for _, row in top_flows.iterrows():
        alpha = min(1.0, 0.3 + (row['width'] / max_width) * 0.7)
        linewidth = (row['width'] / max_width) * 5
        
        # Arrows use Mercator coordinates (start_x, start_y)
        ax.arrow(
            row['start_x'], row['start_y'], 
            row['end_x'] - row['start_x'], row['end_y'] - row['start_y'],
            color='#0044cc', # Professional Blue
            alpha=alpha,
            width=30 * linewidth, 
            head_width=250, 
            length_includes_head=True,
            zorder=3
        )

    # Formatting
    ax.set_axis_off()
    ax.set_title(f"NYC Taxi Demand: {hour:02d}:00", fontsize=24, fontweight='bold', pad=20)
    
    # Save Frame
    plt.savefig(f'map_frame_{hour:02d}.png', dpi=100, bbox_inches='tight')
    plt.close(fig)

print("Done! Frames created. Stitch them together to see the flow on the map.")


import imageio.v3 as iio
import os

# 1. Define the output filename
output_gif_name = 'nyc_mobility_flow.gif'

# 2. Locate and sort the frames
# We sort to ensure 00 comes before 01, etc.
# Adjust the prefix 'map_frame_' if you changed the filename in the previous step.
frames = sorted([f for f in os.listdir('.') if f.startswith('map_frame_') and f.endswith('.png')])

# Critical Check: Ensure frames exist before trying to build
if not frames:
    print("Error: No frames found. Please run the frame generation code first.")
else:
    print(f"Found {len(frames)} frames. Compiling GIF...")

    # 3. Read frames into memory
    images = [iio.imread(filename) for filename in frames]

    # 4. Write the GIF
    # duration=500 means 500ms (0.5 seconds) per frame.
    # loop=0 means loop forever.
    iio.imwrite(output_gif_name, images, duration=500, loop=0)

    print(f"Success! Animation saved as: {output_gif_name}")


from IPython.display import Image
Image(filename='nyc_mobility_flow.gif')


# converting the pickup_datetime_hour and pickup_datetime_month to cyclic features:

df['hour_sin'] = np.sin(2 * np.pi * df['pickup_datetime_hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['pickup_datetime_hour'] / 24)

df['month_sin'] = np.sin(2 * np.pi * df['pickup_datetime_month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['pickup_datetime_month'] / 12)


df.columns


columns_to_drop = ['id', 'pickup_longitude', 'pickup_latitude', 'trip_duration', 
                   'trip_duration_minutes', 'avg_velocity_km/hr', 'dropoff_longitude', 'pickup_datetime_year',
                   'dropoff_latitude', 'pickup_datetime', 'dropoff_datetime', 'pickup_datetime_day', 
                   'pickup_datetime_month', 'pickup_datetime_hour']

numerical_cols = ['passenger_count', 'manhattan_dist_km',
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos']

categorical_cols = ['vendor_id', 'store_and_fwd_flag', 'pickup_cluster', 'dropoff_cluster']


#df_test = pd.read_csv('test.csv')
#df_test.shape


#df_test.head()


# preprocessing for test data

'''

df_test['manhattan_dist_km'] = manhattan_distance_vectorized(
    df_test['pickup_latitude'], 
    df_test['pickup_longitude'], 
    df_test['dropoff_latitude'], 
    df_test['dropoff_longitude']
)

df_test = df_test[(df_test.pickup_longitude> xlim[0]) & (df_test.pickup_longitude < xlim[1])]
df_test = df_test[(df_test.dropoff_longitude> xlim[0]) & (df_test.dropoff_longitude < xlim[1])]
df_test = df_test[(df_test.pickup_latitude> ylim[0]) & (df_test.pickup_latitude < ylim[1])]
df_test = df_test[(df_test.dropoff_latitude> ylim[0]) & (df_test.dropoff_latitude < ylim[1])]

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
df_test['pickup_datetime_year'] = df_test['pickup_datetime'].dt.year
df_test['pickup_datetime_month'] = df_test['pickup_datetime'].dt.month
df_test['pickup_datetime_day'] = df_test['pickup_datetime'].dt.day
df_test['pickup_datetime_hour'] = df_test['pickup_datetime'].dt.hour

df_test['hour_sin'] = np.sin(2 * np.pi * df_test['pickup_datetime_hour'] / 24)
df_test['hour_cos'] = np.cos(2 * np.pi * df_test['pickup_datetime_hour'] / 24)
df_test['month_sin'] = np.sin(2 * np.pi * df_test['pickup_datetime_month'] / 12)
df_test['month_cos'] = np.cos(2 * np.pi * df_test['pickup_datetime_month'] / 12)

'''


#print(df_test.shape)
#df_test.head()


'''
test_coords = df_test[['pickup_latitude', 'pickup_longitude']].copy()
test_coords_scaled = scaler.transform(test_coords)
df_test['pickup_cluster'] = kmeans.predict(test_coords_scaled)

test_dropoff = df_test[['dropoff_latitude', 'dropoff_longitude']].copy()
test_dropoff.columns = ['pickup_latitude', 'pickup_longitude']
test_dropoff_scaled = scaler.transform(test_dropoff)
df_test['dropoff_cluster'] = kmeans.predict(test_dropoff_scaled)
'''


#print(df_test.shape)
#df_test.head()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Define X and y

'''
X_train = df[numerical_cols + categorical_cols]
y_train = (df['trip_duration']) # Log transform target

X_test = df_test[numerical_cols + categorical_cols] 
# y_test is what we want to predict
'''


# --- Building the Pipeline ---

# 1. Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])


# 2. Full Pipeline with Model (e.g., XGBoost)
from xgboost import XGBRegressor

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=1000, 
                           random_state=42, 
                           learning_rate=0.05,
                           max_depth=8,
                           n_jobs=-1,)
    )
])


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error


# 1. Define X (Features) and y (Target) from your TRAINING data
# Note: We use the raw seconds since you decided against log-scaling
X = df[numerical_cols + categorical_cols]
y = df['trip_duration'] 

# 2. Create a Hold-out Validation Set (The "y_test" you were missing)
# This reserves 20% of your data to check accuracy
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Pipeline on the 80% split
pipeline.fit(X_train, y_train)

# 4. Predict on the 20% Validation split
# These predictions can be compared to y_val (which is your y_test equivalent)
predictions = pipeline.predict(X_val)

# --- NOW YOUR EVALUATION CODE WILL WORK ---
# (I swapped 'y_test' with 'y_val' to match the split above)

# Clip negatives (Safety check for raw-target regression)
predictions = np.maximum(predictions, 0) 

# Metrics
rmse = np.sqrt(mean_squared_error(y_val, predictions))
mae = mean_absolute_error(y_val, predictions)
r2 = r2_score(y_val, predictions)
rmsle = np.sqrt(mean_squared_log_error(y_val, predictions))

print("="*30)
print("  MODEL PERFORMANCE REPORT (VALIDATION) ")
print("="*30)
print(f"RMSE (Root Mean Sq Error) : {rmse:.2f} seconds")
print(f"MAE (Mean Absolute Error) : {mae:.2f} seconds")
print(f"R2 Score                  : {r2:.4f}")
print(f"RMSLE (Log Error)         : {rmsle:.5f}")
print("="*30)


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 1. Define the Parameter Grid
# Note the 'model__' prefix. This tells sklearn to tune the 'model' step inside your Pipeline.
param_dist = {
    'model__n_estimators': randint(500, 1500),      # Number of trees
    'model__max_depth': randint(6, 15),             # Depth of tree (complexity)
    'model__learning_rate': uniform(0.01, 0.1),     # Step size (smaller is usually more accurate but slower)
    'model__subsample': uniform(0.6, 0.4),          # Fraction of rows to use per tree (prevents overfitting)
    'model__colsample_bytree': uniform(0.6, 0.4),   # Fraction of columns to use per tree
    'model__min_child_weight': randint(1, 10)       # Minimum weight allowed in a child (regularization)
}

# 2. Setup Randomized Search
# n_iter=10 means it will try 10 random combinations. 
# In a real project with more time, you might set this to 50 or 100.
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=10,           # Try 10 candidates
    scoring='neg_root_mean_squared_error', # Optimize for RMSE
    cv=3,                # 3-Fold Cross Validation (Robustness check)
    verbose=1,
    n_jobs=-1,           # Use all cores
    random_state=42
)

# 3. Run the Search (Fit on Train)
print("Starting Hyperparameter Tuning (this may take a while)...")
random_search.fit(X_train, y_train)

# 4. Get Best Results
print(f"\nBest Parameters: {random_search.best_params_}")
print(f"Best CV RMSE: {-random_search.best_score_:.2f}")

# 5. Evaluate Best Model on Validation Set
best_model = random_search.best_estimator_
preds_tuned = best_model.predict(X_val)
preds_tuned = np.maximum(preds_tuned, 0) # Clip negatives

# Calculate RMSLE for the tuned model
tuned_rmsle = np.sqrt(mean_squared_log_error(y_val, preds_tuned))
print(f"Tuned Model Validation RMSLE: {tuned_rmsle:.5f}")


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. Transform the Target (The "Squashing" Step) ---
# We use log1p which calculates log(1 + x) automatically
y_log = np.log1p(df['trip_duration']) 

# Define X (Features remain the same)
X = df[numerical_cols + categorical_cols]

# --- 2. Split Data ---
# We split X and the NEW LOG-TRANSFORMED y
X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# --- 3. Train on Log Targets ---
print("Training Model on Log-Transformed Target...")
# Note: We are fitting to y_train_log, NOT the raw seconds
pipeline.fit(X_train, y_train_log)

# --- 4. Predict (Output is in Log Scale) ---
log_predictions = pipeline.predict(X_val)

# --- 5. Inverse Transform (The "Expanding" Step) ---
# To verify business metrics (MAE in seconds), we must convert back.
# expm1 is the inverse of log1p
predictions_seconds = np.expm1(log_predictions)
actual_seconds = np.expm1(y_val_log)

# Critical Safety Clip: Exp can explode if predictions are wild, 
# and seconds can't be negative.
predictions_seconds = np.maximum(predictions_seconds, 0)


# --- 6. Evaluation ---

# A. The Competition Metric (RMSLE)
# Since our predictions are ALREADY logs, RMSE of logs == RMSLE
rmsle = np.sqrt(mean_squared_error(y_val_log, log_predictions))

# B. The Business Metrics (converted back to seconds)
rmse_real = np.sqrt(mean_squared_error(actual_seconds, predictions_seconds))
mae_real = mean_absolute_error(actual_seconds, predictions_seconds)
r2 = r2_score(actual_seconds, predictions_seconds)

print("="*40)
print("  MODEL PERFORMANCE REPORT (LOG TRANSFORMED) ")
print("="*40)
print(f"RMSLE (The Goal)          : {rmsle:.5f}")
print("-" * 40)
print(f"RMSE (Seconds)            : {rmse_real:.2f}")
print(f"MAE (Seconds)             : {mae_real:.2f}")
print(f"R2 Score                  : {r2:.4f}")
print("="*40)


# --- Training and Prediction ---

'''
# 1. Fit the entire pipeline on TRAINING data
# This fits the Scaler, fits the OneHotEncoder, and trains the Model
pipeline.fit(X_train, y_train)

# 2. Predict on TEST data
# The pipeline automatically transforms X_test using the params learned from X_train
predictions_log = pipeline.predict(X_test)

# 3. Convert back from Log scale
predictions = (predictions_log)

print("Pipeline training and prediction complete.")
'''




