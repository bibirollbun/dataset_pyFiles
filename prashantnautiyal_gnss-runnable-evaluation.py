import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import re
import joblib
from sklearn.metrics import mean_squared_error

# (Optional) Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# IMPORTANT: Update this path if your data is in a different location
DATA_DIR = '/kaggle/input/smartphone-decimeter-2023/sdc2023'

# This MUST match the sequence length the model was trained on
SEQUENCE_LENGTH = 50


TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
trace_paths = glob.glob(os.path.join(TRAIN_DATA_DIR, "*", "*"))
unique_traces = sorted([path for path in trace_paths if os.path.isdir(path)])
print(f"Found {len(unique_traces)} total traces.")

# Split to get the same validation set as during training
train_paths, val_paths = train_test_split(
    unique_traces,
    test_size=0.2,  # Must be same as training
    random_state=42 # Must be same as training
)
print(f"Using {len(val_paths)} validation traces for evaluation.")


def _safe_column_lookup(df, pattern_list):
    cols = df.columns.tolist()
    found = []
    for pat in pattern_list:
        pat_re = re.compile(pat, re.I)
        match = next((c for c in cols if pat_re.search(c)), None)
        if match is None:
            return None
        found.append(match)
        cols.remove(match)
    return found

def _extract_from_message_style(imu_df):
    accel_df = pd.DataFrame()
    gyro_df = pd.DataFrame()
    if 'MessageType' in imu_df.columns and any(c.lower().startswith('measurement') for c in imu_df.columns):
        meas_cols = {
            'x': next((c for c in imu_df.columns if re.match(r'measurement.*x', c, re.I)), None),
            'y': next((c for c in imu_df.columns if re.match(r'measurement.*y', c, re.I)), None),
            'z': next((c for c in imu_df.columns if re.match(r'measurement.*z', c, re.I)), None)
        }
        if all(meas_cols.values()) and 'utcTimeMillis' in imu_df.columns:
            mx, my, mz = meas_cols['x'], meas_cols['y'], meas_cols['z']
            accel_rows = imu_df[imu_df['MessageType'].str.contains('accel', na=False, case=False)]
            if not accel_rows.empty:
                accel_df = accel_rows[['utcTimeMillis', mx, my, mz]].copy()
                accel_df.columns = ['utcTimeMillis', 'accel_x', 'accel_y', 'accel_z']
            gyro_rows = imu_df[imu_df['MessageType'].str.contains('gyro', na=False, case=False)]
            if not gyro_rows.empty:
                gyro_df = gyro_rows[['utcTimeMillis', mx, my, mz]].copy()
                gyro_df.columns = ['utcTimeMillis', 'gyro_x', 'gyro_y', 'gyro_z']
    return accel_df, gyro_df

def load_and_preprocess_trace(trace_path):
    trace_name = os.path.basename(os.path.dirname(trace_path)) + "/" + os.path.basename(trace_path)
    try:
        gnss_df = pd.read_csv(os.path.join(trace_path, 'device_gnss.csv'), low_memory=False)
        imu_df = pd.read_csv(os.path.join(trace_path, 'device_imu.csv'))
        ground_truth_df = pd.read_csv(os.path.join(trace_path, 'ground_truth.csv'))
        
        # --- TRIP ID LOGIC (PART 1) ---
        # Create a unique ID from the folder name (drive_id) and subfolder name (phone)
        drive_id = os.path.basename(os.path.dirname(trace_path))
        phone = os.path.basename(trace_path)
        ground_truth_df['trip_id'] = f"{drive_id}_{phone}" # <--- THIS IS THE TRIP ID LOGIC
        # --- END OF TRIP ID LOGIC (PART 1) ---
        
        if 'UnixTimeMillis' in ground_truth_df.columns:
            ground_truth_df.rename(columns={'UnixTimeMillis': 'utcTimeMillis'}, inplace=True)
        accel_df, gyro_df = _extract_from_message_style(imu_df)
        if accel_df.empty or gyro_df.empty:
            accel_patterns = [
                ['UncalibratedAccelerometerMps2_x', 'UncalibratedAccelerometerMps2_y', 'UncalibratedAccelerometerMps2_z'],
                ['AccelerometerMps2_x', 'AccelerometerMps2_y', 'AccelerometerMps2_z'],
                [r'accel.*_x', r'accel.*_y', r'accel.*_z']
            ]
            gyro_patterns = [
                ['UncalibratedGyroscopeRps_x', 'UncalibratedGyroscopeRps_y', 'UncalibratedGyroscopeRps_z'],
                ['GyroscopeRps_x', 'GyroscopeRps_y', 'GyroscopeRps_z'],
                [r'gyro.*_x', r'gyro.*_y', r'gyro.*_z']
            ]
            time_col = next((c for c in imu_df.columns if 'time' in c.lower()), None)
            if not time_col: return pd.DataFrame()
            accel_cols = next((cols for pat in accel_patterns if (cols := _safe_column_lookup(imu_df, pat))), None)
            gyro_cols = next((cols for pat in gyro_patterns if (cols := _safe_column_lookup(imu_df, pat))), None)
            if accel_cols:
                accel_df = imu_df[[time_col] + accel_cols].copy()
                accel_df.columns = ['utcTimeMillis', 'accel_x', 'accel_y', 'accel_z']
            if gyro_cols:
                gyro_df = imu_df[[time_col] + gyro_cols].copy()
                gyro_df.columns = ['utcTimeMillis', 'gyro_x', 'gyro_y', 'gyro_z']
        if accel_df.empty and gyro_df.empty: return pd.DataFrame()
        if not accel_df.empty and gyro_df.empty:
            gyro_df = accel_df[['utcTimeMillis']].copy()
            gyro_df[['gyro_x', 'gyro_y', 'gyro_z']] = 0.0
        if accel_df.empty and not gyro_df.empty:
            accel_df = gyro_df[['utcTimeMillis']].copy()
            accel_df[['accel_x', 'accel_y', 'accel_z']] = 0.0
        for df in [accel_df, gyro_df]:
            df['utcTimeMillis'] = df['utcTimeMillis'].astype(np.int64)
            df.sort_values('utcTimeMillis', inplace=True)
        imu_merged = pd.merge_asof(accel_df, gyro_df, on='utcTimeMillis', direction='nearest', tolerance=20).dropna()
        required_gnss_cols = ['utcTimeMillis', 'WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']
        if not all(col in gnss_df.columns for col in required_gnss_cols): return pd.DataFrame()
        gnss_df = gnss_df[required_gnss_cols].copy()
        gnss_df.rename(columns={
            'WlsPositionXEcefMeters': 'WlsPositionEcefMeters_x',
            'WlsPositionYEcefMeters': 'WlsPositionEcefMeters_y',
            'WlsPositionZEcefMeters': 'WlsPositionEcefMeters_z'
        }, inplace=True)
        if 'utcTimeMillis' not in ground_truth_df.columns: return pd.DataFrame()
        ground_truth_df.sort_values('utcTimeMillis', inplace=True)
        gnss_df.sort_values('utcTimeMillis', inplace=True)
        merged_df = pd.merge_asof(ground_truth_df, gnss_df, on='utcTimeMillis', direction='nearest', tolerance=1000)
        merged_df = pd.merge_asof(merged_df, imu_merged, on='utcTimeMillis', direction='nearest', tolerance=50)
        essential_cols = ['LatitudeDegrees', 'LongitudeDegrees', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        merged_df.dropna(subset=essential_cols, inplace=True)
        return merged_df.reset_index(drop=True)
    except Exception as e:
        print(f"Warning: Error processing trace {trace_name}: {e}")
        return pd.DataFrame()


def create_sliding_windows(feature_data, target_data, sequence_length):
    X, y = [], []
    for i in range(len(feature_data) - sequence_length):
        X.append(feature_data[i:(i + sequence_length)])
        y.append(target_data[i + sequence_length])
    return np.array(X), np.array(y)

def ecef_to_lla(ecef_coords):
    a = 6378137.0
    e = 8.1819190842622e-2
    x, y, z = ecef_coords[:, 0], ecef_coords[:, 1], ecef_coords[:, 2]
    p = np.sqrt(x**2 + y**2)
    lon = np.arctan2(y, x)
    lat_initial = np.arctan2(z, p * (1 - e**2))
    lat = lat_initial
    for _ in range(5):
        N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e**2 * N / (N + h)))
    N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    lat_deg = np.rad2deg(lat)
    lon_deg = np.rad2deg(lon)
    return lat_deg, lon_deg, alt


def calculate_gsdc_score(ground_truth, predictions):
    R = 6371000
    lat_gt, lon_gt = np.deg2rad(ground_truth[:, 0]), np.deg2rad(ground_truth[:, 1])
    lat_pred, lon_pred = np.deg2rad(predictions[:, 0]), np.deg2rad(predictions[:, 1])
    dlon, dlat = lon_pred - lon_gt, lat_pred - lat_gt
    a = np.sin(dlat/2)**2 + np.cos(lat_gt) * np.cos(lat_pred) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distances = R * c
    return np.mean([np.percentile(distances, 50), np.percentile(distances, 95)])


# --- Feature definitions based on your 3-scaler logic ---
coords_features = ['WlsLatitudeDegrees', 'WlsLongitudeDegrees']
imu_features = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
target_cols = ['LatOffsetDegrees', 'LonOffsetDegrees']
ecef_cols = ['WlsPositionEcefMeters_x', 'WlsPositionEcefMeters_y', 'WlsPositionEcefMeters_z']


print("Loading pre-trained artifacts...")

try:
    # 1. Load the Coords scaler
    coords_scaler = joblib.load("/kaggle/input/finalweight/coords_scaler.joblib")
    print("✅ Coords scaler loaded successfully.")
    
    # 2. Load the IMU scaler
    imu_scaler = joblib.load("/kaggle/input/finalweight/imu_scaler.joblib")
    print("✅ IMU scaler loaded successfully.")

    # 3. Load the Target scaler
    target_scaler = joblib.load("/kaggle/input/finalweight/target_scaler (2).joblib")
    print("✅ Target scaler loaded successfully.")

    # 4. Load the pre-trained Keras model
    model_path = "/kaggle/input/gnss-prediction/keras/default/1/gnss_correction_model_final.keras"
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully from {model_path}")
    model.summary()

except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    print("Please check the file paths and ensure the files are available in the input directory.")


print("\nLoading and preprocessing all validation traces...")
all_traces_df_list_val = []
for path in val_paths:
    trace_df = load_and_preprocess_trace(path)
    if not trace_df.empty:
        all_traces_df_list_val.append(trace_df)
print(f"Successfully loaded and preprocessed {len(all_traces_df_list_val)} validation traces.")

full_val_df = pd.concat(all_traces_df_list_val, ignore_index=True)

# --- Feature Engineering (as per your snippet) ---
# 1. Convert ECEF to LLA
ecef_coords_val = full_val_df[ecef_cols].values
lat_wls_val, lon_wls_val, alt_wls_val = ecef_to_lla(ecef_coords_val)
full_val_df['WlsLatitudeDegrees'] = lat_wls_val
full_val_df['WlsLongitudeDegrees'] = lon_wls_val

# 2. Calculate Target Offsets
full_val_df['LatOffsetDegrees'] = full_val_df['LatitudeDegrees'] - full_val_df['WlsLatitudeDegrees']
full_val_df['LonOffsetDegrees'] = full_val_df['LongitudeDegrees'] - full_val_df['WlsLongitudeDegrees']

# 3. Filter outliers
MAX_OFFSET = 0.01 # Use the same 0.01 threshold
initial_val_rows = len(full_val_df)
full_val_df = full_val_df[
    (full_val_df['LatOffsetDegrees'].abs() < MAX_OFFSET) &
    (full_val_df['LonOffsetDegrees'].abs() < MAX_OFFSET)
].reset_index(drop=True)
print(f"Removing {initial_val_rows - len(full_val_df)} rows with extreme offsets.")

print("\nCreating validation sequences...")
X_val_list = []
y_val_list = []
ground_truth_coords_list = []
wls_coords_list = []

# --- TRIP ID LOGIC (PART 2) ---
# We loop over the DataFrame grouped by the 'trip_id' we created in Cell 4.
for trip_id, trip_df in full_val_df.groupby('trip_id'): # <--- THIS IS THE TRIP ID LOGIC
    
    # --- Apply all 3 loaded scalers ---
    scaled_coords = coords_scaler.transform(trip_df[coords_features])
    scaled_imu = imu_scaler.transform(trip_df[imu_features])
    scaled_target = target_scaler.transform(trip_df[target_cols])
    
    # 4. Stack SCALED Coords + SCALED IMU (matching 3-scaler logic)
    scaled_features = np.hstack([scaled_coords, scaled_imu])
    scaled_features = np.clip(scaled_features, -100.0, 100.0)

    X_trace, y_trace = create_sliding_windows(
        scaled_features.astype(np.float32), 
        scaled_target.astype(np.float32), 
        SEQUENCE_LENGTH
    )
    
    if len(X_trace) > 0:
        X_val_list.append(X_trace)
        y_val_list.append(y_trace)
        ground_truth_coords_list.append(trip_df[['LatitudeDegrees', 'LongitudeDegrees']].iloc[SEQUENCE_LENGTH:].values)
        wls_coords_list.append(trip_df[['WlsLatitudeDegrees', 'WlsLongitudeDegrees']].iloc[SEQUENCE_LENGTH:].values)
# --- END OF TRIP ID LOGIC (PART 2) ---

plot_len = 0 # Initialize plot length
if len(X_val_list) > 0:
    # --- FIX FOR PLOTTING SPIKE ---
    # Store the length of the *first trip* and limit it to 500 for plotting
    len_first_trip = len(X_val_list[0]) 
    plot_len = min(len_first_trip, 500) # <--- THIS IS THE FIX (added this line)
    # --- END OF FIX ---
    
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    ground_truth_coords_for_eval = np.concatenate(ground_truth_coords_list, axis=0)
    wls_coords_for_eval = np.concatenate(wls_coords_list, axis=0)
    
    print(f"\nFinal X_val (evaluation) shape: {X_val.shape}")
    print(f"Final y_val (evaluation) shape: {y_val.shape}")
else:
    print("No validation data available after processing. Check paths and filters.")


print("\nEvaluating model on validation set...")
predicted_scaled_offsets = model.predict(X_val)

# Inverse transform to get offsets in degrees
predicted_offsets = target_scaler.inverse_transform(predicted_scaled_offsets)

# Add offsets to the baseline WLS coordinates
final_predictions = wls_coords_for_eval + predicted_offsets

# Compute GSDC score
score = calculate_gsdc_score(ground_truth_coords_for_eval, final_predictions)
print(f"\nValidation GSDC Score: {score:.4f} meters")


print("Plotting sample trajectory...")

# --- FIX FOR PLOTTING ---
# Use plot_len to only plot the first 500 points (or less) of the *first trip*.
plot_len = min(len_first_trip, 500) 
# --- END OF FIX ---

plt.figure(figsize=(12, 8))
plt.plot(ground_truth_coords_for_eval[:plot_len, 1], ground_truth_coords_for_eval[:plot_len, 0], '.-', label='Ground Truth', alpha=0.7)
plt.plot(final_predictions[:plot_len, 1], final_predictions[:plot_len, 0], '.-', label='Prediction', alpha=0.7)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Predicted vs. Ground Truth Trajectory (Sample)")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()


R = 6371000 # Earth radius
lat_gt = np.deg2rad(ground_truth_coords_for_eval[:, 0])
lon_gt = np.deg2rad(ground_truth_coords_for_eval[:, 1])
lat_pred = np.deg2rad(final_predictions[:, 0])
lon_pred = np.deg2rad(final_predictions[:, 1])

dlon = lon_pred - lon_gt
dlat = lat_pred - lat_gt
a = np.sin(dlat/2)**2 + np.cos(lat_gt) * np.cos(lat_pred) * np.sin(dlon/2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distances = R * c

# Model performance
mae_distance = np.mean(distances)
rmse_distance = np.sqrt(np.mean(distances**2))
p50 = np.percentile(distances, 50)
p95 = np.percentile(distances, 95)
gsdc_score_calculated = np.mean([p50, p95])


lat_wls_rad = np.deg2rad(wls_coords_for_eval[:, 0])
lon_wls_rad = np.deg2rad(wls_coords_for_eval[:, 1])
dlon_base = lon_wls_rad - lon_gt
dlat_base = lat_wls_rad - lat_gt
a_base = np.sin(dlat_base/2)**2 + np.cos(lat_gt) * np.cos(lat_wls_rad) * np.sin(dlon_base/2)**2
c_base = 2 * np.arctan2(np.sqrt(a_base), np.sqrt(1 - a_base))
baseline_distances = R * c_base

# Baseline performance
baseline_p50 = np.percentile(baseline_distances, 50)
baseline_p95 = np.percentile(baseline_distances, 95)
baseline_gsdc = np.mean([baseline_p50, baseline_p95])
baseline_mae = np.mean(baseline_distances)
baseline_rmse = np.sqrt(np.mean(baseline_distances**2))


metrics_data = {
    'Metric': ['GSDC Score', 'Median Error (p50)', '95th Percentile (p95)', 'MAE', 'RMSE'],
    'Baseline GNSS (meters)': [
        baseline_gsdc,
        baseline_p50,
        baseline_p95,
        baseline_mae,
        baseline_rmse
    ],
    'Final Model (meters)': [
        gsdc_score_calculated,
        p50,
        p95,
        mae_distance,
        rmse_distance
    ]
}

df_results = pd.DataFrame(metrics_data)
df_results['Improvement (meters)'] = df_results['Baseline GNSS (meters)'] - df_results['Final Model (meters)']
print("--- Performance Comparison: Baseline vs. Final Model ---")
print(df_results.to_string(index=False))


print("\nPlotting error distributions...")

# --- FIX FOR PLOTTING ---
# Slice the data to only show the first trip for a cleaner histogram
plot_len = min(len_first_trip, 500)
# --- END OF FIX ---

plt.figure(figsize=(12, 7))
plt.hist(baseline_distances[:plot_len], bins=100, label='Baseline GNSS Error', alpha=0.7, color='darkorange', range=[0, 15])
plt.hist(distances[:plot_len], bins=100, label='Final Model Error', alpha=0.7, color='royalblue', range=[0, 15])
plt.title('Distribution of Baseline Error vs. Final Model Error (Sample)')
plt.xlabel('Position Error (meters)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


print("\nPlotting metrics comparison bar chart...")

# Get data from the DataFrame created in Cell 15
labels = df_results['Metric']
baseline_values = df_results['Baseline GNSS (meters)']
model_values = df_results['Final Model (meters)']

x = np.arange(len(labels)) # the label locations
width = 0.35 # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, baseline_values, width, label='Baseline GNSS Error', color='darkorange')
rects2 = ax.bar(x + width/2, model_values, width, label='Final Model Error', color='royalblue')

# Add text for labels, title, and axes ticks
ax.set_ylabel('Error (meters)')
ax.set_title('Model Performance Improvement Over Baseline', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.legend()
ax.set_ylim(0, max(baseline_values.max(), model_values.max()) * 1.15) # Dynamic y-limit
ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f')
fig.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


print("\nPlotting Predicted vs. True values...")

# Calculate the true offsets (Ground Truth - WLS Baseline)
true_offsets_for_eval = ground_truth_coords_for_eval - wls_coords_for_eval

# --- FIX FOR PLOTTING ---
plot_len = min(len_first_trip, 500)
# --- END OF FIX ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Predicted vs. True Offset Comparison (Sample)', fontsize=16)

# --- Latitude Plot ---
ax1.scatter(true_offsets_for_eval[:plot_len, 0], predicted_offsets[:plot_len, 0], alpha=0.1, s=10)
lims = [min(ax1.get_xlim()[0], ax1.get_ylim()[0]), max(ax1.get_xlim()[1], ax1.get_ylim()[1])]
ax1.plot(lims, lims, 'r--', alpha=0.75, zorder=5, label='Perfect Prediction (y=x)')
ax1.set_xlabel('True Latitude Offset (Degrees)')
ax1.set_ylabel('Predicted Latitude Offset (Degrees)')
ax1.set_title('Latitude Offset')
ax1.grid(True)
ax1.legend()
ax1.axis('equal')

# --- Longitude Plot ---
ax2.scatter(true_offsets_for_eval[:plot_len, 1], predicted_offsets[:plot_len, 1], alpha=0.1, s=10)
lims = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]), max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
ax2.plot(lims, lims, 'r--', alpha=0.75, zorder=5, label='Perfect Prediction (y=x)')
ax2.set_xlabel('True Longitude Offset (Degrees)')
ax2.set_ylabel('Predicted Longitude Offset (Degrees)')
ax2.set_title('Longitude Offset')
ax2.grid(True)
ax2.legend()
ax2.axis('equal')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


print("\nPlotting Residuals...")

# Calculate the true offsets (Ground Truth - WLS Baseline)
true_offsets_for_eval = ground_truth_coords_for_eval - wls_coords_for_eval

# Calculate residuals (Error = True - Predicted)
residuals = true_offsets_for_eval - predicted_offsets

# --- FIX FOR PLOTTING ---
plot_len = min(len_first_trip, 500)
# --- END OF FIX ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Residuals Plot (Error vs. True Value) (Sample)', fontsize=16)

# --- Latitude Residuals ---
ax1.scatter(true_offsets_for_eval[:plot_len, 0], residuals[:plot_len, 0], alpha=0.1, s=10)
ax1.axhline(0, color='r', linestyle='--', label='y = 0 (No Error)')
ax1.set_xlabel('True Latitude Offset (Degrees)')
ax1.set_ylabel('Residual (Error) in Degrees')
ax1.set_title('Latitude Residuals')
ax1.grid(True)
ax1.legend()

# --- Longitude Residuals ---
ax2.scatter(true_offsets_for_eval[:plot_len, 1], residuals[:plot_len, 1], alpha=0.1, s=10)
ax2.axhline(0, color='r', linestyle='--', label='y = 0 (No Error)')
ax2.set_xlabel('True Longitude Offset (Degrees)')
ax2.set_ylabel('Residual (Error) in Degrees')
ax2.set_title('Longitude Residuals')
ax2.grid(True)
ax2.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


print("\nPlotting Error (in meters) over time for a sample...")

# --- FIX FOR PLOTTING SPIKE ---
# Get the 'distances' (in meters) from Cell 12
# Slice using the plot_len (defined in Cell 9) to only show the first trip.
sample_distances = distances[:plot_len] 
# --- END OF FIX ---
time_steps = np.arange(len(sample_distances))

plt.figure(figsize=(14, 6))
plt.plot(time_steps, sample_distances, label='Model Error (meters)', color='b', alpha=0.8)
plt.xlabel(f'Time Step (for first {plot_len} points)') # <--- Label is now dynamic
plt.ylabel('Position Error (meters)')
plt.title('Model Error Over Time (Sample)')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Evaluation Complete ---")

