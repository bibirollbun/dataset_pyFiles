import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visualizations
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
import os

import pandas as pd


import kaggle_evaluation.cmi_inference_server


train = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")  # Replace with your actual path
train_demo = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")



print("Train Shape:", train.shape)
print("Train Demographics Shape:", train_demo.shape)

# Display column types and null values
print(train.info())
print(train_demo.info())

# Unique labels
print("Unique gestures:", train['gesture'].unique())
print("Unique behaviors:", train['behavior'].unique())
print("Unique orientations:", train['orientation'].unique())



print(train_demo.describe(include='all'))

# Distribution of subjects, age, sex
sns.countplot(data=train_demo, x='sex')
plt.title("Sex Distribution")
plt.show()

sns.histplot(train_demo['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()



# Gesture distribution
sns.countplot(data=train, y='gesture', order=train['gesture'].value_counts().index)
plt.title("Gesture Frequency")
plt.show()

# Target vs Non-target
sns.countplot(data=train, x='sequence_type')
plt.title("Target vs Non-Target Gestures")
plt.show()

# Orientation distribution
sns.countplot(data=train, y='orientation', order=train['orientation'].value_counts().index)
plt.title("Body Orientation")
plt.show()



example_seq_id = train['sequence_id'].unique()[0]
example_seq = train[train['sequence_id'] == example_seq_id]

plt.plot(example_seq['sequence_counter'], example_seq['acc_x'], label='acc_x')
plt.plot(example_seq['sequence_counter'], example_seq['acc_y'], label='acc_y')
plt.plot(example_seq['sequence_counter'], example_seq['acc_z'], label='acc_z')
plt.title(f"Accelerometer Data for Sequence {example_seq_id}")
plt.xlabel("Sequence Step")
plt.ylabel("Acceleration (m/s^2)")
plt.legend()
plt.show()



tof_cols = [col for col in train.columns if col.startswith('tof_')]

missing_tof = (train[tof_cols] == -1).mean().sort_values(ascending=False) * 100
missing_tof.head(30).plot(kind='bar', title="Top ToF Pixels with Missing Signals (-1)")
plt.ylabel("% Missing")
plt.show()



from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import seaborn as sns
import matplotlib.pyplot as plt

# Aggregate features per sequence
agg_features = train.groupby("sequence_id").agg({
    'acc_x': 'mean', 'acc_y': 'mean', 'acc_z': 'mean',
    'rot_x': 'mean', 'rot_y': 'mean', 'rot_z': 'mean',
    'thm_1': 'mean', 'thm_2': 'mean', 'thm_3': 'mean', 'thm_4': 'mean', 'thm_5': 'mean'
}).reset_index()

# Add gesture labels
gesture_labels = train.groupby("sequence_id")["gesture"].first().reset_index()
agg_features = agg_features.merge(gesture_labels, on="sequence_id")

# Drop rows with any NaN values
agg_features_clean = agg_features.dropna()

# Dimensionality reduction with UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding_umap = reducer.fit_transform(
    agg_features_clean.drop(["sequence_id", "gesture"], axis=1)
)

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=embedding_umap[:, 0],
    y=embedding_umap[:, 1],
    hue=agg_features_clean["gesture"],
    palette="tab20",
    s=50
)
plt.title("UMAP projection of aggregated sensor features by gesture")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



seq_subject_map = train.groupby("sequence_id")["subject"].first().value_counts()
seq_subject_map.plot(kind='bar', figsize=(12, 5), title="Number of Sequences per Subject")
plt.xlabel("Subject")
plt.ylabel("Sequence Count")
plt.tight_layout()
plt.show()



from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Select features and replace NaN values with column means
X = agg_features.drop(["sequence_id", "gesture"], axis=1)
X_filled = X.fillna(X.mean())

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=8, random_state=42)
agg_features['cluster'] = kmeans.fit_predict(X_scaled)

# Visualize first four features
sns.pairplot(agg_features, hue='cluster', vars=X.columns[:4])
plt.suptitle("Clustering of Sequences Using KMeans", y=1.02)
plt.show()



example_sequence = train[train["sequence_id"] == train["sequence_id"].unique()[0]]

fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

axs[0].plot(example_sequence["acc_x"], label="acc_x")
axs[0].plot(example_sequence["acc_y"], label="acc_y")
axs[0].plot(example_sequence["acc_z"], label="acc_z")
axs[0].set_title("IMU - Accelerometer")
axs[0].legend()

axs[1].plot(example_sequence["rot_x"], label="rot_x")
axs[1].plot(example_sequence["rot_y"], label="rot_y")
axs[1].plot(example_sequence["rot_z"], label="rot_z")
axs[1].set_title("IMU - Rotation")
axs[1].legend()

axs[2].plot(example_sequence[["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"]])
axs[2].set_title("Thermopile Sensors")

axs[3].imshow(
    example_sequence[[f"tof_1_v{i}" for i in range(64)]].values.T,
    aspect='auto', cmap='viridis', interpolation='none'
)
axs[3].set_title("Time-of-Flight Sensor 1 - 8x8 Grid Over Time")

plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# Calculate mean per phase per feature
phase_summary = train.groupby('behavior')[['acc_x', 'acc_y', 'acc_z', 'thm_1', 'tof_1_v0']].mean().reset_index()

# Melt for easier plotting
phase_melted = phase_summary.melt(id_vars='behavior', var_name='feature', value_name='mean_value')

plt.figure(figsize=(10, 6))
sns.barplot(data=phase_melted, x='feature', y='mean_value', hue='behavior')
plt.title("Sensor Mean per Feature by Behavior Phase")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import random

# Pick a random sequence
seq_id = random.choice(train['sequence_id'].unique())
df_seq = train[train['sequence_id'] == seq_id]

# Plot time-series for a few key sensors
plt.figure(figsize=(14, 5))
for col in ['acc_x', 'acc_y', 'acc_z']:
    plt.plot(df_seq['sequence_counter'], df_seq[col], label=col)

plt.title(f"Accelerometer Signals for Sequence {seq_id} ({df_seq['behavior'].iloc[-1]})")
plt.xlabel("Time (sequence_counter)")
plt.ylabel("Acceleration")
plt.legend()
plt.tight_layout()
plt.show()



from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Prepare training data
X = agg_features.drop(["sequence_id", "gesture"], axis=1)
y = agg_features["gesture"]
X = X.fillna(X.mean())

# Train classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Plot top features
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
top_features.plot(kind='barh')
plt.title("Top 10 Important Features (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sensor_groups = {
    'IMU only': ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z'],
    'Thermopiles': ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'],
    'ToF avg': [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]
}

# Compute average per sensor (ToF is dense)
agg_features_tof = train.groupby("sequence_id").agg({
    **{col: 'mean' for col in train.columns if "tof" in col}
}).reset_index()
agg_features = agg_features.merge(agg_features_tof, on="sequence_id", how='left')

for name, cols in sensor_groups.items():
    cols_existing = [col for col in cols if col in agg_features.columns]
    X = agg_features[cols_existing].fillna(0)
    y = agg_features["gesture"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    print(f"\n{name} Only:")
    print(classification_report(y_test, clf.predict(X_test)))



print(train["gesture"].dropna().unique())



# Clean whitespace just in case
train["gesture"] = train["gesture"].str.strip()

# Define gestures for comparison
target_gesture = "Above ear - pull hair"
non_target_gesture = "Glasses on/off"

# Filter rows for the two gestures
df_target = train[train["gesture"] == target_gesture]
df_non_target = train[train["gesture"] == non_target_gesture]

# Print basic info
print("Target gesture sequences:", df_target["sequence_id"].nunique())
print("Non-target gesture sequences:", df_non_target["sequence_id"].nunique())



import seaborn as sns
import matplotlib.pyplot as plt

features_to_compare = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z']

for feature in features_to_compare:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_target, x=feature, hue="gesture", element="step", stat="density", common_norm=False)
    plt.title(f"Distribution of {feature} by Gesture")
    plt.tight_layout()
    plt.show()



# Select gestures to compare
target_gesture = "Above ear - pull hair"
non_target_gesture = "Glasses on/off"

# Ensure consistent whitespace handling
train["gesture"] = train["gesture"].str.strip()

# Filter the DataFrame for selected gestures
df_target = train[train["gesture"] == target_gesture]
df_non_target = train[train["gesture"] == non_target_gesture]

# Plot acceleration comparison across gestures
plt.figure(figsize=(16, 6))
for i, axis in enumerate(['acc_x', 'acc_y', 'acc_z']):
    plt.subplot(1, 3, i + 1)
    sns.kdeplot(df_target[axis], label=target_gesture, fill=True)
    sns.kdeplot(df_non_target[axis], label=non_target_gesture, fill=True)
    plt.title(f"{axis} Distribution")
    plt.xlabel(axis)
    plt.ylabel("Density")
    plt.legend()
plt.suptitle("Acceleration Comparison Between Gestures")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


train[['acc_x', 'acc_y', 'acc_z']] = train[['acc_x', 'acc_y', 'acc_z']].fillna(train[['acc_x', 'acc_y', 'acc_z']].mean())

# Compute magnitude of acceleration vector
train['acc_magnitude'] = (train['acc_x']**2 + train['acc_y']**2 + train['acc_z']**2)**0.5

# Aggregate mean acceleration magnitude by gesture
gesture_acc = train.groupby('gesture')['acc_magnitude'].mean().sort_values(ascending=False).reset_index()

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=gesture_acc, x='gesture', y='acc_magnitude', palette='viridis')
plt.xticks(rotation=90)
plt.title("Average Acceleration Magnitude per Gesture")
plt.ylabel("Mean Acceleration Magnitude")
plt.xlabel("Gesture")
plt.tight_layout()
plt.show()


# Fill missing values in thermopile sensor columns
thm_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
train[thm_cols] = train[thm_cols].fillna(train[thm_cols].mean())

# Compute average temperature across thermopiles
train['thm_avg'] = train[thm_cols].mean(axis=1)

# Group by gesture and calculate mean
gesture_thm = train.groupby('gesture')['thm_avg'].mean().sort_values(ascending=False).reset_index()

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=gesture_thm, x='gesture', y='thm_avg', palette='coolwarm')
plt.xticks(rotation=90)
plt.title("Average Temperature (Thermopiles) per Gesture")
plt.ylabel("Mean Temperature (째C)")
plt.xlabel("Gesture")
plt.tight_layout()
plt.show()



# Fill missing values in thermopile sensor columns
thm_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
train[thm_cols] = train[thm_cols].fillna(train[thm_cols].mean())

# Compute average temperature across thermopiles
train['thm_avg'] = train[thm_cols].mean(axis=1)

# Group by gesture and calculate mean
gesture_thm = train.groupby('gesture')['thm_avg'].mean().sort_values(ascending=False).reset_index()

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=gesture_thm, x='gesture', y='thm_avg', palette='coolwarm')
plt.xticks(rotation=90)
plt.title("Average Temperature (Thermopiles) per Gesture")
plt.ylabel("Mean Temperature (째C)")
plt.xlabel("Gesture")
plt.tight_layout()
plt.show()



# Compute phase-wise means
train['thm_avg'] = train[['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']].mean(axis=1)
train['tof_avg'] = train[tof_cols].mean(axis=1)

# Melt for plotting
melted = train.melt(id_vars=['behavior'], value_vars=['thm_avg', 'tof_avg'], var_name='Sensor', value_name='Value')

plt.figure(figsize=(10, 6))
sns.boxplot(data=melted, x='behavior', y='Value', hue='Sensor')
plt.title("Sensor Readings by Sequence Phase (Behavior)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



from scipy.stats import skew, kurtosis, entropy
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.fft import fft

def compute_features(df):
    feature_list = []

    for seq_id, group in df.groupby("sequence_id"):
        feats = {"sequence_id": seq_id}

        for sensor in ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z',
                       'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']:

            # Clean data: replace -1 only if applicable (not general)
            x = group[sensor].copy()
            x = x.replace(-1, np.nan)  # Replace -1 by NaN if -1 is invalid
            x = x.dropna()

            # If empty after dropna, fill with zeros or skip
            if len(x) == 0:
                x = np.array([0])
            else:
                x = x.values

            feats[f"{sensor}_mean"] = np.mean(x)
            feats[f"{sensor}_std"] = np.std(x)
            feats[f"{sensor}_min"] = np.min(x)
            feats[f"{sensor}_max"] = np.max(x)

            # Only calculate skew/kurtosis if variance > 0 and length > 2
            if len(x) > 2 and np.std(x) > 1e-6:
                feats[f"{sensor}_skew"] = skew(x)
                feats[f"{sensor}_kurtosis"] = kurtosis(x)
            else:
                feats[f"{sensor}_skew"] = 0
                feats[f"{sensor}_kurtosis"] = 0

            feats[f"{sensor}_energy"] = np.sum(x ** 2)
            feats[f"{sensor}_zerocross"] = ((np.diff(np.sign(x)) != 0).sum())
            peaks, _ = find_peaks(x, height=np.mean(x) + np.std(x))
            feats[f"{sensor}_n_peaks"] = len(peaks)

            probs, _ = np.histogram(x, bins=10, density=True)
            feats[f"{sensor}_entropy"] = entropy(probs + 1e-6)

        # Similar clean-up for magnitude computations:
        acc_mag = np.sqrt(group["acc_x"]**2 + group["acc_y"]**2 + group["acc_z"]**2)
        acc_mag = acc_mag.replace(-1, np.nan).dropna().values
        if len(acc_mag) == 0:
            acc_mag = np.array([0])
        feats["acc_mag_mean"] = np.mean(acc_mag)

        rot_mag = np.sqrt(group["rot_x"]**2 + group["rot_y"]**2 + group["rot_z"]**2)
        rot_mag = rot_mag.replace(-1, np.nan).dropna().values
        if len(rot_mag) == 0:
            rot_mag = np.array([0])
        feats["rot_mag_mean"] = np.mean(rot_mag)

        # FFT dominant frequency with checks:
        for sensor in ['acc_x', 'acc_y', 'acc_z']:
            x = group[sensor].replace(-1, np.nan).dropna().values
            if len(x) == 0:
                feats[f"{sensor}_fft_dom_freq"] = 0
            else:
                freqs = np.abs(fft(x))
                feats[f"{sensor}_fft_dom_freq"] = np.argmax(freqs[:len(freqs)//2])

        # TOF processing: same, handle missing properly
        tof_cols = [col for col in group.columns if col.startswith("tof_")]
        if tof_cols:
            tof_data = group[tof_cols].replace(-1, 0).fillna(0).values
            feats["tof_active_pixels"] = np.sum(tof_data > 0)
            feats["tof_mean_intensity"] = np.mean(tof_data[tof_data > 0]) if np.sum(tof_data > 0) > 0 else 0
            if tof_data.shape[1] >= 64:
                mask = (tof_data[:, :64] > 0).sum(axis=0)
                if np.sum(mask) > 0:
                    tof_centroid_x = np.average([i % 8 for i in range(64)], weights=mask)
                    tof_centroid_y = np.average([i // 8 for i in range(64)], weights=mask)
                else:
                    tof_centroid_x, tof_centroid_y = 0, 0
                feats["tof_centroid_x"] = tof_centroid_x
                feats["tof_centroid_y"] = tof_centroid_y
        else:
            feats["tof_active_pixels"] = 0
            feats["tof_mean_intensity"] = 0
            feats["tof_centroid_x"] = 0
            feats["tof_centroid_y"] = 0

        # Linear trend with checks:
        for sensor in ['acc_x', 'acc_y', 'acc_z']:
            x = group[sensor].replace(-1, np.nan).dropna().values
            if len(x) > 1:
                lr = LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x)
                feats[f"{sensor}_trend"] = lr.coef_[0]
            else:
                feats[f"{sensor}_trend"] = 0

        feature_list.append(feats)

    return pd.DataFrame(feature_list)

# Usage:
df_features = compute_features(train)



df_features.head()


# Example: df_labels has columns ['sequence_id', 'target']
df_labels = train[['sequence_id', 'gesture']].drop_duplicates()

# Merge features and labels on sequence_id
df_all = df_features.merge(df_labels, on='sequence_id', how='inner')

X = df_all.drop(columns=['sequence_id', 'gesture'])
y = df_all['gesture']



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model (classification example)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_val)

# For classification
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# For regression, use:
# print("Validation RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB

# Your features and target
X = df_features.drop(columns=["sequence_id"])
y = train[["sequence_id", "gesture"]].drop_duplicates().set_index("sequence_id").loc[df_features["sequence_id"]]["gesture"].values

# Step 1: Encode string labels into integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Converts to integer labels 0 ... N-1

# Optional: Get mapping of label to class name
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label mapping:", label_mapping)
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Models dictionary
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RidgeClassifier": RidgeClassifier(),
    "SGDClassifier": SGDClassifier(loss='log_loss', max_iter=1000, random_state=42),
    "SVC_RBF": SVC(kernel='rbf', probability=True, random_state=42),
    "LinearSVC": LinearSVC(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "GaussianNB": GaussianNB(),
    "BernoulliNB": BernoulliNB()
}

# Cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("Training models...\n")
for name, model in models.items():
    accs, f1s = [], []

    for train_idx, val_idx in kf.split(X_scaled, y_encoded):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        accs.append(accuracy_score(y_val, preds))
        f1s.append(f1_score(y_val, preds, average='macro'))

    print(f"{name:15s} | Accuracy: {np.mean(accs):.4f} | F1 Macro: {np.mean(f1s):.4f}")
    results.append({
        "Model": name,
        "Accuracy": np.mean(accs),
        "F1_macro": np.mean(f1s)
    })

# Results DataFrame
results_df = pd.DataFrame(results).sort_values(by="F1_macro", ascending=False)





