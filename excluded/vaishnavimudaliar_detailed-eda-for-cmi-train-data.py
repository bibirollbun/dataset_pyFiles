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





