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


# ---------------------------
# Import libraries
# ---------------------------
from sklearn.neighbors import KNeighborsRegressor
from functools import reduce
import pandas as pd

# ---------------------------
# Define sources and weights
# ---------------------------
sources = {
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submission17.csv": 0.3,
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submission16.csv": 0.5,
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submission13.csv": 0.3,
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submissionothers.csv": 2.0,
}

# ---------------------------
# Read and merge submissions
# ---------------------------
dfs = []
for i, (src, w) in enumerate(sources.items(), start=1):
    df = pd.read_csv(src).rename(columns={"BeatsPerMinute": f"BeatsPerMinute_{i}"})
    dfs.append(df)

# Merge on 'id'
df = reduce(lambda left, right: left.merge(right, on="id"), dfs)

# ==================================================
# ⚖️ Blend Predictions (KNN if 3+ sources)
# ==================================================
if len(sources) >= 3:
    # Feature matrix X = predictions from all submissions
    feature_cols = [col for col in df.columns if col.startswith("BeatsPerMinute_")]
    X = df[feature_cols]

    # Pseudo-target = weighted mean of predictions (soft supervision)
    weights = list(sources.values())
    total = sum(weights)
    norm_weights = [w / total for w in weights]
    y = sum(norm_weights[i] * df[f"BeatsPerMinute_{i+1}"] for i in range(len(norm_weights)))

    # Initialize KNN regressor
    knn = KNeighborsRegressor(n_neighbors=5, weights="distance")

    # Fit KNN on base predictions
    knn.fit(X, y)

    # Predict final BPM using KNN blending
    df["BeatsPerMinute"] = knn.predict(X)
    print("✅ Used KNN blending for final ensemble.")
else:
    # Fallback: simple weighted average
    weights = list(sources.values())
    total = sum(weights)
    norm_weights = [w / total for w in weights]
    df["BeatsPerMinute"] = sum(
        norm_weights[i] * df[f"BeatsPerMinute_{i+1}"] for i in range(len(norm_weights))
    )
    print("⚠️ Fallback to weighted average (less than 3 models).")

# ---------------------------
# Save final submission
# ---------------------------
final = df[["id", "BeatsPerMinute"]]
final.to_csv("/kaggle/working/submission1.csv", index=False)
print("✅ Ensemble submission saved as submission1.csv")


# ---------------------------
# Import libraries
# ---------------------------
from sklearn.neighbors import KNeighborsRegressor
from functools import reduce
import pandas as pd

# ---------------------------
# Define sources and weights
# ---------------------------
sources = {
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submissionothers.csv": 0.5,
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submission16.csv": 0.1,
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submissionothers.csv": 0.5,
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submissionothers.csv": 2.0,
}

# ---------------------------
# Read and merge submissions
# ---------------------------
dfs = []
for i, (src, w) in enumerate(sources.items(), start=1):
    df = pd.read_csv(src).rename(columns={"BeatsPerMinute": f"BeatsPerMinute_{i}"})
    dfs.append(df)

# Merge on 'id'
df = reduce(lambda left, right: left.merge(right, on="id"), dfs)

# ==================================================
# ⚖️ Blend Predictions (KNN if 3+ sources)
# ==================================================
if len(sources) >= 3:
    # Feature matrix X = predictions from all submissions
    feature_cols = [col for col in df.columns if col.startswith("BeatsPerMinute_")]
    X = df[feature_cols]

    # Pseudo-target = weighted mean of predictions (soft supervision)
    weights = list(sources.values())
    total = sum(weights)
    norm_weights = [w / total for w in weights]
    y = sum(norm_weights[i] * df[f"BeatsPerMinute_{i+1}"] for i in range(len(norm_weights)))

    # Initialize KNN regressor
    knn = KNeighborsRegressor(n_neighbors=5, weights="distance")

    # Fit KNN on base predictions
    knn.fit(X, y)

    # Predict final BPM using KNN blending
    df["BeatsPerMinute"] = knn.predict(X)
    print("✅ Used KNN blending for final ensemble.")
else:
    # Fallback: simple weighted average
    weights = list(sources.values())
    total = sum(weights)
    norm_weights = [w / total for w in weights]
    df["BeatsPerMinute"] = sum(
        norm_weights[i] * df[f"BeatsPerMinute_{i+1}"] for i in range(len(norm_weights))
    )
    print("⚠️ Fallback to weighted average (less than 3 models).")

# ---------------------------
# Save final submission
# ---------------------------
final = df[["id", "BeatsPerMinute"]]
final.to_csv("/kaggle/working/submission2.csv", index=False)
print("✅ Ensemble submission saved as submission2.csv")


# ---------------------------
# Import libraries
# ---------------------------
from sklearn.neighbors import KNeighborsRegressor
from functools import reduce
import pandas as pd

# ---------------------------
# Define sources and weights
# ---------------------------
sources = {
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submission20.csv": 0.5,
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submissionothers.csv": 2.0,
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submission19.csv": 0.1,
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submissionothers.csv": 2.0,
}

# ---------------------------
# Read and merge submissions
# ---------------------------
dfs = []
for i, (src, w) in enumerate(sources.items(), start=1):
    df = pd.read_csv(src).rename(columns={"BeatsPerMinute": f"BeatsPerMinute_{i}"})
    dfs.append(df)

# Merge on 'id'
df = reduce(lambda left, right: left.merge(right, on="id"), dfs)

# ==================================================
# ⚖️ Blend Predictions (KNN if 3+ sources)
# ==================================================
if len(sources) >= 3:
    # Feature matrix X = predictions from all submissions
    feature_cols = [col for col in df.columns if col.startswith("BeatsPerMinute_")]
    X = df[feature_cols]

    # Pseudo-target = weighted mean of predictions (soft supervision)
    weights = list(sources.values())
    total = sum(weights)
    norm_weights = [w / total for w in weights]
    y = sum(norm_weights[i] * df[f"BeatsPerMinute_{i+1}"] for i in range(len(norm_weights)))

    # Initialize KNN regressor
    knn = KNeighborsRegressor(n_neighbors=5, weights="distance")

    # Fit KNN on base predictions
    knn.fit(X, y)

    # Predict final BPM using KNN blending
    df["BeatsPerMinute"] = knn.predict(X)
    print("✅ Used KNN blending for final ensemble.")
else:
    # Fallback: simple weighted average
    weights = list(sources.values())
    total = sum(weights)
    norm_weights = [w / total for w in weights]
    df["BeatsPerMinute"] = sum(
        norm_weights[i] * df[f"BeatsPerMinute_{i+1}"] for i in range(len(norm_weights))
    )
    print("⚠️ Fallback to weighted average (less than 3 models).")

# ---------------------------
# Save final submission
# ---------------------------
final = df[["id", "BeatsPerMinute"]]
final.to_csv("/kaggle/working/submissio3.csv", index=False)
print("✅ Ensemble submission saved as submission3.csv")


# ---------------------------
# Import libraries
# ---------------------------
from sklearn.neighbors import KNeighborsRegressor
from functools import reduce
import pandas as pd

# ---------------------------
# Define sources and weights
# ---------------------------
sources = {
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submission21.csv": 0.8,
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submissionothers.csv": 0.2,
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submission21.csv": 1.2,
    "/kaggle/input/predicting-the-beats-per-minute-of-songs-sub-files/submissionothers.csv": 2.0,
}

# ---------------------------
# Read and merge submissions
# ---------------------------
dfs = []
for i, (src, w) in enumerate(sources.items(), start=1):
    df = pd.read_csv(src).rename(columns={"BeatsPerMinute": f"BeatsPerMinute_{i}"})
    dfs.append(df)

# Merge on 'id'
df = reduce(lambda left, right: left.merge(right, on="id"), dfs)

# ==================================================
# ⚖️ Blend Predictions (KNN if 3+ sources)
# ==================================================
if len(sources) >= 3:
    # Feature matrix X = predictions from all submissions
    feature_cols = [col for col in df.columns if col.startswith("BeatsPerMinute_")]
    X = df[feature_cols]

    # Pseudo-target = weighted mean of predictions (soft supervision)
    weights = list(sources.values())
    total = sum(weights)
    norm_weights = [w / total for w in weights]
    y = sum(norm_weights[i] * df[f"BeatsPerMinute_{i+1}"] for i in range(len(norm_weights)))

    # Initialize KNN regressor
    knn = KNeighborsRegressor(n_neighbors=5, weights="distance")

    # Fit KNN on base predictions
    knn.fit(X, y)

    # Predict final BPM using KNN blending
    df["BeatsPerMinute"] = knn.predict(X)
    print("✅ Used KNN blending for final ensemble.")
else:
    # Fallback: simple weighted average
    weights = list(sources.values())
    total = sum(weights)
    norm_weights = [w / total for w in weights]
    df["BeatsPerMinute"] = sum(
        norm_weights[i] * df[f"BeatsPerMinute_{i+1}"] for i in range(len(norm_weights))
    )
    print("⚠️ Fallback to weighted average (less than 3 models).")

# ---------------------------
# Save final submission
# ---------------------------
final = df[["id", "BeatsPerMinute"]]
final.to_csv("/kaggle/working/submission.csv", index=False)
print("✅ Ensemble submission saved as submission.csv")

