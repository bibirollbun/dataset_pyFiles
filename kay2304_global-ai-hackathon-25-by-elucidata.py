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


import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import optuna

def load_h5_data(file_path, dataset_path):
    with h5py.File(file_path, "r") as h5file:
        return {key: np.array(h5file[dataset_path][key]) for key in h5file[dataset_path].keys()}

file_path = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"
train_images = load_h5_data(file_path, "images/Train")
train_spots = load_h5_data(file_path, "spots/Train")
test_images = load_h5_data(file_path, "images/Test")
test_spots = load_h5_data(file_path, "spots/Test")

def create_spot_tables(spots_data):
    return {slide: pd.DataFrame(spots_data[slide]) for slide in spots_data}

train_spot_tables = create_spot_tables(train_spots)
test_spot_table = pd.DataFrame(test_spots['S_7'])

def add_features(df, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['x', 'y']])
    df['density'] = df.groupby('cluster')['x'].transform('count')
    
    # Calculate distance from slide center
    center_x, center_y = df['x'].mean(), df['y'].mean()
    df['distance_to_center'] = np.sqrt((df['x'] - center_x)**2 + (df['y'] - center_y)**2)
    
    # Nearest neighbor distance
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(df[['x', 'y']])
    distances, _ = nn.kneighbors(df[['x', 'y']])
    df['nearest_neighbor_dist'] = distances[:, 1]  # Second closest point (self is closest)
    
    # Local spot density (spots within a 50-pixel radius)
    nn_radius = NearestNeighbors(radius=50)
    nn_radius.fit(df[['x', 'y']])
    density_within_radius = nn_radius.radius_neighbors_graph(df[['x', 'y']]).sum(axis=1)
    df['local_density'] = np.array(density_within_radius).flatten()
    
    return df

train_data = pd.concat(train_spot_tables.values())
train_data = add_features(train_data)
test_spot_table = add_features(test_spot_table)

X = train_data[['x', 'y', 'cluster', 'density', 'distance_to_center', 'nearest_neighbor_dist', 'local_density']]
y = train_data.iloc[:, 2:-5]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_spot_table[['x', 'y', 'cluster', 'density', 'distance_to_center', 'nearest_neighbor_dist', 'local_density']])

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'max_depth': trial.suggest_int('max_depth', 7, 20),
        'num_leaves': trial.suggest_int('num_leaves', 31, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 50)
    }
    model = MultiOutputRegressor(LGBMRegressor(**params, random_state=42))
    model.fit(X_train, y_train)
    return -model.score(X_val, y_val)

study = optuna.create_study()
study.optimize(objective, n_trials=50)
best_params = study.best_params

lgb_model = MultiOutputRegressor(LGBMRegressor(**best_params, random_state=42))
lgb_model.fit(X_scaled, y)

predictions = lgb_model.predict(X_test_scaled)
predicted_labels = pd.DataFrame(predictions, columns=y.columns)

submission_df = predicted_labels.copy()
submission_df.insert(0, 'ID', test_spot_table.index)
submission_df.to_csv("./submission.csv", index=False)
print("Submission file created!")


