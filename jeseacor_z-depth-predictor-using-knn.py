# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''


test_df = pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/test.csv")
train_df = pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/train.csv")


train_df.head(5)
input_cols = [str(i) for i in range(-299, 1)]
target_cols = [str(i) for i in range(1, 301)]


num_samples = 100

z_input = train_df.loc[:num_samples-1, input_cols].values
z_target = train_df.loc[:num_samples-1, target_cols].values

z_combined = np.hstack([z_input, z_target])

plt.figure(figsize=(18, 8))
plt.imshow(z_combined, aspect='auto', cmap='Oranges', origin='lower')
plt.axvline(x=299, color='black', linestyle='--', linewidth=1.5, label="Current Drill Position (0)")
plt.colorbar(label='Z-depth (oriented up)')
plt.title('Combined Heatmap: Input (-299 to 0) and Target (1 to 300) Z-Depth Sequences')
plt.xlabel('Position Index (-299 to 300)')
plt.ylabel('Sample Index')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


X = train_df[input_cols].fillna(0)
y = train_df[target_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])

knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(n_neighbors=10, weights='uniform', metric='manhattan'))])
knn_model = knn_pipeline.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_knn)
rmse = np.sqrt(mse)
print(f"RMSE : {rmse:.2f}")


input_cols = [str(i) for i in range(-299, 1)]
X_submit = test_df[input_cols]

y_pred = knn_model.predict(X_submit)

y_pred_10x = np.stack([y_pred] * 10, axis=0) 

realization_frames = []

for i in range(10):
    if i == 0:
        cols = [str(j+1) for j in range(300)]
    else:
        cols = [f"r_{i}_pos_{j+1}" for j in range(300)]
    df = pd.DataFrame(y_pred_10x[i], columns=cols)
    realization_frames.append(df)

realizations_df = pd.concat(realization_frames, axis=1)

realizations_df.insert(0, 'geology_id', test_df['geology_id'])

realizations_df.to_csv("submission.csv", index=False)

