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


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


train


train.isna().sum()


train.describe()


train = train.drop(columns="id")


train.hist(bins=50, figsize=(15, 10))
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()



import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(train.corr(), cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()



features = ["RhythmScore", "AudioLoudness", "VocalContent", "AcousticQuality",
            "InstrumentalScore", "LivePerformanceLikelihood", "MoodScore", 
            "Energy", "TrackDurationMs"]

target = "BeatsPerMinute" 


import numpy as np

for col in features:
    temp = train[[col, "BeatsPerMinute"]].copy()
    temp["bin"] = pd.qcut(temp[col], q=50, duplicates="drop")  # 50 quantile bins
    bin_means = temp.groupby("bin").mean(numeric_only=True)
    
    plt.figure(figsize=(8,4))
    plt.plot(bin_means[col], bin_means[target], marker="o")
    plt.xlabel(col)
    plt.ylabel(target)
    plt.title(f"{col} vs {target} (Binned Mean)")
    plt.show()



from sklearn.cluster import KMeans

# More interaction features
train['Loudness_x_Energy'] = train['AudioLoudness'] * train['Energy']
train['Acoustic_x_Instrumental'] = train['AcousticQuality'] * train['InstrumentalScore']

# Create clusters based on acoustic properties
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
acoustic_features = ['AcousticQuality', 'VocalContent', 'InstrumentalScore', 'Energy']
train['Acoustic_Cluster'] = kmeans.fit_predict(train[acoustic_features])

# --- Don't forget to apply the same transformations to the test set! ---
test['Loudness_x_Energy'] = test['AudioLoudness'] * test['Energy']
test['Acoustic_x_Instrumental'] = test['AcousticQuality'] * test['InstrumentalScore']
test['Acoustic_Cluster'] = kmeans.predict(test[acoustic_features])

# Update your features list to include the new ones
features.extend(['Loudness_x_Energy', 'Acoustic_x_Instrumental', 'Acoustic_Cluster'])


for col in train.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=train[col])
    plt.title(f"Outliers in {col}")
    plt.show()


def cap_outliers(train, features):
    for col in features:
        Q1 = train[col].quantile(0.25)
        Q3 = train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        train[col] = np.where(train[col] < lower, lower,
        np.where(train[col] > upper, upper, train[col]))
    return train

# In Cell 13, modify the last line
features_to_cap = [col for col in train.columns if col != 'BeatsPerMinute']
train = cap_outliers(train, features_to_cap)

# Now, when you define X and Y in Cell 15, the target 'Y' will be the original, uncapped values.
X = train[features] # Your original features list
Y = train[target]


for col in train.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=train[col])
    plt.title(f"Outliers in {col}")
    plt.show()


X


train['TrackDurationMin'] = train['TrackDurationMs'] / 60000
train['Energy_Acoustic_Ratio'] = train['Energy'] / (train['AcousticQuality'] + 1e-5)
train['Vocal_Instrument_Balance'] = train['VocalContent'] / (train['InstrumentalScore'] + 1e-5)
train['MoodRhythm'] = train['MoodScore'] * train['RhythmScore']
train['PerformanceIntensity'] = train['LivePerformanceLikelihood'] * train['AudioLoudness']
train['RhythmEnergy'] = train['RhythmScore'] * train['Energy']
train['MoodAcoustic'] = train['MoodScore'] * train['AcousticQuality']


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train1 = scaler.fit_transform(train)
train1 = pd.DataFrame(train, columns=train.columns, index=train.index)


from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
X = train1.drop(columns=[target])
y = train1[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 1. Define GPU-enabled XGBoost model
xgb = XGBRegressor(
    objective="reg:squarederror",
    tree_method="gpu_hist",       # GPU training
    predictor="gpu_predictor",    # GPU prediction
    random_state=42
)

# 2. Define parameter search space
param_dist = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

# 3. RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=5,            # keep small for speed
    scoring="neg_root_mean_squared_error",
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# 4. Fit search
random_search.fit(X_train, y_train)

# 5. Best model
best_model = random_search.best_estimator_


best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))


plt.figure(figsize=(10,6))
plt.barh(X.columns, best_model.feature_importances_)
plt.title("XGBoost Feature Importance")
plt.show()


import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


cat_model = CatBoostRegressor(
    loss_function='RMSE',       # regression loss
    random_seed=42,
    verbose=0,               # suppress training output
    task_type="GPU" 
)

param_grid = {
    'depth': [6, 8,10],
    'learning_rate': [0.01,  0.1],
    'iterations': [100, 500, 1000],
    'l2_leaf_reg': [1,  5]
}

grid = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2
)

grid.fit(X_train, y_train)

best_model_cat = grid.best_estimator_
print("✅ Best Parameters:", grid.best_params_)


y_pred_cat = best_model_cat.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_cat))
r2 = r2_score(y_test, y_pred_cat)

print(f"✅ Tuned CatBoost RMSE: {rmse:.4f}")
print(f"✅ Tuned CatBoost R² Score: {r2:.4f}")


from sklearn.metrics import mean_squared_error
import numpy as np

# Stack validation predictions
y_true = y_test.values  # or your y array

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

best_rmse = 1e9
best_weights = None

# Grid search over weights (step size can be refined later)
for w1 in np.linspace(0,1,21):
    for w2 in np.linspace(0,1-w1,21):
        y_pred = w1*y_pred + w2*y_pred_cat
        score = rmse(y_true, y_pred)
        if score < best_rmse:
            best_rmse = score
            best_weights = (w1,w2)

print("Best RMSE:", best_rmse)
print("Best Weights:", best_weights)



test['TrackDurationMin'] = test['TrackDurationMs'] / 60000
test['Energy_Acoustic_Ratio'] = test['Energy'] / (test['AcousticQuality'] + 1e-5)
test['Vocal_Instrument_Balance'] = test['VocalContent'] / (test['InstrumentalScore'] + 1e-5)
test['MoodRhythm'] = test['MoodScore'] * test['RhythmScore']
test['PerformanceIntensity'] = test['LivePerformanceLikelihood'] * test['AudioLoudness']
test['RhythmEnergy'] = test['RhythmScore'] * test['Energy']
test['MoodAcoustic'] = test['MoodScore'] * test['AcousticQuality']

train_features = best_model.get_booster().feature_names  # features used in training

X_test_final = test[train_features]  # select only training features (drop id automatically)

y_xg_pred_test = best_model.predict(X_test_final)
y_cat_pred_test = best_model_cat.predict(X_test_final)
y_pred_test = y_xg_pred_test*0+y_cat_pred_test*1

output = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": y_pred_test
})

output.to_csv("submission.csv" , index=False)

print("Predictions saved to test_predictions.csv")
print(output.head())





