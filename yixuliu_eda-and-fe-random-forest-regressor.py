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


# EDA and FE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Regression model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Random Forest (tree-based) model
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error, make_scorer
import joblib

# LightGBM
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer, mean_squared_error


# # 1. Data‐Quality Checks


# 1.1 Load your training data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df = df_train.copy()  

df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# 1.2 See the schema and non-null counts
df.info()
df_test.info()

# 1.3 Count missing values per column
print(df.isnull().sum())



# # 2. Univariate Exploration

# 2.1 Quick summary statistics
print(df[['Age','Height','Weight','Duration','Heart_Rate','Body_Temp','Calories']].describe())

# 2.2 Histograms to see shapes
import matplotlib.pyplot as plt

df[['Age','Height','Weight','Duration','Heart_Rate','Body_Temp','Calories']] \
    .hist(bins=30, figsize=(12,8))
plt.tight_layout()
plt.show()

# 2.3 Boxplots to highlight outliers
df[['Age','Height','Weight','Duration','Heart_Rate','Body_Temp','Calories']] \
    .plot.box(subplots=True, layout=(2,4), figsize=(12,6))
plt.tight_layout()
plt.show()



# # 3. Group Comparisons by Sex

# 3.1 Compute mean & std for each numeric feature, grouped by Sex
group_stats = df.groupby('Sex')[['Age','Height','Weight','Duration','Heart_Rate','Body_Temp','Calories']] \
                .agg(['mean','std','count'])
print(group_stats)

# 3.2 Visualize Calories distribution by Sex
df.boxplot(column='Calories', by='Sex', figsize=(6,4))
plt.title('Calories by Sex')
plt.suptitle('')    # remove automatic subtitle
plt.ylabel('Calories')
plt.show()


# # 4. Bivariate Relationships

# Pick numeric features
features = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp','Calories']

# Compute correlation
corr = df[features].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.matshow(corr, fignum=1)
plt.xticks(range(len(features)), features, rotation=90)
plt.yticks(range(len(features)), features)
plt.colorbar()
plt.title('Correlation Matrix Heatmap', pad=20)
plt.show()


# Draw scatter-matrix
scatter_matrix(df[features], alpha=0.2, diagonal='hist', figsize=(12, 12))
plt.suptitle('Scatter Matrix of Features', y=0.9)
plt.show()


# # 5. Multivariate Patterns

fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(df['Duration'], df['Heart_Rate'], df['Calories'], alpha=0.2)
ax.set_xlabel('Duration (min)')
ax.set_ylabel('Heart Rate (bpm)')
ax.set_zlabel('Calories Burned')
plt.title('3D Scatter: Duration vs Heart Rate vs Calories')
plt.show()

plt.show()


fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(df['Duration'], df['Body_Temp'], df['Calories'], alpha=0.2)
ax.set_xlabel('Duration (min)')
ax.set_ylabel('Body Temperature (°C)')
ax.set_zlabel('Calories Burned')
plt.title('3D Scatter: Duration vs Body Temp vs Calories')
plt.show()


fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(df['Heart_Rate'], df['Body_Temp'], df['Calories'], alpha=0.2)
ax.set_xlabel('Heart Rate (bpm)')
ax.set_ylabel('Body Temperature (°C)')
ax.set_zlabel('Calories Burned')
plt.title('3D Scatter: Heart Rate vs Body Temp vs Calories')
plt.show()


# # 6. Derived‐Feature Ideas

# create the interaction feature
df['dur_temp'] = df['Duration'] * df['Body_Temp']
# correlation with target
print(df['dur_temp'].corr(df['Calories']))


# 1.1 baseline = 37.0
# df['norm_intensity'] = (df['Body_Temp'] - baseline) / df['Duration'].replace(0, np.nan)
# df['norm_intensity'].fillna(0, inplace=True)   # treat zero-duration as zero intensity
# print(df['norm_intensity'].describe())
# print("Corr with Calories:", df['norm_intensity'].corr(df['Calories']))

# Why it’s negative: By dividing the Δtemp by Duration, longer workouts (even if they burn lots of calories) yield smaller per-minute heat-up rates—and so the ratio actually decreases as Duration grows.

# What to try instead:
# Swap numerator/denominator: Duration / (Body_Temp–37) will correlate positively. Or drop the division entirely and just use ΔTemp (Body_Temp–37) as a standalone feature.


# 1.2 Define a resting‐temp baseline
baseline = 37.0

# 1.3 Compute normalized intensity
#    - Subtract baseline from Body_Temp
#    - Divide by Duration, but avoid dividing by zero
df['norm_intensity'] = (df['Body_Temp'] - baseline) / df['Duration'].replace(0, np.nan)

# 1.4 Replace any NaN (from zero-duration) with 0
df['norm_intensity'] = df['norm_intensity'].fillna(0)

# 1.5 Check the correlation with Calories
corr1 = df['norm_intensity'].corr(df['Calories'])
print("Corr((Body_Temp–37)/Duration, Calories) =", corr1)



# 2.1 Compute the flipped ratio
df['dur_over_delta'] = df['Duration'] / (df['Body_Temp'] - baseline).replace(0, np.nan)

# 2.2 Again, fill any infinities / NaNs (e.g. if Body_Temp==37)
df['dur_over_delta'] = df['dur_over_delta'].replace([np.inf, -np.inf], np.nan).fillna(0)

# 2.3 Check its correlation
corr2 = df['dur_over_delta'].corr(df['Calories'])
print("Corr(Duration/(Body_Temp–37), Calories) =", corr2)




# 3.1 Compute raw temperature rise
df['delta_temp'] = df['Body_Temp'] - baseline

# 3.2 Correlation with Calories
corr3 = df['delta_temp'].corr(df['Calories'])
print("Corr(Body_Temp–37, Calories) =", corr3)




# flag when body temp exceeds 38°C
df['temp_high'] = (df['Body_Temp'] > 38.0).astype(int)

# flag when heart rate exceeds 100 bpm
df['hr_high']   = (df['Heart_Rate'] > 100).astype(int)

# average calories burned when flag is on vs off
print(df.groupby('temp_high')['Calories'].mean())
print(df.groupby('hr_high')['Calories'].mean())

# Average calories burned when temp_high is off vs on
print("Temp ≤ 38 °C  →", df.groupby('temp_high')['Calories'].mean()[0], "kcal")
print("Temp >  38 °C →", df.groupby('temp_high')['Calories'].mean()[1], "kcal\n")

# Average calories burned when hr_high is off vs on
print("HR ≤ 100 bpm →", df.groupby('hr_high')['Calories'].mean()[0], "kcal")
print("HR >  100 bpm →", df.groupby('hr_high')['Calories'].mean()[1], "kcal")


# How many sessions are high vs low?
print(df['temp_high'].value_counts(), "\n")
print(df['hr_high'].value_counts(), "\n")

# A quick bar plot
df['temp_high'].value_counts().sort_index().plot.bar(
    title='Count of Sessions by temp_high', xlabel='temp_high', ylabel='Count')
plt.show()

df['hr_high'].value_counts().sort_index().plot.bar(
    title='Count of Sessions by hr_high', xlabel='hr_high', ylabel='Count')
plt.show()





# 1) Extract HR into a 2D array
hr = df['Heart_Rate'].values.reshape(-1,1)

# 2) Fit KMeans with k clusters
k = 4
km = KMeans(n_clusters=k, random_state=42).fit(hr)
labels = km.labels_
centroids = km.cluster_centers_.flatten()

# 3) Order clusters by centroid value so labels go 0=lowest, 3=highest
order = np.argsort(centroids)
new_label = {old: new for new, old in enumerate(order)}
df['hr_zone_km'] = [ new_label[l] for l in labels ]

# 4) Check the cluster centers & burn rates
print("Cluster centers (bpm):", sorted(centroids))
print(df.groupby('hr_zone_km')['Calories'].mean())



# BMI = Weight / (Height/100)²: often a better health indicator than raw height or weight.

# Age groups: bin into decades if you suspect non-linear age effects.


# 2. Compute BMI
#    BMI = weight (kg) / [height (m)]²
df['BMI'] = df['Weight'] / ( (df['Height'] / 100) ** 2 )

# 3. Bin Age into decades
#    Creates labels “10s”, “20s”, … up to “80s”
age_bins  = list(range(10, 100, 10))           # [10,20,30,…,90]
age_labels = [f'{b}s' for b in age_bins[:-1]]  # ['10s','20s',…,'80s']
df['age_decade'] = pd.cut(
    df['Age'],
    bins=age_bins,
    right=False,
    labels=age_labels
)

# 4. Quick checks
print("BMI ↔ Calories corr:", df['BMI'].corr(df['Calories']))
print("\nMean Calories by age_decade:")
print(df.groupby('age_decade')['Calories'].mean())


# pick your feature and reshape
X = df['Age'].values.reshape(-1,1)

# fit k clusters
k = 5
km = KMeans(n_clusters=k, random_state=42).fit(X)

# get the raw labels and centroids
labels    = km.labels_
centroids = km.cluster_centers_.flatten()

# order them so bin 0 is the lowest centroid, etc.
order_map = {old: new for new, old in enumerate(np.argsort(centroids))}
df['age_km_bin'] = [order_map[l] for l in labels]

# see what ranges each cluster covers
ranges = df.groupby('age_km_bin')['Age'].agg(['min','max','count'])
print("Cluster centers:", sorted(centroids))
print(ranges)



# # 7. Check for Multicollinearity

# 1) Build your feature matrix as before
X = pd.concat([
    df[['dur_temp','dur_over_delta','delta_temp','BMI']],
    pd.get_dummies(df['hr_zone_km'], prefix='hr',    drop_first=True),
    pd.get_dummies(df['age_km_bin'], prefix='age',   drop_first=True),
], axis=1)

# 2) Ensure everything is numeric and finite
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X = X.astype(float)   # <-- force every column to float

# 3) (Optional) add constant
X['const'] = 1.0

# 4) Compute VIFs
vif_data = pd.DataFrame({
    'feature': X.columns,
    'VIF':     [variance_inflation_factor(X.values, i)
                for i in range(X.shape[1])]
})

print(vif_data.sort_values('VIF').reset_index(drop=True))


# 1.copy data
df_RF  = df_train.copy()

# 2. Re-create the k-means bins from  EDA

# 2.1 Heart-Rate zones (4 clusters)
hr_km = KMeans(n_clusters=4, random_state=42).fit(df_RF[['Heart_Rate']])
df_RF['hr_zone_km'] = hr_km.predict(df_RF[['Heart_Rate']])
# reorder labels so that 0 < 1 < 2 < 3 in terms of centroid value
hr_centroids = hr_km.cluster_centers_.flatten()
hr_order     = np.argsort(hr_centroids)
hr_map       = {old: new for new, old in enumerate(hr_order)}
df_RF['hr_zone_km'] = df_RF['hr_zone_km'].map(hr_map)

# 2.2 Age bins (5 clusters)
age_km = KMeans(n_clusters=5, random_state=42).fit(df_RF[['Age']])
df_RF['age_km_bin'] = age_km.predict(df_RF[['Age']])
age_centroids = age_km.cluster_centers_.flatten()
age_order     = np.argsort(age_centroids)
age_map       = {old: new for new, old in enumerate(age_order)}
df_RF['age_km_bin'] = df_RF['age_km_bin'].map(age_map)

# 3. Recompute continuous FE
df_RF['dur_temp']       = df_RF['Duration'] * df_RF['Body_Temp']
df_RF['delta_temp']     = df_RF['Body_Temp'] - 37
df_RF['dur_over_delta'] = df_RF['Duration'] / df_RF['delta_temp'].replace(0, np.nan)

# 4. Select features & target
X = df_RF[['dur_temp','dur_over_delta','delta_temp','hr_zone_km','age_km_bin']]
y = df_RF['Calories']

# 5. Build pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', sparse=False), ['hr_zone_km','age_km_bin']),
], remainder='passthrough')

rmsle_scorer = make_scorer(
    lambda y_true, y_pred: np.sqrt(mean_squared_log_error(y_true, y_pred)),
    greater_is_better=False
)

rf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )),
])

# 6. CV with RMSLE
scores = cross_val_score(
    rf_pipeline, X, y,
    cv=5,
    scoring=rmsle_scorer,
    error_score='raise'
)
print(f"RF RMSLE: {(-scores.mean()):.4f} ± {scores.std():.4f}")

# 7. Fit & save
rf_pipeline.fit(X, y)
joblib.dump(rf_pipeline, 'rf_calorie_pipeline.pkl')


# 1. Load trained pipeline
rf_pipeline = joblib.load('rf_calorie_pipeline.pkl')

# 2. Reload training data to refit your K-Means bins
df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')

# Heart-rate K-Means (4 clusters)
hr_km = KMeans(n_clusters=4, random_state=42)
hr_km.fit(df_train[['Heart_Rate']])
hr_centroids = hr_km.cluster_centers_.flatten()
hr_order     = np.argsort(hr_centroids)
hr_map       = {old: new for new, old in enumerate(hr_order)}

# Age K-Means (5 clusters)
age_km = KMeans(n_clusters=5, random_state=42)
age_km.fit(df_train[['Age']])
age_centroids = age_km.cluster_centers_.flatten()
age_order     = np.argsort(age_centroids)
age_map       = {old: new for new, old in enumerate(age_order)}

# 3. Load & FE the test set
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_test['dur_temp']       = df_test['Duration'] * df_test['Body_Temp']
df_test['delta_temp']     = df_test['Body_Temp'] - 37
df_test['dur_over_delta'] = df_test['Duration'] / df_test['delta_temp'].replace(0, np.nan)

# 3a. Assign clusters
df_test['hr_zone_km'] = hr_km.predict(df_test[['Heart_Rate']])
df_test['hr_zone_km'] = df_test['hr_zone_km'].map(hr_map)

df_test['age_km_bin'] = age_km.predict(df_test[['Age']])
df_test['age_km_bin'] = df_test['age_km_bin'].map(age_map)

# 4. Select the same feature columns
X_test = df_test[['dur_temp','dur_over_delta','delta_temp','hr_zone_km','age_km_bin']]

# 5. Predict
preds = rf_pipeline.predict(X_test)

# 6. Build & save submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission['Calories'] = preds
submission.to_csv('submission.csv', index=False)

print("Wrote submission.csv — ready to submit!")


# # 0. Custom RMSLE scorer
# rmse_scorer = make_scorer(
#     mean_squared_error,
#     greater_is_better=False,
#     squared=False    # squared=False makes it return the root MSE
# )

# # 1. Load & copy
# df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
# df_LightGBM = df.copy()

# # 2. Rebuild your k-means bins (with explicit n_init to silence warnings)
# hr_km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(df_LightGBM[['Heart_Rate']])
# age_km = KMeans(n_clusters=5, n_init=10, random_state=42).fit(df_LightGBM[['Age']])

# # 2a. Map labels so they ascend by centroid value
# def map_clusters(km, arr):
#     cents   = km.cluster_centers_.flatten()
#     order   = np.argsort(cents)
#     mapping = {old: new for new, old in enumerate(order)}
#     # apply mapping in pure Python → returns NumPy array
#     return np.array([mapping[x] for x in arr])

# df_LightGBM['hr_zone_km'] = map_clusters(
#     hr_km, hr_km.predict(df_LightGBM[['Heart_Rate']])
# )
# df_LightGBM['age_km_bin'] = map_clusters(
#     age_km, age_km.predict(df_LightGBM[['Age']])
# )


# # 3. Feature engineering
# df_LightGBM['dur_temp']       = df_LightGBM['Duration'] * df_LightGBM['Body_Temp']
# df_LightGBM['delta_temp']     = df_LightGBM['Body_Temp'] - 37
# df_LightGBM['dur_over_delta'] = df_LightGBM['Duration'] / df_LightGBM['delta_temp'].replace(0, np.nan)

# # 4. Split X/y
# X = df_LightGBM[['dur_temp','dur_over_delta','delta_temp','hr_zone_km','age_km_bin']]
# y = df_LightGBM['Calories']

# # 5. Preprocessor: one-hot your cluster bins
# preprocessor = ColumnTransformer([
#     ('cat', OneHotEncoder(drop='first', sparse_output=False), ['hr_zone_km','age_km_bin']),
# ], remainder='passthrough')

# # 6. GPU-powered LightGBM regressor
# lgbm = LGBMRegressor(
#     device='gpu',            
#     gpu_platform_id=0,       
#     gpu_device_id=0,         
#     n_estimators=500,
#     learning_rate=0.05,
#     num_leaves=31,
#     max_depth=-1,
#     random_state=42
# )

# pipeline = Pipeline([
#     ('prep', preprocessor),
#     ('lgb',  lgbm),
# ])

# # 7. Cross-validate with RMSLE
# rmsle_scorer = make_scorer(
#     lambda yt, yp: np.sqrt(mean_squared_log_error(yt, np.maximum(0, yp))),
#     greater_is_better=False
# )

# lgb_rmsle = -cross_val_score(
#     pipeline, X, y,
#     cv=5,
#     scoring=rmsle_scorer,
#     n_jobs=-1
# ).mean()
# print(f"LGBM RMSLE: {lgb_rmsle:.4f}")

# # 8. Fit on all data
# pipeline.fit(X, y)

# # 9. Save for later
# joblib.dump(pipeline, 'lgbm_calorie_gpu.pkl')


# # Load saved LightGBM pipeline
# pipeline = joblib.load('lgbm_calorie_gpu.pkl')

# # Reload training data to refit K-means (for consistent bins)
# df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
# hr_km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(df_train[['Heart_Rate']])
# age_km = KMeans(n_clusters=5, n_init=10, random_state=42).fit(df_train[['Age']])
# hr_centroids = hr_km.cluster_centers_.flatten()
# hr_order = hr_centroids.argsort()
# hr_map = {old:new for new,old in enumerate(hr_order)}
# age_centroids = age_km.cluster_centers_.flatten()
# age_order = age_centroids.argsort()
# age_map = {old:new for new,old in enumerate(age_order)}

# # Load & feature-engineer test set
# df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
# df_test['dur_temp']       = df_test['Duration'] * df_test['Body_Temp']
# df_test['delta_temp']     = df_test['Body_Temp'] - 37
# df_test['dur_over_delta'] = df_test['Duration'] / df_test['delta_temp'].replace(0, pd.NA)

# # Assign clusters
# df_test['hr_zone_km']  = hr_km.predict(df_test[['Heart_Rate']])
# df_test['hr_zone_km']  = df_test['hr_zone_km'].map(hr_map)
# df_test['age_km_bin']  = age_km.predict(df_test[['Age']])
# df_test['age_km_bin']  = df_test['age_km_bin'].map(age_map)

# # Predict
# features = ['dur_temp','dur_over_delta','delta_temp','hr_zone_km','age_km_bin']
# preds = pipeline.predict(df_test[features])

# # Build submission
# submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
# submission['Calories'] = preds
# submission.to_csv('submission.csv', index=False)
# print("submission.csv written")

