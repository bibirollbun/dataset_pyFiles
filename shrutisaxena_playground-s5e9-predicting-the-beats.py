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


# =====================================
# 1. Setup
# =====================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Settings
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

#test  = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")  
print("Shape of dataset:", df.shape)

# Quick look
df.head()


# =====================================
# 2. Basic Info & Data Quality
# =====================================
df.info()
df.describe(include='all')

# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if not missing.empty:
    plt.figure(figsize=(10, 6))
    missing.plot(kind='bar')
    plt.title("Missing Values by Feature")
    plt.show()
else:
    print("âœ… No missing values found.")

# Duplicates
print("Duplicate rows:", df.duplicated().sum())



print("\n--- Descriptive Statistics ---")
print(df.describe().T)


# ========================
# 3. Target Variable Analysis (Target: BeatsPerMinute)
# ========================
target = "BeatsPerMinute"

plt.figure(figsize=(8,5))
sns.histplot(df[target], kde=True, bins=30)
plt.title(f"Distribution of {target}")
plt.show()



# ========================
# 4. Feature Distributions
# ========================
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
num_features.remove("id") # remove id if present

# Plot histograms for all numeric features
df[num_features].hist(bins=30, figsize=(15,12), layout=(4,3))
plt.suptitle("Feature Distributions")
plt.show()


# ========================
# 5. Correlation Analysis
# ========================
plt.figure(figsize=(10,8))
sns.heatmap(df[num_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Correlation with target
print("\n--- Correlation with Target ---")
print(df.corr()[target].sort_values(ascending=False))


# ========================
# 6. Feature vs Target Relationships
# ========================
for col in num_features:
    if col != target:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[col], y=df[target])
        plt.title(f"{col} vs {target}")
        plt.show()


# ========================
# 7. Outlier Detection (Boxplots)
# ========================
for col in num_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.show()


#df['RhythmScore'] = np.where(df['RhythmScore'] < lower, lower,
#                             np.where(df['RhythmScore'] > upper, upper, df['RhythmScore']))
#df['AudioLoudness'] = np.where(df['AudioLoudness'] < lower, lower,
#                             np.where(df['AudioLoudness'] > upper, upper, df['AudioLoudness']))

def cap_outliers(df, features):
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower,
        np.where(df[col] > upper, upper, df[col]))
    return df

df = cap_outliers(df, num_features)


# ========================
# 8. Feature Engineering Ideas
# ========================
# Example: Convert TrackDurationMs to minutes
df['TrackDurationMin'] = df['TrackDurationMs'] / 60000
df['Energy_Acoustic_Ratio'] = df['Energy'] / (df['AcousticQuality'] + 1e-5)
df['Vocal_Instrument_Balance'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-5)
df['MoodRhythm'] = df['MoodScore'] * df['RhythmScore']
df['PerformanceIntensity'] = df['LivePerformanceLikelihood'] * df['AudioLoudness']
df['RhythmEnergy'] = df['RhythmScore'] * df['Energy']
df['MoodAcoustic'] = df['MoodScore'] * df['AcousticQuality']



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df1 = scaler.fit_transform(df)
df1 = pd.DataFrame(df1, columns=df.columns, index=df.index)



# ========================
# 8. XGBoost Model
# ========================

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
X = df1.drop(columns=['id', target])
y = df1[target]

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


# ========================
# 10. Evaluate Best Model
# ========================
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("RÂ² Score:", r2_score(y_test, y_pred))



# ========================
# 11. Feature Importance
# ========================
plt.figure(figsize=(10,6))
plt.barh(X.columns, best_model.feature_importances_)
plt.title("XGBoost Feature Importance")
plt.show()



# ============================
# CatBoost Regressor Example
# ============================
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------
# Example dataset
# (replace df and target with your own)
# ----------------------------
#X = df.drop(columns=['id', target])
#y = df[target]

#X_train, X_val, y_train, y_val = train_test_split(
 #   X, y, test_size=0.2, random_state=42
#)

# ----------------------------
# CatBoost Model
# ----------------------------
cat_model = CatBoostRegressor(
    loss_function='RMSE',       # regression loss
    random_seed=42,
    verbose=0,               # suppress training output
    task_type="GPU" 
)

# ----------------------------
# Hyperparameter grid
# ----------------------------
param_grid = {
    'depth': [6, 8,10],
    'learning_rate': [0.01,  0.1],
    'iterations': [100, 500, 1000],
    'l2_leaf_reg': [1,  5]
}

# ----------------------------
# GridSearchCV for tuning
# ----------------------------
grid = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2
)

grid.fit(X_train, y_train)

best_model_cat = grid.best_estimator_
print("âœ… Best Parameters:", grid.best_params_)




# ----------------------------
# Evaluate
# ----------------------------
y_pred_cat = best_model_cat.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_cat))
r2 = r2_score(y_test, y_pred_cat)

print(f"âœ… Tuned CatBoost RMSE: {rmse:.4f}")
print(f"âœ… Tuned CatBoost RÂ² Score: {r2:.4f}")



import pandas as pd

# ========================
# 1. Load Test Data
# ========================
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

# ========================
# 2. Apply Same Feature Engineering
# ========================
test_df['TrackDurationMin'] = test_df['TrackDurationMs'] / 60000
test_df['Energy_Acoustic_Ratio'] = test_df['Energy'] / (test_df['AcousticQuality'] + 1e-5)
test_df['Vocal_Instrument_Balance'] = test_df['VocalContent'] / (test_df['InstrumentalScore'] + 1e-5)
test_df['MoodRhythm'] = test_df['MoodScore'] * test_df['RhythmScore']
test_df['PerformanceIntensity'] = test_df['LivePerformanceLikelihood'] * test_df['AudioLoudness']
test_df['RhythmEnergy'] = test_df['RhythmScore'] * test_df['Energy']
test_df['MoodAcoustic'] = test_df['MoodScore'] * test_df['AcousticQuality']

# ========================
# 3. Ensure Consistent Features
# ========================
train_features = best_model.get_booster().feature_names  # features used in training

X_test_final = test_df[train_features]  # select only training features (drop id automatically)

# ========================
# 4. Predict with Best Model
# ========================
y_xg_pred_test = best_model.predict(X_test_final)
y_cat_pred_test = best_model_cat.predict(X_test_final)
y_pred_test = y_xg_pred_test*0.6+y_cat_pred_test*0.4

# ========================
# 5. Save Predictions
# ========================
output = pd.DataFrame({
    "id": test_df["id"],
    "Predicted_BPM": y_pred_test
})
# ========================
output_cat = pd.DataFrame({
    "id": test_df["id"],
    "Predicted_BPM": y_cat_pred_test
})
output_xg = pd.DataFrame({
    "id": test_df["id"],
    "Predicted_BPM": y_xg_pred_test
})
output.to_csv("test_predictions_xg.csv", index=False)
output.to_csv("test_predictions_cat.csv", index=False)
output.to_csv("test_predictions.csv", index=False)

print("Predictions saved to test_predictions.csv")
print(output.head())

