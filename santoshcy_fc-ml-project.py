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


rsds = pd.read_parquet("/kaggle/input/the-future-crop-challenge/rsds_wheat_train.parquet")
pr = pd.read_parquet("/kaggle/input/the-future-crop-challenge/pr_wheat_train.parquet")
tas = pd.read_parquet("/kaggle/input/the-future-crop-challenge/tas_wheat_train.parquet")
tmin = pd.read_parquet("/kaggle/input/the-future-crop-challenge/tasmin_wheat_train.parquet")
tmax = pd.read_parquet("/kaggle/input/the-future-crop-challenge/tasmax_wheat_train.parquet")
soilco2 = pd.read_parquet("/kaggle/input/the-future-crop-challenge/soil_co2_wheat_train.parquet")
yeild = pd.read_parquet("/kaggle/input/the-future-crop-challenge/train_solutions_wheat.parquet")


rsds.head()


rsds.info()


pr.head()


tas.head()


tmin.head()


tmax.head()


soilco2.head()


soilco2.info()


soilco2.reset_index()


yeild.head()


def summarize_time_series(df, id_cols, prefix):
    # If ID is in index, reset it to bring as column(s)
    if not all(col in df.columns for col in id_cols):
        df = df.reset_index()

    # Select numeric columns excluding id_cols
    ts_cols = df.drop(columns=id_cols, errors='ignore').select_dtypes(include='number').columns.tolist()

    # Copy ID columns for summary_df
    summary_df = df[id_cols].copy()

    # Compute summary statistics row-wise for ts_cols
    summary_df[f'{prefix}_mean'] = df[ts_cols].mean(axis=1)
    summary_df[f'{prefix}_std'] = df[ts_cols].std(axis=1)
    summary_df[f'{prefix}_min'] = df[ts_cols].min(axis=1)
    summary_df[f'{prefix}_max'] = df[ts_cols].max(axis=1)
    summary_df[f'{prefix}_median'] = df[ts_cols].median(axis=1)
    summary_df[f'{prefix}_trend'] = df[ts_cols].apply(lambda row: row.diff().mean(), axis=1)

    return summary_df



# Define your ID columns
id_cols = ['ID','crop','year', 'lat', 'lon']

# File mapping (update file paths accordingly)
files = {
    'precip': '/kaggle/input/the-future-crop-challenge/pr_wheat_train.parquet',
    'srad': '/kaggle/input/the-future-crop-challenge/rsds_wheat_train.parquet',
    'tmean': '/kaggle/input/the-future-crop-challenge/tas_wheat_train.parquet',
    'tmax': '/kaggle/input/the-future-crop-challenge/tasmax_wheat_train.parquet',
    'tmin': '/kaggle/input/the-future-crop-challenge/tasmin_wheat_train.parquet'
}



summarized_dfs = []

for prefix, file in files.items():
    print(f"Processing: {file}")
    df = pd.read_parquet(file)

    # Reset index if needed (your function also does this)
    if df.index.name == 'ID' and 'ID' not in df.columns:
        df = df.reset_index()

    summary = summarize_time_series(df, id_cols=id_cols, prefix=prefix)
    summarized_dfs.append(summary)


from functools import reduce
final_summary = reduce(lambda left, right: pd.merge(left, right, on=id_cols, how='outer'), summarized_dfs)
final_summary.head()


# Assuming final_summary is the combined summary DataFrame from before
soil = pd.read_parquet("/kaggle/input/the-future-crop-challenge/soil_co2_wheat_train.parquet")

# If soil DataFrame has ID as index, reset index
if soil.index.name == 'ID' and 'ID' not in soil.columns:
    soil = soil.reset_index()

# Check if all id_cols are in soil
missing_cols = [col for col in id_cols if col not in soil.columns]
if missing_cols:
    print(f"Warning: Missing columns in soil dataset: {missing_cols}")

# Merge soil data with final summary on id_cols
final_df = pd.merge(final_summary, soil, on=id_cols, how='inner')

final_df.head()



final_df.info()


yield_df = pd.read_parquet('/kaggle/input/the-future-crop-challenge/train_solutions_wheat.parquet')



# If yield_df index is ID and not a column, reset it
if yield_df.index.name == 'ID' and 'ID' not in yield_df.columns:
    yield_df = yield_df.reset_index()

# Merge on common keys
m_df = pd.merge(final_df, yield_df, on=['ID'], how='inner')

m_df.head()



m_df.isnull().sum()


m_df = m_df.drop(columns='crop')


m_df.head()


m_df.dtypes


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(m_df['yield'], kde=True)
plt.title("Distribution of Yield")
plt.show()


sns.heatmap(m_df.corr(numeric_only=True), fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


plt.figure(figsize=(10,6))
sc = plt.scatter(m_df['lon'], m_df['lat'], c=m_df['yield'], cmap='YlGn', alpha=0.7)
plt.colorbar(sc, label='Yield')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geospatial Distribution of Yield")
plt.show()



sns.lineplot(data=m_df, x='year', y='yield')
plt.title("Year-wise Yield Trend")
plt.show()



sns.scatterplot(data=m_df, x='co2', y='yield')
plt.title("CO2 vs Yield")

sns.scatterplot(data=m_df, x='nitrogen', y='yield')
plt.title("Nitrogen vs Yield")
plt.show()


sns.boxplot(data=m_df, x='texture_class', y='yield')
plt.title("Soil Texture Class vs Yield")
plt.show()


summary_cols = [col for col in m_df.columns if any(stat in col for stat in ['mean', 'std', 'trend'])]
for col in summary_cols:
    sns.scatterplot(data=m_df, x=col, y='yield')
    plt.title(f"{col} vs Yield")
    plt.show()


m_df.head()


from sklearn.model_selection import train_test_split

# Drop non-feature columns
drop_cols = ['id', 'lat', 'lon','yield']  # 'yield' is target
feature_cols = [col for col in m_df.columns if col not in drop_cols]

# Define X (features) and y (target)
X = m_df[feature_cols]
y = m_df['yield']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)



X_test.head()


y_train.head()


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Initialize and train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)


# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"ðŸ“Š RMSE: {rmse:.4f}")
print(f"ðŸ“‰ MAE: {mae:.4f}")
print(f"ðŸ“ˆ RÂ² Score: {r2:.4f}")


import matplotlib.pyplot as plt

importances = rf_model.feature_importances_
features = X_train.columns

# Sort features by importance
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(features)), importances[indices], align='center')
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd


models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
}


results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append((name, rmse, mae, r2))

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R2_Score"])
results_df.sort_values(by="R2_Score", ascending=False, inplace=True)
results_df


plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="R2_Score", y="Model", palette="viridis")
plt.title("Model Comparison Based on RÂ² Score", fontsize=14)
plt.xlabel("RÂ² Score")
plt.ylabel("Model")
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

search = RandomizedSearchCV(rf, param_distributions=param_dist, 
                            n_iter=20, cv=3, scoring='neg_root_mean_squared_error', verbose=2)
search.fit(X_train, y_train)

print("Best RF params:", search.best_params_)



import matplotlib.pyplot as plt

importances = search.best_estimator_.feature_importances_
feat_names = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh(feat_names, importances)
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.show()



from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

# Define the model
xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

# Parameter grid
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2.0]
}




# Randomized Search
xgb_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=25,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42
)

# Fit the search
xgb_search.fit(X_train, y_train)

# Best params and score
print("âœ… Best XGBoost Parameters:")
print(xgb_search.best_params_)

print("ðŸ“‰ Best RMSE Score:")
print(-xgb_search.best_score_)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

best_rf_model = search.best_estimator_
best_xgb_model = xgb_search.best_estimator_

# Predict with tuned models
rf_preds = best_rf_model.predict(X_test)
xgb_preds = best_xgb_model.predict(X_test)

# Evaluate
def evaluate_model(y_true, y_pred, name="Model"):
    print(f"ðŸ“Š {name} Evaluation:")
    print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RÂ² Score: {r2_score(y_true, y_pred):.4f}")

evaluate_model(y_test, rf_preds, "Random Forest")
evaluate_model(y_test, xgb_preds, "XGBoost")


import joblib
joblib.dump(best_rf_model, "best_random_forest_model.pkl")


X_train.to_csv('/kaggle/working/X_train.csv', index=False)


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Assuming your training features are in X_train
importances = best_rf_model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title("Top 15 Important Features (Random Forest)")
plt.tight_layout()
plt.show()



from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

y_pred = best_rf_model.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title(f'Actual vs Predicted Yield (RÂ² = {r2_score(y_test, y_pred):.3f})')
plt.grid(True)
plt.tight_layout()
plt.show()



print(X_train.columns.tolist())


X_train.shape


import pandas as pd
import matplotlib.pyplot as plt

# Get feature importances from the best model
importances = best_rf_model.feature_importances_
feature_names = X_train.columns  # Assuming you used pandas DataFrame

# Create a DataFrame for feature importance
feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# Show top N features
top_n = 10
top_features = feat_imp_df.head(top_n)
print(top_features)

# Optional: plot
plt.figure(figsize=(10, 5))
plt.barh(top_features['feature'], top_features['importance'], color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Top 10 Features by Random Forest")
plt.gca().invert_yaxis()
plt.show()




