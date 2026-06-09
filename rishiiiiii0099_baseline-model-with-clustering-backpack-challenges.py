from google.colab import drive
drive.mount('/content/drive')


# lets import tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
import joblib


df= pd.read_csv("/content/drive/MyDrive/Backpack Prediction /train.csv")

# only numeric data
df_numeric = df.select_dtypes(include=np.number)

#test data
df_test = pd.read_csv("/content/drive/MyDrive/Backpack Prediction /test.csv")

# copy the data set
df_copy = df.copy()
df_test_copy = df_test.copy()


# lets drop id columns in dataset
df_copy.drop('id',axis=1,inplace=True)
df_test_copy.drop('id',axis=1,inplace=True)


df_copy.head()


df_test_copy.head()


df_numeric.head()


df.info()


df.isnull().sum()


df.describe().T


# objects data types value counts
for label,content in df.items():
  if pd.api.types.is_object_dtype(content):
    print(f'value count for Column Object:{label}')
    print(df[label].value_counts())
    print('--'*40)


# value counts for numeric columns

for label,content in df.items():
  if pd.api.types.is_numeric_dtype(content):
    print(f'value count for numeric column:{label}')
    print(df[label].value_counts())
    print('--'*40)



import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define categorical and numerical columns
cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
num_cols = ['Compartments', 'Weight Capacity (kg)']

# Define Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessing Pipeline
preprocessor_cluster = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# Apply Preprocessing
df_transformed_for_cluster = preprocessor_cluster.fit_transform(df_copy)

# Determine optimal clusters using Elbow Method
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_transformed_for_cluster)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()





!pip install kneed


from kneed import KneeLocator

# Find the elbow point
knee_locator = KneeLocator(K_range, inertia, curve="convex", direction="decreasing")
optimal_k = knee_locator.elbow

print(f"Optimal number of clusters: {optimal_k}")



# Fit the final clustering model with optimal k (replace with determined value)
optimal_k =  4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_copy['Cluster'] = kmeans.fit_predict(df_transformed_for_cluster)


df_copy


# Apply Preprocessing
df_test_transformed_for_cluster = preprocessor_cluster.fit_transform(df_test_copy)

# Determine optimal clusters using Elbow Method
inertia1 = []
K_range1 = range(1, 11)

for j in K_range1:
    kmeans = KMeans(n_clusters=j, random_state=42, n_init=10)
    kmeans.fit(df_test_transformed_for_cluster)
    inertia1.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(K_range1, inertia1, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k (test data)")
plt.show()



from kneed import KneeLocator

# Find the elbow point
knee_locator = KneeLocator(K_range1, inertia1, curve="convex", direction="decreasing")
optimal_k = knee_locator.elbow

print(f"Optimal number of clusters: {optimal_k}")


# Fit the final clustering model with optimal k (replace with determined value)
optimal_k =  4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_test_copy['Cluster'] = kmeans.fit_predict(df_test_transformed_for_cluster)


df_test_copy


import datetime
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib

# Load and preprocess data
X = df_copy.drop(columns=['Price'])
y = df_copy['Price']
X_test = df_test_copy.copy()

# Extract 'id' for later use
X['id'] = df['id']
X_test['id'] = df_test['id']

# Define categorical and numerical columns
cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
num_cols = ['Compartments', 'Weight Capacity (kg)']

# Define Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# Model Pipeline with Gradient Descent
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SGDRegressor(max_iter=1000, tol=1e-3, random_state=42))
])

# Generate Timestamp
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Prepare OOF array
oof_predictions = np.zeros(len(X))

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for fold, (train_index, valid_index) in enumerate(kf.split(X), 1):
    # Split Data
    X_train_cv, X_valid_cv = X.iloc[train_index], X.iloc[valid_index]
    y_train_cv, y_valid_cv = y.iloc[train_index], y.iloc[valid_index]

    # Train Model
    model_pipeline.fit(X_train_cv, y_train_cv)
    preds = model_pipeline.predict(X_valid_cv)

    # Store OOF Predictions
    oof_predictions[valid_index] = preds

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_valid_cv, preds))
    rmse_scores.append(rmse)

    print(f"Fold {fold} - RMSE: {rmse:.4f}")

# Print Average Scores
print("RMSE Scores:", rmse_scores)
print("Average RMSE:", np.mean(rmse_scores))

# Save OOF Predictions
oof_df = pd.DataFrame({
    'id': X['id'],
    'oof_predictions': oof_predictions
})

oof_filename = f"oof_predictions_{timestamp_str}.csv"
oof_df.to_csv(oof_filename, index=False)
print(f"OOF predictions saved as {oof_filename}")

# Train on Full Dataset & Predict on Test
model_pipeline.fit(X, y)
X_test_transformed = model_pipeline.named_steps['preprocessor'].transform(X_test)
test_preds = model_pipeline.named_steps['regressor'].predict(X_test_transformed)

# Save Trained Model
model_filename = f"model_{timestamp_str}.pkl"
joblib.dump(model_pipeline, model_filename)
print(f"Trained model saved as {model_filename}")

# Prepare Submission File
submission = pd.DataFrame({
    'id': X_test['id'],
    'Price': test_preds
})

submission_filename = f"submission_{timestamp_str}.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission saved as {submission_filename}")


!pip install catboost


import datetime
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib
import catboost
from catboost import CatBoostRegressor

# Load and preprocess data
X = df_copy.drop(columns=['Price'])
y = df_copy['Price']
X_test = df_test_copy.copy()

# Extract 'id' for later use
X['id'] = df['id']
X_test['id'] = df_test['id']

# Define categorical and numerical columns
cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
num_cols = ['Compartments', 'Weight Capacity (kg)']

# Define Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# Model Pipeline with CatBoost
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, loss_function='RMSE', random_seed=42, verbose=0))
])

# Generate Timestamp
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Prepare OOF array
oof_predictions = np.zeros(len(X))

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for fold, (train_index, valid_index) in enumerate(kf.split(X), 1):
    # Split Data
    X_train_cv, X_valid_cv = X.iloc[train_index], X.iloc[valid_index]
    y_train_cv, y_valid_cv = y.iloc[train_index], y.iloc[valid_index]

    # Train Model
    model_pipeline.fit(X_train_cv, y_train_cv)
    preds = model_pipeline.predict(X_valid_cv)

    # Store OOF Predictions
    oof_predictions[valid_index] = preds

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_valid_cv, preds))
    rmse_scores.append(rmse)

    print(f"Fold {fold} - RMSE: {rmse:.4f}")

# Print Average Scores
print("RMSE Scores:", rmse_scores)
print("Average RMSE:", np.mean(rmse_scores))

# Save OOF Predictions
oof_df = pd.DataFrame({
    'id': X['id'],
    'oof_predictions': oof_predictions
})

oof_filename = f"oof_predictions_{timestamp_str}.csv"
oof_df.to_csv(oof_filename, index=False)
print(f"OOF predictions saved as {oof_filename}")

# Train on Full Dataset & Predict on Test
model_pipeline.fit(X, y)
X_test_transformed = model_pipeline.named_steps['preprocessor'].transform(X_test)
test_preds = model_pipeline.named_steps['regressor'].predict(X_test_transformed)

# Save Trained Model
model_filename = f"model_{timestamp_str}.pkl"
joblib.dump(model_pipeline, model_filename)
print(f"Trained model saved as {model_filename}")

# Prepare Submission File
submission = pd.DataFrame({
    'id': X_test['id'],
    'Price': test_preds
})

submission_filename = f"submission_{timestamp_str}.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission saved as {submission_filename}")



!pip install --upgrade scikit-learn
!pip install --upgrade xgboost
!pip install kneed
!pip install catboost

import datetime
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin #for wrapper


# Define categorical and numerical columns
cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
num_cols = ['Compartments', 'Weight Capacity (kg)']


# Define Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# Define XGBRegressor wrapper for compatibility
class XGBRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.regressor = xgb.XGBRegressor(**kwargs)

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.regressor.predict(X)

    def __sklearn_is_fitted__(self):  # Add this method for compatibility
        return hasattr(self.regressor, "best_iteration")

# Model Pipeline with XGBoost
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressorWrapper(objective='reg:squarederror', n_estimators=100, random_state=42))
])

# Data Preparation
X = df_copy.drop(columns=['Price'])
y = df_copy['Price']
X_test = df_test_copy.copy()

# Extract 'id' for later use
X['id'] = df['id']
X_test['id'] = df_test['id']

# K-Fold Cross-Validation and Training
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
oof_predictions = np.zeros(len(X))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for fold, (train_index, valid_index) in enumerate(kf.split(X), 1):
    X_train_cv, X_valid_cv = X.iloc[train_index], X.iloc[valid_index]
    y_train_cv, y_valid_cv = y.iloc[train_index], y.iloc[valid_index]

    model_pipeline.fit(X_train_cv, y_train_cv)
    preds = model_pipeline.predict(X_valid_cv)
    oof_predictions[valid_index] = preds
    rmse = np.sqrt(mean_squared_error(y_valid_cv, preds))
    rmse_scores.append(rmse)
    print(f"Fold {fold} - RMSE: {rmse:.4f}")

print("RMSE Scores:", rmse_scores)
print("Average RMSE:", np.mean(rmse_scores))

# Save OOF Predictions
oof_df = pd.DataFrame({'id': X['id'], 'oof_predictions': oof_predictions})
oof_filename = f"oof_predictions_{timestamp_str}.csv"
oof_df.to_csv(oof_filename, index=False)
print(f"OOF predictions saved as {oof_filename}")

# Train on Full Dataset and Predict on Test Data
model_pipeline.fit(X, y)
X_test_transformed = model_pipeline.named_steps['preprocessor'].transform(X_test)
test_preds = model_pipeline.named_steps['regressor'].predict(X_test_transformed)

# Save Trained Model
model_filename = f"model_{timestamp_str}.pkl"
joblib.dump(model_pipeline, model_filename)
print(f"Trained model saved as {model_filename}")

# Prepare Submission File
submission = pd.DataFrame({'id': X_test['id'], 'Price': test_preds})
submission_filename = f"submission_{timestamp_str}.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission saved as {submission_filename}")


# iam going to save clustering data

# iam going to save clustering data

#Save df_copy to Google Drive
df_copy_path = '/content/drive/MyDrive/Clustering /df_copy.csv'  # Add a filename to the path
df_copy.to_csv(df_copy_path, index=False)
print(f"df_copy saved to: {df_copy_path}")

df_test_path = '/content/drive/MyDrive/Clustering /df_test_copy.csv'  # Add a filename to the path
df_test_copy.to_csv(df_test_path,index=False)
print(f'df_test_copy saved to: {df_test_path}')





