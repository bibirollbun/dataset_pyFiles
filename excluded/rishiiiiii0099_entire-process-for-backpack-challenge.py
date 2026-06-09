from google.colab import drive
drive.mount('/content/drive')


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

# Elbow Method
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



# Fit the final clustering model with optimal k
optimal_k =  4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_copy['Cluster'] = kmeans.fit_predict(df_transformed_for_cluster)


df_copy
# now we successfully added cluster column to our original dataset


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


# Fit the final clustering model with optimal k
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



# Now iam going to use Clustering dataset
df_copy = pd.read_csv('/content/drive/MyDrive/Clustering /df_copy.csv')
df_test_copy = pd.read_csv('/content/drive/MyDrive/Clustering /df_test_copy.csv')
df = pd.read_csv('/content/drive/MyDrive/Backpack Prediction /train.csv')
df_test = pd.read_csv('/content/drive/MyDrive/Backpack Prediction /test.csv')


df_copy


df_test_copy


# prompt: skopt install

!pip install scikit-optimize



# prompt: catboost install

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
from skopt import BayesSearchCV  # Bayesian Optimization for Hyperparameter Tuning
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
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# Model Pipeline with CatBoostRegressor
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', CatBoostRegressor(verbose=0, random_state=42))
])

# Define Bayesian Search Space
param_grid = {
    'regressor__iterations': (500, 2000),
    'regressor__learning_rate': (0.01, 0.3, 'log-uniform'),
    'regressor__depth': (4, 10),
    'regressor__l2_leaf_reg': (1e-4, 10, 'log-uniform'),
    'regressor__border_count': (32, 255),
    'regressor__bagging_temperature': (0.0, 1.0)
}

# Bayesian Hyperparameter Tuning
bayes_search = BayesSearchCV(
    model_pipeline,
    param_grid,
    n_iter=20,  # Number of iterations
    cv=3,  # Cross-validation
    n_jobs=-1,  # Use all processors
    random_state=42,
    scoring='neg_root_mean_squared_error'
)

# Train with Bayesian Optimization
bayes_search.fit(X, y)
print("Best Parameters Found:", bayes_search.best_params_)

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

    # Create a fresh pipeline with best parameters (remove the 'regressor__' prefix)
    best_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', CatBoostRegressor(verbose=0, random_state=42,
                                        **{k.split('__')[1]: v for k, v in bayes_search.best_params_.items()}))
    ])

    # Train the fresh model
    best_model_pipeline.fit(X_train_cv, y_train_cv)

    preds = best_model_pipeline.predict(X_valid_cv)

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
final_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', CatBoostRegressor(verbose=0, random_state=42,
                                    **{k.split('__')[1]: v for k, v in bayes_search.best_params_.items()}))
])

final_model_pipeline.fit(X, y)
X_test_transformed = final_model_pipeline.named_steps['preprocessor'].transform(X_test)
test_preds = final_model_pipeline.named_steps['regressor'].predict(X_test_transformed)

# Save Trained Model
model_filename = f"model_{timestamp_str}.pkl"
joblib.dump(final_model_pipeline, model_filename)
print(f"Trained model saved as {model_filename}")

# Prepare Submission File
submission = pd.DataFrame({
    'id': X_test['id'],
    'Price': test_preds
})

submission_filename = f"submission_{timestamp_str}.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission saved as {submission_filename}")



# prompt: future warning

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



!pip install optuna



import datetime
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
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
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

def objective(trial):
    """Objective function for Optuna"""
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-4, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0.0, 1.0),
        'random_state': 42,
        'verbose': 0
    }

    # Split Data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', CatBoostRegressor(**params))
    ])

    model_pipeline.fit(X_train, y_train)
    preds = model_pipeline.predict(X_valid)

    return np.sqrt(mean_squared_error(y_valid, preds))  # RMSE

# Optuna Hyperparameter Optimization
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)

print("Best Parameters Found:", study.best_params)

# Prepare OOF array
oof_predictions = np.zeros(len(X))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for fold, (train_index, valid_index) in enumerate(kf.split(X), 1):
    X_train_cv, X_valid_cv = X.iloc[train_index], X.iloc[valid_index]
    y_train_cv, y_valid_cv = y.iloc[train_index], y.iloc[valid_index]

    best_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', CatBoostRegressor(**study.best_params))
    ])

    best_model_pipeline.fit(X_train_cv, y_train_cv)
    preds = best_model_pipeline.predict(X_valid_cv)

    oof_predictions[valid_index] = preds
    rmse = np.sqrt(mean_squared_error(y_valid_cv, preds))
    rmse_scores.append(rmse)
    print(f"Fold {fold} - RMSE: {rmse:.4f}")

# Print Average Scores
print("RMSE Scores:", rmse_scores)
print("Average RMSE:", np.mean(rmse_scores))

# Save OOF Predictions
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
oof_df = pd.DataFrame({'id': X['id'], 'oof_predictions': oof_predictions})
oof_filename = f"oof_predictions_{timestamp_str}.csv"
oof_df.to_csv(oof_filename, index=False)
print(f"OOF predictions saved as {oof_filename}")

# Train on Full Dataset & Predict on Test
final_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', CatBoostRegressor(**study.best_params))
])

final_model_pipeline.fit(X, y)
X_test_transformed = final_model_pipeline.named_steps['preprocessor'].transform(X_test)
test_preds = final_model_pipeline.named_steps['regressor'].predict(X_test_transformed)

# Save Trained Model
model_filename = f"model_{timestamp_str}.pkl"
joblib.dump(final_model_pipeline, model_filename)
print(f"Trained model saved as {model_filename}")

# Prepare Submission File
submission = pd.DataFrame({'id': X_test['id'], 'Price': test_preds})
submission_filename = f"submission_{timestamp_str}.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission saved as {submission_filename}")




import datetime
import numpy as np
import pandas as pd
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold, train_test_split
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
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

def objective(trial):
    """Objective function for Optuna"""
    params = {
        'alpha': trial.suggest_loguniform('alpha', 1e-6, 1e-1),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
        'eta0': trial.suggest_loguniform('eta0', 1e-4, 1),
        'max_iter': trial.suggest_int('max_iter', 500, 2000),
        'tol': trial.suggest_loguniform('tol', 1e-4, 1e-2)
    }

    # Split Data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SGDRegressor(**params, random_state=42))
    ])

    model_pipeline.fit(X_train, y_train)
    preds = model_pipeline.predict(X_valid)

    return np.sqrt(mean_squared_error(y_valid, preds))  # RMSE

# Optuna Hyperparameter Optimization
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20)

print("Best Parameters Found:", study.best_params)

# Prepare OOF array
oof_predictions = np.zeros(len(X))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for fold, (train_index, valid_index) in enumerate(kf.split(X), 1):
    X_train_cv, X_valid_cv = X.iloc[train_index], X.iloc[valid_index]
    y_train_cv, y_valid_cv = y.iloc[train_index], y.iloc[valid_index]

    best_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SGDRegressor(**study.best_params, random_state=42))
    ])

    best_model_pipeline.fit(X_train_cv, y_train_cv)
    preds = best_model_pipeline.predict(X_valid_cv)

    oof_predictions[valid_index] = preds
    rmse = np.sqrt(mean_squared_error(y_valid_cv, preds))
    rmse_scores.append(rmse)
    print(f"Fold {fold} - RMSE: {rmse:.4f}")

# Print Average Scores
print("RMSE Scores:", rmse_scores)
print("Average RMSE:", np.mean(rmse_scores))

# Save OOF Predictions
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
oof_df = pd.DataFrame({'id': X['id'], 'oof_predictions': oof_predictions})
oof_filename = f"oof_predictions_{timestamp_str}.csv"
oof_df.to_csv(oof_filename, index=False)
print(f"OOF predictions saved as {oof_filename}")

# Train on Full Dataset & Predict on Test
final_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SGDRegressor(**study.best_params, random_state=42))
])

final_model_pipeline.fit(X, y)
X_test_transformed = final_model_pipeline.named_steps['preprocessor'].transform(X_test)
test_preds = final_model_pipeline.named_steps['regressor'].predict(X_test_transformed)

# Save Trained Model
model_filename = f"model_{timestamp_str}.pkl"
joblib.dump(final_model_pipeline, model_filename)
print(f"Trained model saved as {model_filename}")

# Prepare Submission File
submission = pd.DataFrame({'id': X_test['id'], 'Price': test_preds})
submission_filename = f"submission_{timestamp_str}.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission saved as {submission_filename}")


so

