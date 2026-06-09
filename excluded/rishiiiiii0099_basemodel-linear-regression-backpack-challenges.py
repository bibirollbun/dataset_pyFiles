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



import datetime
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
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

# Apply K-Means Clustering BEFORE splitting into train/test
num_features = X[num_cols]
imputer = SimpleImputer(strategy='median')
num_features_imputed = imputer.fit_transform(num_features)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
X['Cluster'] = kmeans.fit_predict(num_features_imputed)
X_test['Cluster'] = kmeans.predict(imputer.transform(X_test[num_cols]))

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
    ('cat', categorical_transformer, cat_cols + ['Cluster'])
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
model_pipeline.fit(X,y)
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


