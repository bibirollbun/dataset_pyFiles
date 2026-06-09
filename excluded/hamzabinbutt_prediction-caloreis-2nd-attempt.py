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


# Basic packages
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Model selection and evaluation
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_squared_log_error,
    mean_squared_error,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

# Regressors
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb

# Visualization
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.simplefilter(action='ignore', category=FutureWarning)


# File paths
train_path = "/kaggle/input/playground-series-s5e5/train.csv"
test_path = "/kaggle/input/playground-series-s5e5/test.csv"

# Read CSV files
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Drop 'id' column from both datasets
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])

# Display the first few rows
print("Train DataFrame:")
print(train_df.head())

print("\nTest DataFrame:")
print(test_df.head())


# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to 'Sex' column
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])

# Display the encoded DataFrame
train_df.head()


import numpy as np

def generate_features(train_data):
    # Height in meters
    train_data['Height_m'] = train_data['Height'] / 100

    # BMI
    train_data['BMI'] = train_data['Weight'] / (train_data['Height_m'] ** 2)

    # Heart rate per minute
    train_data['HeartRate_per_Min'] = train_data['Heart_Rate'] / train_data['Duration']

    # Temperature deviation from normal (37°C)
    train_data['Temp_Deviation'] = train_data['Body_Temp'] - 37

    # Weight normalized by age
    train_data['Weight_per_Age'] = train_data['Weight'] / train_data['Age']

    # Squared features for non-linear effects
    train_data['Age_squared'] = train_data['Age'] ** 2
    train_data['Duration_squared'] = train_data['Duration'] ** 2

    # Composite intensity score
    train_data['Intensity_Score'] = (train_data['Heart_Rate'] * train_data['Body_Temp']) / train_data['Duration']

    # ----------------------------- NEW FEATURES -----------------------------

    # Interaction terms (capturing combined effects)
    train_data['Age_Duration'] = train_data['Age'] * train_data['Duration']
    train_data['Weight_HeartRate'] = train_data['Weight'] * train_data['Heart_Rate']
    train_data['BMI_Temp'] = train_data['BMI'] * train_data['Body_Temp']
    
    # Non-linear transformations
    train_data['Log_Duration'] = np.log1p(train_data['Duration'])  # log1p avoids log(0)
    train_data['Sqrt_HeartRate'] = np.sqrt(train_data['Heart_Rate'])

    # Temperature-to-BMI ratio
    train_data['Temp_BMI_Ratio'] = train_data['Body_Temp'] / train_data['BMI']

    # Feature for possible fatigue or effort
    train_data['Effort_Index'] = train_data['HeartRate_per_Min'] * train_data['BMI']

    # Age-normalized intensity
    train_data['Norm_Intensity'] = train_data['Intensity_Score'] / train_data['Age']

    # Heart rate deviation from population mean (if mean is known; otherwise can be per group later)
    train_data['HeartRate_Deviation'] = train_data['Heart_Rate'] - train_data['Heart_Rate'].mean()

    # Z-score normalization feature (optional: if needed by model)
    train_data['BMI_zscore'] = (train_data['BMI'] - train_data['BMI'].mean()) / train_data['BMI'].std()

    return train_data



train_df_featured = generate_features(train_df)
train_df_featured


y = train_df_featured['Calories']  
# Drop the target column from the feature set (X)
X = train_df_featured.drop('Calories', axis=1)
# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Optional: Check the shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor

# Function to compute RMSLE
def rmsle_score(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Initialize CatBoost Regressor (using RMSE)
catboost_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=0
)

# Define K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
rmsle_scores_train = []
rmsle_scores_test = []

# Perform cross-validation with log1p transformation
for fold, (train_index, test_index) in enumerate(kf.split(X_train)):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    
    # Log-transform the target variable
    y_train_log = np.log1p(y_train_fold)
    
    # Train the model on log-transformed target
    catboost_model.fit(X_train_fold, y_train_log)
    
    # Predict in log scale, then inverse-transform
    y_pred_train_log = catboost_model.predict(X_train_fold)
    y_pred_test_log = catboost_model.predict(X_test_fold)
    
    y_pred_train_fold = np.expm1(y_pred_train_log)
    y_pred_test_fold = np.expm1(y_pred_test_log)
    
    # Ensure non-negative predictions
    y_pred_train_fold = np.maximum(0, y_pred_train_fold)
    y_pred_test_fold = np.maximum(0, y_pred_test_fold)
    
    # Calculate RMSLE for this fold
    rmsle_train = rmsle_score(y_train_fold, y_pred_train_fold)
    rmsle_test = rmsle_score(y_test_fold, y_pred_test_fold)
    
    rmsle_scores_train.append(rmsle_train)
    rmsle_scores_test.append(rmsle_test)

# Mean RMSLE across all folds
mean_rmsle_train = np.mean(rmsle_scores_train)
mean_rmsle_test = np.mean(rmsle_scores_test)

# Print CV results
print(f"Mean RMSLE (Training Data): {mean_rmsle_train:.4f}")
print(f"Mean RMSLE (Test Data): {mean_rmsle_test:.4f}")

# Plot RMSLE scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rmsle_scores_train) + 1), rmsle_scores_train, label="Training RMSLE", marker='o')
plt.plot(range(1, len(rmsle_scores_test) + 1), rmsle_scores_test, label="Test RMSLE", marker='o')
plt.xlabel('Fold')
plt.ylabel('RMSLE')
plt.title('RMSLE for Training and Test Data Across Folds')
plt.legend()
plt.grid(True)
plt.show()

# Final model evaluation on test set
catboost_model.fit(X_train, np.log1p(y_train))
y_pred_log = catboost_model.predict(X_test)
y_pred = np.expm1(y_pred_log)

# Ensure non-negative predictions
y_pred = np.maximum(0, y_pred)
y_test = np.maximum(0, y_test)

# Final RMSLE
final_rmsle = rmsle_score(y_test, y_pred)
print(f"Final RMSLE on Test Data (CatBoost): {final_rmsle:.4f}")



# Apply the same LabelEncoder to 'Sex' in test_df
test_df['Sex'] = label_encoder.transform(test_df['Sex'])

# Display the encoded test DataFrame
test_df.head()


test_df_featured = generate_features(test_df)
test_df_featured


# Predict in log scale
y_test_pred_log = catboost_model.predict(test_df_featured)

# Convert back to original scale
y_test_pred = np.expm1(y_test_pred_log)

# Ensure non-negative predictions
y_test_pred = np.maximum(0, y_test_pred)

print(y_test_pred)

