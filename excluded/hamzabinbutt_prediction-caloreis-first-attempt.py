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



train_df.describe()


# Identify numerical and categorical variables
numerical_vars = train_df.select_dtypes(include=['number']).columns.tolist()
categorical_vars = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

# Print the results
print("Numerical Variables:", numerical_vars)
print("Categorical Variables:", categorical_vars)



# Combine both lists
all_vars = numerical_vars + categorical_vars

# Calculate missing value percentage for each variable
missing_percent = train_df[all_vars].isnull().mean() * 100

# Display result sorted by highest missing %
print(missing_percent.sort_values(ascending=False))



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


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plot style for better visualization
sns.set(style="whitegrid")

# Get the numerical columns in the DataFrame
numerical_columns = train_df_featured.select_dtypes(include=['float64', 'int64']).columns

# Calculate the number of rows and columns for the grid (rounded up to fit all plots)
n_cols = 5
n_rows = int(np.ceil(len(numerical_columns) / n_cols))

# Create a figure with multiple subplots
plt.figure(figsize=(n_cols * 5, n_rows * 5))

# Loop over each numerical column in the dataset
for i, column in enumerate(numerical_columns):
    plt.subplot(n_rows, n_cols, i + 1)  # Arrange the plots in a grid
    sns.histplot(train_df_featured[column], kde=True, bins=30, color='skyblue', stat='density')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')

# Adjust the layout to prevent overlap
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Compute the correlation matrix
corr_matrix = train_df_featured.corr(numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(16, 12))

# Generate a heatmap
sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    cbar=True, 
    square=True,
    linewidths=0.5
)

plt.title("Correlation Matrix Heatmap", fontsize=18)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



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
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor

# Function to compute RMSLE
def rmsle_score(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Initialize CatBoost Regressor
catboost_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=0
)

# Define K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to hold RMSLE for each fold
train_rmsle_scores = []
test_rmsle_scores = []

# Perform cross-validation with RMSLE as the metric
for fold, (train_index, test_index) in enumerate(kf.split(X_train)):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    
    # Train the model
    catboost_model.fit(X_train_fold, y_train_fold)
    
    # Predict on training and testing sets
    y_pred_train_fold = catboost_model.predict(X_train_fold)
    y_pred_test_fold = catboost_model.predict(X_test_fold)
    
    # Ensure non-negative values for RMSLE
    y_pred_train_fold = np.maximum(0, y_pred_train_fold)
    y_pred_test_fold = np.maximum(0, y_pred_test_fold)
    y_train_fold = np.maximum(0, y_train_fold)
    y_test_fold = np.maximum(0, y_test_fold)
    
    # Calculate RMSLE for train and test data
    train_rmsle = rmsle_score(y_train_fold, y_pred_train_fold)
    test_rmsle = rmsle_score(y_test_fold, y_pred_test_fold)
    
    train_rmsle_scores.append(train_rmsle)
    test_rmsle_scores.append(test_rmsle)

# Calculate the mean and standard deviation of RMSLE across all folds
mean_train_rmsle = np.mean(train_rmsle_scores)
std_train_rmsle = np.std(train_rmsle_scores)
mean_test_rmsle = np.mean(test_rmsle_scores)
std_test_rmsle = np.std(test_rmsle_scores)

# Print results
print(f"Mean Train RMSLE (CatBoost): {mean_train_rmsle:.4f}")
print(f"Train RMSLE Standard Deviation (CatBoost): {std_train_rmsle:.4f}")
print(f"Mean Test RMSLE (CatBoost): {mean_test_rmsle:.4f}")
print(f"Test RMSLE Standard Deviation (CatBoost): {std_test_rmsle:.4f}")

# Plotting RMSLE for each fold
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_rmsle_scores) + 1), train_rmsle_scores, label='Train RMSLE', marker='o')
plt.plot(range(1, len(test_rmsle_scores) + 1), test_rmsle_scores, label='Test RMSLE', marker='o')
plt.xlabel('Fold')
plt.ylabel('RMSLE')
plt.title('RMSLE for Train and Test Set Across Folds (CatBoost)')
plt.legend()
plt.grid(True)
plt.show()

# Final model evaluation on the test set
catboost_model.fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

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


# Predict
y_pred_test = catboost_model.predict(test_df_featured)
print(y_pred_test)

