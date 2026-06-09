import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from sklearn.metrics import mean_squared_error, r2_score
import warnings 
warnings.filterwarnings('ignore')
# Disable LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.getLogger('lightgbm').setLevel(logging.INFO)
logging.getLogger('lightgbm').setLevel(logging.ERROR)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_data = pd.read_csv(r"/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv(r"/kaggle/input/playground-series-s5e2/test.csv")
data = pd.read_csv(r"/kaggle/input/playground-series-s5e2/training_extra.csv")
sample_submission = pd.read_csv(r"/kaggle/input/playground-series-s5e2/sample_submission.csv")

print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("data shape :",data.shape)
print("sample_submission shape :",sample_submission.shape)


train_data.head()


train_data.info()


# Identify categorical columns
cat_cols = train_data.select_dtypes(include=["object", "category"]).columns.tolist()

# Define subplot grid (3 rows, auto columns)
num_plots = len(cat_cols)
rows = 3
cols = (num_plots // rows) + (num_plots % rows > 0)  # Adjust columns dynamically

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = axes.flatten()  # Flatten for easy indexing

# Plot each categorical column
for i, col in enumerate(cat_cols):
    value_counts = train_data[col].value_counts()
    axes[i].pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=140)
    axes[i].set_title(f"Distribution of {col}")

# Hide unused subplots (if any)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


train_data.isna().sum().sort_values(ascending=False)




# # Use seaborn heatmap to visualize missing values
# sns.heatmap(train_data.isnull(), cmap='coolwarm', yticklabels=False)

# # Title
# plt.title("Missing Values Heatmap", fontsize=14)

# plt.show()

# Calculate missing values per column
missing_values = train_data.isnull().sum()

# Bar plot of missing values
plt.figure(figsize=(8, 5))
sns.barplot(x=missing_values.index, y=missing_values.values, palette="Blues_r")
plt.xticks(rotation=45) 
# Labels
plt.xlabel("Columns")
plt.ylabel("Count of Missing Values")
plt.title("Missing Values Distribution Across Columns")

# Show plot
plt.show()


# Remove 'id' column as it's not needed for training
train_data.drop(columns=['id'], inplace=True)
test_ids = test_data['id']  # Save test IDs for submission
test_data.drop(columns=['id'], inplace=True)




# Handle missing values in both categorical and numerical columns
for col in train_data.columns:
    if train_data[col].dtype == 'object':  # Categorical features
        train_data[col] = train_data[col].fillna("Unknown")
        if col in test_data.columns:
            test_data[col] = test_data[col].fillna("Unknown")
    else:  # Numerical features
        train_data[col] = train_data[col].fillna(train_data[col].median())
        if col in test_data.columns:
            test_data[col] = test_data[col].fillna(test_data[col].median())


# Encode categorical features using LabelEncoder
label_encoders = {}
for col in train_data.select_dtypes(include=['object']).columns:
    encoder = LabelEncoder()
    train_data[col] = encoder.fit_transform(train_data[col])
    if col in test_data.columns:
        test_data[col] = encoder.transform(test_data[col])
    label_encoders[col] = encoder


# Define features (X) and target variable (y)
if 'Price' in train_data.columns:
    X = train_data.drop(columns=['Price'])
    y = train_data['Price']
else:
    raise KeyError("The target column 'Price' is missing from the dataset.")


# Split data into training and validation sets (80-20 split)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
test_scaled = scaler.transform(test_data)



# Initialize and train XGBoost model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

# Make predictions on validation set
y_pred = xgb_model.predict(X_valid_scaled)

# Evaluate model performance using Mean Absolute Error
mae = mean_absolute_error(y_valid, y_pred)
print(f"Validation Mean Absolute Error (MAE): {mae:.4f}")




# Plot feature importance
# plt.figure(figsize=(10, 6))
# xgb.plot_importance(xgb_model)
# plt.title("Feature Importance")
# plt.show()

# Make predictions on the test set
test_predictions = xgb_model.predict(test_scaled)

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'Price': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file saved successfully as 'submission.csv'.")

