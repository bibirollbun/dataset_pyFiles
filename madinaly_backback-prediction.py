# For data manipulation
import numpy as np 
import pandas as pd 

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# For XGBoost Implementation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import warnings
warnings.simplefilter("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import training data
train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col="id")
train_data.head()


# Import testing data
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col="id")
test_data.head()


# Import training_extra_data
train_extra_data = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col="id")
train_extra_data.head()


# Combine train and test - to transform columns
# Add a column to differentiage the sets
train_data["dataset_type"] = "train"
test_data["dataset_type"] = "test"

# Concatenate both datasets
combined_df = pd.concat([train_data, test_data])

# Display first few rows
combined_df.head()


# Check columns with missing values
print("Missing Values in Combined Data:\n", combined_df.isnull().sum())
print("Missing Values in Train_Extra_Data:\n", train_extra_data.isnull().sum())


from sklearn.impute import SimpleImputer

# For numerical columns
num_cols = ['Weight Capacity (kg)', 'Compartments']
imputer_num = SimpleImputer(strategy='mean')
combined_df[num_cols] = imputer_num.fit_transform(combined_df[num_cols])

# For categorical columns
cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
imputer_cat = SimpleImputer(strategy='most_frequent')
combined_df[cat_cols] = imputer_cat.fit_transform(combined_df[cat_cols])

# Check columns again for missing values
print("Missing Values in Combined Data:\n", combined_df.isnull().sum())


# Describe numerical features
print(combined_df.describe())
print(train_extra_data.describe())


# Histogram for key numerical features (Train Data)
for col in ["Price", "Weight Capacity (kg)", "Compartments"]:
    plt.figure(figsize=(6, 4))
    sns.histplot(combined_df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col} in Combined Data")
    plt.show()

# Histogram for train_extra_data
for col in ["Price", "Weight Capacity (kg)", "Compartments"]:
    plt.figure(figsize=(6, 4))
    sns.histplot(train_extra_data[col], bins=30, kde=True)
    plt.title(f"Distribution of {col} in Train Extra Data")
    plt.show()


# Check duplicate rows - Keeping as noise for now
print(f"Duplicates in Combined Data: {combined_df.duplicated().sum()}")
print(f"Duplicates in Train Extra Data: {train_extra_data.duplicated().sum()}")


# Check data types
print(combined_df.dtypes)

# Find non-numeric columns
non_numeric_cols = combined_df.select_dtypes(exclude=['number']).columns
print("Non-numeric columns:", non_numeric_cols)


# Copy the original data to avoid modifying it directly
df1 = combined_df.copy()

# Convert binary categorical features to 0/1
df1["Laptop Compartment"] = df1["Laptop Compartment"].map({"Yes": 1, "No": 0})
df1["Waterproof"] = df1["Waterproof"].map({"Yes": 1, "No": 0})

# One-Hot Encoding for categorical features
categorical_cols = ["Brand", "Material", "Size", "Style", "Color"]
df1 = pd.get_dummies(df1, columns=categorical_cols, drop_first=True)  # Avoid multicollinearity

# View transformed data
print(df1.head())



df1.head()


# Convert binary data types to integers
df1[['Brand_Jansport', 'Brand_Nike', 'Brand_Puma', 'Brand_Under Armour',
    'Material_Leather', 'Material_Nylon', 'Material_Polyester', 'Size_Medium',
    'Size_Small', 'Style_Messenger', 'Style_Tote', 'Color_Blue', 'Color_Gray',
    'Color_Green', 'Color_Pink', 'Color_Red']] = df1[['Brand_Jansport', 'Brand_Nike', 'Brand_Puma', 'Brand_Under Armour',
    'Material_Leather', 'Material_Nylon', 'Material_Polyester', 'Size_Medium',
    'Size_Small', 'Style_Messenger', 'Style_Tote', 'Color_Blue', 'Color_Gray',
    'Color_Green', 'Color_Pink', 'Color_Red']].astype(int)

df1.head()


# Split dataset to train and test as originally provided by Kaggle
train_data = df1[df1["dataset_type"] == "train"].drop(columns=["dataset_type"])
test_data = df1[df1["dataset_type"] == "test"].drop(columns=["dataset_type", "Price"])



train_data.head()


test_data.head()


# Obtain the predictor y
y = train_data["Price"]
y.head()


# Select the feature variables
X = train_data.drop(columns=["Price"])
X.head()


# Split training dataset
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize XGBoost Regressor
xgb_model = XGBRegressor(
    objective="reg:squarederror", 
    n_estimators=200, 
    learning_rate=0.1, 
    max_depth=6, 
    random_state=42
)


# Fit the model to training data
xgb_model.fit(X_train, y_train)


# Make predictions on training data
y_valid_pred = xgb_model.predict(X_valid)


# Calculate Mean Squared Error
mse = mean_squared_error(y_valid, y_valid_pred)
print(f"Mean Squared Error: {mse:.2f}")


# Root Mean Squared Error (RMSE)
rmse = mse ** 0.5
print(f"Root Mean Squared Error: {rmse:.2f}")


# Making sure I have 200000 rows this time
X_test = test_data

X_test.shape


# Make predictions on test data (which is the one for submission)
y_pred = xgb_model.predict(X_test)


# Create the submission dataframe with 'id' from the index of test_data and predicted 'Price'
submission = pd.DataFrame({
    "id": X_test.index,  # 'id' is the index
    "Price": y_pred     # Predicted prices
})

# Save the submission DataFrame to a CSV file
submission.to_csv("submission.csv", index=False)

