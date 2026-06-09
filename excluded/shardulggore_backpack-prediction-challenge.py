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


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
extra_df.info()


test_df.info()


train_df.info()


train_df.isna().sum()


# Handle missing values
categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_cols = ["Compartments", "Weight Capacity (kg)"]

# Fill missing categorical values with mode
def fill_mode(df, columns):
    for col in columns:
        df[col] = df[col].fillna(df[col].mode()[0])

fill_mode(train_df, categorical_cols)
fill_mode(test_df, categorical_cols)

# Fill missing numerical values with median
def fill_median(df, columns):
    for col in columns:
        df[col] = df[col].fillna(df[col].median())

fill_median(train_df, numerical_cols)
fill_median(test_df, numerical_cols)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# Define features and target
features = categorical_cols + numerical_cols
X = train_df[features]
y = train_df["Price"]
X_test = test_df[features]

# Standardize numerical features
scaler = StandardScaler()
X.loc[:, numerical_cols] = scaler.fit_transform(X[numerical_cols])
X_test.loc[:, numerical_cols] = scaler.transform(X_test[numerical_cols])


# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize Random Forest regressor with optimized hyperparameters
rf_model = RandomForestRegressor(
    n_estimators=500, max_depth=12, min_samples_split=4, min_samples_leaf=2,
    random_state=42, n_jobs=-1)

# Train the model
rf_model.fit(X_train, y_train)


# Make predictions
y_pred = rf_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")

# Predict on test dataset
test_predictions = rf_model.predict(X_test)

# Ensure no negative predictions
test_predictions = np.maximum(test_predictions, 0)

# Create submission file
submission = pd.DataFrame({"id": test_df["id"], "Price": test_predictions})
submission.to_csv("submission.csv", index=False)

print("Improved submission file saved as 'submission.csv'")


submission.head()


submission.to_csv("submission.csv", index=False)




