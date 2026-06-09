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


import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Re-run preprocessing to ensure everything’s fresh
train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

def feature_engineering(df):
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Compartment_Category'] = pd.cut(df['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)
    df['Style_Size'] = df['Style'] + '_' + df['Size']
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

X = train_df.drop(columns=['id', 'Price'])
y = train_df['Price']
test = test_df.drop(columns=['id'])
test_id = test_df['id']

cat_cols_to_encode = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
X_encoded = pd.get_dummies(X, columns=cat_cols_to_encode, drop_first=True)
test_encoded = pd.get_dummies(test, columns=cat_cols_to_encode, drop_first=True)
test_encoded = test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Initialize XGBoost with tuned parameters
xgb_model = XGBRegressor(
    n_estimators=200,      # More trees for better fit
    max_depth=6,           # Depth for complexity
    learning_rate=0.05,    # Slower learning for generalization
    random_state=42,
    n_jobs=-1              # Use all cores
)

# Train the model
print("Training XGBoost...")
xgb_model.fit(X_encoded, y)
print("Training complete!")

# Predict and clip
test_predictions = xgb_model.predict(test_encoded)
test_predictions = np.clip(test_predictions, a_min=0, a_max=None)

# Save submission
submission_df = pd.DataFrame({'id': test_id, 'Price': test_predictions})
submission_df.to_csv('/kaggle/working/submission_xgb.csv', index=False)
print("Submission saved as '/kaggle/working/submission_xgb.csv'")
print(submission_df.head())


import pandas as pd
import numpy as np
from xgboost import XGBRegressor

train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

def feature_engineering(df):
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Compartment_Category'] = pd.cut(df['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)
    df['Style_Size'] = df['Style'] + '_' + df['Size']
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

X = train_df.drop(columns=['id', 'Price'])
y = train_df['Price']
test = test_df.drop(columns=['id'])
test_id = test_df['id']

cat_cols_to_encode = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
X_encoded = pd.get_dummies(X, columns=cat_cols_to_encode, drop_first=True)
test_encoded = pd.get_dummies(test, columns=cat_cols_to_encode, drop_first=True)
test_encoded = test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

xgb_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
xgb_model.fit(X_encoded, y)
test_predictions = xgb_model.predict(test_encoded)
test_predictions = np.clip(test_predictions, a_min=0, a_max=None)

submission_df = pd.DataFrame({'id': test_id, 'Price': test_predictions})
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission saved! Submit this now!")


# Enhanced feature engineering
def feature_engineering_enhanced(df):
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})
    df['Compartment_Category'] = pd.cut(df['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)
    df['Style_Size'] = df['Style'] + '_' + df['Size']
    # New features
    df['Compartments_per_Weight'] = df['Compartments'] / (df['Weight Capacity (kg)'] + 1)  # Avoid division by zero
    df['Quality_Score'] = df['Waterproof'] + df['Laptop Compartment']  # Simple quality indicator
    return df

# Apply enhanced features
train_df = feature_engineering_enhanced(train_df)
test_df = feature_engineering_enhanced(test_df)

X = train_df.drop(columns=['id', 'Price'])
y = np.log1p(train_df['Price'])  # Log-transform target for better fit
test = test_df.drop(columns=['id'])

cat_cols_to_encode = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
X_encoded = pd.get_dummies(X, columns=cat_cols_to_encode, drop_first=True)
test_encoded = pd.get_dummies(test, columns=cat_cols_to_encode, drop_first=True)
test_encoded = test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Train XGBoost with log-transformed target
xgb_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
xgb_model.fit(X_encoded, y)

# Predict and reverse log-transform
test_predictions = np.expm1(xgb_model.predict(test_encoded))  # Reverse log1p with expm1
test_predictions = np.clip(test_predictions, a_min=0, a_max=None)

# Save submission
submission_df = pd.DataFrame({'id': test_id, 'Price': test_predictions})
submission_df.to_csv('/kaggle/working/submission_xgb_enhanced.csv', index=False)
print("Enhanced submission saved!")

