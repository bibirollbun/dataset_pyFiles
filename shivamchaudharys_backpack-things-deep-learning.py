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


pip install dask



# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import dask.dataframe as dd


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# Create the imputer with most frequent strategy
si = SimpleImputer(strategy='most_frequent')

# Reshape to 2D before applying fit_transform and flatten back to 1D
df[['Brand','Size','Material','Style']] = si.fit_transform(df[['Brand','Size','Material','Style']])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit the encoder on the training data
encoded_array_train = encoder.fit_transform(df[['Brand','Size','Style']])

# Convert the encoded array to a DataFrame
encoded_df_train = pd.DataFrame(encoded_array_train, columns=encoder.get_feature_names_out(['Brand','Size','Style']))

# Concatenate the original training DataFrame with the new encoded columns and drop the old ones
df = pd.concat([df, encoded_df_train], axis=1).drop(['Brand','Size','Style'], axis=1)

knn = KNNImputer(n_neighbors=20)
df[['Weight Capacity (kg)','Compartments']] = df[['Weight Capacity (kg)','Compartments']].fillna(df[['Weight Capacity (kg)','Compartments']].median())


# StandardScaler only works with numerical columns, so exclude 'Compartment'
scaler = StandardScaler()
d['Weight Cf[apacity (kg)','Compartments']] = scaler.fit_transform(df[['Weight Capacity (kg)','Compartments']])
# Apply fit_transform and convert back to 1D
df['Weight Capacity (kg)'] = knn.fit_transform(df[['Weight Capacity (kg)']]).ravel()
df[['Weight Capacity (kg)','Compartments']] = scaler.inverse_transform(df[['Weight Capacity (kg)','Compartments']])
df = df.fillna(df['Color'])

 map_color = {'Black':    1.1,'Green':   1.2,'Red':  1.3,'Blue':  1.4,'Gray':1.05,'Pink':1.5,'NaN':0}
 df['Color'] = df['Color'].map(map_color)


df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})
df['Waterproof'] = df['Waterproof'].map({'Yes':1,'No':0})

lb = LabelEncoder()
df['Material'] = lb.fit_transform(df['Material'])

df[['Material']] = scaler.fit_transform(df[['Material']])
print(df[['Laptop Compartment', 'Waterproof']].isnull().sum())  # Check for missing values
df[['Laptop Compartment', 'Waterproof']] = df[['Laptop Compartment', 'Waterproof']].fillna(df[['Laptop Compartment', 'Waterproof']].median())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df[['Laptop Compartment', 'Waterproof']] = scaler.fit_transform(df[['Laptop Compartment', 'Waterproof']])
df[['Laptop Compartment', 'Waterproof']] = knn.fit_transform(df[['Laptop Compartment', 'Waterproof']])
df[['Laptop Compartment', 'Waterproof']] = scaler.inverse_transform(df[['Laptop Compartment', 'Waterproof']])



X = df.drop(columns=['Price'])


y = df['Price']


# Load dataset and separate features/target
X = df.drop(columns=['Price']).values  # Ensure the correct target column name
y = df['Price'].values


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Assuming X is the feature set and y is the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)














import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load datasets
df_main = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")

# Fix column names
df_main.columns = df_main.columns.str.replace(' ', '_')
df_extra.columns = df_extra.columns.str.replace(' ', '_')

# Concatenate datasets
df_train = pd.concat([df_main, df_extra], ignore_index=True)

# Remove outliers using IQR method
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
    return df

df_train = remove_outliers(df_train, 'Price')

# Separate target variable
target = df_train['Price']
df_train.drop(columns=['Price'], inplace=True)

# Identify numerical and categorical columns
num_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()

# Feature Engineering - Apply log transformation to price-related numerical features
for col in num_cols:
    if df_train[col].skew() > 1:
        df_train[col] = np.log1p(df_train[col])

# Preprocessing for numerical data
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with median
    ("scaler", StandardScaler()),  # Normalize numerical features
    ("transform", PowerTransformer())  # Normalize skewed data
])

# Preprocessing for categorical data
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with mode
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # One-hot encoding
])

# Combine transformers
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Split train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(df_train, target, test_size=0.2, random_state=42)

# Define Optuna Hyperparameter Optimization
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 6, 16),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
    }
    
    model = lgb.LGBMRegressor(**params, random_state=42)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    
    return rmse

# Run Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)  # Increase n_trials for better results

# Best parameters
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train final model with best hyperparameters
model = lgb.LGBMRegressor(**best_params, random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f"Optimized Validation RMSE: {rmse:.4f}")


import lightgbm as lgb
import optuna


import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Fix whitespace in feature names
X_train.columns = X_train.columns.str.replace(' ', '_')
X_valid.columns = X_valid.columns.str.replace(' ', '_')

# Convert categorical features to category dtype
categorical_cols = ['Brand', 'Material','Laptop_Compartment', 'Waterproof', 'Size','Style', 'Color']
for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')
    X_valid[col] = X_valid[col].astype('category')

# Optimized hyperparameters
params = {
      'n_estimators': 1158,
    'max_depth': 12, 
    'learning_rate': 0.005522307143159027,
    'subsample': 0.6681235405078286, 
    'colsample_bytree': 0.8828573220841124, 
    'min_child_samples': 10
}
# Instantiate the model
model = lgb.LGBMRegressor(**params)

# Train with early stopping
model.fit(
    X_train, y_train, 
    eval_set=[(X_valid, y_valid)], 
    categorical_feature=categorical_cols,
    # early_stopping_rounds=100, 
    # verbose=100
)

# Make predictions
y_pred = model.predict(X_valid)

# Calculate RMSE
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f"Improved Validation RMSE: {rmse}")



from catboost import CatBoostClassifier, CatBoostRegressor


import catboost
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Fix whitespace in column names
X_train.columns = X_train.columns.str.replace(' ', '_')
X_valid.columns = X_valid.columns.str.replace(' ', '_')

# Define categorical features
categorical_cols = ['Brand', 'Material','Laptop_Compartment', 'Waterproof', 'Size','Style', 'Color']

# Convert categorical columns to string (CatBoost handles them internally)
X_train[categorical_cols] = X_train[categorical_cols].astype(str)
X_valid[categorical_cols] = X_valid[categorical_cols].astype(str)

# Define CatBoost parameters
catboost_params = {
    'iterations': 1000,           # More iterations for better training
    'depth': 8,                   # Optimal depth to prevent overfitting
    'learning_rate': 0.02,        # Reduce learning rate for better convergence
    'l2_leaf_reg': 5,             # Regularization to prevent overfitting
    'random_strength': 1,         # Prevents overfitting by adding noise to splits
    'bagging_temperature': 0.2,   # Helps with randomness for better generalization
    'border_count': 256,          # Number of bins for numerical features
    'eval_metric': 'RMSE',        # Root Mean Squared Error
    'random_seed': 42,
    'verbose': 100,               # Prints progress every 100 iterations
    'od_type': 'Iter',            # Early stopping
    'od_wait': 200,               # Stops training if no improvement
    'task_type': 'CPU',           # Change to 'GPU' if available for faster training
}

# Initialize and train the model
model = CatBoostRegressor(**catboost_params)
model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    cat_features=categorical_cols,
    # early_stopping_rounds=200,
    # verbose=100
)

# Make predictions
y_pred = model.predict(X_valid)

# Calculate RMSE
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f"CatBoost Validation RMSE: {rmse:.4f}")



import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load datasets
df_main = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")

# Fix column names
df_main.columns = df_main.columns.str.replace(' ', '_')
df_extra.columns = df_extra.columns.str.replace(' ', '_')

# Concatenate datasets
df_train = pd.concat([df_main, df_extra], ignore_index=True)

# Remove outliers using IQR method
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
    return df

df_train = remove_outliers(df_train, 'Price')

# Separate target variable
target = df_train['Price']
df_train.drop(columns=['Price'], inplace=True)

# Identify numerical and categorical columns
num_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()

# Feature Engineering - Apply log transformation to price-related numerical features
for col in num_cols:
    if df_train[col].skew() > 1:
        df_train[col] = np.log1p(df_train[col])

# Preprocessing for numerical data
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with median
    ("scaler", StandardScaler()),  # Normalize numerical features
    ("transform", PowerTransformer())  # Normalize skewed data
])

# Preprocessing for categorical data
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with mode
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # One-hot encoding
])

# Combine transformers
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Split train and validation data
X_train, X_valid, y_train, y_valid = train_test_split(df_train, target, test_size=0.2, random_state=42)

# Define Optuna Hyperparameter Optimization
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_iter': trial.suggest_int('max_iter', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 10.0),
        'max_bins': trial.suggest_int('max_bins', 128, 256),
        'early_stopping': True
    }
    
    model = HistGradientBoostingRegressor(**params, random_state=42)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    
    return rmse

# Run Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)  # Increase n_trials for better results

# Best parameters
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train final model with best hyperparameters
model = HistGradientBoostingRegressor(**best_params, random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f"Optimized Validation RMSE: {rmse:.4f}")



np.sqrt(mse_hist)


from sklearn.linear_model import LinearRegression

# Use Linear Regression for RMSE calculation
lin = LinearRegression()
lin.fit(X_train, y_train)

# Predict continuous values
y_pred = lin.predict(X_test)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")


import matplotlib.pyplot as plt

# Feature importance
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('LightGBM Feature Importance')
plt.show()

# Drop low-importance features if needed
low_importance_features = feature_importance[feature_importance['Importance'] < 5]['Feature'].tolist()
X_train.drop(columns=low_importance_features, inplace=True)
X_valid.drop(columns=low_importance_features, inplace=True)



# Load test data
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# Fix column names
df_test.columns = df_test.columns.str.replace(' ', '_')

# Ensure test data has the same features as training
for col in df_train.columns:
    if col not in df_test.columns:
        df_test[col] = np.nan  # Add missing columns with NaN

# Reorder test data columns to match training
df_test = df_test[df_train.columns]

# Apply the same transformations as in training
X_test_preprocessed = pipeline.named_steps['preprocessor'].transform(df_test)

# Make predictions
y_test_pred = pipeline.named_steps['model'].predict(X_test_preprocessed)

# Save predictions
submission = pd.DataFrame({'id': df_test['id'], 'Price': y_test_pred})
submission.to_csv("submission.csv", index=False)

print("Test predictions saved to submission.csv")





