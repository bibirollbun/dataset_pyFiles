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
import lightgbm as lgb
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import gc



train_df = pd.read_csv('/kaggle/input/carnival-risk-analytics-challenge/train.csv')
target = 'Premium Amount'
id_col = 'id'
test_df = pd.read_csv('/kaggle/input/carnival-risk-analytics-challenge/test.csv')
print(f"\nFull Training Data Shape: {train_df.shape}")
print(f"Test Data Shape: {test_df.shape}")


print("\n Basic Dataset Overview:")
print(train_df.info())

plt.figure(figsize=(6,4))
sns.histplot(train_df['Premium Amount'].dropna(), bins=40, kde=True)
plt.title("Distribution of Premium Amount (Target)")
plt.show()


target = 'Premium Amount'
id_col = 'id'
train_df.head()




import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(12,5))
sns.heatmap(train_df.isnull(), cbar=False)
plt.title("Missing Value Heatmap")
plt.show()

missing = train_df.isnull().mean().sort_values(ascending=False)
print("\nMissing Values (%):")
print((missing*100).round(1))


num_cols = train_df.select_dtypes(include=np.number).columns
print("\nNumerical Feature Summary:")
display(train_df[num_cols].describe().T)

plt.figure(figsize=(14, 10))
cols_to_plot = ["Age", "Annual Income", "Health Score", "Credit Score", "Vehicle Age", "Insurance Duration", "Premium Amount"]
for i, col in enumerate(cols_to_plot, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train_df[col].dropna(), bins=40, kde=True)
    plt.title(f"{col} distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(train_df[num_cols].corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Matrix (Numeric Features)")
plt.show()




train_df['Policy Start Date'] = pd.to_datetime(train_df['Policy Start Date'])
train_df['start_year'] = train_df['Policy Start Date'].dt.year
train_df['start_month'] = train_df['Policy Start Date'].dt.month
train_df['start_dayofweek'] = train_df['Policy Start Date'].dt.dayofweek

X = train_df.drop(columns=[target, id_col, 'Policy Start Date'])
y = train_df[target]

valid_y_indices = y.dropna().index
X = X.loc[valid_y_indices]
y = y.loc[valid_y_indices]

num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include=['object']).columns

print(f"\nNumerical Features: {len(num_cols)} | Categorical Features: {len(cat_cols)}")


# --- 5. PREPROCESSING PIPELINE ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    n_jobs=-1
)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n Data split into training and validation sets.")




print("\n Training Random Forest Regressor...")
start_time = time.time()

rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=100, 
        max_depth=15,
        n_jobs=-1,
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)
rf_train_time = time.time() - start_time
rf_preds = rf_model.predict(X_valid)

rf_rmse = np.sqrt(mean_squared_error(y_valid, rf_preds))

print(f"\n Random Forest RMSE: {rf_rmse:.4f}")
print(f"Training Time: {rf_train_time/60:.2f} minutes")


print("\n Training LightGBM Regressor...")
start_time = time.time()

X_train_proc = preprocessor.fit_transform(X_train)
X_valid_proc = preprocessor.transform(X_valid)

lgb_model = lgb.LGBMRegressor(
    objective='regression_l1', 
    metric='rmse',
    n_estimators=2000,         
    learning_rate=0.035,        
    feature_fraction=0.75,      
    bagging_fraction=0.75,      
    bagging_freq=1,
    lambda_l1=0.2,            
    lambda_l2=0.2,             
    num_leaves=64,             
    verbose=-1,
    n_jobs=-1,
    seed=42
)


lgb_model.fit(X_train_proc, y_train,
              eval_set=[(X_valid_proc, y_valid)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])

lgb_train_time = time.time() - start_time
lgb_preds = lgb_model.predict(X_valid_proc)

lgb_rmse = np.sqrt(mean_squared_error(y_valid, lgb_preds))

print(f"\n LightGBM RMSE: {lgb_rmse:.4f}")
print(f"Training Time: {lgb_train_time/60:.2f} minutes")



print("\n Model Comparison:")
print(pd.DataFrame({
    'Model': ['Random Forest', 'LightGBM'],
    'RMSE': [rf_rmse, lgb_rmse],
    'Training Time (min)': [rf_train_time/60, lgb_train_time/60]
}))



gc.collect()
print("\n Pipeline completed successfully!")
test_df = pd.read_csv('/kaggle/input/carnival-risk-analytics-challenge/train.csv')
test_df['Policy Start Date'] = pd.to_datetime(test_df['Policy Start Date'])
test_df['start_year'] = test_df['Policy Start Date'].dt.year
test_df['start_month'] = test_df['Policy Start Date'].dt.month
test_df['start_dayofweek'] = test_df['Policy Start Date'].dt.dayofweek

X_test = test_df.drop(columns=['id', 'Policy Start Date'])

X_test_proc = preprocessor.transform(X_test)

test_preds = lgb_model.predict(X_test_proc)




submission = pd.DataFrame({
    'id': test_df['id'],
    'Premium Amount': test_preds
})

submission.to_csv('submission.csv', index=False)
print("\n Submission file 'submission.csv' created successfully!")




