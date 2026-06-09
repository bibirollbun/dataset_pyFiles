import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Import the data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_ids = test['id'] # Save IDs for submission
train.head()


train.info()
print("\nMissing Values:")
print(train.isnull().sum())


train.describe()


# Check for duplicate IDs and feature rows
print("Train Duplicates (by ID): ", train['id'].duplicated().sum())
print("Test Duplicates (by ID): ", test['id'].duplicated().sum())

# Exclude id and target before checking for duplicate feature rows
train_duplicates = train.drop(columns=['id', 'accident_risk']).duplicated().sum()
print(f"Duplicate rows in train (excluding ID and target): {train_duplicates}")


numeric_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
for col in numeric_features:
    plt.figure(figsize=(6, 1))
    sns.boxplot(x=train[col])
    plt.title(col)
    plt.show()


# Histograms to check distributions
for col in ['curvature', 'speed_limit', 'num_reported_accidents']:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(train[col], label='Train', fill=True)
    sns.kdeplot(test[col], label='Test', fill=True)
    plt.title(f'{col} Distribution: Train vs Test')
    plt.legend()
    plt.show()


cat_features = ['road_type', 'lighting', 'weather', 'time_of_day']

for col in cat_features:
    train_vals = set(train[col].unique())
    test_vals = set(test[col].unique())
    only_in_train = train_vals - test_vals
    only_in_test = test_vals - train_vals

    print(f"\nColumn: {col}")
    print(f"  Unique in train: {train_vals}")
    print(f"  Unique in test:  {test_vals}")
    if only_in_train:
        print(f"  ðŸš¨ Only in train: {only_in_train}")
    if only_in_test:
        print(f"  ðŸš¨ Only in test: {only_in_test}")


sns.histplot(train['accident_risk'], bins=30, kde=True)
plt.title('Distribution of Accident Risk')
plt.show()

print(train['accident_risk'].describe())


# Calculate mean and count of accident_risk for each group of num_reported_accidents
accident_group_stats = train.groupby('num_reported_accidents')['accident_risk'].agg(['mean', 'count']).reset_index()
print(accident_group_stats)

plt.figure(figsize=(8, 4))
sns.barplot(x='num_reported_accidents', y='mean', data=accident_group_stats)
plt.title('Mean Accident Risk by Number of Reported Accidents')
plt.show()


def bin_accidents(df):
    # Create a categorical feature based on ranges: [0, 2], (2, 4], (4, 8]
    df['accident_group'] = pd.cut(df['num_reported_accidents'], 
                                  bins=[-1, 2, 4, 8], # Group 0,1,2 together, then 3,4, then 5+
                                  labels=['low', 'medium', 'high'],
                                  right=True)
    return df

train = bin_accidents(train)
test = bin_accidents(test)

print(train[['num_reported_accidents', 'accident_group', 'accident_risk']].head())


# Define feature lists for the pipeline
cat_features = ['road_type', 'lighting', 'weather', 'time_of_day', 'accident_group']
num_features = ['num_lanes', 'speed_limit']
bool_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']

# Features that require a log transformation for normalization
# np.log1p(x) is used for log(1+x) to handle zero values
log_features = ['num_reported_accidents', 'curvature']

# Transformer for logarithmic transformation
log_transformer = FunctionTransformer(lambda x: np.log1p(x), validate=True)

# Create the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
        ('num', StandardScaler(), num_features),
        ('log_trans', Pipeline(steps=[
            ('log', log_transformer),
            ('scaler', StandardScaler())
        ]), log_features),
        ('bool', 'passthrough', bool_features)
    ], 
    remainder='drop'
)


# Initialize the model and pipeline
lgbm = lgb.LGBMRegressor(
    objective='regression_l1',
    metric='rmse',
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgbm)
])

# Prepare data for CV
X = train.drop(columns=['id', 'accident_risk'])
y = train['accident_risk']


# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
fold = 0

print(f"Starting 5-Fold Cross-Validation...")

for train_index, val_index in kf.split(X, y):
    fold += 1
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Fit the pipeline
    pipe.fit(X_train, y_train)
    
    # Predict and evaluate
    val_preds = pipe.predict(X_val)
    
    # Clip predictions to the valid range [0, 1]
    val_preds = np.clip(val_preds, 0, 1)
    
    rmse = mean_squared_error(y_val, val_preds, squared=False)
    rmse_scores.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:.6f}")

print("\n--- Cross-Validation Results ---")
print(f"Mean CV RMSE: {np.mean(rmse_scores):.6f}")
print(f"Std Dev CV RMSE: {np.std(rmse_scores):.6f}")


# Train the final pipeline on the entire training dataset
pipe.fit(X, y)

# Predict on test set
test_features = test.drop(columns=['id', 'accident_risk'], errors='ignore')
preds = pipe.predict(test_features)
preds = np.clip(preds, 0, 1)  # Keep in valid range [0, 1]


submission = pd.DataFrame({'id': test_ids, 'accident_risk': preds})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")

