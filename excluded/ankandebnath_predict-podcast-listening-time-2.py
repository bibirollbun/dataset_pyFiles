seed1 = 2025

import os
os.environ['PYTHONHASHSEED'] = str(seed1) # Fix the environment-level hash seed (for data shuffling and other hash-based functions)

import random
import numpy as np
PI_VALUE = np.pi

# Set Python's built-in random seed
random.seed(seed1)
# Set NumPy's random seed
np.random.seed(seed1)


import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

train.shape, test.shape


train


extra_data = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')   # leveraging additional-data
extra_data.shape


train.isna().sum()


for idx, df in enumerate([train, test]):
    if idx == 0:
        print('Train')
    else:
        print('Test')
    null_percentage = (df.isna().sum() / len(df)) * 100
    print(null_percentage, '\n')


print(set(train.columns)-set(extra_data.columns), '\n')
print(extra_data.isna().sum())


extra_data = extra_data.dropna(subset=['Listening_Time_minutes'])
print(extra_data.shape, '\n')

extra_data.isna().sum()


'Train\n', train.info(), 'Test\n', test.info(), 'Extra-data\n', extra_data.info()


'Train\n', train.describe().T, 'Test\n', test.describe().T, 'Extra-data\n', extra_data.describe().T


train = train.dropna(subset=['Number_of_Ads'])
train.shape


num_cols = train.select_dtypes(include=['number']).columns.tolist()
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols.remove('id')

num_cols, cat_cols


# Examining the categorical columns
for col in cat_cols:
    print()
    print(train[col].value_counts())
    print(f'Number of different categories: {train[col].nunique()}')


train = pd.concat([train, extra_data], ignore_index=True)
train = train.drop_duplicates()
train.shape


# Dealing with the peculiar 'Number_of_Ads' column...

# To convert or not convert (ain't that the question)?
train['Number_of_Ads'] = train['Number_of_Ads'].round().astype(int).copy()
test['Number_of_Ads'] = test['Number_of_Ads'].round().astype(int).copy()


from sklearn.preprocessing import StandardScaler

cols_to_scale = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage']
l3_ = 0

for col in cols_to_scale:
    scaler = StandardScaler()
    train[col] = scaler.fit_transform(train[col].values.reshape(-1, 1))
    test[col] = scaler.transform(test[col].values.reshape(-1, 1))

# train['Number_of_Ads'] = np.log1p(train['Number_of_Ads'])
# test['Number_of_Ads'] = np.log1p(test['Number_of_Ads'])

#train['Listening_Time_minutes'], l3_ = stats.boxcox(train['Listening_Time_minutes'] + 1)
print(f'Lambda-value: {l3_}')


from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelEncoder

cat_cols.remove('Publication_Time')
cat_cols.remove('Publication_Day')

# Binary Encoding for high-cardinality columns
binary_encoder = BinaryEncoder(cols=cat_cols)

# Transform both train and test datasets using the fitted encoder
train_encoded = binary_encoder.fit_transform(train[cat_cols])
test_encoded = binary_encoder.transform(test[cat_cols])

numerical_columns_test = [col for col in num_cols if col != 'Listening_Time_minutes']

# Ensure the column order matches between train and test datasets after one-hot encoding
train_encoded = pd.concat(
    [train_encoded, train[num_cols], train[['Publication_Time', 'Publication_Day']]],
    axis=1
)
test_encoded = pd.concat(
    [test_encoded, test[numerical_columns_test], test[['Publication_Time', 'Publication_Day']]],
    axis=1
)

cols_to_le = ['Publication_Time', 'Publication_Day']

for col in cols_to_le:
    le = LabelEncoder()
    train_encoded[col] = le.fit_transform(train_encoded[col].values.ravel())
    test_encoded[col] = le.transform(test_encoded[col].values.ravel())

for df in [train_encoded, test_encoded]:
    df['Publication_Time_sin'] = np.sin(PI_VALUE * df['Publication_Time'] / 2)
    df['Publication_Time_cos'] = np.cos(PI_VALUE * df['Publication_Time'] / 2)
    df['Publication_Day_sin'] = np.sin(2 * PI_VALUE * df['Publication_Day'] / 7)
    df['Publication_Day_cos'] = np.cos(2 * PI_VALUE * df['Publication_Day'] / 7)
    #df['is_Weekend'] = df['Publication_Day'].apply(lambda x: 1 if x in [2, 3] else 0)
    #df.drop(columns=cols_to_le, inplace=True)


train_encoded


missing_cols = ['Episode_Length_minutes', 'Guest_Popularity_percentage']

def fill_missing_values(df, columns):
    """Fill NaN values with (-1) and create a 'was_missing' indicator column."""
    for col in columns:
        df[f'{col}_was_missing'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(-1)
    return df

train_encoded = fill_missing_values(train_encoded, missing_cols)
test_encoded = fill_missing_values(test_encoded, missing_cols)


from sklearn.model_selection import train_test_split

X = train_encoded.drop(columns=['Listening_Time_minutes'])
y = train_encoded['Listening_Time_minutes']
X_test = test_encoded

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed1)


X.shape, y.shape


from scipy.stats import zscore

# Apply Z-score to all columns
z_scores = train_encoded[num_cols].apply(zscore)

# Identify potential outliers (absolute Z-score > 3)
df_zscore_summary = pd.DataFrame({
    "Column": num_cols,
    "Avg_Abs_Z-Score": z_scores.abs().mean().values  # Average absolute Z-score
})

print(df_zscore_summary)


import matplotlib.pyplot as plt
import seaborn as sns

# Number of plots
n_cols = len(num_cols)
fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(5 * n_cols, 5))

# Plot each column on its subplot
for ax, col in zip(axes, num_cols):
    sns.boxplot(y=train[col], ax=ax, color="skyblue")
    ax.set_title(col)
    
plt.tight_layout()
plt.show()


from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import gc

model = XGBRegressor(random_state=seed1, eval_metric='rmse', n_jobs=-1)

# 1a) Standard 5-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=seed1)
rmse_kf = []
for tr_idx, va_idx in kf.split(X):
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    preds = model.predict(X.iloc[va_idx])
    rmse_kf.append(np.sqrt(mean_squared_error(y.iloc[va_idx], preds)))
print("5-Fold CV RMSE:", np.mean(rmse_kf))

# 1b) Stratified on binned target
# bin y into 5 quantiles
y_binned = pd.qcut(y, q=5, labels=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed1)
rmse_skf = []
for tr_idx, va_idx in skf.split(X, y_binned):
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    preds = model.predict(X.iloc[va_idx])
    rmse_skf.append(np.sqrt(mean_squared_error(y.iloc[va_idx], preds)))
print("Stratified 5-Fold RMSE:", np.mean(rmse_skf))

# 1c) GroupKFold by Podcast_Name
gkf = GroupKFold(n_splits=5)
rmse_gkf = []
for tr_idx, va_idx in gkf.split(X, y, groups=train['Podcast_Name']):
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    preds = model.predict(X.iloc[va_idx])
    rmse_gkf.append(np.sqrt(mean_squared_error(y.iloc[va_idx], preds)))
print("GroupKFold RMSE:", np.mean(rmse_gkf))

del model, preds, kf, rmse_kf, y_binned, skf, rmse_skf, gkf, rmse_gkf
gc.collect()


from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, 
                              BaggingRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor)
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
import time

# Function to evaluate regression model
def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    del model, y_pred
    gc.collect()
    return rmse

# Define a list of base estimators for composite ensemble models.
base_estimators = [
    ('rf', RandomForestRegressor(random_state=seed1, n_jobs=-1)),
    ('xgb', XGBRegressor(random_state=seed1, eval_metric='rmse', n_jobs=-1))
]

models = {
    'Voting Regressor': VotingRegressor(estimators=base_estimators, n_jobs=-1),
    'Stacking Regressor': StackingRegressor(estimators=base_estimators, final_estimator=Ridge(random_state=seed1), n_jobs=-1),
    'Random Forest': RandomForestRegressor(random_state=seed1, n_jobs=-1),
    'CatBoost': CatBoostRegressor(verbose=0, random_state=seed1),
    'XGBoost': XGBRegressor(random_state=seed1, eval_metric='rmse', n_jobs=-1),
    'LightGBM': LGBMRegressor(verbose=0, random_state=seed1),
    'Lasso (L1)': Lasso(alpha=0.9, random_state=seed1),
    'Ridge (L2)': Ridge(alpha=0.9, random_state=seed1),
    'Elastic Net': ElasticNet(alpha=0.9, l1_ratio=0.5, random_state=seed1),
    'AdaBoost': AdaBoostRegressor(random_state=seed1),
    'Gradient Boosting': GradientBoostingRegressor(random_state=seed1),
    'HistGradient Boosting': HistGradientBoostingRegressor(random_state=seed1),
    'Bagging': BaggingRegressor(estimator=DecisionTreeRegressor(), random_state=seed1, n_jobs=-1),
    'Extra Trees': ExtraTreesRegressor(random_state=seed1, n_jobs=-1)
}

c = 0
# Evaluate each model
results = {}
for name, model in models.items():
    start_time = time.time()
    rmse = evaluate_model(model, X_train, y_train, X_val, y_val)
    end_time = time.time()
    results[name] = {'RMSE': rmse}
    c += 1
    print(f'Done: {c}, time-taken: {end_time-start_time:.2f}s')

del name, model, models, c, base_estimators
gc.collect()


# Print the results
for model_name, metrics in results.items():
    print(f"{model_name}: RMSE = {metrics['RMSE']:.4f}")


from sklearn.svm import SVR

# Initialize SVR model
svr = SVR()

# Train the model
svr.fit(X_train, y_train)

# Make predictions
y_pred = svr.predict(X_val)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'RMSE: {rmse}')


from sklearn.neighbors import KNeighborsRegressor

# Define range of k values to test
k_values = range(1, 21)
errors = []

# Loop through different values of k
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    errors.append(rmse)

# Plot RMSE vs. k
plt.figure(figsize=(8, 5))
plt.plot(k_values, errors, marker='o', linestyle='dashed', label='RMSE')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.title("Elbow Method for Optimal k")
plt.legend()
plt.grid(True)
plt.show()



del X_train, y_train, X_val, y_val
gc.collect()


# After using 'Optuna' for Hyper-Parameter tuning, using the RandomForestRegressor(i.e; the best_model); and, using cross_val_score to calculate & find the best-score, I got the folln. values—
best_params = {'n_estimators': 279, 'max_depth': 25, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'auto', 'bootstrap': True}


from scipy.special import inv_boxcox

# Predict using the best-model
best_model = RandomForestRegressor(**best_params, random_state=seed1, n_jobs=-1)
best_model.fit(X, y)
test_preds = best_model.predict(X_test)
#test_preds = inv_boxcox(test_preds, l3_) - 1

# Create the submission file
submission = pd.DataFrame({'id': test['id'], 'Listening_Time_minutes': test_preds})
submission.to_csv('submission.csv', index=False)

