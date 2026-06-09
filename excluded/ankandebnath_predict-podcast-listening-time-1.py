seed = 2025

import os
os.environ['PYTHONHASHSEED'] = str(seed) # Fix the environment-level hash seed (for data shuffling and other hash-based functions)

import random
import numpy as np
PI_VALUE = np.pi

# Set Python's built-in random seed
random.seed(seed)
# Set NumPy's random seed
np.random.seed(seed)


import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

train.shape, test.shape


train


train.isna().sum()


for idx, df in enumerate([train, test]):
    if idx == 0:
        print('Train')
    else:
        print('Test')
    null_percentage = (df.isna().sum() / len(df)) * 100
    print(null_percentage, '\n')


'Train', train.info(), 'Test', test.info()


'Train', train.describe().T, 'Test', test.describe().T


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


# Dealing with the peculiar 'Number_of_Ads' column...

# To convert or not convert (ain't that the question)?
train['Number_of_Ads'] = train['Number_of_Ads'].round().astype(int).copy()
test['Number_of_Ads'] = test['Number_of_Ads'].round().astype(int).copy()


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = train.copy()
df = df.dropna()

# Transformation
for col in num_cols:
    # Define a new column name for the transformed data
    transformed_col = col + '_transformed'
    
    # Box-Cox transformation
    df[transformed_col], _ = stats.boxcox(df[col] + 1)
    
    # Plotting the distributions
    plt.figure(figsize=(14, 6))
    
    # Original distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of column {col} (Original)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # Transformed distribution
    plt.subplot(1, 2, 2)
    sns.histplot(df[transformed_col], kde=True, bins=30)
    plt.title(f'Distribution of column {col} (Transformed)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()


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

# Binary Encoding for high-cardinality columns
binary_encoder = BinaryEncoder(cols=cat_cols)

# Transform both train and test datasets using the fitted encoder
train_encoded = binary_encoder.fit_transform(train[cat_cols])
test_encoded = binary_encoder.transform(test[cat_cols])

numerical_columns_test = [col for col in num_cols if col != 'Listening_Time_minutes']

# Ensure the column order matches between train and test datasets after one-hot encoding
train_encoded = pd.concat([train_encoded, train[num_cols]], axis=1)
test_encoded = pd.concat([test_encoded, test[numerical_columns_test]], axis=1)


train_encoded


missing_cols = ['Episode_Length_minutes', 'Guest_Popularity_percentage']

for col in missing_cols:
    train_encoded[col] = train_encoded[col].fillna(-1)
    test_encoded[col] = test_encoded[col].fillna(-1)


from sklearn.model_selection import train_test_split

X = train_encoded.drop(columns=['Listening_Time_minutes'])
y = train_encoded['Listening_Time_minutes']
X_test = test_encoded

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)


from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, 
                              BaggingRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
import time
import gc

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
    ('rf', RandomForestRegressor(random_state=seed, n_jobs=-1)),
    ('xgb', XGBRegressor(random_state=seed, eval_metric='rmse', n_jobs=-1))
]

models = {
    'Voting Regressor': VotingRegressor(estimators=base_estimators, n_jobs=-1),
    'Stacking Regressor': StackingRegressor(estimators=base_estimators, final_estimator=Ridge(random_state=seed), n_jobs=-1),
    'Random Forest': RandomForestRegressor(random_state=seed, n_jobs=-1),
    'CatBoost': CatBoostRegressor(verbose=0, random_state=seed),
    'XGBoost': XGBRegressor(random_state=seed, eval_metric='rmse', n_jobs=-1),
    'LightGBM': LGBMRegressor(verbose=0, random_state=seed),
    'Lasso (L1)': Lasso(alpha=0.9, random_state=seed),
    'Ridge (L2)': Ridge(alpha=0.9, random_state=seed),
    'Elastic Net': ElasticNet(alpha=0.9, l1_ratio=0.5, random_state=seed),
    'AdaBoost': AdaBoostRegressor(random_state=seed),
    'Gradient Boosting': GradientBoostingRegressor(random_state=seed),
    'HistGradient Boosting': HistGradientBoostingRegressor(random_state=seed),
    'Bagging': BaggingRegressor(estimator=DecisionTreeRegressor(), random_state=seed, n_jobs=-1),
    'Extra Trees': ExtraTreesRegressor(random_state=seed, n_jobs=-1)
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


# Print the results
for model_name, metrics in results.items():
    print(f"{model_name}: RMSE = {metrics['RMSE']:.4f}")


# After using 'Optuna' for Hyper-Parameter tuning, using the RandomForestRegressor(i.e; the best_model); and, using cross_val_score to calculate & find the best-score, I got the folln. values—
best_params = {'n_estimators': 279, 'max_depth': 25, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'auto', 'bootstrap': True}


from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit

model = RandomForestRegressor(**best_params, random_state=seed, n_jobs=-1)

# 1a) Standard 5-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
rmse_kf = []
for tr_idx, va_idx in kf.split(X):
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    preds = model.predict(X.iloc[va_idx])
    rmse_kf.append(np.sqrt(mean_squared_error(y.iloc[va_idx], preds)))
print("5-Fold CV RMSE:", np.mean(rmse_kf))

# 1b) Stratified on binned target
# bin y into 5 quantiles
y_binned = pd.qcut(y, q=5, labels=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
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



# from scipy.special import inv_boxcox

# # Predict using the best-model
# best_model = RandomForestRegressor(**best_params, random_state=seed, n_jobs=-1)
# best_model.fit(X, y)
# test_preds = best_model.predict(X_test)
# #test_preds = inv_boxcox(test_preds, l3_) - 1

# # Create the submission file
# submission = pd.DataFrame({'id': test['id'], 'Listening_Time_minutes': test_preds})
# submission.to_csv('submission.csv', index=False)


model1 = XGBRegressor(random_state=seed, eval_metric='rmse', n_jobs=-1); model2 = LGBMRegressor(verbose=0, random_state=seed)
model1.fit(X, y)
model2.fit(X, y)
test_preds1 = model1.predict(X_test); test_preds2 = model2.predict(X_test)
test_preds = (test_preds1 + test_preds2)/2

# Create the submission file
submission = pd.DataFrame({'id': test['id'], 'Listening_Time_minutes': test_preds})
submission.to_csv('submission.csv', index=False)


%%time

import shap
from sklearn.feature_selection import RFE

model = XGBRegressor(random_state=seed, eval_metric='rmse', n_jobs=-1)
model.fit(X, y)

# 2a) SHAP values
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X)
shap.summary_plot(shap_vals, X, feature_names=X.columns)
print('Step-1 Done!\n')

# 2b) Recursive Feature Elimination with Ridge
selector = RFE(Ridge(alpha=1.0), n_features_to_select=30, step=0.1)
selector = selector.fit(X, y)
selected_feats = np.array(X.columns)[selector.support_]
print("Top features:", selected_feats, '\n')


""" sample-code
# Incorporate original dataset
orig = pd.read_csv('original_podcast_dataset.csv')
full = pd.concat([train, orig], ignore_index=True).reset_index(drop=True)
# re-engineer features on `full`, then re-split into train/test

# Adversarial validation
full_feat = pd.concat([train.drop('Listening_Time_minutes',1), test], ignore_index=True)
labels_adv = np.concatenate([np.zeros(len(train)), np.ones(len(test))])
adv_model = RandomForestClassifier(n_estimators=100, random_state=seed)
adv_scores = cross_val_score(adv_model, full_feat, labels_adv, cv=5, scoring='roc_auc')
print("Adversarial AUC:", adv_scores.mean())"""

# if AUC≫0.5, features differ—consider re-sampling or domain adaptation.


