import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


train_df = pd.read_csv('/kaggle/input/sparta-2024-data-science-competition/train.csv')
test_df = pd.read_csv('/kaggle/input/sparta-2024-data-science-competition/test.csv')


print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Missing values:\n", train_df.isnull().sum().sort_values(ascending=False).head(10))

plt.figure(figsize=(10, 5))
sns.histplot(train_df['price'], bins=100, kde=True)
plt.title("Distribution of Price")
plt.show()



def preprocess_features(df):
    df = df.copy()

    # Convert boolean features
    bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'has_availability']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({'t': 1, 'f': 0}).fillna(0)

    # Convert percentage strings to floats
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in df.columns:
            df[col] = df[col].str.replace('%', '', regex=False).astype(float)/100

    # Convert date columns to durations
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    current_date = pd.to_datetime('2024-01-01')
    df['host_duration_days'] = (current_date - df['host_since']).dt.days
    df['days_since_last_review'] = (current_date - df['last_review']).dt.days

    # Amenities
    if 'amenities' in df.columns:
        df['num_amenities'] = df['amenities'].apply(lambda x: len(json.loads(x)) if pd.notnull(x) and x.startswith('[') else 0)
        df['has_wifi'] = df['amenities'].apply(lambda x: 1 if pd.notnull(x) and 'Wifi' in str(x) else 0)
        df['has_kitchen'] = df['amenities'].apply(lambda x: 1 if pd.notnull(x) and 'Kitchen' in str(x) else 0)

    # Fill NaN in categorical features
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("missing").astype(str)

    # Drop unused columns
    drop_cols = ['id', 'name', 'description', 'neighborhood_overview', 'host_id', 'host_url', 
                 'host_name', 'host_about', 'host_neighbourhood', 'host_verifications', 
                 'neighbourhood', 'bathrooms_text', 'amenities', 'host_since', 'first_review', 
                 'last_review', 'price']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    return df

train_processed_df = preprocess_features(train_df)
test_processed_df = preprocess_features(test_df)

# Align columns
missing_cols = set(train_processed_df.columns) - set(test_processed_df.columns)
for col in missing_cols:
    test_processed_df[col] = 0

extra_cols = set(test_processed_df.columns) - set(train_processed_df.columns)
test_processed_df = test_processed_df.drop(columns=list(extra_cols))

# Features and Target
X = train_processed_df.drop(columns=['price_log'], errors='ignore')
y = np.log1p(train_df['price'])  # log-transform

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert categorical columns ke category
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')
    X_val[col] = X_val[col].astype('category')
    test_processed_df[col] = test_processed_df[col].astype('category')


categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']

model_lgb = LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    learning_rate=0.05,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    n_estimators=5000,
    random_state=42
)

print("Training LightGBM...")
model_lgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=200)
    ]
)

y_pred_lgb = np.expm1(model_lgb.predict(X_val))
rmse_lgb = rmse(np.expm1(y_val), y_pred_lgb)
print("LightGBM RMSE:", rmse_lgb)


# setelah kita ubah jadi category
categorical_cols = [col for col in X_train.columns if str(X_train[col].dtype) == 'category']
cat_features = [X_train.columns.get_loc(c) for c in categorical_cols]

model_cat = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    early_stopping_rounds=100,
    verbose=200
)

model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features)
y_pred_cat = np.expm1(model_cat.predict(X_val))
rmse_cat = rmse(np.expm1(y_val), y_pred_cat)
print("CatBoost RMSE:", rmse_cat)


ensemble_pred = 0.7 * y_pred_lgb + 0.3 * y_pred_cat
rmse_ensemble = rmse(np.expm1(y_val), ensemble_pred)
print("Ensemble RMSE:", rmse_ensemble)


pred_lgb_test = np.expm1(model_lgb.predict(test_processed_df))
pred_cat_test = np.expm1(model_cat.predict(test_processed_df))
pred_ensemble_test = 0.7 * pred_lgb_test + 0.3 * pred_cat_test
pred_ensemble_test[pred_ensemble_test < 0] = 0

submission = pd.DataFrame({
    'id': test_df['id'],
    'price': pred_ensemble_test
})

submission.to_csv('submission_ensemble.csv', index=False)
print("Submission saved as submission_ensemble.csv")

