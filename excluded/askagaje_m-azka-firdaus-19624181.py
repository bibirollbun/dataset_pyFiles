import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully.")
train_df = pd.read_csv('/kaggle/input/sparta-2024-data-science-competition/train.csv')
test_df = pd.read_csv('/kaggle/input/sparta-2024-data-science-competition/test.csv')

print("Starting feature engineering...")
train_df['price'] = train_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
train_df['price_log'] = np.log1p(train_df['price'])

all_df = pd.concat([train_df.drop(['price', 'price_log'], axis=1), test_df], axis=0)

all_df['host_since'] = pd.to_datetime(all_df['host_since'])
ref_date = pd.to_datetime('2025-08-01')
all_df['host_duration_days'] = (ref_date - all_df['host_since']).dt.days

all_df['host_response_rate'] = all_df['host_response_rate'].str.replace('%', '', regex=False).astype(float) / 100
all_df['host_acceptance_rate'] = all_df['host_acceptance_rate'].str.replace('%', '', regex=False).astype(float) / 100
all_df['amenities'] = all_df['amenities'].fillna('[]')
all_df['num_amenities'] = all_df['amenities'].apply(lambda x: len(x.split(',')))
top_amenities = ['Wifi', 'Kitchen', 'Heating', 'Air conditioning', 'Washer', 'Dryer']
for amenity in top_amenities:
    all_df[f'has_{amenity.lower().replace(" ", "_")}'] = all_df['amenities'].str.contains(amenity, case=False)

for col in ['host_is_superhost', 'host_identity_verified']:
    all_df[col] = all_df[col].replace({'t': 1, 'f': 0})
for col in ['room_type', 'host_response_time']:
    all_df[col] = LabelEncoder().fit_transform(all_df[col].astype(str))
neighborhood_map = train_df.groupby('neighbourhood_cleansed')['price_log'].mean()
all_df['neighbourhood_encoded'] = all_df['neighbourhood_cleansed'].map(neighborhood_map)

num_cols_to_impute = [
    'host_listings_count', 'host_acceptance_rate', 'host_response_rate',
    'bathrooms', 'bedrooms', 'beds', 'review_scores_rating', 'reviews_per_month'
]
for col in num_cols_to_impute:
    all_df[col].fillna(all_df[col].median(), inplace=True)

cols_to_drop = [
    'id', 'name', 'description', 'neighborhood_overview', 'host_id', 'host_name',
    'host_since', 'host_location', 'host_about', 'host_neighbourhood', 'host_verifications',
    'neighbourhood', 'neighbourhood_cleansed', 'amenities', 'first_review', 'last_review',
    'bathrooms_text', 'host_has_profile_pic'
]
all_df.drop(columns=cols_to_drop, inplace=True)

all_df = pd.get_dummies(all_df, dummy_na=False)
all_df.columns = ["".join(c if c.isalnum() else '_' for c in str(x)) for x in all_df.columns]
all_df = all_df.loc[:, ~all_df.columns.duplicated()]

X = all_df[:len(train_df)]
X_test = all_df[len(train_df):]
y = train_df['price_log']

print(f"Feature engineering complete. Final training shape: {X.shape}")


print("\n--- Training LightGBM Model ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_params = {
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000,
    'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
    'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42
}

oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training Fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    oof_predictions[val_idx] = model.predict(X_val)
    test_predictions += model.predict(X_test) / kf.n_splits

rmse_score = np.sqrt(mean_squared_error(y, oof_predictions))
print(f"\nFinal OOF RMSE Score: {rmse_score:.5f}")

print("\n--- Creating Submission File ---")

# Convert predictions back from log scale to original price scale
final_predictions = np.expm1(test_predictions)

# Create submission DataFrame and save
submission = pd.DataFrame({'id': test_df['id'], 'price': final_predictions})
submission['price'] = submission['price'].clip(0)
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
print(submission.head())

