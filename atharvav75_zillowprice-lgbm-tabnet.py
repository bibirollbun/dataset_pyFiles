!pip install pytorch-tabnet -q
!pip install category_encoders -q


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import gc
import warnings
warnings.filterwarnings('ignore')


#Loading the 2016 train + properties
train2016 = pd.read_csv('/kaggle/input/zillow-prize-1/train_2016_v2.csv')
props2016 = pd.read_csv('/kaggle/input/zillow-prize-1/properties_2016.csv')


#Loadin the 2017 train + properties (for extra data)
train2017 = pd.read_csv('/kaggle/input/zillow-prize-1/train_2017.csv')
props2017 = pd.read_csv('/kaggle/input/zillow-prize-1/properties_2017.csv')


#Merging 2016
df2016 = train2016.merge(props2016, on='parcelid', how='left')


#Merging 2017(fill missing from 2016 props where possible)
df2017 = train2017.merge(props2017, on='parcelid', how='left')


#For 2017 missing props filling from 2016 if same parcel
missing_2017 = df2017[df2017['latitude'].isna()]  # Example: if key features missing
df2017 = df2017.combine_first(missing_2017.merge(props2016, on='parcelid', how='left', suffixes=('', '_2016')))


#Combining both years
df = pd.concat([df2016, df2017], ignore_index=True, sort=False)


#Cliping the outliers (important for low error)
df['logerror'] = df['logerror'].clip(lower=-0.4, upper=0.419)


print(f"Combined data shape: {df.shape}")
del train2016, props2016, train2017, props2017
gc.collect()


# Cell 3: Feature Engineering (Updated - Safe Column Filtering)
print("Debug: Checking columns after merge...")

# First, print actual columns in merged df to debug
print("2016 Merged Columns (first 10):", df2016.columns[:10].tolist())
print("2017 Merged Columns (first 10):", df2017.columns[:10].tolist())
print(f"Total unique columns in combined df: {len(df.columns)}")

# Useful columns list (same as before, but now we'll filter safe way)
useful_cols = [
    'parcelid', 'logerror', 'transactiondate', 'airconditioningtypeid', 'architecturalstyletypeid',
    'basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid', 'buildingqualitytypeid',
    'calculatedbathnbr', 'calculatedbedroomnbr', 'calculatedfinishedsquarefeet',
    'calculatedfinishedsquarefeet6', 'calculatedfinishedsquarefeet12', 'calculatedfinishedsquarefeet13',
    'calculatedfinishedsquarefeet15', 'calculatedparkingnbr', 'cityid', 'coolingtypeid',
    'countyuse1code', 'decktypeid', 'finishedfloor1squarefeet', 'finishedsquarefeet6',
    'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15', 'fips', 'fireplacecnt',
    'fullbathcnt', 'garagecarcnt', 'garagetotalsqft', 'hashottuborspa', 'heatingtypeid',
    'landmarkpoint1', 'landmarkpoint2', 'landmarkpoint3', 'landmarkpoint4', 'latitude',
    'longitude', 'lotsizesquarefeet', 'numberofstories', 'poolcnt', 'pooltypeid10',
    'pooltypeid2', 'pooltypeid7', 'propertylandusetypeid', 'propertyzoningdesc',
    'rawcensustractandblock', 'regionidcity', 'regionidcounty', 'regionidneighborhood',
    'regionidzip', 'roomcnt', 'threequarterbathnbr', 'typeconstructiontypeid',
    'unitcnt', 'utilitytype', 'yardbuildingsqft17', 'yearbuilt', 'numberofstories',
    'fireplaceflag', 'taxamount', 'taxassessedvalue', 'taxdelinquencyflag', 'taxdelinquencyyear',
    'taxlien', 'assessmentyear', 'taxassessorvalue', 'taxassessedyear'
]

#SAFE FILTER: Only keep columns that actually exist in df to avoid key error
existing_cols = [col for col in useful_cols if col in df.columns]
missing_cols = [col for col in useful_cols if col not in df.columns]
print(f"Missing columns (ignored): {missing_cols}")  # Yeh print hoga, e.g., ['landmarkpoint1', ...] if any
print(f"Using {len(existing_cols)} columns instead.")

df = df[existing_cols].copy()  # Now safe - no error!

#Rest of engineering same
#Date features
df['transactiondate'] = pd.to_datetime(df['transactiondate'])
df['transaction_month'] = df['transactiondate'].dt.month
df['transaction_year'] = df['transactiondate'].dt.year
df['transaction_day'] = df['transactiondate'].dt.day

#Engineering new features
if 'yearbuilt' in df.columns:
    df['age'] = 2017 - df['yearbuilt'].fillna(0)
if all(col in df.columns for col in ['taxamount', 'calculatedfinishedsquarefeet']):
    df['tax_per_sqft'] = df['taxamount'] / (df['calculatedfinishedsquarefeet'] + 1)
if all(col in df.columns for col in ['bedroomcnt', 'bathroomcnt']):
    df['bed_bath_ratio'] = df['bedroomcnt'] / (df['bathroomcnt'] + 1)
if all(col in df.columns for col in ['garagetotalsqft', 'garagecarcnt']):
    df['garage_area'] = df['garagetotalsqft'].fillna(0) + df['garagecarcnt'].fillna(0) * 200

#Location clusters (safe)
if all(col in df.columns for col in ['latitude', 'longitude']):
    from sklearn.cluster import KMeans
    coords = df[['latitude', 'longitude']].fillna(0)
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    df['region_cluster'] = kmeans.fit_predict(coords)

# Handle NaNs: Numerical median, Categorical mode
num_cols = df.select_dtypes(include=[np.number]).columns.drop(['parcelid', 'logerror', 'transaction_month', 'transaction_year', 'transaction_day'], errors='ignore')
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    mode_val = df[col].mode()
    df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')

#Label encode categoricals (safe)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1, errors='ignore')
y = df['logerror']

print("Checking for duplicate columns...")
duplicate_cols = df.columns[df.columns.duplicated()].tolist()
if duplicate_cols:
    print(f"Found duplicate columns: {duplicate_cols}")
    df = df.loc[:, ~df.columns.duplicated()]
else:
    print("No duplicate columns – All good!")

print(f"Final columns count: {df.shape[1]}")

X = df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1, errors='ignore')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

#Cleaning up
del df
if 'coords' in locals():
    del coords
gc.collect()


# CELL 4: LightGBM Training (Super Stable & Fast)
print("=== Training LightGBM ===")
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

lgb_params = {
    'n_estimators': 20000,
    'learning_rate': 0.008,
    'num_leaves': 128,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.85,
    'colsample_bytree': 0.65,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'objective': 'regression',
    'metric': 'mae',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}

lgb_model = lgb.LGBMRegressor(**lgb_params)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)]
)

lgb_pred_test = lgb_model.predict(X_test)
lgb_mae = mean_absolute_error(y_test, lgb_pred_test)
print(f"LightGBM Local MAE: {lgb_mae:.6f}")

# Full train pe retrain for final submission
lgb_model.fit(X, y)
lgb_full_pred = lgb_model.predict(X)
print("LightGBM full train done!")


print("Training TabNet")

from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import numpy as np

y_train_2d = y_train.values.reshape(-1, 1)
y_test_2d  = y_test.values.reshape(-1, 1)

tabnet = TabNetRegressor(
    n_d=64, n_a=64,
    n_steps=5,
    gamma=1.5,
    lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 30, "gamma": 0.95},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax',
    verbose=10,
    seed=42
)

tabnet.fit(
    X_train=X_train.values, 
    y_train=y_train_2d,                
    eval_set=[(X_test.values, y_test_2d)],
    eval_name=['valid'],
    eval_metric=['mae'],
    max_epochs=300,
    patience=60,
    batch_size=2048,
    virtual_batch_size=256
)

tabnet_pred_test = tabnet.predict(X_test.values).flatten()
tabnet_pred_full = tabnet.predict(X.values).flatten()

print(f"TabNet Local MAE: {mean_absolute_error(y_test, tabnet_pred_test):.6f}")
print("TabNet successfully trained!")


print("Training Second LightGBM")
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

lgb2_params = {
    'n_estimators': 20000,
    'learning_rate': 0.009,
    'num_leaves': 96,
    'max_depth': -1,
    'min_child_samples': 25,
    'subsample': 0.82,
    'colsample_bytree': 0.68,
    'reg_alpha': 0.3,
    'reg_lambda': 0.1,
    'random_state': 101,
    'n_jobs': -1,
    'verbosity': -1
}

lgb2 = lgb.LGBMRegressor(**lgb2_params)
lgb2.fit(X_train, y_train,
         eval_set=[(X_test, y_test)],
         eval_metric='mae',
         callbacks=[lgb.early_stopping(250)])

lgb2_pred = lgb2.fit(X, y).predict(X)
print(f"Second LGBM done! Local MAE: {mean_absolute_error(y_test, lgb2.predict(X_test)):.6f}")


lgb1_pred = lgb_model.predict(X)

lgb2_pred = lgb2.predict(X)

tabnet_pred = tabnet.predict(X.values).flatten()

final_pred = (0.40 * lgb1_pred +
              0.50 * tabnet_pred +
              0.10 * lgb2_pred)

from sklearn.metrics import mean_absolute_error
print(f"FINAL LOCAL CV MAE = {mean_absolute_error(y, final_pred):.6f}")

import pandas as pd
sub = pd.read_csv('/kaggle/input/zillow-prize-1/sample_submission.csv')
value = final_pred.mean()

sub['201610'] = value
sub['201611'] = value
sub['201612'] = value
sub['201710'] = value
sub['201711'] = value
sub['201712'] = value

sub.to_csv('submission.csv', index=False)

