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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


train_16 = pd.read_csv("/kaggle/input/zillow-prize-1/train_2016_v2.csv")
train_17 = pd.read_csv("/kaggle/input/zillow-prize-1/train_2017.csv")
props_16 = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2016.csv")
props_17 = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2017.csv")
sample_submission = pd.read_csv("/kaggle/input/zillow-prize-1/sample_submission.csv")


print("DATASET SHAPES")
print(f"Properties 2016: {props_16.shape}")
print(f"Properties 2017: {props_17.shape}")
print(f"Training 2016: {train_16.shape}")
print(f"Training 2017: {train_17.shape}")
print(f"Sample Submission: {sample_submission.shape}")

print("PROPERTIES 2016 - INFO")
print(props_16.info())

print("TRAINING 2016 - INFO")
print(train_16.info())


print("TARGET VARIABLE ANALYSIS (logerror)")
print(train_16['logerror'].describe())

print("UNIQUE VALUES")

print(f"Unique ParcelIDs in properties: {props_16['parcelid'].nunique()}")
print(f"Unique ParcelIDs in training:   {train_16['parcelid'].nunique()}")
print(f"Training rows per ParcelID:     {train_16['parcelid'].value_counts().mean():.2f}")


df_2016 = train_16.merge(props_16, on="parcelid", how="left")
df_2017 = train_17.merge(props_17, on="parcelid", how="left")
df = pd.concat([df_2016, df_2017], axis=0).reset_index(drop=True)


df.head()


missing_pct = (df.isnull().sum() / len(df) * 100).sort_values()
missing_features = missing_pct[missing_pct > 0]
print(missing_features)
high_missing = missing_pct[missing_pct > 80].index
df = df.drop(columns=high_missing)


df = df[(df['logerror'] > -0.4) & (df['logerror'] < 0.4)]


df.shape


def preprocessing(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    for col in categorical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col].fillna('MISSING', inplace=True)
            print(f"  {col}: filled {missing_count} missing with 'MISSING'")
    return df

df = preprocessing(df)


plt.hist(df['logerror'], bins=50)
plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(x=df["logerror"], color="skyblue")
plt.show()


def datetime(df):
    if 'transactiondate' in df.columns:
        df['date'] = pd.to_datetime(df['transactiondate'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['quarter'] = df['date'].dt.quarter
        df['month_name'] = df['date'].dt.strftime('%b')
    return df

df = datetime(df)


plt.figure(figsize=(10,8))
daily_counts = df.groupby('date').size()
plt.plot(daily_counts.index, daily_counts.values, alpha=0.7, color='steelblue')


def feature_engineering(df):
    if 'yearbuilt' in df.columns:
        df['property_age'] = 2017 - df['yearbuilt']
        df['property_age'] = df['property_age'].clip(lower=0, upper=200)
        
    if 'calculatedfinishedsquarefeet' in df.columns and 'lotsizesquarefeet' in df.columns:
        df['living_area_ratio'] = df['calculatedfinishedsquarefeet'] / (df['lotsizesquarefeet'] + 1)
        df['living_area_ratio'] = df['living_area_ratio'].clip(upper=1)
        df['extra_space'] = df['lotsizesquarefeet'] - df['calculatedfinishedsquarefeet']
        
    if 'taxamount' in df.columns and 'taxvaluedollarcnt' in df.columns:
        df['tax_rate'] = df['taxamount'] / (df['taxvaluedollarcnt'] + 1)
    
    if 'structuretaxvaluedollarcnt' in df.columns and 'landtaxvaluedollarcnt' in df.columns:
        df['structure_land_ratio'] = df['structuretaxvaluedollarcnt'] / (df['landtaxvaluedollarcnt'] + 1)
        df['total_tax_value'] = df['structuretaxvaluedollarcnt'] + df['landtaxvaluedollarcnt']
    
    if 'bedroomcnt' in df.columns and 'bathroomcnt' in df.columns:
        df['total_rooms'] = df['bedroomcnt'] + df['bathroomcnt']
        df['bath_bed_ratio'] = df['bathroomcnt'] / (df['bedroomcnt'] + 1)
    
    if 'bedroomcnt' in df.columns and 'calculatedfinishedsquarefeet' in df.columns:
        df['sqft_per_bedroom'] = df['calculatedfinishedsquarefeet'] / (df['bedroomcnt'] + 1)
    
    if 'roomcnt' in df.columns and 'calculatedfinishedsquarefeet' in df.columns:
        df['sqft_per_room'] = df['calculatedfinishedsquarefeet'] / (df['roomcnt'] + 1)
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['location_sum'] = df['latitude'] + df['longitude']
        df['location_diff'] = df['latitude'] - df['longitude']
        df['location_product'] = df['latitude'] * df['longitude']
    return df

df = feature_engineering(df)


def log_transform(df):
    log_transform_cols = ['taxvaluedollarcnt', 'taxamount', 'calculatedfinishedsquarefeet', 
                          'lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt']
    
    for col in log_transform_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
    return df

df = log_transform(df)


from sklearn.preprocessing import LabelEncoder


def encoding(df):
    exclude_cols = ['parcelid','logerror','transactiondate','date','month_name','age_category']
    all_cols = df.columns
    features = [col for col in all_cols if col not in exclude_cols]
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    return df

df = encoding(df)


exclude_cols = ['parcelid','logerror','transactiondate','date','month_name','age_category']
all_cols = df.columns
features = [col for col in all_cols if col not in exclude_cols]


X = df[features]
y = df['logerror']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)


X_train.info()


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# X_scaled_full = scaler.fit_transform(X)


y_mean = y_train.mean()
y_std = y_train.std() if y_train.std() != 0 else 1.0
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std


mae = {}
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train_scaled)
lin_preds_scaled = lin_reg.predict(X_test_scaled)
lin_preds = lin_preds_scaled * y_std + y_mean
lr_mae = mean_absolute_error(y_test, lin_preds)
mae['lin-reg'] = lr_mae
print("Linear Regression MAE :", lr_mae)


ridge = Ridge(alpha=0.5)
ridge.fit(X_train_scaled, y_train_scaled)
ridge_preds_scaled = ridge.predict(X_test_scaled)
ridge_preds = ridge_preds_scaled * y_std + y_mean
ridge_mae = mean_absolute_error(y_test, ridge_preds)
mae['ridge'] = ridge_mae
print("Ridge MAE:", ridge_mae)


from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train_scaled, y_train_scaled)
xgb_preds_scaled = xgb.predict(X_test_scaled)
xgb_preds = xgb_preds_scaled * y_std + y_mean
xgb_mae = mean_absolute_error(y_test, xgb_preds)
mae['xgboost'] = xgb_mae
print("XGBoost MAE:", xgb_mae)


best_model = None
best_mae = float('inf')

for model, mae in mae.items():
    if mae < best_mae:
        best_mae = mae
        best_model = model

print(best_model, best_mae)


sample_submission.columns


submission_parcel_ids = sample_submission['ParcelId'].unique()
X_final = props_17[props_17['parcelid'].isin(submission_parcel_ids)]


test_parcelids = X_final['parcelid']


X_final = datetime(X_final) 
X_final = feature_engineering(X_final)


X_final = X_final.drop(columns=high_missing, errors='ignore')
X_final = preprocessing(X_final)
X_final = encoding(X_final)
X_final = log_transform(X_final)


X_final = X_final.reindex(columns=X.columns, fill_value=0)


X_final_scaled = scaler.transform(X_final)
X_final_scaled = pd.DataFrame(X_final_scaled, 
                                   index=X_final.index, 
                                   columns=X_final.columns)


xgb_preds_final= xgb.predict(X_final_scaled)
final_test_pred = (xgb_preds_final * y_std) + y_mean


final_predictions_df = pd.DataFrame({
    'parcelid': test_parcelids,
    'logerror_pred': final_test_pred
})
final_predictions_df['parcelid'] = final_predictions_df['parcelid'].astype(int)
print(f"Final predictions generated for {len(final_predictions_df)} properties.")


final_predictions_df


for col in sample_submission.columns[1:]:
    sample_submission[col] = final_predictions_df['logerror_pred']


sample_submission.to_csv("zillow_submission.csv", index=False)
print("Submission file saved as zillow_submission.csv")




