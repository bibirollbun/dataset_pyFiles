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


train_2016 = pd.read_csv("/kaggle/input/zillow-prize-1/train_2016_v2.csv")

train_2016


train_2017 = pd.read_csv("/kaggle/input/zillow-prize-1/train_2017.csv")

train_2017





properties_2016 = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2016.csv")

properties_2016


properties_2017 = pd.read_csv("/kaggle/input/zillow-prize-1/properties_2017.csv")

properties_2017


sample_sub = pd.read_csv("/kaggle/input/zillow-prize-1/sample_submission.csv")

sample_sub


print("Properties 2016 shape:", properties_2016.shape)
print("Properties 2017 shape:", properties_2017.shape)
print("Train 2016 shape:", train_2016.shape)
print("Train 2017 shape:", train_2017.shape)
print("Sample Submission shape:", sample_sub.shape)


z_dict = pd.read_excel("/kaggle/input/zillow-prize-1/zillow_data_dictionary.xlsx")

z_dict


train_df = pd.concat([
    train_2016.merge(properties_2016, on = "parcelid", how = "left"), 
    train_2017.merge(properties_2017, on = "parcelid", how = "left")
])
train_df['transactiondate'] = pd.to_datetime(train_df['transactiondate'], errors='coerce')

train_df


train_df.describe()


train_df.info()


train_df.isna().sum()


import matplotlib.pyplot as plt
import seaborn as sns


sns.histplot(train_df["logerror"], bins=100, kde=True)
plt.title("Distribution of Logerror")
plt.show()


plt.figure(figsize=(8,6))
sns.boxplot(x=train_df['logerror'])
plt.title("Logerror Outliers")
plt.show()



# for decing whether to use IQR
train_df['logerror'].describe()



missing = train_df.isnull().mean().sort_values(ascending=False)

plt.figure(figsize=(10,12))
sns.barplot(x=missing.head(30), y=missing.head(30).index)
plt.title("Top 30 Features by % Missing")
plt.show()


numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()


plt.figure(figsize=(14,10))

corr = train_df[numeric_cols].corr()['logerror'].sort_values(ascending=False)
sns.heatmap(corr.head(20).to_frame(), cmap="coolwarm", annot=True)
plt.title("Top Correlations with Logerror")
plt.show()



plt.figure(figsize=(10,8))
plt.scatter(train_df['longitude'], train_df['latitude'], 
            c=train_df['logerror'], s=1, cmap='coolwarm')
plt.colorbar(label='Logerror')
plt.title("Logerror Heatmap by Latitude/Longitude")
plt.show()



train_df.hist(figsize=(20, 20), bins=30)
plt.suptitle(f"Numerical Feature Distributions", fontsize=14)
plt.show()


continuous_cols = [
    'taxvaluedollarcnt',
    'calculatedfinishedsquarefeet',
    'lotsizesquarefeet',
    'structuretaxvaluedollarcnt',
    'landtaxvaluedollarcnt'
]

for col in continuous_cols:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=train_df[col], y=train_df['logerror'], s=5, alpha=0.3)
    plt.title(f"{col} vs Logerror")
    plt.xlabel(col)
    plt.ylabel("logerror")
    plt.show()



#data is normal but not standard, Z_score won't work
#IQR also won't work as seen in train_df['logerror'].describe()

train_df = train_df[ train_df['logerror'].abs() < 0.38]




from sklearn.preprocessing import LabelEncoder

def preprocess(df, is_train=True):
    
    #Removing columns with high proportion NaNs (>90%)
    missing_ratio = df.isnull().mean()
    to_drop = missing_ratio[missing_ratio > 0.90].index
    df = df.drop(columns=to_drop, errors='ignore')

    #Seasonality
    if 'transactiondate' in df.columns:
        df['transactiondate'] = pd.to_datetime(df['transactiondate'])
        df['transaction_month'] = df['transactiondate'].dt.month
        df['transaction_quarter'] = df['transactiondate'].dt.quarter

    #age of home
    if 'yearbuilt' in df.columns:
        df['home_age'] = 2017 - df['yearbuilt']
        df['is_historic'] = (df['home_age'] > 50).astype(int) 

    # #tax Ratios
    # if 'taxvaluedollarcnt' in df.columns:
    #     # Tax vs Land
    #     if 'landtaxvaluedollarcnt' in df.columns:
    #         df['tax_ratio'] = df['taxvaluedollarcnt'] / (df['landtaxvaluedollarcnt'] + 1)
    #     # Tax vs Structure
    #     if 'structuretaxvaluedollarcnt' in df.columns:
    #         df['structure_tax_ratio'] = df['structuretaxvaluedollarcnt'] / (df['landtaxvaluedollarcnt'] + 1)
    #     # Tax per SqFt (Value density)
    #     if 'calculatedfinishedsquarefeet' in df.columns:
    #         df['value_per_sqft'] = df['taxvaluedollarcnt'] / (df['calculatedfinishedsquarefeet'] + 1)
    #     # Tax Rate Proxy
    #     if 'taxamount' in df.columns:
    #         df['tax_rate'] = df['taxamount'] / (df['taxvaluedollarcnt'] + 1)

    # # Space Ratios
    # if 'calculatedfinishedsquarefeet' in df.columns:
    #     # Living area vs Lot size
    #     if 'lotsizesquarefeet' in df.columns:
    #         df['living_area_ratio'] = df['calculatedfinishedsquarefeet'] / (df['lotsizesquarefeet'] + 1)
    #         # Extra space (yard size)
    #         df['yard_size'] = df['lotsizesquarefeet'] - df['calculatedfinishedsquarefeet']
    #     # Avg Room Size
    #     if 'roomcnt' in df.columns:
    #         # Avoid division by zero
    #         df['avg_room_size'] = df['calculatedfinishedsquarefeet'] / (df['roomcnt'].clip(lower=1))

    # # Bed/Bath Ratios
    # if 'bedroomcnt' in df.columns and 'bathroomcnt' in df.columns:
    #     df['bath_bed_ratio'] = df['bathroomcnt'] / (df['bedroomcnt'].clip(lower=1))
    #     df['total_rooms'] = df['bathroomcnt'] + df['bedroomcnt']

    # Compare this house to the average house in its zip code
    if 'regionidzip' in df.columns and 'calculatedfinishedsquarefeet' in df.columns:
        # Calculate median sqft per zip code
        zip_sqft = df.groupby('regionidzip')['calculatedfinishedsquarefeet'].transform('median')
        # How much bigger/smaller is this house compared to neighbors?
        df['sqft_rel_to_zip'] = df['calculatedfinishedsquarefeet'] / (zip_sqft + 1)

    if 'regionidzip' in df.columns and 'taxvaluedollarcnt' in df.columns:
        # Calculate median tax value per zip code
        zip_val = df.groupby('regionidzip')['taxvaluedollarcnt'].transform('median')
        df['val_rel_to_zip'] = df['taxvaluedollarcnt'] / (zip_val + 1)

    
    # lat-long bining as seen in EDA

    if 'latitude' in df.columns:
        df['lat_bin'] = (df['latitude'] // 2500).astype('float32')

    if 'longitude' in df.columns:
        df['long_bin'] = (df['longitude'] // 2500).astype('float32')


    if 'calculatedfinishedsquarefeet' in df.columns:
        df['poly_sqft'] = df['calculatedfinishedsquarefeet'] ** 2
    if 'taxvaluedollarcnt' in df.columns:
        df['poly_tax_val'] = df['taxvaluedollarcnt'] ** 2

    #Encoding
    
    df['missing_count'] = df.isnull().sum(axis=1)

    # categorical features
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'transactiondate']

    #Numerical features
    num_cols = [col for col in df.columns if col not in cat_cols and col != 'transactiondate']
    for col in num_cols:
        #handling numerical missing values
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    #Encoding Categoricals
    for col in cat_cols:
        df[col] = df[col].fillna('Missing')
        lbl = LabelEncoder()
        df[col] = lbl.fit_transform(df[col].astype(str))

    return df

train_df = preprocess(train_df, is_train=True)


train_df.columns


# #engineered features EDA

# engineered = [
#     'home_age', 
#     'tax_ratio', 
#     'living_area_ratio',
#     'missing_count'
# ]

# for col in engineered:
#     plt.figure(figsize=(8,6))
#     sns.scatterplot(x=train_df[col], y=train_df['logerror'])
#     plt.title(f"Logerror vs {col}")
#     plt.show()



numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# removing 'logerror' column 
if 'logerror' in numeric_cols:
    numeric_cols.remove('logerror')

X = train_df[numeric_cols].copy()


#filling numeric missing with median
for col in X.columns:
    X[col] = X[col].fillna(X[col].median())


y = train_df['logerror'].values


train_df.columns


X.columns


from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler

#Scaling only on train data to avoid leakage

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)



dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

lgbm = lgb.LGBMRegressor(objective='mae', metric='mae', n_jobs=-1, random_state=42)


from scipy.stats import uniform, randint

param_dist = {
    'n_estimators': randint(1000, 4000),
    'learning_rate': uniform(0.005, 0.03),
    'num_leaves': randint(20, 100),
    'max_depth': randint(5, 15),
    'min_child_samples': randint(50, 500),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': uniform(0.01, 0.5),
    'reg_lambda': uniform(0.01, 0.5)
}


from sklearn.model_selection import RandomizedSearchCV

lgbm_search = RandomizedSearchCV(
    estimator=lgbm, 
    param_distributions=param_dist,
    n_iter=10,                 
    scoring='neg_mean_absolute_error',
    cv=3,                     
    verbose=2,
    random_state=42,
    return_train_score=True
)


print("Starting Randomized Hyperparameter Search...")
lgbm_search.fit(X_train, y_train)

best_params = lgbm_search.best_params_
best_score = -lgbm_search.best_score_

print("\ntuned parameters : ", best_params)
print("best MAE : ", best_score)


params = {
    'objective': 'regression',
    'metric': 'mae',
    'max_bin': 255,   # Standard high value for complex tabular data
    'seed': 42,
    'verbosity': -1
}
params.update(best_params)


model_lgb = lgb.train(
    params,
    dtrain,
    num_boost_round=4000,
    valid_sets=[dtrain, dval],
    callbacks=[
        lgb.early_stopping(stopping_rounds=200),
        lgb.log_evaluation(period=100)
    ]
)


preds = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)


from sklearn.metrics import mean_absolute_error

model_mae = mean_absolute_error(y_val, preds)
print("Validation MAE =", model_mae)



import matplotlib.pyplot as plt
import seaborn as sns

# Assuming y_val and y_pred are available from your prediction step
plt.figure(figsize=(10, 6))

# Plot the distribution of actual values (test set)
sns.kdeplot(y_val, color='blue', label='Actual Values (Test)', fill=True, alpha=0.3, linewidth=2)

# Plot the distribution of predicted values
sns.kdeplot(preds, color='orange', label='Predicted Values', fill=True, alpha=0.3, linewidth=2)

plt.title('Distribution Comparison: Actual vs Predicted Log Error', fontsize=16)
plt.xlabel('Log Error', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=11)
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming y_val and y_pred are available
plt.figure(figsize=(8, 8))

# Scatter plot of actual vs. predicted values
plt.plot(y_val, alpha=0.4, color='blue')
plt.plot(preds, alpha=0.4, color='orange')

# Determine plot limits for the perfect prediction line
min_val = min(np.min(y_val), np.min(preds))
max_val = max(np.max(y_val), np.max(preds))
lims = [min_val, max_val]

# Add the "Perfect Prediction" line
plt.plot(lims, lims, 'r--', lw=3, label='Perfect Prediction')

plt.title('Actual vs Predicted Log Error', fontsize=16)
plt.xlabel('Actual Log Error (y_val)', fontsize=12)
plt.ylabel('Predicted Log Error (y_pred)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()



test_df = properties_2017.copy()          
test_df = preprocess(test_df, is_train=False)

numeric_cols = test_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

X_test = test_df[numeric_cols].copy()
for col in X_test.columns:
    if X_test[col].isnull().any():
        X_test[col] = X_test[col].fillna(X_test[col].median())

# X_test = scaler.transform(X_test)

# Reorder to match training
X_test = X_test[X.columns]


type(X_test)


set(X.columns) - set(X_test.columns)



test_df.columns


preds = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)

#submission file 
submission = sample_sub.copy()
for c in submission.columns:
    if c != 'ParcelId':
        submission[c] = preds


submission.to_csv('submission.csv', index=False)
print('Submission file successfully generated')





submission.head()


import os
os.listdir('/kaggle/working')



submission


