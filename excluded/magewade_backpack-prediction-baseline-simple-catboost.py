import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col=0)
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv", index_col=0)
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv", index_col=0)


train.head()


train.shape


train_extra.shape


print(f"Number of duplicates in train_extra: {train_extra.duplicated().sum()}")
print(f"Number of duplicates in train: {train.duplicated().sum()}")


n = 12 # coefficient 
train_extra_sampled = train_extra.sample(n=train.shape[0] * n, random_state=42) 
train = pd.concat([train, train_extra_sampled], ignore_index=True)
print(f"Shape of the combined dataset: {train.shape}")


categorical = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']


for cat in categorical:
    print(f'Categories in {cat}:')
    print(train[cat].value_counts())
    print(f'Nan: {train[cat].isna().sum()}')
    print('---')


for cat in categorical:
    train[cat] = train[cat].fillna('no_data')
    test[cat] = test[cat].fillna('no_data')
    train[cat] = train[cat].astype('category')
    test[cat] = test[cat].astype('category')


train.head()


numerical = ['Compartments', 'Weight Capacity (kg)']


for col in numerical:
    print(f"Range for {col}:")
    print(f"Type: {train[col].dtype}")
    print(f"Min: {train[col].min()}")
    print(f"Max: {train[col].max()}")
    print(f'Nans: {train[col].isna().sum()}')
    print("---")


train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].median())


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical, 1):
    plt.subplot(1, 2, i)
    sns.histplot(train[col], kde=False, bins=30, color='#D67F54')
    plt.title(f'Histogram of {col}')
    
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical, 1):
    plt.subplot(1, 2, i)
    sns.boxplot(x=train[col], color='#A4C392')
    plt.title(f'Boxplot of {col}')
    
plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
import lightgbm as lgb

X = train.drop('Price', axis=1) 
y = train['Price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model_cb = CatBoostRegressor(depth=10, learning_rate=0.05, loss_function='RMSE', cat_features=categorical, verbose=100)
model_cb.fit(X_train, y_train, eval_set=(X_val, y_val), plot=False)


X_train["cb_pred"] = model_cb.predict(X_train)
X_val["cb_pred"] = model_cb.predict(X_val)
test["cb_pred"] = model_cb.predict(test)  


for col in categorical:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_val[col] = le.transform(X_val[col])
    test[col] = le.transform(test[col])

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(test) 


xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 6,
    'random_state': 42
}

model = xgb.train(xgb_params, dtrain, num_boost_round=500, evals=[(dvalid, 'valid')], early_stopping_rounds=50, verbose_eval=50)


y_pred = model.predict(dtest)


submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
submission['Price'] = y_pred


submission.to_csv('/kaggle/working/submission.csv', index=False)

