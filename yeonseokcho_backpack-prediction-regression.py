import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
sample_submission.head(2)


test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
print(test.shape)
test.head(2)


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
print(train.shape)
train.head(2)


training_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
print(training_extra.shape)
training_extra.tail(2)
# id 500000 ~ 4194317


# missing_info
test_NaN_num = test.isnull().sum()
test_NaN_percentage = test_NaN_num / len(test)

train_NaN_num = train.isnull().sum()
train_NaN_percentage = train_NaN_num / len(train)

training_extra_NaN_num = training_extra.isnull().sum()
training_extra_NaN_percentage = training_extra_NaN_num / len(training_extra)

missing_info = pd.concat([test_NaN_percentage, train_NaN_percentage, 
                          training_extra_NaN_percentage], axis=1, 
                         keys=['test_NaN_percentage', 'train_NaN_percentage', 
                               'training_extra_NaN_percentage'])
missing_info


# target value, 'Price'
plt.figure(figsize=(10,3))

plt.subplot(2,1,1)
plt.title('Price of Backpack in train')
sns.histplot(train['Price'], kde=True, bins=150)

plt.subplot(2,1,2)
plt.title('Price of Backpack in training_extra')
sns.histplot(training_extra['Price'], kde=True, bins=150)

plt.tight_layout()
plt.show()


training_extra = training_extra[training_extra['Price'] != 150].reset_index(drop=True)


# target value, 'Price'
plt.figure(figsize=(10,3))

plt.subplot(2,1,1)
plt.title('Price of Backpack in train')
sns.histplot(train['Price'], kde=True, bins=150)

# target value, 'Price'
plt.subplot(2,1,2)
plt.title('Price of Backpack in training_extra')
sns.histplot(training_extra['Price'], kde=True, bins=150)

plt.tight_layout()
plt.show()


# combine to train_df
train_df = pd.concat([train, training_extra], axis=0)
print(train_df.shape)

plt.figure(figsize=(10, 1))
plt.title('Price of Backpack in combined train_df')
sns.histplot(train_df['Price'], kde=True, bins=150)
plt.show()


# Fill NaN with median and mode[0]
train_wo_price = train_df.drop(['Price'], axis=1)

numeric_columns = train_wo_price.select_dtypes(include=['int', 'float']).columns
object_columns = train_wo_price.select_dtypes(include=['object']).columns

train_df[numeric_columns] = train_df[numeric_columns].fillna(train_wo_price[numeric_columns].median())
test[numeric_columns] = test[numeric_columns].fillna(test[numeric_columns].median())

train_df[object_columns] = train_df[object_columns].fillna(train_df[object_columns].mode().iloc[0])
test[object_columns] = test[object_columns].fillna(test[object_columns].mode().iloc[0])

train_df.isna().sum().sum(), test.isna().sum().sum()


train_df.describe()


compartments_order = train_df.groupby('Compartments')['Price'].median().sort_values().index.tolist()

train_df['Weight Capacity Bins'] = pd.cut(train_df['Weight Capacity (kg)'], 
                                       bins=[5,10,15,20,25,30], 
                                       labels=['5-10','10-15','15-20','20-25','25-30'])
weight_order = train_df.groupby('Weight Capacity Bins')['Price'].median().sort_values().index.tolist()

plt.figure(figsize=(10,3))

plt.subplot(1,2,1)
sns.boxplot(x='Compartments', y='Price', data=train_df, order=compartments_order)
plt.title('Compartments vs Price (Median Ordered)', fontsize=10)
plt.xlabel('Compartments', fontsize=9)
plt.ylabel('Price (USD)', fontsize=9)

plt.subplot(1,2,2)
sns.boxplot(x='Weight Capacity Bins', y='Price', data=train_df, order=weight_order)
plt.title('Weight Capacity vs Price (Median Ordered)', fontsize=10)
plt.xlabel('Weight Capacity (kg)', fontsize=9)
plt.ylabel('Price (USD)', fontsize=9)

plt.tight_layout()
plt.show()



Brand_order = train_df.groupby('Brand')['Price'].median().sort_values().index.tolist()
Material_order = train_df.groupby('Material')['Price'].median().sort_values().index.tolist()

plt.figure(figsize=(10,3))

plt.subplot(1,2,1)
sns.boxplot(x='Brand', y='Price', data=train_df, order=Brand_order)
plt.title('Brand vs Price (Median Ordered)', fontsize=10)
plt.xlabel('Brand', fontsize=9)
plt.ylabel('Price (USD)', fontsize=9)
plt.xticks(rotation=15, ha='center')

plt.subplot(1,2,2)
sns.boxplot(x='Material', y='Price', data=train_df, order=Material_order)
plt.title('Material vs Price (Median Ordered)', fontsize=10)
plt.xlabel('Material', fontsize=9)
plt.ylabel('Price (USD)', fontsize=9)

plt.tight_layout()
plt.show()


Size_order = train_df.groupby('Size')['Price'].median().sort_values().index.tolist()
Laptop_Compartment_order = train_df.groupby('Laptop Compartment')['Price'].median().sort_values().index.tolist()
Waterproof_order = train_df.groupby('Waterproof')['Price'].median().sort_values().index.tolist()

plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
sns.boxplot(x='Size', y='Price', data=train_df, order=Size_order)
plt.title('Size vs Price (Median Ordered)', fontsize=10)
plt.xlabel('Size', fontsize=9)
plt.ylabel('Price (USD)', fontsize=9)

plt.subplot(1, 3, 2)
sns.boxplot(x='Laptop Compartment', y='Price', data=train_df, order=Laptop_Compartment_order)
plt.title('Laptop Compartment vs Price (Median Ordered)', fontsize=10)
plt.xlabel('Laptop Compartment', fontsize=9)
plt.ylabel('Price (USD)', fontsize=9)

plt.subplot(1, 3, 3)
sns.boxplot(x='Waterproof', y='Price', data=train_df, order=Waterproof_order)
plt.title('Waterproof vs Price (Median Ordered)', fontsize=10)
plt.xlabel('Waterproof', fontsize=9)
plt.ylabel('Price (USD)', fontsize=9)

plt.tight_layout()
plt.show()


Style_order = train_df.groupby('Style')['Price'].median().sort_values().index.tolist()
Color_order = train_df.groupby('Color')['Price'].median().sort_values().index.tolist()

plt.figure(figsize=(10,3))

plt.subplot(1,2,1)
sns.boxplot(x='Style', y='Price', data=train_df, order=Style_order)
plt.title('Style vs Price (Median Ordered)', fontsize=10)
plt.xlabel('Style', fontsize=9)
plt.ylabel('Price (USD)', fontsize=9)

plt.subplot(1,2,2)
sns.boxplot(x='Color', y='Price', data=train_df, order=Color_order)
plt.title('Color vs Price (Median Ordered)', fontsize=10)
plt.xlabel('Color', fontsize=9)
plt.ylabel('Price (USD)', fontsize=9)

plt.tight_layout()
plt.show()


target = train_df['Price']
features = train_df.drop(['id', 'Price'], axis=1)

test['Weight Capacity Bins'] = pd.cut(test['Weight Capacity (kg)'], bins=[5,10,15,20,25,30], 
                                      labels=['5-10','10-15','15-20','20-25','25-30'])
test_features = test.drop(['id'], axis=1)
target.shape, features.shape, test_features.shape


# categoric variables in features
features_cat = features.select_dtypes(include=['object'])
test_features_cat = test_features.select_dtypes(include=['object'])
print(features_cat.shape, test_features_cat.shape)
features_cat.head(2)


features_cat_encoded = features_cat.copy()

features_cat_encoded['Brand'] = features_cat_encoded['Brand'].replace({
    'Adidas': 1, 'Puma': 2, 'Nike': 3, 'Jansport': 4, 'Under Armour': 5})

features_cat_encoded['Material'] = features_cat_encoded['Material'].replace({
    'Leather': 1, 'Nylon': 2, 'Canvas': 3, 'Polyester': 4})

features_cat_encoded['Size'] = features_cat_encoded['Size'].replace({
    'Medium': 1, 'Small': 2, 'Large': 3})

features_cat_encoded['Laptop Compartment'] = features_cat_encoded['Laptop Compartment'].replace({
    'No': 1, 'Yes': 2})

features_cat_encoded['Waterproof'] = features_cat_encoded['Waterproof'].replace({
    'Yes': 1, 'No': 2})

features_cat_encoded['Style'] = features_cat_encoded['Style'].replace({
    'Backpack': 1, 'Messenger': 2, 'Tote': 3})

features_cat_encoded['Color'] = features_cat_encoded['Color'].replace({
    'Black': 1, 'Gray': 2, 'Red': 3, 'Pink': 4, 'Blue': 5, 'Green': 6})

print(features_cat_encoded.shape)
features_cat_encoded.head(2)


test_features_cat_encoded = test_features_cat.copy()

test_features_cat_encoded['Brand'] = test_features_cat_encoded['Brand'].replace({
    'Adidas': 1, 'Puma': 2, 'Nike': 3, 'Jansport': 4, 'Under Armour': 5})

test_features_cat_encoded['Material'] = test_features_cat_encoded['Material'].replace({
    'Leather': 1, 'Nylon': 2, 'Canvas': 3, 'Polyester': 4})

test_features_cat_encoded['Size'] = test_features_cat_encoded['Size'].replace({
    'Medium': 1, 'Small': 2, 'Large': 3})

test_features_cat_encoded['Laptop Compartment'] = test_features_cat_encoded['Laptop Compartment'].replace({
    'No': 1, 'Yes': 2})

test_features_cat_encoded['Waterproof'] = test_features_cat_encoded['Waterproof'].replace({
    'Yes': 1, 'No': 2})

test_features_cat_encoded['Style'] = test_features_cat_encoded['Style'].replace({
    'Backpack': 1, 'Messenger': 2, 'Tote': 3})

test_features_cat_encoded['Color'] = test_features_cat_encoded['Color'].replace({
    'Black': 1, 'Gray': 2, 'Red': 3, 'Pink': 4, 'Blue': 5, 'Green': 6})

print(test_features_cat_encoded.shape)
test_features_cat_encoded.head()


# numeric variables in features
features_num = features.select_dtypes(include=['int', 'float'])
test_features_num = test_features.select_dtypes(include=['int', 'float'])
print(features_num.shape, test_features_num.shape)
features_num.head(2)


# combine 
features_encoded_num = pd.concat([features_cat_encoded, features_num], axis=1)
test_features_encoded_num = pd.concat([test_features_cat_encoded, test_features_num], axis=1)
print(features_encoded_num.shape, test_features_encoded_num.shape)
features_encoded_num.head(2)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(features_encoded_num[features_encoded_num.columns])

features_tf = pd.DataFrame(scaler.transform(features_encoded_num[features_encoded_num.columns]), 
                        columns=features_encoded_num.columns, index=features_encoded_num.index)

test_features_tf = pd.DataFrame(scaler.transform(test_features_encoded_num[test_features_encoded_num.columns]), 
                       columns=test_features_encoded_num.columns, index=test_features_encoded_num.index)

print(features_tf.shape, test_features_tf.shape)
features_tf.head(2)


temp_for_cor = pd.concat([target, features_tf], axis=1)
print(temp_for_cor.shape)
temp_for_cor.head(2)


corrmat = temp_for_cor.corr()
corrmat["Price"].sort_values(ascending=False)


features_tf.columns = features_tf.columns.str.replace(r'\s+', '_', regex=True)
test_features_tf.columns = test_features_tf.columns.str.replace(r'\s+', '_', regex=True)

features_tf = features_tf.astype('float32')
test_features_tf = test_features_tf.astype('float32')
target = target.astype('float32')

features_tf.info()


# split into train and test data
from sklearn.model_selection import (train_test_split, StratifiedKFold)
X_train, X_val, y_train, y_val = train_test_split(features_tf, target, test_size=0.2, random_state=42)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# XGBoost model
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import time

start_time = time.time()
xgb_model = XGBRegressor(random_state=42, n_jobs=-1, 
                         learning_rate=0.25, gamma=0.3, colsample_bytree=0.6, 
                         max_depth=4,
                        )  
xgb_model.fit(X_train, y_train)

train_xgb_pred = xgb_model.predict(X_train)
val_xgb_pred = xgb_model.predict(X_val)

end_time = time.time()
xgb_time = end_time - start_time

train_xgb_mse = mean_squared_error(y_train, train_xgb_pred)
val_xgb_mse = mean_squared_error(y_val, val_xgb_pred)

train_xgb_rmse = np.sqrt(train_xgb_mse)
val_xgb_rmse = np.sqrt(val_xgb_mse)

print('XGBoost Results:')
print(f"train_MSE: {train_xgb_mse:.6f}")
print(f"val_MSE: {val_xgb_mse:.6f}")
print(f"train_RMSE: {train_xgb_rmse:.6f}")
print(f"val_RMSE: {val_xgb_rmse:.6f}")
print(f"Time: {xgb_time:.6f}")


# LightGBM model
from lightgbm import LGBMRegressor

start_time = time.time()
lgbm_model = LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1, 
                           n_estimators=500, learning_rate=0.05, max_depth=9, 
                           colsample_bytree=0.8, subsample=0.1, num_leaves=15,
                          ) 
lgbm_model.fit(X_train, y_train)

train_lgbm_pred = lgbm_model.predict(X_train)
val_lgbm_pred = lgbm_model.predict(X_val)

end_time = time.time()
lgbm_time = end_time - start_time

train_lgbm_mse = mean_squared_error(y_train, train_lgbm_pred)
val_lgbm_mse = mean_squared_error(y_val, val_lgbm_pred)

train_lgbm_rmse = np.sqrt(train_lgbm_mse)
val_lgbm_rmse = np.sqrt(val_lgbm_mse)

print('LightGBM Results:')
print(f"train_MSE: {train_lgbm_mse:.6f}")
print(f"val_MSE: {val_lgbm_mse:.6f}")
print(f"train_RMSE: {train_lgbm_rmse:.6f}")
print(f"val_RMSE: {val_lgbm_rmse:.6f}")
print(f"Time: {lgbm_time:.6f}")


# CatBoost model
from catboost import CatBoostRegressor

start_time = time.time()
cat_model = CatBoostRegressor(random_state=42, verbose=0, 
                              n_estimators=100, depth=3, learning_rate=0.3, l2_leaf_reg=3, 
                              random_strength=0
                             )
cat_model.fit(X_train, y_train)

train_cat_pred = cat_model.predict(X_train)
val_cat_pred = cat_model.predict(X_val)

end_time = time.time()
cat_time = end_time - start_time

train_cat_mse = mean_squared_error(y_train, train_cat_pred)
val_cat_mse = mean_squared_error(y_val, val_cat_pred)

train_cat_rmse = np.sqrt(train_cat_mse)
val_cat_rmse = np.sqrt(val_cat_mse)

print('CatBoost Results:')
print(f"train_MSE: {train_cat_mse:.6f}")
print(f"val_MSE: {val_cat_mse:.6f}")
print(f"train_RMSE: {train_cat_rmse:.6f}")
print(f"val_RMSE: {val_cat_rmse:.6f}")
print(f"Time: {cat_time:.6f}")


# Ensemble Model (VotingRegressor)
from sklearn.ensemble import VotingRegressor

ensemble_models = [
    ('XGB', xgb_model),  
    ('LGBM', lgbm_model), 
    ('CatBoost', cat_model) 
]

ensemble_model = VotingRegressor(estimators=ensemble_models)

start_time = time.time()
ensemble_model.fit(X_train, y_train)

train_ensemble_pred = ensemble_model.predict(X_train)
val_ensemble_pred = ensemble_model.predict(X_val)

end_time = time.time()
ensemble_time = end_time - start_time

train_ensemble_mse = mean_squared_error(y_train, train_ensemble_pred)
val_ensemble_mse = mean_squared_error(y_val, val_ensemble_pred)

train_ensemble_rmse = np.sqrt(train_ensemble_mse)
val_ensemble_rmse = np.sqrt(val_ensemble_mse)

print('Ensemble Model (VotingRegressor)')
print(f"train_MSE: {train_ensemble_mse:.6f}")
print(f"val_MSE: {val_ensemble_mse:.6f}")
print(f"train_RMSE: {train_ensemble_rmse:.6f}")
print(f"val_RMSE: {val_ensemble_rmse:.6f}")
print(f"Time: {ensemble_time:.6f}")


# Ensemble Model (StackingRegressor)
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

base_models = [
    ('XGB', xgb_model),  
    ('LGBM', lgbm_model), 
    ('CatBoost', cat_model)
]

meta_model = LinearRegression()
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)

start_time = time.time()
stacking_model.fit(X_train, y_train)

train_stacking_pred = stacking_model.predict(X_train)
val_stacking_pred = stacking_model.predict(X_val)

end_time = time.time()
stacking_time = end_time - start_time

train_stacking_mse = mean_squared_error(y_train, train_stacking_pred)
val_stacking_mse = mean_squared_error(y_val, val_stacking_pred)

train_stacking_rmse = np.sqrt(train_stacking_mse)
val_stacking_rmse = np.sqrt(val_stacking_mse)

print('Ensemble Model (StackingRegressor)')
print(f"train_MSE: {train_stacking_mse:.6f}")
print(f"val_MSE: {val_stacking_mse:.6f}")
print(f"train_RMSE: {train_stacking_rmse:.6f}")
print(f"val_RMSE: {val_stacking_rmse:.6f}")
print(f"Time: {stacking_time:.6f}")


# stacking_model
test_stacking_pred = stacking_model.predict(test_features_tf)
test_stacking_pred = test_stacking_pred
test_stacking_pred


submission = pd.DataFrame({'id': test.id, 'Price': test_stacking_pred})
print(submission.shape)
submission.head()


plt.figure(figsize=(10, 1))
plt.title('Price of Backpack in submission')
sns.histplot(submission['Price'], kde=True, bins=150)
plt.show()
# why? overfitting..


submission.to_csv('submission.csv', index=False)

