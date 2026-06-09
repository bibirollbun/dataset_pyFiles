import os as os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


input_path = '/kaggle/input/competitive-data-science-predict-future-sales'
sales_train = pd.read_csv(os.path.join(input_path,'sales_train.csv'))
test = pd.read_csv(os.path.join(input_path,'test.csv'))
items = pd.read_csv(os.path.join(input_path,'items.csv'))
item_categories = pd.read_csv(os.path.join(input_path,'item_categories.csv'))
shops = pd.read_csv(os.path.join(input_path,'shops.csv'))
sample_submission = pd.read_csv(os.path.join(input_path,'sample_submission.csv'))

print("Sales Train shape: ", sales_train.shape)
print("test shape: ", test.shape)
print("items shape: ", items.shape)
print("item_categories shape: ", item_categories.shape)
print("shops shape: ", shops.shape)



print("Sales train info: ")
sales_train.info()
print("Sales train Head: ")
print(sales_train.head())
print("Sales train Description: ")
print(sales_train.describe())

print("\ntest info: ")
test.info()
print("test Head: ")
print(test.head())
print("test Description: ")
print(test.describe())

print("\nitems info: ")
items.info()
print("items Head: ")
print(items.head())
print("items Description: ")
print(items.describe())

print("\nitem_categories info: ")
item_categories.info()
print("item_categories Head: ")
print(item_categories.head())
print("item_categories Description: ")
print(item_categories.describe())

print("\nshops info: ")
shops.info()
print("shops Head: ")
print(shops.head())
print("shops Description: ")
print(shops.describe())


sales_train["date"] = pd.to_datetime(sales_train["date"], format='%d.%m.%Y')
print("Sales train info after date conversion: ")
sales_train.info()
print(sales_train.head())


plt.figure(figsize=(10,4))
sns.boxplot(x=sales_train['item_cnt_day'])
plt.title('Boxplot of item_cnt_day')
plt.show()

plt.figure(figsize=(10,4))
sns.boxplot(x=sales_train['item_price'])
plt.title('Boxplot of item_price')
plt.show()


print("\nUninque shop names: ", shops['shop_name'].nunique())
print("Total shops: ", len(shops))
print(shops['shop_name'].value_counts().head(10))


monthly_total_sales = sales_train.groupby('date_block_num')['item_cnt_day'].sum()

plt.figure(figsize=(12,6))
plt.plot(monthly_total_sales.index, monthly_total_sales.values)
plt.xlabel('Date Block Num (Month)')
plt.ylabel('Total Items Sold')
plt.title('Total Monthly Sales Trend')
plt.grid(True)
plt.show()


print("\nMissing values in sales_train:", sales_train.isnull().sum().sum())
print("Missing values in test:", test.isnull().sum().sum())
print("Missing values in items:", items.isnull().sum().sum())
print("Missing values in items:", item_categories.isnull().sum().sum())
print("Missing values in items:", shops.isnull().sum().sum())



monthly_sales = sales_train.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False).agg({
    'item_cnt_day':'sum'
}).rename(columns={'item_cnt_day':'item_cnt_month'})

print(monthly_sales.head())
print(monthly_sales.describe())
monthly_sales.info()


monthly_sales['item_cnt_month'] = monthly_sales['item_cnt_month'].clip(0,20) 

print(monthly_sales.head())
print(monthly_sales.describe())
monthly_sales.info()


from itertools import product
import time 

start_time = time.time()

grid = []

for block_num in sales_train['date_block_num'].unique():
    unique_shops = sales_train.loc[sales_train['date_block_num'] == block_num, 'shop_id'].unique()
    unique_items = sales_train.loc[sales_train['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product([block_num], unique_shops, unique_items)), dtype='int32'))

grid = pd.DataFrame(np.vstack(grid), columns=['date_block_num', 'shop_id', 'item_id'], dtype=np.int32)

train_grid = pd.merge(grid, monthly_sales, on=['date_block_num', 'shop_id', 'item_id'], how='left')

train_grid['item_cnt_month'] = train_grid['item_cnt_month'].fillna(0).astype(np.float32)

# Clip again just in case

train_grid['item_cnt_month'] = train_grid['item_cnt_month'].clip(0,20)

print("Full training grid head:")
print(train_grid.head())
print("Grid shape: ", train_grid.shape)
print(f"Grid creating took: {time.time() - start_time:.2f} seconds")
train_grid['item_cnt_month'].describe()


train_grid.sort_values(['date_block_num','shop_id', 'item_id'], inplace=True)

train_grid['lag_1_month'] = train_grid.groupby(['shop_id','item_id'])['item_cnt_month'].shift(1).fillna(0).astype(np.float32)
train_grid['lag_2_month'] = train_grid.groupby(['shop_id','item_id'])['item_cnt_month'].shift(2).fillna(0).astype(np.float32)
train_grid['lag_3_month'] = train_grid.groupby(['shop_id','item_id'])['item_cnt_month'].shift(3).fillna(0).astype(np.float32)
train_grid['lag_6_month'] = train_grid.groupby(['shop_id','item_id'])['item_cnt_month'].shift(6).fillna(0).astype(np.float32)
train_grid['lag_12_month'] = train_grid.groupby(['shop_id','item_id'])['item_cnt_month'].shift(12).fillna(0).astype(np.float32)
print("Grid with lag features head:")
print(train_grid[['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'lag_1_month', 'lag_2_month', 'lag_3_month', 'lag_6_month', 'lag_12_month']].head())


train_grid['month'] = (train_grid['date_block_num']%12)+1

train_grid['year'] = (train_grid['date_block_num'] // 12) + 2013

print("Grid with Date Features head:")
print(train_grid[['date_block_num', 'month', 'year']].head())


mean_item_hist = train_grid.groupby(['date_block_num','item_id'])['item_cnt_month'].mean().reset_index()
mean_item_hist = mean_item_hist.rename(columns={'item_cnt_month':'avg_item_cnt_prev_month'})

mean_item_hist['date_block_num'] += 1

train_grid = pd.merge(train_grid, mean_item_hist, on=['date_block_num','item_id'], how='left')
train_grid['avg_item_cnt_prev_month'] = train_grid['avg_item_cnt_prev_month'].fillna(0).astype(np.float32)

mean_shop_hist = train_grid.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].mean().reset_index()
mean_shop_hist = mean_shop_hist.rename(columns={'item_cnt_month':'avg_shop_cnt_prev_month'})
mean_shop_hist['date_block_num'] += 1
train_grid = pd.merge(train_grid, mean_shop_hist, on=['date_block_num','shop_id'], how='left')
train_grid['avg_shop_cnt_prev_month'] = train_grid['avg_shop_cnt_prev_month'].fillna(0).astype(np.float32)

print("\nGrid with simple Mean Encoding head:")
print(train_grid[['date_block_num', 'item_id', 'shop_id', 'avg_item_cnt_prev_month', 'avg_shop_cnt_prev_month']].head())


train_grid['delta_lag1_lag2'] = train_grid['lag_1_month'] - train_grid['lag_2_month']
train_grid['delta_lag1_lag2'].describe()


train_grid['item_first_month'] = train_grid.groupby('item_id')['date_block_num'].transform('min')
train_grid['item_age_months'] = train_grid['date_block_num'] - train_grid['item_first_month']

train_grid['shop_first_month'] = train_grid.groupby('shop_id')['date_block_num'].transform('min')
train_grid['shop_age_months'] = train_grid['date_block_num'] - train_grid['shop_first_month']


test = pd.read_csv(os.path.join(input_path, 'test.csv')) 
test['date_block_num'] = 34;
print("Columns in test BEFORE merge:", test.columns)
test = pd.merge(test,
                items[['item_id', 'item_category_id']], 
                on='item_id',                           
                how='left')                             
print("Columns in test AFTER merge:", test.columns)
print(test.head())
print(test.shape)



cols_to_concat = ['date_block_num', 'shop_id', 'item_id', 'item_category_id']
target_col = ['item_cnt_month']

train_grid = pd.merge(train_grid, items[['item_id','item_category_id']], on='item_id', how='left')

print(train_grid.columns)

print("Preparing subsets for concatenation...")

train_subset = train_grid[train_grid['date_block_num'] <= 33][cols_to_concat + target_col].copy()
test_subset = test[cols_to_concat].copy()

print(f"Train subset shape: {train_subset.shape}")
print(f"Test subset shape: {test_subset.shape}")

combined_data = pd.concat([train_subset, test_subset], ignore_index=True, sort=False)
print(f"Combined data shape: {combined_data.shape}")


print("Calculating date features...")
combined_data['month'] = (combined_data['date_block_num'] % 12) + 1
combined_data['year'] = (combined_data['date_block_num'] // 12) + 2013


print("Calculating lag features...")

combined_data.sort_values(['date_block_num', 'shop_id', 'item_id'], inplace=True)
combined_data['lag_1_month'] = combined_data.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(1).fillna(0).astype(np.float32)
combined_data['lag_2_month'] = combined_data.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(2).fillna(0).astype(np.float32)
combined_data['lag_3_month'] = combined_data.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(3).fillna(0).astype(np.float32)
combined_data['lag_6_month'] = combined_data.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(6).fillna(0).astype(np.float32)
combined_data['lag_12_month'] = combined_data.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(12).fillna(0).astype(np.float32)

print("Calculating mean-coded features...")

mean_item_hist_comb = combined_data.groupby(['date_block_num', 'item_id'])['item_cnt_month'].mean().reset_index()
mean_item_hist_comb = mean_item_hist_comb.rename(columns={'item_cnt_month':'avg_item_cnt_prev_month'})
mean_item_hist_comb['date_block_num'] += 1
combined_data = pd.merge(combined_data, mean_item_hist_comb, on=['date_block_num', 'item_id'], how='left')
combined_data['avg_item_cnt_prev_month'] = combined_data['avg_item_cnt_prev_month'].fillna(0).astype(np.float32)

mean_shop_hist_comb = combined_data.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].mean().reset_index()
mean_shop_hist_comb = mean_shop_hist_comb.rename(columns={'item_cnt_month': 'avg_shop_cnt_prev_month'})
mean_shop_hist_comb['date_block_num'] += 1
combined_data = pd.merge(combined_data, mean_shop_hist_comb, on=['date_block_num', 'shop_id'], how='left')
combined_data['avg_shop_cnt_prev_month'] = combined_data['avg_shop_cnt_prev_month'].fillna(0).astype(np.float32)

print("Calculating trend features...")

combined_data['delta_lag1_lag2'] = combined_data['lag_1_month'] - combined_data['lag_2_month']

print("Calculating age features...")

combined_data['item_first_month'] = combined_data.groupby('item_id')['date_block_num'].transform('min')
combined_data['item_age_months'] = combined_data['date_block_num'] - combined_data['item_first_month']
combined_data['shop_first_month'] = combined_data.groupby('shop_id')['date_block_num'].transform('min')
combined_data['shop_age_months'] = combined_data['date_block_num'] - combined_data['shop_first_month']


print("Separating final train and test sets...")

final_test_features = combined_data[combined_data['date_block_num'] == 34].copy()
final_train_features = combined_data[combined_data['date_block_num'] <= 33].copy()

# --- Drop early months from training ---
min_train_month = 12 
print(f"Training features shape before dropping early months: {final_train_features.shape}")
final_train_features = final_train_features[final_train_features['date_block_num'] >= min_train_month].copy()
print(f"Training features shape after dropping months < {min_train_month}: {final_train_features.shape}")

final_test_features = pd.merge(final_test_features, test[['ID','shop_id','item_id']], on=['shop_id', 'item_id'], how='left')

print("Test set preparation complete.")
print(final_test_features.head())




print("Defining feature columns and splitting train/validating data...")

feature_cols = [
    'lag_1_month', 'lag_2_month', 'lag_3_month', 'lag_6_month', 'lag_12_month',
    'month', 'year',
    'avg_item_cnt_prev_month', 'avg_shop_cnt_prev_month',
    'delta_lag1_lag2',
    'item_age_months', 'shop_age_months',
    'item_category_id'
]

# --- Create Train/Validation Split ---
x_train = final_train_features[final_train_features['date_block_num'] < 33][feature_cols]
y_train = final_train_features[final_train_features['date_block_num'] < 33]['item_cnt_month']

x_valid = final_train_features[final_train_features['date_block_num'] == 33][feature_cols]
y_valid = final_train_features[final_train_features['date_block_num'] == 33]['item_cnt_month']

x_test = final_test_features[feature_cols]

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_valid shape: {x_valid.shape}, y_valid shape: {y_valid.shape}")
print(f"x_test shape: {x_test.shape}")


import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error # For evaluating base models if desired
import pandas as pd
import numpy as np
import gc

print("Step 7: Training Stacked Ensemble Model and Predicting...")

# --- Ensure x_train, y_train, x_valid, y_valid, x_test are defined from Step 6 ---
# And final_test_features for the 'ID' column for submission

# --- Level 0: Base Models ---
# We will store the validation predictions and test predictions from each base model

# Initialize dictionaries to store out-of-fold (OOF) predictions for meta-model training
# and test predictions for meta-model inference
oof_valid_predictions = pd.DataFrame()
oof_test_predictions = pd.DataFrame()

# --- Model 1: LightGBM ---
print("Training LightGBM base model...")
lgb_params = {
    'objective': 'rmse', 'metric': 'rmse', 'n_estimators': 2000,
    'learning_rate': 0.02, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'num_leaves': 31,
    'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt',
}

model_lgb = lgb.LGBMRegressor(**lgb_params)
model_lgb.fit(
    x_train, y_train,
    eval_set=[(x_train, y_train), (x_valid, y_valid)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=10)] # Using verbose=10 for less output
)

oof_valid_predictions['lgbm'] = model_lgb.predict(x_valid)
oof_test_predictions['lgbm'] = model_lgb.predict(x_test)
print("LightGBM RMSE on validation:", np.sqrt(mean_squared_error(y_valid, oof_valid_predictions['lgbm'])))

# --- Model 2: XGBoost ---
print("Training XGBoost base model...")
xgb_params = {
    'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 2000,
    'learning_rate': 0.02, 'max_depth': 6, # Typical starting depth
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'seed': 42, 'n_jobs': -1,
    'reg_alpha': 0.1, 'reg_lambda': 0.1
}

model_xgb = xgb.XGBRegressor(**xgb_params)
model_xgb.fit(
    x_train, y_train,
    eval_set=[(x_valid, y_valid)],
    early_stopping_rounds=100,
    verbose=False # Set to True or a number for more output
)

oof_valid_predictions['xgb'] = model_xgb.predict(x_valid)
oof_test_predictions['xgb'] = model_xgb.predict(x_test)
print("XGBoost RMSE on validation:", np.sqrt(mean_squared_error(y_valid, oof_valid_predictions['xgb'])))

# --- Model 3: CatBoost ---
print("Training CatBoost base model...")
cb_params = {
    'iterations': 2000, 'learning_rate': 0.02, 'depth': 6,
    'loss_function': 'RMSE', 'eval_metric': 'RMSE',
    'random_seed': 42, 'verbose': 0, # Suppress verbose output from CatBoost
    'early_stopping_rounds': 100,
    'l2_leaf_reg': 3, # Default L2 regularization
    'border_count': 64 # Default for CPU
}

# CatBoost can use item_category_id directly if it's categorical.
# Ensure it's in feature_cols and of appropriate type if you want CatBoost to treat it as categorical.
# For simplicity here, assuming all features are numerical or CatBoost handles them.
# If item_category_id is crucial as a categorical feature for CatBoost,
# you might need to pass cat_features=[index_of_item_category_id_in_feature_cols]

model_cb = cb.CatBoostRegressor(**cb_params)
model_cb.fit(
    x_train, y_train,
    eval_set=[(x_valid, y_valid)],
    # cat_features=[feature_cols.index('item_category_id')] if 'item_category_id' in feature_cols else None
)

oof_valid_predictions['cb'] = model_cb.predict(x_valid)
oof_test_predictions['cb'] = model_cb.predict(x_test)
print("CatBoost RMSE on validation:", np.sqrt(mean_squared_error(y_valid, oof_valid_predictions['cb'])))


# --- Level 1: Meta-Model (Ridge Regression) ---
print("Training Meta-Model (Ridge Regression)...")
meta_model = Ridge(alpha=1.0, random_state=42) # Alpha is the regularization strength

# The OOF predictions on the validation set are the features for the meta-model
# The actual y_valid values are the target for the meta-model
meta_model.fit(oof_valid_predictions, y_valid)

# --- Predict with Meta-Model ---
print("Predicting on test set using stacked ensemble...")
stacked_predictions = meta_model.predict(oof_test_predictions)

# --- Clip ---
stacked_predictions_clipped = stacked_predictions.clip(0, 20)

# --- Create Submission ---
print("Creating submission file...")
submission_df = pd.DataFrame({
    'ID': final_test_features['ID'], # Ensure final_test_features is available from Step 5
    'item_cnt_month': stacked_predictions_clipped
})
submission_df.to_csv('submission_stacked.csv', index=False)

print("Submission file created: submission_stacked.csv")
print(submission_df.head())

# --- Evaluate Stacked Model (Optional) ---
# You can see the RMSE of the meta-model's predictions on the validation set
meta_model_valid_preds = meta_model.predict(oof_valid_predictions)
stacked_rmse_on_valid = np.sqrt(mean_squared_error(y_valid, meta_model_valid_preds))
print(f"Stacked Ensemble RMSE on validation set: {stacked_rmse_on_valid}")

