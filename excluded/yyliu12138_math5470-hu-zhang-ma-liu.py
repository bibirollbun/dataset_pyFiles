# import os
# import pickle
# import numpy as np
# import pandas as pd
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import matplotlib.pyplot as plt


# data_dir = '/kaggle/input/m5-forecasting-accuracy/'
# save_dir = '/kaggle/working/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)


# calendar = pd.read_csv(data_dir + 'calendar.csv', header=0, index_col=None)
# evaluation = pd.read_csv(data_dir + 'sales_train_evaluation.csv', header=0, index_col=None)
# validation = pd.read_csv(data_dir + 'sales_train_validation.csv', header=0, index_col=None)
# prices = pd.read_csv(data_dir + 'sell_prices.csv', header=0, index_col=None)


# main_data = evaluation.copy()

# # reshape sales data so that each day's sales appear in a separate row
# main_data = main_data.drop(columns=['id'])
# columns_to_preserve = [col for col in main_data.columns if not col.startswith('d_')]
# main_data = main_data.melt(id_vars=columns_to_preserve, var_name='d', value_name='sales')

# # merge data from calender.csv with main_data
# main_data = main_data.merge(calendar, on='d', how='left')
# main_data = main_data.drop(columns=['date', 'weekday'])

# # Remove 'd_' prefix and convert to integer
# main_data['d'] = main_data['d'].str.extract(r'(\d+)').astype(int)

# # update the 'snap' column based on 'state_id'
# main_data = main_data.rename(columns={'snap_CA': 'snap'})
# main_data.loc[main_data['state_id'] == 'TX', 'snap'] = main_data['snap_TX']
# main_data.loc[main_data['state_id'] == 'WI', 'snap'] = main_data['snap_WI']
# main_data = main_data.drop(columns=['snap_TX', 'snap_WI'])

# # merge data from sell_prices.csv with main_data
# main_data = main_data.merge(prices, on=['item_id', 'store_id', 'wm_yr_wk'], how='left')

# # remove where sell price is NaN, which means not selling
# main_data = main_data.dropna(subset=['sell_price'])

# main_data


# # extract and save data from day 1 to day 1913
# d1_d1913_data = main_data[main_data['d'] <= 1913]
# d1_d1913_data.to_csv(data_dir + 'd1_d1913_data.csv', index=False, header=True)

# # extract and save data from day 1914 to day 1941
# d1914_d1941_data = main_data[main_data['d'] > 1913]
# d1914_d1941_data.to_csv(data_dir + 'd1914_d1941_data.csv', index=False, header=True)  


# submission_file = pd.read_csv(data_dir + 'sample_submission.csv', header=0, index_col=None)
# submission = submission_file.copy()

# calender_copy = calendar.copy()
# calender_copy['d'] = calender_copy['d'].str.extract(r'(\d+)').astype(int)

# split_ids = submission['id'].str.rsplit('_', n=5)
# submission['item_id'] = split_ids.str[0] + '_' + split_ids.str[1] + '_' + split_ids.str[2]
# submission['store_id'] = split_ids.str[3] + '_' + split_ids.str[4]
# submission['state_id'] = split_ids.str[3]

# columns_to_preserve = [col for col in submission.columns if not col.startswith('F')]
# submission = submission.melt(id_vars=columns_to_preserve, var_name='d', value_name='sales')

# sub_valid = submission[submission['id'].str.endswith('_validation')]
# sub_valid['d'] = sub_valid['d'].str.extract(r'(\d+)').astype(int) + 1913
# sub_valid = sub_valid.merge(calender_copy, on='d', how='left')
# sub_valid = sub_valid.drop(columns=['date', 'weekday'])
# sub_valid = sub_valid.rename(columns={'snap_CA': 'snap'})
# sub_valid.loc[sub_valid['state_id'] == 'TX', 'snap'] = sub_valid['snap_TX']
# sub_valid.loc[sub_valid['state_id'] == 'WI', 'snap'] = sub_valid['snap_WI']
# sub_valid = sub_valid.drop(columns=['snap_TX', 'snap_WI'])
# sub_valid = sub_valid.merge(prices, on=['item_id', 'store_id', 'wm_yr_wk'], how='left')

# sub_eval = submission[submission['id'].str.endswith('_evaluation')]
# sub_eval['d'] = sub_eval['d'].str.extract(r'(\d+)').astype(int) + 1941
# sub_eval = sub_eval.merge(calender_copy, on='d', how='left')
# sub_eval = sub_eval.drop(columns=['date', 'weekday'])
# sub_eval = sub_eval.rename(columns={'snap_CA': 'snap'})
# sub_eval.loc[sub_eval['state_id'] == 'TX', 'snap'] = sub_eval['snap_TX']
# sub_eval.loc[sub_eval['state_id'] == 'WI', 'snap'] = sub_eval['snap_WI']
# sub_eval = sub_eval.drop(columns=['snap_TX', 'snap_WI'])
# sub_eval = sub_eval.merge(prices, on=['item_id', 'store_id', 'wm_yr_wk'], how='left')


# sub_valid


# sub_eval


# sub_valid.to_csv(data_dir + 'sub_valid.csv', index=False, header=True)
# sub_eval.to_csv(data_dir + 'sub_eval.csv', index=False, header=True)  


# d1_d1913_data = pd.read_csv(data_dir + 'd1_d1913_data.csv', index_col=None, header=0)  # 46881677
# d1914_d1941_data = pd.read_csv(data_dir + 'd1914_d1941_data.csv', index_col=None, header=0) 
# sub_valid = pd.read_csv(data_dir + 'sub_valid.csv', index_col=None, header=0)
# sub_eval = pd.read_csv(data_dir + 'sub_eval.csv', index_col=None, header=0)  


# sorted(set(sub_valid['store_id']))


# # plot the histogram of the 'sales' column
# sales_counts = d1_d1913_data['sales'].value_counts().sort_index()  # Sort by sales values

# print(sales_counts)

# plt.figure(figsize=(10, 6))
# plt.bar(sales_counts.index, sales_counts.values, edgecolor='black')
# plt.yscale('log')  # Set the y-axis to log scale
# plt.title('Sales Distribution')
# plt.xlabel('Sales')
# plt.ylabel('Frequency (log scale)')
# plt.show()


# # Calculate the 99th percentile of the 'sales' column (non-zeros, no event, no snap)
# tmp = d1_d1913_data[d1_d1913_data['sales'] > 0].copy()
# tmp = tmp[tmp['event_type_1'].isna()]
# tmp = tmp[tmp['snap'] == 0]
# percentile = tmp['sales'].quantile(0.99)
# print(f'99-th percentile: {percentile}')

# # Retain rows where the 'sales' value is less than or equal to the 99th percentile
# tmp1 = d1_d1913_data[(~d1_d1913_data['event_type_1'].isna()) | (d1_d1913_data['snap'] == 1) | (d1_d1913_data['sales'] == 0)]
# filtered_data = tmp[tmp['sales'] <= percentile]
# # filtered_data = tmp
# print(tmp.shape)
# print(filtered_data.shape)
# filtered_data = pd.concat([filtered_data, tmp1])

# # plot the histogram of the 'sales' column
# sales_counts = filtered_data['sales'].value_counts().sort_index()  # Sort by sales values

# print(sales_counts)

# plt.figure(figsize=(10, 6))
# plt.bar(sales_counts.index, sales_counts.values, edgecolor='black')
# plt.yscale('log')  # Set the y-axis to log scale
# plt.title('Sales Distribution')
# plt.xlabel('Sales')
# plt.ylabel('Frequency (log scale)')
# plt.show()


# # check zero counts and non-zero counts
# zero_sales_count = (d1_d1913_data['sales'] == 0).sum()
# non_zero_sales_count = (d1_d1913_data['sales'] != 0).sum()

# print(f'Zeros: {zero_sales_count}')
# print(f'Non-zeros: {non_zero_sales_count}')


# # sample zero sales data at random to match the number of non-zero sales data
# non_zero_data = d1_d1913_data[d1_d1913_data['sales'] != 0].copy()
# zero_data = d1_d1913_data[d1_d1913_data['sales'] == 0].copy()
# sampled_data = d1_d1913_data.copy()
# # non_zero_data = filtered_data[filtered_data['sales'] != 0].copy()
# # zero_data = filtered_data[filtered_data['sales'] == 0].copy()
# # random_zero_data = zero_data.sample(n=non_zero_sales_count, random_state=42)
# # sampled_data = pd.concat([random_zero_data, non_zero_data])
# sampled_data.shape


# features = ['sell_price', 'year', 'wday', 'month', 'snap', 'item_id', 'event_type_1', 'event_type_2']#, 'cat_id', 'dept_id', 'state_id']
# # X = d1_d1913_data[features].copy()
# # y = d1_d1913_data['sales'].copy()
# # X_all = sampled_data[features].copy()
# # y_all = sampled_data['sales'].copy()

# categorical_features = ['wday', 'month', 'snap', 'item_id', 'event_type_1', 'event_type_2']

# # train a model for each store
# store_id = sorted(set(sub_valid['store_id']))

# for sid in store_id:

#     current_train = sampled_data[sampled_data['store_id'] == sid].copy()
#     X = current_train[features].copy()
#     y = current_train['sales'].copy()
#     print(X.shape)

#     X[categorical_features] = X[categorical_features].astype('category')

#     # split the data to train set and validation set
#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

#     # construct lightgbm dataset
#     train_dataset = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
#     valid_dataset = lgb.Dataset(X_valid, label=y_valid, reference=train_dataset, categorical_feature=categorical_features)

#     # parameters
#     params = {
#         'objective': 'tweedie',
#         'tweedie_variance_power': 1.1,
#         'metric': 'huber',  
#         'boosting_type': 'gbdt',
#         'n_estimators': 3000,
#         'subsample': 0.5,
#         'subsample_freq': 1,
#         'num_leaves': 1024,
#         'learning_rate': 0.05,
#         'feature_fraction': 0.5,
#     }

#     # train the model
#     model = lgb.train(params, train_dataset, valid_sets=[valid_dataset], num_boost_round=100, callbacks=[lgb.early_stopping(10)])

#     with open(save_dir + f'{sid}_model.pkl', 'wb') as f:
#         pickle.dump(model, f)
#     # with open(save_dir + f'{sid}_model.pkl', 'rb') as f:
#     #     model = pickle.load(f)

#     valid_preds = model.predict(X_valid)
#     valid_preds = np.round(valid_preds)
#     valid_preds[valid_preds < 0] = 0
#     rmse = np.sqrt(mean_squared_error(y_valid, valid_preds))
#     print(f'Validation RMSE: {rmse:.4f}')

#     # test data
#     current_test = d1914_d1941_data[d1914_d1941_data['store_id'] == sid].copy()
#     X_test = current_test[features].copy()
#     y_test = current_test['sales'].copy()
#     print(X_test.shape)

#     X_test[categorical_features] = X_test[categorical_features].astype('category').copy()
#     y_pred = model.predict(X_test)
#     y_pred = np.round(y_pred)
#     y_pred[y_pred < 0] = 0
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     print(f'Test RMSE: {rmse:.4f}')

#     # predict and save
#     sub_valid_X = sub_valid[sub_valid['store_id'] == sid].copy()
#     sub_valid_X = sub_valid_X[features]
#     print(sub_valid_X.shape)

#     sub_valid_X[categorical_features] = sub_valid_X[categorical_features].astype('category')
#     sub_valid_y = model.predict(sub_valid_X)
#     sub_valid_y = np.round(sub_valid_y)
#     sub_valid_y[sub_valid_y < 0] = 0

#     sub_valid.loc[sub_valid['store_id'] == sid, 'sales'] = sub_valid_y


#     sub_eval_X = sub_eval[sub_eval['store_id'] == sid].copy()
#     sub_eval_X = sub_eval_X[features]
#     print(sub_eval_X.shape)

#     sub_eval_X[categorical_features] = sub_eval_X[categorical_features].astype('category')
#     sub_eval_y = model.predict(sub_eval_X)
#     sub_eval_y = np.round(sub_eval_y)
#     sub_eval_y[sub_eval_y < 0] = 0

#     sub_eval.loc[sub_eval['store_id'] == sid, 'sales'] = sub_eval_y

# # save
# sub_valid.to_csv(save_dir + 'sub_valid_results.csv', index=False, header=True)
# sub_valid['d'] = 'F' + (sub_valid['d'] - 1913).astype(str)
# sub_eval.to_csv(save_dir + 'sub_eval_results.csv', index=False, header=True)  
# sub_eval['d'] = 'F' + (sub_eval['d'] - 1941).astype(str)

# result = pd.concat([sub_valid, sub_eval], axis=0, ignore_index=True)
# pivot_df = result.pivot(index='id', columns='d', values='sales')
# pivot_df.columns = [f"{col}" for col in pivot_df.columns]
# pivot_df = pivot_df.reset_index()
# pivot_df.to_csv(save_dir + 'sample_submission_results.csv', index=False, header=True)  

# pivot_df = pd.read_csv('../results/sample_submission_results.csv', header=0, index_col=None)
# new_order = ['id'] + [f'F{i}' for i in range(1, 29)]
# pivot_df = pivot_df[new_order]
# pivot_df.to_csv('../results/sample_submission.csv', index=False, header=True)


import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


data_dir = '/kaggle/input/m5-forecasting-accuracy/'
model_dir = '/kaggle/input/trained-lgbm/other/default/1/'
save_dir = '/kaggle/working/'


calendar = pd.read_csv(data_dir + 'calendar.csv', header=0, index_col=None)
prices = pd.read_csv(data_dir + 'sell_prices.csv', header=0, index_col=None)
submission_file = pd.read_csv(data_dir + 'sample_submission.csv', header=0, index_col=None)
submission = submission_file.copy()

calender_copy = calendar.copy()
calender_copy['d'] = calender_copy['d'].str.extract(r'(\d+)').astype(int)

split_ids = submission['id'].str.rsplit('_', n=5)
submission['item_id'] = split_ids.str[0] + '_' + split_ids.str[1] + '_' + split_ids.str[2]
submission['store_id'] = split_ids.str[3] + '_' + split_ids.str[4]
submission['state_id'] = split_ids.str[3]

columns_to_preserve = [col for col in submission.columns if not col.startswith('F')]
submission = submission.melt(id_vars=columns_to_preserve, var_name='d', value_name='sales')

sub_valid = submission[submission['id'].str.endswith('_validation')]
sub_valid['d'] = sub_valid['d'].str.extract(r'(\d+)').astype(int) + 1913
sub_valid = sub_valid.merge(calender_copy, on='d', how='left')
sub_valid = sub_valid.drop(columns=['date', 'weekday'])
sub_valid = sub_valid.rename(columns={'snap_CA': 'snap'})
sub_valid.loc[sub_valid['state_id'] == 'TX', 'snap'] = sub_valid['snap_TX']
sub_valid.loc[sub_valid['state_id'] == 'WI', 'snap'] = sub_valid['snap_WI']
sub_valid = sub_valid.drop(columns=['snap_TX', 'snap_WI'])
sub_valid = sub_valid.merge(prices, on=['item_id', 'store_id', 'wm_yr_wk'], how='left')

sub_eval = submission[submission['id'].str.endswith('_evaluation')]
sub_eval['d'] = sub_eval['d'].str.extract(r'(\d+)').astype(int) + 1941
sub_eval = sub_eval.merge(calender_copy, on='d', how='left')
sub_eval = sub_eval.drop(columns=['date', 'weekday'])
sub_eval = sub_eval.rename(columns={'snap_CA': 'snap'})
sub_eval.loc[sub_eval['state_id'] == 'TX', 'snap'] = sub_eval['snap_TX']
sub_eval.loc[sub_eval['state_id'] == 'WI', 'snap'] = sub_eval['snap_WI']
sub_eval = sub_eval.drop(columns=['snap_TX', 'snap_WI'])
sub_eval = sub_eval.merge(prices, on=['item_id', 'store_id', 'wm_yr_wk'], how='left')


store_id = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 
            'TX_1', 'TX_2', 'TX_3', 
            'WI_1', 'WI_2', 'WI_3']


features = ['sell_price', 'year', 'wday', 'month', 'snap', 'item_id', 'event_type_1', 'event_type_2']

categorical_features = ['wday', 'month', 'snap', 'item_id', 'event_type_1', 'event_type_2']

for sid in store_id:
    
    with open(model_dir + f'{sid}_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # predict and save
    sub_valid_X = sub_valid[sub_valid['store_id'] == sid].copy()
    sub_valid_X = sub_valid_X[features]
    print(sub_valid_X.shape)

    sub_valid_X[categorical_features] = sub_valid_X[categorical_features].astype('category')
    sub_valid_y = model.predict(sub_valid_X)
    sub_valid_y = np.round(sub_valid_y)
    sub_valid_y[sub_valid_y < 0] = 0

    sub_valid.loc[sub_valid['store_id'] == sid, 'sales'] = sub_valid_y


    sub_eval_X = sub_eval[sub_eval['store_id'] == sid].copy()
    sub_eval_X = sub_eval_X[features]
    print(sub_eval_X.shape)

    sub_eval_X[categorical_features] = sub_eval_X[categorical_features].astype('category')
    sub_eval_y = model.predict(sub_eval_X)
    sub_eval_y = np.round(sub_eval_y)
    sub_eval_y[sub_eval_y < 0] = 0

    sub_eval.loc[sub_eval['store_id'] == sid, 'sales'] = sub_eval_y

# save
# sub_valid.to_csv(save_dir + 'sub_valid_results.csv', index=False, header=True)
sub_valid['d'] = 'F' + (sub_valid['d'] - 1913).astype(str)
# sub_eval.to_csv(save_dir + 'sub_eval_results.csv', index=False, header=True)  
sub_eval['d'] = 'F' + (sub_eval['d'] - 1941).astype(str)

result = pd.concat([sub_valid, sub_eval], axis=0, ignore_index=True)
pivot_df = result.pivot(index='id', columns='d', values='sales')
pivot_df.columns = [f"{col}" for col in pivot_df.columns]
pivot_df = pivot_df.reset_index()

new_order = ['id'] + [f'F{i}' for i in range(1, 29)]
pivot_df = pivot_df[new_order]
pivot_df.to_csv(save_dir + 'submission.csv', index=False, header=True)

