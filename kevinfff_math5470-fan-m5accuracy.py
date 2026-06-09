import pandas as pd
sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')


import warnings
warnings.filterwarnings('ignore')


print(sales.shape)
sales.head(10)


for d in range(1942,1970):
    col = 'd_' + str(d)
    sales[col] = 0


print(calendar.shape)
calendar.head(10)


print(prices.shape)
prices.head(10)


import numpy as np

def optimize_dataframe_memory(df):
    old_memory = df.memory_usage().sum() / 1024**2
    print(f"old memory usage is:{old_memory} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type not in [object, 'category']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).startswith('int'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)
        else:
            if col_type == object:
                if df[col].nunique() / len(df[col]) < 0.5:
                    df[col] = df[col].astype('category')

    new_memory = df.memory_usage().sum() /1024**2
    print(f"new memory usage is:{new_memory} MB")

    return df

sales1 = optimize_dataframe_memory(sales)
calendar1 = optimize_dataframe_memory(calendar)
prices1 = optimize_dataframe_memory(prices)

del sales
del calendar
del prices


df = pd.melt(sales1, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='d', value_name='number')
df = pd.merge(df, calendar1, on='d', how='left')
df = pd.merge(df, prices1, on=['store_id','item_id','wm_yr_wk'], how='left') 
df.head(20)

del sales1
del calendar1
del prices1


print(df.columns.tolist())


df.drop(columns=['weekday'], inplace=True)
df.drop(columns=['wm_yr_wk'], inplace=True)


df['sell_price'] = df['sell_price'].fillna(df.groupby('id')['sell_price'].transform('median'))


lags = [29,30,31,32,33,34,35,50,60]
for lag in lags:
    df['number_lag_'+str(lag)] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],as_index=False)['number'].shift(lag).astype(np.float16)


df["state_mean"] = df.groupby("state_id")['number'].transform("mean").astype(np.float16)
df["store_mean"] = df.groupby("store_id")['number'].transform("mean").astype(np.float16)
df["item_id_mean"] = df.groupby("item_id")['number'].transform("mean").astype(np.float16)
df["item_state_mean"] = df.groupby(["item_id", "state_id"])['number'].transform("mean").astype(np.float16)
df["item_store_mean"] = df.groupby(["item_id", "store_id"])['number'].transform("mean").astype(np.float16)
df["state_median"] = df.groupby("state_id")['number'].transform("median").astype(np.float16)
df["store_median"] = df.groupby("store_id")['number'].transform("median").astype(np.float16)
df["item_id_median"] = df.groupby("item_id")['number'].transform("median").astype(np.float16)
df["item_state_median"] = df.groupby(["item_id", "state_id"])['number'].transform("median").astype(np.float16)
df["item_store_median"] = df.groupby(["item_id", "store_id"])['number'].transform("median").astype(np.float16)


df['d'] = df['d'].str.replace('d_', '').astype(int)
df = df[df['d']>=60]
df = optimize_dataframe_memory(df)

df.info()


# items = df["id"].unique()
# random_sample = pd.Series(items).sample(n=500, random_state=42)
# df_s = df[df['id'].isin(random_sample)]
# df_s.head(10)


# def train_val(df, lr, est, leaves):
#     val_mse = 0
#     count = 0
    
#     for store in stores:
#         data = df[df['store_id']==store]
    
#         X_train, y_train = data[data['d']<1914].drop('number',axis=1), data[data['d']<1914]['number']
#         X_valid, y_valid = data[(data['d']>=1914) & (data['d']<1942)].drop('number',axis=1), data[(data['d']>=1914) & (data['d']<1942)]['number']

#         if y_valid.empty:
#             continue
            
#         model = LGBMRegressor(
#             n_estimators=est,
#             learning_rate=lr,
#             num_leaves=leaves,
#             random_state=42
#         )
        
#         print(f'*****Training on Store: {store}*****')
        
#         model.fit(
#             X_train, y_train, 
#             eval_set=[(X_train, y_train), (X_valid, y_valid)],
#             eval_metric='rmse',
#             callbacks = [lgb.log_evaluation(period=10), lgb.early_stopping(stopping_rounds=20)]
#         )
    
#         val_pred = model.predict(X_valid)
#         val_mse += np.sum((val_pred - y_valid)**2)
#         count += len(val_pred)

#     return np.sqrt(val_mse/count)


# column_names = ['lr', 'n_estimators', 'num_leaves', 'rmse']
# rmse_df = pd.DataFrame(columns=column_names)


# for lr in [0.2, 0.1, 0.05]:
#     for est in [100, 200]:
#         for leaves in [50, 60, 70]:
#             print(f"——————————Train on sample df with lr={lr}, n_estimators={est}, num_leaves={leaves}——————————")
#             rmse = train_val(df_s, lr, est, leaves)
#             print(f"*****RMSE for lr={lr}, n_estimators={est}, num_leaves={leaves} is {rmse}.*****")
#             rmse_df.loc[len(rmse_df)] = [lr, est, leaves, rmse]


# rmse_df


X_train, y_train = df[df['d']<1914].drop('number',axis=1), df[df['d']<1914]['number']
X_valid, y_valid = df[(df['d']>=1914) & (df['d']<1942)].drop('number',axis=1), df[(df['d']>=1914) & (df['d']<1942)]['number']
X_test = df[df['d']>=1942].drop('number',axis=1)


stores = df['store_id'].unique()


import lightgbm as lgb
from lightgbm import LGBMRegressor

eval_results = []
val_results = []

for store in stores:
    val_mse = 0
    count = 0
    
    data = df[df['store_id']==store]

    X_train, y_train = data[data['d']<1914].drop('number',axis=1), data[data['d']<1914]['number']
    X_valid, y_valid = data[(data['d']>=1914) & (data['d']<1942)].drop('number',axis=1), data[(data['d']>=1914) & (data['d']<1942)]['number']
    X_test = data[data['d']>=1942].drop('number',axis=1)

    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=50,
        random_state=42
    )
    
    print(f'*****Training on Store: {store}*****')
    
    model.fit(
        X_train, y_train, 
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric='rmse',
        callbacks = [lgb.log_evaluation(period=10), lgb.early_stopping(stopping_rounds=20)]
    )

    eval_result = data[data['d']>=1942][['id', 'd']]
    eval_result['pred'] = model.predict(X_test)
    eval_results.append(eval_result)

    val_result = data[(data['d']>=1914) & (data['d']<1942)][['id', 'd']]
    val_result['pred'] = model.predict(X_valid)
    val_results.append(val_result)

    val_mse += np.sum((val_result['pred'] - y_valid)**2)
    count += len(y_valid)


print(f"Validation RMSE is {np.sqrt(val_mse/count)}")


def handle_format(results, is_val):
    wide_results = []
    for result in results:
        wide_df = result.pivot(index='id', columns='d', values='pred').reset_index()
        wide_results.append(wide_df)
    combined_df = pd.concat(wide_results, ignore_index=True)
    combined_df.columns = ['id']+[f'F{i+1}' for i in range(28)]
    if is_val:
        combined_df['id'] = combined_df['id'].str.replace('evaluation', 'validation')
    return combined_df


eval_submission = handle_format(eval_results, False)
val_submission = handle_format(val_results, True)
final_submission = pd.concat([eval_submission, val_submission], ignore_index=True)
final_submission


final_submission.to_csv('submission.csv', index=False)




