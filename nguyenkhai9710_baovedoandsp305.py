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


df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
df2 = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
price = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')


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

    new_memory = df.memory_usage().sum() / 1024**2
    print(f"new memory usage is:{new_memory} MB")

    return df

df = optimize_dataframe_memory(df)
df2 = optimize_dataframe_memory(df2)
prices = optimize_dataframe_memory(price)


sale = df.melt(
id_vars = ['id','item_id','dept_id','cat_id','store_id','state_id'],
var_name = 'd', value_name = 'sales')


sale = sale.merge(
df2[['d','date','wm_yr_wk']], on = 'd', how = 'left')


sale['date'] = pd.to_datetime(sale['date'])


mer = sale.merge(price[['sell_price','store_id','item_id','wm_yr_wk']], on = ['store_id','item_id','wm_yr_wk'], how = 'left')


mer2 = mer.drop(columns=['state_id','wm_yr_wk'])


mer2['sell_price'] = mer2.groupby('item_id')['sell_price'].transform('mean')


store_ids = mer2['store_id'].unique()
state_list = []
for store in store_ids:
    var_name = f"store_{store}_df"
    print(var_name)
    globals()[var_name] = mer2[mer2['store_id'] == store].copy()
    state_list.append(var_name)


# !pip install -U "flaml[automl]"
# from flaml.automl import AutoML

# from flaml.automl import AutoML
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
# import numpy as np
# import pandas as pd

# # HÃ m RMSLE custom
# def rmsle(y_true, y_pred):
#     return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

# # HÃ m RMSSE
# def rmsse(y_true, y_pred):
#     numerator = np.mean((y_true - y_pred) ** 2)
#     denominator = np.mean(np.square(np.diff(y_true)))
#     return np.sqrt(numerator / denominator) if denominator != 0 else np.nan
    
# # HÃ m chÃ­nh
# def evaluate_store_df(df, sample_size=10000, lag_days=[1, 7, 14], roll_windows=[7, 14]):
#     # Láº¥y máº«u vÃ  sáº¯p xáº¿p theo ngÃ y
#     # df_sample = df.sample(sample_size, random_state=42).copy()
#     df_sample = df.copy()
#     df_sample['date'] = pd.to_datetime(df_sample['date'])
#     df_sample = df_sample.sort_values('date')

#     # ThÃªm Ä‘áº·c trÆ°ng lag
#     for lag in lag_days:
#         df_sample[f'sales_lag_{lag}'] = df_sample['sales'].shift(lag)

#     # ThÃªm rolling mean
#     for window in roll_windows:
#         df_sample[f'sales_roll_mean_{window}'] = df_sample['sales'].shift(1).rolling(window=window).mean()

#     # Loáº¡i bá»� cÃ¡c dÃ²ng bá»‹ NaN do lag/rolling
#     df_sample = df_sample.dropna()

#     # Encode cÃ¡c cá»™t dáº¡ng object
#     for col in df_sample.select_dtypes(include='object').columns:
#         df_sample[col] = LabelEncoder().fit_transform(df_sample[col])

#     # TrÃ­ch Ä‘áº·c trÆ°ng thá»�i gian trÆ°á»›c
#     df_sample['dayofweek'] = df_sample['date'].dt.dayofweek
#     df_sample['month'] = df_sample['date'].dt.month

#     # XÃ³a nhá»¯ng cá»™t thá»±c sá»± khÃ´ng cáº§n
#     drop_cols = []  # giá»¯ láº¡i item_id, store_id, dept_id
#     df_sample = df_sample.drop(columns=[col for col in drop_cols if col in df_sample.columns])

#     # TÃ¡ch X, y
#     X = df_sample.drop(columns=['sales'])
#     y = df_sample['sales']

#     # TÃ¡ch train/test theo thá»�i gian: 80% Ä‘áº§u tiÃªn lÃ m train, 20% cuá»‘i lÃ m test
#     split_index = int(len(X) * 0.8)
#     X_train, X_test = X.iloc[:-28], X.iloc[-28:]
#     y_train, y_test = y.iloc[:-28], y.iloc[-28:]
    
#     # Train AutoML
#     automl = AutoML()
#     automl.fit(X_train=X_train, y_train=y_train, task="regression", time_budget=60)

#     # Dá»± Ä‘oÃ¡n
#     y_pred = automl.predict(X_test)

#     # Káº¿t quáº£
#     return {
#         'model': automl.model.estimator.__class__.__name__,
#         'RMSSE': rmsse(y_test.values, y_pred),
#         'MAE': mean_absolute_error(y_test, y_pred),
#         'MSE': mean_squared_error(y_test, y_pred),
#         'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
#         'R2': r2_score(y_test, y_pred),
#         'MAPE': mean_absolute_percentage_error(y_test, y_pred),
#         'RMSLE': rmsle(y_test, y_pred),
        
#         'automl': automl 
#     }



# results = {}

# for store_name in state_list:
#     print(f"ğŸ”� Evaluating {store_name}...")
#     df = globals()[store_name]
#     try:
#         metrics = evaluate_store_df(df)
#         results[store_name] = metrics
#     except Exception as e:
#         results[store_name] = {'error': str(e)}
#         print(f"âš ï¸� Lá»—i táº¡i {store_name}: {e}")



# results_df = pd.DataFrame(results).T
# results_df = results_df[['model', 'RMSSE', 'MAE', 'MSE', 'RMSE', 'R2', 'MAPE', 'RMSLE']]
# results_df


def make_features(df):
    df = df.sort_values(['item_id', 'store_id', 'date'])

    for lag in [1, 7, 14]:
        df[f'lag_{lag}'] = df.groupby(['item_id', 'store_id'],observed=False)['sales'].shift(lag)

    for win in [7, 14]:
        df[f'rolling_mean_{win}'] = df.groupby(['item_id', 'store_id'],observed=False)['sales'].shift(1).rolling(window=win).mean()

    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    return df



import pandas as pd
import numpy as np
import lightgbm as lgb

def predict_28days(df, model, feature_cols):
    df_all = df.copy()
    df_all['date'] = pd.to_datetime(df_all['date'])
    last_date = df_all['date'].max()
    last_d = int(df_all[df_all['date'] == last_date]['d'].iloc[0].split('_')[1])

    future_predictions = []

    for i in range(1, 29):  # 28 ngÃ y tiáº¿p theo
        next_date = last_date + pd.Timedelta(days=i)

        # Táº¡o 1 báº£n copy tá»« ngÃ y cuá»‘i cÃ¹ng (d_1913)
        new_rows = df_all[df_all['date'] == last_date].copy()
        new_rows['date'] = next_date
        new_rows['d'] = f'd_{last_d + i}'  # náº¿u cáº§n giá»¯ d dáº¡ng string
        new_rows['sales'] = np.nan

        # ThÃªm vÃ o dá»¯ liá»‡u gá»‘c
        df_all = pd.concat([df_all, new_rows], ignore_index=True)

        # Táº¡o Ä‘áº·c trÆ°ng
        df_all = make_features(df_all)  # báº¡n cáº§n Ä‘á»‹nh nghÄ©a hÃ m nÃ y

        # Chá»‰ láº¥y dÃ²ng cá»§a ngÃ y cáº§n dá»± Ä‘oÃ¡n
        pred_input = df_all[df_all['date'] == next_date].copy()
        X_pred = pred_input[feature_cols]

        # Dá»± Ä‘oÃ¡n (KhÃ´ng cáº§n DMatrix)
        y_pred = model.predict(X_pred)
        pred_input['predicted_sales'] = y_pred

        # GÃ¡n láº¡i káº¿t quáº£ vÃ o df_all
        df_all.loc[df_all['date'] == next_date, 'sales'] = y_pred

        # LÆ°u káº¿t quáº£
        future_predictions.append(pred_input[['id','item_id', 'dept_id' , 'cat_id','store_id', 'd', 'date', 'predicted_sales']])

    result = pd.concat(future_predictions, ignore_index=True)
    return result



import lightgbm as lgb

for i in state_list:
    df3 = globals()[i]
    df_features = make_features(df3)
    
    # Bá»� cÃ¡c dÃ²ng NaN sau khi táº¡o Ä‘áº·c trÆ°ng
    df_train = df_features.dropna()

    feature_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_14', 'dayofweek', 'month']
    X_train = df_train[feature_cols]
    y_train = df_train['sales']

    # âœ… DÃ¹ng lgb.Dataset thay vÃ¬ lgb.DMatrix
    dtrain = lgb.Dataset(X_train, label=y_train)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'seed': 42
    }

    # âœ… DÃ¹ng lgb.train Ä‘Ãºng cÃ¡ch
    model = lgb.train(params, dtrain, num_boost_round=100)

    # âœ… Dá»± Ä‘oÃ¡n 28 ngÃ y
    forecast = predict_28days(df_features, model, feature_cols)

    # âœ… LÆ°u vÃ o biáº¿n Ä‘á»™ng
    globals()[f"{i}_x2"] = forecast



combined_df = pd.concat([store_CA_1_df_x2,store_CA_2_df_x2, store_CA_3_df_x2, store_CA_4_df_x2, store_TX_1_df_x2, store_TX_2_df_x2, store_TX_3_df_x2, store_WI_1_df_x2, store_WI_2_df_x2, store_WI_3_df_x2 ], axis=0, ignore_index=True)
combined_df


combined_df2 = combined_df.drop(['item_id',	'dept_id',	'cat_id',	'store_id', 'date'], axis = 1)


combined_df2


pivot_df = combined_df2.pivot(index='id', columns='d', values='predicted_sales').reset_index()

# Láº¥y danh sÃ¡ch cÃ¡c cá»™t d_1942 Ä‘áº¿n d_1969
d_cols = [f'd_{i}' for i in range(1914, 1942)]  # 1970 vÃ¬ range khÃ´ng bao gá»“m cuá»‘i

# Táº¡o dictionary Ä‘á»•i tÃªn: d_1942 -> f1, d_1943 -> f2, ..., d_1969 -> f28
rename_dict = {old: f'F{idx+1}' for idx, old in enumerate(d_cols)}

# Ä�á»•i tÃªn cÃ¡c cá»™t
pivot_df = pivot_df.rename(columns=rename_dict)

# Sáº¯p xáº¿p theo cá»™t f1 tÄƒng dáº§n
pivot_df = pivot_df.sort_values(by='F1', ascending=True).reset_index(drop=True)

pivot_df.head()


pivot_df


# pivot_df.to_csv('submission.csv', index=False)


sales_evaluation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')


sales_evaluation_4 = sales_evaluation[['id','d_1914','d_1915','d_1916','d_1917',	'd_1918','d_1919','d_1920','d_1921','d_1922','d_1923','d_1924','d_1925','d_1926','d_1927','d_1928','d_1929','d_1930','d_1931','d_1932','d_1933','d_1934','d_1935','d_1936','d_1937','d_1938','d_1939','d_1940','d_1941']]
sales_evaluation_4


# Láº¥y danh sÃ¡ch cÃ¡c cá»™t d_1942 Ä‘áº¿n d_1969
d_cols2 = [f'd_{i}' for i in range(1914, 1942)]  # 1970 vÃ¬ range khÃ´ng bao gá»“m cuá»‘i

# Táº¡o dictionary Ä‘á»•i tÃªn: d_1942 -> f1, d_1943 -> f2, ..., d_1969 -> f28
rename_dict2 = {old: f'F{idx+1}' for idx, old in enumerate(d_cols)}

# Ä�á»•i tÃªn cÃ¡c cá»™t
sales_evaluation_4 = sales_evaluation_4.rename(columns=rename_dict)

# Sáº¯p xáº¿p theo cá»™t f1 tÄƒng dáº§n
sales_evaluation_5 = sales_evaluation_4.sort_values(by='F1', ascending=True).reset_index(drop=True)

sales_evaluation_5.head()


pivot_df3 = pivot_df.copy()
pivot_df3.index.name = None  # XoÃ¡ tÃªn index "d"
pivot_df3 = pivot_df3.reset_index(drop=True)
pivot_df3


final_result1 = pd.concat([pivot_df,sales_evaluation_5], axis=0, ignore_index=True)
final_result1


final_result2 = final_result1.sort_values(by='id').reset_index(drop=True)
final_result2


final_result2.to_csv('submission.csv', index=False)


pivot_df5 = pivot_df.sort_values(by='id').reset_index(drop=True)
pivot_df5


sales_evaluation_6 = sales_evaluation_5.sort_values(by='id').reset_index(drop=True)
sales_evaluation_6


import plotly.express as px


department = combined_df.groupby('dept_id')['predicted_sales'].sum().reset_index()
fig = px.line(department, x='dept_id', y='predicted_sales',
              title='Total dept_id Sales Over Time')
fig.show()


combined_df


sales_evaluation


sales_evaluation_6 = sales_evaluation[['id','dept_id','d_1914','d_1915','d_1916','d_1917',	'd_1918','d_1919','d_1920','d_1921','d_1922','d_1923','d_1924','d_1925','d_1926','d_1927','d_1928','d_1929','d_1930','d_1931','d_1932','d_1933','d_1934','d_1935','d_1936','d_1937','d_1938','d_1939','d_1940','d_1941']]
sales_evaluation_6


sale_evaluation_8 = sales_evaluation_6.melt(
id_vars = ['id','dept_id'],
var_name = 'd', value_name = 'sales')


sale_evaluation_8


department4 = sale_evaluation_8.groupby('dept_id')['sales'].sum().reset_index()
fig = px.line(department4, x='dept_id', y='sales',
              title='Total dept_id Sales Over Time')
fig.show()


import seaborn as sns
import matplotlib.pyplot as plt
# Táº¡o má»™t biá»ƒu Ä‘á»“ má»›i
plt.figure(figsize=(10, 6))

# Váº½ Ä‘Æ°á»�ng cho df1
sns.lineplot(data=department, x='dept_id', y='predicted_sales', label='Ä�Æ°á»�ng 1 light gbm', marker='o')

# Váº½ Ä‘Æ°á»�ng cho df2
sns.lineplot(data=department4, x='dept_id', y='sales', label='Ä�Æ°á»�ng 2 cá»§a dá»¯ liá»‡u so sÃ¡nh', marker='s')

# ThÃªm tiÃªu Ä‘á»� vÃ  nhÃ£n cho trá»¥c
plt.title('So sÃ¡nh hai Ä‘Æ°á»�ng trÃªn cÃ¹ng má»™t biá»ƒu Ä‘á»“')
plt.xlabel('dept_id')
plt.ylabel('predicted_sales')

# Hiá»ƒn thá»‹ legend
plt.legend()

# Hiá»ƒn thá»‹ biá»ƒu Ä‘á»“
plt.grid(True)
plt.show()

