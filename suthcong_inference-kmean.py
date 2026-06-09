import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mat
import matplotlib.pyplot as plt
import os
 
sns.set(color_codes = True)
# Set seed for reproducibility
np.random.seed(2020)
import warnings
import matplotlib.pyplot as plt

# Ignore all warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv').drop(['warehouse','product_unique_id'],axis=1)
inventory.head()

def fe_date(df):
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    df['days_since_2020'] = (df['date'] - pd.to_datetime('2020-01-01')).dt.days.astype('int')
    df['day_of_year'] = df['date'].dt.dayofyear
    df['cos_day'] = np.cos(df['day_of_year']*2*np.pi/365)
    df['sin_day'] = np.sin(df['day_of_year']*2*np.pi/365)

def fe_other(df):
    discount_cols = ['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount']
    df[discount_cols] = df[discount_cols].clip(0)
    df['max_discount'] = df[['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount']].max(axis=1)
    
    # Given that we're using XGBoost, which is in theory invariant to monotonic transformations of features, this transformation in isolation doesn't really do anything. I mainly did it because it made the shap plot look more linear. However, I think it did make further feature engineering that used price more effective.
    df['sell_price_main'] = np.log(df['sell_price_main']) 

    df['common_name'] = df['name'].apply(lambda x: x[:x.find('_')])
    df['CN_total_products'] = df.groupby(['date','warehouse','common_name'])['unique_id'].transform('nunique')
    df['CN_discount_avg'] = df.groupby(['date','warehouse','common_name'])['max_discount'].transform('mean')
    df['CN_WH'] = df['common_name'] + '_' + df['warehouse']
    df['name_num_warehouses'] = df.groupby(['date','name'])['unique_id'].transform('nunique')

def fe_combined(df):
    df['num_sales_days_28D'] = pd.MultiIndex.from_frame(df[['unique_id','date']]).map(df.sort_values('date').groupby('unique_id').rolling(
        window='28D', on='date', closed='left')['date'].count().fillna(0))

    # This 'price_detrended' feature was one I found pretty late into the game, but I think it helped out a lot. I was trying to make a feature that captured whether an item was cheap or expensive relative to its usual price, which is what 'price_scaled' represents. What I found was that the prices of things generally increase over time. So I removed that time-based trend to construct price_detrended, and that proved very effective.
    mean_prices = df.groupby(df['unique_id'])['sell_price_main'].mean()
    std_prices = df.groupby(df['unique_id'])['sell_price_main'].std()
    df['price_scaled'] = np.where(df['unique_id'].map(std_prices) == 0, 0, 
                                  (df['sell_price_main'] - df['unique_id'].map(mean_prices))/df['unique_id'].map(std_prices))
    df['price_detrended'] = df['price_scaled'] - df.groupby(['days_since_2020','warehouse'])['price_scaled'].transform('mean')
    df.drop('price_scaled',axis=1,inplace=True)

    warehouse_stats = df.groupby(['date','warehouse'])['total_orders'].median().rename('med_total_orders').reset_index().sort_values('date')
    warehouse_stats['ewmean_orders_56'] = warehouse_stats.groupby('warehouse')['med_total_orders'].transform(lambda x:x.ewm(alpha=1/56).mean())
    df['mean_orders_14d'] = pd.MultiIndex.from_frame(df[['warehouse','date']]).map(
        warehouse_stats.groupby('warehouse').rolling(on='date',window='14D')['med_total_orders'].mean())
    df['ewmean_orders_56'] = pd.MultiIndex.from_frame(df[['warehouse','date']]).map(
        warehouse_stats.set_index(['warehouse','date'])['ewmean_orders_56'])
    return df
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv', parse_dates=['date'])
calendar.loc[calendar['holiday_name'].isna(), 'holiday'] = 0 # V3
calendar['last_holiday_date'] = calendar['date']
calendar['next_holiday_date'] = calendar['date']
calendar.loc[calendar['holiday'] == 0, ['last_holiday_date','next_holiday_date']] = np.nan
calendar['last_holiday_date'] = calendar.sort_values('date').groupby('warehouse')['last_holiday_date'].ffill()
calendar['next_holiday_date'] = calendar.sort_values('date').groupby('warehouse')['next_holiday_date'].bfill()
calendar['days_since_last_holiday'] = ((calendar['date'] - calendar['last_holiday_date']).dt.days)
calendar['days_to_next_holiday'] = ((calendar['next_holiday_date'] - calendar['date']).dt.days)
calendar['day_before_holiday'] = calendar['days_to_next_holiday'] == 1
calendar['day_after_holiday'] = calendar['days_since_last_holiday'] == 1
calendar.drop(['last_holiday_date','next_holiday_date'],axis=1,inplace=True)
calendar.drop(['days_since_last_holiday','days_to_next_holiday'],axis=1,inplace=True)
calendar.drop(['shops_closed','winter_school_holidays','school_holidays','holiday_name'],axis=1,inplace=True)


test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
test['id'] = test['unique_id'].astype('str') + '_' + test['date'].astype('str')
test.set_index('id',inplace=True)
test = test.reset_index().merge(inventory, on='unique_id').set_index('id').loc[test.index]
test = test.reset_index().merge(calendar, on=['date','warehouse']).set_index('id')
fe_date(test)
fe_other(test)

all_data = pd.concat([test])
all_data = fe_combined(all_data)
test = all_data.loc[test.index]
test['date'] = pd.to_datetime(test['date'])
# Sort the DataFrame by the 'date' column
test = test.sort_values(by='date')


data = test[['warehouse', 'total_orders','sell_price_main', 'L2_category_name_en',
       'holiday', 'day_before_holiday', 'day_after_holiday','common_name','num_sales_days_28D', 'price_detrended',
       'mean_orders_14d', 'ewmean_orders_56','max_discount','CN_total_products','CN_discount_avg']]


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

df = pd.DataFrame(data)
df['price_detrended'].fillna(df['price_detrended'].mean(), inplace=True)
# del data
# Define feature categories
numerical_features = [
    'total_orders', 'sell_price_main', 'num_sales_days_28D', 'price_detrended',
    'mean_orders_14d', 'ewmean_orders_56', 'max_discount', 'CN_total_products', 'CN_discount_avg'
]
categorical_features = ['warehouse', 'L2_category_name_en', 'common_name']
binary_features = ['holiday', 'day_before_holiday', 'day_after_holiday']

# Define transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num_std', StandardScaler(), numerical_features),  # Standard scaling for numerical features
        ('cat_onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),  # One-hot encoding for categorical
        ('binary', 'passthrough', binary_features)  # Keep binary features as is
    ]
)

transformed_data = preprocessor.fit_transform(df)
transformed_data[0].toarray() 


from sklearn.metrics.pairwise import cosine_similarity

cos_sim = cosine_similarity(transformed_data)
cos_sim


from sklearn.decomposition import PCA
pca = PCA(n_components=150)  # Reduce to 2 components (adjust as needed)
cos_sim = pca.fit_transform(cos_sim)


import joblib
# Load the trained KNN model
knn_model = joblib.load('/kaggle/input/k/sthnhcng/knn-kmean-cosine/knn_model.joblib')

clusters = knn_model.predict(cos_sim)

clusters



clusters.shape


test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
test['date'] = pd.to_datetime(test['date'])
# Sort the DataFrame by the 'date' column
test = test.sort_values(by='date')


# test_sub = test_pred_df.mean(axis=1)
# test_sub.name = 'sales_hat'
# test_sub.to_csv('submission_top3.csv')


F = test.copy()
F.insert(5, 'Kmean_cluster', clusters, True)
F


F.to_csv('test_with_cluster.csv', index = False)




