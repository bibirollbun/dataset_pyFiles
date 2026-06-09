import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor, DMatrix
import shap
shap.initjs()


inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv').drop(['warehouse','product_unique_id'],axis=1)
inventory.head()


def fe_date(df):
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    df['days_since_2020'] = (df['date'] - pd.to_datetime('2020-01-01')).dt.days.astype('int')
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_month'] = df['date'].dt.day.apply(lambda x: (x + 6) // 7)
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
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


train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
train['id'] = train['unique_id'].astype('str') + '_' + train['date'].astype('str')
train.set_index('id',inplace=True)
train = train[~train['sales'].isna()]
train = train.reset_index().merge(inventory, on='unique_id').set_index('id').loc[train.index]
train = train.reset_index().merge(calendar, on=['date','warehouse']).set_index('id').loc[train.index]
fe_date(train)
fe_other(train)

test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
test['id'] = test['unique_id'].astype('str') + '_' + test['date'].astype('str')
test.set_index('id',inplace=True)
test = test.reset_index().merge(inventory, on='unique_id').set_index('id').loc[test.index]
test = test.reset_index().merge(calendar, on=['date','warehouse']).set_index('id')
fe_date(test)
fe_other(test)

all_data = pd.concat([train,test])
all_data = fe_combined(all_data)
train = all_data.loc[train.index]
test = all_data.loc[test.index].drop(['sales','availability'],axis=1)


X_train = train.drop('sales',axis=1)
y_train = train['sales']
train_availability = X_train['availability']
X_train.drop('availability',inplace=True,axis=1)
weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv').set_index('unique_id')
X_train_weights = X_train['unique_id'].map(weights['weight'])


cat_cols = ['unique_id'] + list(X_train.columns[X_train.dtypes == 'object'])
all_data = pd.concat([X_train, test])
add_cols = ['last_sales_ema005','CN_sales_sum','last_sales_zs']

# Here there are a few additional features engineered from historical sales data. These are done separately from the rest of my feature engineering because when I go to test model performance on a time-based holdout validation set, I need to make sure these features aren't using sales data from that validation set.
train_cp = train.groupby('unique_id')['date'].apply(lambda s: pd.date_range(s.min(), test.date.max())).explode().reset_index()
train_cp = train_cp.merge(
    pd.concat([train[['unique_id','date','sales','warehouse',]], 
               test[['unique_id','date','warehouse']]]),
    on=['unique_id','date'],how='left')
train_cp = train_cp.merge(inventory, left_on='unique_id', right_index=True)
train_cp['common_name'] = train_cp['name'].apply(lambda x: x[:x.find('_')])
train_cp.sort_values('date',inplace=True)
train_cp['last_sales_ema005'] = train_cp.groupby(['unique_id'])['sales'].transform(lambda x: x.shift(1).ewm(alpha=.005).mean()).fillna(0)
train_cp['CN_sales_sum'] = train_cp.groupby(['common_name','warehouse','date'])['last_sales_ema005'].transform('sum')
all_data = all_data.merge(train_cp.set_index(['unique_id','date'])[[
    'last_sales_ema005','CN_sales_sum'
]], left_on=['unique_id','date'],right_index=True,how='left')
sales_stats = train_cp.groupby(['common_name','warehouse'])['sales'].agg(['mean','std'])
all_data['last_sales_zs'] = (all_data['last_sales_ema005'] - pd.MultiIndex.from_frame(all_data[['common_name','warehouse']]).map(
    sales_stats['mean']))/ pd.MultiIndex.from_frame(all_data[['common_name','warehouse']]).map(sales_stats['std'])

# Cutting all data prior to 2022 seems to help. This could be due to COVID effects, and also the fact that there is little data from the Germany warehouses before 2022.
X_train = X_train[X_train['date'] >= '2022-01-01']
y_train = y_train.loc[X_train.index]
X_train_weights = X_train_weights.loc[X_train.index]

X_train[add_cols] = all_data[add_cols]
test[add_cols] = all_data[add_cols]
all_data[cat_cols] = all_data[cat_cols].astype('str').astype('category')


lr = .1
es = 10
n_est = round(5000/lr)
seed = 2
base_params = {
    'n_estimators':n_est
    ,'learning_rate':lr
    ,'verbosity':0
    ,'enable_categorical':True
    ,'early_stopping_rounds':es
    ,'random_state':seed
    ,'objective':'reg:squarederror'
    ,'eval_metric':'rmse'
    ,'device':'cuda'
    ,'reg_lambda':0
    ,'min_child_weight':1
}
kf_params = {
    'n_splits':3
    ,'n_repeats':1
    ,'random_state':seed
}


drop_cols = ['date','name','L1_category_name_en']
oof_preds = []
test_preds = []
pow_trans=True
pow_degree=.5

kf = RepeatedKFold(**kf_params)
X,y = deepcopy(X_train),deepcopy(y_train)
X[cat_cols] = all_data[cat_cols]
X.drop(drop_cols,axis=1,inplace=True)
test_copy = deepcopy(test)
test_copy[cat_cols] = all_data[cat_cols]
test_copy.drop(drop_cols,axis=1,inplace=True)
oof_pred_df = pd.DataFrame(index=X.index, columns=[
    'Pred_{0}'.format(i) for i in range(kf_params['n_repeats'])])
for i, (idx_t, idx_v) in enumerate(kf.split(X)):
    X_t, X_v = X.iloc[idx_t], X.iloc[idx_v]        
    y_t, y_v = y.loc[X_t.index], y.loc[X_v.index]
    if pow_trans:
        y_t, y_v = np.power(y_t, pow_degree), np.power(y_v, pow_degree)
    xgb = XGBRegressor(**base_params)
    xgb.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=100*es)
    model_test_preds = np.power(xgb.predict(test_copy).clip(0), 1/pow_degree) if pow_trans else xgb.predict(test_copy).clip(0)
    test_preds.append(model_test_preds)
    model_oof_preds = np.power(xgb.predict(X_v).clip(0), 1/pow_degree) if pow_trans else xgb.predict(X_v).clip(0)
    oof_pred_df.iloc[idx_v,int(i/kf_params['n_splits'])] = model_oof_preds
oof_preds.append(oof_pred_df)


oof_pred_df = pd.concat(oof_preds,axis=1)
test_pred_df = pd.DataFrame(np.transpose(test_preds), index=test.index)
oof_pred_vals = oof_pred_df.mean(axis=1)
np.round(mean_absolute_error(y_train, oof_pred_vals, sample_weight=X_train_weights), 3)


test_dm = DMatrix(test_copy, enable_categorical=True)
shap_values = xgb.get_booster().predict(test_dm, pred_contribs=True)
shap.summary_plot(shap_values[:,:-1], test_copy, max_display=40)


shap.dependence_plot('type_6_discount', shap_values[:,:-1], test_copy)


test_sub = test_pred_df.mean(axis=1)
test_sub.name = 'sales_hat'
# print(test_sub.sort_values(axis=0).head())
# test_sub.to_csv('submission.csv')


test_with_preds = test[['unique_id', 'date']].copy()
test_with_preds['sales_hat'] = test_pred_df.mean(axis=1).values
test_with_preds.sort_values(['unique_id', 'date'], inplace=True)

alpha = 0.2
test_with_preds['sales_hat_smoothed'] = test_with_preds.groupby('unique_id')['sales_hat'].transform(
    lambda x: x.ewm(alpha=alpha, adjust=False).mean()
)

test_with_preds['sales_hat_smoothed'] = test_with_preds['sales_hat_smoothed'].clip(0)
print(test_with_preds.head())
# test_with_preds['sales_hat_smoothed'].to_csv('submission.csv')
test_with_preds['sales_hat_clipped'] = test_with_preds.groupby('unique_id')['sales_hat'].transform(
    lambda x: np.clip(x, 0, x.quantile(0.99))
)
print(test_with_preds)
test_with_preds['sales_hat_clipped'].to_csv('submission.csv')




