import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


base_path="/kaggle/input/rohlik-sales-forecasting-challenge-v2/"
train=pd.read_csv(f'{base_path}sales_train.csv' ,parse_dates=['date'])
inventory=pd.read_csv(f'{base_path}inventory.csv')
test=pd.read_csv(f'{base_path}sales_test.csv',parse_dates=['date'])
calender_extended=pd.read_csv(f'{base_path}calendar.csv',parse_dates=['date'])


train=train.drop(columns=['availability'])
train.dropna(subset=['sales'],inplace=True)


test['sales']=0
df = pd.concat([train, test], ignore_index=True).sort_values('date')
df = df.merge(calender_extended, on=['date', 'warehouse'], how='left')
df = df.merge(inventory, on=['unique_id', 'warehouse'], how='left')
df['date'] = pd.to_datetime(df['date'])

del train, test



df.tail()


PERIODS = [14,16,18,21,30,60,90,120,180,270,350,600,1000]



def add_date_features(df):
    df['date_month'] = df['date'].dt.month
    df['date_year'] = df['date'].dt.year
    df['date_day'] = df['date'].dt.day
    df['date_weekofyear'] = df['date'].dt.isocalendar().week
    df['date_weekday'] = df['date'].dt.weekday 
    df['date_dayofyear'] = df['date'].dt.dayofyear
    df['date_year_sin'] = np.sin((df['date_year'] - df['date_year'].min()) / (df['date_year'].max() - df['date_year'].min()) * 2 * np.pi)
    df['date_month_sin'] = np.sin(df['date_year'] / 1 * 2 * np.pi)
    df['date_month_cos'] = np.cos(df['date_month'] / 12 * 2 * np.pi)
    return df

def add_product_category(df):
    df['category'] = df['name'].str.split('_',expand=True)[0]
    return df


def add_lagged_product_sales(df, periods):
    # Sort by warehouse, name, and date for correct time order
    df = df.sort_values(by=['warehouse', 'name', 'date'])
    
    # Iterate over each defined period (e.g., 14, 30, 90 days)
    for shift in periods:
        # Create a new column for each period's lagged sales using shift
        df[f'product_sales_{shift}'] = np.nan
        
        # Group by 'warehouse' and 'name', then apply shift for each group
        df[f'product_sales_{shift}'] = df.groupby(['warehouse', 'name'])['sales'] \
            .shift(periods=shift)
        
    return df



df = add_date_features(df)
df = add_product_category(df)
df = add_lagged_product_sales(df,PERIODS)



## Simple sanity check: product_sales_feature

df[(df.name=='Croissant_36')&(df.warehouse=="Brno_1")][['date','warehouse','name','sales','product_sales_14']].head(20)



## Ensuring correct datatypes
for col in df.select_dtypes("object").columns:
    df[col] = df[col].astype('category')


## Splitting of datasets

train_start_date  = '2020-08-01'
train_end_date  = '2024-01-01'

train = df[(df['date'] >= train_start_date) & (df['date'] <= train_end_date)]
test  = df[(df['date'] >  train_end_date)]

X_train = train.drop(['sales', 'date'], axis=1)
y_train = train['sales']

X_test = test.drop(['sales', 'date'], axis=1)





## Parameter from here: https://www.kaggle.com/code/meryentr/rohlik-sales-lightgbm-lb-20-75

params={ 
    'learning_rate': 0.021796506746095975,
    'num_leaves': 93,
    'max_depth': 10,
    'min_child_samples': 25,
    'subsample': 0.7057135664023435,
    'colsample_bytree': 0.8528497905459008,
    'reg_alpha': 0.036786449788597686,
    'reg_lambda': 0.3151110021900479,
    'num_boost_round': 9800,
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
}


%%time
import lightgbm as lgb
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)



feature_importance_df = pd.DataFrame({'name':model.feature_name_})
feature_importance_df['importance'] = model.feature_importances_
feature_importance_df['group'] = feature_importance_df['name'].apply(
    lambda x: 'lagged sales features' if 'product_sales_' in x else 'date features' if 'date_' in x else 'other features')
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 20))
ax = sns.barplot(data=feature_importance_df, x='importance', y='name',  hue='group', dodge=False)
ax.set_title(f"Feature importances")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()


y_pred = model.predict(X_test)
y_pred[y_pred<0]=0


## Creation of submission dataset

test['id'] = test['unique_id'].astype(str) + "_" + test['date'].astype(str)
test['sales_hat'] = y_pred
test[['id','sales_hat']].to_csv("submission1.csv",index=False)

