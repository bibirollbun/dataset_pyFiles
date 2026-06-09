import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


def generate_submission_file(res,suffix):
    """
    The function is mainly responsible for generating the final submission file
    which can be directly submitted to the competition. takes mainly the dataframe
    and the filename to be given in suffix
    """
    res = pd.DataFrame(res)
    res.rename(columns = {0:'num_sold'},inplace = True)
    res['id'] = id_1
    res = res[['id','num_sold']]
    file_name = 'submission_'+suffix+'.csv'
    res.to_csv(file_name,index = False)



train_data['Date'] = pd.to_datetime(train_data['date'])
train_data['Year'] = train_data['Date'].dt.year
train_data['Month'] = train_data['Date'].dt.month
train_data['day_name'] = train_data['Date'].dt.day_name()
train_data['day_of_week'] = train_data['Date'].dt.dayofweek
train_data['is_weekend'] = train_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
train_data['YearMonth'] =train_data['Date'].dt.to_period('M')  # Creates Year-Month period
monthly_totals = train_data.groupby(['YearMonth','country','store'])['num_sold'].sum().reset_index()


test_data['date'] = pd.to_datetime(test_data['date'])
test_data['Year'] = test_data['date'].dt.year
test_data['Month'] = test_data['date'].dt.month
test_data['day_name'] = test_data['date'].dt.day_name()
test_data['day_of_week'] = test_data['date'].dt.dayofweek
test_data['is_weekend'] = test_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)


train_data.head()



for i in monthly_totals['country'].unique():
    temp_df = monthly_totals[monthly_totals['country'] == i]
    temp_df = temp_df.sort_values(by = ['YearMonth'],ascending = False)
    temp_df['YearMonth'] = temp_df['YearMonth'].astype(str)
    plt.figure(figsize = (10,10))
    sns.lineplot(x = 'YearMonth',y = 'num_sold',data = temp_df.head(50),hue = 'store')
    title = 'Plot for country: '+i
    plt.title(title)
    plt.xticks(rotation = 45)
    plt.show()


yearly_totals = train_data.groupby(['Year','country','store'])['num_sold'].sum().reset_index()
for i in yearly_totals['country'].unique():
    temp_df = yearly_totals[yearly_totals['country'] == i]
    temp_df = temp_df.sort_values(by = ['Year'],ascending = False)
    temp_df['Year'] = temp_df['Year'].astype(str)
    plt.figure(figsize = (10,10))
    sns.lineplot(x = 'Year',y = 'num_sold',data = temp_df,hue = 'store')
    title = 'Plot for country: '+i
    plt.title(title)
    plt.xticks(rotation = 45)
    plt.show()


train_data.head()


weekly_df = train_data.groupby(['country','day_name','store'],as_index = False)['num_sold'].mean()

weekly_df.reset_index(inplace = True)
for i in weekly_df['country'].unique():
    temp_df = weekly_df[weekly_df['country'] == i]
    sns.barplot(x = 'day_name',y = 'num_sold',hue = 'store',data = temp_df)
    plot_title = 'Analysis for country: '+i
    plt.title(plot_title)
    plt.show()


weekly_df.head()


train_data = train_data.merge(monthly_totals,how = 'left',on = ['YearMonth','country','store'])
train_data = train_data.merge(yearly_totals,how = 'left',on = ['Year','country','store'])


train_data.rename(columns = {'num_sold_x':'num_sold_actual','num_sold_y':'num_sold_monthly','num_sold':'num_sold_yearly'},inplace = True)


train_data.head()


train_data['country'].unique()


store_map = {'Discount Stickers':0,'Stickers for Less':1,'Premium Sticker Mart':2}
product_map = {'Holographic Goose':0,'Kaggle':1,'Kaggle Tiers':2,'Kerneler':3,'Kerneler Dark Mode':4}

train_data['product'] = train_data['product'].map(product_map)
train_data['store'] = train_data['store'].map(store_map)


country_map = {'Canada':0,'Finland':1,'Italy':2,'Kenya':3,'Norway':4,'Singapore':5}
train_data['country'] = train_data['country'].map(country_map)


store_map = {'Discount Stickers':0,'Stickers for Less':1,'Premium Sticker Mart':2}
product_map = {'Holographic Goose':0,'Kaggle':1,'Kaggle Tiers':2,'Kerneler':3,'Kerneler Dark Mode':4}
country_map = {'Canada':0,'Finland':1,'Italy':2,'Kenya':3,'Norway':4,'Singapore':5}

test_data['product'] = test_data['product'].map(product_map)
test_data['store'] = test_data['store'].map(store_map)
test_data['country'] = test_data['country'].map(country_map)


train_data['num_sold_actual'] = train_data['num_sold_actual'].fillna(
    train_data.groupby(['country', 'product'])['num_sold_actual'].transform('mean')
)


train_data.head()


X = train_data.drop(['date','id','Date','YearMonth','num_sold_monthly','num_sold_yearly','num_sold_actual','day_name'],axis = 1)
y = train_data['num_sold_actual']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
mdl_xgb = XGBRegressor()
mdl_xgb.fit(X,y)


print(mdl_xgb.score(X_train,y_train))
mdl_xgb.score(X_test,y_test)


id_1 = test_data['id']

res = mdl_xgb.predict(test_data.drop(['id','date','day_name'],axis = 1))





generate_submission_file(res,'xgb_new_features1')


## Exploring Stacking Ensembles
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

stacked_model = StackingRegressor(
    estimators=[
        ('xgb', XGBRegressor()),
        ('rf', RandomForestRegressor())
    ],
    final_estimator=Ridge()
)

stacked_model.fit(X_train, y_train)



res_stack = stacked_model.predict(test_data.drop(['id','date','day_name'],axis = 1))
generate_submission_file(res_stack,'xgb_8Jan_stack')



def mape_metric(y_pred, dataset):
    y_true = dataset.get_label()
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return 'mape', mape, False

train_data_lgb = lgb.Dataset(X, label=y)
val_data = lgb.Dataset(X_test, label=y_test, reference=train_data_lgb)
params = {
    'objective': 'regression',
    'metric': 'l2',  # Default metric (you can replace it or add others)
    'boosting_type': 'gbdt',
    'num_leaves': 35,
    'learning_rate': 0.05,
    'n_estimators':200,
    'verbose': -1
}

# Train the model
model_lgb = lgb.train(
    params,
    train_data_lgb,
    num_boost_round=100,
    valid_sets=[train_data_lgb, val_data],
    valid_names=['train', 'val'],
    feval=mape_metric  # Add custom evaluation function
)


##0.9656
y_pred = model_lgb.predict(X_test)
final_mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"Final MAPE on validation set: {final_mape:.2f}%")


res_lgbm = model_lgb.predict(test_data.drop(['id','date','day_name'],axis = 1))


generate_submission_file(res_lgbm,'lgbm')




