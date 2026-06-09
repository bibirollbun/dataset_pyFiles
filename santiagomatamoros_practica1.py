

import numpy as np
import pandas as pd



import os
for dirname, _, filenames in os.walk('D:\favorita'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




items = pd.read_csv("D:\favorita/items.csv")
holiday_events = pd.read_csv("D:\favorita/holidays_events.csv", parse_dates=['date'])
stores = pd.read_csv("D:\favorita/stores.csv")
oil = pd.read_csv("D:\favorita/oil.csv", parse_dates=['date'])
transactions = pd.read_csv("D:\favorita/transactions.csv", parse_dates=['date'])
train = pd.read_csv('D:\favorita/train.csv', parse_dates = ['date'])



train.head()


train.tail()


print("Nulls in Oil columns: {0} => {1}".format(oil.columns.values,oil.isnull().any().values))
print("="*70)
print("Nulls in holiday_events columns: {0} => {1}".format(holiday_events.columns.values,holiday_events.isnull().any().values))
print("="*70)
print("Nulls in stores columns: {0} => {1}".format(stores.columns.values,stores.isnull().any().values))
print("="*70)
print("Nulls in transactions columns: {0} => {1}".format(transactions.columns.values,transactions.isnull().any().values))
print("="*70)
print("Nulls in train columns: {0} => {1}".format(train.columns.values,train.isnull().any().values))



oil['dcoilwtico'] = oil['dcoilwtico'].interpolate()



oil['dcoilwtico'] = oil['dcoilwtico'].fillna(method='bfill')


oil.isnull().sum()


train.isnull().sum()


train['onpromotion'] = train['onpromotion'].fillna('False')


train.isnull().sum()


sales_oil = train.groupby("date")["unit_sales"].sum().reset_index()
sales_oil = sales_oil.merge(oil, on="date", how="left") 

plt.figure(figsize=(14, 6))


plt.plot(sales_oil["date"], sales_oil["unit_sales"], label="Total Unit Sales", color="blue", alpha=0.6)


plt.twinx()
plt.plot(sales_oil["date"], sales_oil["dcoilwtico"], label="Oil Prices", color="red", alpha=0.6)

plt.title("Daily Sales vs Oil Prices")
plt.xlabel("Date")
plt.legend(["Unit Sales", "Oil Prices"])
plt.show()


correlation = sales_oil["unit_sales"].corr(sales_oil["dcoilwtico"])
print(f"Correlation between oil price and sales: {correlation:.4f}")




sales_oil["oil_lag_7"] = sales_oil["dcoilwtico"].shift(7)  # Lag by 7 days
sales_oil["oil_lag_3"] = sales_oil["dcoilwtico"].shift(3)  # Lag by 3 days


correlation_lag_7 = sales_oil["unit_sales"].corr(sales_oil["oil_lag_7"])
correlation_lag_3 = sales_oil["unit_sales"].corr(sales_oil["oil_lag_3"])

print(f"Correlation with 7-day lag: {correlation_lag_7:.4f}")
print(f"Correlation with 3-day lag: {correlation_lag_3:.4f}")



store_sales = train.groupby("store_nbr")["unit_sales"].sum().reset_index()


store_sales = store_sales.sort_values(by="unit_sales", ascending=False)

print(store_sales.head())

plt.figure(figsize=(12, 6))
plt.bar(store_sales["store_nbr"], store_sales["unit_sales"], color="skyblue")
plt.title("Total Sales Per Store")
plt.xlabel("Store Number")
plt.ylabel("Total Sales")
plt.xticks(rotation=90)  # Rotate labels for better visibility
plt.show()


train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month

monthly_sales_by_year = train.groupby(["year", "month"])["unit_sales"].sum().reset_index()

import seaborn as sns

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sales_by_year, x="month", y="unit_sales", hue="year", palette="tab10")
plt.title("Monthly Sales Trend Across Years")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(range(1, 13))
plt.legend(title="Year")
plt.show()




train["year_month"] = train["date"].dt.to_period("M")  # Format: YYYY-MM


monthly_sales = train.groupby("year_month")["unit_sales"].sum().reset_index()


monthly_sales["year_month"] = monthly_sales["year_month"].astype(str)

plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_sales, x="year_month", y="unit_sales", marker="o", linestyle="-", color="b")
plt.title("Consecutive Monthly Sales Trend Over Years")
plt.xlabel("Year-Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.grid(True)
plt.show()


train_with_clusters = pd.merge(train, stores[['store_nbr', 'cluster']], on='store_nbr', how='left')
total_sales_by_cluster = train_with_clusters.groupby('cluster')['unit_sales'].sum().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(x='cluster', y='unit_sales', data=total_sales_by_cluster, palette='viridis')
plt.title('Total Sales by Store Cluster')
plt.xlabel('Cluster')
plt.ylabel('Total Sales')
plt.show()


daily_sales = train.groupby('date')['unit_sales'].sum().reset_index()
daily_sales['rolling_avg'] = daily_sales['unit_sales'].rolling(window=30).mean()


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(daily_sales['date'], daily_sales['unit_sales'], label='Daily Sales', alpha=0.5)
plt.plot(daily_sales['date'], daily_sales['rolling_avg'], label='30-Day Rolling Avg', color='red')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()



daily_sales['month'] = daily_sales['date'].dt.month


monthly_sales = daily_sales.groupby('month')['unit_sales'].mean().reset_index()


plt.figure(figsize=(10, 6))
plt.plot(monthly_sales['month'], monthly_sales['unit_sales'], marker='o')
plt.title('Monthly Seasonality in Sales')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()



daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek


weekly_sales = daily_sales.groupby('day_of_week')['unit_sales'].mean().reset_index()


plt.figure(figsize=(10, 6))
plt.plot(weekly_sales['day_of_week'], weekly_sales['unit_sales'], marker='o')
plt.title('Weekly Seasonality in Sales')
plt.xlabel('Day of Week')
plt.ylabel('Average Sales')
plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose


decomposition = seasonal_decompose(daily_sales.set_index('date')['unit_sales'], model='additive', period=365)


plt.figure(figsize=(12, 8))
decomposition.plot()
plt.show()



train['day_of_week'] = train['date'].dt.dayofweek  # Monday=0, Sunday=6
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train['week_of_year'] = train['date'].dt.isocalendar().week

train['is_weekend'] = train['day_of_week'].isin([5, 6]).astype(int)


train['sales_lag_7'] = train.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(7)

oil['oil_price_lag_7'] = oil['dcoilwtico'].shift(7)

transactions['transactions_lag_7'] = transactions['transactions'].shift(7)


train['sales_rolling_avg_7'] = train.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(lambda x: x.rolling(window=7).mean())

oil['oil_price_rolling_avg_7'] = oil['dcoilwtico'].rolling(window=7).mean()

transactions['transactions_rolling_avg_7'] = transactions['transactions'].rolling(window=7).mean()


holiday_events['is_holiday'] = holiday_events['type'].apply(lambda x: 1 if x in ['Holiday', 'Event'] else 0)

train = pd.merge(train, holiday_events[['date', 'is_holiday']], on='date', how='left')


del holiday_events


train['is_holiday'] = train['is_holiday'].fillna(0)


train = pd.merge(train, stores[['store_nbr', 'state', 'cluster']], on='store_nbr', how='left')

train = pd.merge(train, items[['item_nbr', 'family', 'perishable']], on='item_nbr', how='left')


del stores


del items



import gc

gc.collect()



train = pd.merge(train, oil[['date', 'dcoilwtico','oil_price_lag_7']], on='date', how='left')
# train = pd.merge(train, transactions[['date','transactions']], on='date', how='left')


del oil


del transactions


gc.collect()


train = train.iloc[7:].reset_index(drop=True)



train.head()


train.isnull().sum()



train = train.drop(columns=['id', 'year_month'])



train['dcoilwtico'] = train['dcoilwtico'].fillna(method='ffill')
train['oil_price_lag_7'] = train['oil_price_lag_7'].fillna(method='bfill')

# Fill missing values with the mean of the column
train['sales_lag_7'] = train['sales_lag_7'].fillna(train['sales_lag_7'].mean())
train['sales_rolling_avg_7'] = train['sales_rolling_avg_7'].fillna(train['sales_rolling_avg_7'].mean())


train.isnull().sum()


from sklearn.preprocessing import LabelEncoder

cat_features  = ['store_nbr', 'item_nbr', 'state', 'cluster', 'family', 'onpromotion', 'weekofyear']
label_encoders = {}

for col in cat_features:
    label_encoders[col] = LabelEncoder()
    train[col] = label_encoders[col].fit_transform(train[col])


from sklearn.preprocessing import StandardScaler

num_features = ["unit_sales", "dcoilwtico", "sales_rolling_avg_7", "oil_price_lag_7"]
scaler = StandardScaler()
train[num_features] = scaler.fit_transform(train[num_features])



train['onpromotion'] = train['onpromotion'].astype(bool)



train['week_of_year'] = train['week_of_year'].astype(int)




features = [col for col in train.columns if col not in ['unit_sales', 'date']]
X = train[features]
y = train['unit_sales']

print("Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())


from sklearn.model_selection import train_test_split


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")



xgb.plot_importance(model)
plt.show()


import numpy as np
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))


def create_sequences(data, timesteps=7):
    X_seq, y_seq = [], []
    for i in range(timesteps, len(data)):
        X_seq.append(data[i-timesteps:i])
        y_seq.append(data[i])
    return np.array(X_seq), np.array(y_seq)


timesteps = 7
X_seq, y_seq = create_sequences(X_scaled, timesteps)


X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)


print(f"LSTM Training data shape: {X_train_lstm.shape}, {y_train_lstm.shape}")
print(f"LSTM Validation data shape: {X_val_lstm.shape}, {y_val_lstm.shape}")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])


lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = lstm_model.fit(
    X_train_lstm, y_train_lstm,
    validation_data=(X_val_lstm, y_val_lstm),
    epochs=20,
    batch_size=32,
    verbose=1
)


plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM - Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


sale_day_item_level= pd.read_csv("../input/memory-optimization-data-manipulation/sale_day_item_level.csv")
sale_day_store_level= pd.read_csv("../input/memory-optimization-data-manipulation/sale_day_store_level.csv")
sale_store_item_level= pd.read_csv("../input/memory-optimization-data-manipulation/sale_store_item_level.csv")



sale_store_level=sale_day_store_level.groupby(['store_nbr'],as_index=False)['store_sales','item_variety'].agg(['sum'])


sale_store_level.columns = sale_store_level.columns.droplevel(1)
sale_store_level=sale_store_level.reset_index()
sale_store_level.head()



sale_item_level=sale_day_item_level.groupby(['item_nbr'],as_index=False)['item_sales'].agg(['sum'])

sale_item_level=sale_item_level.reset_index()
sale_item_level.head()



temp=sale_store_level.sort_values('store_sales',ascending=False).reset_index(drop=True)
temp=temp.set_index('store_nbr').head(10)

plt.figure(figsize=(12,8))
sns.barplot(temp.index,temp.store_sales, alpha=0.6, color='blue')
plt.ylabel('Overall Sales', fontsize=12)
plt.xlabel('Store Number', fontsize=12)
plt.title('Top Stores by Overall sale', fontsize=15)

plt.show()



temp1=sale_item_level.sort_values('sum',ascending=False).reset_index(drop=True)
temp1=temp1.set_index('item_nbr').head(10)
plt.figure(figsize=(12,8))
x=temp1.index.values
y=temp1['sum'].values
sns.barplot(x,y, alpha=0.6, color='purple')
plt.ylabel('Overall Sales', fontsize=12)
plt.xlabel('Store Number', fontsize=12)
plt.title('Top Items by Overall sale', fontsize=15)
plt.show()



temp=sale_day_store_level.groupby('Year')['store_sales'].sum()
plt.figure(figsize=(13,4))
sns.pointplot(temp.index,temp.values, alpha=0.8)
plt.ylabel('Overall Sales', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.title('Sale Year Over Year', fontsize=15)
plt.xticks(rotation='vertical')

plt.show()



temp=sale_day_store_level.groupby(['Year','Month']).aggregate({'store_sales':np.sum,'Year':np.min,'Month':np.min})
temp=temp.reset_index(drop=True)
sns.set(style="whitegrid", color_codes=True)

plt.figure(figsize=(15,8))
plt.plot(range(1,13),temp.iloc[0:12,0],label="2013")
plt.plot(range(1,13),temp.iloc[12:24,0],label="2014")
plt.plot(range(1,13),temp.iloc[24:36,0],label="2015")
plt.plot(range(1,13),temp.iloc[36:48,0],label="2015")
plt.ylabel('Overall Sales', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.title('Monthly sales variation', fontsize=15)
plt.xticks(rotation='vertical')
plt.legend(['2013', '2014', '2015', '2016'], loc='upper left')
plt.show()



plt.figure(figsize=(15,12))

plt.subplot(221)

temp = stores['cluster'].value_counts()

sns.barplot(temp.index,temp.values,color=color[5])
plt.ylabel('Count of stores', fontsize=12)
plt.xlabel('Cluster', fontsize=12)
plt.title('Store distribution across cluster', fontsize=15)

plt.subplot(222)

temp = stores['type'].value_counts()

sns.barplot(temp.index,temp.values,color=color[7])
plt.ylabel('Count of stores', fontsize=12)
plt.xlabel('Type of store', fontsize=12)
plt.title('Store distribution across store types', fontsize=15)

plt.subplot(223)

temp = stores['state'].value_counts()

sns.barplot(temp.index,temp.values,color=color[8])
plt.ylabel('Count of stores', fontsize=12)
plt.xlabel('state', fontsize=12)
plt.title('Store distribution across states', fontsize=15)
plt.xticks(rotation='vertical')

plt.subplot(224)

temp = stores['city'].value_counts()

sns.barplot(temp.index,temp.values,color=color[9])
plt.ylabel('Count of stores', fontsize=12)
plt.xlabel('City', fontsize=12)
plt.title('Store distribution across cities', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


sale_store_level=sale_store_level.iloc[:,0:2]

merge=pd.merge(sale_store_level,stores,how='left',on='store_nbr')



plt.figure(figsize=(15,12))

plt.subplot(221)

temp = merge.groupby(['cluster'])['store_sales'].sum()
#plot
sns.barplot(temp.index,temp.values,color=color[5])
plt.ylabel('Sales', fontsize=12)
plt.xlabel('Cluster', fontsize=12)
plt.title('Cumulative sales across store clusters', fontsize=15)

plt.subplot(222)

temp = merge.groupby(['type'])['store_sales'].sum()
#plot
sns.barplot(temp.index,temp.values,color=color[7])
plt.ylabel('sales', fontsize=12)
plt.xlabel('Type of store', fontsize=12)
plt.title('Cumulative sales across store types', fontsize=15)

plt.subplot(223)

temp = merge.groupby(['state'])['store_sales'].sum()
#plot
sns.barplot(temp.index,temp.values,color=color[8])
plt.ylabel('sales', fontsize=12)
plt.xlabel('state', fontsize=12)
plt.title('Cumulative sales across states', fontsize=15)
plt.xticks(rotation='vertical')

plt.subplot(224)

temp = merge.groupby(['city'])['store_sales'].sum()
#plot
sns.barplot(temp.index,temp.values,color=color[9])
plt.ylabel('sales', fontsize=12)
plt.xlabel('City', fontsize=12)
plt.title('Cumulative sales across cities', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


store_items=pd.merge(sale_store_item_level,items,on='item_nbr')
store_items=pd.merge(store_items,stores,on='store_nbr')
store_items['item_sales']=store_items['item_sales']

#item

top_items_by_type=store_items.groupby(['type','item_nbr'])['item_sales'].sum()
top_items_by_type=top_items_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])


top_items_by_type=top_items_by_type.groupby(['type']).head(5)



top_class_by_type=store_items.groupby(['type','class'])['item_sales'].sum()
top_class_by_type=top_class_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])


top_class_by_type=top_class_by_type.groupby(['type']).head(5)


top_family_by_type=store_items.groupby(['type','family'])['item_sales'].sum()
top_family_by_type=top_family_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])


top_family_by_type=top_family_by_type.groupby(['type']).head(5)


top_family_by_type=store_items.groupby(['type','family'])['item_sales'].sum()
top_family_by_type=top_family_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])
x=top_family_by_type.pivot(index='family',columns='type')
cm = sns.light_palette("orange", as_cmap=True)
x = x.style.background_gradient(cmap=cm)
x


top_items_by_type=store_items.groupby(['type','item_nbr'])['item_sales'].sum()
top_items_by_type=top_items_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])
top_items_by_type=top_items_by_type.groupby(['item_nbr']).head(20)
#print(top_items_by_type)
x=top_items_by_type.pivot(index='item_nbr',columns='type')
x['total']=x.sum(axis=1)
x=x.sort_values('total',ascending=False)
del(x['total'])
x=x.head(30)
cm = sns.light_palette("green", as_cmap=True)
x = x.style.background_gradient(cmap=cm,axis=1)
x


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipe_processing = Pipeline([
        ('prepare_data', prepare_data()),
        ('split_data', split_data()),
        ('process_data', process_data()),
        ('join_data', join_df())
    ])


data_df = pipe_processing.fit_transform([train_large, stores, oil, items, transactions, holiday_events])


X = data_df.drop(['unit_sales', 'transactions'], axis=1)
Y = data_df[['unit_sales', 'transactions']]


from sklearn.linear_model import LinearRegression,SGDRegressor,ElasticNet,Ridge
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error


def checkModelPerformane(model):
    model.fit(x_train.values, y_train.values)
    
    pred = model.predict(x_test.values)
    
    print("mean_squared_error: ",np.sqrt(mean_squared_error(y_test.values, pred))) 
    print("mean_absolute_error: ", np.sqrt(mean_absolute_error(y_test.values, pred)))


print("LinearRegression")
checkModelPerformane(LinearRegression())


print("Random Forest")
checkModelPerformane(RandomForestRegressor(random_state=42)) 


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]

forest_reg = RandomForestRegressor(random_state=42)
 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(x_train.values, y_train.values)


grid_search.best_params_


grid_search.best_estimator_


!head test.csv


final_model = grid_search.best_estimator_


test = pd.read_csv("../working/test.csv", parse_dates=['date'])

pipe_processing2 = Pipeline([
        ('split_data', split_data()),
        ('process_data', process_data()),
        ('join_data', join_df())
    ])

test_df = pipe_processing2.fit_transform(test)




test_df



# final_predictions = final_model.predict(test_x)

