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


import json
from pandas import json_normalize
def load_df(csv_path='../input/train_v2.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


import pandas as pd
df=load_df('/kaggle/input/ga-customer-revenue-prediction/train.csv')
df['totals.transactionRevenue']=df['totals.transactionRevenue'].fillna(0).astype('int')
df_test=load_df('/kaggle/input/ga-customer-revenue-prediction/test_v2.csv')


total_visitor_rev=df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

count_vis=len(total_visitor_rev[total_visitor_rev['totals.transactionRevenue']>0])
              
print("Number of unique visitor with some revenue",count_vis)
print("Number of unique visitor with some revenue",len(total_visitor_rev))
print( ' % of visitor generating revenue',count_vis/len(total_visitor_rev))


df.date=pd.to_datetime(df.date, format="%Y%m%d")
df['trans_month']=df.date.dt.month


df['month_year'] = df['date'].dt.strftime('%b-%Y')
df['transact']=np.where(df['totals.transactionRevenue']>0,1,0)


first_purchase = df[df['transact'] == 1].groupby('fullVisitorId')['sessionId'].min().reset_index()
first_purchase['is_first_purchase']=1



df = df.merge(first_purchase, on=['fullVisitorId','sessionId'], how='left')
df.groupby(df.month_year)["totals.transactionRevenue"].sum().reset_index()

df_mt=df.groupby('month_year').agg({
    'totals.transactionRevenue': 'sum',         # Sum values
    'transact': 'sum',         # Sum counts,
    'is_first_purchase':'sum',
    'fullVisitorId':  pd.Series.nunique    # Unique count of user_id
}).reset_index()




df_mt['%first_trans']=df_mt['is_first_purchase']/df_mt['transact']


df_mt['month_year_dt'] = pd.to_datetime(df_mt['month_year'], format='%b-%Y')

# Sort by month & year
df_mt = df_mt.sort_values(by='month_year_dt')

# Drop the datetime column if not needed
df_mt = df_mt.drop(columns=['month_year_dt'])

# Reset index
df_mt = df_mt.reset_index(drop=True)


df_mt



import plotly.graph_objects as go

# Create the figure
fig = go.Figure()

# Add Transaction Revenue (Primary Y-axis)
fig.add_trace(go.Bar(
    x=df_mt['month_year'],
    y=df_mt['totals.transactionRevenue'],
    name="Total Revenue",
    marker_color='blue',
    yaxis='y1'  # Primary Y-axis
))

# Add Number of Transactions (Secondary Y-axis)
fig.add_trace(go.Scatter(
    x=df_mt['month_year'],
    y=df_mt['transact'],
    name="Total Transactions",
    mode='lines+markers',
    marker=dict(color='red', size=8),
    yaxis='y2'  # Secondary Y-axis
))

# Add Unique Visitors (Secondary Y-axis)
fig.add_trace(go.Scatter(
    x=df_mt['month_year'],
    y=df_mt['fullVisitorId'],
    name="Unique Visitors",
    mode='lines+markers',
    marker=dict(color='green', size=8),
    yaxis='y2'
))

# Add New Users (Secondary Y-axis)
fig.add_trace(go.Scatter(
    x=df_mt['month_year'],
    y=df_mt['is_first_purchase'],
    name="New Users",
    mode='lines+markers',
    marker=dict(color='purple', size=8),
    yaxis='y2'
))

# Layout adjustments
fig.update_layout(
    title="Monthly Comparison of Revenue, Transactions, and Visitors",
    xaxis=dict(title="Month-Year"),
    
    yaxis=dict(
        title="Total Transaction Revenue",
        titlefont=dict(color="blue"),
        tickfont=dict(color="blue"),
        side="left"
    ),

    yaxis2=dict(
        title="Transactions / Visitors",
        titlefont=dict(color="red"),
        tickfont=dict(color="red"),
        overlaying="y",
        side="right"
    ),
    
    legend=dict(x=0.1, y=1.1),
    template="plotly_white"
)

# Show the figure
fig.show()






import plotly.express as px
df_grouped = df.groupby(['month_year', 'trafficSource.campaign'])['totals.transactionRevenue'].sum().reset_index()



# Create stacked bar chart
fig = px.bar(df_grouped, 
             x='month_year', 
             y='totals.transactionRevenue', 
             color='trafficSource.campaign', 
             title='Monthly Revenue Split by Campaign',
             labels={'revenue': 'Total Revenue', 'month_year': 'Month-Year'},
             barmode='stack')  # Stack bars by campaign

# Show the plot
fig.show()



df_grouped = df.groupby(['month_year', 'trafficSource.campaign'])['totals.transactionRevenue'].sum().reset_index()
df_grouped.columns=['month_year','trafficSource.campaign','revenue_by_campaign']
df_total = df.groupby(['month_year'])['totals.transactionRevenue'].sum().reset_index()
df_ana=pd.merge(df_grouped,df_total,on=['month_year'])
df_ana['%contri']=df_ana['revenue_by_campaign']/df_ana['totals.transactionRevenue']*100
df_ana[df_ana['trafficSource.campaign']!='(not set)']
df_ana['month_year'] = pd.to_datetime(df_ana['month_year'], format='%b-%Y')
df_ana = df_ana.sort_values(by='month_year')

# Filter out '(not set)' campaign
df_filtered = df_ana[df_ana['trafficSource.campaign'] != '(not set)']

# Create a stacked bar chart for % contribution
fig = px.bar(df_filtered, 
             x='month_year', 
             y='%contri', 
             color='trafficSource.campaign', 
             title='% Contribution in Monthly Revenue Split by Campaign',
             labels={'%contri': '% Contribution', 'month_year': 'Month-Year'},
             barmode='stack')

# Add a secondary y-axis for total transaction revenue
fig.add_trace(go.Scatter(
    x=df_filtered['month_year'],
    y=df_filtered['totals.transactionRevenue'],
    name="Total Revenue",
    mode='lines+markers',
    marker=dict(color='red', size=8),
    yaxis='y2'  # Secondary Y-axis
))

# Update layout for better readability
fig.update_layout(
    xaxis=dict(title='Month-Year', tickformat='%b-%Y'),
    yaxis=dict(title='% Contribution', side='left'),
    yaxis2=dict(title='Total Revenue', overlaying='y', side='right', showgrid=False),
    legend=dict(title='Campaign', orientation='h', yanchor='bottom', y=-0.3),
    margin=dict(l=50, r=50, t=50, b=50),
    height=600
)

# Show plot
fig.show()


import plotly.express as px
cols=['channelGrouping',  'device.browser',
       'device.operatingSystem', 'device.deviceCategory',
       'geoNetwork.continent', 'geoNetwork.subContinent', 'geoNetwork.country',
       'geoNetwork.region', 'geoNetwork.metro', 'geoNetwork.city',
       'geoNetwork.networkDomain', 'totals.hits', 'totals.pageviews',
       'totals.bounces', 'totals.newVisits', 'trafficSource.campaign',
       'trafficSource.source', 'trafficSource.medium', 'trafficSource.keyword',
       'trafficSource.isTrueDirect', 'trafficSource.referralPath']
for i in cols:
    df_t=df.groupby(i)['fullVisitorId'].count().reset_index()
    
    fig = px.bar(df_t, 
             x=i, 
             y='fullVisitorId', 
             title="Number of Records per "+i,
             labels={ 'fullVisitorId': 'Number of Records'},
             color='fullVisitorId', 
             color_continuous_scale='Viridis')

    
    fig.show()


one_hot_encoder=['channelGrouping','device.browser','device.operatingSystem', 'device.deviceCategory', 'geoNetwork.continent','trafficSource.medium', 'trafficSource.adwordsClickInfo.slot']
drop_col=['geoNetwork.subContinent', 'geoNetwork.country','visitStartTime',
       'geoNetwork.region', 'geoNetwork.metro', 'geoNetwork.city','trafficSource.referralPath',
       'geoNetwork.networkDomain','trafficSource.campaign','trafficSource.source', 'trafficSource.keyword',
         'trafficSource.adwordsClickInfo.page','sessionId','visitId','socialEngagementType',
       'device.browserVersion', 'device.browserSize','customDimensions','hits', 'totals.timeOnSite',
       'totals.sessionQualityDim', 'totals.transactions',
       'totals.totalTransactionRevenue',
       'device.operatingSystemVersion', 'device.mobileDeviceBranding',
       'device.mobileDeviceModel', 'device.mobileInputSelector',
       'device.mobileDeviceInfo', 'device.mobileDeviceMarketingName',
       'device.flashVersion', 'device.language', 'device.screenColors',
       'device.screenResolution', 'geoNetwork.cityId', 'geoNetwork.latitude',
       'geoNetwork.longitude', 'geoNetwork.networkLocation', 'totals.visits',
       'trafficSource.adwordsClickInfo.criteriaParameters',
       'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adContent', 'trafficSource.campaignCode']
const_cols = [c for c in df.columns if df[c].nunique(dropna=False)==1 ]


const_cols = [col for col in const_cols if col != 'tag']

drop_col=drop_col+const_cols

from sklearn.preprocessing import LabelEncoder
def transform_df(df):
    
    df.date=pd.to_datetime(df.date, format="%Y%m%d")
    df['trans_month']=df.date.dt.month
    df['device.browser']=df['device.browser'].str.replace('Safari (in-app)','Safari')
    df['device.browser']=df['device.browser'].str.replace('Apple-iPhone7C2','Safari')
    df['device.browser']=df['device.browser'].apply(lambda x: x if x in ['Chrome', 'Safari', 'Firefox', 'Internet Explorer'] else 'other')
    df['device.operatingSystem']=df['device.operatingSystem'].apply(lambda x: x if x in ['Windows','Macintosh','Android','iOS','Linux','Chrome OS'] else 'other') 
    df[['totals.hits','device.isMobile', 'totals.pageviews','totals.bounces', 'totals.newVisits','trafficSource.isTrueDirect']]=df[['totals.hits','device.isMobile', 'totals.pageviews','totals.bounces', 'totals.newVisits','trafficSource.isTrueDirect']].fillna(0).astype('int')
    df['trafficSource.adwordsClickInfo.isVideoAd']=df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True).astype('int')
    df['bounce_rate'] = df['totals.bounces'] / df['totals.hits']  
    df['pages_per_session'] = df['totals.pageviews'] / df['visitNumber']
    df['engagement_score'] = df['totals.hits'] + df['totals.pageviews']
    df['is_campaign']=np.where(df['trafficSource.campaign']== '(not set)',0,1)
    drop_col_existing = [col for col in drop_col if col in df.columns]
    
    df=df.drop(drop_col_existing,axis=1)
    #df=pd.get_dummies(df, columns=one_hot_encoder)

    for col in one_hot_encoder:
        df[col] = LabelEncoder().fit_transform(df[col])
   
    return df


df_test['tag']='test'
df['tag']='train'



df_base=pd.concat([df,df_test])
df_base=transform_df(df_base)


df_base[df_base['tag']=='train']['date']
df_base2 = df_base[df_base['tag']=='train'].sort_values(by='date')

split_index = int(len(df_base2) * 0.8)  # Get index for 80% split

# âœ… Step 3: Get the Split Date
split_date = df_base2.iloc[split_index]['date']
print(f"ðŸ“… Split Date for 80:20 Split: {split_date}")


X_train=df_base[(df_base['tag']=='train') & (df_base.date<'2017-05-11')].drop(['date','tag','totals.transactionRevenue','fullVisitorId', 'month_year'],axis=1)
y_train=np.log1p(df_base[(df_base['tag']=='train') & (df_base.date<'2017-05-11')]['totals.transactionRevenue'].astype(float))
X_val=df_base[(df_base['tag']=='train') & (df_base.date>='2017-05-11')].drop(['date','tag','totals.transactionRevenue','fullVisitorId', 'month_year'],axis=1)
y_val=np.log1p(df_base[(df_base['tag']=='train') & (df_base.date>='2017-05-11')]['totals.transactionRevenue'].astype(float))
X_test=df_base[(df_base['tag']=='test')].drop(['date','tag','totals.transactionRevenue','fullVisitorId', 'month_year'],axis=1)


import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
import logging
import optuna

# Suppress warnings
warnings.simplefilter("ignore")  # Ignores all warnings

# Reduce Optuna logging verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Reduce LightGBM logs
logging.getLogger("lightgbm").setLevel(logging.WARNING)
logging.getLogger("xgboost").setLevel(logging.WARNING)

# Function to optimize XGBoost
def objective_xgb(trial):
    params = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        "random_state": 42
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)
    
    rmse_val = np.sqrt(mean_squared_error(y_val, val_pred))
    rmse_train = np.sqrt(mean_squared_error(y_train, train_pred))

    return rmse_val + 0.1 * rmse_train  # Prioritizing validation RMSE, but considering training too

# Function to optimize LightGBM
def objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        "random_state": 422
    }

    model = lgb.LGBMRegressor(**params)
    
    # Early stopping via callbacks
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)

    rmse_val = np.sqrt(mean_squared_error(y_val, val_pred))
    rmse_train = np.sqrt(mean_squared_error(y_train, train_pred))

    return rmse_val + 0.1 * rmse_train  # Balancing validation and train RMSE

# Running Optuna for XGBoost
study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=20)  # Adjust trials as needed
best_xgb_params = study_xgb.best_params

# Running Optuna for LightGBM
study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=20)
best_lgb_params = study_lgb.best_params

# Train models with best parameters
xgb_model = xgb.XGBRegressor(**best_xgb_params)
xgb_model.fit(X_train, y_train)

lgb_model = lgb.LGBMRegressor(**best_lgb_params)
lgb_model.fit(X_train, y_train)

# Predictions
xgb_pred_train = xgb_model.predict(X_train)
xgb_pred_val = xgb_model.predict(X_val)
xgb_pred_test = xgb_model.predict(X_test)

lgb_pred_train = lgb_model.predict(X_train)
lgb_pred_val = lgb_model.predict(X_val)
lgb_pred_test = lgb_model.predict(X_test)

# RMSE
xgb_rmse_train = np.sqrt(mean_squared_error(y_train, xgb_pred_train))
xgb_rmse_val = np.sqrt(mean_squared_error(y_val, xgb_pred_val))
lgb_rmse_train = np.sqrt(mean_squared_error(y_train, lgb_pred_train))
lgb_rmse_val = np.sqrt(mean_squared_error(y_val, lgb_pred_val))

print(f"XGBoost RMSE - Train: {xgb_rmse_train:.4f}, Validation: {xgb_rmse_val:.4f}")
print(f"LightGBM RMSE - Train: {lgb_rmse_train:.4f}, Validation: {lgb_rmse_val:.4f}")




# import xgboost as xgb
# import lightgbm as lgb

# from sklearn.metrics import mean_absolute_error, mean_squared_error

# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
# xgb_model.fit(X_train, y_train)

# lgb_model = lgb.LGBMRegressor(n_estimators=100, min_data_in_leaf=10, min_split_gain=0.1, random_state=422)
# lgb_model.fit(X_train, y_train)
# xgb_pred_train = xgb_model.predict(X_train)
# lgb_pred_train = lgb_model.predict(X_train)

# xgb_pred_val = xgb_model.predict(X_val)
# lgb_pred_val = lgb_model.predict(X_val)

# xgb_pred_test = xgb_model.predict(X_test)
# lgb_pred_test = lgb_model.predict(X_test)

# def evaluate_model_log_rmse(y_true, y_pred, model_name):
#     # Apply log transformation (log1p to handle zero values)
#     # log_y_true = np.log1p(y_true)
#     # log_y_pred = np.log1p(y_pred)
#     # log_y_true=np.nan_to_num(log_y_true, nan=0)
#     # log_y_pred=np.nan_to_num(log_y_pred, nan=0)

#     # Compute RMSE
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     print(f"{model_name} - Log RMSE: {rmse:.4f}")

# # Evaluate models using log-transformed RMSE
# evaluate_model_log_rmse(y_val, xgb_pred_val, "XGBoost")
# evaluate_model_log_rmse(y_val, lgb_pred_val, "LightGBM")
# evaluate_model_log_rmse(y_train, xgb_pred_train, "XGBoost")
# evaluate_model_log_rmse(y_train, lgb_pred_train, "LightGBM")


df_sub=pd.merge(df_base[(df_base['tag']=='test')]['fullVisitorId'],pd.DataFrame(lgb_pred_test),left_index=True, right_index=True)
df_sub.columns=['fullVisitorId','PredictedLogRevenue']
df_sub["PredictedLogRevenue"] = np.expm1(df_sub["PredictedLogRevenue"])  
df_sub['PredictedLogRevenue']= df_sub['PredictedLogRevenue']#.astype(float).round(1)
df_sub['PredictedLogRevenue'].value_counts()
df_sub=df_sub.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
df_sub['PredictedLogRevenue']=np.log1p(df_sub['PredictedLogRevenue'])
df_sub.to_csv("submission.csv",index=False)

