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


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'iframe'

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


ross_df=pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv',low_memory=False)
test_df=pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')
store_df=pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')




ross_df



store_df


merged_df=ross_df.merge(store_df,how='left',on='Store')
merged_df


test_df


merged_test_df = test_df.merge(store_df, how='left', on='Store') 
merged_test_df


merged_df.info()


round(merged_df.describe().T,2)


merged_df.duplicated().sum()


merged_df['Date']=pd.to_datetime(merged_df.Date)


merged_test_df['Date']=pd.to_datetime(merged_test_df.Date)


merged_df.Date.min(), merged_df.Date.max() #train data from 2013-01-01 to 2015-07-31


merged_test_df.Date.min(), merged_test_df.Date.max() 


sns.histplot(data=merged_df, x='Sales')


merged_df.Open.value_counts()


merged_df.Open.value_counts()[0]


merged_df=merged_df[merged_df.Open==1].copy()


sns.histplot(data=merged_df, x='Sales')


plt.figure(figsize=(18,8))
temp_df = merged_df.sample(40000)
sns.scatterplot(x=temp_df.Sales, y=temp_df.Customers, hue=temp_df.Date.dt.year, alpha=0.8)
plt.title("Sales Vs Customers")
plt.show()


plt.figure(figsize=(18,8))
temp_df = merged_df.sample(10000)
sns.scatterplot(x=temp_df.Store, y=temp_df.Sales, hue=temp_df.Date.dt.year, alpha=0.8)
plt.title("Stores Vs Sales")
plt.show()


sns.barplot(data=merged_df, x='DayOfWeek', y='Sales',palette="husl")


sns.barplot(data=merged_df, x='Promo', y='Sales',palette="husl")



# Converting all numeric columns to float (ignoring errors)
merged_df_numeric = merged_df.apply(pd.to_numeric, errors='coerce')

# Checking for non-numeric columns (if any)
non_numeric_columns = merged_df.columns[merged_df.dtypes == 'object']
print("Non-numeric columns:", non_numeric_columns)

# Now calculating correlation, ignoring NaN values
correlation = merged_df_numeric.corr()['Sales'].sort_values(ascending=False)
print(correlation)


merged_df['Date']


merged_df['Day']=merged_df.Date.dt.day
merged_df['Month']=merged_df.Date.dt.month
merged_df['Year']=merged_df.Date.dt.year


merged_test_df['Day'] = merged_test_df.Date.dt.day
merged_test_df['Month'] = merged_test_df.Date.dt.month
merged_test_df['Year'] = merged_test_df.Date.dt.year


sns.barplot(data=merged_df, x='Year', y='Sales',palette="husl")


sns.barplot(data=merged_df, x='Month', y='Sales',palette="husl")


merged_df['Day']


len(merged_df)


train_size=int(.75*len(merged_df))
train_size


sorted_df=merged_df.sort_values('Date')
train_df,val_df=sorted_df[:train_size],sorted_df[train_size:]
len(train_df),len(val_df)


train_df


val_df


train_df.Date.min(), train_df.Date.max()


val_df.Date.min(), val_df.Date.max()


merged_test_df.Date.min(), merged_test_df.Date.max()


train_df.columns


input_cols=['Store', 'DayOfWeek', 'Promo',
       'StateHoliday',  'StoreType', 'Assortment',
       'Day', 'Month', 'Year']



target_cols='Sales'


merged_df[input_cols].nunique()


train_inputs,train_targets=train_df[input_cols].copy(),train_df[target_cols].copy()
val_inputs,val_targets=val_df[input_cols].copy(),val_df[target_cols].copy()


test_inputs=merged_test_df[input_cols].copy()
# Test data does not have targets


numeric_cols= ['Store', 'Day', 'Promo','Month', 'Year']


categorical_cols=['DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment']


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='mean').fit(train_inputs[numeric_cols])



train_inputs[numeric_cols]=imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])


test_inputs[numeric_cols].isna().sum()


from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler().fit(train_inputs[numeric_cols])


train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


train_inputs


from sklearn.preprocessing import OneHotEncoder


encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore').fit(train_inputs[categorical_cols])
encoded_cols=list(encoder.get_feature_names_out(categorical_cols))


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


train_inputs


X_train=train_inputs[numeric_cols+encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


X_train


merged_df.Sales.mean()


def return_mean(input):
    return np.full(len(input),merged_df.Sales.mean())


train_preds = return_mean(X_train)
train_preds


from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(train_preds, train_targets))


def rmspe(target, pred):
    target = np.array(target)
    pred = np.array(pred)
    
    # Filter out zero values in target
    non_zero_mask = target != 0
    target = target[non_zero_mask]
    pred = pred[non_zero_mask]
    
    # Calculate RMSPE
    rmspe_value = np.sqrt(np.mean(((target - pred) / target) ** 2)) * 100
    return f"{rmspe_value:.2f}%"



rmspe(train_targets,train_preds)


np.sqrt(mean_squared_error(return_mean(X_val), val_targets)),rmspe(train_targets,train_preds)



def try_model(model):
    # Fit the model
    model.fit(X_train,train_targets)

    # Generate predictions
    train_preds=model.predict(X_train)
    val_preds=model.predict(X_val)

    # Compute RMSE
    train_rmse=np.sqrt(mean_squared_error(train_preds,train_targets))
    val_rmse =np.sqrt(mean_squared_error(val_targets, val_preds))
    return train_rmse,val_rmse
    


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor


try_model(LinearRegression())


try_model(Ridge())


try_model(Lasso())


try_model(ElasticNet())


try_model(SGDRegressor())


from sklearn.tree import DecisionTreeRegressor, plot_tree


tree=DecisionTreeRegressor(random_state=42)
try_model(tree)


plt.figure(figsize=(40, 20))
plot_tree(tree, max_depth=3, filled=True, feature_names=numeric_cols+encoded_cols);


%%time
#Let's try a random forest.

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
try_model(rf)


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

# Initialize models
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
ridge_model = Ridge()
dt_model = DecisionTreeRegressor(random_state=42)

# Train models separately
rf_model.fit(X_train, train_targets)
ridge_model.fit(X_train, train_targets)
dt_model.fit(X_train, train_targets)

# Predict using all three models
rf_preds = rf_model.predict(X_val)
ridge_preds = ridge_model.predict(X_val)
dt_preds = dt_model.predict(X_val)

# Stacked Ensemble with Weighted Averaging
ensemble_preds = (0.6 * rf_preds + 0.3 * dt_preds + 0.1 * ridge_preds)

# Calculate RMSE for each model and ensemble
rf_rmse,rf_rmspe = np.sqrt(mean_squared_error(val_targets, rf_preds)),rmspe(val_targets, rf_preds)
ridge_rmse,ridge_rmspe = np.sqrt(mean_squared_error(val_targets, ridge_preds)),rmspe(val_targets, ridge_preds)
dt_rmse,dt_rmspe = np.sqrt(mean_squared_error(val_targets, dt_preds)),rmspe(val_targets, dt_preds)
ensemble_rmse, ensemble_rmspe= np.sqrt(mean_squared_error(val_targets, ensemble_preds)),rmspe(val_targets, ensemble_preds)

print("RandomForest RMSE & RMSPE:", rf_rmse,rf_rmspe)
print("Decision Tree RMSE & RMSPE:", dt_rmse,dt_rmspe)
print("Ridge RMSE & RMSPE:", ridge_rmse,ridge_rmspe)
print("Stacked Ensemble RMSE & RMSPE:", ensemble_rmse,ensemble_rmspe)


# Sorting Feature Importances in Descending Order for Each Model
rf_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
dt_importances = pd.Series(dt_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
ridge_importances = pd.Series(np.abs(ridge_model.coef_), index=X_train.columns).sort_values(ascending=False)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Feature Importance of Stacked Ensemble Components (Descending Order)', fontsize=16)

# RandomForest
axes[0].barh(rf_importances.index, rf_importances.values)
axes[0].set_title('RandomForest Feature Importance')
axes[0].invert_yaxis()

# DecisionTree
axes[1].barh(dt_importances.index, dt_importances.values)
axes[1].set_title('DecisionTree Feature Importance')
axes[1].invert_yaxis()

# Ridge
axes[2].barh(ridge_importances.index, ridge_importances.values)
axes[2].set_title('Ridge Feature Importance')
axes[2].invert_yaxis()

plt.tight_layout()
plt.show()


# Predict on X_test using all three models
rf_test_preds = rf_model.predict(X_test)
dt_test_preds = dt_model.predict(X_test)
ridge_test_preds = ridge_model.predict(X_test)

# Apply the Stacked Ensemble (same weights)
test_preds = (0.6 * rf_test_preds + 0.3 * dt_test_preds + 0.1 * ridge_test_preds)

# Display first 10 predictions
print("Stacked Ensemble Test Predictions (First 10):")
print(test_preds[:10])


submission_df=pd.read_csv("/kaggle/input/rossmann-store-sales/sample_submission.csv")

submission_df['Sales'] = test_preds * test_df['Open'].astype('float')
submission_df.fillna(0, inplace=True)
submission_df.to_csv('submission.csv', index=None)



submission_df.head()

