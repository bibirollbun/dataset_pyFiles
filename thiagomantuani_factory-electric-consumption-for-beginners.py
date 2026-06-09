import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.signal
import statsmodels.api as sm
import datetime as dt
import math
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from statsmodels.tsa.filters.hp_filter import hpfilter
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from colorama import Fore, Style
import optuna
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor, plot_importance, log_evaluation
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import plot_tree, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import shap
from statsmodels.graphics.tsaplots import plot_pacf

from sklearn import set_config


plt.style.use('ggplot')
set_config(transform_output='pandas')


train = pd.read_csv("/kaggle/input/prediction-of-factory-electric-consumption/train_df.csv",parse_dates=['Date'])
test = pd.read_csv("/kaggle/input/prediction-of-factory-electric-consumption/test_df.csv",parse_dates=['Date'])
submission = pd.read_csv('/kaggle/input/prediction-of-factory-electric-consumption/submission.csv')


train.head(1)


train['Date'].describe()


if not train.isna().any().any():
    print('There are no missing values in train.')
if not test.isna().any().any():
    print('There are no missing values in test.')


if ((train.Date - train.Date.shift(1)).iloc[1:] == pd.to_timedelta(1, unit='h')).all():
    print('Train has a sample every 1 hour.')
if ((test.Date - test.Date.shift(1)).iloc[1:] == pd.to_timedelta(1, unit='h')).all():
    print('Test has a sample every 1 hour.')


initial_features = [ f for f in test.columns.tolist() if f not in ['Date']]
target = 'Electric_Consumption'


_, axs = plt.subplots(7, 1, sharex=True, figsize=(12, 12), constrained_layout=True)
for ax, col in zip(axs, initial_features + [target]):
    ax.set_title(f"{col}")
    ax.plot(train.Date, train[col], 
            label='train', color='lightblue' if col != target else 'blue')
    if col != target:
        ax.plot(test.Date, test[col], label='test', color='lightgreen')
  
    else:
        ax.text(pd.to_datetime('2024-09-01'), 3, '?', fontsize=72, color='orange')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)    

plt.show()


train['Factor_F'] = train['Factor_F'].clip(None,621.00)


index_min = train['Factor_A'].nsmallest(1).index
train.loc[index_min,'Factor_A'] = 0


_, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 4), constrained_layout=True)
for ax, col in zip(axs, ['Factor_A','Factor_F']):
    ax.set_title(f"{col}")
    ax.plot(train.Date, train[col], 
            label='train', color='lightblue' if col != target else 'b')      
    if col != target:
        ax.plot(test.Date, test[col], label='test', color='lightgreen')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)    

plt.show()


_, axs = plt.subplots(2,2, constrained_layout=True, sharex=False, figsize=(10,5))
ax = axs.ravel()
train.groupby(train.Date.dt.hour)[target].mean().plot(ax=ax[0])
ax[0].set_title('hourly consumption', fontweight='bold', fontsize=10)
ax[0].set_xticks(range(0,24))
ax[0].set_xlabel('Hour')

train.groupby(train.Date.dt.month)[target].mean().plot(ax=ax[1])
ax[1].set_title('monthly consumption', fontweight='bold',fontsize=10)
ax[1].set_xticks(range(1,13))
ax[1].set_xlabel('Month')

day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
train.groupby(train.Date.dt.dayofweek)[target].mean().plot(ax=ax[2])
ax[2].set_title('daily consumption', fontweight='bold',fontsize=10)
ax[2].set_xlabel('Day')
ax[2].set_xticks(range(len(day_names))) 
ax[2].set_xticklabels(day_names, rotation=45, ha='right')

train.groupby(train.Date.dt.isocalendar().week)[target].mean().plot(ax=ax[3])
ax[3].set_title('weekly consumption', fontweight='bold', fontsize=10)
plt.suptitle('Seasonal plots', fontsize=18, color='green', y=1.05, va='center');


earliest_time = train['Date'].min()
for df in [train,test]:    
    
    df['datetime'] = pd.to_datetime(df['Date'])    
    df['date'] = df['datetime'].dt.date    
    
    df['year'] = df['datetime'].dt.year
    df['datediff_in_days'] = (
        df['datetime']- earliest_time
    ).dt.days
    # dictionary with time features as keys
    # and min and max as values
    time_features = {
        'hour': [0, 23],
        'dayofweek': [0, 6],
        #'minute': [0,59],
        #'second': [0,59],
        'week': [1, 52],
        'month': [1, 12],
        'day': [1,365]
    }
    #df['part_of_day'] = pd.cut(df['datetime'].dt.hour, bins=[0, 6, 12, 18, 24], 
    #                           labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5,6]).astype(int)
    for col in time_features:
        if col=='week':
            df[col] = df['datetime'].dt.isocalendar().week.astype(np.int32)
        else:
            df[col] = getattr(df['datetime'].dt,col)
        
        
        ## sin and cosine features to capture the circular continuity
        col_min,col_max = time_features[col]
        angles = 2*np.pi*(df[col]-col_min)/(col_max-col_min+1)
        
        # add sin and cos
        df[col+'_sine'] = np.sin(angles).astype('float')
        df[col+'_cosine'] = np.cos(angles).astype('float')
    df['half_hour']    = df['hour']/12
    df['working'] = df['hour'].between(6,18).astype('int')  
    
not_feature_columns = ['datetime','date']  
train.drop(not_feature_columns,axis=1,inplace=True)
test.drop(not_feature_columns,axis=1,inplace=True)



train.shape, test.shape


try:
    for df in [train,test]:
        df.set_index('Date',inplace=True)
except Exception as e :
    print({e})


dt = DecisionTreeRegressor(max_depth=3)
dt.fit(train[initial_features], train[target]);

plt.figure(figsize=(16, 6))
plot_tree(dt, feature_names=initial_features, fontsize=7, impurity=False, filled=True, ax=plt.gca())
plt.show()



all_features = list(test.columns)


scores, oofs  = {}, {}
df_shap = pd.DataFrame(columns=train.drop(target, axis=1).columns)
df_shap_vl = pd.DataFrame()
shap_values_fold, shap_valid = None, None


SPLITS = 12
ts = TimeSeriesSplit(n_splits=SPLITS)
seed = 42

def cross_validate(estimator, label='', features = initial_features, fimp = False):
    lscores = []    

    global df_shap, df_shap_vl, shap_values_fold, shap_valid
    
    X = train.copy()
    y = X.pop(target)
    val_preds = np.zeros((len(X)))
    
    
    for fold, (trx_idx, val_idx) in enumerate(ts.split(X,y)):        
        X_train = X.iloc[trx_idx][features]
        y_train = y.iloc[trx_idx]
        X_val = X.iloc[val_idx][features]
        y_val = y.iloc[val_idx]

        model = clone(estimator)        
        model.fit(X=X_train,y=y_train) 
                  #eval_set=[(X_val, y_val)],
                  #callbacks=[log_evaluation(100)])
        y_pred = model.predict(X_val)
        val_preds[val_idx] += y_pred
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        lscores.append(rmse)
        verbose = False
        if verbose:
            print(f"Fold {fold:2}: RMSE={rmse}")
        if fimp and (fold+1==SPLITS):
            
            explainer = shap.TreeExplainer(model)
            X_val = X_val.reset_index(drop=True)
            shap_val = explainer.shap_values(X_val)    
            shap_values_fold = shap_val
            shap_valid = X_val.copy()
            
            df_shap_fold = pd.DataFrame(shap_val, columns=X_val.columns)
            if df_shap.empty:
                df_shap = df_shap_fold
                df_shap_vl = X_val.copy()
            else:
                df_shap = pd.concat([df_shap, df_shap_fold])  
                df_shap_vl = pd.concat([df_shap_vl, X_val])
        
            
    oofs[label] = val_preds                    
    scores[label] = lscores
    print(f"{Fore.GREEN}avg rmse={np.mean(lscores)}{Style.RESET_ALL}")

    


cross_validate(make_pipeline(
                    StandardScaler(),                    
                    Ridge()),'Ridge',all_features)


cross_validate(LGBMRegressor(n_estimators=85, verbose=0),'LGBM',all_features,True)


cross_validate(XGBRegressor(),'XGBRegressor',all_features,False)


cross_validate(HistGradientBoostingRegressor(random_state=2024),'HistGradient',all_features,False)


shap.summary_plot(shap_values_fold, shap_valid)


shap.summary_plot(df_shap.values, df_shap_vl, plot_type="bar")


def plot_shap():
    fig, ax = plt.subplots(6, 5, figsize=(25, 15))
    ax = ax.flatten()
    cat_features = []
    
    i = 0
    for _, f in enumerate(all_features):
        if f in cat_features:
            continue
        
        shap.dependence_plot(
             f, shap_values_fold, shap_valid,
                ax=ax[i], show=False)    
        i += 1

    fig.suptitle(f'SHAP, Partial Dependance', 
            x=0, horizontalalignment='left', fontsize=24)
    fig.subplots_adjust(bottom=0.1)
    plt.tight_layout()

    for axe in ax.ravel():
        if not axe.has_data():
            axe.axis('off')
plot_shap()


result_score = []
for label in scores.keys():
    result_score.append((label,np.mean(scores[label])))
ax = pd.DataFrame(result_score,columns=['model','score']).sort_values(by='score',ascending=False).set_index('model').plot(kind='barh',color='lightblue')
ax.bar_label(ax.containers[0],label_type='center');
if ax.get_legend() is not None:
    ax.get_legend().remove()


model = HistGradientBoostingRegressor(random_state=2024)
model.fit(train[all_features],
          train[target])
y_pred = model.predict(test[all_features])
submission['Electric_Consumption'] = np.clip(y_pred,train[target].min(),None)
submission.to_csv('submission.csv',index=False)
submission[target].hist();

