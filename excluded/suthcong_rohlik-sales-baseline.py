!pip -q install calplot


import numpy as np 
import pandas as pd
import datetime as dt
import seaborn as sns
from colorama import Style, Fore
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, SplineTransformer, FunctionTransformer
from category_encoders import OneHotEncoder, TargetEncoder
from datetime import datetime
from lightgbm import LGBMRegressor
from scipy.optimize import differential_evolution, minimize
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline, Pipeline
import gc
from scipy.signal import periodogram
from scipy.stats import kurtosis
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.base import clone, BaseEstimator, TransformerMixin
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import PredictionErrorDisplay, mean_absolute_error
import os
from sklearn.kernel_approximation import Nystroem
import plotly_express as px
import calplot
from sklearn import set_config
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
set_config(transform_output='pandas')
plt.style.use('ggplot')


sales_train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv',parse_dates=['date'])
sales_test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv',parse_dates=['date'])
inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv',parse_dates=['date'])
test_weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')
solution = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')


sales_train = pd.merge(sales_train, inventory, how='left', on =['unique_id','warehouse'])
sales_test = pd.merge(sales_test, inventory, how='left', on =['unique_id','warehouse'])


sales_train = pd.merge(sales_train, calendar, how='left', on =['date','warehouse'])
sales_test = pd.merge(sales_test, calendar, how='left', on =['date','warehouse'])


for df in [sales_train,sales_test]:
    df.set_index('date',inplace=True)


np.setdiff1d(sales_train.columns,sales_test.columns)


sales_train.drop(['availability'], axis=1, inplace=True)
sales_train.sort_values(['date','warehouse'],inplace=True)


result = sales_train.reset_index().groupby(['warehouse']).agg(
    count = ('date','size'),
    first_date = ('date','min'),
    last_date = ('date','max'),    
    date_difference=('date', lambda x: x.max() - x.min()),
    var_sales = ('sell_price_main','var'),
    mean_price = ('sell_price_main','mean'),
    skew_price = ('sell_price_main','skew'),    
    max_price = ('sell_price_main','max'),
    kurtosis_sales = ('sell_price_main',kurtosis)
)
result


result = sales_test.reset_index().groupby(['warehouse']).agg(
    count = ('date','size'),
    first_date = ('date','min'),
    last_date = ('date','max'),    
    date_difference=('date', lambda x: (x.max() - x.min()).days+1)
)
result


del result


u_warehouses = sales_train['warehouse'].unique()

for  w in u_warehouses:
    missing = pd.date_range(start=sales_train.loc[sales_train.warehouse==w].index.min(),end=sales_train.loc[sales_train.warehouse==w].index.max()).difference(sales_train.loc[sales_train.warehouse==w].index)
    if missing.size>0:
        print(f'{Style.BRIGHT}{Fore.BLUE}**{w}**{Style.RESET_ALL}')
        first_date = sales_train.loc[sales_train.warehouse==w].index.min().strftime("%Y-%m-%d")
        last_date  = sales_train.loc[sales_train.warehouse==w].index.max().strftime("%Y-%m-%d")        
        print(f'{Style.BRIGHT}{Fore.YELLOW} Missing Dates-> {Style.RESET_ALL}{pd.date_range(start=sales_train.loc[sales_train.warehouse==w].index.min(),end=sales_train.loc[sales_train.warehouse==w].index.max()).difference(sales_train.loc[sales_train.warehouse==w].index)}\n')
    


sales_train[sales_train.sales.isnull()]['warehouse'].value_counts()


cat_cols = list(sales_test.select_dtypes(include='O'))
num_cols = list(sales_test._get_numeric_data())
target = 'sales'
initial_features = list(sales_test.columns)


for c in cat_cols:    
    A = sales_train[c].fillna('None').astype(str).unique()
    B = sales_test[c].fillna('None').astype(str).unique()
    C = np.setdiff1d(B,A)
    if C.size>0:
        print(C)
        sales_train.iloc[~sales_train[c].isin(C), c ] = 'None'
    sales_train[c] = sales_train[c].astype('category')
    sales_test[c] = sales_test[c].astype('category')    


%%time
fig,axs= plt.subplots(11,2, figsize=(15,25),  constrained_layout=True)
for c, ax in zip(initial_features,axs.ravel()):
    if sales_train[c].dtype=='float':
        ax.hist(sales_train[c],color='blue')        
    elif sales_train[c].dtype=='category':
        vc = sales_train[c].value_counts() / len(sales_train)
        ax.bar(vc.index,vc, color='brown')
        ax.yaxis.set_major_formatter('{x:.0%}')
        if len(vc)<=15:
            ax.set_xticks(np.arange(len(sales_train[c].dtype.categories)), sales_train[c].dtype.categories)
            ax.set_xticklabels(ax.get_xticklabels(),rotation=0)
        else:
            ax.set_xticks([])
    elif sales_train[c].dtype=='int64':        
        ax.hist(sales_train[c],color='green')        
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.set_title(f'{c}', fontweight='bold')
#axs.flat[-1].set_visible(False)
plt.suptitle('Features', y=1.03, fontsize=25);



plt.figure(figsize=(12, 8))
for i, (combi, df) in enumerate(sales_train.groupby(['warehouse'],observed=False)):
    ax = plt.subplot(3, 3, i+1)
    ax.hist(df.sales, bins=20, color='blue')        
    #ax.set_xscale('log')    
    ax.set_title(combi)
plt.suptitle('Histograms of sales', y=1.03)
plt.tight_layout(h_pad=3.0)
plt.show()


plt.figure(figsize=(18,22))
for i, (comb, df) in enumerate(sales_train.groupby(['warehouse'],observed=False)):
    ax = plt.subplot(7,3, i+1)        
    sales = df.sort_values(by='date').groupby('date')['sales'].sum().reset_index()
    trend = (sales.date - sales.iloc[0].date) // dt.timedelta(days=1)
    trend = trend.values.reshape(-1,1)
    model = make_pipeline(PolynomialFeatures(degree=2),
                          LinearRegression())
    model.fit(trend,sales.sales)
    y_pred = pd.Series(model.predict(trend), index=sales.date)
    
    ax.plot(sales.date,sales.sales,label='sales', color='black',marker='o',ls='--',markersize=1)
    y_pred.plot(ax=ax,color='red',label='trend')
    ax.set_title(comb)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)    
    ax.legend()   
    
    
plt.tight_layout()
plt.suptitle('sales over time by warehouse',fontsize=20,y=1.02)
plt.show()
    


plt.figure(figsize=(18,10))
for i, (comb, df) in enumerate(sales_train.groupby(sales_train.warehouse,observed=False)):    
    ax = plt.subplot(4, 3, i+1, ymargin=0.5)
    
    resampled = df.resample('MS').sales.sum()
    colors = ['blue'] * len(resampled)
    ax.set_title(comb)
    ax.set_ylim(resampled.min(), resampled.max())
    ax.bar(range(len(resampled)), resampled)
    ax.set_xticks(range(0, 48, 12), [f"Jan {y}" for y in range(2020, 2024)])        
    ax.bar(range(len(resampled)), resampled, color=colors)
    
plt.suptitle('Monthly sales for 2020-2024', y=1.03)
plt.tight_layout(h_pad=3.0)
plt.show()    



plt.figure(figsize=(18,10))
for i, (comb, df) in enumerate(sales_train.groupby(sales_train.warehouse,observed=False)):    
    ax = plt.subplot(4, 3, i+1, ymargin=0.5)
    
    resampled = df.resample('YS')[['sales']].mean().reset_index()
    ax.bar(resampled.date.dt.year, resampled.sales, color='brown')
    ax.set_title(comb)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # only integer labels
    #ax.set_ylim(0, resampled.sales.max())   

    
plt.suptitle('Monthly sales for 2020-2024', y=1.03)
plt.tight_layout(h_pad=3.0)
plt.show()    



plt.figure(figsize=(18,12))
for i, (comb, df) in enumerate(sales_train.groupby('warehouse',observed=False)):
    ax = plt.subplot(4,2, i+1)
    resampled = df.sort_values(by='date').groupby(df.index.month)['sales'].sum()
    colors = ['b'] * 5 + ['r'] * 2 + ['b'] * 5
    ax.bar(range(1, 13), resampled, color= colors)
    ax.set_xticks(ticks=range(1, 13), labels='JFMAMJJASOND')
    ax.set_title(comb)
    #ax.set_ylim(resampled.min(), resampled.max())
        
    
plt.suptitle('Monthly sales', y=1.03)
plt.tight_layout(h_pad=3.0)
plt.show()
    


%%time
yearp = sales_train.groupby(['warehouse',sales_train.index.year],observed=True)['sales'].mean().reset_index()
plt.figure(figsize=(12,11))
for i,w in enumerate(u_warehouses):
    ax = plt.subplot(len(u_warehouses),1,i+1)
    ax.plot(yearp[yearp.warehouse==w]['date'],yearp[yearp.warehouse==w]['sales'])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(w)
plt.tight_layout()
plt.show()


mean_sales = sales_train.groupby(['shops_closed','warehouse'],observed=False)['sales'].mean().reset_index().sort_values(by='sales')
#pivot_data = mean_sales.pivot(index='shops_closed', columns='warehouse', values='sales').fillna(0)
#pivot_data.plot(kind='bar',stacked=True)
_, axs = plt.subplots(1,2,figsize=(8,3), constrained_layout=True)
ax = axs.ravel()
sns.barplot(data=mean_sales, x='shops_closed', y='sales', hue='warehouse', errorbar=None, ax=ax[0])
ax[0].legend(bbox_to_anchor=[1,1])
sales_train.groupby(['shops_closed'])['sales'].mean().plot(kind='bar',ax=ax[1],color='green')
ax[1].bar_label(ax[1].containers[0])
plt.suptitle('sales by shops_closed');


tmp =sales_train.groupby(['warehouse','holiday'], observed=False).agg(
    sales_mean = ('sales','mean'),
    sales_min  = ('sales','min'),
    sales_max  = ('sales','max')
)
ax = tmp.unstack(level='warehouse')[['sales_mean']].plot(kind='bar', stacked=True)
ax.legend(bbox_to_anchor=[1,1]);
plt.title('sales by holiday');


from IPython.display import display


crosstab1 = pd.crosstab(index=sales_train['shops_closed'], 
                         columns=sales_train['holiday'], 
                         margins=False)

styled_crosstab1 = crosstab1.style.background_gradient(axis=0, cmap='YlOrRd')
styled_crosstab2 = pd.crosstab(index=[sales_train['shops_closed'], sales_train['warehouse']], 
                        columns=sales_train['holiday'], 
                        margins=False).style.background_gradient(axis=0, cmap='coolwarm')

display(styled_crosstab1)



display(styled_crosstab2)


sales_train.loc[sales_train.sales.isnull(),:].reset_index().groupby(['warehouse'],observed=False). \
agg(size=('warehouse','size'),
    min_date=('date','min'),
    max_date=('date','max'),
    days = ('date', lambda x: x.max() - x.min()),
    split_date=('date', lambda x: list(np.unique(np.unique(x.dt.strftime('%Y-%m-%d'))))) 
   ).dropna()


sales_train['sales'] = sales_train['sales'].fillna(0)
sales_train['total_orders'] = sales_train['total_orders'].fillna(0)
sales_train['sell_price_main'] = sales_train['sell_price_main'].interpolate()


def plot_periodogram(serie,wh,ax=None):     
    fs = pd.Timedelta('365D') / pd.Timedelta('1D')
    freq, spec = periodogram(serie, fs=fs,detrend='linear',scaling='spectrum')
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))  
    ax.step(freq, spec, color="blue")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title(f"Periodogram {wh}")

for u in u_warehouses:
    plot_periodogram(sales_train[sales_train.warehouse==u].groupby('date').sales.mean(),u)


#fig = calplot.calplot(sales_train.query('holiday==1').sales, how = "sum", cmap='jet')


weight_map = test_weights.set_index('unique_id')['weight'].to_dict()


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


class DropColsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,cols):
        self.cols = cols
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return X.drop(self.cols,axis=1)



class CreateTimeFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()                
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['weekday'] = df.index.weekday
        df['week'] = df.index.isocalendar().week
        df['weekend'] = df.index.weekday // 5
        df['semiweekly'] = np.where(df.index.weekday <3,0,1)    
        df['year_sin'] = np.sin(df['year'] / 1 * 2 * np.pi)
        df['year_cos'] = np.cos(df['year'] / 1 * 2 * np.pi)
        df['month_sin'] = np.sin(df['month'] / 12 * 2 * np.pi)
        df['month_cos'] = np.cos(df['month'] / 12 * 2 * np.pi)
        

        return df
    


ctf = CreateTimeFeatures()
sales_train = ctf.fit_transform(sales_train).copy()  
sales_test = ctf.fit_transform(sales_test).copy() 

my_index = sales_train.index
my_index_ts = sales_test.index

agg_df = sales_train.reset_index().groupby(['name'],observed=False).agg(days_in_sale=('date','nunique'),
                                                                   purchase_interval=('date',lambda x: (x.max() - x.min()).days)                                                               
                                                                  ).reset_index()
sales_train = sales_train.merge(agg_df[['name', 'days_in_sale', 'purchase_interval']], on='name', how='left')
sales_test = sales_test.merge(agg_df[['name', 'days_in_sale', 'purchase_interval']], on='name', how='left')
sales_train.set_index(my_index,inplace=True)
  
sales_train.loc[:,'date_diff'] = sales_train.reset_index().groupby('name',observed=False)['date'].diff().dt.days.fillna(0).reset_index()['date'].values
sales_train['gap'] = sales_train['date_diff'] > 1
sales_train['gap_group'] = sales_train.groupby(['name'],observed=False)['gap'].cumsum()
agg_df = sales_train.groupby(['name', 'gap_group'],observed=False)['date_diff'].max().reset_index()
agg_df = agg_df.groupby('name',observed=False)['date_diff'].max().rename('days_without_sale')
sales_train = sales_train.merge(agg_df, on='name', how='left')
sales_test = sales_test.merge(agg_df, on='name', how='left')

sales_train.set_index(my_index,inplace=True)
sales_test.set_index(my_index_ts,inplace=True)


oofs = {}
scores = {}
test_preds = {}
COMPUTE_TEST = True


def cross_validate(estimator, features, plot_residuals=False, fit_params={}):
    kf = TimeSeriesSplit(n_splits=5,test_size=dt.timedelta(weeks=2).days)
    X = sales_train[features].copy()
    y = sales_train[target]
       
    model = clone(estimator)
    val_preds = np.zeros(len(X))
    list_scores = []
    
    for fold, (trx_idx, val_idx) in enumerate(kf.split(X,y,groups=X['warehouse'])):        
        X_train, y_train = X.iloc[trx_idx], y.iloc[trx_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]      
                
        model.fit(X_train.drop('unique_id',axis=1),y_train, **fit_params)
        y_pred = model.predict(X_val.drop('unique_id',axis=1)).clip(0,None)
        val_preds[val_idx] += y_pred
        wmape = mean_absolute_error(y_val,y_pred,sample_weight=X_val["unique_id"].map(weight_map).values)
        list_scores.append(wmape)
        
        print(f' #{fold} - wmae: {wmape}')
        if plot_residuals:
            display = PredictionErrorDisplay.from_predictions(y_val,y_pred)            
            plt.show()
    if isinstance(model,Pipeline):
        name_model = type(model[-1]).__name__
    else:
        name_model = type(model).__name__                              
    

    oofs[name_model] = val_preds
    scores[name_model] = list_scores
    print(f'wmae mean: {np.mean(list_scores)}')   
    
    if COMPUTE_TEST:
        print('Computing Test prediction....')
        model = clone(estimator)
        model.fit(X,y)
        
        test_pred = model.predict(sales_test[features]).clip(0,None)
        test_preds[name_model] = test_pred
        print('Computing Test prediction - Ok')


cross_validate(make_pipeline(             
             TargetEncoder(cols=['name',
                                 'holiday_name',
                                 'L2_category_name_en',
                                 'L3_category_name_en',
                                 'L4_category_name_en']),                                                    
             OneHotEncoder(cols=['warehouse','L1_category_name_en']),      
             StandardScaler(),                          
             Ridge()),initial_features+['days_in_sale', 'purchase_interval','days_without_sale',
                                        'year_sin','year_cos','month_sin','month_cos'])


cross_validate(make_pipeline(             
             TargetEncoder(cols=['name',
                                 'holiday_name',
                                 'L2_category_name_en',
                                 'L3_category_name_en',
                                 'L4_category_name_en']),                                     
             LGBMRegressor(verbosity=-1)),initial_features+['days_in_sale', 'purchase_interval','days_without_sale',
                                                           'year_sin','year_cos','week','weekday'])


cross_validate(make_pipeline(             
             TargetEncoder(cols=['name',
                                 'holiday_name',
                                 'L2_category_name_en',
                                 'L3_category_name_en',
                                 'L4_category_name_en']),                                     
             XGBRegressor(verbosity=0,enable_categorical=True)),
             initial_features+['days_in_sale', 'purchase_interval'])


features = [
 'warehouse',
 'total_orders',
 'sell_price_main',
 'type_0_discount',
 'type_1_discount',
 'type_2_discount',
 'type_3_discount',
 'type_4_discount',
 'type_5_discount',
 'type_6_discount',
 'product_unique_id',
 'name',
 'L1_category_name_en',
 'L2_category_name_en',
 'L3_category_name_en',
 'L4_category_name_en',
 'holiday_name',
 'holiday',
 'shops_closed',
 'winter_school_holidays',
 'school_holidays',
 'days_in_sale',
 'purchase_interval',
 'days_without_sale',
 'year_sin',
 'year_cos',
 'week',
 'weekday']



train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
solution = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')
inv = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
cle = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")
test_weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')

train = train.merge(cle,on=['warehouse','date'],how='left') 
train = train.merge(inv,on=['warehouse','unique_id'],how='left')

test = test.merge(cle,on=['warehouse','date'],how='left')
test = test.merge(inv,on=['warehouse','unique_id'],how='left')

train = train.dropna(subset=['sales']) # Null in Target

train.drop('availability', axis=1, inplace=True) # Not Available in Test
# train.drop('name', axis=1, inplace=True) # Creating Problem in Training
# test.drop('name', axis=1, inplace=True) 

# SOURCE: https://www.kaggle.com/code/abdmental01/rsf-abd-base#AbdBase-%7C%7C-LGBM-
def date(df):
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.day_name()
    df['week'] = df['date'].dt.isocalendar().week
    df['year_sin'] = np.sin(2 * np.pi * df['year'])
    df['year_cos'] = np.cos(2 * np.pi * df['year'])
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12) 
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)  
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['group'] = (df['year'] - 2020) * 48 + df['month'] * 4 + df['day'] // 7
    
    df.drop('date', axis=1, inplace=True)
    
    cols = ['warehouse', 'month_name', 'day_of_week','holiday_name','L1_category_name_en',
             'L2_category_name_en','L3_category_name_en','L4_category_name_en']
    df['holiday_name'] = df['holiday_name'].fillna('None')
    for c in cols:
        df[c] = df[c].astype('category')

    return df


train = date(train)
test = date(test)


def cross_validate(estimator, features, plot_residuals=False, fit_params={}):
    kf = TimeSeriesSplit(n_splits=5,test_size=dt.timedelta(weeks=2).days)
    X = train[features].copy()
    y = train['sales']
       
    model = clone(estimator)
    val_preds = np.zeros(len(X))
    list_scores = []
    
    for fold, (trx_idx, val_idx) in enumerate(kf.split(X,y,groups=X['warehouse'])):        
        X_train, y_train = X.iloc[trx_idx], y.iloc[trx_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]      
                
        model.fit(X_train.drop('unique_id',axis=1),y_train, **fit_params)
        y_pred = model.predict(X_val.drop('unique_id',axis=1)).clip(0,None)
        val_preds[val_idx] += y_pred
        wmape = mean_absolute_error(y_val,y_pred,sample_weight=X_val["unique_id"].map(weight_map).values)
        list_scores.append(wmape)
        
        print(f' #{fold} - wmae: {wmape}')
        if plot_residuals:
            display = PredictionErrorDisplay.from_predictions(y_val,y_pred)            
            plt.show()
    if isinstance(model,Pipeline):
        name_model = type(model[-1]).__name__
    else:
        name_model = type(model).__name__                              
    

    oofs[name_model] = val_preds
    scores[name_model] = list_scores
    print(f'wmae mean: {np.mean(list_scores)}')   
    
    if COMPUTE_TEST:
        print('Computing Test prediction....')
        model = clone(estimator)
        model.fit(X,y)
        
        test_pred = model.predict(test[features]).clip(0,None)
        test_preds[name_model] = test_pred
        print('Computing Test prediction - Ok')


from catboost import CatBoostRegressor
cat_features = [
                'name', 'warehouse', 'month_name', 'day_of_week','holiday_name','L1_category_name_en', 
                'L2_category_name_en','L3_category_name_en','L4_category_name_en',
               ]

cross_validate(                                              
             CatBoostRegressor(
                 task_type='CPU',
                 iterations=100,
                 cat_features=cat_features,
                 verbose=10
             ),
             test.columns
)


oofs


estimator = make_pipeline(             
             TargetEncoder(cols=['name',
                                 'holiday_name',
                                 'L2_category_name_en',
                                 'L3_category_name_en',
                                 'L4_category_name_en']),                                     
             LGBMRegressor(verbosity=-1))
features = initial_features+['days_in_sale', 'purchase_interval','days_without_sale',
                                                           'year_sin','year_cos','week','weekday']
plot_residuals=False
fit_params={}
kf = TimeSeriesSplit(n_splits=5,test_size=dt.timedelta(weeks=2).days)
X = sales_train[features].copy()
y = sales_train[target]
   
model = clone(estimator)
val_preds = np.zeros(len(X))
list_scores = []

for fold, (trx_idx, val_idx) in enumerate(kf.split(X,y,groups=X['warehouse'])):        
    X_train, y_train = X.iloc[trx_idx], y.iloc[trx_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]      
            
    model.fit(X_train.drop('unique_id',axis=1),y_train, **fit_params)
    y_pred = model.predict(X_val.drop('unique_id',axis=1)).clip(0,None)
    val_preds[val_idx] += y_pred
    wmape = mean_absolute_error(y_val,y_pred,sample_weight=X_val["unique_id"].map(weight_map).values)
    list_scores.append(wmape)
    
    print(f' #{fold} - wmae: {wmape}')
    if plot_residuals:
        display = PredictionErrorDisplay.from_predictions(y_val,y_pred)            
        plt.show()
if isinstance(model,Pipeline):
    name_model = type(model[-1]).__name__
else:
    name_model = type(model).__name__                              

oofs[name_model] = val_preds
scores[name_model] = list_scores
print(f'wmae mean: {np.mean(list_scores)}')   


importances = model.named_steps['lgbmregressor'].feature_importances_


feature_importance_df = pd.DataFrame({"Feature": features, "Importance": importances})

feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance_df["Importance"], y=feature_importance_df["Feature"], palette="viridis")
plt.title("Feature Importance of LightGBM Model")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


df_score = pd.DataFrame().from_dict(scores)
ax = df_score.mean().sort_values(ascending=False).plot(kind='barh')
bars = ax.patches
bars[-1].set_color('green')
ax.bar_label(ax.containers[0],label_type='center',color='white',fontweight='bold')
plt.title('scores')
ax.set_xlabel('wmae')
plt.show()


test_preds


if COMPUTE_TEST:
    solution['sales_hat'] = test_preds['Ridge'].clip(0,None)
    solution.to_csv('Ridge_submission.csv',index=False)
    solution['sales_hat'] = test_preds['LGBMRegressor'].clip(0,None)
    solution.to_csv('LGBMRegressor_submission.csv',index=False)
    solution['sales_hat'] = test_preds['XGBRegressor'].clip(0,None)
    solution.to_csv('XGBRegressor_submission.csv',index=False)
    solution['sales_hat'] = test_preds['CatBoostRegressor'].clip(0,None)
    solution.to_csv('CatBoostRegressor_submission.csv',index=False)

