!pip -q install calplot


pip install pytabkit


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
from pytabkit import RealMLP_TD_Regressor
from pytabkit import TabM_D_Regressor
from catboost import CatBoostRegressor
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


from IPython.display import display


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



# 选出所有的折扣列
discount_cols = ["type_0_discount", "type_1_discount", "type_2_discount", 
                 "type_3_discount", "type_4_discount", "type_5_discount", "type_6_discount"]

# 计算最大折扣（如果有负值，意味着无折扣，需要替换为 0）
sales_train["max_discount"] = sales_train[discount_cols].max(axis=1)
sales_test["max_discount"] = sales_test[discount_cols].max(axis=1)

# 确保所有负折扣值转换为 0（负值意味着没有折扣）
sales_train["max_discount"] = sales_train["max_discount"].clip(lower=0)
sales_test["max_discount"] = sales_test["max_discount"].clip(lower=0)
target_column = 'sales'  

# 选择要编码的类别特征
category_cols = ["name", "holiday_name", "L1_category_name_en", "L2_category_name_en", "L3_category_name_en", "L4_category_name_en"]

# 初始化 TargetEncoder
encoder = TargetEncoder(
    cols=category_cols,
    handle_unknown='value',  # 测试集新类别用训练集全局均值代替
    smoothing=10,            # 平滑系数，防止过拟合
    min_samples_leaf=5       # 最小样本数，增加鲁棒性
)

# 对训练集进行拟合和转换
sales_train[category_cols] = encoder.fit_transform(
    sales_train[category_cols], 
    sales_train[target_column]  # 需要传入目标变量
)

# 对测试集进行转换（使用训练集的统计量）
sales_test[category_cols] = encoder.transform(
    sales_test[category_cols]
)
from sklearn.preprocessing import LabelEncoder

# 选择要编码的类别特征
category_cols = ['warehouse', 'L1_category_name_en']

# 逐列进行 Label Encoding
label_encoders = {}
for col in category_cols:
    le = LabelEncoder()
    sales_train[col] = le.fit_transform(sales_train[col])
    sales_test[col] = le.transform(sales_test[col])  # 确保 test 数据集用相同编码
    label_encoders[col] = le

sales_train['gap'] = sales_train['gap'].astype(int)


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
    
    # if COMPUTE_TEST:
    #     print('Computing Test prediction....')
    #     model = clone(estimator)
    #     model.fit(X,y)
        
    #     test_pred = model.predict(sales_test[features]).clip(0,None)
    #     test_preds[name_model] = test_pred
    #     print('Computing Test prediction - Ok')


features = ['unique_id', 'name','total_orders','warehouse',
            'product_unique_id','max_discount','sell_price_main',
            'gap_group','week']


seed = 2025


cross_validate(make_pipeline(             
             # TargetEncoder(cols=['name',
             #                     'holiday_name',
             #                     'L2_category_name_en',
             #                     'L3_category_name_en',
             #                     'L4_category_name_en']),                                                    
             # OneHotEncoder(cols=['warehouse','L1_category_name_en']),      
             StandardScaler(),                          
             Ridge(random_state=seed)),features)


cross_validate(make_pipeline(             
             # TargetEncoder(cols=['name',
             #                     'holiday_name',
             #                     'L2_category_name_en',
             #                     'L3_category_name_en',
             #                     'L4_category_name_en']),                                     
             LGBMRegressor(verbosity=-1, random_state=seed)),features)


cross_validate(make_pipeline(             
             # TargetEncoder(cols=['name',
             #                     'holiday_name',
             #                     'L2_category_name_en',
             #                     'L3_category_name_en',
             #                     'L4_category_name_en']),                                     
             XGBRegressor(verbosity=0,enable_categorical=True, random_state=seed)),
             features)


cb_params = {
    'grow_policy'        : 'Lossguide',
    #'task_type'          : 'GPU',
    'iterations'         : 800,
    'bagging_temperature': 0.5,
    'learning_rate'      : 0.1,
    'max_leaves'         : 128,
    'max_depth'          : 12,
    'l2_leaf_reg'        : 1.25,
    'min_data_in_leaf'   : 24,
    'verbose'            : 0,
    'border_count'       : 256,
   # 'cat_features'       : cat_columns,
}

cross_validate(make_pipeline(             
    #          TargetEncoder(cols=['name',
    #                              'holiday_name',
    #                              'L2_category_name_en',
    #                              'L3_category_name_en',
    #                              'L4_category_name_en']), 
    # OneHotEncoder(cols=['warehouse','L1_category_name_en']), 
             CatBoostRegressor(**cb_params, random_state=seed)),
             features)


df_score = pd.DataFrame().from_dict(scores)
ax = df_score.mean().sort_values(ascending=False).plot(kind='barh')
bars = ax.patches
bars[-1].set_color('green')
ax.bar_label(ax.containers[0],label_type='center',color='white',fontweight='bold')
plt.title('scores')
ax.set_xlabel('wmae')
plt.show()


# if COMPUTE_TEST:
#     solution['sales_hat'] = test_preds['LGBMRegressor'].clip(0,None)
#     solution.to_csv('submission.csv',index=False)

