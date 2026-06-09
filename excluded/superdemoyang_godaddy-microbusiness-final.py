import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
from tqdm.auto import tqdm

import datetime
import dateutil.parser

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style = 'whitegrid')

%matplotlib inline

import plotly.express as px

import copy
from collections import defaultdict

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline


from statsmodels.tsa.forecasting.stl import STLForecast, STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing as ES

import xgboost as xgb

import statsmodels.formula.api as smf    
    
        
BASE = '/kaggle/input/godaddy-microbusiness-density-forecasting/'


train_data = pd.read_csv(BASE+"train.csv")
test_data = pd.read_csv(BASE+"test.csv")
revealed_data = pd.read_csv(BASE+"revealed_test.csv")


train_data


test_data


revealed_data


train_data['is_test'] = 0
revealed_data['is_test'] = 1
test_data['is_test'] = 2

### 提取验证集
test_data = test_data[~test_data.row_id.isin(revealed_data.row_id)]

data = pd.concat([train_data, revealed_data, test_data])


### 基础的时间信息特征
data['first_day_of_month'] = pd.to_datetime(data['first_day_of_month'])

data['month_from_month'] =data['first_day_of_month'].dt.month
data['year_from_month'] =data['first_day_of_month'].dt.year

data['population_year_minus_two']=(data['active']/data['microbusiness_density'])*100
data['population_year'] = data.year_from_month - 2


import missingno as msno

msno.matrix(data)


### 拼接额外的地理位置信息
coords = pd.read_csv("/kaggle/input/usa-counties-coordinates/cfips_location.csv", sep=',')
data = data.merge(coords.drop(['name'], axis=1), on="cfips", how="left")

for cfips in tqdm(data['cfips'].unique()):
    mask = data.cfips == cfips
    data.loc[mask, 'county'] = data.loc[mask, 'county'].iloc[0]
    data.loc[mask, 'state'] = data.loc[mask, 'state'].iloc[0]
    data.loc[mask, 'population_year_minus_two'] = data.loc[mask, 'population_year_minus_two'].ffill()


data.reset_index(drop=True, inplace=True)


### 根据县城id进行拼接可以获得对应的经纬度信息  也可以后续结合其他地理数据  比如气候、地形等
coords


### 拼接额外的人口信息数据并提取特征 工作人口数和退休人口数
for year in data['population_year'].unique():
    print(f'Processing {year}')
    filename = f'/kaggle/input/census-data-for-godaddy/ACSST5Y{year}.S0101-Data.csv'
    COLS = ['GEO_ID','NAME','S0101_C01_026E', 'S0101_C01_028E']
    temp = pd.read_csv(filename,usecols=COLS)
    temp = temp.iloc[1:]
    temp['S0101_C01_026E'] = temp['S0101_C01_026E'].astype('int')
    temp['S0101_C01_028E'] = temp['S0101_C01_028E'].astype('int')
    temp['cfips'] = temp.GEO_ID.apply(lambda x: int(x.split('US')[-1]) )
    adult = temp.set_index('cfips').S0101_C01_026E.to_dict()
    retired = temp.set_index('cfips').S0101_C01_028E.to_dict()
    mask = data.population_year == year
    data.loc[mask, 'used_population'] = data.loc[mask, 'cfips'].map(adult)
    data.loc[mask, 'retired_population'] = data.loc[mask, 'cfips'].map(retired)


### 根据cfips进行拼接 获得对应县的成年人口数和老年人口数  在进行微活企业分析时主要考虑年轻人
temp


### 添加额外的人口普查数据并构建特征  
census_data = pd.read_csv(BASE + "census_starter.csv")
census_data.set_index('cfips', inplace=True)

def add_last_year_data(cfips, year, census_data, col_stub):
    col_name = f"{col_stub}{year}"
    return census_data.loc[cfips][col_name]

# 使用宽带人口的百分比
data['pct_bb_last_year'] = data.apply(lambda row: add_last_year_data(cfips=row['cfips'], year=row['population_year'], census_data=census_data, col_stub='pct_bb_'), axis=1)
# 大学学位人口百分比
data['pct_college_last_year'] = data.apply(lambda row: add_last_year_data(cfips=row['cfips'], year=row['population_year'], census_data=census_data, col_stub='pct_college_'), axis=1)
# 在美国境外出生的人口百分比
data['pct_foreign_born_last_year'] = data.apply(lambda row: add_last_year_data(cfips=row['cfips'], year=row['population_year'], census_data=census_data, col_stub='pct_foreign_born_'), axis=1)
# 从事信息相关行业的劳动力百分比
data['pct_it_workers_last_year'] = data.apply(lambda row: add_last_year_data(cfips=row['cfips'], year=row['population_year'], census_data=census_data, col_stub='pct_it_workers_'), axis=1)
# 家庭收入中位数
data['median_hh_inc_last_year'] = data.apply(lambda row: add_last_year_data(cfips=row['cfips'], year=row['population_year'], census_data=census_data, col_stub='median_hh_inc_'), axis=1)


### 根据cfips进行拼接 获得对应县的人口普查数据信息 可以获得近几年对应县的各类人口百分比变化情况 
# 比如从宽带人口数变化和大学学位人口数变化就可以看出对应的年轻人流入流出情况
# 家庭收入的变化也可以反应出当地的经济形式
census_data


### 添加相邻县的相关信息数据并构建特征

# 通过对比的方式可以更加突出每个地的变化趋势
# 并且一个地区的发展也会在一定程度上受到周围地区发展的影响
def add_neighbor_value(df, weighted=False):
    county_neighbors=pd.read_csv('/kaggle/input/county-neighbours/county-neighbours.csv')
    county_neighbors.rename(columns = {'Neighbour county code':'cfips'}, inplace = True)
    
    res = defaultdict(float)
    all_cfips = df['cfips'].unique()
    for cfips in tqdm(all_cfips):
        ### 获取相邻县
        neighbors = county_neighbors[county_neighbors['Countycode'] == cfips]['cfips'].unique()
        if len(neighbors) == 0:  # If no neighbors, set to self
            tdf = df[df['cfips'] == cfips]
            res.update({(cfips, row['first_day_of_month']): row['microbusiness_density'] for __, row in tdf.iterrows()})
        else:
            temp = df[df['cfips'].isin(neighbors)]

            if not weighted:
                ### 统计某县某时间点周围邻居县的目标值
                res.update({(cfips, dt): df['microbusiness_density'].mean() for dt, df in temp.groupby('first_day_of_month')})
            else:
                res.update({(cfips, dt): (df['microbusiness_density'] * df['used_population']).sum() / df['used_population'].sum() for dt, df in temp.groupby('first_day_of_month')})


    df['neighbor_average'] = df.apply(lambda row: res[(row['cfips'], row['first_day_of_month'])], axis=1)
    df['neighbor_diff'] = df['neighbor_average']/df['microbusiness_density'] - 1
    
    df.loc[df['microbusiness_density'] == 0, 'neighbor_diff'] = 0

add_neighbor_value(data)


data


### 由于预测目标是企业的活跃数量除以落后2年的工作年龄人口，如果微型企业的比例大致不变，预测人口变化可以增加一些预测能力。
### 引入人口组成数据

working_age = {5,6,7,8,9,10,11}
pop_df = pd.read_csv('/kaggle/input/annual-county-population-by-age-data-1969-to-2020/us.1969_2020.19ages.adjusted.csv')
total_pop = pop_df.groupby(['Year', 'cfips'])['population'].sum()
### 统计各种人口数
working_pop = pop_df[pop_df['agecode'].isin(working_age)].groupby(['Year', 'cfips'])['population'].sum() / total_pop
working_male = pop_df[pop_df['agecode'].isin(working_age) & (pop_df['sex'] == 'M')].groupby(['Year', 'cfips'])['population'].sum() / total_pop
white_pop = pop_df[pop_df['race'] == 'W'].groupby(['Year', 'cfips'])['population'].sum() / total_pop
black_pop = pop_df[pop_df['race'] == 'B'].groupby(['Year', 'cfips'])['population'].sum() / total_pop


pop_df


# 这里的agecode不是对应年龄 5-11差不多对应工作年龄阶段
pop_df['agecode'].value_counts()


pop_df.groupby('cfips')['agecode'].mean()


def get_pop(row, pop_df_grp, dflt='population_year_minus_two'):
    try:
        return pop_df_grp.loc[row['population_year'], row['cfips']]
    except KeyError:
        return row[dflt]


def get_pop_pct(row, pop_df_grp, dflt=np.nan):
    try:
        return pop_df_grp.loc[row['population_year'], row['cfips']]
    except KeyError:
        return dflt


def get_10yr_pop_chg(row, pop_df_grp):
    try:
        yr = row['population_year']
        now = pop_df_grp.loc[yr, row['cfips']]
        then = pop_df_grp.loc[yr-10, row['cfips']]
        return now / then - 1
    except KeyError:
        return 0

for c in data.columns:
    if 'seer' in c:
        del data[c]


### 统计同一年的相关人口特征
data['seer_pop'] = data.apply(lambda row: get_pop(row, total_pop), axis=1)

data['seer_working_pct'] = data.apply(lambda row: get_pop_pct(row, working_pop), axis=1)
data['seer_working_male_pct'] = data.apply(lambda row: get_pop_pct(row, working_male), axis=1)
data['seer_white_pct'] = data.apply(lambda row: get_pop_pct(row, white_pop), axis=1)
data['seer_black_pct'] = data.apply(lambda row: get_pop_pct(row, black_pop), axis=1)

data['seer_pop_chg'] = data.apply(lambda row: get_10yr_pop_chg(row, total_pop), axis=1)
data['seer_pop_ratio'] = data['seer_pop']/data['used_population'] - 1


for cfips in tqdm(data['cfips'].unique()):
    mask = data.cfips == cfips
    for c in data.columns:
        if 'seer' in c:
            data.loc[mask, c] = data.loc[mask, c].ffill()



from matplotlib import pyplot as plt
from featexp import get_univariate_plots


features_list = ['seer_working_pct','seer_working_male_pct','seer_white_pct','seer_black_pct','seer_pop_chg']
# 生成单变量分析图
get_univariate_plots(data=data, target_col='microbusiness_density', features_list=features_list, bins=30)


### 统计每个县的劳动力相关数据和特征
for col in ['unemployment_rate', 'labor_force']:
    full_dict = defaultdict(lambda: np.nan)
    for yr in data['population_year'].unique():
        file = '/kaggle/input/unemployment-data-of-us/laucnty{}.xlsx'.format(yr % 100)
        columns = ['LUASCode', 'state_fips', 'county_fips', 'name', 'year', 'dummy', 'labor_force', 'employed', 'unemployed', 'unemployment_rate']
        df = pd.read_excel(file, header=4)
        df.columns = columns
        del df['dummy']
        df.dropna(inplace=True)
        df['year'] = df['year'].map(int)
        df['cfips'] = (df['state_fips']*1000 + df['county_fips']).map(int)
        temp = {(row['cfips'],yr): row[col] for __, row in df.iterrows()}
        full_dict.update(temp)

    data[col] = data.apply(lambda row: full_dict[(row['cfips'], row['population_year'])], axis=1)


### 包括总劳动力 无工作人数 工作人数 
# 可以反映出一个地区目前的失业情况
df


for cfips in tqdm(data['cfips'].unique()):
    mask = data.cfips == cfips
    for c in ['unemployment_rate', 'labor_force']:
        data.loc[mask, c] = data.loc[mask, c].ffill()


### 劳动参与率 劳动力除以总人口  分析劳动力占比 该区域是属于年轻化还是老龄化
data['labor_participation'] = data['labor_force'] / data['seer_pop']
data['labor_participation'].hist(bins=100)


### 引入微型企业相关活动数据进行特征统计
tmp = pd.read_csv('/kaggle/input/us-microbusiness-activity-index-counties/VF_mai_counties_Q222.csv')
tmp['first_day_of_month'] = pd.to_datetime(tmp['date'])
data = data.merge(tmp.drop(['total_pop_20', 'county_name', 'date'], axis=1), how='left', on=['first_day_of_month', 'cfips'])
data


# Fill missing data
for cfips in tqdm(data['cfips'].unique()):
    mask = data.cfips == cfips
    for c in ['MAI_composite', 'engagement', 'participation', 'infrastructure']:
        data.loc[mask, c] = data.loc[mask, c].bfill().ffill()
        
data


### 查看构建的特征分布
data.describe().T


### 设置损失函数和目标函数
def smape(y_true, y_pred):
    
    # CONVERT TO NUMPY
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # WHEN BOTH EQUAL ZERO, METRIC IS ZERO
    both = np.abs(y_true) + np.abs(y_pred)
    idx = np.where(both==0)[0]
    y_true[idx]=1; y_pred[idx]=1
    
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def vsmape(y_true, y_pred):
    
    # CONVERT TO NUMPY
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # WHEN BOTH EQUAL ZERO, METRIC IS ZERO
    both = np.abs(y_true) + np.abs(y_pred)
    idx = np.where(both==0)[0]
    y_true[idx]=1
    y_pred[idx]=1
    
    return 200 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))


import xgboost as xgb

### 初始化xgb模型
xgb_model = xgb.XGBRegressor(
        objective='reg:pseudohubererror',
        #objective='reg:squarederror',
        tree_method="hist",
        n_estimators=795,
        learning_rate=0.0075,
        max_leaves = 17,
        subsample=0.50,
        colsample_bytree=0.50,
        max_bin=4096,
        n_jobs=2,
)


### 定义简单的上个值模型用于测试
class LastValueModel(object):
    def __init__(self):
        self.last_values = defaultdict(float)
        
    def reset(self):
        pass
    
    def fit(self, data):
        max_dt = data.first_day_of_month.max()
        print('Take last values from', max_dt)
        self.last_values = {row.cfips: row.microbusiness_density for __, row in data[data.first_day_of_month == max_dt].iterrows()}
        
    def predict(self, data):
        data['pred'] = data['cfips'].map(self.last_values)
        
        return data.set_index('row_id')['pred']


### 添加额外的目标滑动窗口特征
def build_features(raw, target='microbusiness_density', target_act='active_tmp', lags = 6):
    feats = []
    for lag in tqdm(range(1, lags)):
        raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')[target].shift(lag)
        raw[f'act_lag_{lag}'] = raw.groupby('cfips')[target_act].diff(lag)
        feats.append(f'mbd_lag_{lag}')
        feats.append(f'act_lag_{lag}')
        
    lag = 1
    for window in [2, 4, 6, 8, 10]:
        raw[f'mbd_rollmea{window}_{lag}'] = raw.groupby('cfips')[f'mbd_lag_{lag}'].transform(lambda s: s.rolling(window, min_periods=1).sum())        
        #raw[f'mbd_rollmea{window}_{lag}'] = raw[f'mbd_lag_{lag}'] - raw[f'mbd_rollmea{window}_{lag}']
        feats.append(f'mbd_rollmea{window}_{lag}')
    
    return raw, feats
def train_clip_none(s):
    return s

def train_clip_small(s):
    return s.clip(-0.0045, 0.0045)


from sklearn.base import TransformerMixin, BaseEstimator

### 定义异常值处理部分，使用平滑处理的方式
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, active_threshold, pct_change_threshold):
        self.active_threshold = active_threshold
        self.pct_change_threshold = pct_change_threshold
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        changes = defaultdict(list)

        X_ = X.copy()
        for o in X_.cfips.unique():
            indices = (X_['cfips']==o)
            tmp = X_.loc[indices].copy().reset_index(drop=True)
            var = tmp.active.values.copy()
            dts = tmp.first_day_of_month.copy()

            for i in range(len(var)-1, 1, -1):
                if var[i] > self.active_threshold:
                    pct_chg = var[i]/var[i-1] - 1
                    if abs(pct_chg) > self.pct_change_threshold:
                        var[:i] += (var[i] - var[i-1])
                        changes[o].append(dts[i])

            X_.loc[indices, 'smoothed_active'] = var
            X_.loc[indices, 'average_population'] = np.mean(X_.loc[indices, 'used_population'])
                
        X_['smoothed_microbusiness_density'] = X_['smoothed_active'] / X_['average_population'] * 100

        cnt = sum([len(v) for v in changes.values()])
        print(f'Adjusted {len(changes)} cfips and {cnt} points for threshold = {self.pct_change_threshold}')
        return X_


### 定义预测模型
class BDTRollFwdModel(LastValueModel):
    def __init__(self, get_model, pop_split, train_clip, use_blacklist, features, scale, probing_changes=None, jan_replace_file=None):
        super().__init__()
        
        self.use_blacklist = use_blacklist
        self.train_clip = train_clip
        self.pop_split = pop_split
        self.model = get_model
        self.features = features
        self.scale = scale
        self.probing_changes = probing_changes
        self.jan_replace_file = jan_replace_file
        
        self.save_data = None
        
    def reset(self):
        super().reset()
        self.save_data = None
        
    def prep_data(self, df, training):
        ### 异常值处理变化超过10%就算作异常值
        smoother = OutlierRemover(active_threshold=5, pct_change_threshold=0.1)
        xgb_data = smoother.transform(df.copy())

        ### 由于评价指标的特殊性，将目标进行转换，变成求取目标值变化的幅度
        xgb_data['target_0'] = xgb_data.groupby('cfips')['smoothed_microbusiness_density'].shift(-1)
        xgb_data['target'] = xgb_data['target_0'] / xgb_data['smoothed_microbusiness_density'] - 1

        if training:
            xgb_data['target'] = xgb_data['target'].fillna(0)
            xgb_data.loc[xgb_data.first_day_of_month == xgb_data.first_day_of_month.max(), 'target'] = np.nan
        
            xgb_data.dropna(subset=['target'], inplace=True)
            mask = xgb_data['target'] == np.inf
            xgb_data.loc[mask, 'target'] = 0

        ### 添加额外的滑动窗口特征
        xgb_data, __ = build_features(xgb_data, target='target', target_act='smoothed_active', lags=8)

        xgb_data.reset_index(inplace=True, drop=True)

        xgb_data['county_i'] = (xgb_data['county'] + xgb_data['state']).factorize()[0]
        xgb_data['state_i'] = xgb_data['state'].factorize()[0]

#         xgb_data['top0ind_i'] = xgb_data['top0ind'].factorize()[0]
#         xgb_data['top1ind_i'] = xgb_data['top1ind'].factorize()[0]
#         xgb_data['top2ind_i'] = xgb_data['top2ind'].factorize()[0]
        
        return xgb_data
        
    def get_mask(self, df):
        return df.average_population > self.pop_split
        
    def fit(self, data):
        super().fit(data)
        
        self.saved_data = data.copy()
        
        fit_data = self.prep_data(data, training=True)
        
        print('Fitting model')
    
        mask = self.get_mask(fit_data)
        
#         fcst_dt = fit_data.first_day_of_month.max()
#         t = pd.to_datetime(fcst_dt).strftime('%Y%m%d')
#         fit_data[mask].to_pickle(f'v10_train_{t}.p')
        
        self.model.fit(fit_data.loc[mask, self.features], self.train_clip(fit_data.loc[mask, 'target']))
        
    def predict(self, data):
        last_value_pred = super().predict(data)

        tmp = pd.concat([self.saved_data.copy(), data.copy()])
        
        for fcst_dt in data['first_day_of_month'].unique():
            print('Fcst Date', fcst_dt)
            
            fit_data = self.prep_data(tmp[(tmp.first_day_of_month < fcst_dt)].copy(), training=False)
            
#             t = pd.to_datetime(fcst_dt).strftime('%Y%m%d')
#             fit_data.to_pickle(f'v10_validation_{t}.p')
            
            max_date = fit_data.first_day_of_month.max()
            print('Max Date: ', max_date)
            validation = fit_data[fit_data.first_day_of_month == max_date].copy()
            validation_pred_y = (self.model.predict(validation.loc[:, self.features]) * self.scale + 1) * validation['microbusiness_density']
    
            # Below pop split, use last value
            validation['pred'] = validation_pred_y
            # validation.loc[~self.get_mask(validation), 'pred'] = validation.loc[~self.get_mask(validation), 'microbusiness_density']
    
            # if self.use_blacklist:
            #     mask = validation['state'].isin(blacklist) | validation.cfips.isin(blacklistcfips)
            #     validation.loc[mask, 'pred'] = validation.loc[mask, 'microbusiness_density']
    
            prd = validation['pred'].copy()

            prd.loc[prd.isna()] = 0  
            
            # NOTE THE ROUNDING.  It improves scores.
            validation['active_pred'] = (prd * validation['used_population'] / 100).map(round)
    
            d = {row['cfips']: row['active_pred'] for __, row in validation.iterrows()}
            mask = tmp.first_day_of_month == fcst_dt
            tmp.loc[mask, 'active'] = tmp.loc[mask, 'cfips'].map(d)
            tmp.loc[mask, 'microbusiness_density'] = tmp.loc[mask, 'active'] / tmp.loc[mask, 'used_population'] * 100
                
        d = {row['row_id']: row['microbusiness_density'] for __, row in tmp.iterrows()}
        data['pred'] = data['row_id'].map(d)
        return data.set_index('row_id')['pred']


### 进行特征选择
features = ['state_i',  
 'mbd_lag_1',
 'act_lag_1',
 'mbd_lag_2',
 'act_lag_2',
 'mbd_lag_3',
 'act_lag_3',
 'mbd_lag_4',
 'act_lag_4',
 'mbd_lag_5',
 'act_lag_5',
 'mbd_lag_6',
 'act_lag_6',
 'mbd_lag_7',
 'act_lag_7',
 'mbd_rollmea2_1',
 'mbd_rollmea4_1',
 'mbd_rollmea6_1',
 'mbd_rollmea8_1',
 'mbd_rollmea10_1',
 'pct_bb_last_year',
 'used_population',
 'neighbor_diff',
 'pct_college_last_year',
 'pct_foreign_born_last_year',
 'pct_it_workers_last_year',
 'median_hh_inc_last_year',
 'seer_pop_chg',
 'labor_force',
 'labor_participation',
 'participation',
 'lat',
 'lng',
 'engagement',
 'seer_black_pct',
           ]

train_clip = train_clip_small
pop_split = 5000


model = BDTRollFwdModel(get_model=xgb_model, pop_split=pop_split, train_clip=train_clip, use_blacklist=False, features=features, scale=1.0)


### 使用k折交叉验证

# 设置训练参数
train_data_all = data[data.is_test == 0].copy()
start_test_month=2
number_months=3
number_folds=5


### k折加滚动预测 每次使用截止到两个月前的数据去预测这个月 滚动预测直到结束
values = []
months = train_data.first_day_of_month.unique()
for n in reversed(range(number_folds)):
    last_forecast_month = months[-n-1]
    begin_forecast_month = months[-n-number_months]
    train_max_month = months[-n-number_months-start_test_month-1]
    print('------------------------------------------------------------------------------------------------------------------------------')
    print('Fold', number_folds - n, ': Train up to and including data from ', train_max_month, 'and evaluate forecasts from ', begin_forecast_month,'to', last_forecast_month)
    train_data = train_data_all[train_data_all.first_day_of_month <= train_max_month].copy()
    validation_data = train_data_all[(train_data_all.first_day_of_month > train_max_month) & (train_data_all.first_day_of_month <= last_forecast_month)].copy()
    validation_data['microbusiness_density'] = np.nan
    validation_data['active'] = np.nan
    
    model.reset()
    print('Training model')
    t1 = datetime.datetime.utcnow()
    model.fit(train_data)
    t2 = datetime.datetime.utcnow()
    print('Training took ', (t2-t1).seconds, 'seconds')
    print('Predicting')
    pred = model.predict(validation_data)
    
    
    test = train_data_all[train_data_all.first_day_of_month > train_max_month].copy()
    test.set_index('row_id', inplace=True)
    test['pred'] = pred
    
    evaluation_data = test[(test.first_day_of_month >= begin_forecast_month) & (test.first_day_of_month <= last_forecast_month)]
    
    value = smape(evaluation_data['microbusiness_density'], evaluation_data['pred'])
    if np.isnan(value):
        raise ValueError('Smape is NaN')
    print(f'SMAPE = ', value)
    values.append([train_max_month, begin_forecast_month, last_forecast_month, value])
    
print('------------------------------------------------------SUMMARY-------------------------------------------------------------------')
    
for train_max_month, begin_forecast_month, last_forecast_month, value in values:
    print('Train up to and including data from ', train_max_month, 'and evaluate forecasts from ', begin_forecast_month,'to', last_forecast_month,'=', value)
    
print('Average SMAPE =', np.mean([x[-1] for x in values]))

       


import seaborn as sns
import scipy.stats as stats
for col in ['microbusiness_density', 'active','pct_it_workers_last_year']:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.kdeplot(data[col], shade=True, color='blue')
    plt.title(col)
    
    plt.subplot(122)
    stats.probplot(data[col], dist="norm", plot=plt)
    plt.title(col)
    plt.show();


!zip -r ./annual-county-population-by-age-data-1969-to-2020.zip /kaggle/input/annual-county-population-by-age-data-1969-to-2020


!zip -r ./county-neighbours.zip /kaggle/input/county-neighbours


!zip -r ./godaddy-best-public-and-home-changes.zip /kaggle/input/godaddy-best-public-and-home-changes


!zip -r ./unemployment-data-of-us.zip /kaggle/input/unemployment-data-of-us


!zip -r ./us-microbusiness-activity-index-counties.zip /kaggle/input/us-microbusiness-activity-index-counties


!zip -r ./census-data-for-godaddy.zip /kaggle/input/census-data-for-godaddy


!zip -r ./godaddy-microbusiness-density-forecasting.zip /kaggle/input/godaddy-microbusiness-density-forecasting


!zip -r ./usa-counties-coordinates.zip /kaggle/input/usa-counties-coordinates




