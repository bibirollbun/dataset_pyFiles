import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
from IPython.display import clear_output

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split, cross_val_score, cross_validate, GridSearchCV
from random import random, randint, randrange, uniform
from lightgbm import LGBMRegressor
from lightgbm import log_evaluation, early_stopping

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.metrics import *
from sklearn.metrics import make_scorer, mean_absolute_percentage_error


rs = 9


# load dataset
dfTrain=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
dfTest=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample_result = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')

# dfTrain = dfTrain.loc[:1999]
# dfTest = dfTest.loc[:999]


# print shapes of dataset
print(dfTrain.shape, dfTest.shape)


# check Nan values
print(dfTrain.isnull().sum(),dfTest.isnull().sum())


# get info about numerical stats
print(dfTrain.describe(include='object'))
print(dfTest.describe(include='object'))


print(dfTrain.info())
print(dfTest.info())


# 날짜 변수 생성
dfTrain['date'] = pd.to_datetime(dfTrain['date'])
dfTrain['year'] = dfTrain['date'].dt.year
dfTrain['month'] = dfTrain['date'].dt.month
dfTrain['day'] = dfTrain['date'].dt.day


cols = ['year', 'country', 'store', 'product']

for col in cols:
    print(f'column:{col}')
    print(dfTrain[col].unique())
    print()


# yearSales = dfTrain.groupby('year')['num_sold'].mean()
# month_sales = dfTrain.groupby('month')['num_sold'].mean()
# day_sales = dfTrain.groupby('day')['num_sold'].mean()
# countrySales = dfTrain.groupby(['year','country'])['num_sold'].mean().reset_index()
# countryMsales = dfTrain.groupby(['month','country'])['num_sold'].mean().reset_index()
# countryDsales = dfTrain.groupby(['day','country'])['num_sold'].mean().reset_index()


# # year sales
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=yearSales, marker='o', linewidth=2, color='gold')
# plt.title('Average Sales (Year)')
# plt.xlabel('Year')
# plt.ylabel('Average num_sold')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # month sales
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=month_sales, marker='o', linewidth=2, color='gold')
# plt.title('Average Sales (month)')
# plt.xlabel('month')
# plt.ylabel('Average num_sold')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # day sales
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=day_sales, marker='o', linewidth=2, color='gold')
# plt.title('Average Sales (day)')
# plt.xlabel('day')
# plt.ylabel('Average num_sold')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # country sales - year
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=countrySales,  # 데이터프레임 이름
#              x='year',         # x축: 기간
#              y='num_sold',          # y축: 매출액
#              hue='country',      # 범례: 국가별 구분
#              marker='o',         # 데이터 포인트 마커
#              linewidth=2)        # 선 굵기
# plt.title('Average Sales(country,year)')
# plt.xlabel('year')
# plt.ylabel('Average num_sold')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# # country sales - month
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=countryMsales,  # 데이터프레임 이름
#              x='month',         # x축: 기간
#              y='num_sold',          # y축: 매출액
#              hue='country',      # 범례: 국가별 구분
#              marker='o',         # 데이터 포인트 마커
#              linewidth=2)        # 선 굵기
# plt.title('Average Sales(country,month)')
# plt.xlabel('month')
# plt.ylabel('Average num_sold')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


# # country sales - day
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=countryDsales,  # 데이터프레임 이름
#              x='day',         # x축: 기간
#              y='num_sold',          # y축: 매출액
#              hue='country',      # 범례: 국가별 구분
#              marker='o',         # 데이터 포인트 마커
#              linewidth=2)        # 선 굵기
# plt.title('Average Sales(country,day)')
# plt.xlabel('day')
# plt.ylabel('Average num_sold')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


def plot_sales_trend(data, group_cols, x_col=None, title_suffix=''):
    # 데이터 집계
    if isinstance(group_cols, list):
        sales_data = data.groupby(group_cols)['num_sold'].mean().reset_index()
        x_col = x_col or group_cols[0]
        hue_col = group_cols[1] if len(group_cols) > 1 else None
    else:
        sales_data = data.groupby(group_cols)['num_sold'].mean()
        x_col = group_cols
        hue_col = None

    # 그래프 생성
    plt.figure(figsize=(12, 6))
    if hue_col:
        sns.lineplot(data=sales_data, x=x_col, y='num_sold', hue=hue_col,
                    marker='o', linewidth=2)
    else:
        sns.lineplot(data=sales_data, marker='o', linewidth=2, color='gold')

    plt.title(f'Average Sales ({title_suffix})')
    plt.xlabel(x_col)
    plt.ylabel('Average num_sold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# 사용 예시
time_periods = {
    'Year': 'year',
    'Month': 'month',
    'Day': 'day'
}

# 시간별 추세
for period_name, period_col in time_periods.items():
    plot_sales_trend(dfTrain, period_col, title_suffix=period_name)

# 국가별 시간 추세
for period_name, period_col in time_periods.items():
    plot_sales_trend(dfTrain, [period_col, 'country'], 
                    title_suffix=f'country, {period_name.lower()}')


# 국가별 결측치 수 파악하기
countries = ['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']
anl_countries = ['Canada','Kenya']

# 국가별 연도별 스토어별 제품별 결측치 수 파악하기
for country in anl_countries:
    for year in dfTrain['year'].unique():
        for store in dfTrain['store'].unique():
            for product in dfTrain['product'].unique():
                nan_values = dfTrain.loc[(dfTrain['country'] == country) & (dfTrain['year'] == year) & (dfTrain['store'] == store) & (dfTrain['product'] == product), 'num_sold'].isnull().sum()
                if nan_values>0:
                    print(f'{country} {year} {store} {product} nan values:', nan_values)
                else:
                    pass


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
    
    def fit(self, X, y=None):
        for col in ['country', 'store', 'product']:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le
        return self
    
    def fillnaMean(self, X):
        def fill_missing_products(df):
            df = df.copy()
            # 각 연도와 국가별로 처리
            for year in df['date'].dt.year.unique():
                for country in df['country'].unique():
                    # 해당 연도와 국가의 데이터 필터링
                    year_country_mask = (df['date'].dt.year == year) & (df['country'] == country)
                    year_country_data = df[year_country_mask]
                    
                    # 각 제품별로 처리
                    for product in ['Holographic Goose', 'Kerneler']:
                        # Premium과 Stickers for Less의 해당 제품 평균 판매량
                        other_stores_mask = year_country_data['store'].isin(['Premium Sticker Mart', 'Stickers for Less']) & \
                                          (year_country_data['product'] == product)
                        product_other_stores = year_country_data[other_stores_mask]['num_sold'].mean()
                        
                        # Discount Stickers의 다른 제품 대비 판매 비율 계산
                        discount_mask = (year_country_data['store'] == 'Discount Stickers') & \
                                      (year_country_data['product'] != product)
                        other_stores_products_mask = ~year_country_data['store'].isin(['Discount Stickers'])
                        
                        # 판매 비율 계산 (다른 제품들의 평균 판매량 대비)
                        discount_ratio = year_country_data[discount_mask]['num_sold'].mean() / \
                                       year_country_data[other_stores_products_mask]['num_sold'].mean()
                        
                        # 결측치 채우기
                        fill_mask = year_country_mask & \
                                  (df['store'] == 'Discount Stickers') & \
                                  (df['product'] == product)
                        
                        df.loc[fill_mask, 'num_sold'] = product_other_stores * discount_ratio
            
            return df
        
        # num_sold 컬럼이 있을 때만 처리
        if 'num_sold' in X.columns:
            X = fill_missing_products(X)
            # 혹시 남은 결측치가 있다면 기존 방식으로 처리
            X['num_sold'] = X['num_sold'].fillna(X.groupby(['country', 'product'])['num_sold'].transform('mean'))
        
        return X
    
    def transform(self, X):
        X = X.copy()
        
        # 먼저 결측치 처리를 수행
        if 'num_sold' in X.columns:
            X = self.fillnaMean(X)
        
        X['date'] = pd.to_datetime(X['date'])
        
        X['month'] = X['date'].dt.month
        X['day'] = X['date'].dt.day
        X['quarter'] = X['date'].dt.quarter
        X['season'] = X['month'].apply(self.get_season)
        X.drop('date', axis=1, inplace=True)
        
        for col in ['country', 'store', 'product']:
            le = self.label_encoders[col]
            X[col] = le.transform(X[col])
        
        X['month'] = X['month'].astype(int)
        X['day'] = X['day'].astype(int)
        X['quarter'] = X['quarter'].astype(int)
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        X['day_sin'] = np.sin(2 * np.pi * X['day'] / 31)
        X['day_cos'] = np.cos(2 * np.pi * X['day'] / 31)
        X['quarter_sin'] = np.sin(2 * np.pi * X['quarter'] / 4)
        X['quarter_cos'] = np.cos(2 * np.pi * X['quarter'] / 4)
        return X
    
    @staticmethod
    def get_season(month: int) -> int:
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        elif month in [9, 10, 11]:
            return 3  # Autumn
        return 4





prep = Pipeline([
    ('preprocessor', Preprocessor())
])


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
import numpy as np


dfTrain = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')

dfTrain = dfTrain.drop('id', axis=1)

dfTrain = dfTrain.dropna(subset=['num_sold'])

X = dfTrain.drop('num_sold', axis=1)
y = dfTrain['num_sold'] 


params = {
    'colsample_bytree': 0.9394214279109504,
    'learning_rate': 0.013184103905618631,
    'max_depth': 8,
    'min_child_samples': 5,
    'n_estimators': 878,
    'num_leaves': 147,
    'subsample': 0.5885725864594232
}


modelLGB = lgb.LGBMRegressor(**params)

kf = KFold(n_splits=5, shuffle=True, random_state=rs)

mape_scores = []


for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

    X_train_prep = prep.fit_transform(X_train)
    X_val_prep   = prep.transform(X_val)
                    
    modelLGB.fit(X_train_prep, y_train)

    y_pred = modelLGB.predict(X_val_prep)

    mape_val = mean_absolute_percentage_error(y_val, y_pred)
    mape_scores.append(mape_val)
    print(f"[Fold {fold}] MAPE: {mape_val:.4f}")

print(f"Average MAPE over folds: {np.mean(mape_scores):.4f}")

X_all = X.copy()
y_all = y.copy()

X_all_processed = prep.fit_transform(X_all, y_all)

modelLGB.fit(X_all_processed, y_all)


dfTest = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
dfSub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
dfTest = dfTest.drop('id', axis=1)


dfTest = prep.transform(dfTest)
yPred = modelLGB.predict(dfTest)


submission = pd.DataFrame({
    'id': dfSub['id'], 
    'num_sold': yPred
})
submission.to_csv('submission.csv', index=False)


dfConfirm = pd.read_csv('submission.csv')
dfConfirm.head() 

