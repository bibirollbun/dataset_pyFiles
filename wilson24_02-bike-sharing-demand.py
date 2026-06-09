import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import json
import sklearn
import re

#sklearn library
# 1.model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold

# 2.preprocessing
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

mpl.rc('font',size=12)
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity="all"


train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')


train.head()


test.head()


train.shape, test.shape


submission = pd.read_csv('./data/sampleSubmission.csv')
submission.head()


train.info()


test.info()


df = pd.read_csv('./data/train.csv')
df.head()


df['datetime'] = pd.to_datetime(df['datetime'])
df.info()


df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['weekday'] = df['datetime'].dt.weekday
df['weekday_name'] = df['datetime'].dt.day_name()
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['second'] = df['datetime'].dt.second


df.head()


# 분석을 위해 season, weather 피처 문자형으로 변환
df['season'] = df['season'].replace({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
df['weather'] = df['weather'].replace({1: 'Clear', 2: 'Mist, Cloudy', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'})


df.head()


sns.displot(x='count', data=df)
plt.title("Target Distribution")


sns.displot(np.log(df['count']))
plt.title("Target Log Distribution")


# datetime 시각화
mpl.rc('font',size=6)
figure, axes = plt.subplots(nrows = 6, ncols = 2)
plt.tight_layout()


sns.barplot(data = df, x='year', y='count', ax = axes[0,0])
sns.barplot(data = df, x='month', y='count', ax = axes[0,1])
sns.barplot(data = df, x='day', y='count', ax = axes[1,0])
sns.barplot(data = df, x='hour', y='count', ax = axes[1,1])
sns.barplot(data = df, x='minute', y='count', ax = axes[2,0])
sns.barplot(data = df, x='second', y='count', ax = axes[2,1])
sns.barplot(data = df, x='weekday_name', y='count', ax = axes[3,0])
sns.barplot(data = df, x='season', y='count', ax = axes[3,1])
sns.barplot(data = df, x='weather', y='count', ax = axes[4,0])
sns.barplot(data = df, x='holiday', y='count', ax = axes[4,1])
sns.barplot(data = df, x='workingday', y='count', ax = axes[5,0])


# 수치형 변수들과 Target의 상관계수
corr = df[['temp','atemp','humidity','windspeed','count']].corr()
sns.heatmap(corr, annot=True, vmax=1.0,vmin=-1.0)


# windspeed와 count의 산점도
# 풍속이 0인 값이 많다 => 풍속이 0일수 있는가?
plt.scatter(df['windspeed'], df['count'], color='blue', alpha=0.7)


plt.scatter(df['temp'], df['atemp'], color='blue', alpha=0.7)


train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')


df = pd.concat([train,test],ignore_index=True)
df.head()


df['datetime'] = pd.to_datetime(df['datetime'])
df.info()


df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['weekday'] = df['datetime'].dt.weekday
df['hour'] = df['datetime'].dt.hour


df.info()


# 필요 없는 피처 제거
df.drop(['datetime','casual','registered','windspeed','month'],axis=1,inplace=True)
df.info()


df.head()


# 데이터 분리(train, test)
train = df[~pd.isnull(df['count'])]
test = df[pd.isnull(df['count'])]


train.shape


test.head()


X_test = test.drop(columns = ['count'])


X = train.drop(columns = ['count'])
y = train['count'].values
X.head()
y


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1,random_state=42)


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 모델 리스트
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Support Vector Regressor': SVR(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42),
    'LightGBM': LGBMRegressor(random_state=42),
}


# 평가지표 (RMSLE)
def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 로그 변환 (log1p = log(x + 1)과 동일)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    
    # 로그 차이의 제곱 평균 후 제곱근
    rmsle_value = np.sqrt(np.mean((log_true - log_pred) ** 2))
    
    return rmsle_value


# 모델 학습 및 평가
results = {}
for model_name, model in models.items():
    log_y = np.log(y_train)
    model.fit(X_train, log_y)
    preds = model.predict(X_valid)
    exp_preds = np.exp(preds)
    result_rmsle = rmsle(y_valid, exp_preds)
    results[model_name] = result_rmsle
    print(f"{model_name} RMSLE : {result_rmsle:.4f}")


import optuna
from lightgbm import LGBMRegressor


# Optuna objective 함수 정의
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
    }
    
    model = LGBMRegressor(**params)
    
    # 모델 훈련 후 예측
    log_y = np.log(y_train)
    model.fit(X_train, log_y)
    preds = model.predict(X_valid)
    exp_preds = np.exp(preds)
    result_rmsle = rmsle(y_valid, exp_preds)
    
    return result_rmsle

# Optuna Study 생성
study = optuna.create_study(direction="minimize")  # RMSLE는 최소화해야 하므로 'minimize' 설정
study.optimize(objective, n_trials=100)

# 최적 하이퍼파라미터 출력
print("Best parameters:", study.best_params)
print("Best RMSLE:", study.best_value)

# Train the best model
best_params = study.best_params
best_model = LGBMRegressor(**best_params, random_state=42)
log_y = np.log(y_train)
best_model.fit(X_train, log_y)

# Valid Set 성능 확인
val_preds = best_model.predict(X_valid)
exp_val_preds = np.exp(val_preds)
result_rmlse = rmsle(y_valid, exp_val_preds)
print("Final validation accuracy with best model: ", result_rmlse)


test_preds = best_model.predict(X_test)
exp_test_preds = np.exp(test_preds)
exp_test_preds


submission = pd.read_csv('./data/sampleSubmission.csv')
submission.head()


submission['count'] = exp_test_preds


submission.to_csv('./week2_submission.csv',index=False)

