import numpy as np
import pandas as pd

import sys
import datetime
import time
import math

import lightgbm as lgb
!pip install optuna optuna-integration
import optuna.integration.lightgbm as lgbo

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer
le = preprocessing.LabelEncoder()
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error # 平均絶対誤差
from sklearn.metrics import mean_squared_error # 平均二乗誤差
from sklearn.metrics import mean_squared_log_error # 対数平均二乗誤差
from sklearn.metrics import r2_score # 決定係数
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import missingno as msno
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


sample_submission


sample_submission.info()


df_train


df_test


df_train.info()


df_test.info()


country_list = df_train['country'].unique().tolist()
country_list


df_test['country'].unique().tolist()


store_list = df_train['store'].unique().tolist()
store_list


df_test['store'].unique().tolist()


product_list = df_train['product'].unique().tolist()
product_list


df_test['product'].unique().tolist()


df_train['num_sold'].describe()


# 欠損値の確認
# Check missing values

print('train')
msno.matrix(df=df_train, figsize=(10,6), color=(0,.3,.3))


print('test')
msno.matrix(df=df_test, figsize=(10,6), color=(0,.3,.3))


# dateを日付型に変更

df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date'] = pd.to_datetime(df_test['date'])


# dateを分割
# split date to year, month, day

df_train['year'] = df_train['date'].dt.year
df_train['month'] = df_train['date'].dt.month
df_train['day'] = df_train['date'].dt.day
df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day


# 曜日のカラムを追加
df_train['dayofweek'] = df_train['date'].dt.dayofweek
df_test['dayofweek'] = df_test['date'].dt.dayofweek

ary =[]
for i in range(len(df_train)):
    if df_train['dayofweek'][i] >= 5:
        is_weekend = 1
    else:
        is_weekend = 0
    ary.append(is_weekend)
df_train['is_weekend'] = ary
ary =[]
for i in range(len(df_test)):
    if df_test['dayofweek'][i] >= 5:
        is_weekend = 1
    else:
        is_weekend = 0
    ary.append(is_weekend)
df_test['is_weekend'] = ary



# 月末のカラムを追加


ary = []
for i in range(len(df_train)):
    year = int(df_train['year'][i])
    month = int(df_train['month'][i])
    day = int(df_train['day'][i])
    is_monthend = 0
    if year % 400 == 0 or (year % 100 != 0 and year % 4 == 0):
        # 閏年
        if month == 2 and day == 29:
            is_monthend = 1
    else:
        # 閏年以外
        if month == 2 and day == 28:
            is_monthend = 1
    if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
        if day == 31:
            is_monthend = 1
    elif month == 4 or month == 6 or month == 9 or month == 11:
        if day == 30:
            is_monthend = 1
    ary.append(is_monthend)
df_train['is_monthend'] = ary

ary = []
for i in range(len(df_test)):
    year = int(df_test['year'][i])
    month = int(df_test['month'][i])
    day = int(df_test['day'][i])
    is_monthend = 0
    if year % 400 == 0 or (year % 100 != 0 and year % 4 == 0):
        # 閏年
        if month == 2 and day == 29:
            is_monthend = 1
    else:
        # 閏年以外
        if month == 2 and day == 28:
            is_monthend = 1
    if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
        if day == 31:
            is_monthend = 1
    elif month == 4 or month == 6 or month == 9 or month == 11:
        if day == 30:
            is_monthend = 1
    ary.append(is_monthend)
df_test['is_monthend'] = ary


group_columns = ['country', 'store', 'product']
plots = {}

for col in group_columns:
    grouped = df_train.groupby([col, 'date'])['num_sold'].sum().unstack(0)
    
    # Plot each category
    grouped.plot(title=f'Date vs Num_Sold for {col}', figsize=(10, 6))
    plt.ylabel('Num Sold')
    plt.xlabel('Date')
    plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# get_dummiesメソッドを使ってカラムを分割
# Split columns using get_dummies method

columns = ['country', 'store', 'product']
df_train_dum = pd.get_dummies(df_train, columns=columns, dtype=int)
df_test_dum = pd.get_dummies(df_test, columns=columns, dtype=int)
df_train_dum


# 欠損値を含む行を削除

df_train_dum_dn = df_train_dum.dropna()
df_train_dum_dn


df_test_dum


# トレーニングデータとテストデータの分布を可視化（ヒストグラム）
feat_ary = ['year', 'month', 'day', 'dayofweek', 'is_monthend', 'country_Canada', 'country_Finland', 'country_Italy', 'country_Norway', 'country_Singapore', 'store_Discount Stickers', 'store_Premium Sticker Mart', 'store_Stickers for Less', 'product_Holographic Goose', 'product_Kaggle', 'product_Kaggle Tiers', 'product_Kerneler', 'product_Kerneler Dark Mode']
for feat in feat_ary:
    plt.figure(figsize=(12,3))
    ax1 = plt.subplot(1,2,1)
    df_train_dum_dn[feat].plot(kind='hist', bins=50, color='blue')
    plt.title(feat + ' / train')
    ax2 = plt.subplot(1,2,2, sharex=ax1) # sharexでｘ軸を同じにする
    df_test_dum[feat].plot(kind='hist', bins=50, color='green')
    plt.title(feat + ' / test')
    plt.show()


plt.figure(figsize=(12,3))
df_train_dum_dn['num_sold'].plot(kind='hist', bins=50, color='blue')
plt.title('num_sold / train')
plt.show()


# Heatmap(train)

corr = df_train_dum_dn.corr().round(2)
plt.figure(figsize=(20,10))
sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=False, annot=True, cmap='coolwarm')
plt.show()


# Heatmap(train)

corr = df_test_dum.corr().round(2)
plt.figure(figsize=(20,10))
sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=False, annot=True, cmap='coolwarm')
plt.show()


X = df_train_dum_dn.drop(columns=['id', 'date', 'year', 'num_sold'])
value = df_train_dum_dn['num_sold']


# 学習用データと検証用データを作成する関数

def make_lgb_data(test_size, random_state, objective, metric, X, value):
    X_train, X_test, t_train, t_test = train_test_split(
        X,
        value,
        test_size=test_size,
        random_state=random_state
    )

    lgb_train = lgb.Dataset(
        X_train,
        t_train
    )
    
    lgb_eval = lgb.Dataset(
        X_test,
        t_test,
        reference=lgb_train
    )
    
    dic_return = {
        'X_train' : X_train,
        'X_test' : X_test,
        't_train' : t_train,
        't_test' : t_test,
        'lgb_train' : lgb_train,
        'lgb_eval' : lgb_eval
    }
    
    return dic_return


# optuna を使ってパラメーターチューニングする関数

def tuneParams(test_size, random_state, objective, metric, X, value):

    opt_params = {
        "objective" : objective,
        "metric" : metric
    }
    
    dic = make_lgb_data(test_size, random_state, objective, metric, X, value)
    X_train = dic['X_train']
    X_test = dic['X_test']
    t_train = dic['t_train']
    t_test = dic['t_test']
    lgb_train = dic['lgb_train']
    lgb_eval = dic['lgb_eval']
    
    opt = lgbo.train(
        opt_params,
        lgb_train,
        valid_sets = lgb_eval,
        #verbose_eval = False,
        num_boost_round = 10,
        #early_stopping_rounds = 10
    )
    
    return opt


# 学習（モデル作成）関数

def make_model(X, value, test_size, random_state, objective, metric, learning_rate, num_iterations, max_depth, paramObj):
    dic = make_lgb_data(test_size, random_state, objective, metric, X, value)
    lgb_train = dic['lgb_train']
    lgb_eval = dic['lgb_eval']
    X_test = dic['X_test'] # 検証用
    t_test = dic['t_test'] # 　〃

    params = {
        #'device': 'gpu',
        'task': 'train',
        'objective': objective,
        'metric': metric,
        #'num_class': num_class,
        'boosting_type': 'gbdt',
        'learning_rate': learning_rate,
        'num_iterations': num_iterations,
        'max_depth': max_depth,
        'feature_pre_filter': paramObj['feature_pre_filter'],
        'lambda_l1': paramObj['lambda_l1'],
        'lambda_l2': paramObj['lambda_l2'],
        'num_leaves': paramObj['num_leaves'],
        'feature_fraction': paramObj['feature_fraction'],
        'bagging_fraction': paramObj['bagging_fraction'],
        'bagging_freq': paramObj['bagging_freq'],
        'min_child_samples': paramObj['min_child_samples'],
        'verbosity': -1
    }

    evaluation_results = {}               # 学習の経過を保存するためのオブジェクト
    model = lgb.train(
        params,
        valid_names=['train', 'valid'],   # 学習経過で表示する名称
        valid_sets=[lgb_train, lgb_eval], # モデル検証のデータセット
#        evals_result=evaluation_results,
        train_set=lgb_train,
        callbacks=[lgb.early_stopping(1000),
                   lgb.record_evaluation(evaluation_results),# 学習の経過を保存
                   lgb.log_evaluation(1000) ]
    )
    
    resultObj = {'evaluation_results' : evaluation_results,
                 'model' : model,
                 'X_test' : X_test, # 検証用
                 't_test' : t_test} #   〃
    return resultObj


#----------------------------------------------------
# 基本設定（Base setting）
#----------------------------------------------------
test_size = 0.2
random_state_ary = [0]
objective = 'regression_l1'
metric = 'l1'
#num_class = 4 # FIX

#----------------------------------------------------
# for optuna
#----------------------------------------------------
optuna_switch = 'off'
opt_count = 1
num_choose = 1 # FIX

if opt_count < num_choose:
    num_choose = opt_count

#----------------------------------------------------
# for lightGBM
#----------------------------------------------------
learning_rate = 0.01 # 0.0005
num_iterations = 200000
max_depth = -1


# optuna

opt_ary = []
if optuna_switch == 'on':
    count_rs = 0
    for random_state in random_state_ary:
        param_ary = []
        for i in range(opt_count):
            count_rs += 1
            print()
            print('=' * 100)
            print(f'Round : {i + 1} / {opt_count} (random_state : {random_state}) ---------->')
            print(f'(Total Round {count_rs} / {len(random_state_ary) * opt_count})')
            print()
            opt = tuneParams(test_size, random_state, objective, metric, X, value)
            score = opt.best_score['valid_0'][metric]
            dic = {'score' : score, 'params' : opt.params}
            param_ary.append(dic)

        # スコアの高い順にソート
        #param_ary = sorted(param_ary, key=lambda x: x['score'], reverse=True)
        # スコアの低い順にソート
        param_ary = sorted(param_ary, key=lambda x: x['score'], reverse=False)
        
        opt_ary.append({'random_state' : random_state, 'param_ary' : param_ary})


if optuna_switch == 'on':
    total_count = 0
    for inner in opt_ary:
        random_state = inner['random_state']
        param_ary = inner['param_ary']
        count = 0
        for dic in param_ary:
            score = dic['score']
            params = dic['params']
            total_count += 1
            count += 1
            print('')
            print('=' * 80)
            print(f'Round : {count} / {len(param_ary)} (random_state : {random_state})')
            print(f'Total Round : {total_count} / {len(opt_ary) * len(param_ary)}')
            print(f'val_score : {score}')
            print(f'params : {params}')
            print('=' * 80)
            print('')


if optuna_switch == 'on':
    params = {
        'task': 'train',
        'objective': objective,
        'metric': metric,
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'num_iterations': 1000000,
        'max_depth': -1,
         # チューニングされたパラメータをセット
        'feature_pre_filter': opt.params['feature_pre_filter'],
        'lambda_l1': opt.params['lambda_l1'],
        'lambda_l2': opt.params['lambda_l2'],
        'num_leaves': opt.params['num_leaves'],
        'feature_fraction': opt.params['feature_fraction'],
        'bagging_fraction': opt.params['bagging_fraction'],
        'bagging_freq': opt.params['bagging_freq'],
        'min_child_samples': opt.params['min_child_samples'],
        'verbosity': -1
    }
elif optuna_switch != 'on':
    # optunaを使わないときはここにパラメーターをセット（注意！ random_state_aryと同じ順に並べる）
    # In case of unusing optuna, set params here(Caution! Set in the same order as the corresponding "random_state")
    #===============================================================================
    param_ary = [
        {'objective': 'regression_l1', 'metric': 'l1', 'feature_pre_filter': False, 'lambda_l1': 0.0013176658375885523, 'lambda_l2': 2.4780915104704975e-07, 'num_leaves': 253, 'feature_fraction': 0.9520000000000001, 'bagging_fraction': 0.8193198822322743, 'bagging_freq': 3, 'min_child_samples': 100, 'num_iterations': 10}
    ]
    #===============================================================================
    opt_ary = []
    for i in range(len(param_ary)):
        random_state = random_state_ary[i]
        opt_ary.append({'random_state' : random_state, 'param_ary' : [param_ary[i]]})
    opt_count = len(opt_ary)
    num_choose = 1


count = 0
for j in range(len(opt_ary)):
    count += 1

    random_state = opt_ary[j]['random_state']
    param_ary = opt_ary[j]['param_ary']
    param_ary = param_ary[0 : num_choose]
    
    print()
    print('='*80)
    print(f'random_state : {random_state} ---------->')
    print(f'(Total Round : {count} / {len(param_ary) * len(opt_ary)})')

    if optuna_switch == 'on':
        score = param_ary[0]['score']
        print(f'opt score : {score}')
        paramObj = param_ary[0]['params']
    else:
        paramObj = param_ary[0]

    print(f'params : {paramObj}')
    print('-'*80)
    resultObj = make_model(X, value, test_size, random_state, objective, metric, learning_rate, num_iterations, max_depth, paramObj)
        
    opt_ary[j]['result_ary'] = resultObj


count = 0
for i in range(len(opt_ary)):
    count += 1
    random_state = opt_ary[i]['random_state']
    resultObj = opt_ary[i]['result_ary']
    print()
    print(f'random_state {random_state}')
    #print(f'(Total count {total_count} / {len(opt_ary) * len(result_ary)})')

    evaluation_results = resultObj['evaluation_results']

    plt.plot(evaluation_results['train'][metric], label='train')
    plt.plot(evaluation_results['valid'][metric], label='valid')
    plt.ylabel(metric)
    plt.xlabel('Boosting round')
    plt.title('Training performance')
    plt.legend()
    plt.show()


# Feature importance
def show_FI(model, random_state, count):
    fig, ax = plt.subplots(figsize=(10, 10))
    lgb.plot_importance(model, ax=ax)
    plt.title('random_state : ' + str(random_state) + ' (Round' + str(count) + ')')


count = 0
for i in range(len(opt_ary)):
    count += 1
    result_ary = opt_ary[i]['result_ary']
    random_state = opt_ary[i]['random_state']

    model = result_ary['model']
    show_FI(model, random_state, count)


x = df_test_dum.drop(columns=['id', 'date', 'year'])


result = model.predict(x)
result


sample_submission['num_sold'] = result
print(sample_submission)
sample_submission.to_csv('submission.csv', index=False)

