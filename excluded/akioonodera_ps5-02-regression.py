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


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
df_origin = pd.read_csv('/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv')


sample_submission


df_train


df_origin


df_test


df_train.info()


df_origin.info()


df_test.info()


# 欠損値の確認
# Check missing values

print('train')
msno.matrix(df=df_train, figsize=(10,6), color=(0,.3,.3))


print('origin')
msno.matrix(df=df_origin, figsize=(10,6), color=(0,.3,.3))


print('test')
msno.matrix(df=df_test, figsize=(10,6), color=(0,.3,.3))


df_train['Brand'].unique().tolist()


df_origin['Brand'].unique().tolist()


df_test['Brand'].unique().tolist()


df_train['Material'].unique().tolist()


df_origin['Material'].unique().tolist()


df_test['Material'].unique().tolist()


df_train['Size'].unique().tolist()


df_origin['Size'].unique().tolist()


df_test['Size'].unique().tolist()


df_train['Compartments'].unique().tolist()


df_origin['Compartments'].unique().tolist()


df_test['Compartments'].unique().tolist()


df_train['Laptop Compartment'].unique().tolist()


df_origin['Laptop Compartment'].unique().tolist()


df_test['Laptop Compartment'].unique().tolist()


df_train['Waterproof'].unique().tolist()


df_origin['Waterproof'].unique().tolist()


df_test['Waterproof'].unique().tolist()


df_train['Style'].unique().tolist()


df_origin['Style'].unique().tolist()


df_test['Style'].unique().tolist()


df_train['Color'].unique().tolist()


df_origin['Color'].unique().tolist()


df_test['Color'].unique().tolist()


df_train[['Weight Capacity (kg)', 'Price']].describe()


df_origin[['Weight Capacity (kg)', 'Price']].describe()


df_test[['Weight Capacity (kg)']].describe()


# df_trainとdf_originを結合
# Concat df_train and df_origin

df_t_o = pd.concat([df_train, df_origin], ignore_index=True)
df_t_o = df_t_o.drop_duplicates() # 重複データを削除

df_t_o = df_t_o.drop(columns=['id']) # idの列を削除
df_t_o


# 欠損値を含む行を削除
# Remove rows with missing values

df_t_o_dn = df_t_o.dropna()
df_t_o_dn


msno.matrix(df=df_t_o_dn, figsize=(10,6), color=(0,.3,.3))


# get_dummiesメソッドを使ってカラムを分割（df_testには欠損値が含まれているがそのまま処理）
# Split columns using get_dummies method

columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
df_t_o_dn_dum = pd.get_dummies(df_t_o_dn, columns=columns, dtype=int)
df_test_dum = pd.get_dummies(df_test, columns=columns, dtype=int)
df_test_dum = df_test_dum.drop(columns=['id']) # idの列を削除
df_t_o_dn_dum


df_test_dum


# トレーニングデータとテストデータの分布を可視化（ヒストグラム）
feat_ary = ['Compartments', 'Weight Capacity (kg)']
for feat in feat_ary:
    plt.figure(figsize=(12,3))
    ax1 = plt.subplot(1,2,1)
    df_t_o_dn_dum[feat].plot(kind='hist', bins=50, color='blue')
    plt.title(feat + ' / train + origin')
    ax2 = plt.subplot(1,2,2, sharex=ax1) # sharexでｘ軸を同じにする
    df_test_dum[feat].plot(kind='hist', bins=50, color='green')
    plt.title(feat + ' / test')
    plt.show()


plt.figure(figsize=(12,3))
df_t_o_dn_dum['Price'].plot(kind='hist', bins=50, color='blue')
plt.title('Price / train')
plt.show()


# Heatmap(train + origin)

corr = df_t_o_dn_dum.corr().round(2)
plt.figure(figsize=(20,10))
sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=False, annot=True, cmap='coolwarm')
plt.show()


# Heatmap(test)

corr = df_test_dum.corr().round(2)
plt.figure(figsize=(20,10))
sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=False, annot=True, cmap='coolwarm')
plt.show()


X = df_t_o_dn_dum.drop(columns=['Price'])
value = df_t_o_dn_dum['Price']


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
random_state_ary = [1]
objective = 'regression'
metric = 'rmse'
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
learning_rate = 0.0005 # 0.0005
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
        'learning_rate': learning_rate,
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
        {'objective': 'regression', 'metric': 'rmse', 'feature_pre_filter': False, 'lambda_l1': 0.15541647760091407, 'lambda_l2': 1.0058821881464524e-07, 'num_leaves': 31, 'feature_fraction': 1.0, 'bagging_fraction': 1.0, 'bagging_freq': 0, 'min_child_samples': 100, 'num_iterations': 10}
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
    #plt.title('Feature importance')


count = 0
for i in range(len(opt_ary)):
    count += 1
    result_ary = opt_ary[i]['result_ary']
    random_state = opt_ary[i]['random_state']

    model = result_ary['model']
    show_FI(model, random_state, count)


x = df_test_dum


result = model.predict(x)
result


sample_submission['Price'] = result
print(sample_submission)
sample_submission.to_csv('submission.csv', index=False)

