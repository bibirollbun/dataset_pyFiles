# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
TIME=time.time()
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import xgboost as xgb
import catboost as cb
import seaborn as sns
import lightgbm as lgb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt



from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error

%matplotlib inline




def process_df(df):#変換用
    df['Sex']=df['Sex'].map({'male':0, 'female':1})#o
    df['Bmi']=df['Weight']/((df['Height']**2)/10000)#BMI#x
    df['Bmr']=(13.397-4.15*df['Sex'])*df['Weight']+(4.799-1.701*df['Sex'])*df['Height']-(5.677-1.347*df['Sex'])*df['Age']+88.362+359.231*df['Sex']#基礎代謝#x
    df['heart_dur']=df['Heart_Rate']*df['Duration']#心拍数×時間#o
    df['intensity']=df['Heart_Rate']/df['Duration']#o
    df['h_rate']=100*df['Heart_Rate']/(220-df['Age'])#最大心拍数に対する比#o
    #df['temp_Curvature']=(train_df['Body_Temp'].mean()-df['Body_Temp'])**2
    #df['temp_2']=df['Body_Temp']**2
    df['bmr_time']=df['Bmr']*df['Duration']#o

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df = df.drop(['Height','Weight'], axis=1)
    return df 

def process_2(df):#変換用2
    df['log_heart']=np.log(df['heart_dur'])
    df['log_inte']=np.log(df['intensity'])
    df['log_hrate']=np.log(df['h_rate'])
    df['log_bmrt']=np.log(df['bmr_time'])
    df['log_bmi']=np.log(df['Bmi'])
    df['log_dur']=np.log(df['Duration'])
    df['log_wei']=np.log(df['Weight'])
    df['log_hei']=np.log(df['Height'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #df = df.drop(['Bmi','Bmr','heart_dur','intensity','h_rate','temp_2','bmr_time'], axis=1)
    return df 
drops=['id','Bmi','Bmr','heart_dur','intensity','h_rate','bmr_time','Duration','Height','Weight','Heart_Rate']

train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
train_df.describe()
X2=process_df(train_df)
X=process_2(X2)
X=X.drop(drops, axis=1)
print(X.head())
temp_mean=train_df['Body_Temp'].mean()
temp_min=min(train_df['Body_Temp'].min(),test_data['Body_Temp'].min())


#df_corr = X.corr()
#print(df_corr)
#X=train_df.drop(['Calories'],axis=1)
#X['temp_Curvature']=(temp_mean-X['Body_Temp'])**2
#sns.scatterplot(x="bmr_time", y="Calories",hue="Sex", data=X)

#X['Sex']=X['Sex'].map({'male':0, 'female':1})
y=train_df['Calories']
y_log=np.log1p(y)
X2_test=process_df(test_data)
X_test=process_2(X2_test)
X_test=X_test.drop(drops, axis=1)
#X_test['temp_Curvature']=(temp_mean-X_test['Body_Temp'])**2





plt.scatter(np.log(X["log_wei"]), y_log)#グラフの概形
plt.title("This is a title")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid(True)


# インデックスと目的変数を定義
#filter_columns = ['PassengerId', 'Survived']

# 説明変数のデータ型を確認
dtypes = X.dtypes
print('説明変数　　　　型')
print(dtypes)
print()

# float型のカラムのみを抽出
float_columns = dtypes[dtypes == 'float64'].index.tolist()
print('数値の説明変数       ：　', float_columns)

# float型以外のカラムを抽出
categorical_columns = dtypes[dtypes != 'float64'].index.tolist()
print('カテゴリカルな説明変数：　', categorical_columns)

float_columns += ['Calories']
X[float_columns].corr()#相関係数


from scipy.stats import chi2_contingency
#カイ二乗検定
chi2_results = []
# クロステーブルの作成
for col in categorical_columns:
    cross_table = pd.crosstab(X[col], X['Calories'])
    chi2, p, _, _ = chi2_contingency(cross_table)
    chi2_results.append((col, p))
    
print(f"変数名,  p値")
# p値が小さい順に並べる
sorted_chi2_results = sorted(chi2_results, key=lambda x: x[1])
sorted_chi2_results


X=X.drop(['Calories'],axis=1)
#X_test=X_test.drop(['id'],axis=1)
#X_test['Sex']=X_test['Sex'].map({'male':0, 'female':1})
print(X.head())
#print(X_test.head())
#id Sex Age Height Weight Duration Heart_Rate Body_Temp Calories

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
ty_l=np.log1p(train_y)

print(time.time()-TIME)


import statsmodels.api as sm
# 切片追加
X_ = sm.add_constant(X)
# 重回帰学習
lr = sm.OLS(y,X_).fit()
# 各係数のp値
lr.pvalues


model_xgb = xgb.XGBRegressor(
    n_estimators=1000,#決定木の個数
    learning_rate=0.01,#学習率
    max_depth=9,#決定木の最大深度
    subsample=0.7, #決定木のサンプルとして取り出す割合
    colsample_bytree=0.7, #説明変数のサンプル抽出比
    min_child_weight=3,#決定木の葉の重みの下限
    reg_alpha=0.001,#L1
    reg_lambda=5,#L2
    gamma=0.001,
    random_state=42,#乱数シード
    #tree_method='gpu_hist',
    #objective='reg:squaredlogerror',
    objective='reg:squarederror',
    eval_metric='rmse'
)#本提出用パラメータ

# xgboostモデルの作成

#学習
model_xgb.fit(train_X,ty_l)

# 学習モデルの評価（RMSEを計算）
y_pred_train = np.exp(model_xgb.predict(train_X))-1
ypt_log=model_xgb.predict(val_X)
y_pred_test = np.exp(ypt_log)-1

#y_pred_train = model.predict(train_X)
#y_pred_test = model.predict(val_X)

print(X.sample())

#print(min(y_pred_train),max(y_pred_train))
print(mean_squared_log_error(train_y, y_pred_train, squared=False),"RMSLE,train")
print(mean_squared_log_error(val_y, y_pred_test, squared=False),"RMSLE,test")
print(time.time()-TIME)


for i in [6,7,8,9,10]:#テスト用パラメータの調子絵
    xgb_model_ep = xgb.XGBRegressor(
        n_estimators=1250,#決定木の個数
        learning_rate=0.01,#学習率
        max_depth=i,#決定木の最大深度
        subsample=0.7, #決定木のサンプルとして取り出す割合
        colsample_bytree=0.7,#説明変数のサンプル抽出比
        reg_alpha=0.1,#L1
        reg_lambda=5,#L2 0.001
        min_child_weight=3,#決定木の葉の重みの下限
        gamma=0.001,
        random_state=42,#乱数シード
        #tree_method='gpu_hist',
        #objective='reg:squaredlogerror',
        objective='reg:squarederror',
        eval_metric='rmse'
    )
    xgb_model_ep.fit(train_X,ty_l)
    ypt_l = xgb_model_ep.predict(val_X)
    ypt=np.exp(ypt_l)-1
    yll=xgb_model_ep.predict(train_X)
    ytr=np.exp(yll)-1
    p=mean_squared_log_error(train_y, ytr, squared=False)
    q=mean_squared_log_error(val_y, ypt, squared=False)
    print(q,p,abs(p-q),i,"RMSLE")
print(time.time()-TIME)  



lgb_params_train = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.016,
    "n_estimators": 1500, 
    "num_leaves": 256,  
    "max_depth": 8, 
    "min_child_samples": 8, 
    "min_split_gain": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.01, 
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbosity": -1,
    "force_col_wise": True
}#lightgbm 提出用パラメータ

# LGBM
model_lgb = lgb.LGBMRegressor(**lgb_params_train)



model_lgb.fit(train_X, ty_l)
ypt_l_lgb = model_lgb.predict(val_X)
yll_lgb = model_lgb.predict(train_X)
ypt3=np.exp(ypt_l_lgb)-1
ytr3=np.exp(yll_lgb)-1
p=mean_squared_log_error(train_y, ytr3, squared=False)
q=mean_squared_log_error(val_y, ypt3, squared=False)
print(q,p,abs(p-q),"RMSLE",time.time()-TIME)


T2=time.time()
for i in []:#[0.5,0.6,0.7,0.8,0.9]:
    lgb_params_ep = {
        "objective": "regression",#目的関数
        "metric": "rmse",#損失関数
        "learning_rate": 0.016,#学習率,計算時間はこの値の大きさに反比例
        "n_estimators": 1500, #正比例
        "num_leaves": 256,  #各木の分岐の個数、正比例？
        "max_depth": 8, #木の最大深度
        "min_child_samples": 8, 
        "min_split_gain": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.01, 
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbosity": -1,
        "force_col_wise": True
    }
    model_lgb_ep = lgb.LGBMRegressor(**lgb_params_ep)
    model_lgb_ep.fit(train_X, ty_l)
    ypt_l = model_lgb_ep.predict(val_X)
    yll = model_lgb_ep.predict(train_X)
    ypt=np.exp(ypt_l)-1
    ytr=np.exp(yll)-1
    p=mean_squared_log_error(train_y, ytr, squared=False)
    q=mean_squared_log_error(val_y, ypt, squared=False)
    print(q,p,abs(p-q),i,"RMSLE",time.time()-T2)
print(time.time()-TIME)  


T3=time.time()
for i in []:#[6,7,8,9,10]:
    model_cat = cb.CatBoostRegressor(
        iterations=1500,
        learning_rate=0.029,
        depth=9,
        subsample=0.6,
        colsample_bylevel=0.8,
        random_seed=42,
        verbose=0  # Set to 100 if you want iteration logs
    )
    model_cat.fit(train_X,ty_l, early_stopping_rounds=20)
    ypt_l = model_cat.predict(val_X)
    ypt=np.exp(ypt_l)-1
    yll=model_cat.predict(train_X)
    ytr=np.exp(yll)-1
    p=mean_squared_log_error(train_y, ytr, squared=False)
    q=mean_squared_log_error(val_y, ypt, squared=False)
    print(q,p,abs(p-q),i,"RMSLE",time.time()-T3)


model_cat = cb.CatBoostRegressor(
    iterations=1500,
    learning_rate=0.029,
    depth=9,
    subsample=0.6,
    colsample_bylevel=0.8,
    random_seed=42,
    verbose=0  # Set to 100 if you want iteration logs
)

model_cat.fit(train_X, ty_l, early_stopping_rounds=20)


predictions = model_cat.predict(val_X)
ex_y=np.exp(predictions)-1
#val_X['xgb_pred']=y_pred_test
#val_X['cat_pred']=ex_y
print(val_X.head())



#final_pred=(y_pred_test+ex_y)/2
para=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
dl=[[0 for j in range(11)] for i in range(11)]
dl2=[[0 for j in range(11)] for i in range(11)]
for i in range(11):
    for j in range(11):
        #pp=max(0,para[i]*y_pred_test+para[j]*ypt3+(1-para[i]-para[j])*ex_y)
        #dl[i][j]=int(mean_squared_log_error(val_y, pp, squared=False)*1000000)
        qq=np.exp(para[i]*ypt_log+para[j]*ypt_l_lgb+(1-para[i]-para[j])*predictions)-1
        dl2[i][j]=int(mean_squared_log_error(val_y, qq, squared=False)*1000000)
#print(dl)
print(dl2)
for i in [0.2,0.23,0.26,0.3]:
    for j in [0.2]:#,0.3,0.4,0.5,0.6,0.7]:
        pp=i*y_pred_test+j*ypt3+(1-i-j)*ex_y
        qq=np.exp(i*ypt_log+j*ypt_l_lgb+(1-i-j)*predictions)-1
        
        #print(mean_squared_log_error(val_y, pp, squared=False),i,j,"RMSLE")
        #print(mean_squared_log_error(val_y, qq, squared=False),i,j,"RMSLE,2")
for i in [0.4]:
    pp=i*y_pred_test+(1-i)*ex_y
    qq=np.exp(i*ypt_log+(1-i)*predictions)-1
    #print(mean_squared_log_error(val_y, pp, squared=False),i,j,"RMSLE")
    #print(mean_squared_log_error(val_y, qq, squared=False),i,j,"RMSLE,2")

#final_pred_2=np.exp((ypt_log+predictions)/2)-1
print(ex_y,y_pred_test,ypt3)
print(val_y)
#print(final_pred)
#print(mean_squared_log_error(val_y, final_pred, squared=False),"ave,RMSLE")
#print(mean_squared_log_error(val_y, final_pred_2, squared=False),"ave,RMSLE,fpred2")
#model.fit(X, y)
#predictions = model.predict(X_test)
print(time.time()-TIME)
#output = pd.DataFrame({'id': test_data.id, 'Calories': predictions})
#output.to_csv('submission.csv', index=False)


#model.fit(X, y)#提出解の作成
y_l=np.log1p(y)

model_xgb.fit(X,y_l)
pre_xgb=model_xgb.predict(X_test)
ex_y1=np.exp(pre_xgb)-1
#predictions = model.predict(X_test)
print(time.time()-TIME,"xgb")

model_cat.fit(X, y_l)
pre_cat = model_cat.predict(X_test)
ex_y3=np.exp(pre_cat)-1
print(time.time()-TIME,"cat")

model_lgb.fit(X, y_l)
pre_lgb=model_lgb.predict(X_test)
ex_y2=np.exp(pre_lgb)-1
print(time.time()-TIME,"lgb")

#val_X['xgb_pred']=y_pred_test
#val_X['cat_pred']=ex_y
#print(val_X.head())
#submit_pred=(ex_y1+ex_y2)/2
#submit_pred=0.375*ex_y1+(1-0.375)*ex_y2
#submit_pred=np.exp(0.6*pre_xgb+(1-0.6)*pre_cat)-1
#submit_pred=0.45*ex_y1+0.225*ex_y2+(1-0.45-0.225)*ex_y3
submit_pred=np.exp(0.4*pre_xgb+0.2*pre_lgb+(1-0.4-0.2)*pre_cat)-1
output = pd.DataFrame({'id': test_data.id, 'Calories': submit_pred})#出力
output.to_csv('submission.csv', index=False)
print(time.time()-TIME)

