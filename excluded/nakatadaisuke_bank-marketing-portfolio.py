import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

import time


class Paths:
    p = "/kaggle/input/playground-series-s5e8/"
    train = p+"train.csv"
    test = p+"test.csv"
    sample = p+"sample_submission.csv"


train = pd.read_csv(Paths.train)
test = pd.read_csv(Paths.test)


con =train["y"].value_counts()[0]
ret =train["y"].value_counts()[1]
per_con = con /(con + ret) * 100
per_ret = ret / (con + ret) * 100
print(f"非加入割合:{int(per_con)}%、加入(申し込み)割合:{int(per_ret)}%")
sns.countplot(x="y",data=train)
plt.show()


train.info()


train.head()


test.head()


def describe_data(data,name):
    print(f"\n{name} shape:{data.shape}")
    print(f"\n{name} 欠損値:\n{data.isnull().sum()}")
    print(f"\n{name} 数値特徴量:\n{data.select_dtypes(include=np.number).columns}")
    print(f"\n{name} カテゴリ特徴量:\n{data.select_dtypes(include='object').columns}")
    
describe_data(train,"train")
describe_data(test,"test")


def nums_describe(df):
    df_ = df.select_dtypes(include = np.number).drop(columns=["id","y"])
    des = df_.describe().T
    des['skewness'] = df_.skew()
    des['kurtosis'] = df_.kurtosis()
    des['count'] = des['count'].astype('int')
    return des

des = nums_describe(train)
des


#数値特徴量ごとに y=0/1 の分布を比較し、ターゲットとの関係を把握する
def num_view():
    cols = train.select_dtypes(include=np.number).columns.drop(["y","id"])
    
    # 外れ値カットを適用したいカラム
    need_cut = ["balance", "duration", "campaign", "previous"]

    n_cols = 2
    n_rows = (len(cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(16, 24))

    for i, col in enumerate(cols, 1):
        plt.subplot(n_rows, n_cols, i)

        if col in need_cut:
            upper = train[col].quantile(0.99)
            min_v = train[col].min()
        else:
            # 外れ値が少ない特徴量は “本来の分布” を見せる
            upper = train[col].max()
            min_v = train[col].min()

        sns.histplot(train[train["y"]==0][col],
                     bins=30, color="blue",
                     binrange=(min_v, upper),
                     alpha=0.6, kde=False, label="0")

        sns.histplot(train[train["y"]==1][col],
                     bins=30, color="red",
                     binrange=(min_v, upper),
                     alpha=0.6, kde=False, label="1")

        plt.title(col)
        plt.legend()

    plt.tight_layout()
    plt.show()


col_y = ["duration","balance"]
plt.figure(figsize=(16,12))
for i,col in enumerate(col_y,1):
    plt.subplot(len(col_y), 1, i)
    sns.boxplot(x='y', y=col, data=train)
    plt.title(f"{col} vs y")
plt.tight_layout()
plt.show()


def cats_view():
    cats = train.select_dtypes(include= "object").columns.tolist()
    plt.figure(figsize=(25,5*len(cats)))
    for i , cat in enumerate(cats):
        plt.subplot(len(cats),2,2*i+1)
        sns.countplot(x=cat,hue="y",data=train)
        plt.title(f"{cat} countplot (y)")
        plt.xticks(rotation=45)
        plt.subplot(len(cats),2,2*i+2)
        col_per = train[cat].value_counts(normalize=True)
        plt.pie(col_per,labels=col_per.index,autopct="%1.1f%%")
        plt.title(f"{cat} proportion")
    
    plt.tight_layout()
    plt.show()

cats_view()


def feature_engineering(df):
    df_ = df.copy()
    
    #特徴量追加
    #年齢をグループ化
    #age_order = ['13-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    #df["age_group"] = df["age"].apply(create_age_groups)
    #df_['age_group'] = pd.Categorical(df["age_group"], 
    #                                    categories=age_order)

    # ===== 数値特徴量の変換 =====
    df_["balance_log"] = np.log1p(df_["balance"].clip(lower=0))
    df_['previous_log'] = np.log1p(df_['previous'])
    df_['campaign_log'] = np.log1p(df_['campaign'])
    
    
    #カテゴリ×カテゴリ
    df_["job_edu"] = df_["job"].astype(str) + "_" + df_["education"].astype(str)
    
    #2値分類化
    df_["pdays_missing"] = (df_["pdays"] == -1).astype(int)
    
    #カテゴリ化
    #df_["campaign_cat"] = df["campaign"].apply(lambda x : f"{x}回" if x < 5 else "5回以上")
    
    all_cat_cols = df_.select_dtypes(include="object").columns.tolist()
    for cal in all_cat_cols:
        df_[cal] = df_[cal].astype("category")

    return df_
    
x_train = train.drop(columns=["id","y"])
x_test = test.drop(columns = ["id"])
y_train = train["y"]

x_train = feature_engineering(x_train)
x_test = feature_engineering(x_test)


params = dict(
    n_estimators=10000, 
    learning_rate=0.07,
    num_leaves=64,
    min_child_samples =128,
    subsample=0.8,
    colsample_bytree=0.8,
    subsample_freq=1,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="binary",
    metric="auc",
    random_state=123,
    max_bin=3600,
)


print(x_train.shape)
print(x_test.shape)


print(x_train.columns)
print(x_test.columns)


n_splits = 5

cv = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state = 123)

metrics = []
models = []
imp = pd.DataFrame()

cat_models=[]

cat_cols = x_train.select_dtypes(include="category").columns.tolist()

start_all = time.time()
for nfold ,(train_idx, val_idx) in enumerate(cv.split(x_train,y_train)):
    print("="*10,nfold,"="*10)
    start_fold = time.time()
    x_tr, y_tr = x_train.iloc[train_idx],y_train.iloc[train_idx]
    x_va, y_va = x_train.iloc[val_idx],y_train.iloc[val_idx]

    model = lgb.LGBMClassifier(**params,verbose=-1)
    model.fit(x_tr,y_tr,categorical_feature=cat_cols,eval_set = [(x_tr,y_tr),(x_va,y_va)],callbacks=[lgb.early_stopping(stopping_rounds=200,verbose=True),lgb.log_evaluation(200)],)

    # foldごとの split / gain を取得（Boosterから）
    names = model.booster_.feature_name()
    split_imp = model.booster_.feature_importance(importance_type="split")
    gain_imp  = model.booster_.feature_importance(importance_type="gain")
    
    
    y_tr_pred = model.predict_proba(x_tr)[:,1]
    y_va_pred = model.predict_proba(x_va)[:,1]

    metric_tr = roc_auc_score(y_tr,y_tr_pred)
    metric_va = roc_auc_score(y_va,y_va_pred)

    metrics.append([nfold,metric_tr,metric_va])
    models.append(model)

    _imp = pd.DataFrame({
        "col":x_train.columns,
        "imp":model.feature_importances_,
        "nfold":nfold
    })
    imp = pd.concat([imp,_imp],axis = 0,ignore_index= True)
    
    # foldごとの所要時間を出力
    end_fold = time.time()
    print(f"Fold {nfold} time: {end_fold - start_fold:.2f} seconds")

# 全体の所要時間を出力
end_all = time.time()
print(f"Total training time: {end_all - start_all:.2f} seconds")


metrics_array = np.array(metrics)
print("[cv] tr:{:.2f}+-{:.4f}, va:{:.2f}+-{:.4f}".format(metrics_array[:,1].mean(),metrics_array[:,1].std(),
                                                        metrics_array[:,2].mean(), metrics_array[:,1].std(),))


imp_df = imp.groupby("col")["imp"].agg(["mean","std"])
imp_df.columns = ["imp","imp_std"]
imp_df = imp_df.sort_values(by="imp",ascending = False)
imp_df.head(30)


def average_proba(data,models):
    preds =[]
    for mod in models:
        y_test_pred = mod.predict_proba(x_test)[:,1]
        preds.append(y_test_pred)

    preds = np.array(preds)
    preds = np.mean(preds,axis=0)

    return preds


y_test_pred = average_proba(x_test,models)
df_submit = pd.DataFrame({"id":test["id"],"y":y_test_pred})
df_submit


df_submit.to_csv("submission.csv",index = False)

