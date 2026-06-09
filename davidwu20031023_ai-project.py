# %% Cell 0 — 安裝所需套件
!pip install polars tqdm keras torch colorama lightgbm xgboost catboost
!pip install mord


import numpy as np
import pandas as pd
import os
import re
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim

from colorama import Fore, Style
from IPython.display import clear_output
import warnings
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None


target_labels = ['None', 'Mild', 'Moderate', 'Severe']


season_dtype = pl.Enum(['Spring', 'Summer', 'Fall', 'Winter'])

train = (
    pl.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
    .with_columns(pl.col('^.*Season$').cast(season_dtype))
)

test = (
    pl.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
    .with_columns(pl.col('^.*Season$').cast(season_dtype))
)


supervised_usable = (
    train
    .filter(pl.col('sii').is_not_null())
)

missing_count = (
    supervised_usable
    .null_count()
    .transpose(include_header=True,
               header_name='feature',
               column_names=['null_count'])
    .sort('null_count', descending=True)
    .with_columns((pl.col('null_count') / len(supervised_usable)).alias('null_ratio'))
)
plt.figure(figsize=(6, 15))
plt.title(f'Missing values over the {len(supervised_usable)} samples which have a target')
plt.barh(np.arange(len(missing_count)), missing_count.get_column('null_ratio'), color='coral', label='missing')
plt.barh(np.arange(len(missing_count)), 
         1 - missing_count.get_column('null_ratio'),
         left=missing_count.get_column('null_ratio'),
         color='darkseagreen', label='available')
plt.yticks(np.arange(len(missing_count)), missing_count.get_column('feature'))
plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
plt.xlim(0, 1)
plt.legend()
plt.show()


print(train.select(pl.col('PCIAT-PCIAT_Total').is_null() == pl.col('sii').is_null()).to_series().mean())

(train
 .select(pl.col('PCIAT-PCIAT_Total'))
 .group_by(train.get_column('sii'))
 .agg(pl.col('PCIAT-PCIAT_Total').min().alias('PCIAT-PCIAT_Total min'),
      pl.col('PCIAT-PCIAT_Total').max().alias('PCIAT-PCIAT_Total max'),
      pl.col('PCIAT-PCIAT_Total').len().alias('count'))
 .sort('sii')
)


print('Columns missing in test:')
print([f for f in train.columns if f not in test.columns])


vc = train.get_column('Basic_Demos-Enroll_Season').value_counts()
plt.pie(vc.get_column('count'), labels=vc.get_column('Basic_Demos-Enroll_Season'))
plt.title('Season of enrollment')
plt.show()


vc = train.get_column('Basic_Demos-Sex').value_counts()
plt.pie(vc.get_column('count'), labels=['boys', 'girls'])
plt.title('Sex of participant')
plt.show()


_, axs = plt.subplots(2, 1, sharex=True)
for sex in range(2):
    ax = axs.ravel()[sex]
    vc = train.filter(pl.col('Basic_Demos-Sex') == sex).get_column('Basic_Demos-Age').value_counts()
    ax.bar(vc.get_column('Basic_Demos-Age'),
           vc.get_column('count'),
           color=['lightblue', 'coral'][sex],
           label=['boys', 'girls'][sex])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('count')
    ax.legend()
plt.suptitle('Age distribution')
axs.ravel()[1].set_xlabel('years')
plt.show()


_, axs = plt.subplots(2, 1, sharex=True, sharey=True)
for sex in range(2):
    ax = axs.ravel()[sex]
    vc = train.filter(pl.col('Basic_Demos-Sex') == sex).get_column('sii').value_counts()
    ax.bar(vc.get_column('sii'),
           vc.get_column('count') / vc.get_column('count').sum(),
           color=['lightblue', 'coral'][sex],
           label=['boys', 'girls'][sex])
    ax.set_xticks(np.arange(4), target_labels)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel('count')
    ax.legend()
plt.suptitle('Target distribution')
axs.ravel()[1].set_xlabel('Severity Impairment Index (sii)')
plt.show()


plt.figure(figsize=(14, 12))
corr_matrix = supervised_usable.select([
    'PCIAT-PCIAT_Total', 'Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-BMI', 
    'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
    'Physical-Diastolic_BP', 'Physical-Systolic_BP', 'Physical-HeartRate',
    'PreInt_EduHx-computerinternet_hoursday', 'SDS-SDS_Total_T', 'PAQ_A-PAQ_A_Total',
    'PAQ_C-PAQ_C_Total', 'Fitness_Endurance-Max_Stage', 'Fitness_Endurance-Time_Mins','Fitness_Endurance-Time_Sec',
    'FGC-FGC_CU', 'FGC-FGC_GSND','FGC-FGC_GSD','FGC-FGC_PU','FGC-FGC_SRL','FGC-FGC_SRR','FGC-FGC_TL','BIA-BIA_Activity_Level_num', 
    'BIA-BIA_BMC', 'BIA-BIA_BMI', 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
    'BIA-BIA_FFMI','BIA-BIA_FMI', 'BIA-BIA_Fat','BIA-BIA_Frame_num','BIA-BIA_ICW','BIA-BIA_LDM','BIA-BIA_LST',
    'BIA-BIA_SMM','BIA-BIA_TBW'
    # Add other relevant columns
]).to_pandas().corr()

sii_corr = corr_matrix['PCIAT-PCIAT_Total'].drop('PCIAT-PCIAT_Total')
filtered_corr = sii_corr[(sii_corr > 0.1) | (sii_corr < -0.1)]

print(filtered_corr)

plt.figure(figsize=(8, 6))
filtered_corr.sort_values().plot(kind='barh', color='coral')
plt.title('Features with Correlation > 0.1 or < -0.1 with PCIAT-PCIAT_Total')
plt.xlabel('Correlation coefficient')
plt.ylabel('Features')
plt.show()


actigraphy = pl.read_parquet('/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet/id=0417c91e/part-0.parquet')
actigraphy


def analyze_actigraphy(id, only_one_week=False, small=False):
    actigraphy = pl.read_parquet(f'/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet/id={id}/part-0.parquet')
    day = actigraphy.get_column('relative_date_PCIAT') + actigraphy.get_column('time_of_day') / 86400e9
    sample = train.filter(pl.col('id') == id)
    age = sample.get_column('Basic_Demos-Age').item()
    sex = ['boy', 'girl'][sample.get_column('Basic_Demos-Sex').item()]
    actigraphy = (
        actigraphy
        .with_columns(
            (day.diff() * 86400).alias('diff_seconds'),
            (np.sqrt(np.square(pl.col('X')) + np.square(pl.col('Y')) + np.square(pl.col('Z'))).alias('norm'))
        )
    )

    if only_one_week:
        start = np.ceil(day.min())
        mask = (start <= day.to_numpy()) & (day.to_numpy() <= start + 7*3)
        mask &= ~ actigraphy.get_column('non-wear_flag').cast(bool).to_numpy()
    else:
        mask = np.full(len(day), True)
        
    if small:
        timelines = [
            ('enmo', 'forestgreen'),
            ('light', 'orange'),
        ]
    else:
        timelines = [
            ('X', 'm'),
            ('Y', 'm'),
            ('Z', 'm'),
#             ('norm', 'c'),
            ('enmo', 'forestgreen'),
            ('anglez', 'lightblue'),
            ('light', 'orange'),
            ('non-wear_flag', 'chocolate')
    #         ('diff_seconds', 'k'),
        ]
        
    _, axs = plt.subplots(len(timelines), 1, sharex=True, figsize=(12, len(timelines) * 1.1 + 0.5))
    for ax, (feature, color) in zip(axs, timelines):
        ax.set_facecolor('#eeeeee')
        ax.scatter(day.to_numpy()[mask],
                   actigraphy.get_column(feature).to_numpy()[mask],
                   color=color, label=feature, s=1)
        ax.legend(loc='upper left', facecolor='#eeeeee')
        if feature == 'diff_seconds':
            ax.set_ylim(-0.5, 20.5)
    axs[-1].set_xlabel('day')
    axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    axs[0].set_title(f'id={id}, {sex}, age={age}')
    plt.show()

analyze_actigraphy('0417c91e', only_one_week=False)


def process_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    
    # 提取所需的統計特徵
    mean_values = df.mean().values
    std_values = df.std().values
    min_values = df.min().values
    max_values = df.max().values
    
    # 計算活動比例
    active_ratio = (df[['X', 'Y', 'Z']] != 0).mean().values
    still_ratio = (df['enmo'] < 0.01).mean()
    
    # 合併特徵
    features = np.concatenate([mean_values, std_values, min_values, max_values, active_ratio, [still_ratio]])
    
    # 創建特徵名稱
    columns = list(df.columns)
    feature_names = (
        [f"{col}_mean" for col in columns] +
        [f"{col}_std" for col in columns] +
        [f"{col}_min" for col in columns] +
        [f"{col}_max" for col in columns] +
        [f"{col}_active_ratio" for col in ['X', 'Y', 'Z']] +
        ['enmo_still_ratio']
    )
    
    return features, feature_names, filename.split('=')[1]



def load_time_series(dirname) -> pd.DataFrame:
    ids = os.listdir(dirname)
    
    with ThreadPoolExecutor() as executor:
        # 修改：接收返回的特徵值、特徵名稱和 ID
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
    
    # 解壓結果
    stats_list, feature_names_list, indexes = zip(*results)
    
    # 獲取統一的特徵名稱（假設所有檔案的特徵名稱相同）
    feature_names = feature_names_list[0]
    
    # 構建 DataFrame
    df = pd.DataFrame(stats_list, columns=feature_names)
    df['id'] = indexes
    
    return df



train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
sample = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')

train_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")

df_train = train_ts.drop('id', axis=1)
df_test = test_ts.drop('id', axis=1)


train['sii'].value_counts()


# 定義季節類型的映射字典
season_mapping = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}

# 找到所有以 Season 結尾的欄位
season_columns = [col for col in train.columns if col.endswith('Season')]

# 將這些列的值轉換為數值
for col in season_columns:
    train[col] = train[col].map(season_mapping)

train


# 找到 test 中的列名
test_columns = set(test.columns)

# 找到 train 中需要保留的列（目標列 'sii' 和 test 中存在的列）
columns_to_keep = [col for col in train.columns if col in test_columns or col == 'sii']

# 篩選 train 中需要保留的列
filtered_train = train[columns_to_keep]
filtered_train = filtered_train[filtered_train['sii'].notnull()]
# 打印結果
print(f"原始 train 大小: {train.shape}")
print(f"篩選後 train 大小: {filtered_train.shape}")
print(f"保留的列: {columns_to_keep}")


# 獲取每列的非空值數量
non_null_counts = filtered_train.notnull().sum()

# 計算總行數
total_rows = len(filtered_train)

# 計算缺失比率
train_null_ratio = 1 - (non_null_counts / total_rows)

# 設定閾值
threshold = 0.3

# 篩選出 null_ratio 小於等於閾值的列
columns_to_keep_train = train_null_ratio[train_null_ratio <= threshold].index.tolist()

# 保留篩選出的列
filtered_train = filtered_train[columns_to_keep_train]

# 打印結果
print(f"保留的列: {columns_to_keep_train}")
print(f"篩選後的 train 大小: {filtered_train.shape}")



# 定義季節類型的映射字典
season_mapping = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}

# 找到所有以 Season 結尾的欄位
season_columns = [col for col in test.columns if col.endswith('Season')]

# 將這些列的值轉換為數值
for col in season_columns:
    test[col] = test[col].map(season_mapping)


# 合併兩個 DataFrame，根據 id
# merged_train = pd.merge(filtered_train, train_ts, on='id', how='inner')
merged_train = pd.merge(filtered_train, train_ts, on='id', how='left')


# 將 id 設置為索引
merged_train.set_index('id', inplace=True)

# 打印結果
merged_train


import matplotlib.pyplot as plt
# 篩選數值型列
numeric_columns = merged_train.select_dtypes(include=['number']).columns

# 計算相關係數
correlation_with_sii = merged_train[numeric_columns].corr()['sii'].drop('sii')

# 設置圖形大小
plt.figure(figsize=(20, 20))  # 調整寬和高

# 繪製條形圖
correlation_with_sii.sort_values().plot(kind='barh', color='skyblue')

# 添加標題和軸標籤
plt.title("Correlation with 'sii'", fontsize=16)  # 設置標題字體大小
plt.xlabel("Correlation coefficient", fontsize=14)  # 設置 X 軸標籤字體大小
plt.ylabel("Features", fontsize=14)  # 設置 Y 軸標籤字體大小

# 調整刻度字體大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 顯示圖形
plt.tight_layout()  # 自動調整子圖參數以防止文字重疊
plt.show()

# 打印相關係數
print("其他欄位與 'sii' 的相關係數：")
print(correlation_with_sii)



# 計算相關係數
correlation_with_sii = merged_train.corr()['sii']

# 篩選相關性大於 0.1 的列
columns_to_keep = correlation_with_sii[correlation_with_sii.abs() > 0.1].index.tolist()

# 保留相關列
filtered_train = merged_train[columns_to_keep]

# 打印結果
print(f"保留的列: {columns_to_keep}")
print(f"篩選後的 train 大小: {filtered_train.shape}")



# 將 'sii' 欄位移到最後
columns = [col for col in filtered_train.columns if col != 'sii']  # 除了 'sii' 的其他列
columns.append('sii')  # 將 'sii' 加到最後
filtered_train = filtered_train[columns]  # 重新排列列順序


filtered_train


merged_test = pd.merge(test, test_ts, on='id', how='left')
# 將 id 設置為索引
merged_test.set_index('id', inplace=True)
merged_test


merged_test = merged_test[['Basic_Demos-Age', 'Basic_Demos-Sex', 'Physical-BMI', 'Physical-Height', 'Physical-Weight', 'Physical-Systolic_BP', 'FGC-FGC_CU', 'FGC-FGC_TL', 'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 'PreInt_EduHx-computerinternet_hoursday', 'enmo_mean', 'non-wear_flag_mean', 'X_std', 'enmo_std', 'light_std', 'X_min', 'Y_min', 'Y_max', 'enmo_max', 'enmo_still_ratio']]
merged_test


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# --- 0. 读取数据 -------------------------------------------------------------
# filtered_train = pd.read_csv("your_data.csv")
X = filtered_train.drop('sii', axis=1)
y = filtered_train['sii'].astype(int)

# --- 1. 划分训练/测试 (80/20) --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 2. 前处理 Pipeline ---------------------------------------------------
pre = Pipeline([
    ('imp', SimpleImputer(strategy='mean')),
    ('sc', StandardScaler())
])

# --- 3. 定义搜索空间 & 模型 ----------------------------------------------
# 每个模型只挑最重要的两项超参数来调，其他保持默认／固定

XGB_params = {
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__n_estimators':  [200, 300, 400]
}

CAT_params = {
    'clf__depth':         [6, 8, 10]
}

ADA_params = {
    'clf__n_estimators':  [50, 100, 200],
    'clf__learning_rate': [0.01, 0.1, 1.0]
}


search_space = {
    'XGB': (
        XGBClassifier(
            random_state=42,
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=-1
        ), XGB_params
    ),
    'CAT': (
        CatBoostClassifier(
            random_state=42,
            verbose=0,
            loss_function='MultiClass'
        ), CAT_params
    ),
    'ADA': (
        AdaBoostClassifier(random_state=42), ADA_params
    )
}

best_estimators = {}

# --- 4. GridSearchCV 全面搜索 & 训练每个基础模型 -----------------------------
for name, (model, params) in search_space.items():
    pipe = Pipeline([('pre', pre), ('clf', model)])
    grid = GridSearchCV(
        pipe,
        param_grid=params,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    best_estimators[name] = best

    # 测试集预测 & 报告
    y_pred = best.predict(X_test)
    print(f"\n{name} best parameter：{grid.best_params_}")
    print(f"=== {name} test report ===")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print('-'*50)

# --- 5. VotingClassifier 集成 -----------------------------------------------
voting = VotingClassifier(
    estimators=[
        ('xgb', best_estimators['XGB']),
        ('cat', best_estimators['CAT']),
        ('ada', best_estimators['ADA'])
    ],
    voting='soft',
    weights=[4, 4, 4],
    n_jobs=-1
)
voting.fit(X_train, y_train)

y_vote = voting.predict(X_test)
print("\n=== Voting Ensemble test report ===")
print(classification_report(y_test, y_vote, digits=4))
print(f"Ensemble Accuracy: {accuracy_score(y_test, y_vote):.4f}")

cm = confusion_matrix(y_test, y_vote, labels=np.unique(y))
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Voting Ensemble confusion matrix')
plt.xlabel('Pred')
plt.ylabel('True')
plt.tight_layout()
plt.show()

def per_class_acc(y_true, y_pred):
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {lbl: cm[i,i]/cm[i,:].sum() for i,lbl in enumerate(labels)}

print("Ensemble per class Accuracy:", per_class_acc(y_test, y_vote))

# --- 6. 序列化所有模型 & 预处理器 ----------------------------------------------
joblib.dump(best_estimators, 'best_gradient_models.pkl')
joblib.dump(voting, 'voting_ensemble.pkl')
joblib.dump(pre, 'preprocessor.pkl')



#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

warnings.filterwarnings("ignore")

# --- 类别权重映射 ------------------------------------------------------------
weight_map = {0: 0.417, 1: 0.73, 2: 0.86, 3: 1.0}

# --- 0. 读取数据 -------------------------------------------------------------
# filtered_train = pd.read_csv("your_data.csv")
X = filtered_train.drop('sii', axis=1).values
y = filtered_train['sii'].astype(int).values

# 计算每个样本的权重
sample_weight = np.vectorize(weight_map.get)(y)

# --- 1. 划分训练/测试 (80/20) --------------------------------------------
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weight, test_size=0.2, random_state=42, stratify=y
)

# --- 2. 前处理 Pipeline ---------------------------------------------------
pre = Pipeline([
    ('imp', SimpleImputer(strategy='mean')),
    ('sc',  StandardScaler())
])

# --- 3. 定义搜索空间 & 模型 ----------------------------------------------
XGB_params = {
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__n_estimators':  [200, 300, 400]
}
CAT_params = {
    'clf__depth': [6, 8, 10]
}
ADA_params = {
    'clf__n_estimators':  [50, 100, 200],
    'clf__learning_rate': [0.01, 0.1, 1.0]
}

search_space = {
    'XGB': (
        XGBClassifier(
            random_state=42,
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_jobs=-1
        ), XGB_params
    ),
    'CAT': (
        CatBoostClassifier(
            random_state=42,
            verbose=0,
            loss_function='MultiClass'
        ), CAT_params
    ),
    'ADA': (
        AdaBoostClassifier(random_state=42), ADA_params
    )
}

best_estimators = {}

# --- 4. GridSearchCV 全面搜索 & 训练基础模型 -----------------------------
for name, (model, params) in search_space.items():
    pipe = Pipeline([('pre', pre), ('clf', model)])
    grid = GridSearchCV(
        pipe,
        param_grid=params,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    # 传入 sample_weight 给 clf
    grid.fit(X_train, y_train, **{'clf__sample_weight': sw_train})
    best_estimators[name] = grid.best_estimator_

    # 测试集预测 & 报告
    y_pred = grid.predict(X_test)
    print(f"\n=== {name} test report ===")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print('-' * 50)

# --- 5. Soft Voting 手动集成 -----------------------------------------------
# 收集各模型的 predict_proba 结果
probas = []
for name, clf in best_estimators.items():
    probas.append(clf.predict_proba(X_test))
# 平均概率
avg_proba = np.mean(probas, axis=0)
# 取最大概率对应的类别
y_vote = np.argmax(avg_proba, axis=1)

print("\n=== Voting Ensemble test report ===")
print(classification_report(y_test, y_vote, digits=4))
print(f"Ensemble Accuracy: {accuracy_score(y_test, y_vote):.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_vote, labels=[0,1,2,3])
plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=[0,1,2,3], yticklabels=[0,1,2,3]
)
plt.title('Voting Ensemble confusion matrix')
plt.xlabel('Pred')
plt.ylabel('True')
plt.tight_layout()
plt.show()

def per_class_acc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    return {lbl: cm[i,i]/cm[i,:].sum() for i,lbl in enumerate([0,1,2,3])}

print("Ensemble 每类 Accuracy:", per_class_acc(y_test, y_vote))

# --- 6. 序列化所有模型 & 预处理器 ----------------------------------------------
joblib.dump(best_estimators, 'best_classifiers.pkl')
joblib.dump(pre,             'preprocessor.pkl')



#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.ensemble import VotingRegressor, AdaBoostRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# --- 0. 读取数据 -------------------------------------------------------------
# filtered_train = pd.read_csv("your_data.csv")
X = filtered_train.drop('sii', axis=1)
y = filtered_train['sii'].astype(float)  # 回归取 float

# 确定分类范围
min_label, max_label = int(y.min()), int(y.max())
labels = list(range(min_label, max_label + 1))

# --- 1. 划分训练/测试 (80/20) --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 2. 前处理 Pipeline ---------------------------------------------------
pre = Pipeline([
    ('imp', SimpleImputer(strategy='mean')),
    ('sc', StandardScaler())
])

# --- 3. 定义网格搜索空间 & 模型 ------------------------------------------
# 每个模型只挑最重要的两项超参数来调，其他保持默认／固定

XGB_params = {
    'reg__learning_rate': [0.01, 0.05, 0.1],
    'reg__n_estimators':  [200, 300, 400]
}

CAT_params = {
    'reg__depth': [6, 8, 10]
}

ADA_params = {
    'reg__n_estimators':  [50, 100, 200],
    'reg__learning_rate': [0.01, 0.1, 1.0]
}

search_space = {
    'XGB': (
        XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            n_jobs=-1
        ),
        XGB_params
    ),
    'CAT': (
        CatBoostRegressor(
            random_state=42,
            verbose=0,
            loss_function='RMSE'
        ),
        CAT_params
    ),
    'ADA': (
        AdaBoostRegressor(random_state=42),
        ADA_params
    )
}

best_estimators = {}
for name, (model, params) in search_space.items():
    pipe = Pipeline([('pre', pre), ('reg', model)])
    grid = GridSearchCV(
        pipe,
        param_grid=params,
        cv=5,
        scoring='r2',      # 以 R² 作为搜索指标
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    best_estimators[name] = grid.best_estimator_

    # 回归评估
    y_pred = grid.predict(X_test)
    print(f"\n{name} best parameter：", grid.best_params_)
    print(f"=== {name} regression evaluation ===")
    print(f"MSE:  {mean_squared_error(y_test, y_pred):.4f}")
    print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R²:   {r2_score(y_test, y_pred):.4f}")
    print('-'*30)

    # 回归→分类评估
    y_pred_cls = np.rint(y_pred).astype(int)
    y_pred_cls = np.clip(y_pred_cls, min_label, max_label)

    print(f"\n=== {name} regression classification report ===")
    print(classification_report(
        y_test.astype(int),
        y_pred_cls,
        labels=labels,
        digits=4
    ))
    print(f"{name} Accuracy: {accuracy_score(y_test.astype(int), y_pred_cls):.4f}")

    # 混淆矩阵
    cm = confusion_matrix(
        y_test.astype(int),
        y_pred_cls,
        labels=labels
    )
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'{name} confusion matrix')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    # 每类 Accuracy
    per_acc = {lbl: cm[i,i] / cm[i].sum() for i, lbl in enumerate(labels)}
    print(f"{name} per class Accuracy:", per_acc)
    print('='*50)

# --- 4. 构建 VotingRegressor ------------------------------------------------
voting = VotingRegressor(
    estimators=[
        ('xgb', best_estimators['XGB']),
        ('cat', best_estimators['CAT']),
        ('ada', best_estimators['ADA'])
    ]
)
voting.fit(X_train, y_train)

y_vote = voting.predict(X_test)
print("\n=== VotingRegressor regression evaluation ===")
print(f"MSE:  {mean_squared_error(y_test, y_vote):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_vote):.4f}")
print(f"R²:   {r2_score(y_test, y_vote):.4f}")

# VotingRegressor → 分类评估
y_vote_cls = np.rint(y_vote).astype(int)
y_vote_cls = np.clip(y_vote_cls, min_label, max_label)

print("\n=== VotingRegressor regression classification report ===")
print(classification_report(
    y_test.astype(int),
    y_vote_cls,
    labels=labels,
    digits=4
))
print(f"Voting Accuracy: {accuracy_score(y_test.astype(int), y_vote_cls):.4f}")

cm = confusion_matrix(y_test.astype(int), y_vote_cls, labels=labels)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Voting confusion natrix')
plt.xlabel('Pred')
plt.ylabel('True')
plt.tight_layout()
plt.show()

per_acc = {lbl: cm[i,i] / cm[i].sum() for i, lbl in enumerate(labels)}
print("Voting per class Accuracy:", per_acc)

# --- 5. 序列化所有模型 & 预处理器 ----------------------------------------------
joblib.dump(best_estimators, 'best_regressors.pkl')
joblib.dump(voting, 'voting_regressor.pkl')
joblib.dump(pre, 'preprocessor.pkl')



#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
# 全局屏蔽 “invalid value encountered in scalar divide” RuntimeWarning
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning
)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    make_scorer
)
from sklearn.ensemble import VotingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

# filtered_train = pd.read_csv("your_data.csv")
X = filtered_train.drop('sii', axis=1).values
y_orig = filtered_train['sii'].astype(int).values  # 原始 0–3

# 回归连续映射值直接就是 0,1,2,3
mapped_values = np.array([0, 1, 2, 3])

# --- 1. 划分训练/测试 --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_orig.astype(float), test_size=0.2, random_state=42
)

# --- 2. 手动过采样少数类（原始类别3） ----------------------------------------
mask_min = (y_train == 3)
X_min, y_min = X_train[mask_min], y_train[mask_min]
X_maj, y_maj = X_train[~mask_min], y_train[~mask_min]

X_min_ups, y_min_ups = resample(
    X_min, y_min,
    replace=True,
    n_samples=len(y_maj),
    random_state=42
)

X_train_bal = np.vstack([X_maj, X_min_ups])
y_train_bal = np.concatenate([y_maj, y_min_ups])

# --- 3. 前处理 Pipeline -----------------------------------------------------
pre = Pipeline([
    ('imp', SimpleImputer(strategy='mean')),
    ('sc',  StandardScaler())
])

# --- 4. 映射连续预测到类别索引 ----------------------------------------------
def map_cont_to_cls(arr_cont: np.ndarray) -> np.ndarray:
    diffs = np.abs(arr_cont.reshape(-1, 1) - mapped_values.reshape(1, -1))
    return diffs.argmin(axis=1)

# --- 5. 自定义 QWK scorer（含 NaN 保护） -----------------------------------
def qwk_score(y_true_cont, y_pred_cont):
    true_cls = map_cont_to_cls(np.array(y_true_cont))
    pred_cls = map_cont_to_cls(np.array(y_pred_cont))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        kappa = cohen_kappa_score(true_cls, pred_cls, weights='quadratic')
    return 0.0 if np.isnan(kappa) else kappa

qwk_scorer = make_scorer(qwk_score, greater_is_better=True)

# --- 6. 定义搜索空间与模型 --------------------------------------------------
XGB_params = {
    'reg__learning_rate': [0.01, 0.05, 0.1],
    'reg__n_estimators':  [200, 300, 400]
}

CAT_params = {
    'reg__depth': [6, 8, 10]
}

ADA_params = {
    'reg__n_estimators':  [50, 100, 200],
    'reg__learning_rate': [0.01, 0.1, 1.0]
}

search_space = {
    'XGB': (
        XGBRegressor(
            random_state=42,
            objective='reg:squarederror',
            n_jobs=-1
        ),
        XGB_params
    ),
    'CAT': (
        CatBoostRegressor(
            random_state=42,
            verbose=0,
            loss_function='RMSE'
        ),
        CAT_params
    ),
    'ADA': (
        AdaBoostRegressor(random_state=42),
        ADA_params
    )
}

best_estimators = {}

# --- 7. GridSearchCV 全面搜索 & 训练基础模型 -----------------------------
for name, (model, params) in search_space.items():
    pipe = Pipeline([('pre', pre), ('reg', model)])
    grid = GridSearchCV(
        pipe,
        param_grid=params,
        cv=5,
        scoring=qwk_scorer,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train_bal, y_train_bal)
    best_estimators[name] = grid.best_estimator_

    # --- QWK & 分类报告 ---
    y_pred_cont = grid.predict(X_test)
    y_pred_cls  = map_cont_to_cls(y_pred_cont)

    qwk_val = qwk_score(y_test, y_pred_cont)
    print(f"\n{name} best QWK = {qwk_val:.4f}")
    print(f"{name} best parameter：{grid.best_params_}")
    print(classification_report(y_test.astype(int), y_pred_cls, labels=mapped_values, digits=4))
    print(f"{name} Accuracy: {accuracy_score(y_test.astype(int), y_pred_cls):.4f}")

    cm = confusion_matrix(y_test.astype(int), y_pred_cls, labels=mapped_values)
    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=mapped_values, yticklabels=mapped_values
    )
    plt.title(f'{name} confusion matrix')
    plt.xlabel('Pred'); plt.ylabel('True')
    plt.tight_layout()
    plt.show()

# --- 8. VotingRegressor 集成 -----------------------------------------------
voting = VotingRegressor([
    ('xgb', best_estimators['XGB']),
    ('cat', best_estimators['CAT']),
    ('ada', best_estimators['ADA'])
])
voting.fit(X_train_bal, y_train_bal)

y_vote_cont = voting.predict(X_test)
y_vote_cls  = map_cont_to_cls(y_vote_cont)

qwk_vote = qwk_score(y_test, y_vote_cont)
print("\n=== VotingRegressor overall evaluation (QWK) ===")
print(f"QWK: {qwk_vote:.4f}")
print(classification_report(y_test.astype(int), y_vote_cls, labels=mapped_values, digits=4))
print(f"Voting Accuracy: {accuracy_score(y_test.astype(int), y_vote_cls):.4f}")

cm = confusion_matrix(y_test.astype(int), y_vote_cls, labels=mapped_values)
plt.figure(figsize=(5,4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=mapped_values, yticklabels=mapped_values
)
plt.title('Voting confusion matrix')
plt.xlabel('Pred'); plt.ylabel('True')
plt.tight_layout()
plt.show()

# --- 9. 序列化 --------------------------------------------------------------
joblib.dump(best_estimators, 'best_regressors.pkl')
joblib.dump(voting,          'voting_regressor.pkl')
joblib.dump(pre,             'preprocessor.pkl')



# %% Cell 0 — 安裝所需套件
!pip install polars tqdm keras torch colorama lightgbm xgboost catboost
!pip install mord


import os, warnings, gc
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# 读取 train.csv 并映射 Season 列
DATA_DIR = '/kaggle/input/child-mind-institute-problematic-internet-use'
train = pd.read_csv(f'{DATA_DIR}/train.csv')

season_map = {'Spring':0,'Summer':1,'Fall':2,'Winter':3}
for c in train.columns:
    if c.endswith('Season'):
        train[c] = train[c].map(season_map)

# 丢掉 sii 缺失
train = train.dropna(subset=['sii']).reset_index(drop=True)
print("原始 train 大小：", train.shape)



from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_file(folder, base):
    df = pd.read_parquet(f'{base}/{folder}/part-0.parquet')
    df = df.drop(columns='step', errors='ignore')
    stats = np.hstack([
        df.mean().values,
        df.std().values,
        df.min().values,
        df.max().values,
        (df[['X','Y','Z']]!=0).mean().values,
        [(df['enmo']<0.01).mean()]
    ])
    cols = df.columns.tolist()
    names = (
        [f'{c}_mean' for c in cols] +
        [f'{c}_std'  for c in cols] +
        [f'{c}_min'  for c in cols] +
        [f'{c}_max'  for c in cols] +
        [f'{c}_active_ratio' for c in ['X','Y','Z']] +
        ['enmo_still_ratio']
    )
    return stats, names, folder.split('=')[1]

def load_series(base):
    folders = os.listdir(base)
    with ThreadPoolExecutor() as ex:
        results = list(tqdm(
            ex.map(process_file, folders, [base]*len(folders)),
            total=len(folders), desc="萃取时序特征"
        ))
    stats, names, ids = zip(*results)
    df = pd.DataFrame(stats, columns=names[0])
    df['id'] = ids
    return df

train_ts = load_series(f'{DATA_DIR}/series_train.parquet')
print("时序特征大小：", train_ts.shape)



from sklearn.model_selection import train_test_split

# 建立两个标签：分类的 y_clf，以及回归的 y_raw
y_clf = df['sii'              ].astype(int)
y_raw = df['PCIAT-PCIAT_Total'].astype(float)
X_all = df.drop(columns=['sii','PCIAT-PCIAT_Total'])

# 用 same split 保持一致性，stratify 用 y_clf
X_train, X_test, y_train_clf, y_test_clf, y_train_raw, y_test_raw = train_test_split(
    X_all, y_clf, y_raw,
    test_size=0.20, random_state=42, stratify=y_clf
)

print("训练集样本数：", X_train.shape[0], "测试集样本数：", X_test.shape[0])
print("训练集 SII 分布：\n", y_train_clf.value_counts(normalize=True))



from sklearn.model_selection import train_test_split

# 建立两个标签：分类的 y_clf，以及回归的 y_raw
y_clf = df['sii'              ].astype(int)
y_raw = df['PCIAT-PCIAT_Total'].astype(float)
X_all = df.drop(columns=['sii','PCIAT-PCIAT_Total'])

# 用 same split 保持一致性，stratify 用 y_clf
X_train, X_test, y_train_clf, y_test_clf, y_train_raw, y_test_raw = train_test_split(
    X_all, y_clf, y_raw,
    test_size=0.20, random_state=42, stratify=y_clf
)

print("训练集样本数：", X_train.shape[0], "测试集样本数：", X_test.shape[0])
print("训练集 SII 分布：\n", y_train_clf.value_counts(normalize=True))



from sklearn.pipeline          import Pipeline
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing     import StandardScaler
from sklearn.feature_selection import SelectFpr, f_regression
from sklearn.base              import clone

# 要剔除的泄漏特征 PCIAT-1~20
leak_cols = [f"PCIAT-PCIAT_{i:02d}" for i in range(1,21)]

# —— 1. 回归器用的预处理 Pipeline（保留所有特征） ——
alpha = 0.05
print(f"[Regression] SelectFpr(p≤{alpha}) 筛特征（含 PCIAT）")
pre_reg = Pipeline([
    ('imp', SimpleImputer(strategy='mean')),
    ('sc' , StandardScaler()),
    ('sel', SelectFpr(score_func=f_regression, alpha=alpha))
])

# 临时 inspect
pre_reg_inspect = clone(pre_reg)
pre_reg_inspect.fit(X_train, y_train_raw)
mask_reg = pre_reg_inspect.named_steps['sel'].get_support()
print("Regression 保留特征（含 PCIAT）共", mask_reg.sum())

# —— 2. 分类器用的预处理 Pipeline（剔除 PCIAT-1~20） ——
print(f"\n[Classification] 先剔除 PCIAT-1~20，再 SelectFpr(p≤{alpha})")
X_train_clf = X_train.drop(columns=leak_cols, errors='ignore')
X_test_clf  = X_test .drop(columns=leak_cols, errors='ignore')

pre_clf = Pipeline([
    ('imp', SimpleImputer(strategy='mean')),
    ('sc' , StandardScaler()),
    ('sel', SelectFpr(score_func=f_regression, alpha=alpha))
])

# inspect 分类保留特征
pre_clf_inspect = clone(pre_clf)
pre_clf_inspect.fit(X_train_clf, y_train_clf)
mask_clf = pre_clf_inspect.named_steps['sel'].get_support()
selected_clf = X_train_clf.columns[mask_clf]
print("Classification 保留特征（剔除 PCIAT-1~20）共", len(selected_clf))
for feat in selected_clf:
    print(" -", feat)

# 回归器和分类器的超参配置
from xgboost          import XGBRegressor, XGBClassifier
from catboost         import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier

param_reg = {
    'xgb': {'model': XGBRegressor(objective='reg:squarederror',
                                  random_state=42, n_jobs=-1),
            'grid' : {'model__learning_rate':[0.01,0.05,0.1],
                      'model__n_estimators': [200,300,400]}},
    'cat': {'model': CatBoostRegressor(verbose=0, loss_function='RMSE',
                                       random_state=42),
            'grid' : {'model__depth':[6,8,10]}},
    'ada': {'model': AdaBoostRegressor(random_state=42),
            'grid' : {'model__n_estimators':[50,100,200],
                      'model__learning_rate':[0.01,0.1,1.0]}}
}

param_clf = {
    'xgb': {'model': XGBClassifier(use_label_encoder=False,
                                    eval_metric='mlogloss', random_state=42),
            'grid' : {'model__learning_rate':[0.01,0.05,0.1],
                      'model__n_estimators':[200,300,400]}},
    'cat': {'model': CatBoostClassifier(verbose=0, random_state=42),
            'grid' : {'model__depth':[6,8,10]}},
    'ada': {'model': AdaBoostClassifier(random_state=42),
            'grid' : {'model__n_estimators':[50,100,200],
                      'model__learning_rate':[0.01,0.1,1.0]}}
}



from sklearn.model_selection import GridSearchCV

best_regressors = {}
for name,spec in param_reg.items():
    pipe = Pipeline([('pre', pre), ('model', spec['model'])])
    gs   = GridSearchCV(pipe, spec['grid'], cv=5, scoring='r2',
                        n_jobs=-1, verbose=1, error_score='raise')
    gs.fit(X_train, y_train_raw)
    best_regressors[name] = gs.best_estimator_
    print(f">>> 回归器 {name} 最佳参数：", gs.best_params_)

best_classifiers = {}
for name,spec in param_clf.items():
    pipe = Pipeline([('pre', pre), ('model', spec['model'])])
    gs   = GridSearchCV(pipe, spec['grid'], cv=5, scoring='accuracy',
                        n_jobs=-1, verbose=1, error_score='raise')
    gs.fit(X_train, y_train_clf)
    best_classifiers[name] = gs.best_estimator_
    print(f">>> 分类器 {name} 最佳参数：", gs.best_params_)



from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, accuracy_score
)

def raw_to_sii(raw):
    if raw < 30: return 0
    if raw < 50: return 1
    if raw < 80: return 2
    return 3

labels = [0,1,2,3]

print("=== 回归器(预测 PCIAT_Total) & 映射回 SII 的评估 ===")
for name, model in best_regressors.items():
    y_pred_raw = model.predict(X_test)
    # 回归指标
    print(f"\n[{name.upper()} 回归指标]")
    print(f" MSE={mean_squared_error(y_test_raw, y_pred_raw):.4f}")
    print(f" MAE={mean_absolute_error(y_test_raw, y_pred_raw):.4f}")
    print(f" R² ={r2_score            (y_test_raw, y_pred_raw):.4f}")
    # 映射 & 分类报告
    y_pred_sii = np.array([raw_to_sii(v) for v in y_pred_raw])
    print(f"\n[{name.upper()} 映射后分类报告]")
    print(classification_report(y_test_clf, y_pred_sii, labels=labels, digits=4))
    print(f"Accuracy: {accuracy_score(y_test_clf, y_pred_sii):.4f}")

print("\n=== 直接分类器 的评估 ===")
for name, model in best_classifiers.items():
    y_pred = model.predict(X_test)
    print(f"\n[{name.upper()}]")
    print(classification_report(y_test_clf, y_pred, labels=labels, digits=4))
    print(f"Accuracy: {accuracy_score(y_test_clf, y_pred):.4f}")


