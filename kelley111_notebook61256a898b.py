# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 数据加载
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
train_solution = train[['ID', 'efs', 'efs_time', 'race_group']].copy()
train.tail()


# 数值特征填充
num_features = train.select_dtypes(include=['float64', 'int64']).columns
print(num_features)
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 提取需要保留的列
efs_columns = ['efs', 'efs_time']

# 提取数值特征并进行缺失值填充、标准化、PCA降维
num_features = train.select_dtypes(include=['float64', 'int64']).columns
num_features = [f for f in num_features if f not in efs_columns]  # 确保不包含 'efs' 和 'efs_time'

# KNN填充
imputer = KNNImputer(n_neighbors=4)
train[num_features] = imputer.fit_transform(train[num_features])
test[num_features] = imputer.transform(test[num_features])  # 对测试集进行相同的填充

# 标准化
scaler = StandardScaler()
train[num_features] = scaler.fit_transform(train[num_features])
test[num_features] = scaler.transform(test[num_features])  # 对测试集进行相同的标准化

# PCA降维
pca = PCA(n_components=0.85)  # 保留95%的方差
train_pca = pca.fit_transform(train[num_features])  # 拟合 PCA 模型并转换训练集
test_pca = pca.transform(test[num_features])  # 使用训练集的 PCA 模型转换测试集

# 将PCA降维后的特征添加到DataFrame中
train_pca_df = pd.DataFrame(train_pca, columns=[f'PC{i+1}' for i in range(train_pca.shape[1])])
test_pca_df = pd.DataFrame(test_pca, columns=[f'PC{i+1}' for i in range(test_pca.shape[1])])

# 删除原始的数值特征并添加PCA特征
train = train.drop(columns=num_features)
test = test.drop(columns=num_features)
train = pd.concat([train, train_pca_df], axis=1)
test = pd.concat([test, test_pca_df], axis=1)

# 将 'efs' 和 'efs_time' 列重新添加到数据中
train[efs_columns] = train[efs_columns]

# 查看数据的新形状
print(train.shape)
print(test.shape)


print(train.columns)


# 特征工程
features = [f for f in test.columns if f != 'ID']

# 确保 features 列表中的列名存在于 train DataFrame 中
missing_features = [f for f in features if f not in train.columns]
if missing_features:
    print(f"以下列名在 train DataFrame 中不存在: {missing_features}")
    features = [f for f in features if f in train.columns]

# 分类特征处理
cat_features = list(train.select_dtypes(object).columns)
train[cat_features] = train[cat_features].astype(str).astype('category')
test[cat_features] = test[cat_features].astype(str).astype('category')  # 对测试集进行相同的处理

# 种族分组
race_groups = np.unique(train.race_group)


print(train.columns)



from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

def evaluate_fold(y_va_pred, fold):
    """Compute and print the metrics (concordance index) per race group for a single fold.

    Global variables:
    - train, X_va, idx_va
    - The metrics are saved in the global list all_scores.
    """
    metric_list = []
    for race in race_groups:
        mask = X_va.race_group.values == race
        c_index_race = concordance_index(
            train.efs_time.iloc[idx_va][mask],
            - y_va_pred[mask],
            train.efs.iloc[idx_va][mask]
        )
        # print(f"# {race:42} {c_index_race:.3f}")
        metric_list.append(c_index_race)
    fold_score = np.mean(metric_list) - np.sqrt(np.var(metric_list))
    print(f"# Total fold {fold}:{' ':29} {fold_score:.3f} mean={np.mean(metric_list):.3f} std={np.std(metric_list):.3f}")
    all_scores.append(metric_list)

def display_overall(label):
    """Compute and print the overall metrics (concordance index)"""
    df = pd.DataFrame(all_scores, columns=race_groups)
    df['mean'] = df[race_groups].mean(axis=1)
    df['std'] = np.std(df[race_groups], axis=1)
    df['score'] = df['mean'] - df['std']
    df = df.T
    df['Overall'] = df.mean(axis=1)
    temp = df.drop(index=['std']).values
    print(f"# Overall:                                   {df.loc['score', 'Overall']:.3f} {label}")
    all_model_scores[label] = df.loc['score', 'Overall']
    display(df
            .iloc[:len(race_groups)]
            .style
            .format(precision=3)
            .background_gradient(axis=None, vmin=temp.min(), vmax=temp.max(), cmap="cool")
            .concat(df.iloc[len(race_groups):].style.format(precision=3))
           )


!pip install lifelines



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import numpy as np
import xgboost
import catboost
import warnings
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy.stats import rankdata

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, quantile_transform, FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

all_model_scores = {}


y = np.where(train.efs == 1, train.efs_time, -train.efs_time)
all_scores = []
for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train.race_group)):
    X_tr = train.iloc[idx_tr][features]
    X_va = train.iloc[idx_va][features]
    y_tr = y[idx_tr]
    
    xgb_cox_params = {'objective': 'survival:cox', 'grow_policy': 'depthwise', 
                      'n_estimators': 700, 'learning_rate': 0.0254, 'max_depth': 8, 
                      'reg_lambda': 0.116, 'reg_alpha': 0.139, 'min_child_weight': 23.8,
                      'colsample_bytree': 0.59, 'subsample': 0.7, 'tree_method': 'hist',
                      'enable_categorical': True}
    model = xgboost.XGBRegressor(**xgb_cox_params)
    model.fit(X_tr, y_tr) # negative values are considered right censored
    y_va_pred = model.predict(X_va) # predicts hazard factor
    evaluate_fold(y_va_pred, fold)
display_overall('Cox Proportional Hazards XGBoost')
# 在测试集上预测
test_predictions = model.predict(test[features])  # 使用测试集进行预测
print("测试集预测结果:", test_predictions)
# Overall:                                   0.670



# # XGBoost Cox regression
# y = np.where(train.efs == 1, train.efs_time, -train.efs_time)
# all_scores = []
# for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train.race_group)):
#     X_tr = train.iloc[idx_tr][features]
#     X_va = train.iloc[idx_va][features]
#     y_tr = y[idx_tr]
    
#     xgb_cox_params = {'objective': 'survival:cox', 'grow_policy': 'depthwise', 
#                       'n_estimators': 700, 'learning_rate': 0.0254, 'max_depth': 8, 
#                       'reg_lambda': 0.116, 'reg_alpha': 0.139, 'min_child_weight': 23.8,
#                       'colsample_bytree': 0.59, 'subsample': 0.7, 'tree_method': 'hist',
#                       'enable_categorical': True}
#     model = xgboost.XGBRegressor(**xgb_cox_params)
#     model.fit(X_tr, y_tr) # negative values are considered right censored
#     y_va_pred = model.predict(X_va) # predicts hazard factor
#     evaluate_fold(y_va_pred, fold)
# display_overall('Cox Proportional Hazards XGBoost')
# # Overall:                                   0.670


# from sklearn.preprocessing import OneHotEncoder

# # One-Hot Encoding 
# encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# encoded_features = encoder.fit_transform(train[cat_features])
# encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cat_features))
# train = pd.concat([train.drop(cat_features, axis=1), encoded_df], axis=1)




# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# train[num_features] = scaler.fit_transform(train[num_features])


# # HLA 匹配总分
# hla_features = [f for f in train.columns if 'hla_match' in f]
# train['hla_total_score'] = train[hla_features].sum(axis=1)


# #对数几率变换 并打印最大小值
# def logit(p):
#     return np.log(p) - np.log(1 - p)

# max_efs_time, min_efs_time = 80, -100
# train['efs_time'] = train['efs_time'] / (max_efs_time - min_efs_time)
# train['efs_time'] = np.log(train['efs_time']) - np.log(1 - train['efs_time'])
# train['efs_time'] += 10
# print(train['efs_time'].max(), train['efs_time'].min())


# #定义目标y值 并进行归一化处理
# train["y"] = train.efs_time.values
# efs_1_data = train.loc[train.efs == 1, "efs_time"]
# efs_0_data = train.loc[train.efs == 0, "efs_time"]
# mx = efs_1_data.max()
# mn = efs_0_data.min()
# train.loc[train.efs == 0, "y"] = efs_0_data + mx - mn
# train.y = train.y.rank()
# train.loc[train.efs == 0, "y"] += len(train) // 2
# train.y = train.y / train.y.max()




# # 生存概率变换
# def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
#     kmf = KaplanMeierFitter()
#     kmf.fit(df[time_col], event_observed=df[event_col])
#     survival_probabilities = kmf.survival_function_at_times(df[time_col]).values.flatten()
#     return survival_probabilities

# race_group = sorted(train['race_group'].unique())
# for race in race_group:
#     train.loc[train['race_group'] == race, "target"] = transform_survival_probability(train[train['race_group'] == race], time_col='efs_time', event_col='efs')
#     gap = 0.7 * (train.loc[(train['race_group'] == race) & (train['efs'] == 0)]['target'].max() - train.loc[(train['race_group'] == race) & (train['efs'] == 1)]['target'].min()) / 2
#     train.loc[(train['race_group'] == race) & (train['efs'] == 0), 'target'] -= gap







