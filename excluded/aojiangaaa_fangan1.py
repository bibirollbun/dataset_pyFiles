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


!pwd


cd ../input/equity-post-HCT-survival-predictions/


!pwd


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# 加载数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
data_dict = pd.read_csv('data_dictionary.csv')

# 查看数据
print(train.head())
print(test.head())


# 分离特征和目标变量
X = train.drop(columns=['ID', 'efs', 'efs_time'])
y = train['efs']


# 定义数值型和类别型特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# 定义预处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 应用预处理
X_processed = preprocessor.fit_transform(X)
test_processed = preprocessor.transform(test.drop(columns=['ID']))


# # 统计分析
# print(train_data.describe())

# # 可视化分析
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 绘制特征分布图
# for col in train_data.columns:
#     if train_data[col].dtype == 'number':
#         sns.histplot(train_data[col], kde=True)
#         plt.title(f'Distribution of {col}')
#         plt.show()

# # 绘制相关性热力图
# plt.figure(figsize=(10, 8))
# sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()


# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 验证集评估
y_pred = rf.predict_proba(X_val)[:, 1]
print(f'Validation ROC AUC: {roc_auc_score(y_val, y_pred)}')


# # XGBoost模型
# xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
# xgb_model.fit(X_train, y_train)
# y_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
# print(f'XGBoost Validation ROC AUC: {roc_auc_score(y_val, y_pred_xgb)}')

# # LightGBM模型
# lgb_model = lgb.LGBMClassifier(objective='binary', n_estimators=100, random_state=42)
# lgb_model.fit(X_train, y_train)
# y_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
# print(f'LightGBM Validation ROC AUC: {roc_auc_score(y_val, y_pred_lgb)}')




# # 指定一个已存在的目录作为 train_dir
# train_dir = '/kaggle/working/catboost_info'

# # 确保目录存在
# import os
# if not os.path.exists(train_dir):
#     os.makedirs(train_dir)

# # 使用指定的 train_dir
# cb_model = cb.CatBoostClassifier(random_state=42, verbose=0, train_dir=train_dir)
# cb_model.fit(X_train, y_train)
# y_pred_cb = cb_model.predict_proba(X_val)[:, 1]
# print(f'CatBoost Validation ROC AUC: {roc_auc_score(y_val, y_pred_cb)}')


# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import roc_auc_score
# import xgboost as xgb
# import lightgbm as lgb
# import catboost as cb

# # 加载数据
# train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
# test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

# # 分离特征和目标变量
# X = train.drop(columns=['ID', 'efs', 'efs_time'])
# y = train['efs']

# # 定义数值型和类别型特征
# numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
# categorical_features = X.select_dtypes(include=['object', 'category']).columns

# # 定义预处理管道
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # 应用预处理
# X_processed = preprocessor.fit_transform(X)
# test_processed = preprocessor.transform(test.drop(columns=['ID']))

# # 划分训练集和验证集
# X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 指定一个已存在的目录作为 train_dir
train_dir = '/kaggle/working/catboost_info'

# 确保目录存在
import os
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

# XGBoost模型
xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
print(f'XGBoost Validation ROC AUC: {roc_auc_score(y_val, y_pred_xgb)}')

# LightGBM模型
lgb_model = lgb.LGBMClassifier(objective='binary', n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict_proba(X_val)[:, 1]
print(f'LightGBM Validation ROC AUC: {roc_auc_score(y_val, y_pred_lgb)}')

# CatBoost模型
cb_model = cb.CatBoostClassifier(random_state=42, verbose=0, train_dir=train_dir)
cb_model.fit(X_train, y_train)
y_pred_cb = cb_model.predict_proba(X_val)[:, 1]
print(f'CatBoost Validation ROC AUC: {roc_auc_score(y_val, y_pred_cb)}')


param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}

grid_search = GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic', random_state=42),
                           param_grid=param_grid,
                           scoring='roc_auc',
                           cv=5,
                           verbose=1)

grid_search.fit(X_train, y_train)
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best ROC AUC: {grid_search.best_score_}')

# 使用最佳参数重新训练
best_xgb_model = grid_search.best_estimator_
y_pred_best_xgb = best_xgb_model.predict_proba(X_val)[:, 1]
print(f'Best XGBoost Validation ROC AUC: {roc_auc_score(y_val, y_pred_best_xgb)}')


# import xgboost as xgb
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import roc_auc_score

# # 定义参数网格
# param_grid = {
#     'max_depth': [3, 5, 7, 9],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [100, 200, 300, 400],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'reg_alpha': [0.01, 0.1, 0.5],
#     'reg_lambda': [0.1, 0.5, 1.0]
# }

# # 初始化XGBoost分类器
# xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss')

# # 使用GridSearchCV进行超参数搜索
# grid_search = GridSearchCV(estimator=xgb_clf,
#                            param_grid=param_grid,
#                            scoring='roc_auc',
#                            cv=5,
#                            verbose=1,
#                            n_jobs=-1)  # 使用所有可用的CPU核心加速搜索

# # 拟合训练数据
# grid_search.fit(X_train, y_train)

# # 输出最佳参数和最佳ROC AUC分数
# print(f'Best parameters: {grid_search.best_params_}')
# print(f'Best ROC AUC: {grid_search.best_score_}')

# # 使用最佳参数重新训练模型
# best_xgb_model = grid_search.best_estimator_

# # 在验证集上进行预测
# y_pred_best_xgb = best_xgb_model.predict_proba(X_val)[:, 1]

# # 计算验证集上的ROC AUC分数
# print(f'Best XGBoost Validation ROC AUC: {roc_auc_score(y_val, y_pred_best_xgb)}')


from tqdm import tqdm  # 导入tqdm库



# import xgboost as xgb
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import roc_auc_score

# # 定义参数网格
# param_grid = {
#     'max_depth': [3, 5, 7, 9],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [100, 200, 300, 400],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'reg_alpha': [0.01, 0.1, 0.5],
#     'reg_lambda': [0.1, 0.5, 1.0]
# }

# # 初始化XGBoost分类器，使用GPU
# xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42, tree_method='gpu_hist', gpu_id=0, eval_metric='logloss')

# # 使用GridSearchCV进行超参数搜索
# grid_search = GridSearchCV(estimator=xgb_clf,
#                            param_grid=param_grid,
#                            scoring='roc_auc',
#                            cv=5,
#                            verbose=1,
#                            n_jobs=-1)  # 使用所有可用的CPU核心加速搜索

# # 拟合训练数据
# grid_search.fit(X_train, y_train)

# # 输出最佳参数和最佳ROC AUC分数
# print(f'Best parameters: {grid_search.best_params_}')
# print(f'Best ROC AUC: {grid_search.best_score_}')

# # 使用最佳参数重新训练模型
# best_xgb_model = grid_search.best_estimator_

# # 在验证集上进行预测
# y_pred_best_xgb = best_xgb_model.predict_proba(X_val)[:, 1]

# # 计算验证集上的ROC AUC分数
# print(f'Best XGBoost Validation ROC AUC: {roc_auc_score(y_val, y_pred_best_xgb)}')


import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm  # 使用tqdm的notebook后端

class TqdmWrapper:
    def __init__(self, cross_validator, tqdm_class=tqdm):
        self.cross_validator = cross_validator
        self.tqdm_class = tqdm_class

    def split(self, X, y=None, groups=None):
        return self.cross_validator.split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.cross_validator.get_n_splits(X, y, groups)

    def __iter__(self):
        total = self.get_n_splits()
        with self.tqdm_class(total=total) as progress:
            for train_index, test_index in self.cross_validator.split(X, y=None, groups=None):
                yield train_index, test_index
                progress.update(1)

# 定义参数网格
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 400],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0.01, 0.1, 0.5],
    'reg_lambda': [0.1, 0.5, 1.0]
}

# 初始化XGBoost分类器，使用GPU
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42, tree_method='gpu_hist', gpu_id=0, eval_metric='logloss')

# 使用TqdmWrapper包装交叉验证
cv = TqdmWrapper(cross_validator=KFold(n_splits=5))

# 使用GridSearchCV进行超参数搜索
grid_search = GridSearchCV(estimator=xgb_clf,
                           param_grid=param_grid,
                           scoring='roc_auc',
                           cv=cv,
                           verbose=3,
                           n_jobs=-1)  # 使用所有可用的CPU核心加速搜索

# 拟合训练数据
grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳ROC AUC分数
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best ROC AUC: {grid_search.best_score_}')

# 使用最佳参数重新训练模型
best_xgb_model = grid_search.best_estimator_

# 在验证集上进行预测
y_pred_best_xgb = best_xgb_model.predict_proba(X_val)[:, 1]

# 计算验证集上的ROC AUC分数
print(f'Best XGBoost Validation ROC AUC: {roc_auc_score(y_val, y_pred_best_xgb)}')





# 模型融合
from sklearn.ensemble import VotingClassifier

ensemble_model = VotingClassifier(estimators=[
    ('rf', rf),
    ('xgb', best_xgb_model),
    ('lgb', lgb_model),
    ('cb', cb_model)
], voting='soft')

ensemble_model.fit(X_train, y_train)
y_pred_ensemble = ensemble_model.predict_proba(X_val)[:, 1]
print(f'Ensemble Validation ROC AUC: {roc_auc_score(y_val, y_pred_ensemble)}')


# # 使用最佳模型对测试集进行预测
# test_predictions = ensemble_model.predict_proba(test_processed)[:, 1]

# # 生成提交文件
# submission = pd.DataFrame({'ID': test['ID'], 'prediction': test_predictions})

# # 确保保存到 /kaggle/working/ 目录
# submission.to_csv('/kaggle/working/submission.csv', index=False)

# print('Submission file saved as /kaggle/working/submission.csv')


# # 计算分层一致性指数（Stratified C-index）
# from sklearn.metrics import roc_auc_score

# # 假设我们有一个函数来计算分层C-index
# def stratified_c_index(y_true, y_pred, race_groups):
#     c_indices = []
#     for group in race_groups.unique():
#         group_mask = race_groups == group
#         c_index = roc_auc_score(y_true[group_mask], y_pred[group_mask])
#         c_indices.append(c_index)
#     mean_c_index = np.mean(c_indices)
#     std_c_index = np.std(c_indices)
#     return mean_c_index - std_c_index

# # 计算分层C-index
# race_groups = train['race_group']
# stratified_c_index_score = stratified_c_index(y, ensemble_model.predict_proba(X_processed)[:, 1], race_groups)
# print(f'Stratified C-index: {stratified_c_index_score}')


# !pip install lifelines








# !pip install lifelines
# 终端rm -r catboost_info

