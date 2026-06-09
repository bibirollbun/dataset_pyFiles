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


train_data = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col="id")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
train_data.describe()


train_data.isnull().sum()


train_data.y.value_counts()


train_data.nunique()


cat_col = [cat for cat in train_data.columns if train_data[cat].dtype=="object"]
for col in cat_col:
    print(f"{col}: {train_data[col].nunique()} unique values")
    print(f"  Values: {sorted(train_data[col].unique())}")


from sklearn.preprocessing import OneHotEncoder,StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Mapping features which may have ordinal relation
month_mapping = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5,
    'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

edu_mapping = {
    'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3
}

yes_no_mapping = {
    'yes': 1, 'no': 0
}

income_mapping = {
    'housemaid': 1,
    'student': 1,
    'unemployed': 1,
    'retired': 1,
    'services': 2,
    'technician': 2,
    'blue-collar': 2,
    'self-employed': 2,
    'admin.': 3,
    'management': 3,
    'entrepreneur': 3,
    'unknown': 0,
}

X = train_data.copy()
y = train_data.y
X.drop(columns="y", axis=1, inplace=True)

def data_process(data_to_map, data):
    data_to_map['month'] = data['month'].map(month_mapping)
    data_to_map['education'] = data['education'].map(edu_mapping)
    data_to_map['default'] = data['default'].map(yes_no_mapping)
    data_to_map['housing'] = data['housing'].map(yes_no_mapping)
    data_to_map['loan'] = data['loan'].map(yes_no_mapping)
    data_to_map['job'] = data['job'].map(income_mapping)
    data_to_map['is_new_customer'] = (data_to_map['pdays'] == -1).astype(int)
    data_to_map['debt'] = data_to_map['housing']  + data_to_map['loan']
    data_to_map.drop(columns=['housing', 'loan'], axis=1, inplace=True)
    return data_to_map
X = data_process(X, train_data)

# Unmapped cat features are left to OHE, or feature engineering?
cat = [cat for cat in X.columns if X[cat].dtype=="object"]
num = [num for num in X.columns if X[num].dtype=="int64"]

ohe = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()
le = LabelEncoder()

X['marital'] = le.fit_transform(X['marital'])
X['contact'] = le.fit_transform(X['contact'])
X['poutcome'] = le.fit_transform(X['poutcome'])
X = scaler.fit_transform(X)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, num),
        # ('cat', ohe, cat)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, shuffle=True, stratify=y_train)


from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# # 计算类别权重
# neg_count = (y_train == 0).sum()
# pos_count = (y_train == 1).sum()
# scale_pos_weight = neg_count / pos_count

# # 创建 XGBoost 模型
# xgb = XGBClassifier(
#     n_estimators=200,
#     tree_method='approx',
#     scale_pos_weight=scale_pos_weight,
#     objective='binary:logistic',
#     eval_metric='aucpr'
# )

# # 创建包含预处理和分类器的 Pipeline
# model_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),  # 假设 preprocessor 是之前定义好的预处理步骤
#     ('classifier', xgb)
# ])

# # 设置验证集用于早停（必须先将验证集预处理）
# X_val_transformed = preprocessor.fit_transform(X_val)

# # 定义 fit 参数（关键步骤）
# fit_params = {
#     'classifier__eval_set': [(X_val_transformed, y_val)],  # 添加验证集监控
#     'classifier__early_stopping_rounds': 3,              # 早停轮数
#     'classifier__verbose': False                          # 关闭训练日志
# }

# # 训练模型（传递 fit 参数）
# model_pipeline.fit(X_train, y_train, **fit_params)

# # 预测与评估
# y_pred = model_pipeline.predict(X_val)
# y_prob = model_pipeline.predict_proba(X_val)[:, 1]

# # 计算各项指标
# f1 = f1_score(y_val, y_pred)
# precision = precision_score(y_val, y_pred)
# recall = recall_score(y_val, y_pred)
# auc_score = roc_auc_score(y_val, y_prob)

# print(f"模型的 F1 分数是: {f1:.4f}")
# print(f"模型的 ROC AUC 分数是: {auc_score:.4f}")
# print(f"模型的 Precision 分数是: {precision:.4f}")
# print(f"模型的 Recall 分数是: {recall:.4f}")

# best_rounds = model_pipeline.named_steps['classifier'].best_iteration
# print(f"最佳迭代轮数: {best_rounds}")


from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# rf = RandomForestClassifier(n_estimators=300,              
#                                 criterion='entropy',             
#                                 # max_depth=20,                 
#                                 max_features='sqrt',           
#                                 # min_samples_split=10,         
#                                 # min_samples_leaf=4,             
#                                 class_weight='balanced_subsample',  
#                                 oob_score=True,                
#                                 n_jobs=-1,                    
#                                 random_state=1)
# model_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', rf)
# ])

# model_pipeline.fit(X_train, y_train)

# y_pred = model_pipeline.predict(X_val)
# y_prob = model_pipeline.predict_proba(X_val)[:, 1]

# f1 = f1_score(y_val, y_pred)
# precision = precision_score(y_val, y_pred)
# recall = recall_score(y_val, y_pred)
# auc_score = roc_auc_score(y_val, y_prob)
# print(f"F1 score on val set: {f1:.4f}")
# print(f"ROC AUC score on val set: {auc_score:.4f}")
# print(f"Precision on val set: {precision:.4f}")
# print(f"Recall on val set: {recall:.4f}")

# y_pred = model_pipeline.predict(X_test)
# y_prob = model_pipeline.predict_proba(X_test)[:, 1]
# f1 = f1_score(y_test, y_pred)
# auc_score = roc_auc_score(y_test, y_prob)
# print(f"F1 score on test set: {f1:.4f}")
# print(f"ROC AUC score on test set: {auc_score:.4f}")


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# weight = 395708/54292

# lr = LogisticRegression(
#     solver='lbfgs',
#     max_iter=5000,                        
#     random_state=1,
#     class_weight={0: 1, 1: weight},
#     n_jobs=-1
# )

# model_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', lr)
# ])

# model_pipeline.fit(X_train, y_train)

# y_pred = model_pipeline.predict(X_val)
# y_prob = model_pipeline.predict_proba(X_val)[:, 1]

# f1 = f1_score(y_val, y_pred)
# precision = precision_score(y_val, y_pred)
# recall = recall_score(y_val, y_pred)
# auc_score = roc_auc_score(y_val, y_prob)
# print(f"F1 score on val set: {f1:.4f}")
# print(f"ROC AUC score on val set: {auc_score:.4f}")
# print(f"Precision on val set: {precision:.4f}")
# print(f"Recall on val set: {recall:.4f}")

# y_pred = model_pipeline.predict(X_test)
# y_prob = model_pipeline.predict_proba(X_test)[:, 1]
# f1 = f1_score(y_test, y_pred)
# auc_score = roc_auc_score(y_test, y_prob)
# print(f"F1 score on test set: {f1:.4f}")
# print(f"ROC AUC score on test set: {auc_score:.4f}")


# voting_clf = VotingClassifier(
#     estimators=[
#         ('lr', lr),
#         ('rf', rf),
#         ('xgb', xgb)
#     ],
#     voting='soft',     
#     weights=[1,2,4]    
# )
# voting_clf.fit(X_train, y_train)
# y_pred = voting_clf.predict(X_val)
# y_prob = voting_clf.predict_proba(X_val)[:, 1]

# f1 = f1_score(y_val, y_pred)
# precision = precision_score(y_val, y_pred)
# recall = recall_score(y_val, y_pred)
# auc_score = roc_auc_score(y_val, y_prob)
# print(f"F1 score on val set: {f1:.4f}")
# print(f"ROC AUC score on val set: {auc_score:.4f}")
# print(f"Precision on val set: {precision:.4f}")
# print(f"Recall on val set: {recall:.4f}")




# y_pred = voting_clf.predict(X_test)
# y_prob = voting_clf.predict_proba(X_test)[:, 1]
# f1 = f1_score(y_test, y_pred)
# auc_score = roc_auc_score(y_test, y_prob)
# print(f"F1 score on test set: {f1:.4f}")
# print(f"ROC AUC score on test set: {auc_score:.4f}")


# X_test = test_data.copy()
# X_test = data_process(X_test, test_data)
# X_test['marital'] = le.fit_transform(X_test['marital'])
# X_test['contact'] = le.fit_transform(X_test['contact'])
# X_test['poutcome'] = le.fit_transform(X_test['poutcome'])
# X_test = scaler.fit_transform(X_test)

# voting_clf.fit(X, y)
# pred = voting_clf.predict(X_test)
# prob = voting_clf.predict_proba(X_test)[:, 1]

# submission = pd.DataFrame({"id": test_data["id"], "y": prob})
# submission.to_csv("submission.csv", index=False)
# submission.head()


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

train_data = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col="id")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
cat_col = [cat for cat in train_data.columns if train_data[cat].dtype=="object"]

X = train_data.copy()
X.drop(columns="y", axis=1, inplace=True)
X[cat_col] = X[cat_col].astype('category')
y = train_data.y

X_test = test_data.copy()
X_test.drop(columns="id", axis=1, inplace=True)
X_test[cat_col] = X_test[cat_col].astype('category')

scale_pos_weight = (y == 0).sum() / (y == 1).sum()

xgb = XGBClassifier(
    enable_categorical=True,
    tree_method='hist',
    # tree_method='approx',
    # device='cuda',
    n_estimators=10000,
    learning_rate=0.03,
    max_depth=6,
    scale_pos_weight=scale_pos_weight,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective='binary:logistic',
    eval_metric='aucpr',
    early_stopping_rounds=25,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42, shuffle=True, stratify=y)

xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False )

y_pred = xgb.predict(X_val)
y_prob = xgb.predict_proba(X_val)[:, 1]
f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
auc_score = roc_auc_score(y_val, y_prob)
print(f"F1 score on val set: {f1:.4f}")
print(f"ROC AUC score on val set: {auc_score:.4f}")
print(f"Precision on val set: {precision:.4f}")
print(f"Recall on val set: {recall:.4f}")


pred = xgb.predict(X_test)
prob = xgb.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({"id": test_data["id"], "y": prob})
submission.to_csv("submission.csv", index=False)
submission.head()

