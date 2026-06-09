import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import joblib

import warnings
warnings.filterwarnings("ignore")


train  = pd.read_csv(f"/kaggle/input/stock-pledge-defaults-prediction/train.csv")
test   = pd.read_csv(f"/kaggle/input/stock-pledge-defaults-prediction/test.csv")


train.info()  # 有2个特征为object类型


train.select_dtypes(include=['object']).columns


train['P/E ratio'] = train['P/E ratio'].apply(lambda x:x.replace(',',''))
train['P/E ratio'] = train['P/E ratio'].astype(float)


train.isnull().sum().sum()  # 无缺失值


train.duplicated().sum()  # 无重复值


train.iloc[:,1:].describe().T


train['IsDefault'].value_counts() # 样本不均衡


# 箱型图探索数据
cols = train.columns.to_list()[1:]
fig = plt.figure(figsize=(40, 90))
for i, col in enumerate(cols):
    ax = fig.add_subplot(21, 3, i+1)
    sns.boxplot(x=train[col], orient='v', width=0.5, ax=ax)
    ax.set_ylabel(col, fontsize=8)
fig.tight_layout()
plt.show()


# 查看数据分布
train_cols = 6
train_rows = len(cols)
plt.figure(figsize=(4*train_cols, 6*train_rows))

i=0
for col in cols:
    i+=1
    ax = plt.subplot(train_rows,train_cols, i)
    sns.histplot(
        train[col], kde=True,
        stat="density", kde_kws=dict(cut=3),
        alpha=.4, edgecolor=(1, 1, 1, .4),
    )

    i+=1
    ax = plt.subplot(train_rows, train_cols, i)
    res = stats.probplot(train[col], plot=plt)
fig.tight_layout()
plt.show()


# 热力图
train_corr = train[cols].corr()
ax = plt.subplots(figsize=(80, 60))
ax = sns.heatmap(train_corr, vmax=.8, square=True, annot=True)


cols  # IsDefault 与 Share pledge ratio of controlling shareholders有较强的线性关系


# 上采样
X = train.iloc[:,1:-1]
y = train.iloc[:,-1]
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# xgboost
model = xgb.XGBClassifier(
    objective='binary:logistic',  # 二分类任务
    n_estimators=100,  # 树的数量
    learning_rate=0.1,  # 学习率
    max_depth=3,  # 树的最大深度
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)


cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1')
np.mean(cv_scores)


# 参数优化
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 0.9]
}

# 初始化网格搜索
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1',  # 评估指标
    cv=5,  # 5折交叉验证
    n_jobs=-1,  # 使用所有CPU核心
    verbose=1
)

# 执行网格搜索
grid_search.fit(X_train, y_train)


grid_search.best_params_


grid_search.best_score_


# 使用最优参数重新训练模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


print(classification_report(y_test, y_pred))


# 随机森林填补测试集中的缺失值
test.select_dtypes(include=['object']).columns


test['P/E ratio'] = test['P/E ratio'].apply(lambda x:x.replace(',',''))
test['P/E ratio'] = test['P/E ratio'].astype(float)


sum(test.isnull().sum() !=0) # 7列有缺失值


filled_data = test.iloc[:,1:]
cols = filled_data.columns
# 找到缺失值所在的列
missing_cols = list(cols[filled_data.isnull().any()])
for col in missing_cols:
    all_data = filled_data.drop(columns=missing_cols[1:])
    train_data = all_data[all_data[col].notnull()]
    train_data.reset_index(inplace=True, drop=True)
    X = train_data.drop(columns=col)
    y = train_data[col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 预测缺失值
    missing_index = all_data[all_data[col].isnull()].index
    test_data = all_data[all_data[col].isnull()]
    test_data.reset_index(inplace=True, drop=True)
    test_X = test_data.drop(columns=col)
    y_pred = model.predict(test_X)

    # 填补缺失值
    filled_data.loc[missing_index, col] = y_pred
    missing_cols = missing_cols[1:]


new_test = pd.concat([test.iloc[:,0], filled_data], axis=1)


new_test.isnull().sum(0).sum(0)


# 可视化填补结果，看是否与原始数据一致
missing_cols = list(test.columns[test.isnull().any()])
for col in missing_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(test[col], ax=axes[0], kde=True, color='skyblue', bins=10)
    axes[0].set_title(f'raw data {col} distribution')
    sns.histplot(new_test[col], ax=axes[1], kde=True, color='lightcoral', bins=10)
    axes[1].set_title(f'raw data {col} distribution')
plt.show()


test_y = best_model.predict(new_test.iloc[:,1:])


test_y


test['Stock code']


test_pred = pd.DataFrame(index=test['Stock code'])
test_pred['IsDefault'] = test_y
test_pred.to_csv('submission.csv', index=True)


test_pred.head()

