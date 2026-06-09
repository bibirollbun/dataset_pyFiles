import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 是否有缺失值
# print(df.isna().sum())
# 对年龄进行分箱
def process_data():
    df_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
    df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
    labels = [0, 1, 2]
    # 训练集 qcut，返回分箱和区间
    df_train['age'], bins = pd.qcut(df_train['age'], 3, labels=labels, retbins=True)
    # 测试集按训练集的分箱区间划分
    df_test['age'] = pd.cut(df_test['age'], bins=bins, labels=labels, include_lowest=True)
    df_train['age'] = df_train['age'].astype(int)
    df_test['age'] = df_test['age'].astype(int)
    # print(df_test.age.dtypes)
    # 处理离散数据job
    threshold = 20000
    counts = df_train['job'].value_counts()
    rare_jobs = counts[counts < threshold].index
    df_train['job'] = df_train['job'].replace(rare_jobs, 'Other')
    df_test['job'] = df_test['job'].replace(rare_jobs, 'Other')
    # print(df_train['job'].value_counts())
    le = LabelEncoder()
    df_train['job'] = le.fit_transform(df_train['job'])
    df_test['job'] = le.transform(df_test['job'])
    # 处理离散婚姻
    le = LabelEncoder()
    df_train['marital'] = le.fit_transform(df_train['marital'])
    df_test['marital'] = le.transform(df_test['marital'])
    # print(df.head())
    # 离散教育
    le = LabelEncoder()
    df_train['education'] = le.fit_transform(df_train['education'])
    df_test['education'] = le.transform(df_test['education'])
    # 是否违约
    le = LabelEncoder()
    df_train['default'] = le.fit_transform(df_train['default'])
    df_test['default'] = le.transform(df_test['default'])
    # 账户余额
    # balance 分箱
    df_train['balance'], balance_bins = pd.cut(df_train['balance'], 3, labels=[0,1,2], retbins=True)
    df_test['balance'] = pd.cut(df_test['balance'], bins=balance_bins, labels=[0,1,2], include_lowest=True)
    df_train['balance'] = df_train['balance'].astype(int)
    df_test['balance'] = df_test['balance'].astype(int)
    # 房产
    le = LabelEncoder()
    df_train['housing'] = le.fit_transform(df_train['housing'])
    df_test['housing'] = le.transform(df_test['housing'])
    le = LabelEncoder()
    df_train['loan'] = le.fit_transform(df_train['loan'])
    df_test['loan'] = le.transform(df_test['loan'])
    # print(df.head())
    # 联系方式
    le = LabelEncoder()
    df_train['contact'] = le.fit_transform(df_train['contact'])
    df_test['contact'] = le.transform(df_test['contact'])
    le = LabelEncoder()
    df_train['poutcome'] = le.fit_transform(df_train['poutcome'])
    df_test['poutcome'] = le.transform(df_test['poutcome'])
    df_train = df_train.drop(['day', 'month', 'pdays', 'previous', 'campaign', 'duration', 'id'], axis=1)
    df_test = df_test.drop(['day', 'month', 'pdays', 'previous', 'campaign', 'duration', 'id'], axis=1)
    # print(df_train.head())
    x_train = df_train.iloc[:,:-1]
    y_train = df_train.iloc[:,-1]
    x_test = df_test
    return x_train, y_train, x_test
    # # 显示所有列
    # pd.set_option('display.max_columns', None)
    #
    # # 显示所有行
    # # pd.set_option('display.max_rows', None)
    #
    # # 设置列宽自适应
    # pd.set_option('display.max_colwidth', None)
if __name__ == '__main__':
    x_train, y_train, x_test =process_data()
    print(x_train, y_train, x_test)


# from process import process_data
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import make_scorer, roc_auc_score
# x_train, y_train, x_test= process_data()
pre = StandardScaler()
x_train = pre.fit_transform(x_train)
x_test = pre.transform(x_test)
# 构建模型

# # svm
# model = SVC(kernel='linear')
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# print(roc_auc_score(y_test, y_pred))
# logstic
#提升树
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.05, 0.1, 0.2],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0]
# }
#
# # 自定义评分指标为 ROC AUC
# roc_scorer = make_scorer(roc_auc_score, needs_proba=True)
#
# # 网格搜索
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     scoring=roc_scorer,
#     cv=3,
#     verbose=2,
#     n_jobs=-1
# )
#
# grid_search.fit(x_train, y_train)
#
# print("最佳参数：", grid_search.best_params_)
# print("最佳 ROC AUC:", grid_search.best_score_)
# 随机森林
# model = RandomForestClassifier(
#     n_estimators=200,       # 树的数量
#     max_depth=None,         # 树的最大深度，可调节防止过拟合
#     random_state=42,
#     n_jobs=-1               # 多线程加速
# )
# 决策树
# model = DecisionTreeClassifier(max_depth=5, random_state=42)
# 逻辑回归
# model = LogisticRegression(solver='liblinear',penalty='l2',C=1.5)
model.fit(x_train, y_train)

y_prob = model.predict_proba(x_test)[:, 1]

print(y_prob)

import pandas as pd

# 重新读取原始测试集，保留 id
df_test_raw = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test_ids = df_test_raw['id']

# 生成提交文件
df_submission = pd.DataFrame({
    'id': test_ids,
    'y': y_prob
})

# 保存为 CSV
df_submission.to_csv('submission.csv', index=False)


