# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


!for f in ../input/*.zip; do unzip "$f"; done


# !rm -rf ../input/*


print(os.listdir("./"))



df_gender_age_test = pd.read_csv('./gender_age_test.csv', dtype={'device_id': np.str})
df_gender_age_train = pd.read_csv('./gender_age_train.csv', dtype={'device_id': np.str})

df_app_events = pd.read_csv('./app_events.csv', dtype={'app_id': np.str})
df_events = pd.read_csv('./events.csv', dtype={'device_id': np.str})

df_app_labels = pd.read_csv('./app_labels.csv', dtype={'app_id': np.str})
df_label_categories = pd.read_csv('./label_categories.csv')

df_phone_brands = pd.read_csv('./phone_brand_device_model.csv', dtype={'device_id': np.str})


df_gender_age_test.head()


df_gender_age_test.device_id.nunique(), df_gender_age_test.shape[0]


df_gender_age_train.head()


df_gender_age_train.device_id.nunique(), df_gender_age_train.shape[0]


df_gender_age_train.info()


df_gender_age_train.describe(include='all').T


df_ga_full = pd.concat([df_gender_age_train, df_gender_age_test], axis=0)


df_ga_full.device_id.nunique()


df_events.head()


df_events.event_id.nunique(), df_events.device_id.nunique(), df_events.shape[0]


100 * (df_gender_age_test.device_id.isin(df_events.device_id.unique())).sum()/df_gender_age_test.device_id.nunique()


100 * (df_gender_age_train.device_id.isin(df_events.device_id.unique())).sum()/df_gender_age_train.device_id.nunique()


df_app_events.head()


df_app_events.event_id.nunique(), df_app_events.shape[0]


# df_gender_age_train.device_id[]
in_train_events = df_events[df_events.device_id.isin(set(df_gender_age_train.device_id) & set(df_events.device_id))]
in_train_app_events = df_app_events[df_app_events.event_id.isin(in_train_events.event_id)]
in_train_app_events.event_id.nunique(), in_train_app_events.event_id.size, len(in_train_events)


in_test_events = df_events[df_events.device_id.isin(set(df_gender_age_test.device_id) & set(df_events.device_id))]
in_test_app_events = df_app_events[df_app_events.event_id.isin(in_test_events.event_id)]
in_train_app_events.event_id.nunique(), in_train_app_events.event_id.size, len(in_train_events)


del in_train_events
del in_train_app_events
del in_test_events
del in_test_app_events


import gc
gc.collect()


df_app_labels.head()


df_app_labels.app_id.nunique(), df_app_labels.label_id.nunique(), df_app_labels.shape[0]


df_label_categories.head()


df_label_categories.category.nunique(), df_label_categories.shape[0]


df_phone_brands.head()


df_phone_brands.device_id.nunique(), df_phone_brands.shape[0]


df_phone_brands[df_phone_brands.device_id.isin(df_phone_brands.device_id.value_counts()[df_phone_brands.device_id.value_counts() > 1]\
                                               .index.tolist())].sort_values('device_id')


df_phone_brands.drop_duplicates(subset='device_id', inplace=True)


a = df_phone_brands.groupby(['device_model']).phone_brand.nunique()[df_phone_brands.groupby(['device_model']).phone_brand.nunique() > 1]
a


df_phone_brands[df_phone_brands.device_model.isin(a.index.tolist())].sort_values(['device_model', 'phone_brand'])


a.shape[0]


df_phone_brands.phone_brand = df_phone_brands.phone_brand.map(str.strip).map(str.lower)
df_phone_brands.device_model = df_phone_brands.device_model.map(str.strip).map(str.lower)
df_phone_brands.device_model = df_phone_brands.phone_brand.str.cat(df_phone_brands.device_model)


df_phone_brands.info()


df_phone_brands.describe()


df_ga_full = df_ga_full.merge(df_phone_brands, how='left', on='device_id')


df_train = df_ga_full.loc[df_ga_full.device_id.isin(df_gender_age_train.device_id.tolist())]
df_test = df_ga_full.loc[df_ga_full.device_id.isin(df_gender_age_test.device_id.tolist())]


# sns.kdeplot(df_gender_age_train.age)
fig = plt.figure(figsize=(9, 6))
sns.distplot(df_gender_age_train.age, ax=fig.gca())
plt.title('Age distribution')
sns.despine()


fig = plt.figure(figsize=(7, 4))
sns.barplot(x = df_gender_age_train.gender.value_counts().index, y=df_gender_age_train.gender.value_counts().values, ax=fig.gca())
sns.despine()
plt.title('Gender distribution')


df_gender_age_train.groupby('group').device_id.size().sort_index(ascending=False).plot.barh(title='Age Gender Group Distribution')
sns.despine()


# for brands
c = df_train.phone_brand.value_counts()
# value counts 是自动根据数量按照降序进行排序
market_share = c.cumsum()/c.sum()
# for models
c2 = df_train.device_model.value_counts()
market_share2 = c2.cumsum()/c2.sum()


ax = plt.subplot(1,2,1)
plt.gcf().set_figheight(4)
plt.gcf().set_figwidth(12)
plt.plot(market_share.values, 'b-')
plt.title('Brand share')
sns.despine()

ax = plt.subplot(1,2,2)
plt.plot(market_share2.values, 'g-')
plt.title('Model share')
sns.despine()

plt.subplots_adjust(top=0.8)
plt.suptitle('Brand and model share');


share_majority = market_share[~(market_share>0.95)].index.tolist()
share_others = market_share[market_share>0.95].index.tolist()

share_majority2 = market_share2[~(market_share2>0.60)].index.tolist()
share_others2 = market_share2[market_share2>0.60].index.tolist()


str(share_majority2)


# https://seaborn.pydata.org/tutorial/categorical.html
# sns.swarmplot(x="phone_brand", y="age", hue="gender", data=df_train);
fig = plt.figure(figsize=(20, 6))
ax = sns.boxplot(x="phone_brand", y="age", hue="gender", data=df_train[df_train.phone_brand.isin(share_majority)].sort_values('age'), ax=fig.gca());
ax.set_xticklabels(share_majority, rotation=30);
str(share_majority)


fig = plt.figure(figsize=(20, 6))
ax = sns.boxplot(x="device_model", y="age", hue="gender", data=df_train[df_train.device_model.isin(share_majority2)].sort_values('age'), ax=fig.gca());
ax.set_xticklabels(ax.get_xticklabels(), rotation=30);
str(share_majority2)


df_train.head()


df_app_labels.head()


# groups可以看到每个group长的样子
# df_app_labels.groupby('app_id').label_id.groups
df_app_labels = df_app_labels.groupby('app_id').label_id.apply(lambda x: ' '.join(str(s) for s in x))
df_app_labels.head()


df_app_events.head()


df_app_events ['app_lab'] = df_app_events['app_id'].map(df_app_labels)


df_app_events.head()


df_app_events = df_app_events.groupby('event_id').app_lab.apply(lambda x: ' '.join(str(s) for s in x))


df_app_events.head()


del df_label_categories
del df_app_labels


df_events.head()


df_events['app_lab'] = df_events.event_id.map(df_app_events)


df_events.head()


df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])


df_events['hour'] = df_events['timestamp'].dt.hour


time_large = df_events.groupby('device_id')['hour'].apply(lambda x: max(x))


time_small = df_events.groupby('device_id')['hour'].apply(lambda x: min(x))


from collections import Counter
time_most = df_events.groupby('device_id')['hour'].apply(lambda x: Counter(x).most_common(1)[0][0])


del df_app_events


df_events.app_lab = df_events.app_lab.fillna('Missing')
df_events = df_events.groupby('device_id').app_lab.apply(lambda x: ' '.join(str(s) for s in x))


df_events.head()


df_ga_full['app_lab']= df_ga_full['device_id'].map(df_events)
df_ga_full['time_most']= df_ga_full['device_id'].map(time_most)
df_ga_full['time_large']= df_ga_full['device_id'].map(time_large)
df_ga_full['time_small']= df_ga_full['device_id'].map(time_small)


df_ga_full.head()


del df_train
del df_test
del df_events
del df_phone_brands
del time_large
del time_most
del time_small


fig = plt.figure(figsize=(20, 6))
ax = sns.boxplot(x="time_most", y="age", hue="gender", data=df_ga_full, ax=fig.gca());
ax.set_xticklabels(ax.get_xticklabels(), rotation=30);


fig = plt.figure(figsize=(20, 6))
ax = sns.boxplot(x="time_large", y="age", hue="gender", data=df_ga_full, ax=fig.gca());
ax.set_xticklabels(ax.get_xticklabels(), rotation=30);


fig = plt.figure(figsize=(20, 6))
ax = sns.boxplot(x="time_small", y="age", hue="gender", data=df_ga_full, ax=fig.gca());
ax.set_xticklabels(ax.get_xticklabels(), rotation=30);


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True)
# 将NA当作一个类别来处理。
df_app_lab_vectorized = vectorizer.fit_transform(df_ga_full['app_lab'].fillna('Missing')) 
# 可以考虑使用label category 将feature names替换掉我们更为熟悉的文字表述。
str(vectorizer.get_feature_names())


app_labels = pd.DataFrame(df_app_lab_vectorized.toarray(), columns=vectorizer.get_feature_names(), index=df_ga_full.device_id)
app_labels.head(3)


df_ga_full = df_ga_full.merge(app_labels, how='left', left_on='device_id', right_index=True)


df_ga_full.head(3)


df_ga_full = pd.get_dummies(df_ga_full.drop(columns=['gender', 'age', 'app_lab']), columns=['phone_brand', 'device_model', 'time_most', 'time_large', 'time_small'])


df_ga_full.head(3)


df_ga_full.shape


df_ga_full.info()


df_ga_full.describe()


train = df_ga_full[df_ga_full.device_id.isin(df_gender_age_train.device_id)]
test = df_ga_full[df_ga_full.device_id.isin(df_gender_age_test.device_id)].drop(columns=['group'])

X = train.drop(columns=['group'])
encoder = LabelEncoder()
Y = encoder.fit_transform(train['group'])


X.shape, Y.shape


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# 假设 df_ga_full、df_gender_age_train、df_gender_age_test 已加载并预处理完毕

# 分离训练集和测试集
train = df_ga_full[df_ga_full.device_id.isin(df_gender_age_train.device_id)].copy()
test = df_ga_full[df_ga_full.device_id.isin(df_gender_age_test.device_id)].copy()

# 标签编码
encoder = LabelEncoder()
y = encoder.fit_transform(train['group'])
X = train.drop(columns=['group', 'device_id'])  # 去掉非特征列
test_X = test.drop(columns=['group', 'device_id'])

# 确保训练和测试特征列一致（防止列缺失或顺序不一致）
common_cols = X.columns.tolist()
test_X = test_X.reindex(columns=common_cols, fill_value=0)

# 划分训练/验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 构建 LightGBM Dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# 参数中直接启用早停和日志
params = {
    'objective': 'multiclass',
    'num_class': 12,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'random_state': 42
}

# 训练时通过 evals 和 early_stopping_rounds 控制
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    num_boost_round=2000,
    early_stopping_rounds=50,   # ✅ 直接传参，无需 callback
    verbose_eval=50             # ✅ 每50轮打印一次
)

# 预测测试集（输出概率）
y_pred = model.predict(test_X, num_iteration=model.best_iteration)

# 生成提交文件
submission = pd.DataFrame(
    y_pred,
    columns=encoder.classes_,      # 保持类别顺序（如 F23-26, M32-38...）
    index=test['device_id']
)
submission.index.name = 'device_id'
submission.to_csv('submission_lgbm.csv')

print("✅ Submission file saved as 'submission_lgbm.csv'")


# ==============================
# 1. 解压并加载所有数据
# ==============================
import os
# !for f in ../input/*.zip; do unzip -q "$f"; done

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# 加载基础数据
df_train = pd.read_csv('gender_age_train.csv')
df_test = pd.read_csv('gender_age_test.csv')
df_app_events = pd.read_csv('app_events.csv')
df_events = pd.read_csv('events.csv')
df_phone = pd.read_csv('phone_brand_device_model.csv')

# 所有 device_id
all_device_ids = pd.concat([df_train[['device_id']], df_test[['device_id']]], axis=0)




# ==============================
# 2. 构建 device_id -> app_id 列表（关键修正！）
# ==============================
# 合并 app_events 与 events
df_app_device = df_app_events[['event_id', 'app_id']].merge(
    df_events[['event_id', 'device_id']],
    on='event_id',
    how='inner'
)

# 去重（一个设备可能多次使用同一 App）
df_app_device = df_app_device.drop_duplicates(subset=['device_id', 'app_id'])

# 聚合成 "app_id1 app_id2 ..." 字符串
app_by_device = df_app_device.groupby('device_id')['app_id'].apply(
    lambda x: ' '.join(x.astype(str))
).reset_index()
app_by_device.columns = ['device_id', 'app_list']




# ==============================
# 3. 构建时间特征（每小时是否活跃）
# ==============================
df_events['hour'] = pd.to_datetime(df_events['timestamp']).dt.hour
time_features = df_events.groupby('device_id')['hour'].apply(
    lambda hours: pd.Series({f'time_small_{h}.0': 1 for h in hours.unique()})
).unstack(fill_value=0).reset_index()

# ==============================
# 4. 合并所有信息到主表
# ==============================
df_ga_full = all_device_ids.merge(app_by_device, on='device_id', how='left')
df_ga_full = df_ga_full.merge(time_features, on='device_id', how='left')
df_ga_full = df_ga_full.merge(df_phone, on='device_id', how='left')

# 添加训练标签
df_ga_full = df_ga_full.merge(df_train[['device_id', 'group']], on='device_id', how='left')

# ==============================
# 5. 编码品牌和型号（类别特征）
# ==============================
df_ga_full['phone_brand'] = df_ga_full['phone_brand'].fillna('Unknown')
df_ga_full['device_model'] = df_ga_full['device_model'].fillna('Unknown')

brand_enc = LabelEncoder()
model_enc = LabelEncoder()
df_ga_full['brand'] = brand_enc.fit_transform(df_ga_full['phone_brand'])
df_ga_full['model'] = model_enc.fit_transform(df_ga_full['device_model'])


# ==============================
# 6. 构建 App 二值特征（过滤低频）
# ==============================
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True, min_df=100)  # 只保留 ≥100 次的 App
app_sparse = vectorizer.fit_transform(df_ga_full['app_list'].fillna(''))

app_df = pd.DataFrame(
    app_sparse.toarray(),
    columns=vectorizer.get_feature_names(),
    index=df_ga_full.index
)



# ==============================
# 7. 添加统计特征
# ==============================
df_ga_full['num_apps'] = app_df.sum(axis=1)
time_cols = [col for col in df_ga_full.columns if col.startswith('time_small')]
df_ga_full['num_active_time_slots'] = df_ga_full[time_cols].sum(axis=1)

# 合并 App 特征
df_ga_full = pd.concat([df_ga_full, app_df], axis=1)

# ==============================
# 8. 准备训练/测试集
# ==============================
train = df_ga_full[df_ga_full['device_id'].isin(df_train['device_id'])].copy()
test = df_ga_full[df_ga_full['device_id'].isin(df_test['device_id'])].copy()

le = LabelEncoder()
y = le.fit_transform(train['group'])

# 特征列（排除非特征字段）
exclude_cols = {'device_id', 'app_list', 'phone_brand', 'device_model', 'group'}
feature_cols = [c for c in df_ga_full.columns if c not in exclude_cols]
X = train[feature_cols]
test_X = test[feature_cols].fillna(0)
test_X = test_X.reindex(columns=feature_cols, fill_value=0)

# 划分验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 类别特征声明
categorical_features = ['brand', 'model']










# ==============================
# 9. 训练 LightGBM
# ==============================
train_data = lgb.Dataset(
    X_train, label=y_train,
    categorical_feature=categorical_features,
    free_raw_data=False
)
val_data = lgb.Dataset(
    X_val, label=y_val,
    categorical_feature=categorical_features,
    reference=train_data,
    free_raw_data=False
)

params = {
    'objective': 'multiclass',
    'num_class': 12,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': 6,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'cat_l2': 10,
    'cat_smooth': 10,
    'verbose': -1,
    'random_state': 42
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=2000,
    early_stopping_rounds=50,
    verbose_eval=50
)




# ==============================
# 10. 生成提交文件
# ==============================
y_pred = model.predict(test_X, num_iteration=model.best_iteration)

submission = pd.DataFrame(
    y_pred,
    columns=le.classes_,
    index=test['device_id']
)
submission.index.name = 'device_id'
submission.to_csv('submission.csv')

# 显示下载链接
from IPython.display import FileLink
print("✅ Ready to download:")
display(FileLink('submission.csv'))


from IPython.display import FileLink
FileLink('submission_lgbm.csv')


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
# scores = cross_val_score(LogisticRegression(), X, Y, scoring='neg_log_loss',cv=10, verbose=1)


# scores.mean(), scores


# from sklearn.cross_validation import cross_val_predict
# y_pred = cross_val_predict(LogisticRegression(), X, Y, cv=10, n_jobs=-1, verbose=1)
# log_loss(Y, y_pred)


# from sklearn.model_selection import StratifiedKFold
# kf = StratifiedKFold(n_splits=10, random_state=0)
# pred = np.zeros((Y.shape[0], Y.nunique()))
# for train_index, test_index in kf.split(X, Y):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
#     lr = LogisticRegression(solver='sag').fit(X_train, y_train)
#     pred[test_index,:] = lr.predict_proba(X_test)
#     # Downsize to one fold only for kernels
#     print("{:.5f}".format(log_loss(y_test, pred[test_index, :]), end=' '))

# # log_loss(Y, pred)


import xgboost as xgb
from sklearn.model_selection import train_test_split

X.set_index('device_id', inplace=True)
X_train, X_val, y_train, y_val = train_test_split(X, Y, train_size=.80)

##################
#     XGBoost
##################

dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_val, y_val)

params = {
    "objective": "multi:softprob",
    "num_class": 12, # Y一共有12个类别
    "booster": "gbtree", # 默认为基于树的模型gbtree,还有基于线性模型的gbliner。
    "eval_metric": "mlogloss",
    "eta": 0.3, # 和GBM中的 learning rate 参数类似。
    "silent": 0, # 用于控制输出的信息，1静默模式，0默认，输出更多的，以帮助我们更好的理解。
}
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, 140, evals=watchlist, verbose_eval=True)


test.set_index('device_id', inplace=True)
y_pre = gbm.predict(xgb.DMatrix(test), ntree_limit=gbm.best_iteration)
# scores = cross_val_score(RandomForestClassifier(n_est


# from sklearn.ensemble import RandomForestClassifier
# scores = cross_val_score(RandomForestClassifier(n_estimators=100), X, Y, scoring='neg_log_loss',cv=10, verbose=1)


# scoresmean(), scores.


pd.read_csv('../input/sample_submission.csv').head()


result = pd.DataFrame(y_pre, index=test.index, columns=encoder.classes_)
result.head()


result.to_csv('./predict_prob.csv')


pd.read_csv('./predict_prob.csv').head()




