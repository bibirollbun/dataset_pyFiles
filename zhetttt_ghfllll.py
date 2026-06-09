import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 读取数据
train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 数据预处理
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()

# 编码特征
train_df['Soil_Type'] = le_soil.fit_transform(train_df['Soil Type'])
train_df['Crop_Type'] = le_crop.fit_transform(train_df['Crop Type'])
test_df['Soil_Type'] = le_soil.transform(test_df['Soil Type'])
test_df['Crop_Type'] = le_crop.transform(test_df['Crop Type'])

# 编码目标变量
y = le_fertilizer.fit_transform(train_df['Fertilizer Name'])

# 特征工程
train_df['N+P+K'] = train_df[['Nitrogen', 'Phosphorous', 'Potassium']].sum(axis=1)
test_df['N+P+K'] = test_df[['Nitrogen', 'Phosphorous', 'Potassium']].sum(axis=1)

# 选择特征
features = [
    'Temparature', 'Humidity', 'Moisture',
    'Soil_Type', 'Crop_Type',
    'Nitrogen', 'Potassium', 'Phosphorous',
    'N+P+K'
]
X = train_df[features]
X_test = test_df[features]

# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# 初始化模型（关键参数设置）
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y)),
    booster='gbtree',
    tree_method='hist',
    max_bin=256,
    eval_metric='mlogloss',
    n_estimators=300,
    learning_rate=0.08,
    max_depth=8,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    verbosity=1,  # 正确日志参数
    # scale_pos_weight=1.0  # 已移除
    early_stopping_rounds=20
)

# 训练模型
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],  # 验证集传递方式
    sample_weight_eval_set=[None]  # 可选权重
)

# 预测处理
test_probs = model.predict_proba(X_test)
top5_indices = np.argsort(test_probs, axis=1)[:, -5:][:, ::-1]
top5_labels = le_fertilizer.inverse_transform(top5_indices.reshape(-1))

# 生成提交文件
submission = []
for idx in range(len(test_df)):
    fertilizers = list(dict.fromkeys(top5_labels[idx*5:(idx+1)*5]))[:5]
    while len(fertilizers) < 5:
        fertilizers.append('UNKNOWN')
    submission.append({
        'id': test_df.iloc[idx]['id'],
        'Fertilizer Name': ' '.join(fertilizers)
    })

# 保存结果
pd.DataFrame(submission, columns=['id', 'Fertilizer Name']).to_csv('submission.csv', index=False)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()

train_df['Soil_Type'] = le_soil.fit_transform(train_df['Soil Type'])
test_df['Soil_Type'] = le_soil.transform(test_df['Soil Type'])

train_df['Crop_Type'] = le_crop.fit_transform(train_df['Crop Type'])
test_df['Crop_Type'] = le_crop.transform(test_df['Crop Type'])

y = le_fertilizer.fit_transform(train_df['Fertilizer Name'])


train_df['N+P+K'] = train_df[['Nitrogen', 'Phosphorous', 'Potassium']].sum(axis=1)
test_df['N+P+K'] = test_df[['Nitrogen', 'Phosphorous', 'Potassium']].sum(axis=1)


features = [
    'Temparature', 'Humidity', 'Moisture',
    'Soil_Type', 'Crop_Type',
    'Nitrogen', 'Potassium', 'Phosphorous',
    'N+P+K'
]
X = train_df[features]
X_test = test_df[features]


model = XGBClassifier(
    objective='multi:softprob',   # 多分类概率输出
    num_class=7,                 # 肥料类别数
    max_depth=8,                 # 树深控制
    learning_rate=0.08,          # 学习率衰减
    subsample=0.7,               # 随机采样比例
    colsample_bytree=0.7,        # 特征采样比例
    n_estimators=300,            # 迭代次数
    early_stopping_rounds=20,    # 早停机制
    verbosity=1                  # 日志输出
)


X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)


model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=10  # 每10轮输出日志
)


test_probs = model.predict_proba(X_test)


top5_indices = np.argsort(test_probs, axis=1)[:, -5:][:, ::-1]
top5_labels = le_fertilizer.inverse_transform(top5_indices.reshape(-1))


submission = []
for idx in range(len(test_df)):
    fertilizers = list(dict.fromkeys(top5_labels[idx*5:(idx+1)*5]))[:5]
    while len(fertilizers) < 5:
        fertilizers.append('UNKNOWN')
    submission.append({
        'id': test_df.iloc[idx]['id'],
        'Fertilizer Name': ' '.join(fertilizers)
    })


pd.DataFrame(submission, columns=['id', 'Fertilizer Name']).to_csv('submission.csv', index=False)

