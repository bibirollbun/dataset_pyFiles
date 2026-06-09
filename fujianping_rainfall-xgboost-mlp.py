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


# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models
import xgboost as xgb  



# 加载数据
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
# 查看数据信息
print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)
print("\nTrain Data Columns:", train_df.columns.tolist())
print("\nMissing Values in Train Data:\n", train_df.isnull().sum())
print("\nMissing Values in Test Data:\n", test_df.isnull().sum())

# 修补缺失值
rows,columns = np.where(test_df.isnull())
rows,columns

test_df['winddirection'] = test_df['winddirection'].fillna(value=test_df['winddirection'].mean())
print("\nMissing Values in Test Data:\n", test_df.isnull().sum())


X = train_df.drop(['id','rainfall'], axis=1)
y = train_df['rainfall']

# 划分训练集和测试集  8:2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 初始化分类器 XGBRegressor XGBClassifier
model = xgb.XGBRegressor(
    objective='binary:logistic',
    n_estimators=100,
    random_state=42,
    eval_metric='logloss'
)

# 训练模型
model.fit(X_train, y_train)


# 使用gain作为重要性指标
importance_gain = model.get_booster().get_score(importance_type='gain')

importance_gain_df = pd.DataFrame({
    'feature': list(importance_gain.keys()),
    'importance': list(importance_gain.values())
}).sort_values('importance', ascending=False)

# 绘制gain重要性图
plt.figure(figsize=(12, 8))
plt.barh(importance_gain_df['feature'], importance_gain_df['importance'])
plt.xlabel('Feature Importance (gain)')
plt.title('XGBoost Feature Importance (average gain)')
plt.gca().invert_yaxis()
plt.show()


# 分离特征与标签
drop_features=['id','rainfall','cloud']

X = train_df.drop(columns=drop_features, axis=1)
y = train_df['rainfall']

# 标准化数值特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# 训练参数
batch_size = 64
epochs = 100

# 训练模型
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping],
    verbose=1
)


y_pred_proba = model.predict(X_val).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

print("Validation ROC-AUC:", roc_auc_score(y_val, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


# 预处理Test数据
drop_features.remove('rainfall')
X_test = test_df.drop(columns=drop_features, axis=1)
X_test_scaled = scaler.transform(X_test)

# 预测概率
test_pred_proba1 = model.predict(X_test_scaled).flatten()

#test_pred_proba1


sub1 = pd.DataFrame({"id": test_df.index, "rainfall": list(test_pred_proba1)})
#sub1


# 分离特征与标签
drop_features=['id','rainfall','dewpoint','humidity','sunshine','windspeed',
               'mintemp','winddirection','pressure','maxtemp','day','temparature']

X = train_df.drop(columns=drop_features,axis=1)

y = train_df['rainfall']

# 标准化数值特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# 训练参数
batch_size = 64
epochs = 100

# 训练模型
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping],
    verbose=1
)

# 验证集评估
y_pred_proba = model.predict(X_val).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

print("Validation ROC-AUC:", roc_auc_score(y_val, y_pred_proba))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


drop_features.remove('rainfall')
# 预处理Test数据
X_test = test_df.drop(columns=drop_features, axis=1)
X_test_scaled = scaler.transform(X_test)

# 预测概率
test_pred_proba2 = model.predict(X_test_scaled).flatten()

#test_pred_proba2


sub2 = pd.DataFrame({"id": X_test.index, "rainfall": list(test_pred_proba2)})
#sub2


sub3 = pd.DataFrame({"id": test_df['id'], "rainfall": 0.99* sub1['rainfall'] + 
                                                          0.01 *  sub2['rainfall'] })
#sub3

sub3.to_csv("submission.csv", index=False)

