import pandas as pd
import numpy as np;
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import StandardScaler;
from sklearn.metrics import accuracy_score;
import tensorflow as tf;
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input;
from tensorflow.keras.metrics import AUC;
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping;
from tensorflow.keras.optimizers import Adam


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
print("ok")


def fill_missing_values(df_):
    columns = df_.columns
    for column in columns:
        if df_[column].dtype == 'object':  # 如果是字符列
            mode_value = df_[column].mode()[0]  # 获取出现次数最多的字符
            df_[column].fillna(mode_value, inplace=True)
        else:  # 如果是数值列
            mean_value = df_[column].mean()  # 计算平均值
            df_[column].fillna(mean_value, inplace=True)
    return df_


train = pd.read_csv('/kaggle/input/xdu-hic-math-2025/train.csv')  
train=fill_missing_values(train)
train['Status'] = train['Status'].map({'Developing': 1, 'Developed': 2})

table_feature=['Year', 'Adult Mortality','Status',
       'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
       ' thinness  1-19 years', ' thinness 5-9 years',
       'Income composition of resources', 'Schooling']


X = train[table_feature]
y = train['Life expectancy ']

# 划分训练集 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("ok")


# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("ok")


# 改为全连接网络（更适合表格数据）
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # 线性激活用于回归
])
 
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',  # 均方误差用于回归
    metrics=['mae']
)

model.summary()
print("ok")


# 训练配置
callbacks = [
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
]
 
# 训练
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)
 
# 评估
model.load_weights('best_model.keras')
y_pred = model.predict(X_test).flatten()
 
# 回归任务评估指标
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
 
print(f'Test MAE: {mae:.4f}')
print(f'Test RMSE: {rmse:.4f}')


#预测测试集 
test = pd.read_csv('/kaggle/input/xdu-hic-math-2025/test.csv')
test=fill_missing_values(test)
test['Status'] = test['Status'].map({'Developing': 1, 'Developed': 2})
end_test = scaler.transform(test[table_feature])
y_pred_test = model.predict(end_test).flatten()

#保存结果
sample_submission = pd.read_csv('/kaggle/input/xdu-hic-math-2025/sample_submission.csv')
sample_submission['Life expectancy ']=y_pred_test
sample_submission.to_csv('MLP_100epoch.csv',index=False)
sample_submission.head()




