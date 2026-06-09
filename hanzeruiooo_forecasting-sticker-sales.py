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


train_data=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
train_data.info()


train_data.head()


train_data1 = train_data.dropna()
# 遍历所有object类型的字段，查看这些字段的unique()值
for column in train_data1.select_dtypes(include=['object']).columns:
    unique_values = train_data1[column].unique()
    print(f"Unique values in '{column}': {unique_values}")


import pandas as pd
import numpy as np

def add_time_features(dataframe, date_column):
    # 确保日期列是datetime格式
    dataframe[date_column] = pd.to_datetime(dataframe[date_column])

    # 提取常规时间特征
    dataframe['year'] = dataframe[date_column].dt.year
    dataframe['month'] = dataframe[date_column].dt.month
    dataframe['day'] = dataframe[date_column].dt.day
    # dataframe['day_of_week'] = dataframe[date_column].dt.dayofweek  # 周几 (0-6)
    # dataframe['week_of_year'] = dataframe[date_column].dt.isocalendar().week  # 周数
    # dataframe['quarter'] = dataframe[date_column].dt.quarter  # 季度 (1-4)
    # dataframe['is_weekend'] = dataframe[date_column].dt.weekday >= 5  # 是否周末 (周六日为True)

    # 月份的周期性特征
    dataframe['month_sin'] = np.sin(2 * np.pi * dataframe['month'] / 12)
    dataframe['month_cos'] = np.cos(2 * np.pi * dataframe['month'] / 12)

    # # 周几的周期性特征
    # dataframe['day_of_week_sin'] = np.sin(2 * np.pi * dataframe['day_of_week'] / 7)
    # dataframe['day_of_week_cos'] = np.cos(2 * np.pi * dataframe['day_of_week'] / 7)

  
    return dataframe



# 使用这个函数来处理训练集和测试集数据
train_data1 = add_time_features(train_data1, date_column='date')
test_data = add_time_features(test_data, date_column='date')

# 输出结果查看
print(train_data1.head())
print(test_data.head())



train_data2=train_data1.drop(['id','num_sold','date'], axis=1)
test_data2=test_data.drop(['id','date'], axis=1)
label=train_data1['num_sold']
train_data2.shape, test_data2.shape


from sklearn.preprocessing import OneHotEncoder

def encode_categorical_features(train_df, test_df, categorical_columns):
    """
    对指定的类别型特征进行 One-Hot 编码，并对训练集和测试集进行特征对齐。
    
    参数:
    - train_df: 训练集 DataFrame
    - test_df: 测试集 DataFrame
    - categorical_columns: 需要编码的类别型特征列名列表
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # 创建 OneHotEncoder
    train_encoded = train_data2.copy()  # 使用传入的训练集数据
    test_encoded = test_data2.copy()  # 使用传入的测试集数据
    
    for column in categorical_columns:
        # 对训练集进行 fit_transform
        train_encoded_array = encoder.fit_transform(train_df[[column]])
        # 对测试集使用训练集的规则进行 transform
        test_encoded_array = encoder.transform(test_df[[column]])
        
        # 将编码后的数据转换为 DataFrame
        train_encoded_df = pd.DataFrame(train_encoded_array, 
                                        columns=encoder.get_feature_names_out([column]), 
                                        index=train_df.index)
        test_encoded_df = pd.DataFrame(test_encoded_array, 
                                       columns=encoder.get_feature_names_out([column]), 
                                       index=test_df.index)
        
        # 合并编码后的数据到原始 DataFrame 中
        train_encoded = pd.concat([train_encoded, train_encoded_df], axis=1)
        test_encoded = pd.concat([test_encoded, test_encoded_df], axis=1)
        
        # 删除原始的类别型列
        train_encoded.drop(column, axis=1, inplace=True)
        test_encoded.drop(column, axis=1, inplace=True)
    
    # 确保训练集和测试集的列顺序一致
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    
    return train_encoded, test_encoded

# 需要编码的列名
encoder_columns = ['country', 'store', 'product']

# 调用函数进行编码，传入指定的类别列
train_data_encoded, test_data_encoded = encode_categorical_features(train_data2, test_data2, encoder_columns)

# 查看结果维度
print("训练集维度: ", train_data_encoded.shape)
print("测试集维度: ", test_data_encoded.shape)


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import optuna


x = train_data_encoded
y = label

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        'objective': 'regression', 
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 3, 8),  # 限制树的深度
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),  # 增加最小样本数
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.05),  # 降低学习率
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 50.0),  # 增加 L2 正则化的范围
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 50.0),  # 增加 L1 正则化的范围
    }
    
    model = lgb.LGBMRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mape

# 启动优化
study = optuna.create_study(direction='minimize')  # MAPE 越小越好
study.optimize(objective, n_trials=100)

# 使用最佳参数训练模型
best_params = study.best_trial.params
best_model = lgb.LGBMRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# 评估
y_pred = best_model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)

# 输出结果
print("最佳参数: ", best_params)
print("MAPE: {:.5f}".format(mape))


# 使用模型进行预测

predictions = best_model.predict(test_data_encoded)



ids = test_data['id'].copy()

# 创建一个 DataFrame，将预测结果和 id 组合在一起
result = pd.DataFrame({
    'id': ids,
    'num_sold': predictions
})


result.to_csv('prediction_results.csv', index=False)
result

