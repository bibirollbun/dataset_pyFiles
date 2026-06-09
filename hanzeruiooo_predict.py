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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train_df.info()


train_df.head()



def fill_missing_values(df):
    """
    填充指定列中的缺失值，使用中位数填充 Episode_Length_minutes 和 Number_of_Ads，
    使用均值填充 Guest_Popularity_percentage。
    
    参数:
    df (pd.DataFrame): 包含缺失值的数据集
    
    返回:
    pd.DataFrame: 处理后的数据集
    """
    # 填充 Episode_Length_minutes 的缺失值为中位数
    df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median(), inplace=True)

    # 填充 Guest_Popularity_percentage 的缺失值为均值
    df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].mean(), inplace=True)

    # 填充 Number_of_Ads 的缺失值为中位数
    df['Number_of_Ads'].fillna(df['Number_of_Ads'].median(), inplace=True)
    
    return df

train_df1 = train_df.dropna()
test_df1 = fill_missing_values(test_df)



train_df1.info()


# 遍历所有object类型的字段，查看这些字段的unique()值
for column in train_df1.select_dtypes(include=['object']).columns:
    unique_values = train_df1[column].unique()
    print(f"Unique values in '{column}': {unique_values}")


train_df2 = train_df1.drop(columns=['id','Podcast_Name'])
test_df2 = test_df1.drop(columns=['id','Podcast_Name'])


train_df2[['Episode_Title']]


# 使用 split 方法提取数字部分
train_df2['Episode_Title'] = train_df2['Episode_Title'].str.split().str[1]

# 将提取的数字转换为整数类型
train_df2['Episode_Title'] = train_df2['Episode_Title'].astype(int)


# 使用 split 方法提取数字部分
test_df2['Episode_Title'] = test_df2['Episode_Title'].str.split().str[1]

# 将提取的数字转换为整数类型
test_df2['Episode_Title'] = test_df2['Episode_Title'].astype(int)




train_df2.info()


test_df2.info()


train = train_df2.drop(columns=['Listening_Time_minutes'])

label = train_df2['Listening_Time_minutes']


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

object_cols = ['Genre','Publication_Day','Publication_Time','Episode_Sentiment']


# 创建一个自定义函数来对每列应用 LabelEncoder
def apply_label_encoder(df):
    label_encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # 保存每列的 LabelEncoder，方便以后逆转编码
    return df

# 使用 FunctionTransformer 来对每一列分别应用 LabelEncoder
encoder = ColumnTransformer(
    transformers=[
        ('encode', FunctionTransformer(apply_label_encoder, validate=False), object_cols)
    ],
    remainder='passthrough'  # 保持其他列不变
)

# 对训练集和测试集进行编码
train_encoded = encoder.fit_transform(train)
test_encoded = encoder.transform(test_df2)

# 将编码后的数据转换回 DataFrame
train_df_encoded = pd.DataFrame(train_encoded, columns=train.columns)
test_df_encoded = pd.DataFrame(test_encoded, columns=test_df2.columns)

# 输出训练集和测试集的形状
train_df_encoded.shape, test_df_encoded.shape



train_df_encoded.info()





from sklearn.metrics import mean_squared_error
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import optuna

# 数据准备
x = train_df_encoded
y = label
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ================== 定义LightGBM的超参数优化函数 ==================
def objective_lgb(trial):
    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.05),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 50.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 50.0),
        'random_state': 42
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, pred))  # 这里返回 RMSE

# ================== 执行超参数优化 ==================
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=20)
best_lgb = study_lgb.best_params
best_lgb['random_state'] = 42

# ================== 定义LightGBM模型并训练 ==================
lgb_model = lgb.LGBMRegressor(**best_lgb)
lgb_model.fit(X_train, y_train)

# ================== 评估原始模型 ==================
pred = lgb_model.predict(X_test)
initial_rmse = np.sqrt(mean_squared_error(y_test, pred))  # 计算 RMSE

initial_rmse




# # ================== 特征回溯过程 ==================
# rmse_results = {}

# for feature in X_train.columns:
#     X_backtest = X_train.drop(columns=[feature])
    
#     # 确保X_train和y_train的样本数一致
#     X_train_backtest, X_test_backtest, y_train_backtest, y_test_backtest = train_test_split(
#         X_backtest, y_train, test_size=0.3, random_state=42
#     )
    
#     # 在去除一个特征后重新训练模型
#     lgb_model.fit(X_train_backtest, y_train_backtest)
#     y_pred_backtest = lgb_model.predict(X_test_backtest)
    
#     # 计算去除特征后的RMSE
#     rmse_backtest = np.sqrt(mean_squared_error(y_test_backtest, y_pred_backtest))
#     rmse_results[feature] = rmse_backtest

# # 输出每个特征去除后的RMSE变化
# print("\n特征去除后的RMSE变化：")
# for feature, rmse_value in rmse_results.items():
#     print(f'去除特征 {feature} 后，RMSE: {rmse_value}, RMSE变化: {rmse_value - initial_rmse}')



# 使用训练好的元模型进行预测
predictions = lgb_model.predict(test_df_encoded)

# 获取test2的id列
ids = test_df['id'].copy()

# 创建一个 DataFrame，将预测结果和 id 组合在一起
result = pd.DataFrame({
    'id': ids,
    'Listening_Time_minutes': predictions
})

# 将结果保存到CSV文件中
result.to_csv('prediction_results.csv', index=False)

# 返回结果
result

