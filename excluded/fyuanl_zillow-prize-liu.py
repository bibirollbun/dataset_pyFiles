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


# This code block is to preprocess your data.

# Please document your file path clearly

### BEGIN YOUR CODE HERE
import pandas as pd
import numpy as np

# 加载数据
properties_2016_data = pd.read_csv('/kaggle/input/zillow-prize-1/properties_2016.csv')
properties_2017_data = pd.read_csv('/kaggle/input/zillow-prize-1/properties_2017.csv')
train_2016_data = pd.read_csv('/kaggle/input/zillow-prize-1/train_2016_v2.csv')

# 合并数据，用于训练模型
merged_data = pd.merge(train_2016_data, properties_2016_data, on='parcelid', how='left')

# 筛选出缺失值比例大于60%的特征, 并排序
selected_remove_features = merged_data.columns[merged_data.isnull().sum()/len(merged_data) > 0.6].tolist()
# 去除缺失值比例大于60%的特征
merged_removed_data = merged_data.drop(columns=selected_remove_features)

# 处理transactiondate特征, 提取年、月、日信息
# merged_removed_data['transactiondate_year'] = pd.to_datetime(merged_removed_data['transactiondate']).dt.year
merged_removed_data['transactiondate_month'] = pd.to_datetime(merged_removed_data['transactiondate']).dt.month
# merged_removed_data['transactiondate_day'] = pd.to_datetime(merged_removed_data['transactiondate']).dt.day
merged_removed_data = merged_removed_data.drop(columns=['transactiondate'])

# 月份、天列三角函数编码
merged_removed_data['transactiondate_month_sin'] = np.sin(2 * np.pi * merged_removed_data['transactiondate_month'] / 12)
merged_removed_data['transactiondate_month_cos'] = np.cos(2 * np.pi * merged_removed_data['transactiondate_month'] / 12)
merged_removed_data = merged_removed_data.drop(columns=['transactiondate_month'])
# merged_removed_data['transactiondate_day_sin'] = np.sin(2 * np.pi * merged_removed_data['transactiondate_day'] / 31)
# merged_removed_data['transactiondate_day_cos'] = np.cos(2 * np.pi * merged_removed_data['transactiondate_day'] / 31)
# merged_removed_data = merged_removed_data.drop(columns=['transactiondate_day'])

# 移除object类型的特征
object_type_features = merged_removed_data.select_dtypes(include=['object']).columns.tolist()
merged_removed_data = merged_removed_data.drop(columns=object_type_features)

# 存在缺失值的特征
missing_features = merged_removed_data.columns[merged_removed_data.isnull().sum() > 0].tolist()
# 填充缺失值
for col in missing_features:
    if 'id' in col.lower():
        merged_removed_data[col] = merged_removed_data[col].fillna(-1)
    else:
        median_value = merged_removed_data[col].median()
        merged_removed_data[col] = merged_removed_data[col].fillna(median_value)

# 去除重复列
merged_removed_data = merged_removed_data.loc[:,~merged_removed_data.columns.duplicated()]
# 去除parcelid列
final_data = merged_removed_data.copy()
# print(final_data.isnull().sum())  # 检查是否还有缺失值
# print(final_data.info())

### END YOUR CODE HERE


# The code block is to train, finetuen, and cross validate your model.

### BEGIN YOUR CODE HERE

import xgboost as xgb
from sklearn.metrics import mean_absolute_error # 房屋价格预测常使用 MAE 或 RMSE
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
import time
import matplotlib.pyplot as plt
import seaborn as sns

# === 为模型准备数据 ===
X = final_data.drop(['logerror'], axis=1)
y = final_data['logerror']

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === 初始化和训练 XGBoost 模型 ===
print("开始训练 XGBoost 模型...")
start_time = time.time()

xgb_model = xgb.XGBRegressor()

# 使用训练数据进行拟合
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)], # 监控测试集上的性能
    verbose=False                 # 设置为True可以查看每一轮的性能变化
)

end_time = time.time()
print(f"XGBoost模型训练完成，耗时：{end_time - start_time:.2f} 秒")

# === 进行预测 ===
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# === 评估模型性能 ===
# 计算训练集的平均绝对误差 (Mean Absolute Error, MAE)
mae_train = mean_absolute_error(y_train, y_pred_train)
# 计算测试集的平均绝对误差 (MAE)
mae_test = mean_absolute_error(y_test, y_pred_test)

print("\n--- 初始模型性能评估 ---")
print(f"训练集 MAE (Mean Absolute Error): {mae_train:.4f}")
print(f"测试集 MAE (Mean Absolute Error): {mae_test:.4f}")

# === 通过交叉检验微调模型 ===
# 定义要搜索的参数网格
param_grid = {
    # 缩小 n_estimators 的范围
    'n_estimators': [500, 800],
    # 调整学习率
    'learning_rate': [0.03, 0.05, 0.07],
    # 调整树的最大深度
    'max_depth': [4, 5, 6],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.6, 0.7]
}

# 定义评分指标
# 目标是最小化 MAE，即最大化 -MAE
neg_mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
print("\n--- 开始 Grid Search 交叉验证进行超参数微调 ---")
search_start_time = time.time()
# 使用 KFold 代替整数 '3'，并设置 random_state
cv_folds = KFold(n_splits=3, shuffle=True, random_state=42)
# 初始化 GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(random_state=42),
    param_grid=param_grid,
    scoring=neg_mae_scorer,    # 使用负 MAE 作为评分标准
    cv=cv_folds,               # 使用 K 折交叉验证
    verbose=1,                 # 设置为 2 以查看详细进度
    n_jobs=-1                  # 使用所有核心进行并行计算
)

grid_search.fit(X_train, y_train)
search_end_time = time.time()
print(f"Grid Search 交叉验证完成，耗时：{(search_end_time - search_start_time) / 60:.2f} 分钟")

# 查看微调最佳结果
print("\n--- Grid Search 最佳结果 ---")
# 注意：best_score_ 是负 MAE，所以我们要取其绝对值
best_params = grid_search.best_params_
best_mae = -grid_search.best_score_
print(f"最佳超参数组合: {best_params}")
print(f"最佳交叉验证 MAE (Mean Absolute Error): {best_mae:.4f}")

# 获取最佳模型
best_xgb_model = grid_search.best_estimator_

# 使用最佳模型进行最终评估
y_pred_test_best = best_xgb_model.predict(X_test)
final_mae_test = mean_absolute_error(y_test, y_pred_test_best)

print("\n--- 最佳模型在独立测试集上的评估 ---")
print(f"最佳模型测试集 MAE (Mean Absolute Error): {final_mae_test:.4f}")

# 分析特征重要性
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_xgb_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# 绘制特征重要性
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in xgboost Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('home_feature_importance.png', dpi=300)
plt.show()

### END YOUR CODE HERE


# The code block is to generate the final submission data.

### BEGIN YOUR CODE HERE

toPredict_2016_data = properties_2016_data.copy()

# 初始化结果 DataFrame，并准备基础数据
final_predictions_2016 = pd.DataFrame(toPredict_2016_data['parcelid'].copy())

# 移除没有参与训练的特征
toPredict_2016_data = toPredict_2016_data.drop(columns=selected_remove_features)
# 移除object类型的特征
object_type_features_2016 = toPredict_2016_data.select_dtypes(include=['object']).columns.tolist()
toPredict_2016_data = toPredict_2016_data.drop(columns=object_type_features_2016)
# 存在缺失值的特征missing_features
# 填充缺失值
for col in missing_features:
    if 'id' in col.lower():
        toPredict_2016_data[col] = toPredict_2016_data[col].fillna(-1)
    else:
        median_value = toPredict_2016_data[col].median()
        toPredict_2016_data[col] = toPredict_2016_data[col].fillna(median_value)

# 删除'parcelid'列
#base_data_2016 = toPredict_2016_data.drop(columns='parcelid')
base_data_2016 = toPredict_2016_data.copy()

# 预测时间点列表
predict_time_2016 = ['201610','201611','201612']
# 批量生成时间特征
for predict_time_str in predict_time_2016:
    # 提取年份和月份
    predict_year = int(predict_time_str[0:4])
    predict_month = int(predict_time_str[4:6])
    # 批量计算三角函数值
    month_sin = np.sin(2 * np.pi * predict_month / 12)
    month_cos = np.cos(2 * np.pi * predict_month / 12)
    # 创建完整的预测数据集 (DataFrame)，并添加所有必需的特征
    predict_data = base_data_2016.assign(
        #transactiondate_year=predict_year, # 批量添加年份
        transactiondate_month_sin=month_sin, # 批量添加 sin 值
        transactiondate_month_cos=month_cos  # 批量添加 cos 值
    )

    # 一次性批量预测所有行。predict_data 是一个完整的 DataFrame
    predict_outcome = best_xgb_model.predict(predict_data)
    # 将预测结果作为一个新列添加到结果 DataFrame 中
    final_predictions_2016[predict_time_str] = predict_outcome


##################
# properties_2017_data
toPredict_2017_data = properties_2017_data.copy()
# 初始化结果 DataFrame，并准备基础数据
final_predictions_2017 = pd.DataFrame(toPredict_2017_data['parcelid'].copy())

# 移除没有参与训练的特征
toPredict_2017_data = toPredict_2017_data.drop(columns=selected_remove_features)
# 移除object类型的特征
object_type_features_2017 = toPredict_2017_data.select_dtypes(include=['object']).columns.tolist()
toPredict_2017_data = toPredict_2017_data.drop(columns=object_type_features_2017)
# 存在缺失值的特征missing_features
# 填充缺失值
for col in missing_features:
    if 'id' in col.lower():
        toPredict_2017_data[col] = toPredict_2017_data[col].fillna(-1)
    else:
        median_value = toPredict_2017_data[col].median()
        toPredict_2017_data[col] = toPredict_2017_data[col].fillna(median_value)

# 删除'parcelid'列
#base_data_2017 = toPredict_2017_data.drop(columns='parcelid')
base_data_2017 = toPredict_2017_data.copy()

# 预测时间点列表
predict_time_2017 = ['201710','201711','201712']
# 批量生成时间特征
for predict_time_str in predict_time_2017:
    # 提取年份和月份
    predict_year = int(predict_time_str[0:4])
    predict_month = int(predict_time_str[4:6])
    # 批量计算三角函数值
    month_sin = np.sin(2 * np.pi * predict_month / 12)
    month_cos = np.cos(2 * np.pi * predict_month / 12)
    # 创建完整的预测数据集 (DataFrame)，并添加所有必需的特征
    predict_data = base_data_2017.assign(
        #transactiondate_year=predict_year, # 批量添加年份
        transactiondate_month_sin=month_sin, # 批量添加 sin 值
        transactiondate_month_cos=month_cos  # 批量添加 cos 值
    )

    # 一次性批量预测所有行。predict_data 是一个完整的 DataFrame
    predict_outcome = best_xgb_model.predict(predict_data)
    # 将预测结果作为一个新列添加到结果 DataFrame 中
    final_predictions_2017[predict_time_str] = predict_outcome


# 最终合并
final_predict_results = pd.merge(final_predictions_2016, final_predictions_2017, on='parcelid', how='outer')

final_predict_results.to_csv("/kaggle/working/final_predict_submission.csv", index=False)
print(f"预测结果已经保存文件到final_predict_submission.csv")


### END YOUR CODE HERE

