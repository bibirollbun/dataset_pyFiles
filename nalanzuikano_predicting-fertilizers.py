# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


print(f'训练集维度 / Train shape:{train.shape}\n')
print(f'训练集概况 / Train info:')
print(train.info())
print('训练集特征 / Train statistics:')
display(train.describe(), train.head())


print(f'测试集维度 / Test shape:{test.shape}\n')
print(f'测试集概况 / Test info:')
print(test.info())
print('测试集特征 / Test statistics:')
display(test.describe(), test.head())


# # 查看训练集和测试集中的类别特征是否一致
# # Check if categorical features are consistent between train and test sets
# soil_types_train = train['Soil Type'].dropna().unique().tolist()
# soil_types_test = test['Soil Type'].dropna().unique().tolist()

# crop_types_train = train['Crop Type'].dropna().unique().tolist()
# crop_types_test = test['Crop Type'].dropna().unique().tolist()

# fertilizer_name = train['Fertilizer Name'].dropna().unique().tolist()

# print(f'Soil Type train:{soil_types_train}\n Soil Type test:{soil_types_test}')
# print(f'Crop Type train:{crop_types_train}\n Crop Type test:{crop_types_test}')
# print(f'Fertilizer Name:{fertilizer_name}')
# # Soil Type train:['Clayey', 'Sandy', 'Red', 'Loamy', 'Black']
# #  Soil Type test:['Sandy', 'Red', 'Clayey', 'Black', 'Loamy']
# # Crop Type train:['Sugarcane', 'Millets', 'Barley', 'Paddy', 'Pulses', 'Tobacco', 'Ground Nuts', 'Maize', 'Cotton', 'Wheat', 'Oil seeds']
# #  Crop Type test:['Wheat', 'Sugarcane', 'Ground Nuts', 'Pulses', 'Millets', 'Barley', 'Oil seeds', 'Paddy', 'Cotton', 'Maize', 'Tobacco']
# # Fertilizer Name:['28-28', '17-17-17', '10-26-26', 'DAP', '20-20', '14-35-14', 'Urea']


from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# 指定特征列
# Specify feature columns
categorical_cols = ['Soil Type', 'Crop Type']
numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
target_col = 'Fertilizer Name'

# 处理输入特征和目标变量
# Prepare input features and target variable
X = train.drop(columns='Fertilizer Name')
y = train['Fertilizer Name']

# 划分数据集、训练集和验证集
# Split dataset into training and validation sets
X_train_raw, X_valid_raw, y_train_raw, y_valid_raw = train_test_split(X, y, test_size=0.2, random_state=42)

# 测试集，保持和训练集特征一致
# Prepare test set with same feature columns
test = test[X_train_raw.columns]
X_test_raw = test

# 计算 CatBoost需要的类别特征索引 
# Get categorical feature indices for CatBoost
cat_feature_indices = [X_test_raw.columns.get_loc(col) for col in categorical_cols]

# 定义数值和类别特征的转换器
# Define transformers for numerical and categorical features
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
numerical_transformer = StandardScaler()

# 创建列转换器，分别处理不同类型特征
# Create column transformer to preprocess features separately
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# 预处理训练、验证和测试集特征
# Fit on train, transform on train/valid/test sets
X_train_preprocessed  = preprocessor.fit_transform(X_train_raw)
X_valid_preprocessed  = preprocessor.transform(X_valid_raw)
X_test_preprocessed  = preprocessor.transform(test)

# 对目标变量进行编码，将类别标签转换为整数编码
# Encode target variable labels as integer
label_encoder = LabelEncoder()
y_train_encoded  = label_encoder.fit_transform(y_train_raw)
y_valid_encoded  = label_encoder.transform(y_valid_raw)


from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.5,
    reg_lambda=1.5,
    gamma=0.1,
    use_label_encoder=False,
    eval_metric='mlogloss',
    early_stopping_rounds=30,
    random_state=42,
    verbosity=0
)

xgb_model.fit(
    X_train_preprocessed, y_train_encoded,
    eval_set=[(X_train_preprocessed, y_train_encoded), (X_valid_preprocessed, y_valid_encoded)],
    verbose=False
)


# 预测概率与top-3类别评估
# Predict probabilities and evaluate top-3 accuracy for XGB model
from sklearn.metrics import accuracy_score

xgb_proba = xgb_model.predict_proba(X_valid_preprocessed)
xgb_top3_idx = np.argsort(xgb_proba, axis=1)[:, -3:][:, ::-1]
xgb_top3_labels = label_encoder.inverse_transform(xgb_top3_idx.ravel()).reshape(xgb_top3_idx.shape)
xgb_true_labels = label_encoder.inverse_transform(y_valid_encoded)

# 计算验证集中真实标签是否在预测的top3中
# Calculate if true label is within predicted top 3 labels
xgb_correct = [true in preds for true, preds in zip(xgb_true_labels, xgb_top3_labels)]
xgb_top3_acc = np.mean(xgb_correct)
print(f'XGB Top-3 Accuracy: {xgb_top3_acc:.4f}')
# XGB Top-3 Accuracy:0.5230


# 使用XGBoost单模型预测测试集的类别概率
# Predict class probabilities on the test set using the XGBoost single model
pred_proba = xgb_model.predict_proba(X_test_preprocessed)

# 对每行预测概率排序，取概率最高的前三个类别索引
# For each row, sort predicted probabilities and select the indices of the top 3 classes
top3_idx = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]

# 将类别索引转换回原始标签名称
# Convert class indices back to original label names
top3_labels = label_encoder.inverse_transform(top3_idx.ravel()).reshape(top3_idx.shape)

# 构造提交文件的数据框，包含id和对应的预测前三个类别名称（空格分隔）
# Build submission DataFrame with id and predicted top 3 class names joined by space
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})

# 将提交结果保存为csv文件，不包含行索引
# Save the submission DataFrame as a CSV file without the index
submission.to_csv('/kaggle/working/submission.csv', index=False)


# from lightgbm import LGBMClassifier

# lgb_model = LGBMClassifier(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=7,
#     num_leaves=80,
#     subsample=0.85,
#     colsample_bytree=0.8,
#     min_child_samples=10,
#     reg_alpha=1.0,
#     reg_lambda=1.0,
#     random_state=42,
#     early_stopping_round=30,
#     verbose=-1
# )



# lgb_model.fit(
#     X_train_preprocessed, y_train_encoded,
#     eval_set=[(X_train_preprocessed, y_train_encoded), (X_valid_preprocessed, y_valid_encoded)],
#     eval_metric='multi_logloss'
# )


# # 预测概率与top-3类别评估
# # Predict probabilities and evaluate top-3 accuracy for LGB model
# lgb_proba = lgb_model.predict_proba(X_valid_preprocessed)
# lgb_top3_idx = np.argsort(lgb_proba, axis=1)[:, -3:][:, ::-1]
# lgb_top3_labels = label_encoder.inverse_transform(lgb_top3_idx.ravel()).reshape(lgb_top3_idx.shape)
# lgb_true_labels = label_encoder.inverse_transform(y_valid_encoded)

# # 计算验证集中真实标签是否在预测的top3中
# # Calculate if true label is within predicted top 3 labels
# lgb_correct = [true in preds for true, preds in zip(lgb_true_labels, lgb_top3_labels)]
# lgb_top3_acc = np.mean(lgb_correct)
# print(f'LGB Top-3 Accuracy: {lgb_top3_acc:.4f}')
# # LGB Top-3 Accuracy: 0.5211


# from catboost import CatBoostClassifier

# cat_model = CatBoostClassifier(
#     iterations=700,
#     learning_rate=0.03,
#     depth=6,
#     l2_leaf_reg=8,
#     bagging_temperature=0.5,
#     loss_function='MultiClass',
#     eval_metric='MultiClass',
#     early_stopping_rounds=50,
#     verbose=0,
#     random_state=42
# )

# cat_model.fit(
#     X_train_raw, y_train_encoded,
#     eval_set=[(X_valid_raw, y_valid_encoded)],
#     cat_features=cat_feature_indices
# )


# # 预测概率与top-3类别评估
# # Predict probabilities and evaluate top-3 accuracy for Cat model
# cat_proba = cat_model.predict_proba(X_valid_raw)
# cat_top3_idx = np.argsort(cat_proba, axis=1)[:, -3:][:, ::-1]
# cat_top3_labels = label_encoder.inverse_transform(cat_top3_idx.ravel()).reshape(cat_top3_idx.shape)
# cat_true_labels = label_encoder.inverse_transform(y_valid_encoded)

# # 计算验证集中真实标签是否在预测的top3中
# # Calculate if true label is within predicted top 3 labels
# cat_correct = [true in preds for true, preds in zip(cat_true_labels, cat_top3_labels)]
# cat_top3_acc = np.mean(cat_correct)
# print(f'Cat Top-3 Accuracy: {cat_top3_acc:.4f}')
# # Cat Top-3 Accuracy:0.5047


# # 多模型加权融合，寻找最优权重组合
# # Weighted ensemble of models: find best weights for XGB, LGB, CatBoost
# best_score = 0
# best_weights = (0, 0, 0)

# for w1 in np.arange(0, 1.01, 0.1):
#     for w2 in np.arange(0, 1.01 - w1, 0.1):
#         w3 = 1.0 - w1 - w2
#         if w3 < 0 or w3 > 1:
#             continue

#         # 加权融合预测概率
#         # Weighted fusion of predicted probabilities
#         fused_proba = w1 * xgb_proba + w2 * lgb_proba + w3 * cat_proba

#         # 不进行 inverse_transform，直接用整数标签比较top3预测
#         # Directly compare integer labels for top-3 predicted classes without decoding
#         fused_top3_idx = np.argsort(fused_proba, axis=1)[:, -3:][:, ::-1]

#         # 计算top-3准确率
#         # Calculate top-3 accuracy
#         fused_correct = [true in top3 for true, top3 in zip(y_valid_encoded, fused_top3_idx)]
#         fused_score = np.mean(fused_correct)

#         # 记录最佳组合和得分
#         # Update best weights if current score is better
#         if fused_score > best_score:
#             best_score = fused_score
#             best_weights = (w1, w2, w3)

# print(f"最优加权组合: XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, Cat={best_weights[2]:.2f}")
# print(f"融合 Top-3 Accuracy: {best_score:.4f}")
# # 最优加权组合: XGB=0.50, LGB=0.50, Cat=0.00
# # 融合 Top-3 Accuracy: 0.5233


# # 使用最优权重对三个模型的预测概率进行加权融合
# # Weighted ensemble of predicted probabilities from three models using the best-found weights
# weights = [0.50, 0.50, 0.00]

# weighted_preds = (
#     weights[0] * xgb_model.predict_proba(X_test_preprocessed) +
#     weights[1] * lgb_model.predict_proba(X_test_preprocessed) +
#     weights[2] * cat_model.predict_proba(X_test_raw) 
# )

# # 取加权后预测的top-3类别索引
# # Get top-3 class indices from the weighted predictions
# top3_idx = np.argsort(weighted_preds, axis=1)[:, -3:][:, ::-1]

# # 将类别索引转回标签并调整形状
# # Convert class indices back to original labels and reshape
# top3_labels = label_encoder.inverse_transform(top3_idx.ravel()).reshape(top3_idx.shape)

# submission = pd.DataFrame({
#     'id': test['id'],
#     'Fertilizer Name': [' '.join(row) for row in top3_labels]
# })

# submission.to_csv('kaggle/working/submission.csv', index=False)




