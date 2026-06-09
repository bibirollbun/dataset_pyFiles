import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from copy import deepcopy
# from cuml.preprocessing import TargetEncoder


test_csv = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train_csv = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra_csv = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
train_csv.info()


train_csv = pd.concat([train_csv,train_extra_csv])


# orig = pd.read_csv("/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")
# orig = orig.loc[(orig["Weight Capacity (kg)"]>5)&(orig["Weight Capacity (kg)"]<30)]
# orig.columns = [f"orig_{c}" for c in orig.columns]
# train_csv = train_csv.merge(orig, left_on="Weight Capacity (kg)", right_on="orig_Weight Capacity (kg)", how="left")
# train_csv = train_csv.drop("id",axis=1)
# test_csv = test_csv.merge(orig, left_on="Weight Capacity (kg)", right_on="orig_Weight Capacity (kg)", how="left")
# train_csv.head()


label = 'Price'
cat_columns = []
numeric_columns = []
for col in train_csv.columns:
    if train_csv[col].dtype == 'object':
        cat_columns.append(col)
    if train_csv[col].dtype == 'float64' and col != 'Price':
        numeric_columns.append(col)
cat_columns,numeric_columns


# 缺失值
missing_percentage = train_csv.isna().sum() / len(train_csv) * 100
missing_percentage




# 绘制直方图（按频数排序）
for col in cat_columns:
    # 计算每个大小的频率
    freq = train_csv[col].value_counts().sort_values(ascending=False)
    
    # 绘制饼图
    plt.figure(figsize=(5, 5))
    plt.pie(freq.values,  # 绘图数据
            labels=freq.index,  # 添加类别标签
            autopct='%1.1f%%',  # 显示每个部分的百分比
            startangle=90,  # 设置饼图的起始角度
            wedgeprops={'edgecolor': 'black'})  # 设置饼图边框颜色
    
    plt.title(f'Distribution of {col}')  # 添加标题
    plt.show()


# 绘制直方图（按频数排序）
for col in numeric_columns:
    # 绘制柱状图
    plt.figure()
    sns.histplot(data = train_csv[col], color='blue', edgecolor='black', alpha=0.7,kde=True)
    plt.title(f'{col} (Sorted by Frequency)')
    plt.xlabel('Numeric')
    plt.ylabel('Frequency')
    
    plt.show()


# 标签标准化前
for col in cat_columns:
    sns.boxplot(x=train_csv[col], y=train_csv[label])
    plt.show()



# sns.kdeplot(train_csv['orig_Price'], color="blue",fill=True)
# plt.title("Probability Density Function")
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.show()


# Plotting the scatter plot
plt.figure(figsize=(20, 30))
plt.scatter(train_csv[label].value_counts().index ,train_csv[label].value_counts(), alpha=0.7, color='blue', edgecolor='k', s=50)
plt.xlabel('ID', fontsize=12)
plt.ylabel('Number of Sold Items', fontsize=12)
plt.title('Scatter Plot: ID vs. Number of Sold Items', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()



# Assuming `train_df` is already defined
plt.figure(figsize=(5, 5))  # Increase the size of the plot for better visibility
sns.heatmap(train_csv[numeric_columns + [label]].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})

# Add a title with larger font size
plt.title("Correlation Matrix Heatmap", fontsize=18)
plt.tight_layout()  # Ensures the labels are not cut off

plt.show()



## process data
def feature_engineering(df):
    
    df = deepcopy(df)
    df[cat_columns] = df[cat_columns].fillna('None').astype('string').astype('category')
    # median_weight = df['Weight Capacity (kg)'].median()
    # df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].fillna(median_weight).astype("float64")
    df['Weight Capacity (kg)'].dropna(inplace=True)
    df = df.drop(columns=['id'])
    for col in cat_columns:
        df[col] = df[col].astype("category")
    return df
train_df = feature_engineering(train_csv)
test_df = feature_engineering(test_csv)


train_df.isna().sum()
train_df.info()


cat_columns = []
for col in train_df.columns:
    if train_df[col].dtype == 'category':
        cat_columns.append(col)


# X = train_df.drop(columns=[label])
# y = train_df[label]
# catboost_params = {
#         'loss_function': 'RMSE',
#         'eval_metric': 'RMSE',
#         'learning_rate': 0.40,
#         'iterations': 10000,
#         'depth': 4,
#         'random_strength': 0,
#         'l2_leaf_reg': 5.189087598805998,
#         'task_type':'GPU',
#         'random_seed': 42,
#         'verbose': False    
#     }

# cv = KFold(5, shuffle=True, random_state=0)
# cv_splits = cv.split(X, y)
# scores = []
# cat_test_preds = []
# X_test_pool = Pool(test_df, cat_features=cat_columns)
# for train_idx, val_idx in cv_splits:
#     model = CatBoostRegressor(**catboost_params)
#     X_train_fold, X_val_fold = X.loc[train_idx], X.loc[val_idx]
#     y_train_fold, y_val_fold = y.loc[train_idx], y.loc[val_idx]
#     X_train_pool = Pool(X_train_fold, y_train_fold, cat_features=cat_columns)
#     X_valid_pool = Pool(X_val_fold, y_val_fold, cat_features=cat_columns)
#     model.fit(X=X_train_pool, eval_set=X_valid_pool, verbose=100, early_stopping_rounds=200)
#     val_pred = model.predict(X_valid_pool)
#     score = np.sqrt(mean_squared_error(y_val_fold, val_pred))
#     scores.append(score)
#     test_pred = model.predict(X_test_pool)
#     cat_test_preds.append(test_pred)
    
# print(f'Cross-validated RMSE score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
# print(f'Max RMSE score: {np.max(scores):.3f}')
# print(f'Min RMSE score: {np.min(scores):.3f}')


# # 获取特征重要性
# feature_importance = model.get_feature_importance()

# # 打印特征重要性
# for feature_idx, importance in enumerate(feature_importance):
#     print(f"{train_df.columns[feature_idx]}: {importance}")


# import lightgbm as lgb
# lgb_test_preds = []
# # Define the parameters for the LightGBM regressor
# params = {
#     'objective': 'regression',  # Regression task
#     'metric': 'rmse',           # Root Mean Squared Error
#     'boosting_type': 'gbdt',    # Gradient Boosting Decision Tree
#     'num_leaves': 31,           # Number of leaves in a tree
#     'learning_rate': 0.05,     # Learning rate
#     'feature_fraction': 0.9,    # Fraction of features to use for each tree
#     'bagging_fraction': 0.8,    # Fraction of data to use for each tree
#     'bagging_freq': 5,          # Frequency for bagging
#     'verbose': 1,
#     'device': 'gpu',
#      'n_estimators' : 10000,
#     'early_stopping_rounds' : 200
# }


# X = train_df.drop(columns=[label])
# y = train_df[label]

# cv = KFold(5, shuffle=True, random_state=0)
# cv_splits = cv.split(X, y)
# scores = []
# X_test = test_df.drop(columns=['id'])
# for train_idx, val_idx in cv_splits:
    
#     X_train_fold, X_val_fold = X.loc[train_idx], X.loc[val_idx]
#     y_train_fold, y_val_fold = y.loc[train_idx], y.loc[val_idx]
#     train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
#     valid_data = lgb.Dataset(X_val_fold, label=y_val_fold)
#     # Train the model
#     lgb_model = lgb.train(params, train_data, valid_sets = valid_data)
    
#     # Make predictions on the test set
    
#     val_pred = lgb_model.predict(X_val_fold)
    
#     score = np.sqrt(mean_squared_error(y_val_fold, val_pred))
#     scores.append(score)
#     test_pred = lgb_model.predict(X_test)
#     lgb_test_preds.append(test_pred)
#     print(score)
    
# print(f'Cross-validated RMSE score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
# print(f'Max RMSE score: {np.max(scores):.3f}')
# print(f'Min RMSE score: {np.min(scores):.3f}')


# cat_submi = pd.DataFrame()

# cat_submi['id'] = test_df['id']
# cat_submi[label] = np.mean(cat_test_preds, axis=0)
# cat_submi.to_csv('/kaggle/working/cat_submission.csv', index=False)



# lgb_submi = pd.DataFrame()

# lgb_submi['id'] = test_df['id']
# lgb_submi[label] = np.mean(lgb_test_preds, axis=0)
# lgb_submi.to_csv('/kaggle/working/lgb_submission.csv', index=False)



from cuml.preprocessing import TargetEncoder


# Instantiate TargetEncoder
TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')

features = test_df.columns.tolist()

for col in features:
    if col != 'id':
        TE.fit(train_df[col], train_df[label])  # Fit on training data
        train_df[col] = TE.transform(train_df[col])  # Transform training data
        
        test_df[col] = TE.transform(test_df[col])  # Transform test data

# Function to display summary
def display_summary(df, name):
    print(f"\n{name} Summary:")
    print("-" * 30)
    print("\nData Info:")
    df.info()
    print("\nFirst Rows:")
    display(df.head().T)

# Display summary of the transformed datasets
display_summary(train_df, "Merged Train Dataset")
display_summary(test_df, "Test Dataset")


import lightgbm as lgb
lgb_test_preds = []
# Define the parameters for the LightGBM regressor
params = {
    'objective': 'regression',  # Regression task
    'metric': 'rmse',           # Root Mean Squared Error
    'boosting_type': 'gbdt',    # Gradient Boosting Decision Tree
    'num_leaves': 31,           # Number of leaves in a tree
    'learning_rate': 0.5,     # Learning rate
    'feature_fraction': 0.9,    # Fraction of features to use for each tree
    'bagging_fraction': 0.8,    # Fraction of data to use for each tree
    'bagging_freq': 5,          # Frequency for bagging
    'verbose': -1,
    'device': 'gpu',
     'n_estimators' : 1000,
    'early_stopping_rounds' : 200
}


X = train_df.drop(columns=[label])
y = train_df[label]

cv = KFold(5, shuffle=True, random_state=0)
cv_splits = cv.split(X, y)
scores = []
X_test = test_df
for train_idx, val_idx in cv_splits:
    
    X_train_fold, X_val_fold = X.loc[train_idx], X.loc[val_idx]
    y_train_fold, y_val_fold = y.loc[train_idx], y.loc[val_idx]
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    valid_data = lgb.Dataset(X_val_fold, label=y_val_fold)
    # Train the model
    lgb_model = lgb.train(params, train_data, valid_sets = valid_data)
    
    # Make predictions on the test set
    
    val_pred = lgb_model.predict(X_val_fold)
    
    score = np.sqrt(mean_squared_error(y_val_fold, val_pred))
    scores.append(score)
    test_pred = lgb_model.predict(X_test)
    lgb_test_preds.append(test_pred)
    print(score)
    
print(f'Cross-validated RMSE score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
print(f'Max RMSE score: {np.max(scores):.3f}')
print(f'Min RMSE score: {np.min(scores):.3f}')



# Assuming `train_df` is already defined
plt.figure(figsize=(10, 10))  # Increase the size of the plot for better visibility
sns.heatmap(train_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})

# Add a title with larger font size
plt.title("Correlation Matrix Heatmap", fontsize=18)
plt.tight_layout()  # Ensures the labels are not cut off

plt.show()


