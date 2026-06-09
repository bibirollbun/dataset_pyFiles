import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold,GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import numpy as np



train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e2/train.csv').drop('id', axis=1)
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv').drop('id', axis=1)
sam = pd.read_csv(r'sample_submission.csv')


train_df.head()


train_df.info()


num_col = [i for i in train_df.columns if train_df[i].dtype == 'float64']
cat_col = [i for i in train_df.columns if i not in num_col]
print(len(num_col), len(cat_col))


# target 
plt.style.use('ggplot')
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(data=train_df, x='Price', bins=50, kde=True, color='blue',ax=ax[0])
ax[0].set_title('Histogram of Target')
sns.boxplot(data=train_df, x='Price', ax=ax[1])
ax[1].set_title('Box Plot of Target')
plt.tight_layout()
plt.show()


fig ,ax = plt.subplots(len(num_col), 2, figsize=(24, 16))
for idx, i in enumerate(num_col):
    # kde plot
    sns.kdeplot(data=train_df, x=i, ax=ax[idx, 0], fill=True)
    ax[idx, 0].set_title('destrbution of {}'.format(i))
    ax[idx, 0].set(xlabel=i)
    # boxplot
    sns.boxplot(data=train_df, x=i, ax=ax[idx, 1])
    ax[idx, 1].set_title('boxplot of {}'.format(i))
plt.tight_layout()
plt.show()
    


fig, ax = plt.subplots(len(cat_col), 2, figsize=(16, 24))
for idx, col in enumerate(cat_col):
    # 柱形图
    sns.countplot(data=train_df, x=col, ax=ax[idx, 0])
    ax[idx, 0].set_title('counts of {}'.format(col))
    ax[idx, 0].grid(True, linestyle='--', alpha=0.7)
    # 饼图
    train_df[col].value_counts().plot(kind='pie', ax=ax[idx, 1])
    ax[idx, 1].set_title('pie of {}'.format(col))
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(len(cat_col), 3, figsize=(32, 34))
for i, col in enumerate(cat_col):
    # 密度图
    sns.kdeplot(x='Price', hue=col, data=train_df, ax=axes[i, 0], palette='Set2', fill=True)
    axes[i, 0].set_title(f'he relationship between {col} and the target variable', fontsize=22)
    axes[i, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 小提琴图
    sns.violinplot(x=col, y='Price', data=train_df, ax=axes[i, 1], palette='Set2')
    axes[i, 1].set_title(f'he relationship between {col} and the target variable', fontsize=22)
    axes[i, 1].grid(True, linestyle='--', alpha=0.7)

    # 频率分布直方图
    sns.histplot(x='Price', hue=col, data=train_df, ax=axes[i, 2], palette='Set2', kde=True)
    axes[i, 2].set_title(f'he relationship between {col} and the target variable', fontsize=22)
    axes[i, 2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


train_df.describe()


print(train_df.isnull().sum())


train_df.dropna(inplace=True)


from sklearn.preprocessing import LabelEncoder
import numpy as np


encoders = {}

for col in cat_col:
    encoder = LabelEncoder()
    train_df[col] = encoder.fit_transform(train_df[col])
    encoders[col] = encoder
    
    # 对测试集进行编码，并处理未见过的标签
    test_labels = test_df[col].unique()
    train_labels = encoder.classes_
    
    # 找出测试集中未见过的标签
    unseen_labels = set(test_labels) - set(train_labels)
    
    if len(unseen_labels) > 0:
        print(f"Column {col} has unseen labels: {unseen_labels}")
        
        test_df[col] = test_df[col].replace(list(unseen_labels), 'Unknown')
        
        # 更新编码器以包含新的 'Unknown' 标签
        all_labels = np.append(train_labels, 'Unknown')
        encoder.fit(all_labels)
    
    test_df[col] = encoder.transform(test_df[col])

# 将 Category 类型转换回 int 类型
for col in cat_col:
    train_df[col] = train_df[col].astype('int')
    test_df[col] = test_df[col].astype('int')

train_df.head()


X = train_df.drop('Price', axis=1)
y = train_df.Price
X_train, X_test, y_train, y_test = train_test_split(X, y)


# 
score = {}
# xgb
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
score['xgb'] = mean_squared_error(y_test, y_pred)
# lgb
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
score['lgb'] = mean_squared_error(y_test, y_pred)


# 可视化
plt.figure(figsize=(8, 6))
plt.bar(score.keys(), score.values(), color='skyblue')
plt.title('score', fontsize=16)
plt.ylabel('MSE')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'num_leaves': [31, 50, 100],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 定义五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 定义 GridSearchCV
grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

# 进行网格搜索
grid_search.fit(X, y)

# 输出最佳参数和最佳得分
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", -grid_search.best_score_)

# 使用最佳参数训练模型
best_lgb_model = grid_search.best_estimator_

# 训练模型
best_lgb_model.fit(X, y)

# 预测测试集
y_pred = best_lgb_model.predict(test_df)

# 如果需要，可以计算训练集上的性能
y_train_pred = best_lgb_model.predict(X)
train_rmse = mean_squared_error(y, y_train_pred, squared=False)
print(f"Training RMSE: {train_rmse}")


pred = best_lgb_model.predict(test_df)
sub = pd.DataFrame({
    'id':sam.id,
    'Price':pred
})
sub.to_csv('submission.csv', index=False)




