import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

print(f'训练集：{train_df.shape}')
print(f'测试集：{test_df.shape}')

train_df.head()


print(train_df.info())
print('############################')
print(train_df.isnull().sum())


# EDA
fig, ax = plt.subplots(1,2,figsize=(12,4))
sns.kdeplot(data=train_df, x='Calories', ax=ax[0], fill=True)
ax[0].set_title('目标变量分布图')
sns.boxplot(data=train_df, x='Calories', ax=ax[1])
ax[1].set_title('目标变量箱线图')
plt.tight_layout()
plt.show()


sns.countplot(x='Sex', data=train_df)
plt.title('性别分布')
plt.tight_layout()
plt.show()


# 单变量分布
col = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

fig, axes = plt.subplots(len(col), 2, figsize=(18, 35))
for i, col in enumerate(col):
    sns.histplot(train_df[col], kde=True, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'{col}分布图', fontsize=22)
    axes[i, 0].grid(True, linestyle='--', alpha=0.7)
    
    sns.boxenplot(train_df[col], ax=axes[i, 1], color='lightgreen')
    axes[i, 1].set_title(f'{col}箱线图', fontsize=22)
    axes[i, 1].grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



# 多变量
cat_col = ['Sex']
fig, ax = plt.subplots(1 ,2, figsize=(12, 6))
sns.violinplot(x='Calories', hue='Sex', data=train_df, ax=ax[0], palette='Set2')
ax[0].set_title('性别与目标变量关系', fontsize=22)

sns.kdeplot( x='Calories', hue='Sex', data=train_df, ax=ax[1], palette='Set2')
ax[1].set_title('性别与目标变量关系', fontsize=22)
plt.legend()
plt.tight_layout()
plt.show()


# encoding Onehot_Encoding
train_df['Sex'] = train_df['Sex'].map({'female': 0, 'male': 1})
test_df['Sex'] = test_df['Sex'].map({'female': 0, 'male': 1})


# heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('特征相关性热力图', fontsize=16)
plt.show()


# 相关系数大于0.1的
feature = ['Age', 'Duration', 'Heart_Rate', 'Body_Temp']
X = train_df[feature]
y = train_df.Calories
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


scaler = StandardScaler()  # 标准化
X_train_scaled = scaler.fit_transform(X_train[feature])
X_test_scaled = scaler.transform(X_test[feature])

score = {}
# xgb
xgb_model = XGBRegressor()
xgb_model.fit(X_train_scaled, y_train)
y_pred = xgb_model.predict(X_test)
score['xgb'] = mean_squared_error(y_test, y_pred)
# lgb
lgb_model = LGBMRegressor()
lgb_model.fit(X_train_scaled, y_train)
y_pred = lgb_model.predict(X_test)
score['lgb'] = mean_squared_error(y_test, y_pred)


plt.figure(figsize=(8, 6))
plt.bar(score.keys(), score.values(), color='skyblue')
plt.title('模型评分', fontsize=16)
plt.ylabel('RMSLE')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


pred = xgb_model.predict(test_df[feature])
pred[pred < 0] = 0 
sub = pd.DataFrame({
    'id':test_df.id,
    'Calories':pred
})
sub.to_csv('submission.csv', index=False)




