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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')


df.head()


df.dtypes


df.info()


object_cols = df.select_dtypes('object').columns.tolist()


object_cols


for col in object_cols:
    print(df[col].value_counts())


df.describe(include='all').T


bool_cols = df.select_dtypes('bool').columns.tolist()
for col in bool_cols:
    print(df[col].value_counts())


num_cols = df.select_dtypes(['int', 'float']).columns.tolist()


num_cols


fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    if i != 0:
        sns.histplot(x=col, data=df, ax=axes[i])
        
plt.tight_layout();


corr_matrix = df[num_cols[1:]].corr()


plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.show()


corr_matrix2 = df[num_cols[1:]].corr(method='spearman')


plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix2, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.show()


def cramers_v(x, y):
    contingency = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency)[0]
    n = contingency.sum().sum()
    phi2 = chi2/n
    r, k = contingency.shape
    return np.sqrt(phi2 / min(k-1, r-1))


cols_categorical = df.select_dtypes(include=['bool', 'object']).columns
corr_matrix = pd.DataFrame(index=cols_categorical, columns=cols_categorical, dtype=float)

# 计算相关性矩阵
for i in cols_categorical:
    for j in cols_categorical:
        corr_matrix.loc[i, j] = cramers_v(df[i], df[j])

# 绘制 heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix.astype(float), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.show()


plt.figure()
sns.boxplot(data=df[['accident_risk', 'curvature']])
plt.show()


plt.scatter(df['curvature'], df['accident_risk'])
plt.ylabel('accident_risk')
plt.xlabel('curvature')
plt.title('Relation between curvature and accident_risk');


cols_to_explore = df.select_dtypes(['int','bool', 'object']).columns.tolist()


len(cols_to_explore)


fig, axes = plt.subplots(3, 4, figsize=(12, 8), sharey=True)
axes = axes.flatten()
for i, col in enumerate(cols_to_explore):
    if i != 0:
        sns.boxplot(x=col, y=df['accident_risk'], data=df, ax=axes[i])

plt.tight_layout();


df['num_reported_accidents'].value_counts()


df['high_accident'] = (df['num_reported_accidents'] >= 3).astype(int)


sns.boxplot(x='high_accident', y='accident_risk', data=df);


df['high_accident'].value_counts()


df['speed_limit_high'] = (df['speed_limit'] >= 60).astype(int)


df['is_night'] = (df['lighting'] == 'night').astype(int)


object_cols = df.select_dtypes('object').columns.to_list()


for col in object_cols:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = df.join(dummies)


bool_cols = df.select_dtypes('bool').columns.to_list()


df[bool_cols] = df[bool_cols].astype(int)


df.columns


# -------- 连续 × 连续 --------
df['curvature_x_speed_limit'] = df['curvature'] * df['speed_limit']
df['curvature_x_num_lanes'] = df['curvature'] * df['num_lanes']

# -------- 连续 × 虚拟变量 --------
df['speed_limit_x_weather_rainy'] = df['speed_limit'] * df['weather_rainy']
df['speed_limit_x_lighting_night'] = df['speed_limit'] * df['lighting_night']
df['curvature_x_lighting_night'] = df['curvature'] * df['lighting_night']
df['num_lanes_x_road_type_urban'] = df['num_lanes'] * df['road_type_urban']

# -------- 分类 × 分类（虚拟变量） --------
df['lighting_night_x_weather_rainy'] = df['lighting_night'] * df['weather_rainy']
df['road_type_urban_x_time_of_day_evening'] = df['road_type_urban'] * df['time_of_day_evening']
df['high_accident_x_weather_rainy'] = df['high_accident'] * df['weather_rainy']
df['high_accident_x_weather_foggy'] = df['high_accident'] * df['weather_foggy']
df['high_accident_x_is_night'] = df['high_accident'] * df['is_night']
df['is_night_x_weather_rainy'] = df['is_night'] * df['weather_foggy']
df['speed_limit_high_x_is_night'] = df['speed_limit_high'] * df['is_night']
df['speed_limit_high_x_weather_rainy'] = df['speed_limit_high'] * df['weather_rainy']
df['speed_limit_high_x_weather_foggy'] = df['speed_limit_high'] * df['weather_foggy']
df['speed_limit_high_x_high_accident'] = df['speed_limit_high'] * df['high_accident']




df.dtypes


features_to_scale = [
    'num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents',
    'curvature_x_speed_limit', 'curvature_x_num_lanes',
    'speed_limit_x_weather_rainy', 'speed_limit_x_lighting_night',
    'curvature_x_lighting_night', 'num_lanes_x_road_type_urban'
]

scaler = StandardScaler()
scaled_values = scaler.fit_transform(df[features_to_scale])

# 在原 DataFrame 中增加标准化列，后缀 _scaled
for i, col in enumerate(features_to_scale):
    df[col + '_scaled'] = scaled_values[:, i]

print("标准化列已添加，原列保留")


df.columns


X_columns2 = ['road_signs_present', 'public_road', 'holiday', 'school_season',
       'high_accident', 'speed_limit_high', 'is_night', 'road_type_rural',
       'road_type_urban', 'lighting_dim', 'lighting_night', 'weather_foggy',
       'weather_rainy', 'time_of_day_evening', 'time_of_day_morning',
       'speed_limit_x_weather_rainy', 'speed_limit_x_lighting_night',
       'curvature_x_lighting_night', 'num_lanes_x_road_type_urban',
       'lighting_night_x_weather_rainy',
       'road_type_urban_x_time_of_day_evening',
       'high_accident_x_weather_rainy', 'high_accident_x_weather_foggy',
       'high_accident_x_is_night', 'is_night_x_weather_rainy',
       'speed_limit_high_x_is_night', 'speed_limit_high_x_weather_rainy',
       'speed_limit_high_x_weather_foggy', 'speed_limit_high_x_high_accident',
       'num_lanes_scaled', 'curvature_scaled', 'speed_limit_scaled',
       'num_reported_accidents_scaled', 'curvature_x_speed_limit_scaled',
       'curvature_x_num_lanes_scaled', 'speed_limit_x_weather_rainy_scaled',
       'speed_limit_x_lighting_night_scaled',
       'curvature_x_lighting_night_scaled',
       'num_lanes_x_road_type_urban_scaled']


X = df[X_columns2]
y = df['accident_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# 训练线性回归模型
lr = LinearRegression()
lr.fit(X_train, y_train)

# 预测
y_pred = lr.predict(X_test)

# 评估
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression Performance:\nRMSE: {rmse:.4f}\nR2: {r2:.4f}")


# 将系数和特征整理成 DataFrame
coef_df = pd.DataFrame({
    'feature': X_columns2,
    'coefficient': lr.coef_
}).sort_values(by='coefficient', key=abs, ascending=False)

# 绘制条形图
plt.figure(figsize=(10,8))
sns.barplot(x='coefficient', y='feature', data=coef_df, palette='vlag')
plt.axvline(0, color='red', linestyle='--')
plt.title('Linear Regression Coefficients')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# Ridge
ridge = Ridge(alpha=1.0)  # alpha 调整正则化强度
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f'Ridge RMSE: {rmse_ridge:.4f}')

# Lasso
lasso = Lasso(alpha=0.01)  # alpha 越大，特征越容易被压到0
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print(f'Lasso RMSE: {rmse_lasso:.4f}')

# 可查看 Lasso 选出的非零特征
selected_features = [f for f, c in zip(X_columns2, lasso.coef_) if c != 0]
print("Lasso 选出的特征:", selected_features)


# 残差分析
y_pred = lr.predict(X_test)
residuals = y_test - y_pred


# 残差分布
plt.figure(figsize=(8,4))
sns.histplot(residuals, bins=30, kde=True)
plt.title('Residual Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


# 残差 vs 预测值
plt.figure(figsize=(8,4))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show()


# 计算残差
residuals = y_test - y_pred
residual_std = residuals.std()

# 标记残差较大样本（超过 2 倍标准差）
outlier_mask = np.abs(residuals) > 2 * residual_std
outliers = X_test[outlier_mask].copy()
outliers['y_true'] = y_test[outlier_mask]
outliers['y_pred'] = y_pred[outlier_mask]
outliers['residual'] = residuals[outlier_mask]


# 回归系数
coef = lr.coef_

#系数对应的列名
feature_names = X_train.columns

# 取绝对值最大的前10个特征
k = 10
top_features = np.abs(coef).argsort()[::-1][:k]
top_feature_names = feature_names[top_features]

print("Top features:", top_feature_names)


fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(top_feature_names):
    sns.kdeplot(X_test[col], label='All Samples', ax=axes[i])
    sns.kdeplot(outliers[col], label='Outliers', color='red', ax=axes[i])
    axes[i].set_title(f'Distribution comparison: {col}')
    axes[i].legend()

plt.tight_layout();


# 初始化随机森林
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# 预测
y_pred_rf = rf.predict(X_test)

# 计算 RMSE
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f'Random Forest RMSE: {rmse_rf:.4f}')


# 初始化模型
xg_reg = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 训练
xg_reg.fit(X_train, y_train)

# 预测
y_pred_xgb = xg_reg.predict(X_test)

# 计算 RMSE
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f'XGBoost RMSE: {rmse_xgb:.4f}')


# 特征重要性
feat_imp_df = pd.DataFrame({
    'feature': X_columns2,
    'importance': xg_reg.feature_importances_
}).sort_values(by='importance', ascending=False)

# 可视化
plt.figure(figsize=(10,8))
sns.barplot(x='importance', y='feature', data=feat_imp_df, palette='plasma')
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


feat_imp_df['feature'][:10].to_list()


top_feature_names.to_list()


# XGBoost 基础模型
xg_reg_adj = xgb.XGBRegressor(
    n_estimators=100,
    random_state=42
)

# 参数网格
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3]
}

# 使用 3 折交叉验证
grid_search = RandomizedSearchCV(
    estimator=xg_reg_adj,
    param_distributions=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1,
)

# 执行网格搜索
grid_search.fit(X_train, y_train)



# 最佳模型预测
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
print(f'XGBoost Tuned RMSE: {rmse_best:.4f}')


residuals_best = y_test - y_pred_best


# 残差分布
plt.figure(figsize=(8,4))
sns.histplot(residuals_best, bins=30, kde=True)
plt.title('Residual Distribution (XGBoost_best)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


# 残差 vs 预测值
plt.figure(figsize=(8,4))
plt.scatter(y_pred_best, residuals_best, alpha=0.6, color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values (XGBoost_best)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show();


plt.scatter(y_test, residuals_best, alpha=0.6, color='blue');
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Acutal Values')
plt.xlabel('Acutual Values')
plt.ylabel('Residuals');


# 构建 DataFrame 对比
df_compare = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred_best
})

# 只筛选真实值为0或1的样本
df_compare_filtered = df_compare[df_compare['y_true'].isin([0, 1])]

# 查看对比
print(df_compare_filtered)


df_compare_filtered = df_compare[df_compare['y_true'].isin([0,1])].copy()
df_compare_filtered['error'] = df_compare_filtered['y_pred'] - df_compare_filtered['y_true']


rmse_y0 = np.sqrt(mean_squared_error(df_compare_filtered[df_compare_filtered['y_true']==0]['y_true'],
                                     df_compare_filtered[df_compare_filtered['y_true']==0]['y_pred']))

rmse_y1 = np.sqrt(mean_squared_error(df_compare_filtered[df_compare_filtered['y_true']==1]['y_true'],
                                     df_compare_filtered[df_compare_filtered['y_true']==1]['y_pred']))

print("y=0 RMSE:", rmse_y0)
print("y=1 RMSE:", rmse_y1)


y_train_pred = best_model.predict(X_train)


plt.hist(y_train_pred[y_train==0], bins=20, alpha=0.5, label='y=0')
plt.hist(y_train_pred[y_train==1], bins=20, alpha=0.5, label='y=1')
plt.legend()
plt.show();


y_test_pred = best_model.predict(X_test)


plt.hist(y_test_pred[y_test==0], bins=20, alpha=0.5, label='y=0')
plt.hist(y_test_pred[y_test==1], bins=20, alpha=0.5, label='y=1')
plt.legend()
plt.show();


explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X_test)


# 全局特征重要性
shap.summary_plot(shap_values, X_test)

