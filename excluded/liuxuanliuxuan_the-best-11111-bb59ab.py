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



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.quantiles = {}
        self.mean_values = {}
    
    def fit(self, X, y=None):
        num_cols = ['Monthly_Spending', 'Total_Usage_Hours', 'Account_Age_Months',
                   'Support_Calls', 'Late_Payments', 'Streaming_Usage']
        for col in num_cols:
            if col in X.columns:
                self.quantiles[col] = {
                    'q25': X[col].quantile(0.25),
                    'q75': X[col].quantile(0.75)
                }
                self.mean_values[col] = X[col].mean()
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        # 基础比值特征
        if all(col in X.columns for col in ['Monthly_Spending', 'Total_Usage_Hours']):
            X['Spending_Per_Hour'] = X['Monthly_Spending'] / (X['Total_Usage_Hours'] + 1e-5)
        
        # 交互特征
        if all(col in X.columns for col in ['Gender', 'Location']):
            X['Gender_Location'] = X['Gender'] + '_' + X['Location']
        if all(col in X.columns for col in ['Subscription_Type', 'Last_Interaction_Type']):
            X['Subscription_Interaction'] = X['Subscription_Type'] + '_' + X['Last_Interaction_Type']
        
        # 分位数特征
        for col in self.quantiles:
            X[f'{col}_Above_Q75'] = (X[col] > self.quantiles[col]['q75']).astype(int)
            X[f'{col}_Below_Q25'] = (X[col] < self.quantiles[col]['q25']).astype(int)
        
        # 行为特征组合
        if 'Support_Calls' in X.columns and 'Complaint_Tickets' in X.columns:
            X['Total_Issues'] = X['Support_Calls'] + X['Complaint_Tickets']
        
        if 'Late_Payments' in X.columns and 'Account_Age_Months' in X.columns:
            X['Late_Payment_Rate'] = X['Late_Payments'] / (X['Account_Age_Months'] + 1)
        
        # 二值特征
        if 'Satisfaction_Score' in X.columns:
            X['Is_Low_Satisfaction'] = (X['Satisfaction_Score'] <= 2).astype(int)
        
        if 'Promo_Opted_In' in X.columns:
            X['Promo_Active'] = X['Promo_Opted_In'].astype(int)
        
        return X


train_data = pd.read_csv('/kaggle/input/ultimate-customer-churn-prediction-challenge/train.csv')
test_data = pd.read_csv('/kaggle/input/ultimate-customer-churn-prediction-challenge/test.csv')

# 提取ID和目标
train_ids = train_data['Customer_ID']
test_ids = test_data['Customer_ID']
y = train_data['Churn']




feature_engineer = FeatureEngineer()
train_features = train_data.drop(['Customer_ID', 'Churn'], axis=1)
test_features = test_data.drop(['Customer_ID'], axis=1)

all_data = pd.concat([train_features, test_features])
all_data_engineered = feature_engineer.fit_transform(all_data)

X = all_data_engineered.iloc[:len(train_data)]
X_test = all_data_engineered.iloc[len(train_data):]



categorical_features = [
    'Gender', 'Location', 'Subscription_Type', 
    'Last_Interaction_Type', 'Gender_Location',
    'Subscription_Interaction'
]

numerical_features = [col for col in X.columns 
                     if col not in categorical_features + ['Customer_ID', 'Churn']]

# 删除冗余特征
redundant_features = [
    'Monthly_Spending_ZScore', 
    'Total_Usage_Hours_ZScore',
    'Hourly_Value'
]

X = X.drop(redundant_features, axis=1, errors='ignore')
X_test = X_test.drop(redundant_features, axis=1, errors='ignore')
numerical_features = [col for col in numerical_features if col not in redundant_features]



numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(mutual_info_classif, k=25))
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', TargetEncoder(smoothing=30))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)


base_models = [
    ('xgb', XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=5,
        gamma=0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        eval_metric='auc',
        random_state=42
    )),
    ('cat', CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0
    ))
]

meta_model = LogisticRegression(
    C=0.1,
    penalty='elasticnet',
    solver='saga',
    l1_ratio=0.5,
    max_iter=1000,
    random_state=42
)

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    passthrough=True
)

final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', stacking_model)
])




cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

print("开始交叉验证训练...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    final_pipeline.fit(X_train, y_train)
    val_pred = final_pipeline.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, val_pred)
    cv_scores.append(auc)
    print(f"Fold {fold+1} | AUC: {auc:.5f}")

print(f"\n平均交叉验证AUC: {np.mean(cv_scores):.5f} ± {np.std(cv_scores):.5f}")



val_preds = cross_val_predict(
    final_pipeline, 
    X, 
    y, 
    method='predict_proba',
    cv=cv
)[:, 1]

precision, recall, thresholds = precision_recall_curve(y, val_preds)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"\n最佳F1阈值: {best_threshold:.4f}")

print("\n在全量数据上训练最终模型...")
final_pipeline.fit(X, y)
test_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
test_pred = (test_pred_proba >= best_threshold).astype(int)


submission = pd.DataFrame({
    'Customer_ID': test_ids,
    'Churn': test_pred
})

print("\n预测结果分布:")
print(submission['Churn'].value_counts(normalize=True))

submission.to_csv('optimized_submission.csv', index=False)
print("\n提交文件已保存！")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 自定义颜色方案
colors = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'accent': '#e74c3c',
    'dark': '#34495e',
    'light': '#ecf0f1',
    'churn': '#e74c3c',
    'not_churn': '#2ecc71'
}

# 设置Seaborn样式
sns.set(
    style="whitegrid",
    font="SimHei",
    palette=sns.color_palette([colors['primary'], colors['secondary'], colors['accent'], colors['dark']])
)

# 定义自定义颜色映射
custom_cmap = LinearSegmentedColormap.from_list('churn_cmap', 
                                                [colors['not_churn'], colors['churn']], N=256)

# 加载数据（假设您已经有处理好的数据）
# 这里使用您代码中的数据加载部分
train_data = pd.read_csv('/kaggle/input/ultimate-customer-churn-prediction-challenge/train.csv')
test_data = pd.read_csv('/kaggle/input/ultimate-customer-churn-prediction-challenge/test.csv')
train_ids = train_data['Customer_ID']
test_ids = test_data['Customer_ID']
y = train_data['Churn']

# 假设您已经完成了特征工程
feature_engineer = FeatureEngineer()
train_features = train_data.drop(['Customer_ID', 'Churn'], axis=1)
test_features = test_data.drop(['Customer_ID'], axis=1)
all_data = pd.concat([train_features, test_features])
all_data_engineered = feature_engineer.fit_transform(all_data)
X = all_data_engineered.iloc[:len(train_data)]
X_test = all_data_engineered.iloc[len(train_data):]

# 假设您已经定义了特征类型
categorical_features = [
    'Gender', 'Location', 'Subscription_Type',
    'Last_Interaction_Type', 'Gender_Location',
    'Subscription_Interaction'
]
numerical_features = [col for col in X.columns 
                     if col not in categorical_features + ['Customer_ID', 'Churn']]

# 删除冗余特征
redundant_features = [
    'Monthly_Spending_ZScore',
    'Total_Usage_Hours_ZScore',
    'Hourly_Value'
]
X = X.drop(redundant_features, axis=1, errors='ignore')
X_test = X_test.drop(redundant_features, axis=1, errors='ignore')
numerical_features = [col for col in numerical_features if col not in redundant_features]

# 将目标变量添加到特征数据中以便分析
X_with_churn = X.copy()
X_with_churn['Churn'] = y


# 1.1 客户流失总体分布
plt.figure(figsize=(10, 6))
churn_counts = y.value_counts()
plt.pie(churn_counts, labels=['未流失', '已流失'], autopct='%1.1f%%',
        startangle=90, colors=[colors['not_churn'], colors['churn']])
plt.title('客户流失比例', fontsize=16)
plt.axis('equal')  # 保证饼图是圆形
plt.tight_layout()
plt.show()

# 1.2 数值特征分布与流失关系
plt.figure(figsize=(18, 15))
for i, col in enumerate(numerical_features[:6]):  # 只显示前6个特征
    plt.subplot(2, 3, i+1)
    sns.histplot(X_with_churn, x=col, hue='Churn', multiple='stack',
                 palette=[colors['not_churn'], colors['churn']], kde=True)
    plt.title(f'{col} 分布与流失关系', fontsize=12)
    plt.xlabel(col)
    plt.ylabel('频数')
plt.tight_layout()
plt.show()

# 1.3 分类特征与流失关系
plt.figure(figsize=(18, 12))
for i, col in enumerate(categorical_features[:6]):  # 只显示前6个特征
    if col in X_with_churn.columns:
        plt.subplot(2, 3, i+1)
        churn_rate = X_with_churn.groupby(col)['Churn'].mean()
        churn_rate.plot(kind='bar', color=[colors['primary'], colors['accent']])
        plt.title(f'{col} 的流失率', fontsize=12)
        plt.xlabel(col)
        plt.ylabel('流失率')
        plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 1.4 特征相关性热图
plt.figure(figsize=(16, 12))
numeric_for_corr = X_with_churn.select_dtypes(include=[np.number])
correlation = numeric_for_corr.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', 
            linewidths=.5, cbar_kws={'label': '相关系数'})
plt.title('数值特征相关性热图', fontsize=16)
plt.tight_layout()
plt.show()


# 2.1 原始特征与新特征对比
plt.figure(figsize=(16, 10))
# 选择一些原始特征和对应的新特征进行对比
compare_features = [
    ('Monthly_Spending', 'Spending_Per_Hour'),
    ('Total_Usage_Hours', 'Spending_Per_Hour'),
    ('Support_Calls', 'Total_Issues'),
    ('Late_Payments', 'Late_Payment_Rate')
]

for i, (feat1, feat2) in enumerate(compare_features):
    if feat1 in X_with_churn.columns and feat2 in X_with_churn.columns:
        plt.subplot(2, 2, i+1)
        sns.scatterplot(x=feat1, y=feat2, hue='Churn', data=X_with_churn,
                        palette=[colors['not_churn'], colors['churn']], alpha=0.6)
        plt.title(f'{feat1} 与 {feat2} 的关系', fontsize=12)
plt.tight_layout()
plt.show()

# 2.2 分位数特征效果
plt.figure(figsize=(16, 8))
quantile_features = [col for col in X_with_churn.columns if 'Above_Q75' in col or 'Below_Q25' in col]
for i, col in enumerate(quantile_features[:4]):  # 只显示前4个分位数特征
    base_col = col.replace('_Above_Q75', '').replace('_Below_Q25', '')
    plt.subplot(2, 2, i+1)
    sns.boxplot(x=col, y=base_col, hue='Churn', data=X_with_churn,
                palette=[colors['not_churn'], colors['churn']])
    plt.title(f'{base_col} 的分位数特征效果', fontsize=12)
plt.tight_layout()
plt.show()


# 假设您已经完成了模型训练和交叉验证
# 这里模拟一些交叉验证结果用于可视化
cv_scores = [0.856, 0.842, 0.861, 0.853, 0.849]  # 模拟的5折交叉验证AUC分数

# 3.1 交叉验证结果可视化
plt.figure(figsize=(10, 6))
plt.bar(range(1, 6), cv_scores, color=colors['primary'], edgecolor='black')
plt.axhline(y=np.mean(cv_scores), color=colors['accent'], linestyle='--', label=f'平均AUC: {np.mean(cv_scores):.4f}')
plt.title('5折交叉验证AUC分数', fontsize=16)
plt.xlabel('折数')
plt.ylabel('AUC值')
plt.ylim(0.8, 0.9)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# 3.2 ROC曲线
from sklearn.metrics import roc_curve, auc

# 假设您已经有模型预测概率
# 这里模拟预测概率用于可视化
np.random.seed(42)
y_pred_proba = np.random.rand(len(y))
y_pred_proba = np.where(y == 1, y_pred_proba + 0.2, y_pred_proba)  # 让正例的概率更高

fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color=colors['primary'], lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率', fontsize=14)
plt.ylabel('真正例率', fontsize=14)
plt.title('接收者操作特征曲线 (ROC)', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3.3 精确率-召回率曲线
precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
best_threshold = thresholds[np.argmax(2 * precision * recall / (precision + recall + 1e-8))]
best_f1 = 2 * precision[np.argmax(2 * precision * recall / (precision + recall + 1e-8))] * \
          recall[np.argmax(2 * precision * recall / (precision + recall + 1e-8))] / \
          (precision[np.argmax(2 * precision * recall / (precision + recall + 1e-8))] + 
           recall[np.argmax(2 * precision * recall / (precision + recall + 1e-8))] + 1e-8)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color=colors['secondary'], lw=2)
plt.scatter(recall[np.argmax(f1_scores)], precision[np.argmax(f1_scores)], 
            color=colors['accent'], s=100, marker='*', label=f'最佳F1点: {best_f1:.4f} (阈值: {best_threshold:.4f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('召回率', fontsize=14)
plt.ylabel('精确率', fontsize=14)
plt.title('精确率-召回率曲线 (PR曲线)', fontsize=16)
plt.legend(loc="lower left", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3.4 阈值与F1分数关系
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

plt.figure(figsize=(12, 8))
plt.plot(thresholds, f1_scores[:-1], color=colors['primary'], lw=2)
plt.scatter(best_threshold, np.max(f1_scores), color=colors['accent'], s=100, marker='*', 
            label=f'最佳F1: {np.max(f1_scores):.4f} (阈值: {best_threshold:.4f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('分类阈值', fontsize=14)
plt.ylabel('F1分数', fontsize=14)
plt.title('分类阈值与F1分数关系', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3.5 特征重要性可视化（使用XGBoost模型的特征重要性）
from xgboost import XGBClassifier

# 训练一个XGBoost模型来获取特征重要性
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=5,
    gamma=0.5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    random_state=42
)

# 预处理数据
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder

# 定义预处理管道
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', TargetEncoder(smoothing=30))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# 修正：不依赖ColumnTransformer的内部属性，直接使用定义的转换器
# 处理数值特征
X_numeric = numeric_pipeline.fit_transform(X[numerical_features])

# 处理分类特征，提供y作为目标变量
X_categorical = categorical_pipeline.fit_transform(X[categorical_features], y)

# 合并处理后的特征
X_preprocessed = np.hstack([X_numeric, X_categorical])

# 训练XGBoost模型
xgb_model.fit(X_preprocessed, y)

# 获取特征重要性
feature_importance = xgb_model.feature_importances_

# 修正：手动构建特征名称
numeric_feature_names = [f"num_{col}" for col in numerical_features]
categorical_feature_names = [f"cat_{col}" for col in categorical_features]
feature_names = numeric_feature_names + categorical_feature_names

# 创建特征重要性数据框
importance_df = pd.DataFrame({
    '特征': feature_names,
    '重要性': feature_importance
})
importance_df = importance_df.sort_values('重要性', ascending=False).head(20)  # 只显示前20个

# 绘制特征重要性图
plt.figure(figsize=(12, 8))
sns.barplot(x='重要性', y='特征', data=importance_df, color=colors['primary'])
plt.title('特征重要性排名 (前20个)', fontsize=16)
plt.xlabel('特征重要性', fontsize=14)
plt.ylabel('特征名称', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# 4.1 交互式特征重要性图
import plotly.express as px

fig = px.bar(importance_df, x='重要性', y='特征', orientation='h',
             title='交互式特征重要性排名',
             color='重要性', color_continuous_scale='Viridis',
             labels={'重要性': '特征重要性', '特征': '特征名称'})
fig.update_layout(
    font=dict(family="SimHei, Arial", size=12),
    width=800,
    height=800,
    margin=dict(l=100, r=50, t=80, b=50)
)
fig.show()

# 4.2 3D特征与流失关系图
# 选择三个特征进行3D可视化
if all(col in X_with_churn.columns for col in ['Monthly_Spending', 'Total_Usage_Hours', 'Account_Age_Months']):
    fig = px.scatter_3d(X_with_churn, x='Monthly_Spending', y='Total_Usage_Hours', z='Account_Age_Months',
                        color='Churn', color_continuous_scale=[colors['not_churn'], colors['churn']],
                        title='3D特征与流失关系图',
                        labels={
                            'Monthly_Spending': '月消费额',
                            'Total_Usage_Hours': '总使用时长',
                            'Account_Age_Months': '账户年龄(月)'
                        })
    fig.update_layout(
        font=dict(family="SimHei, Arial", size=12),
        width=900,
        height=700
    )
    fig.show()

# 4.3 交互式ROC曲线
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=fpr, y=tpr,
    mode='lines',
    name=f'ROC曲线 (AUC = {roc_auc:.4f})',
    line=dict(color=colors['primary'], width=2)
))
fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='随机猜测',
    line=dict(color='navy', width=2, dash='dash')
))
fig.update_layout(
    title='交互式ROC曲线',
    xaxis=dict(title='假正例率', range=[0, 1]),
    yaxis=dict(title='真正例率', range=[0, 1.05]),
    font=dict(family="SimHei, Arial", size=14),
    width=800,
    height=600
)
fig.show()

