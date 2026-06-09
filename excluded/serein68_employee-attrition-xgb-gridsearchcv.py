import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


from warnings import simplefilter
import math
from pandas_profiling import ProfileReport

simplefilter('ignore')

train1= pd.read_csv("/kaggle/input/playground-series-s3e3/train.csv")
train2 = pd.read_csv("/kaggle/input/ibmdata/WA_Fn-UseC_-HR-Employee-Attrition.csv")

set(train1.columns.append(train2.columns)) - set(train1.columns)
set(train1.columns.append(train2.columns)) - set(train2.columns)

train2.Attrition = train2.Attrition.map({'Yes': 1, 'No': 0})
train1 = train1.drop('id', axis = 1)
train2 = train2.drop('EmployeeNumber', axis = 1)
train = pd.concat([train1, train2])
print(train.head())
#train.to_csv('HR_Employee_Attrition_Merged.csv', index=False)


# 检查 Attrition 分布
print("Attrition Distribution:")
print(train['Attrition'].value_counts(normalize=True).map("{:.2%}".format).to_frame().T)

# 分析 OverTime 与 Attrition 的关系
print("\nOverTime vs Attrition (Proportion):")
crosstab = pd.crosstab(train['OverTime'], train['Attrition'], normalize='index')
print(crosstab)

import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import seaborn as sns
import pandas as pd

# --- 1. 下载并准备字体文件 ---

# 字体名称和下载链接 (使用 Noto Sans CJK SC，开源免费)
font_name = "Noto Sans CJK SC"
font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
font_path_local = "./NotoSansCJKsc-Regular.otf" # 下载到当前工作目录

# 如果字体文件不存在，则下载
if not os.path.exists(font_path_local):
    print(f"正在下载 {font_name} 字体...")
    !wget -q -O {font_path_local} {font_url}
    print("字体下载完成！")
else:
    print(f"{font_name} 字体已存在。")

# --- 2. 强制 Matplotlib 加载字体 ---

# 将字体文件添加到 Matplotlib 的字体管理器
try:
    font_prop = fm.FontProperties(fname=font_path_local)
    fm.fontManager.addfont(font_path_local)
    print(f"成功将字体文件 '{font_path_local}' 添加到 Matplotlib。")
    
    # 验证字体是否被正确识别
    if font_name in [f.name for f in fm.fontManager.ttflist]:
        print(f"✅ 验证成功：Matplotlib 已识别 '{font_name}' 字体。")
    else:
        print(f"❌ 验证失败：Matplotlib 未识别 '{font_name}' 字体。请检查字体文件路径。")

except Exception as e:
    print(f"❌ 加载字体时出错: {e}")

# --- 3. 全局配置 Matplotlib ---

# 设置全局字体为我们刚刚加载的字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = [font_name]  # 使用字体的名称
rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
print("Matplotlib 全局字体配置已更新。")

# 创建中文标签的交叉表
crosstab_chinese = crosstab.copy()
# 重命名索引为中文
crosstab_chinese.index = crosstab_chinese.index.map({'Yes': '是', 'No': '否'})
    
# 绘制交叉表热图
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(crosstab_chinese, annot=True, fmt='.2%', cmap='Blues', cbar=True, ax=ax)

# 使用 FontProperties 设置标题和标签
font_prop_title = fm.FontProperties(fname=font_path_local, size=14)
font_prop_label = fm.FontProperties(fname=font_path_local, size=12)

ax.set_title('加班状态与员工离职的关联分析', fontproperties=font_prop_title, pad=15)
ax.set_xlabel('离职（0: 否, 1: 是）', fontproperties=font_prop_label)
ax.set_ylabel('是否加班', fontproperties=font_prop_label)

# 设置 y 轴刻度标签（确保中文显示）
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_prop_label, rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_prop_label, rotation=0)

output_image_path = 'overtime_attrition_heatmap.png'
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"热图已保存至 '{output_image_path}'")



print('Any missing values?', train.isna().sum().sum())


# using dictionary to convert specific columns, prepare for modeling

convert_dict = {'Education': object,
                'EnvironmentSatisfaction': object,
                'JobInvolvement':object,
                'JobLevel':object,
                'JobSatisfaction':object,
                'PerformanceRating':object,
                'RelationshipSatisfaction':object,
                'StockOptionLevel':object,
                'WorkLifeBalance':object,             
                }
 
train = train.astype(convert_dict)
train.dtypes




from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_absolute_error,cohen_kappa_score
import sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline


# target
y = train.Attrition

#features
# Select columns corresponding to features
X_full= train.drop(['Attrition'],axis=1)


# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_full.columns if
                    X_full[cname].nunique() < 10 and 
                    X_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_full.columns if 
                X_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X = X_full[my_cols].copy()

# 划分训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=8)


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])




# setup for model training 

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=8)

model = xgb.XGBClassifier(random_state = 1)

pipeline = Pipeline([
    ('preprocessor', preprocessor),  
    ('model', model)
])

param_grid = {
    'model__max_depth': [2, 3, 5, 7, 10],
    'model__n_estimators': [10, 100, 500],
    'model__learning_rate': [0.01, 0.03, 0.05, 0.1]
}




# GridSearchCV
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')

grid.fit(X_train, y_train)


!pip install shap


# 导入 SHAP
import shap

# 获取最佳模型
best_model = grid.best_estimator_

# 对验证集进行预处理
X_valid_transformed = best_model.named_steps['preprocessor'].transform(X_valid)

# 创建 SHAP 解释器
explainer = shap.TreeExplainer(best_model.named_steps['model'])

# 计算验证集的 SHAP 值
shap_values = explainer.shap_values(X_valid_transformed)
feature_names = numerical_cols + list(best_model.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols))

# 设置全局字体为我们刚刚加载的字体
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = [font_name]  # 使用字体的名称
rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
print("Matplotlib 全局字体配置已更新。")
feature_translation = {
    'OverTime_No': '无加班',
    'StockOptionLevel_0': '无股票期权',
    'NumCompaniesWorked': '曾就职公司数',
    'MonthlyIncome': '月收入',
    'Age': '年龄',
    'DistanceFromHome': '居住地距离',
    'BusinessTravel_Travel_Frequently': '商务旅行(频繁)',
    'YearsAtCompany': '公司工龄',
    'JobSatisfaction_4': '工作满意度(4级)',
    'EnvironmentSatisfaction_1': '环境满意度(1级)'
}
chinese_feature_names = [feature_translation.get(name, name) for name in feature_names]


# 绘制 SHAP 摘要图（特征重要性）
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_valid_transformed, feature_names=chinese_feature_names, max_display=10, show=False)

plt.title('特征重要性 SHAP 摘要图 (前 10 特征)')
ax = plt.gca() # 获取主坐标轴
ax.set_xlabel('SHAP 值')  # 修改 X 轴标签

cbar = None
for ax in plt.gcf().axes:
    if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == 'Feature value':  # 检测默认的 Colorbar
        cbar = ax
        break

if cbar:
    cbar.set_ylabel('特征值', fontsize=12)  # 修改 Colorbar 主标签为中文
    cbar.set_yticklabels(['低', '高'])  # 修改刻度标签为“低”和“高”
        
plt.savefig('shap_summary_plot_top10.png', dpi=300, bbox_inches='tight')
plt.show()
print("SHAP 摘要图已保存至 'shap_summary_plot_top10.png'")

# 保存 SHAP 值到 DataFrame
shap_importance = pd.DataFrame({
    'Feature': feature_names,
    'SHAP Importance': np.abs(shap_values).mean(axis=0)
}).sort_values(by='SHAP Importance', ascending=False)

# 输出 SHAP 特征重要性到 CSV
shap_importance.to_csv('shap_feature_importance.csv', index=False)
print("SHAP feature importance saved to 'shap_feature_importance.csv'")


# 在验证集上预测
y_pred = grid.predict(X_valid)
y_pred_proba = grid.predict_proba(X_valid)[:, 1]

# 计算评估指标
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, accuracy_score

auc = roc_auc_score(y_valid, y_pred_proba)
aupr = average_precision_score(y_valid, y_pred_proba)
f1 = f1_score(y_valid, y_pred)
mcc = matthews_corrcoef(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)
recall = recall_score(y_valid, y_pred)
accuracy = accuracy_score(y_valid, y_pred)

# 打印结果
print(f"Best parameters: {grid.best_params_}")
print(f"AUC: {auc:.6f}")
print(f"AUPR: {aupr:.6f}")
print(f"F1 Score: {f1:.6f}")
print(f"MCC: {mcc:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"Accuracy: {accuracy:.6f}")

# 将评估指标保存到 DataFrame
metrics_df = pd.DataFrame({
    'Metric': ['AUC', 'AUPR', 'F1 Score', 'MCC', 'Precision', 'Recall', 'Accuracy'],
    'Value': [auc, aupr, f1, mcc, precision, recall, accuracy]
})

# 输出指标到 CSV
metrics_df.to_csv('evaluation_metrics.csv', index=False)
print("Evaluation metrics saved to 'evaluation_metrics.csv'")


# 绘制关键特征的条形图
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# --- 1. 确保字体已加载（如果前面已执行过，这部分可以跳过）---
font_name = "Noto Sans CJK SC"
font_path_local = "./NotoSansCJKsc-Regular.otf"

# 如果字体文件不存在，则下载
if not os.path.exists(font_path_local):
    print(f"正在下载 {font_name} 字体...")
    !wget -q -O {font_path_local} {font_url}
    print("字体下载完成！")

# 添加字体到 Matplotlib
if font_path_local not in [f.fname for f in fm.fontManager.ttflist]:
    fm.fontManager.addfont(font_path_local)
    print(f"字体已添加到 Matplotlib")

# --- 2. 强制设置全局字体配置 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [font_name]
plt.rcParams['axes.unicode_minus'] = False

# 清除 Matplotlib 缓存，确保字体生效
plt.rcParams.update({'font.size': 15})  
print("Matplotlib 全局字体配置已更新")

# 创建独热编码后的特征 DataFrame
X_full_transformed = pd.DataFrame(
    grid.best_estimator_.named_steps['preprocessor'].transform(X),
    columns=numerical_cols + list(grid.best_estimator_.named_steps['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols))
)
X_full_transformed['Attrition'] = y.values

# 选择关键人本特征并翻译为中文
key_features = ['JobSatisfaction_4', 'EnvironmentSatisfaction_1', 'OverTime_No']
feature_translation = {
    'JobSatisfaction_4': '工作满意度4级',
    'EnvironmentSatisfaction_1': '环境满意度1级',
    'OverTime_No': '不加班'
}

# --- 3. 绘制条形图（无网格线，字体放大）---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, feature in enumerate(key_features):
    ax = axes[i]
    
    # 计算均值用于绘图
    data_to_plot = X_full_transformed.groupby('Attrition')[feature].mean()
    
    # 绘制条形图
    bars = ax.bar([0, 1], data_to_plot.values, color=['#89CFF0', '#4682B4'], width=0.6)
    
    # 设置标题和标签（字体大小放大一倍）
    font_prop = fm.FontProperties(fname=font_path_local, size=20)  # 原来是12，现在24
    ax.set_title(feature_translation[feature] + '与离职的关系', fontproperties=font_prop, pad=10)
    
    font_prop_label = fm.FontProperties(fname=font_path_local, size=15)  # 原来是10，现在20
    ax.set_xlabel('离职状态', fontproperties=font_prop_label)
    ax.set_ylabel(feature_translation[feature]+'比例', fontproperties=font_prop_label)
    
    # 设置刻度标签
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['否', '是'], fontproperties=font_prop_label)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '0.5', '1'], fontproperties=font_prop_label)
    ax.set_ylim(0, 1)
    
    # 添加百分比标签
    for index, value in enumerate(data_to_plot.values):
        ax.text(index, value + 0.02, f'{value:.1%}', 
               ha='center', va='bottom', 
               fontproperties=font_prop_label)
    
    # 美化：只保留左边和底边的坐标轴，去掉所有网格线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.grid(False)  # 关闭所有网格线

plt.tight_layout()
plt.savefig('key_features_barplot_optimized.png', dpi=300, bbox_inches='tight')
plt.show()
print("优化后的条形图已保存至 'key_features_barplot_optimized.png'")

