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


!pip install lifelines


import warnings
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


df_train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv").drop(columns=["ID"], axis=1)
df_test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")


df_train.head().T


df_train.head()


df_test.head()


print("train Data", df_train.info())


df_train.describe().round(2).style.format(precision=2).background_gradient(cmap="Blues")


df_train.isna().sum().reset_index().style.format(precision=2).background_gradient(cmap='Reds')


# 提取所有的"object" 类型的列并获取列名
categorical_columns = df_train.select_dtypes(include="object").columns.tolist()
print(categorical_columns)


plt.figure(figsize=(36,36))
for i ,col in enumerate(categorical_columns ,1):
    plt.subplot(12,3,i)
    sns.countplot(y=col, data=df_train)
    plt.title(col, fontsize=14)

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.tight_layout()
plt.show()


plt.figure(figsize=(38, 38))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(12,3,i)
    sns.boxplot(x="efs_time", y=col, data=df_train)
    plt.title(col, fontsize=14)

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.tight_layout()
plt.show()


# 提取所有的"float和int" 类型的列并获取列名
numerical_columns = df_train.select_dtypes(include=["float64", "int"]).columns.tolist()
print(numerical_columns)



plt.figure(figsize=(36, 36))
sns.set(style="whitegrid")
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(8, 3, i)
    sns.histplot(x=df_train[col], bins=30, kde=True)
    plt.title(col, fontsize=14)

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.tight_layout()
plt.show()   


# 计算相关系数矩阵
plt.figure(figsize=(36, 34))
correlation_matrix = df_train[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='PuBuGn', fmt=".2f", annot_kws={"size":14, "weight":"bold"})

plt.yticks(fontsize=20, fontweight='bold')  
plt.xticks(fontsize=20, fontweight="bold")

plt.title("Correlation Matrix",fontsize=30, fontweight="bold")
plt.show()


from lifelines import KaplanMeierFitter


from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()

T = df_train["efs_time"]
E = df_train["efs"]
kmf.fit(T, event_observed=E)


import numpy as np
from scipy.interpolate import interp1d

survival_times = kmf.survival_function_.index.values
survival_probs = kmf.survival_function_['KM_estimate'].values

interp_func = interp1d(survival_times, survival_probs, kind='cubic')  
new_times = np.linspace(min(survival_times), max(survival_times), 500)
new_probs = interp_func(new_times)

plt.figure(figsize=(10, 6))
plt.plot(new_times, new_probs, label='Smoothed KM estimate')
plt.grid(True, linestyle='--', alpha=0.6)
plt.title('Survival function of political regimes (Smoothed)')
plt.legend()
plt.show()



df_train['km_label'] = kmf.survival_function_at_times(df_train['efs_time']).values
df_train.loc[df_train['efs'] == 0, 'km_label'] -= 0.1


import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# 创建子图
fig, ax = plt.subplots(figsize=(10, 6))

# 定义种族类别
races = df_train["race_group"].unique()  # 

kmf = KaplanMeierFitter()

# 遍历种族类别，并分别绘制生存曲线
for race in races:
    mask = df_train["race_group"] == race
    kmf.fit(T[mask], event_observed=E[mask], label=race)
    kmf.plot_survival_function(ax=ax)

# 添加标题和图例
plt.title("Kaplan-Meier Survival Function by Race")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.legend(title="Race")
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()


df_train[categorical_columns] = df_train[categorical_columns].astype('category')
df_test[categorical_columns] = df_test[categorical_columns].astype('category')



X = df_train.iloc[:,:-3]
y = df_train.iloc[:,-1]


X.head()


from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from bayes_opt import BayesianOptimization

# 初始化XGBoost回归器
xgb_reg = XGBRegressor(
    objective='reg:squarederror', 
    eval_metric='rmse',
    enable_categorical=True  # 启用处理类别变量（需要支持的版本
)


# 五折交叉验证（回归任务使用KFold即可）
kf = KFold(n_splits=5, shuffle=True, random_state=666)

# 对原始数据使用交叉验证并生成预测结果
y_pred_original = cross_val_predict(xgb_reg, X, y, cv=kf)

# 输出回归报告（这里采用R²和均方误差）
print("Original Dataset Regression Report")
print("R2 Score:", r2_score(y, y_pred_original))
print("Mean Squared Error:", mean_squared_error(y, y_pred_original))


def xgb_eval(n_estimators, learning_rate, max_depth, colsample_bytree):
    params = {
        "n_estimators": int(round(n_estimators)),
        "learning_rate": learning_rate,
        "max_depth": int(round(max_depth)),
        "colsample_bytree": colsample_bytree,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "enable_categorical": True  # 启用类别数据支持（实验性参数）
    }
    # 使用KFold进行交叉验证，并采用R²作为评价指标
    cv_result = cross_val_score(
        XGBRegressor(**params),
        X,
        y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="r2"
    ).mean()
    return cv_result

# 贝叶斯优化
xgb_bo = BayesianOptimization(
    f=xgb_eval,
    pbounds={
        "n_estimators": (50, 300),
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 10),
        "colsample_bytree": (0.5, 1.0)
    },
    random_state=42
)

# 优化过程：先初始化5个随机点，再进行30次迭代
xgb_bo.maximize(init_points=5, n_iter=30)

# 输出最优参数
best_params = xgb_bo.max["params"]

print("Best XGBoost Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value:.4f}")



xgb_tuned = XGBRegressor(
    n_estimators=int(round(168.0585)), 
    learning_rate=0.1900,
    max_depth=int(round(3.0753)),
    colsample_bytree=0.8761,
    objective="reg:squarederror",
    eval_metric="rmse",
    enable_categorical=True
)

xgb_tuned.fit(X, y)


from lifelines.utils import concordance_index

# 预测分数（风险分数或生存时间）
y_pred = xgb_tuned.predict(X)

# 计算 C-index
c_index = concordance_index(y, y_pred)
print("C-index:", c_index)



from xgboost import plot_importance

fig, ax = plt.subplots(1, 1, figsize=(24, 22))
ax = plot_importance(
    xgb_tuned,
    show_values=False,  # 不显示数值
    title="Feature importance | XGBoost Model",
    ax=ax,
    xlabel="",
    height=0.7,  
    color="#1f77b4",  # 蓝色
)

# 在条形图上显示数值
ax.bar_label(ax.containers[0], fmt="{:,.01f}", fontsize=8)
ax.grid(False)
plt.show()



prediction = xgb_tuned.predict(df_test.drop(columns=["ID"]))


submission = pd.DataFrame({'id': df_test['ID'], 'prediction': prediction})

# 创建输出文件
submission.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")



submission




