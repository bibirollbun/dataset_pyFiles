import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
#特征处理相关库
from sklearn.preprocessing import PowerTransformer, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#极限梯度模型
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


train_df=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


train_df.head()


train_df.isnull().sum()


test_df.isnull().sum()


train_df.info()


train_copy=train_df.drop(['id'],axis=1)


#箱型图查看数据分布情况
def plot_boxplot_outliers(df, figsize=(15, 10)):
    """
    绘制箱型图检测异常值
    """
    # 选择数值型列
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 创建子图
    n_cols = 3
    n_rows = (len(numeric_df.columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]  # 统一处理一维数组
    
    # 为每个数值列绘制箱型图
    for i, col in enumerate(numeric_df.columns):
        numeric_df[col].plot.box(ax=axes[i])
        axes[i].set_title(f'{col} ')
        axes[i].set_ylabel('number')
    
    
    plt.tight_layout()
    plt.show()
    
    # # 统计异常值
    outlier_summary = detect_outliers(numeric_df)
    return outlier_summary

def detect_outliers(df):
    """
    检测并统计异常值
    """
    outlier_info = {}
    
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_info[col] = {
            '异常值数量': len(outliers),
            '异常值比例': f"{len(outliers)/len(df)*100:.2f}%",
            '异常值范围': f"[{outliers.min():.2f}, {outliers.max():.2f}]" if len(outliers) > 0 else "无"
        }
    
    return pd.DataFrame(outlier_info).T

outlier_summary =plot_boxplot_outliers(train_copy)
print(outlier_summary)


#数值类型相关性热力图
def plot_correlation_heatmap(df, figsize=(10, 8)):
    """
    绘制相关性热力图
    """
    # 选择数值型列
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 计算相关系数矩阵
    corr_matrix = numeric_df.corr()
    
    # 绘制热力图
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, 
                annot=True,      # 显示数值
                fmt=".2f",       # 数值格式
                cmap='coolwarm', # 颜色方案
                center=0,        # 颜色中心点
                square=True)     # 正方形单元格
    
    plt.title('heatmap')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix
corr_matrix=plot_correlation_heatmap(train_df)


#各特征的直方图条形分布趋势
numeric_cols = train_df.drop('id', axis = 1).select_dtypes(include=['int64', 'float64']).columns
n_cols = 4  
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
bins = 30  

colors = cm.tab20(np.linspace(0, 1, len(numeric_cols)))

plt.figure(figsize=(n_cols * 4, n_rows * 3))

for i, (col, color) in enumerate(zip(numeric_cols, colors), 1):
    plt.subplot(n_rows, n_cols, i)
    plt.hist(train_df[col], bins=bins, edgecolor='black', alpha=0.7, color=color)
    plt.title(col)
    plt.tight_layout()

plt.show()


#数据特征标准化
X=train_df.drop(['BeatsPerMinute'],axis=1).iloc[:,1:]
test=test_df.drop('id',axis=1)
Y=train_df['BeatsPerMinute']
# 偏斜特征列（可能分布不均匀的特征）
skewed_cols = ["VocalContent", "AcousticQuality", "InstrumentalScore", "LivePerformanceLikelihood"]
# 需要标准化的特征列保证标准化正态分布
standardize_cols = ["AudioLoudness"]
preprocessor=ColumnTransformer(
    transformers=[
        ('skewed',PowerTransformer(method='yeo-johnson'),skewed_cols),
        ('standardize',StandardScaler(),standardize_cols)
    ],
    remainder='passthrough'
)

X_transformer=preprocessor.fit_transform(X)
X_test=preprocessor.transform(test)

feature=preprocessor.get_feature_names_out()

train_std=pd.DataFrame(X_transformer,columns=feature,index=X.index)
test_std=pd.DataFrame(X_test,columns=feature,index=test.index)


train_std.head()


test_std.head()


#模型训练
kf=KFold(n_splits=5,shuffle=True,random_state=42)
oof_preds={'cat':np.zeros(len(train_std))} #用于交叉验证训练集的预测收集
test_preds={'cat':np.zeros((len(test_std),kf.get_n_splits()))} #用于测试集的训练数据收集

scores = {"cat": []} #模型评估分数


for fold, (train_idx, val_idx) in enumerate(kf.split(train_std,Y),1):
    # 1. 划分训练集和验证集
    X_train, X_val = train_std.iloc[train_idx], train_std.iloc[val_idx]
    y_train, y_val = Y.iloc[train_idx], Y.iloc[val_idx]

    model = CatBoostRegressor(
        iterations = 3000,
        learning_rate = 0.05,
        l2_leaf_reg = 10,
        grow_policy = 'SymmetricTree', 
        random_strength=2,
        loss_function = 'RMSE', 
        eval_metric='RMSE',
        random_seed = 34,
        task_type="GPU",
        depth = 8,
        verbose = 0
    )
    # 2. 训练模型（以"cat"模型为例）
    model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),  # 添加验证集
    use_best_model=True  # 使用验证集上表现最好的模型
    )
    
    # 3. 预测并更新oof_preds（存储验证集预测）
    oof_preds["cat"][val_idx] = model.predict(X_val)
    
    # 4. 预测并更新test_preds（存储测试集预测）
    test_preds["cat"][:, fold-1] = model.predict(X_test) #标准化后的数据集
    
    # 5. 计算分数并更新scores
    score = mean_squared_error(y_val, oof_preds["cat"][val_idx])
    scores["cat"].append(score)


print(score)


final_pred={'cat':test_preds['cat'].mean(axis=1)}

submission = pd.DataFrame({
    "id": test_df["id"],
    "BeatsPerMinute": final_pred["cat"] 
})

submission.to_csv("submission.csv", index=False)




