
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 假设 df 是你的数据集，包括特征和目标变量
# 假设特征列为 X，目标变量列为 y

# 定义 K-fold 交叉验证
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
df= pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv') 
X=df.drop(['id','day','rainfall'], axis=1).columns.tolist()
y='rainfall'



from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
# 假设 df 是你的数据集，包括特征和目标变量
# 假设特征列为 X，目标变量列为 y
from sklearn.model_selection import train_test_split
# 定义 K-fold 交叉验证
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# 初始化 XGBoost 模型，并设置参数
model = XGBClassifier(
    max_depth=6,
    colsample_bytree=0.9,
    subsample=0.9,
    n_estimators=10000,
    learning_rate=0.1,
    eval_metric="auc",
    early_stopping_rounds=100,
    alpha=1
)

# 执行 K-fold 交叉验证
best=0

for train_index, test_index in kf.split(df):
    X_train, X_test = df.iloc[train_index][X], df.iloc[test_index][X]
    y_train, y_test = df.iloc[train_index][y], df.iloc[test_index][y]
    
    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 训练模型并提供验证集进行early stopping
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # 预测概率
    y_scores = model.predict_proba(X_test)[:, 1]

    # 计算 ROC 曲线的参数
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    if roc_auc>best:
        model.save_model('best_model.json')
        best=roc_auc
    print(roc_auc)
    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    


# 预测
loaded_model = XGBClassifier()
loaded_model.load_model('/kaggle/working/best_model.json')
y_pred = loaded_model.predict(df[X])

# 评估模型
accuracy = accuracy_score(df[y], y_pred)
print(f'Accuracy on test set: {accuracy}')

# 预测概率
y_scores = loaded_model.predict_proba(X_test)[:, 1]

# 计算 ROC 曲线的参数
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()




