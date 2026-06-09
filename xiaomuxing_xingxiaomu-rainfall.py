import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


df_train.head(5)


# 统计每列缺失值数量
print(df_train.isnull().sum())
# 重复行
print(df_train.duplicated().sum())


print(df_train.describe().T[['min', 'max']])


rainfall_ratio = df_train["rainfall"].value_counts(normalize=True)
print(f"Rainfall ratio is: {rainfall_ratio}")


df_train_filtered = df_train[(df_train["sunshine"] >= 0) & (df_train["sunshine"] <= 5)]
rainfall_ratio = df_train_filtered["rainfall"].value_counts(normalize=True)
print(f"Rainfall ratio is: {rainfall_ratio}")


scaler = MinMaxScaler()
cols_to_normalize = ["pressure", "maxtemp", "temparature", "mintemp", 
            "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]

df_train[cols_to_normalize] = scaler.fit_transform(df_train[cols_to_normalize])
df_train_cleaned = df_train


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

features = ["pressure", "maxtemp", "temparature", "mintemp", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed", "rainfall"]
corr_matrix = df_train_cleaned[features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={"shrink": 0.75})
plt.title("Correlation Heatmap")
plt.show()


# 选择特征（排除 id 和目标变量 rainfall）
# features = ["pressure", "maxtemp", "temparature", "mintemp", "dewpoint", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]
features = ["pressure", "maxtemp", "temparature", "dewpoint", "humidity", "cloud", "sunshine", "windspeed"]
features = ["humidity", "cloud", "sunshine"]
X = df_train_cleaned[features]
y = df_train_cleaned["rainfall"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 训练 SVM（启用概率预测）
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
svm_model.fit(X_train, y_train)

# 预测概率
y_prob = svm_model.predict_proba(X_val)[:, 1]  # 取 rain=1 的概率
# y_prob

# 计算 AUC 分数（衡量概率预测的效果）
auc_score = roc_auc_score(y_val, y_prob)
print(f"AUC Score: {auc_score:.4f}")


from xgboost import XGBClassifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)

y_prob = xgb_model.predict_proba(X_val)[:, 1]

auc_score = roc_auc_score(y_val, y_prob)
print(f"AUC Score: {auc_score:.4f}")


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_prob = log_model.predict_proba(X_val)[:, 1]

auc_score = roc_auc_score(y_val, y_prob)
print(f"AUC Score: {auc_score:.4f}")


# ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}


results = {}

# 循环训练每个模型并评估其性能
for model_name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测概率（用于计算 AUC）或者直接预测类别
    if hasattr(model, "predict_proba"):  # 一些模型支持概率预测
        y_prob = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_prob)
    else:  # 其他模型（如 KNN、Decision Tree）预测类别
        y_pred = model.predict(X_test)
        auc_score = roc_auc_score(y_val, y_pred)
    
    # 保存模型结果（AUC 或准确率）
    results[model_name] = auc_score

# 输出结果
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'AUC'])
print(results_df.sort_values(by='AUC', ascending=False))


# 绘制 AUC 比较图
plt.figure(figsize=(10, 6))
results_df.sort_values(by='AUC', ascending=True).plot(kind='barh', x='Model', y='AUC', color='skyblue', legend=False)
plt.title('AUC Score Comparison of Different Models')
plt.xlabel('AUC Score')
plt.ylabel('Model')
plt.show()


bayes_model = GaussianNB()
bayes_model.fit(X_train, y_train)

y_prob = bayes_model.predict_proba(X_val)[:, 1]

auc_score = roc_auc_score(y_val, y_prob)
print(f"AUC Score: {auc_score:.4f}")


df_test[cols_to_normalize] = scaler.fit_transform(df_test[cols_to_normalize])
df_test_cleaned = df_test
# 选择特征（排除 id 和目标变量 rainfall）
# features = ["pressure", "maxtemp", "temparature", "mintemp", "dewpoint", "humidity", "cloud", "sunshine", "windspeed"]
# features = ["pressure", "maxtemp", "temparature", "dewpoint", "humidity", "cloud", "sunshine", "windspeed"]
features = ["humidity", "cloud", "sunshine"]
X = df_test_cleaned[features]
y = bayes_model.predict_proba(X)[:, 1]
    
# Prepare the submission dataframe
submission = pd.DataFrame({
    'id': df_test_cleaned['id'],
    'rainfall': y
})

# Save the submission file
submission.to_csv('submission.csv', index=False)

