import pandas as pd

# 没有表头，所以 header=None
X_train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header=None)
y_train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header=None).values.ravel()
X_test = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv', header=None)
X_train



import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# 1. 读取数据
X_train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header=None)
y_train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header=None).values.ravel()
X_test = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv', header=None)

# 2. 定义多个模型（默认参数）
model1 = LogisticRegression(max_iter=1000)
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = lgb.LGBMClassifier()
model5 = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model6 = MLPClassifier(max_iter=500)

# 3. Voting 集成器（Soft voting）
ensemble_model = VotingClassifier(
    estimators=[
        ('lr', model1),
        ('rf', model2),
        ('gb', model3),
        ('lgb', model4),
        ('xgb', model5),
        ('mlp', model6),
    ],
    voting='soft',  # 概率投票
    n_jobs=-1
)

# 4. 可选：交叉验证看效果
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in skf.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    ensemble_model.fit(X_tr, y_tr)
    val_pred = ensemble_model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    scores.append(acc)
print(f"Ensemble CV Accuracy: {sum(scores)/len(scores):.4f}")

# 5. 用全部数据训练最终模型
ensemble_model.fit(X_train, y_train)

# 6. 预测测试集
test_pred = ensemble_model.predict(X_test)

# 7. 写入提交文件
submission = pd.DataFrame(test_pred, columns=["Solution"])
submission.index += 1  # Id 从 1 开始
submission.index.name = "Id"
submission.to_csv("submission.csv")





