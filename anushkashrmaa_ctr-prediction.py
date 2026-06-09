
import numpy as np
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import os
print(os.listdir("/kaggle/input"))



import os


print("Files inside avazu-ctr-prediction:")
print(os.listdir("/kaggle/input/avazu-ctr-prediction"))



import pandas as pd


df = pd.read_csv("/kaggle/input/avazu-ctr-prediction/train.gz", nrows=100000)


df.head()



from sklearn.preprocessing import LabelEncoder


y = df["click"]
X = df.drop(columns=["id", "click"])


for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))



from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



import lightgbm as lgb


model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)


model.fit(X_train, y_train)



from sklearn.metrics import roc_auc_score, log_loss


y_pred = model.predict_proba(X_test)[:, 1]


auc = roc_auc_score(y_test, y_pred)
loss = log_loss(y_test, y_pred)

print(f"AUC Score: {auc:.4f}")
print(f"Log Loss: {loss:.4f}")



import matplotlib.pyplot as plt
lgb.plot_importance(model, max_num_features=10)
plt.title("Top 10 Feature Importances")
plt.show()





