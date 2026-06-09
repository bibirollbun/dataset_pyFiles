!ls /kaggle/input/playground-series-s5e12/



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train.head()


train.shape


train.columns


categorical_cols = train.select_dtypes(include=['object']).columns


for c in categorical_cols:
    print(c)


train.isnull().sum()


from sklearn.preprocessing import LabelEncoder


for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])


train.head()


X = train.drop("diagnosed_diabetes",axis=1)
y = train["diagnosed_diabetes"]


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


neg, pos = y_train.value_counts()
scale_pos_weight = neg / pos



#v2
model = xgb.XGBClassifier(
    n_estimators=1500,         # more trees
    max_depth=5,               # slightly shallower tree
    learning_rate=0.03,        # smaller learning rate
    subsample=0.85,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=1,                   # requires more gain to split
    reg_alpha=0.5,             # L1 regularization
    reg_lambda=2,              # L2 regularization
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method='hist',        # CPU-safe
    device='cpu',
    eval_metric='auc',
    early_stopping_rounds=50
)




model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)



y_pred_proba = model.predict_proba(X_val)[:,1]
auc = roc_auc_score(y_val, y_pred_proba)
print("Validation AUC:", auc)


xgb.plot_importance(model, max_num_features=20)
plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



for col in categorical_cols:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])



preds_proba = model.predict_proba(test)[:, 1]

submission = pd.DataFrame({
    "id": test["id"],               
    "diagnosed_diabetes": preds_proba  
})
submission.to_csv("S.csv", index=False)




