''' Checking dir '''
!ls /kaggle/input/playground-series-s5e11



''' importing '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold 
import catboost as cb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, roc_curve




train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


train.head()


print(train['education_level'].unique())
print(train['loan_purpose'].unique())


len(train)


print(train['gender'].unique())



train['gender'] = train['gender'].map({"Male": 0, "Female": 1}).fillna(-1).astype(int)
test['gender']  = test['gender'].map({"Male": 0, "Female": 1}).fillna(-1).astype(int)



train.head()


train['marital_status'].unique()


train.info()


cat_cols = train.select_dtypes(include=['object']).columns.tolist()
print("Categorical Columns:", cat_cols)


''' cat-boost'''
X = train.drop("loan_paid_back",axis=1)
y = train['loan_paid_back']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42 ,stratify=y 
)
ctrain_data = Pool(X_train, y_train, cat_features=cat_cols)
val_data   = Pool(X_val, y_val, cat_features=cat_cols)


model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    verbose=100
)


model.fit(ctrain_data, eval_set=val_data, verbose=100)


val_probs = model.predict_proba(X_val)[:, 1]
auc_val = roc_auc_score(y_val, val_probs)
print("Validation AUC:", auc_val)

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, val_probs)
plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("Validation ROC Curve")
plt.legend()
plt.show()


drop_cols = ["loan_paid_back", "predicted_probability"]
X_full = train.drop(columns=[c for c in drop_cols if c in train.columns])
y_full = train["loan_paid_back"]

full_pool = Pool(X_full, y_full, cat_features=cat_cols)
model.fit(full_pool, verbose=100)


test = test[X_full.columns]

test_pool = Pool(test, cat_features=cat_cols)
test_preds = model.predict(test_pool) 


sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
sample_sub["loan_paid_back"] = test_preds
sample_sub.to_csv("submission.csv", index=False)




