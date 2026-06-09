'''
My 2nd kaggle competition of Binary Classification with a Bank Dataset

model choice :

LIGHTGBM
i) a large structured dataset with categorical + numerical mix

ii) Likely imbalanced binary target (marketing campaign “yes/no”)

Need a model that is:

Fast to train on 750k rows

High performing (non-linear interactions matter)

Handles imbalance

~ ssmjtc
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import classification_report
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


tr=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
te=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


print("train shape",tr.shape)
print("test shape",te.shape)
print(tr.columns)
print(tr.isnull().sum())

#checking data type of columns
li=[[x,type(x)] for x in tr.columns]
print(li)


print(tr.head(4))


#most of the data is in string form we have to enumerate the dataset
# convert numerical columns
num_cols = ['age','balance','day','duration','campaign','pdays','previous']
tr[num_cols] = tr[num_cols].astype(int)
te[num_cols]=te[num_cols].astype(int)
cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']

#uniques ..used to one hot encode
li=[[x,tr[x].unique()] for x in cat_cols]
print(li)


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# categorical columns + categories
cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']

categories = [
    ['technician','blue-collar','student','admin.','management',
     'entrepreneur','self-employed','unknown','services','retired',
     'housemaid','unemployed'],   # job
    ['married','single','divorced'],  # marital
    ['secondary','primary','tertiary','unknown'],  # education
    ['no','yes'],   # default
    ['no','yes'],   # housing
    ['no','yes'],   # loan
    ['cellular','unknown','telephone'],   # contact
    ['aug','jun','may','feb','apr','nov','jul','jan','oct','mar','sep','dec'],  # month
    ['unknown','other','failure','success']  # poutcome
]

# setup encoder (⚡ add sparse=False)
encoder = OneHotEncoder(categories=categories, sparse_output=False, handle_unknown="ignore")

# fit/transform
encoded_train = encoder.fit_transform(tr[cat_cols])
encoded_test  = encoder.transform(te[cat_cols])

# column names after encoding
encoded_cols = encoder.get_feature_names_out(cat_cols)

# build DataFrames (now correct shape: 750000 x 44)
train_encoded = pd.DataFrame(encoded_train, columns=encoded_cols, index=tr.index)
test_encoded  = pd.DataFrame(encoded_test,  columns=encoded_cols, index=te.index)

# merge with numerical + target
num_cols = ['id','age','balance','day','duration','campaign','pdays','previous']
train_final = pd.concat([tr[num_cols].astype(int), train_encoded, tr['y'].astype(int)], axis=1)
test_final  = pd.concat([te[num_cols].astype(int), test_encoded], axis=1)

print(train_encoded.shape)  # (750000, 44)
print(test_encoded.shape)   # (250000, 44)
print(train_final.shape)    # (750000, 53) → 8 numeric + 44 dummies + 1 target
print(test_final.shape)     # (250000, 52) → 8 numeric + 44 dummies



X = train_final.drop(columns=["y"])
y = train_final["y"]

# Train/Validation split (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = lgb.LGBMClassifier(
    objective="binary",
    learning_rate=0.05,
    num_leaves=64,
    n_estimators=1000,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=100)]
)

y_pred = clf.predict(X_val)
y_pred_proba = clf.predict_proba(X_val)[:, 1]
y_pred = (y_pred_proba >= 0.77).astype(int)

print("Evaluation Metrics")
print(f"Accuracy  : {accuracy_score(y_val, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_val, y_pred_proba):.4f}")
print(f"Precision : {precision_score(y_val, y_pred):.4f}")
print(f"Recall    : {recall_score(y_val, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_val, y_pred):.4f}")
print(classification_report(y_val, y_pred))


y_pred = clf.predict_proba(test_final)[:, 1]

# ----------------------------
# 6. Create submissions.csv
# ----------------------------
submission = pd.DataFrame({
    "id": test_final["id"],
    "y": y_pred
})

submission.to_csv("submission.csv", index=False)

