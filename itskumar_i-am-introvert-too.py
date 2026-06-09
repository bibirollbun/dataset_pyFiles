import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline


# Import the dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col="id")


train.shape, test.shape


train.head()


test.head()


train.describe()


train.hist(figsize=(10, 8));


train["Stage_fear"].value_counts()


train["Drained_after_socializing"].value_counts()


train["Personality"].value_counts()


train.isnull().sum()


num_cols = [c for c in train.select_dtypes("number").columns]
cat_cols = [c for c in train.select_dtypes("object").columns if c !="Personality"]


X, y = train.drop(columns=["Personality"]), train["Personality"]


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=720)


X_train.shape, y_train.shape


X_val.shape, y_val.shape


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

binary_map = {"No": 0, "Yes": 1}
imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

for col in cat_cols:
    most_freq = X_train[col].mode()[0]
    X_train.loc[:, col] = X_train[col].fillna(most_freq)
    X_train.loc[:, col] = X_train[col].map(binary_map).astype("int8")

X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])


for col in cat_cols:
    most_freq = X_val[col].mode()[0]
    X_val.loc[:, col] = X_val[col].fillna(most_freq)
    X_val.loc[:, col] = X_val[col].map(binary_map).astype("int8")

X_val[num_cols] = imputer.transform(X_val[num_cols])
X_val.loc[:, num_cols] = scaler.transform(X_val[num_cols])


X_train[cat_cols] = X_train[cat_cols].astype("int8")
X_val[cat_cols] = X_val[cat_cols].astype("int8")
X_val.dtypes


target_map = {"Extrovert": 0, "Introvert": 1}
y_train = y_train.map(target_map)
y_val = y_val.map(target_map)


y_train.value_counts()


from imblearn.over_sampling import SMOTE


smote = SMOTE(random_state=727)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

y_resampled.value_counts()


# Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

lr_model = LogisticRegression(random_state=720)
lr_model.fit(X_resampled, y_resampled)
preds = lr_model.predict(X_val)

acc = accuracy_score(y_val, preds)
prec = precision_score(y_val, preds)
rec = recall_score(y_val, preds)

print(f"   Accuracy : {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall   : {rec:.4f}")


from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=720)
dt_model.fit(X_resampled, y_resampled)
preds = dt_model.predict(X_val)

acc = accuracy_score(y_val, preds)
prec = precision_score(y_val, preds)
rec = recall_score(y_val, preds)

print(f"   Accuracy : {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall   : {rec:.4f}")


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=720)
rf_model.fit(X_resampled, y_resampled)
preds = rf_model.predict(X_val)

acc = accuracy_score(y_val, preds)
prec = precision_score(y_val, preds)
rec = recall_score(y_val, preds)

print(f"   Accuracy : {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall   : {rec:.4f}")


from lightgbm import LGBMClassifier

lgb_model = LGBMClassifier(random_state=720)
lgb_model.fit(X_resampled, y_resampled)
preds = lgb_model.predict(X_val)

acc = accuracy_score(y_val, preds)
prec = precision_score(y_val, preds)
rec = recall_score(y_val, preds)

print(f"   Accuracy : {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall   : {rec:.4f}")


from xgboost import XGBClassifier

xgb_model = XGBClassifier(random_state=720)
xgb_model.fit(X_resampled, y_resampled)
preds = xgb_model.predict(X_val)

acc = accuracy_score(y_val, preds)
prec = precision_score(y_val, preds)
rec = recall_score(y_val, preds)

print(f"   Accuracy : {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall   : {rec:.4f}")


test.isnull().sum()


# predictions

for col in cat_cols:
    most_freq = test[col].mode()[0]
    test.loc[:, col] = test[col].fillna(most_freq)
    test.loc[:, col] = test[col].map(binary_map).astype("int8")

test[cat_cols] = test[cat_cols].astype("int8")
test[num_cols] = imputer.transform(test[num_cols])
test.loc[:, num_cols] = scaler.transform(test[num_cols])

predictions = lgb_model.predict(test)


submission = pd.DataFrame({"id": test.index,
                           "Personality": predictions})
submission['Personality'] = submission['Personality'].map({0: "Extrovert", 1: "Introvert"})
submission.to_csv("submission.csv", index=False)




