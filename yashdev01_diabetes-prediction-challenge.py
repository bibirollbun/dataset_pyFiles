import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train.info()


train.describe()


train.isna().sum()


sns.heatmap(train.select_dtypes(include=['int', 'float']).corr(), cmap='coolwarm')


target = "diagnosed_diabetes"
X = train.drop(columns=[target, 'id'])
test = test.drop(columns=['id'])
y = train[target]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


cat_cols = X_train.select_dtypes(exclude=['int', 'float']).columns
num_cols = X_train.select_dtypes(include=['int', 'float']).columns

print(cat_cols)
print(num_cols)


from sklearn.preprocessing import LabelEncoder, StandardScaler


encoders = {}

for col in cat_cols:
    label_encoder = LabelEncoder()
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])
    test[col] = label_encoder.transform(test[col])
    encoders[col] = label_encoder


encoders


scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
test[num_cols] = scaler.transform(test[num_cols])


from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=2000,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42
)

model.fit(X_train, y_train)


preds = model.predict(X_test)


final_pred = model.predict(test)


final_pred


from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


probab = model.predict_proba(X_test)[:,1]


print("Accuracy :", accuracy_score(y_test, preds))
print("F1 Score :", f1_score(y_test, preds))
print("ROC-AUC  :", roc_auc_score(y_test, probab))
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))


test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": final_pred
})

submission.to_csv("submission.csv", index=False)


submission




