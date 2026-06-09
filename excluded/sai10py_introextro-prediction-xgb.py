import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train.info()


train.isnull().sum()


X = train.drop(columns=["Personality"])
y = train["Personality"]


from sklearn.preprocessing import LabelEncoder

encoders = []

for col in X.columns:
    if X[col].dtypes == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders.append(le)
        
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)


from sklearn.impute import KNNImputer

imputer = KNNImputer()
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from xgboost import XGBClassifier

model_xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'  # or 'logloss' for binary
)

model_xgb.fit(X_train, y_train)


from sklearn.metrics import accuracy_score

print(f"Accuracy: {accuracy_score(model_xgb.predict(X_test), y_test)}")


test.isnull().sum()


i = 0
for col in test.columns:
    if test[col].dtypes == "object":
        le = encoders[i]
        test[col] = le.transform(test[col])
        i += 1

test = pd.DataFrame(imputer.transform(test), columns=test.columns)
test = scaler.transform(test)


preds = model_xgb.predict(test)
preds = y_encoder.inverse_transform(preds)

sub = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

sub["Personality"] = preds

sub.to_csv("submission.csv",index=False)
sub.head()

