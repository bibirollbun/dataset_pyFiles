import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/emsi-5iir-ds/train.csv")


train_df.info()


train_df.describe().T


train_df.T


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def create_features(df):
    df = df.copy()
    df["f_27"] = df[["f_27"]].iloc[:, 0].apply(lambda x: len(set(x)))
    return df
        

def prepare_data(df):
    df = create_features(df)
    classic_attrs = ["f_07", "f_08", "f_09", "f_10", "f_11", "f_12", "f_13", "f_14", "f_15", "f_16", "f_17", "f_18", "f_27", "f_29", "f_30"]
    num_attrs_df = df.drop(classic_attrs, axis=1)
    classic_attrs_df = df[classic_attrs]

    scaler = RobustScaler()
    num_attrs_df_trf = scaler.fit_transform(num_attrs_df)
    X = np.hstack((num_attrs_df_trf,  classic_attrs_df))
    print(X.shape)

    return X

    


X, y =  prepare_data(train_df.drop(["target", "id"], axis=1)), train_df[["target"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


model = LGBMClassifier(boosting_type="gbdt", n_estimators=100, n_jobs=-1, random_state=42, learning_rate=0.2, num_leaves=40)
# model = XGBClassifier(n_estimators=100, n_jobs=-1, random_state=10, num_parallel_tree=3)
model.fit(X_train, y_train)

print(" Reporting ".center(50, "-"))
train_pred = model.predict(X_train)
print(f"train report:\n{classification_report(y_train, train_pred)} \n roc_auc_score: {roc_auc_score(y_train, train_pred)}")

test_pred =  model.predict(X_test)
print(f"\n\ntest report:\n{classification_report(y_test,test_pred)}\n roc_auc_score: {roc_auc_score(y_test, test_pred)}")
print("".center(50, "-"))


train_df["f_00"].mean()


test_df = pd.read_csv("/kaggle/input/emsi-5iir-ds/test.csv")
test_X = prepare_data(test_df.drop(["id"], axis=1))

test_pred = model.predict_proba(test_X)[:, 1]


sub_df = pd.DataFrame({"id":test_df["id"].values.astype(int), "target":test_pred})


sub_df


sub_df.to_csv("submission.csv", index=False)




