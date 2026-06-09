pip install imblearn


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE


train_df = pd.read_csv("/kaggle/input/playground-series-s3e2/train.csv")



train_df.value_counts()


num_cols = ["age", "avg_glucose_level", "bmi"]


train_df_cat = train_df.drop([*num_cols, "id", "stroke"], axis=1)
train_df_num = train_df[num_cols]
cat_cols = train_df_cat.columns


fig, axes = plt.subplots(1, 7, figsize =(30, 40))

axes = axes.flatten()
for idx, s in enumerate(train_df_cat):
    val_counts = train_df_cat[s].value_counts()
    val_counts.plot.pie(autopct='%1.1f%%', cmap="viridis", ax=axes[idx], title=s, explode = [0.03 for i in range(len(val_counts))])





train_df[["stroke"]].value_counts() / len(train_df)


fig, axes = plt.subplots(4, 2, figsize =(20, 25))

axes = axes.flatten()
for idx, s in enumerate(cat_cols):
    # axes[idx].set_title(s)
    sns.heatmap(pd.crosstab(train_df['stroke'], train_df[s]), annot=True, cmap="viridis", ax=axes[idx])
    



plt.figure(figsize=(20, 5))
sns.heatmap( train_df[[*num_cols, "stroke"]].corr()[["stroke"]].T, annot=True, cmap="viridis")


train_df_num.describe().T


scaler = RobustScaler()
ohe = OneHotEncoder()
sampler = SMOTE()

train_df_cat_trf = ohe.fit_transform(train_df_cat).toarray()
train_df_num_trf = scaler.fit_transform(train_df_num)

X = np.hstack((train_df_cat_trf, train_df_num_trf))
y = train_df[["stroke"]]
X, y = sampler.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# model = LogisticRegression()
model = XGBClassifier()

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"train report: \n{classification_report(y_train, y_pred_train)}\nroc_auc_score: {roc_auc_score(y_train, y_pred_train)}\n\n")
print(f"test report: \n{classification_report(y_test, y_pred_test)}\nroc_auc_score: {roc_auc_score(y_test, y_pred_test)}\n\n")


test_df = pd.read_csv("/kaggle/input/playground-series-s3e2/test.csv")

test_df_cat = test_df.drop([*num_cols, "id"], axis=1)
test_df_num = test_df[num_cols]

test_df_cat_trf = ohe.transform(test_df_cat).toarray()
test_df_num_trf = scaler.transform(test_df_num)

test_X = np.hstack((test_df_cat_trf, test_df_num_trf))

test_pred = model.predict_proba(test_X)

sub_df = pd.DataFrame({
    "id":test_df["id"],
    "stroke": test_pred[:, 1]
})

sub_df.head()


sub_df.to_csv("submission_1.csv", index=False)






















