# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


train.head()


train.shape


train.info()


train.duplicated().sum()


train.describe(include = "object")


train['job'].value_counts()


rare_jobs = ["student", "housemaid", "unemployed", "unknown"]
train["job_grouped"] = train["job"].apply(lambda x: "other" if x in rare_jobs else x)
test["job_grouped"] = test["job"].apply(lambda x: "other" if x in rare_jobs else x)
train["job_grouped"].value_counts()


train = train.drop(columns=["job"])
test = test.drop(columns=["job"])


train['marital'].value_counts()


train['education'].value_counts()


train['default'].value_counts()


print(pd.crosstab(train['default'], train['y'], normalize='index'))


train = train.drop(columns=["default"])
test = test.drop(columns=["default"])


train['housing'].value_counts()


train['loan'].value_counts()


train['contact'].value_counts()


train['month'].value_counts()  


train['poutcome'].value_counts()  


train.groupby("poutcome")["y"].mean()


train.describe()


train = train.drop(columns=["id"])


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(train["age"], bins=30, kde=True, color="skyblue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


train[train["age"] > 85].groupby("job_grouped")["y"].count()  


plt.figure(figsize=(10,6))


sns.histplot(train["balance"], bins=100, kde=True, color="teal")

plt.axvline(0, color="red", linestyle="--", label="Zero Balance") 
plt.title("Distribution of Balance")
plt.xlabel("Balance")
plt.ylabel("Count")
plt.legend()
plt.show()


Q1 = train["balance"].quantile(0.25)
Q3 = train["balance"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = train[(train["balance"] < lower_bound) | (train["balance"] > upper_bound)]

print("Number of outliers:", len(outliers))
print("percentage: {:.2f}%".format(len(outliers) / len(train) * 100))



def day_period(day):
    if day <= 15:
        return "early"
    else:
        return "late"


train["day_period"] = train["day"].apply(day_period)
test["day_period"] = test["day"].apply(day_period)
success_rate = train.groupby("day_period")["y"].mean()

print(success_rate)


train["duration_bin"] = pd.cut(train["duration"], bins=range(0, train["duration"].max()+100, 100))

success_rate = train.groupby("duration_bin")["y"].mean()
#print(success_rate)

plt.figure(figsize=(12,6))
sns.lineplot(x=success_rate.index.astype(str), y=success_rate.values, marker="o")
plt.xticks(rotation=90)
plt.xlabel("Call Duration (seconds)")
plt.ylabel("Subscription Rate")
plt.title("Subscription Rate vs Call Duration")
plt.show()


train = train.drop(columns=["duration_bin"])


# train["duration_bin"] = pd.cut(
#     train["duration"],
#     bins=[-1, 120, 600, train["duration"].max()],
#     labels=["short", "medium", "long"]
# )


Q1 = train["campaign"].quantile(0.25)
Q3 = train["campaign"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = train[(train["campaign"] < lower_bound) | (train["campaign"] > upper_bound)]

print("Number of outliers:", len(outliers))
print("percentage: {:.2f}%".format(len(outliers) / len(train) * 100))





plt.figure(figsize=(8,5))
sns.histplot(train["campaign"], bins=30, kde=True, color="skyblue")
plt.title("campaign Distribution")
plt.xlabel("campaign")
plt.ylabel("Count")
plt.show()


train[(train["campaign"] > 15)].count()


train['campaign_capped'] = np.where(train['campaign'] > 15, 15, train['campaign'])
test['campaign_capped'] = np.where(test['campaign'] > 15, 15, test['campaign'])


print(train[['campaign', 'campaign_capped']].describe())


train = train.drop(columns=["campaign"])
test = test.drop(columns=["campaign"])


train['never_contacted'] = np.where(train['pdays'] == -1, 1, 0)
test['never_contacted'] = np.where(test['pdays'] == -1, 1, 0)


train['days_since_last'] = np.where(train['pdays'] == -1, 0, train['pdays'])


plt.figure(figsize=(8,5))
sns.histplot(train["pdays"], bins=30, kde=True, color="skyblue")
plt.title("pdays Distribution")
plt.xlabel("pdays")
plt.ylabel("Count")
plt.show()


subscription_rate = train.groupby('days_since_last')['y'].mean().reset_index()

subscription_rate.rename(columns={'y': 'subscription_rate'}, inplace=True)

print(subscription_rate.head(10))  


train["days_since_last_bin"] = pd.cut(train["days_since_last"], bins=range(0, train["days_since_last"].max()+100, 100))

success_rate = train.groupby("days_since_last_bin")["y"].mean()
#print(success_rate)

plt.figure(figsize=(12,6))
sns.lineplot(x=success_rate.index.astype(str), y=success_rate.values, marker="o")
plt.xticks(rotation=90)
plt.xlabel("days_since_last Call (seconds)")
plt.ylabel("Subscription Rate")
plt.title("Subscription Rate vs days_since_last call")
plt.show()


train = train.drop(columns=["days_since_last","pdays","days_since_last_bin"])
test = test.drop(columns=["pdays"])


train['previous'].hist(bins=50)
plt.xlabel("previous")
plt.ylabel("count")
plt.show()


Q1 = train["previous"].quantile(0.25)
Q3 = train["previous"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = train[(train["previous"] < lower_bound) | (train["previous"] > upper_bound)]

print("Number of outliers:", len(outliers))
print("percentage: {:.2f}%".format(len(outliers) / len(train) * 100))


train[(train["previous"] < lower_bound)].count()


train[ (train["previous"] > 10)].count()


train = train[train["previous"] <= 10]


train.shape


train.head()


train.describe(include = "object")


train = train.drop(columns=["day"])
test = test.drop(columns=["day"])


train.describe()


!pip install category_encoders


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, PowerTransformer
from category_encoders import TargetEncoder  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score  



onehot_cols  = ["marital", "housing", "loan", "contact", "month", "poutcome", "day_period","job_grouped"]
num_cols = ["age", "balance", "previous", "campaign_capped","duration"]
skewed_cols = ["balance", "previous"]
education_order = [["primary", "secondary", "tertiary", "unknown"]]
education_col = ["education"]



ordinal_transformer = OrdinalEncoder(categories=education_order)
onehot_transformer  = OneHotEncoder(drop="first", handle_unknown="ignore")
skewed_transformer  = PowerTransformer(method="yeo-johnson")
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("ord", ordinal_transformer, education_col),
        ("ohe", onehot_transformer, onehot_cols),
        ("skew", skewed_transformer, skewed_cols)
        #("scale", scaler, num_cols),     
    ],
    remainder="passthrough"  
)





from xgboost import XGBClassifier

# neg, pos = np.bincount(y_train)
# scale_pos_weight = neg / pos


xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss",
        #scale_pos_weight=scale_pos_weight
    ))
])

X = train.drop("y", axis=1)
y = train["y"]


xgb_pipeline.fit(X, y)

# y_proba_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]
# roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)
# print("XGBoost ROC AUC:", roc_auc_xgb)



# XGBoost ROC AUC: 0.9653116935645966
# XGBoost ROC AUC: 0.9655983991434511


test.head()


y_proba_xgb = xgb_pipeline.predict_proba(test.drop(columns=["id"]))[:, 1]
submission = pd.DataFrame({
    "id": test["id"],        
    "y": y_proba_xgb
})
submission.to_csv("submission.csv", index=False)



print(submission.head())
print(submission.shape)


import os

os.listdir("/kaggle/working/")

