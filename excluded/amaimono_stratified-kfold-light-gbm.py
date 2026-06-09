import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")


train_path = "/kaggle/input/playground-series-s5e7/train.csv"
test_path = "/kaggle/input/playground-series-s5e7/test.csv"
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)


df_train.head()


df_train.info()


missing_values = df_train.isnull().sum()
missing_percent = (missing_values / len(df_train) * 100)
missing_df = pd.DataFrame({"Missing_Values": missing_values, "Percentege": missing_percent})
missing_df = missing_df[missing_df["Missing_Values"] > 0]
missing_df


df_train.describe()


plt.figure(figsize=(10,6))
sns.countplot(data=df_train, x="Personality")
plt.title("Distribution of Personality Types")
plt.xlabel("Personality")
plt.ylabel("Count")
plt.show()

print("Personality Value Counts (Proportions):")
print(df_train["Personality"].value_counts(normalize=True))


num_cols_0 = ["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]
for col in num_cols_0:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_train[col])
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

    print(f"Descriptive Stats for {col}")
    print(df_train[col].describe())


num_cols_1 = ["Stage_fear", "Drained_after_socializing"]
for col in num_cols_1:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_train, x=col)
    plt.title(f"count{col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()
    print(f"Descriptive Stats for {col}")
    print(df_train[col].value_counts(normalize=True))


for col in df_train.select_dtypes(include=float).columns:
    df_train.fillna({col:df_train[col].median()}, inplace=True)

for col in df_train.select_dtypes(include=object).columns:
    print( col)
    df_train.fillna({col:df_train[col].mode()[0]}, inplace=True)


for col in df_test.select_dtypes(include=float).columns:
    df_test.fillna({col:df_test[col].median()}, inplace=True)
    
for col in df_test.select_dtypes(include=object).columns:
    print( col)
    df_test.fillna({col:df_test[col].mode()[0]}, inplace=True)


df_train['Stage_fear'] = df_train['Stage_fear'].map({'Yes': 1, 'No': 0})
df_train['Drained_after_socializing'] = df_train['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
le = LabelEncoder()
df_train["Personality"] = le.fit_transform(df_train["Personality"])


X_test = df_test.copy()
X_test['Stage_fear'] = df_test['Stage_fear'].map({'Yes': 1, 'No': 0})
X_test['Drained_after_socializing'] = df_test['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
X_test = X_test.drop(columns='id')


X_train = df_train.drop(columns=["Personality","id"])
y_train = df_train[["Personality"]]


list_nfold=[0,1,2,3,4]
cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(X_train, y_train))
for nfold in list_nfold:
    print("-"*20, nfold, "-"*20)
    idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
    x_tr, y_tr = X_train.iloc[idx_tr], y_train.iloc[idx_tr]
    x_va, y_va = X_train.iloc[idx_va], y_train.iloc[idx_va]
    print(x_tr.shape, x_va.shape)

    model = lgb.LGBMClassifier()
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_tr, y_tr),(x_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True),lgb.log_evaluation(period=10)])
    y_tr_pred = model.predict(x_tr)
    y_va_pred = model.predict(x_va)
    metric_tr = roc_auc_score(y_tr, y_tr_pred)
    metric_va = roc_auc_score(y_va, y_va_pred)
    print(f"[auc] tr:{metric_tr},va:{metric_va}")


predictions = model.predict(X_test)

predictions = le.inverse_transform(predictions)

submission = pd.DataFrame({
    'id': df_test['id'],
    'Personality': predictions
})

submission.to_csv('submission.csv', index=False)

