import dask.dataframe as dd
df = dd.read_csv("/kaggle/input/microsoft-malware-prediction/train.csv",
                 dtype="object",  # everything loaded as string
                 assume_missing=True)


df


sample_df = df.sample(frac=0.168, random_state=42).compute()


sample_df.shape


sample_df.to_csv("trainsample.csv", index=False)


import pandas as pd
train = pd.read_csv("trainsample.csv")


train.shape


train = pd.read_csv("trainsample.csv")


train.head()


train.columns


columnstodrop = ['AutoSampleOptIn',
'Census_InternalBatteryNumberOfCharges',
'Census_InternalBatteryType',
'Census_IsFlightingInternal',
'Census_IsFlightsDisabled',
'Census_IsWIMBootEnabled',
'Census_ProcessorClass',
'Census_ThresholdOptIn',
'DefaultBrowsersIdentifier',
'IsBeta',
'ProductName',
'PuaMode',
'UacLuaenable']


train = train.drop(columns=columnstodrop)


train.shape


train.dtypes.value_counts()


train.info()


null_2 = (train.isnull().sum() / len(train)) * 100
null_2[null_2 > 2].sort_values(ascending=False)


train['HasDetections'].value_counts(normalize=True)


train.duplicated().sum()


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=train["HasDetections"])
plt.title("Distribution of Target Variable (HasDetections)")
plt.show()


sample_corr = train.sample(1000000, random_state=42)

corr_matrix = sample_corr.corr(numeric_only=True)

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.title("Correlation Heatmap (Sample of 1 Million Rows)")
plt.show()



from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


target = "HasDetections"

X = train.drop(columns=[target])
y = train[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)



cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Replace NaN in categorical columns
for col in cat_cols:
    X_train[col] = X_train[col].fillna("missing")
    X_val[col] = X_val[col].fillna("missing")



model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    early_stopping_rounds=50,
    task_type='GPU'
)


model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_cols
)



preds = model.predict(X_val)
pred_probs = model.predict_proba(X_val)[:, 1]

print("Accuracy:", accuracy_score(y_val, preds))
print("AUC:", roc_auc_score(y_val, pred_probs))
print("\nClassification Report:\n", classification_report(y_val, preds))



model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    early_stopping_rounds=50,
    task_type='GPU'
)


model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_cols
)



preds = model.predict(X_val)
pred_probs = model.predict_proba(X_val)[:, 1]

print("Accuracy:", accuracy_score(y_val, preds))
print("AUC:", roc_auc_score(y_val, pred_probs))
print("\nClassification Report:\n", classification_report(y_val, preds))

