import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)



print("\nTrain columns and types:")
print(train_df.dtypes)


print("\nMissing values in train data:")
print(train_df.isnull().sum())


print("\nTarget value counts:")
print(train_df['loan_paid_back'].value_counts())


numeric_features = train_df.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

print("Numeric Features:", numeric_features)
print("Categorical features:", categorical_features)


train_df[numeric_features].hist(figsize=(15,10),bins=30)
plt.suptitle("Numeric Feature Distributions", fontsize=16)
plt.show()


train_df['loan_paid_back'].value_counts(normalize=True).plot(
    kind='bar', color=['skyblue', 'salmon'], title='Target Distribution'
)
plt.xlabel('Loan Paid Back (1 = Yes, 0 = No)')
plt.ylabel('Proportion')
plt.show()


sns.countplot(data=train_df, x='gender', hue='loan_paid_back')
plt.title("Loan Payback by Gender")
plt.show()


sns.countplot(data=train_df, x='marital_status', hue='loan_paid_back')
plt.title("Loan Payback by Marital Status")
plt.show()


df = train_df.copy()
test = test_df.copy()


num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()


#from sklearn.preprocessing import LabelEncoder

#encoder = {}
#for col in cat_cols:
#    le = LabelEncoder()
#    df[col] = le.fit_transform(df[col])
#    test[col] = le.transform(test[col])
#    encoder[col] = le


from category_encoders import TargetEncoder

# target
y = df["loan_paid_back"]

# define encoder
te = TargetEncoder(cols=cat_cols)

# fit & transform
df[cat_cols] = te.fit_transform(df[cat_cols], y)
test[cat_cols] = te.transform(test[cat_cols])




df["loan_to_income"] = df["loan_amount"] / (df["annual_income"] + 1)
test["loan_to_income"] = test["loan_amount"] / (test["annual_income"] + 1)


df["credit_to_debt"] = df["credit_score"] / (df["debt_to_income_ratio"] + 1)
test["credit_to_debt"] = test["credit_score"] / (test["debt_to_income_ratio"] + 1)


df["high_interest"] = (df["interest_rate"] > df["interest_rate"].median()).astype(int)
test["high_interest"] = (test["interest_rate"] > df["interest_rate"].median()).astype(int)



df["debt_to_income_ratio"] = df["debt_to_income_ratio"].clip(0, 200)
test["debt_to_income_ratio"] = test["debt_to_income_ratio"].clip(0, 200)

df["credit_score_log"] = np.log1p(df["credit_score"])
test["credit_score_log"] = np.log1p(test["credit_score"])

df["annual_income_log"] = np.log1p(df["annual_income"])
test["annual_income_log"] = np.log1p(test["annual_income"])



from sklearn.model_selection import train_test_split

X = df.drop("loan_paid_back", axis=1)
y = df["loan_paid_back"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(X_train.shape, X_valid.shape)






from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score



model = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.015,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.6,
    gamma=0.02,
    min_child_weight=5,
    eval_metric="auc",
    random_state=42,
    tree_method="hist"
)



model.fit(X,y)



test_pred = model.predict_proba(test)[:, 1]
print(test_pred)


# Predict probabilities for ROC
valid_pred = model.predict_proba(X_valid)[:, 1]

roc = roc_auc_score(y_valid, valid_pred)
print("Validation ROC-AUC:", roc)



submission = pd.DataFrame({
    "id": test_df["id"],
    "loan_paid_back": test_pred
})

submission.to_csv("submission.csv", index=False)

print("submission.csv created!")


