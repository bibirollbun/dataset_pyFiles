import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#Loading the data
train_data= pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_data= pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample_data= pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


#Let's see the first 5 rows for train data.
train_data.head()


#Let's see the first 5 rows for test data.
test_data.head()


#Let's see the first 5 rows for Sample data.
sample_data.head()


#Checking the data info. 
train_data.info()


#Checking the dataset dimension
print(f"Rows and Columns in the train dataset:", train_data.shape)
print(f"Rows and Columns in the test dataset:", test_data.shape)


#Cheking the columns names.
train_data.columns


#Removing 'id columns'.
train_data.drop(["id"], axis=1, inplace=True)


#Checking null values in the dataset.
train_data.isnull().sum().sum()


#Checking the summary of dataset.
train_data.describe().round()


train_data["loan_paid_back"].value_counts()


#changing the data-type of the train dataset column name "loan_paid_back".
train_data["loan_paid_back"]=train_data["loan_paid_back"].astype("int")
train_data["loan_paid_back"][0]


#Rounding off the annual income because into roundoff.
train_data["annual_income"]=train_data["annual_income"].round()
train_data["annual_income"][1]


train_data.head()


#Sprating the numerical and categorical columns
num_cols= ["annual_income","debt_to_income_ratio","credit_score","loan_amount","interest_rate"]
cat_cols= ["gender","marital_status","education_level","employment_status","loan_purpose","grade_subgrade"]


#Checking the loan paid back count
plt.figure(figsize=(7,5))
sns.countplot(x="loan_paid_back", data=train_data, hue="gender")
plt.title("Loan Paid Back Accordingly Gender")
plt.xlabel("Loan Paid Back")
plt.ylabel("Count")
plt.show()


#Checking the loan paid back accordingly loan purpose.
plt.figure(figsize=(10,8))
sns.barplot(data=train_data, x="loan_paid_back", y="loan_amount", hue="loan_purpose", estimator="mean")
plt.title("Loan Paid Back Accordingly Loan Amount", fontweight="bold")
plt.xlabel("Loan Paid Back")
plt.ylabel("Loan Amount")
plt.show()


plt.figure(figsize=(8,6))
sns.scatterplot(x="annual_income", y="loan_amount", data=train_data)
plt.title("Annual Income Vs Loan Amount", size=18)
plt.show()


for col in cat_cols:
    plt.figure(figsize=(8,5))
    sns.countplot(x=col, data=train_data)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()


for col in num_cols:
    plt.figure(figsize=(8,5))
    sns.histplot(train_data[col], bins=30, kde=True, palette="Paired")
    plt.title(f"Measurment of {col}")
    plt.xlabel(f"{col}")
    plt.ylabel("Frequency")
    plt.tight_layout()


#Creating function for checking the outlier using boxplot.
def boxplot(data):
    plt.figure(figsize=(8,6))
    sns.boxplot(x=data)
    plt.title("Checking the outliers")
    plt.show()


boxplot(train_data["loan_amount"])


boxplot(train_data["interest_rate"])


boxplot(train_data["credit_score"])


boxplot(train_data["annual_income"])


boxplot(train_data["loan_paid_back"])


# Copying the train data.
raw= train_data.copy()

train_cat= raw.copy()
train_xgb= raw.copy()


from catboost import CatBoostClassifier

train_cat["new_education_level"]= train_cat["education_level"].astype('category').cat.codes
train_cat["new_grade_subgrade"]= train_cat["grade_subgrade"].astype('category').cat.codes

mean_target= train_cat.groupby("loan_purpose")["loan_paid_back"].mean()
train_cat["new_loan_purpose"]= train_cat["loan_purpose"].map(mean_target)

train_cat= pd.get_dummies(train_cat, columns=["gender", "marital_status", "employment_status"], drop_first=False)

cols_to_drop= ["education_level",  "grade_subgrade", "loan_purpose"]
train_cat.drop(columns=cols_to_drop, inplace=True)


#Spliting for train_cat

X_cat= train_cat.drop(["loan_paid_back"], axis=1)
y_cat= train_cat["loan_paid_back"]


train_xgb.head()


train_xgb["annual_income"]= np.log1p(train_xgb["annual_income"])
train_xgb["debt_to_income_ratio"]= np.log1p(train_xgb["debt_to_income_ratio"])

train_xgb= pd.get_dummies(train_xgb, columns=["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"], drop_first=False)

X_xgb= train_xgb.drop("loan_paid_back", axis=1)
y_agb= train_xgb["loan_paid_back"]


#Copyping for test data.

test_cat= test_data.copy()
test_xgb= test_data.copy()


test_cat["new_education_level"]= test_cat["education_level"].astype('category').cat.codes
test_cat["new_grade_subgrade"]= test_cat["grade_subgrade"].astype('category').cat.codes

test_cat['new_loan_purpose'] = test_cat['loan_purpose'].map(mean_target)

test_cat= pd.get_dummies(test_cat, columns=["gender", "marital_status", "employment_status"], drop_first=False)

missing_cols= set(X_cat.columns) - set(test_cat.columns)
for col in missing_cols:
    test_cat[col]= 0

test_cat= test_cat[X_cat.columns]
test_cat.head()


test_xgb["annual_income"]= np.log1p(test_xgb["annual_income"])
test_xgb["debt_to_income_ratio"]= np.log1p(test_xgb["debt_to_income_ratio"])

test_xgb= pd.get_dummies(test_xgb, columns=['gender','marital_status','education_level','employment_status','loan_purpose','grade_subgrade'],drop_first=False)

missing_cols= set(X_xgb.columns) - set(test_xgb.columns)
for col in missing_cols:
    test_xgb[col]= 0

test_xgb= test_xgb[X_xgb.columns]
test_xgb.head()


#Catboost algorithum 

from catboost import CatBoostClassifier

cat_model= CatBoostClassifier(
    iterations= 1500,
    learning_rate= 0.03,
    depth= 6,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=False,
    random_state=42)

cat_model.fit(X_cat, y_cat)


#Prediction for CatBoost

cat_pred= cat_model.predict_proba(X_cat)[:, 1]
print(cat_pred)


# XGBoostClassifier algorithum

from xgboost import XGBClassifier 

xgb_model= XGBClassifier(
    n_estimators= 1200,
    learning_rate= 0.05,
    max_depth= 6,
    subsample= 0.8,
    colsample_bytree= 0.8,
    tree_methods="hist",
    eval_metric="auc",
    random_status=42,
    n_jobs=1)

xgb_model.fit(X_xgb, y_agb)


#Prediction for XGBClassier

xgb_pred= xgb_model.predict_proba(X_xgb)[:,1]
print(xgb_pred)


#Bulding meta dataset for the logistic regression model.

meta_X= pd.DataFrame({
    "cat_pred": cat_pred,
    "xgb_pred": xgb_pred})

meta_y= y_cat


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
meta_X = scaler.fit_transform(meta_X)


from sklearn.linear_model import LogisticRegression

logistic_model= LogisticRegression(max_iter=2000)
logistic_model.fit(meta_X, meta_y)


test_cat_pred= cat_model.predict_proba(test_cat)[:,1]
test_xgb_pred= xgb_model.predict_proba(test_xgb)[:,1]

print(test_cat_pred)
print(test_xgb_pred)


meta_test_X= pd.DataFrame({
    "cat_pred": test_cat_pred,
    "xgb_pred": test_xgb_pred})

meta_test_X= scaler.transform(meta_test_X)


final_test_pred = logistic_model.predict_proba(meta_test_X)[:, 1]

submission = pd.DataFrame({
    "id": test_data["id"],
    "loan_paid_back": final_test_pred
})

submission.to_csv("submission.csv", index=False)
print("Submission file created!")

