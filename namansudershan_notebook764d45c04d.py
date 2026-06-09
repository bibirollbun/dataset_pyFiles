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


import pandas as pd
import numpy as np
import seaborn as sns



dataTrain = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
dataTest = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

dfTrain = pd.DataFrame(dataTrain)
dfTest = pd.DataFrame(dataTest)

dfTrain.head()


dfTest.head()


import seaborn as sns
import matplotlib.pyplot as plt



plt.figure(figsize=(10,5))
sns.kdeplot(
    data=dfTrain,
    x="alcohol_consumption_per_week",             # feature to analyse
    hue="diagnosed_diabetes",      # separate curves by class
    common_norm=False,   # normalize each class separately
    fill=True,           # filled KDE (optional, looks nicer)
    alpha=0.4            # transparency
)

plt.title("KDE Plot of BMI for Diabetes vs Non-Diabetes")
plt.show()



plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="alcohol_consumption_per_week",
    hue="diagnosed_diabetes"
)
plt.title("Alcohol Consumption by Diabetes Status")
plt.show() #doesnt seem to be a strong predictor



plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="physical_activity_minutes_per_week",
    hue="diagnosed_diabetes"
)
plt.title(" physical activity vs diabetes")
plt.show() #again not a strong predictor


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="age",
    hue="diagnosed_diabetes"
)
plt.title(" physical activity vs diabetes")
plt.show() #a strong predictor


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="sleep_hours_per_day",
    hue="diagnosed_diabetes"
)
plt.title(" physical activity vs diabetes")
plt.show() #not strong


 #strong predictor


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="waist_to_hip_ratio",
    hue="diagnosed_diabetes"
)
plt.title(" physical activity vs diabetes")
plt.show()  #strong


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="systolic_bp",
    hue="diagnosed_diabetes"
)
plt.title(" physical activity vs diabetes")
plt.show()  #strong


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="gender",
    hue="diagnosed_diabetes"
)
plt.title(" physical activity vs diabetes")
plt.show() #weak


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="income_level",
    hue="diagnosed_diabetes"
)
plt.title(" physical activity vs diabetes")
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="education_level",
    hue="diagnosed_diabetes"
)
plt.title(" physical activity vs diabetes")
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="ethnicity",
    hue="diagnosed_diabetes"
)
plt.title("ethnicity vs diabetes")
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="smoking_status",
    hue="diagnosed_diabetes"
)
plt.title(" smoking status vs diabetes")
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="employment_status",
    hue="diagnosed_diabetes"
)
plt.title(" employment status vs diabetes")
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="family_history_diabetes",
    hue="diagnosed_diabetes"
)
plt.title(" family history of diabetes vs diabetes")
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="hypertension_history",
    hue="diagnosed_diabetes"
)
plt.title(" hypertension history vs diabetes")
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="cardiovascular_history",
    hue="diagnosed_diabetes"
)
plt.title("cardiovascular history vs diabetes")
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="diastolic_bp",
    hue="diagnosed_diabetes"
)
plt.title("diastolic bp vs diabetes")
plt.show() #strong


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="heart_rate",
    hue="diagnosed_diabetes"
)
plt.title("heart rate vs diabetes")
plt.show()


plt.figure(figsize=(8,5))
sns.countplot(
    data=dfTrain,
    x="cholestrol_total",
    hue="diagnosed_diabetes"
)
plt.title("total cholestron vs diabetes")
plt.show()


set(dfTest.columns) - set(dfTrain.columns)




dfTrain.head()


dfTrain["age_bmi"] = dfTrain["age"]*dfTrain["bmi"]
dfTrain["waist_hip_ratio&age"] = dfTrain["waist_to_hip_ratio"]*dfTrain["age"]
dfTrain["bmi_waist"] = dfTrain["waist_to_hip_ratio"]*dfTrain["bmi"]
dfTrain["age_bmi_waist"] = dfTrain["age"]*dfTrain["bmi"]*dfTrain["waist_to_hip_ratio"]


#feature engineering 
dfTest["age_bmi"] = dfTest["age"]*dfTest["bmi"]
dfTest["waist_hip_ratio&age"] = dfTest["waist_to_hip_ratio"]*dfTest["age"]
dfTest["bmi_waist"] = dfTest["waist_to_hip_ratio"]*dfTest["bmi"]
dfTest["age_bmi_waist"] = dfTest["age"]*dfTrain["bmi"]*dfTest["waist_to_hip_ratio"]


#seperating in numerical and categorical columns

cat_cols = dfTrain.select_dtypes(exclude=np.number).columns
cat_cols_train = dfTrain[cat_cols]
cat_cols_test  = dfTest[cat_cols]     # ← use SAME list
num_cols = dfTrain.select_dtypes(include=np.number).columns.drop("diagnosed_diabetes")
num_cols_train = dfTrain[num_cols]
num_cols_test  = dfTest[num_cols]     # ← use SAME list






#ONE HOT ENCODING
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
X_cat = ohe.fit_transform(cat_cols_train)
Y_cat = ohe.transform(cat_cols_test)
dfTrain_final = np.hstack([X_cat , num_cols_train])
dfTest_final = np.hstack([Y_cat , num_cols_test])


from sklearn.model_selection import train_test_split

X = dfTrain_final
y = dfTrain["diagnosed_diabetes"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# Simple XGBoost training, eval and submission
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report


# If inputs are pandas DataFrames, convert to numpy (XGBoost accepts both, but ensure shapes)
X_tr = X_train
X_v  = X_val
y_tr = y_train
y_v  = y_val

# Basic XGBoost classifier (easy-to-tune defaults)
model = XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

# Fit with early stopping on the validation set
model.fit(
    X_tr, y_tr,
    eval_set=[(X_v, y_v)],
    early_stopping_rounds=50,
    verbose=20
)

# Evaluate on validation
val_proba = model.predict_proba(X_v)[:, 1]
val_pred  = model.predict(X_v)
print("Validation ROC-AUC:", round(roc_auc_score(y_v, val_proba), 5))
print("\nClassification report (val):")
print(classification_report(y_v, val_pred, digits=4))

# Predict on test set (dfTest_final must be same feature shape as training)
test_proba = model.predict_proba(dfTest_final)[:, 1]

# Write submission (uses sample_submission structure if available)
try:
    sub = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
    sub["diabetes"] = test_proba
    sub = sub[["id" , "diabetes"]]
    sub.to_csv("submission.csv", index=False)
    print("\nSaved submission.csv — first rows:")
    display(sub.head())
except Exception:
    # fallback: save with index id if you have test ids in dfTest
    pd.DataFrame({"diabetes": test_proba}).to_csv("submission.csv", index=False)
    print("\nSaved submission.csv (fallback).")





