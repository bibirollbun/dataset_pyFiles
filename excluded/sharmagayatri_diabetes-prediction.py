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


from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


df= pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
testdf = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


df.sample(5)


df.shape


df.info()


df.duplicated().sum()


df.describe()


df['diagnosed_diabetes'].value_counts(normalize=True)


import matplotlib.pyplot as plt


df.hist(bins=50,figsize=(20,25))
plt.show()


bins = [0, 18, 35, 60, 100]
names = ['Child', 'Young Adult', 'Middle-Aged', 'Senior']

# 3. Apply the pd.cut() function to create a new 'Age_Group' column
df['Age_Group'] = pd.cut(x=df['age'], bins=bins, labels=names)

df.head()


X= df.drop(columns=['diagnosed_diabetes','id'])
y= df['diagnosed_diabetes']


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_idx, test_idx = next(split.split(X, y))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


num_col = X_train.select_dtypes(include=['int64','float64']).columns
cat_col = X_train.select_dtypes(include=['object']).columns


signal = {}

for col in num_col:
    signal[col] = abs(
        df[df['diagnosed_diabetes']==1][col].mean()
        - df[df['diagnosed_diabetes']==0][col].mean()
    )

pd.Series(signal).sort_values(ascending=False)


# ---------------------------
# Medical Interactions
# ---------------------------
df["bmi_age"] = df["bmi"] * df["age"]
df["bp_product"] = df["systolic_bp"] * df["diastolic_bp"]
df["chol_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1)
df["waist_bmi"] = df["waist_to_hip_ratio"] * df["bmi"]

# ---------------------------
# Binary Clinical Thresholds
# ---------------------------
df["obese"] = (df["bmi"] >= 30).astype(int)
df["high_bp"] = (
    (df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)
).astype(int)
df["low_hdl"] = (df["hdl_cholesterol"] < 40).astype(int)
df["high_ldl"] = (df["ldl_cholesterol"] >= 130).astype(int)
df["high_triglycerides"] = (df["triglycerides"] >= 150).astype(int)

# ---------------------------
# Lifestyle Risk Index
# ---------------------------
df["lifestyle_risk"] = (
    (df["alcohol_consumption_per_week"] > 7).astype(int) +
    (df["physical_activity_minutes_per_week"] < 150).astype(int) +
    (df["sleep_hours_per_day"] < 6).astype(int) +
    (df["screen_time_hours_per_day"] > 6).astype(int)
)

# ---------------------------
# Genetic + Cardio Interaction
# ---------------------------
df["genetic_cardiac_risk"] = (
    df["family_history_diabetes"] *
    (df["hypertension_history"] + df["cardiovascular_history"])
)





from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

ordinal_col = [ 'education_level',
       'smoking_status']

nominal_col = [col for col in cat_col if col not in ordinal_col]


num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])

ord_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(
        categories=[
            ['No formal', 'Highschool', 'Graduate', 'Postgraduate'],
            ['Never', 'Former', 'Current']
        ]
    ))
])

nom_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_col),
    ('ord', ord_pipe, ordinal_col),
    ('nom', nom_pipe, nominal_col)
])



# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import StackingClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score

# from catboost import CatBoostClassifier



# # Base models (NO preprocessing here)
# base_models = [
#     ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    
#     ("dt", DecisionTreeClassifier(
#         max_depth=4,
#         random_state=42
#     )),
    
#     ("cat", CatBoostClassifier(
#         iterations=300,
#         depth=6,
#         learning_rate=0.05,
#         verbose=False
#     ))
# ]

# # Meta model
# meta_model = LogisticRegression(max_iter=1000, class_weight="balanced")

# # Stacking model
# stack_model = StackingClassifier(
#     estimators=base_models,
#     final_estimator=meta_model,
#     cv=5,
#     passthrough=False,
#     n_jobs=-1
# )

# # FINAL PIPELINE (this is the key part)
# final_pipeline = Pipeline([
#     ("preprocessor", preprocessor),   # ðŸ‘ˆ your existing preprocessing
#     ("model", stack_model)
# ])

# # Train
# final_pipeline.fit(X_train, y_train)

# # Evaluate
# y_pred = final_pipeline.predict(X_test)
# y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))



from catboost import CatBoostClassifier

rf_model = Pipeline([
    ('prep',preprocessor),
    ('model',CatBoostClassifier(
    n_estimators=2000, 
    learning_rate=0.05, 
    depth=6,
    random_state=42, 
    eval_metric='AUC', 
    early_stopping_rounds=50,))
])

rf_model.fit(X_train, y_train)

# probabilities, not labels
y_test_prob = rf_model.predict_proba(X_test)[:, 1]
y_train_prob = rf_model.predict_proba(X_train)[:, 1]

auc_val = roc_auc_score(y_test, y_test_prob)
auc_train = roc_auc_score(y_train, y_train_prob)

print("Train AUC:", auc_train)
print("Test AUC:", auc_val)


# ---------------------------
# Medical Interactions
# ---------------------------
testdf["bmi_age"] = testdf["bmi"] * testdf["age"]
testdf["bp_product"] = testdf["systolic_bp"] * testdf["diastolic_bp"]
testdf["chol_ratio"] = testdf["ldl_cholesterol"] / (testdf["hdl_cholesterol"] + 1)
testdf["waist_bmi"] = testdf["waist_to_hip_ratio"] * testdf["bmi"]

# ---------------------------
# Binary Clinical Thresholds
# ---------------------------
testdf["obese"] = (testdf["bmi"] >= 30).astype(int)
testdf["high_bp"] = (
    (testdf["systolic_bp"] >= 140) | (testdf["diastolic_bp"] >= 90)
).astype(int)
testdf["low_hdl"] = (testdf["hdl_cholesterol"] < 40).astype(int)
testdf["high_ldl"] = (testdf["ldl_cholesterol"] >= 130).astype(int)
testdf["high_triglycerides"] = (testdf["triglycerides"] >= 150).astype(int)

# ---------------------------
# Lifestyle Risk Index
# ---------------------------
testdf["lifestyle_risk"] = (
    (testdf["alcohol_consumption_per_week"] > 7).astype(int) +
    (testdf["physical_activity_minutes_per_week"] < 150).astype(int) +
    (testdf["sleep_hours_per_day"] < 6).astype(int) +
    (testdf["screen_time_hours_per_day"] > 6).astype(int)
)

# ---------------------------
# Genetic + Cardio Interaction
# ---------------------------
testdf["genetic_cardiac_risk"] = (
    testdf["family_history_diabetes"] *
    (testdf["hypertension_history"] + testdf["cardiovascular_history"])
)



test_proba = rf_model.predict_proba(testdf)[:, 1]


submission = pd.DataFrame({
    'id': testdf['id'],
    'diagnosed_diabetes': test_proba
})
submission.to_csv('submission.csv', index=False)




