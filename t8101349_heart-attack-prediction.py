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


train_path = "/kaggle/input/unibs-heart-attack-analysis-prediction-2024/train.csv"
train_df = pd.read_csv(train_path)
train_df


train_df.head()


train_df.shape


train_df.describe()


train_df.isnull().sum()


train_df_clean = train_df.dropna()
train_df_clean


y = train_df_clean["target"]
X = train_df_clean.drop(columns = ["target"])


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



X_train


import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=1
)

# 訓練模型
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_valid)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 混淆矩陣
cm = confusion_matrix(y_valid, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


test_path = "/kaggle/input/unibs-heart-attack-analysis-prediction-2024/test.csv"
test_df = pd.read_csv(test_path)
test_df


test_df.isnull().sum()


#觀察分布
print(y.mean())
y.value_counts(normalize=True)



import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = train_df_clean.corr(numeric_only=True)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.show()



corr_with_target = corr_matrix["target"].abs().sort_values(ascending=False)
print(corr_with_target)



#小於0.1的特徵捨棄
features = ["st_slope","exercise_angina","chest_pain_type","oldpeak","pulse","max_heart_rate","sex","age","cholesterol","fasting_blood_sugar","resting_bp_s","target"]
train_df=train_df[features]
train_df


train_df[train_df.cholesterol.isnull()]


#觀察缺值是否代表甚麼意義
train_df[train_df.cholesterol.isnull()]["target"].value_counts(normalize=True)



train_df[train_df.fasting_blood_sugar.isnull()]["target"].value_counts(normalize=True)



train_df[train_df.chest_pain_type.isnull()]["target"].value_counts(normalize=True)



#用平均或眾數填補


cat_cols = ["st_slope", "exercise_angina", "chest_pain_type", "sex", "fasting_blood_sugar"]
num_cols = ["oldpeak", "pulse", "max_heart_rate", "age", "cholesterol","resting_bp_s"]

for col in cat_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)

for col in num_cols:
    train_df[col].fillna(train_df[col].mean(), inplace=True)


train_df[features].isnull().sum()



y = train_df["target"]
X = train_df.drop(columns = ["target"])


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

model = xgb.XGBClassifier(random_state=1, use_label_encoder=False, eval_metric='logloss')

param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [ 7, 8, 9],
    'learning_rate': [0.1, 0.2,0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.3]
}

# 設定 RandomizedSearchCV，用 AUC 作為評分標準
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=100,
    scoring='roc_auc',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)


random_search.fit(X_train, y_train)


print("Best parameters found:", random_search.best_params_)
print("Best AUC Score (CV):", random_search.best_score_)



from sklearn.metrics import roc_auc_score

best_model = random_search.best_estimator_
y_proba = best_model.predict_proba(X_valid)[:, 1]
auc_score = roc_auc_score(y_valid, y_proba)

print("AUC on validation set:", auc_score)




threshold = 0.5
y_pred = (y_proba >= threshold).astype(int)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 混淆矩陣
cm = confusion_matrix(y_valid, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()



threshold = 0.3
y_pred_custom = (y_proba >= threshold).astype(int)




from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 混淆矩陣
cm = confusion_matrix(y_valid, y_pred_custom)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(y_valid, y_pred))
print("Precision:", precision_score(y_valid, y_pred))
print("Recall:", recall_score(y_valid, y_pred))
print("F1 Score:", f1_score(y_valid, y_pred))



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(y_valid, y_pred_custom))
print("Precision:", precision_score(y_valid, y_pred_custom))
print("Recall:", recall_score(y_valid, y_pred_custom))
print("F1 Score:", f1_score(y_valid, y_pred_custom))



features = ["st_slope","exercise_angina","chest_pain_type","oldpeak","pulse","max_heart_rate","sex","age","cholesterol","fasting_blood_sugar","resting_bp_s"]
test_df=test_df[features]
test_df


cat_cols = ["st_slope", "exercise_angina", "chest_pain_type", "sex", "fasting_blood_sugar"]
num_cols = ["oldpeak", "pulse", "max_heart_rate", "age", "cholesterol","resting_bp_s"]

for col in cat_cols:
    test_df[col].fillna(train_df[col].mode()[0], inplace=True)

for col in num_cols:
    test_df[col].fillna(train_df[col].mean(), inplace=True)


test_df.isnull().sum()


test_df.shape


test_proba = best_model.predict_proba(test_df)[:, 1]
test_proba.shape


threshold = 0.3
test_y_pred = (test_proba >= threshold).astype(int)
test_y_pred.shape


sample_path = "/kaggle/input/unibs-heart-attack-analysis-prediction-2024/sampleSubmission.csv"
sample_df = pd.read_csv(sample_path)
sample_df


submission_df = pd.DataFrame({
    "Id": range(1, len(test_y_pred) + 1),
    "target": test_y_pred
})

# 儲存成 CSV
submission_df.to_csv("submission.csv", index=False)



import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

model = xgb.XGBRegressor(random_state=1)

param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [9, 10, 11],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.3]
}

# 設定 RandomizedSearchCV，用 AUC 作為評分標準
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=100,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)


random_search.fit(X_train, y_train)


print("Best parameters found:", random_search.best_params_)
print("Best MSE Score (CV):", -1*random_search.best_score_)


from sklearn.metrics import roc_auc_score

best_model = random_search.best_estimator_
y_proba = best_model.predict(X_valid)
threshold = 0.5
y_pred = (y_proba >= threshold).astype(int)
auc_score = roc_auc_score(y_valid, y_pred)

print("AUC on validation set:", auc_score)


from sklearn.metrics import roc_auc_score

best_model = random_search.best_estimator_
y_proba = best_model.predict(X_valid)
threshold = 0.3
y_pred = (y_proba >= threshold).astype(int)
auc_score = roc_auc_score(y_valid, y_pred)

print("AUC on validation set:", auc_score)


from sklearn.metrics import roc_auc_score

best_model = random_search.best_estimator_
y_proba = best_model.predict(X_valid)
threshold = 0.15
y_pred = (y_proba >= threshold).astype(int)
auc_score = roc_auc_score(y_valid, y_pred)

print("AUC on validation set:", auc_score)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("MSE:", mean_squared_error(y_valid, y_pred))
print("MAE:", mean_absolute_error(y_valid, y_pred))
print("R²:", r2_score(y_valid, y_pred))




from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 混淆矩陣
cm = confusion_matrix(y_valid, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()



test_proba = best_model.predict(test_df)
threshold = 0.3
y_pred = (test_proba >= threshold).astype(int)
y_pred.shape


submission_df = pd.DataFrame({
    "Id": range(1, len(test_y_pred) + 1),
    "target": test_y_pred
})

# 儲存成 CSV
submission_df.to_csv("submission.csv", index=False)


