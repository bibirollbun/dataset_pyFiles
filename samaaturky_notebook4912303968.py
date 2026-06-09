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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report , confusion_matrix
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
from xgboost import XGBClassifier


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")



train.head()


train.isnull().sum()


train.duplicated().sum()


train.info()


train = train.drop(columns=['id'])


numerical_features=['age','balance','day','duration', 'campaign','pdays', 'previous','y']


for col in numerical_features:
    plt.figure(figsize=(6, 4))
    train.boxplot(column=col)
    plt.title(f"Boxplot of {col}")


train[numerical_features].hist(figsize=(12, 10))


for col in train.select_dtypes(include='object').columns:
    plt.figure(figsize=(8,4))
    sns.countplot(data=train, x=col)
    plt.title(f"Countplot of {col}")


train=train.drop(columns=['default','poutcome'])


train.columns


corr=train[numerical_features].corr()

#heatmap:
sns.heatmap(corr,annot=True)


for col in train.select_dtypes(include='object').columns:
    print(f'{col}: {train[col].unique()}')


nominal= ['marital', 'housing','loan']
ordinal= ['education','month']
knn_nominal = ['job', 'contact']
pwr_transform_features = ['balance', 'duration', 'pdays', 'previous']
non_pwr_numerical =['age','day', 'campaign']
final_nominal = nominal + knn_nominal


#split x,y
X = train.drop(columns=['y'])
y = train['y']


# split to train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


education_order = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y',
    'high.school', 'secondary', 'primary', 'tertiary',
    'professional.course', 'university.degree', 'unknown']
month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline



# # Ordinal encoding
# ord_encoder = OrdinalEncoder(categories=[education_order, month_order])
# X_train_ord = ord_encoder.fit_transform(X_train[ordinal])
# X_val_ord = ord_encoder.transform(X_val[ordinal])

# # One-hot encoding
# ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
# X_train_nom = ohe.fit_transform(X_train[nominal])
# X_val_nom = ohe.transform(X_val[nominal])

# # power transformation
# pwr_scaler = PowerTransformer(method='yeo-johnson')
# X_train_log = pwr_scaler.fit_transform(X_train[pwr_transform_features])
# X_val_log = pwr_scaler.transform(X_val[pwr_transform_features])

# #scale others:
# scaler = StandardScaler()
# X_train_num = scaler.fit_transform(X_train[non_pwr_numerical])
# X_val_num = scaler.transform(X_val[non_pwr_numerical])

# #concatenate
# X_train = np.hstack([X_train_num, X_train_log, X_train_ord, X_train_nom, X_train_knn])
# X_val = np.hstack([X_val_num, X_val_log, X_val_ord, X_val_nom, X_val_knn])

# print("Train shape:", X_train.shape)
# print("Val shape:", X_val.shape)


# clf = LogisticRegression(
#     solver="liblinear", 
#     class_weight="balanced", 
#     random_state=42
# )

# clf.fit(X_train, y_train)

# y_pred_train = clf.predict(X_train)
# y_pred_val = clf.predict(X_val)
# y_proba = clf.predict_proba(X_val)[:,1]

# print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
# print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
# print("ROC AUC:", roc_auc_score(y_val, y_proba))


# rclf=RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=20, min_samples_leaf=10, random_state=42,class_weight="balanced")
# rclf.fit(X_train, y_train)

# y_pred_train = rclf.predict(X_train)
# y_pred_val = rclf.predict(X_val)
# y_proba = rclf.predict_proba(X_val)[:,1]

# print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
# print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
# print("ROC AUC:", roc_auc_score(y_val, y_proba))
# print("Classification Report:\n", classification_report(y_val, y_pred_val))


# rclf=RandomForestClassifier(n_estimators=430,max_depth=15,min_samples_split=20, min_samples_leaf=10, random_state=42,class_weight="balanced")
# rclf.fit(X_train, y_train)

# y_pred_train = rclf.predict(X_train)
# y_pred_val = rclf.predict(X_val)
# y_proba = rclf.predict_proba(X_val)[:,1]

# print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
# print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
# print("ROC AUC:", roc_auc_score(y_val, y_proba))
# print("Classification Report:\n", classification_report(y_val, y_pred_val))


# from sklearn.ensemble import BaggingClassifier
# #bagging:
# bagging_clf = BaggingClassifier(estimator=rclf, n_estimators=50, random_state=42)
# bagging_clf.fit(X_train, y_train)

# y_pred_train = bagging_clf.predict(X_train)
# y_pred_val = bagging_clf.predict(X_val)
# y_proba = bagging_clf.predict_proba(X_val)[:,1]

# print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
# print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
# print("ROC AUC:", roc_auc_score(y_val, y_proba))
# print("Classification Report:\n", classification_report(y_val, y_pred_val))


# from xgboost import XGBClassifier
# #initialize the XGBoost classifier
# xgb_clf = XGBClassifier(n_estimators=1000, learning_rate=0.1, random_state=42)

# xgb_clf.fit(X_train, y_train)

# #predict
# y_pred_train = xgb_clf.predict(X_train)
# y_pred_val = xgb_clf.predict(X_val)
# y_proba = xgb_clf.predict_proba(X_val)[:,1]

# #evaluate
# print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
# print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
# print("ROC AUC:", roc_auc_score(y_val, y_proba))
# print("Classification Report:\n", classification_report(y_val, y_pred_val))


# from xgboost import XGBClassifier
# #initialize the XGBoost classifier
# xgb_clf = XGBClassifier(gamma= 0.20584494295802447, learning_rate=0.20398197043239888,max_depth= 6, n_estimators= 863,random_state=42)

# xgb_clf.fit(X_train, y_train)

# #predict
# y_pred_train = xgb_clf.predict(X_train)
# y_pred_val = xgb_clf.predict(X_val)
# y_proba = xgb_clf.predict_proba(X_val)[:,1]

# #evaluate
# print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
# print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
# print("ROC AUC:", roc_auc_score(y_val, y_proba))
# print("Classification Report:\n", classification_report(y_val, y_pred_val))


# from xgboost import XGBClassifier
# #initialize the XGBoost classifier
# xgb_clf = XGBClassifier(colsample_bytree=0.7123738038749523, gamma= 2.7134804157912424, learning_rate=0.03818484499495253,max_depth= 10, min_child_weight= 1, n_estimators=970, scale_pos_weight=7.288437629506838, subsample= 0.7579526072702278
# ,random_state=42)

# xgb_clf.fit(X_train, y_train)

# #predict
# y_pred_train = xgb_clf.predict(X_train)
# y_pred_val = xgb_clf.predict(X_val)
# y_proba = xgb_clf.predict_proba(X_val)[:,1]

# #evaluate
# print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
# print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
# print("ROC AUC:", roc_auc_score(y_val, y_proba))
# print("Classification Report:\n", classification_report(y_val, y_pred_val))


# from xgboost import XGBClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint, uniform

# xgb = XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='auc',
#     use_label_encoder=False,
#     random_state=42
# )

# param_dist = {
#     'n_estimators': randint(450, 1000),
#     'max_depth': randint(3, 10),
#     'learning_rate': uniform(0.01, 0.2),
#     'gamma': uniform(0, 10)
# }

# random_search = RandomizedSearchCV(
#     estimator=xgb,
#     param_distributions=param_dist,
#     n_iter=15,
#     scoring='roc_auc',
#     cv=3,
#     verbose=2,
#     random_state=42,
#     n_jobs=-1
# )

# random_search.fit(X_train, y_train)
# print("Best Params:", random_search.best_params_)
# print("Best AUC:", random_search.best_score_)



# Pipelines

# Non-log numerical
num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

# Log + scale pipeline
pwr_pipeline = Pipeline([
    ("pwr", PowerTransformer(method='yeo-johnson')),
    ("scaler", StandardScaler())
])

# Ordinal features
ord_pipeline = Pipeline([
    ("encoder", OrdinalEncoder(categories=[education_order, month_order]))
])

# Nominal features
nom_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Job/contact pipeline: encode ordinal + KNN impute
knn_nom_pipeline = Pipeline([
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ("imputer", KNNImputer(n_neighbors=5))
])

# Column transformer
preprocessor = ColumnTransformer([
    ("num", num_pipeline, non_pwr_numerical),
    ("log", pwr_pipeline, pwr_transform_features),
    ("ord", ord_pipeline, ordinal),
    ("nom", nom_pipeline, final_nominal),
    ("knn_nom", knn_nom_pipeline, ["job", "contact"])
])
# Final Pipeline with SMOTE and Classifier
model_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', XGBClassifier(gamma= 0.20584494295802447, learning_rate=0.20398197043239888,max_depth= 6, n_estimators= 863,random_state=42))

])


#Train the pipeline 
model_pipeline.fit(X_train, y_train)
# evaluate 
print("train Accuracy:", model_pipeline.score(X_train, y_train))
print("Validation Accuracy:", model_pipeline.score(X_val, y_val))


predictions = model_pipeline.predict_proba(test)[:, 1]

#submission
submission= pd.DataFrame({
    "id": test["id"],
    "y": predictions
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Saved submission.csv in working directory")





