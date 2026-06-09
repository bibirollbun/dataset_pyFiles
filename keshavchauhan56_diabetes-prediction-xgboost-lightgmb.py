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


# Loading the datatset.
train_data=pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_data=pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample_data=pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")


#Setting display option to max 26 columns
pd.set_option('display.max_columns', 26)


# Train_dataset with top 5 rowa
train_data.head()


# Test_dataset with top 5 rowa
test_data.head()


# Sample_dataset with top 5 rowa
sample_data.head()


#Checking train dataset info.
train_data.info()


#Checking test dataset info.
test_data.info()


# Droping the ID columns which not related to our data
train_data.drop(["id"], axis=1, inplace=True)
test_data.drop(["id"], axis=1, inplace=True)


#Checking dimension of the dataset.
print(f"Rows and Columns in train Dataset {train_data.shape}")
print(f"Rows and Columns in test Dataset {test_data.shape}")
print(f"Rows and Columns in sample Dataset {sample_data.shape}")


# Columns names and count fo the columns in train datatset.
print(f"Columns Names {train_data.columns}")
print(f"Columns Count {len(train_data.columns)}")


# Changing the diagnosed_diabetes columns dtype float to int
train_data["diagnosed_diabetes"]= train_data["diagnosed_diabetes"].astype("int")


#Checking statistic of the train dataset.
round(train_data.describe())


# Checking null values in daataset.
print(f"Checking null values in train dataset: {train_data.isnull().sum().sum()}")
print(f"Checking null values in test dataset: {test_data.isnull().sum().sum()}")


#Checking duplicates
print(f"Duplicates in the train dataset: {train_data.duplicated().sum().sum()}")
print(f"Duplicates in the test dataset: {test_data.duplicated().sum().sum()}")


#Checking the unique values.
print(f"Unique values in the train dataset: {train_data.nunique().sum().sum()}")
print(f"Unique values in the test dataset: {test_data.nunique().sum().sum()}")


#Creating new column as per the age group that given in th dataset.
bins=[20,35,55,100]
labels= ["Young adult", "Middle-Age", "Senior"]

train_data["age_group"]= pd.cut(train_data["age"], bins=bins, labels=labels)

print(train_data[["age", "age_group"]].head)


#Value count of age groups having diabetes
train_data["age_group"].value_counts()


# Changing the diagnosed_diabetes columns dtype float to int
train_data["age_group"]= train_data["age_group"].astype("object")


ax=sns.countplot(x ="diagnosed_diabetes", data= train_data)
ax.bar_label(ax.containers[0])
plt.title("Count of People Have Diabetes")
plt.show()


plt.figure(figsize=(8,5))
gb= train_data.groupby("diagnosed_diabetes").agg({"diagnosed_diabetes":"count"})
plt.pie(gb["diagnosed_diabetes"], labels= gb.index, autopct= "%1.2f%%")
plt.title("Percentage of People have Diabetes")
plt.legend(labels=["No","Yes"], fancybox=True, loc="lower left")
plt.show()


ax=sns.countplot(x= "gender", data=train_data, hue="diagnosed_diabetes")
ax.bar_label(ax.containers[0])
plt.title("Gender Count by Diabetes")
plt.show()


ax=sns.countplot(x= "smoking_status", data=train_data, hue="diagnosed_diabetes")
ax.bar_label(ax.containers[0])
plt.title("Smoker Count by Diabetes")
plt.show()


plt.figure(figsize=(5,4))
sns.histplot(x= "age", data=train_data, bins=80, hue="diagnosed_diabetes")
plt.title("Diabetes Accordingly Age")
plt.show()


ax=sns.countplot(x= "age_group", data=train_data, hue="diagnosed_diabetes")
ax.bar_label(ax.containers[0])
plt.title("Age Group Count by Diabetes")
plt.show()


plt.figure(figsize=(5,4))
sns.histplot(x= "bmi", data=train_data, hue="diagnosed_diabetes")
plt.title("Diabetes Accordingly BMI")
plt.show()


ax=sns.countplot(x ="income_level", data= train_data, hue="diagnosed_diabetes")
ax.bar_label(ax.containers[0])
plt.title("Count By Income Level")
plt.show()


# Converting all categorical column to contniues column in train dataset.
from sklearn.preprocessing import LabelEncoder
label_encoder={}
cat_col= train_data.select_dtypes(include=['object']).columns.tolist()

for col in cat_col:
    train_data[col]= train_data[col].astype(str)
    label_encoder[col]= LabelEncoder()
    train_data[col]= label_encoder[col].fit_transform(train_data[col])
    train_data[col]= train_data[col].astype(str)


# Converting all categorical column to contniues column in test dataset.
label_encoder={}
cat_col= test_data.select_dtypes(include=['object']).columns.tolist()

for col in cat_col:
    test_data[col]= test_data[col].astype(str)
    label_encoder[col]= LabelEncoder()
    test_data[col]= label_encoder[col].fit_transform(test_data[col])
    test_data[col]= test_data[col].astype(str)


corrdf= train_data.corr()

plt.figure(figsize=(18,12))
sns.heatmap(corrdf, annot=True, cmap="rainbow")
plt.title("Correlation Heatmap", fontsize=20)
plt.show()


#Checking coor relation of independent variable to target variable.
target_corr= corrdf["diagnosed_diabetes"].abs().sort_values(ascending=False)
selected_features = target_corr[target_corr > 0.01].index.tolist()
low_corr_features= target_corr[(target_corr<0.01)].index.tolist()

print(target_corr)
print("")
print("\nSelected features between 0.1 and 0.5:")
print(selected_features)
print("")
print(len(selected_features))

print("\nLow features between 0.1 and 0.5:")
print(low_corr_features)
print("")
print(len(low_corr_features))


# droping the less correlacted feature from the train dataset.
train_data.drop(low_corr_features, axis=1, inplace=True)
print(train_data.shape)
train_data.head()


# droping the less correlacted feature from the test dataset.
low_corr=['income_level', 'education_level', 'sleep_hours_per_day', 'alcohol_consumption_per_week', 'employment_status', 'gender', 'smoking_status', 'ethnicity']

test_data.drop(low_corr, axis=1, inplace=True)
print(test_data.shape)
test_data.head()


# Spliting the dataset into train test using train_test_split
x= train_data.drop(["diagnosed_diabetes"], axis=1)
y= train_data["diagnosed_diabetes"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(x,y, train_size=0.8, random_state=2410)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


#First going with random forest classifier model
from sklearn.ensemble import RandomForestClassifier

model= RandomForestClassifier().fit(X_train,y_train)
y_test_predict= model.predict(X_test)

y_pred_prob_1 = model.predict_proba(X_test)[:,1]

print(y_test_predict)
print(y_pred_prob_1)


# Evaluation Report for Random Forect Classifier.
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_test_predict))
print(classification_report(y_test, y_test_predict))


# Roc auc score using Random Forest Classifier 
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_pred_prob_1)
print("ROC–AUC:", roc_auc)


from sklearn.linear_model import LogisticRegression

model_lg= LogisticRegression().fit(X_train, y_train)
y_predict= model_lg.predict(X_test)

y_pred_prob_2 = model_lg.predict_proba(X_test)[:,1]

print(y_predict)
print(y_pred_prob_2)


# Evaluation Report for Logistic Regression.
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))


# Roc auc score using logistic Regression

from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_pred_prob_2)
print("ROC–AUC:", roc_auc)


#Standard Scaling the dataset.
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler().fit(X_train)

trainSTD= scaler.transform(X_train)
testSTD= scaler.transform(X_test)

#Conversting into DataFrame
trainstd= pd.DataFrame(trainSTD, columns=X_train.columns)
teststd= pd.DataFrame(testSTD, columns=X_test.columns)

print(trainstd.shape)
print(teststd.shape)


#Checking the model with STD data (Logistic Regression).
lgstd= LogisticRegression().fit(trainstd, y_train)
y_prediction= lgstd.predict(X_test)
y_predict_prob= lgstd.predict_proba(X_test)[:,1]

print(y_prediction)
print(y_predict_prob)


# Evaluation Report for Logistic Regression.
print(confusion_matrix(y_test, y_prediction))
print(classification_report(y_test, y_prediction))


roc_auc = roc_auc_score(y_test, y_predict_prob)
print("ROC–AUC:", roc_auc)


#Using xgboost model for better performance.
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42)

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]
print(y_pred_proba)


roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC Score:", roc_auc)


import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()


from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model, x, y, cv=cv, scoring='roc_auc'
)

print("Mean ROC AUC:", scores.mean())


model_2 = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight = (y == 0).sum() / (y == 1).sum())

model_2.fit(X_train, y_train)

y_pred_proba_2 = model_2.predict_proba(X_test)[:, 1]
print(y_pred_proba_2)


roc_auc = roc_auc_score(y_test, y_pred_proba_2)
print("ROC AUC Score:", roc_auc)


fpr, tpr, _ = roc_curve(y_test, y_pred_proba_2)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()


import lightgbm as lgb

#Scaling the imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


model_lgb = lgb.LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.02,

    max_depth=-1,
    num_leaves=64,          
    min_child_samples=50, 

    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,

    reg_alpha=0.5,          
    reg_lambda=1.0,         
    min_split_gain=0.01,

    objective='binary',
    metric='auc',
    scale_pos_weight=scale_pos_weight,

    random_state=42,
    n_jobs=-1)



model_lgb.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(50)])


y_predict_lgb= model_lgb.predict(X_test)
y_pred_proba_lgb = model_lgb.predict_proba(X_test)[:, 1]

print(y_predict_lgb)
print(y_pred_proba_lgb)


roc_auc = roc_auc_score(y_test, y_pred_proba_lgb)
print("LightGMB ROC AUC Score:", roc_auc)


fpr, tpr, _ = roc_curve(y_test, y_pred_proba_lgb)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model_lgb,
    x,
    y,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)

print("CV ROC-AUC Scores:", cv_scores)
print("Mean CV ROC-AUC:", cv_scores.mean())


lgb.plot_importance(model_lgb)


#Predicting to unseen dataset.
final_prediction= model_lgb.predict(test_data)
y_prodict_proba_f= model_lgb.predict_proba(test_data)[:,1]

print(final_prediction)
print(y_prodict_proba_f)


#Here is the prediction on the test dataset.
test_data['diagnosed_diabetes'] = final_prediction
test_data.head()


#Now as required submitting on the sample data.
sample_data['diagnosed_diabetes'] = final_prediction


sample_data.head()


sample_data.to_csv('submission.csv', index=False)

