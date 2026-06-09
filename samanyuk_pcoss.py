# Importing
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


tr=pd.read_csv(r'/kaggle/input/exploring-predictive-health-factors/train.csv')
tr #Reading and displaying the Training Set


tr.info()


# Checking for null values
plt.figure(figsize=(15,8))
sns.heatmap(tr.isnull())
plt.show()


# Digging deeper into these null values
for col in tr.columns:
    if tr[col].isnull().any()==True:
        print(tr[col].value_counts())


# Dropping the ID column
tr.drop(columns='ID',inplace=True)


# Checking out Age Feature
tr['Age'].value_counts()


# Filling these NA Values
# All the features which contain null values are categorical so filling it with mode
for col in tr.columns:
    if tr[col].isnull().any()==True:
        tr[col]=tr[col].fillna(tr[col].mode()[0])


tr.isnull().any()


# Checking if the Target variable is balanced
tr['PCOS'].value_counts()


# Target Variable is not balanced :(
# The Age Feature values are not in order so replacing them.. and making it uniform
tr['Age']=tr['Age'].str.replace('30-25','25-30')
tr['Age']=tr['Age'].str.replace('Less than 20-25','Less than 20')


# Checking out the Excercise Feature
tr['Exercise_Type'].value_counts()


# Not formatted properly
# Eg: Cardio (e.g., running, cycling, swimming) and Cardio (e.g.  will be treated the same i.e. Cardio
tr['Exercise_Type'] = tr['Exercise_Type'].str.replace(r'^Cardio.*', 'Cardio', regex=True)
tr['Exercise_Type'] = tr['Exercise_Type'].str.replace(r'^Strength training.*', 'Strength Training', regex=True)
tr['Exercise_Type'] = tr['Exercise_Type'].str.replace(r'^Flexibility and balance.*', 'Flexibility and balance', regex=True)


# Formatted Feature
tr['Exercise_Type'].value_counts()


fig,ax=plt.subplots(4,3,figsize=(20,20))
ax=ax.flatten()
i=0
for col in tr.columns[tr.dtypes=='object']:
    sns.countplot(data=tr,x=tr[col],ax=ax[i])
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45, ha="right") 
    ax[i].set_title(col, fontsize=14) 
    i+=1
plt.tight_layout()
plt.show()
    


plt.figure(figsize=(20,5))
sns.histplot(data=tr,x='Weight_kg',kde=True)
plt.show()


# Using Label Encoder to encode the Categorical values
# We can also use one hot encoding.. depends on the Model you will be using
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in tr.columns[tr.dtypes=='object']:
    tr[col]=le.fit_transform(tr[col])


tr.info()


# Importing
from sklearn.feature_selection import mutual_info_classif


# Calculating Mutual Information
x=tr.drop(columns='PCOS')
y=tr['PCOS']

mi=mutual_info_classif(x,y)
mi_df=pd.DataFrame({'cols':x.columns,'mi':mi})
mi_df.sort_values(inplace=True,by='mi',ascending=False)
plt.figure(figsize=(15,6))
sns.barplot(data=mi_df,x='mi',y='cols')
plt.show()


# Correlation
plt.figure(figsize=(15,10))
sns.heatmap(tr.corr(),annot=True)
plt.show()


te=pd.read_csv(r'/kaggle/input/exploring-predictive-health-factors/test.csv')
te.info()
tte=te.copy()


te.drop(columns='ID',inplace=True)
for col in te.columns:
    if te[col].isnull().any()==True:
        te[col]=te[col].fillna(te[col].mode()[0])
le=LabelEncoder()
for col in te.columns[te.dtypes=='object']:
    te[col]=le.fit_transform(te[col])


te.info()


# importing
from sklearn.model_selection import train_test_split


x=tr.drop(columns='PCOS')
y=tr['PCOS']
x_t,x_te,y_t,y_te=train_test_split(x,y,random_state=20)


# importing
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV


lgbm = LGBMClassifier(verbose=-1)
params = {
    'boosting_type': ['gbdt', 'dart'],'num_leaves': np.arange(20, 150, 10),'min_child_samples': np.arange(5, 30),  
    'learning_rate': [0.01, 0.05, 0.1, 0.2], 'n_estimators': [2000]}

nrf = RandomizedSearchCV(lgbm,param_distributions=params,cv=10,random_state=20,scoring='roc_auc',
                         n_jobs=-1)


# Fit the model
nrf.fit(x_t, y_t)
# Print the best parameters and the best score
print("Best parameters:", nrf.best_params_)
print("Best ROC AUC score:", nrf.best_score_)


from sklearn.metrics import roc_curve, auc

y_prob_train = nrf.best_estimator_.predict_proba(x_t)[:, 1]

fpr, tpr, _ = roc_curve(y_t, y_prob_train)
roc_auc = auc(fpr, tpr)

# Plot the ROC Curve
plt.figure(figsize=(20,5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--") 
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Training Set)")
plt.legend(loc="lower right")
plt.show()

print("ROC AUC Score (Training Set):", roc_auc)


pred_prob = nrf.best_estimator_.predict_proba(te)
pcos_probabilities = pred_prob[:, 1]
predictions = (pcos_probabilities >= 0.5).astype(int)


submission = pd.DataFrame({
    'ID': tte['ID'],  
    'PCOS': pcos_probabilities, })

submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'  Yaaaay :)")

