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


import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


test.shape


data.head()


data.info()


non_null = data[~data.isnull().any(axis=1)]


data.shape


non_null.shape


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

for i in non_null.columns:
    if non_null[i].dtype!='object':
        sns.kdeplot(data=non_null,x=i,color='red')
        plt.title(f"{i}")
        plt.show()
    else:
        sns.histplot(data=non_null,x=i,color='green')
        plt.title(f"{i}")
        plt.show()
    


def random_imputation(df):
    df_imputed = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            non_null = df[col].dropna()
            df_imputed[col] = df[col].apply(lambda x: np.random.choice(non_null) if pd.isnull(x) else x)
    return df_imputed


data_imputed = pd.DataFrame()
data_imputed = random_imputation(data)
print(data_imputed.shape)


features = data_imputed.drop('id',axis=1).iloc[:,:-1]
target = data_imputed.iloc[:,-1]


features.head()


for i in features.columns:
    if features[i].dtype=='object':
        print(f"the uniques values of {i} are as follows : {np.unique(features[i].values)}")


features['Stage_fear'] = features['Stage_fear'].map(lambda x : 1 if x=='Yes' else 0)


for i in features.columns:
    if features[i].dtype=='object':
        print(f"the uniques values of {i} are as follows : {np.unique(features[i].values)}")


features['Drained_after_socializing'] = features['Drained_after_socializing'].map(lambda x : 1 if x=='Yes' else 0)


for i in features.columns:
    if features[i].dtype=='object':
        print(f"the uniques values of {i} are as follows : {np.unique(features[i].values)}")


target=target.apply(lambda x : 1 if x=='Introvert' else 0)


def feature_engineering(data):
    print("Original input shape:", data.shape)
    
    data = random_imputation(data)
    print("After random imputation:", data.shape)

    data["Stage_fear"] = data["Stage_fear"].map(lambda x: 1 if x == "Yes" else 0)
    data["Drained_after_socializing"] = data["Drained_after_socializing"].map(lambda x: 1 if x == "Yes" else 0)

    if 'Personality' in data.columns:
        data["Personality"] = data["Personality"].map(lambda x: 1 if x == "Introvert" else 0)

    print("Before feature-target split:", data.shape)

    if 'Personality' in data.columns :
        features = data.drop('id', axis=1).iloc[:, :-1]
        target = data.iloc[:, -1]
        print("Features shape:", features.shape)
        print("Target shape:", target.shape)
        return features, target
    else:
        features = data
        return features
        

    



target.value_counts()
ratio = 13699/4825


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
model = XGBClassifier(scale_pos_weight = ratio)
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.25,random_state=42)
model.fit(X_train,y_train)


from sklearn.metrics import classification_report
preds=model.predict(X_test)
print(classification_report(preds,y_test))


preds_proba = model.predict_proba(X_test)
plt.figure(figsize=(8, 6))


sns.histplot(preds_proba, color='dodgerblue', kde=True, label='Predicted Probabilities', stat='density', bins=25, alpha=0.5)
sns.histplot(y_test, color='goldenrod', kde=True, label='Actual Labels', stat='density', bins=2, alpha=0.4)


plt.xlabel('Probability / Label')
plt.ylabel('Density')
plt.title('Distribution of Predicted Probabilities vs Actual Labels')
plt.legend()
plt.tight_layout()
plt.show()


from sklearn.metrics import roc_curve, auc, RocCurveDisplay



y_proba = model.predict_proba(X_test)[:, 1] 

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Compute AUC
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='dodgerblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()



print(f"shape before : {test.shape}")
test_ids = test['id']
test.drop('id',axis=1,inplace=True)
features_test = feature_engineering(test)


features_test.shape
features_test.columns


test.columns


predictions = model.predict(features_test)


sample.head()


predictions = pd.Series(predictions).apply(lambda x: "Extrovert" if x == 0 else "Introvert")
submission = pd.DataFrame({"id":test_ids.astype("int64"),"Personality":predictions})


submission.head()


submission.to_csv('submission.csv', index=False)


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,features,target,cv=5,scoring='accuracy')
print("Fold-wise scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())


from sklearn.model_selection import RandomizedSearchCV

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_grid = {
    'scale_pos_weight':[ratio],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3, 0.5],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=50,            
    scoring='accuracy',  
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

print("Best params:", search.best_params_)
print("Best score:", search.best_score_)


from xgboost import cv, DMatrix

params = search.best_params_
dtrain = DMatrix(X_train, label=y_train)

cv_results = cv(
    params,
    dtrain,
    num_boost_round=1000,
    nfold=5,
    early_stopping_rounds=20,
    metrics="logloss",
    seed=42
)


best_num_boost_round=len(cv_results)


best_num_boost_round


final_model = XGBClassifier(
    **params,  
    n_estimators=best_num_boost_round,
    use_label_encoder=False,
    eval_metric='logloss'
)

final_model.fit(X_train, y_train)


final_model.score(X_train,y_train)


preds=final_model.predict(X_test)
print(classification_report(preds,y_test))


preds_proba = final_model.predict_proba(X_test)
plt.figure(figsize=(8, 6))


sns.histplot(preds_proba, color='dodgerblue', kde=True, label='Predicted Probabilities', stat='density', bins=25, alpha=0.5)
sns.histplot(y_test, color='goldenrod', kde=True, label='Actual Labels', stat='density', bins=2, alpha=0.4)


plt.xlabel('Probability / Label')
plt.ylabel('Density')
plt.title('Distribution of Predicted Probabilities vs Actual Labels')
plt.legend()
plt.tight_layout()
plt.show()


y_proba = final_model.predict_proba(X_test)[:, 1] 

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Compute AUC
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='dodgerblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()



predictions=final_model.predict(features_test)


predictions = pd.Series(predictions).apply(lambda x: "Extrovert" if x == 0 else "Introvert")
submission = pd.DataFrame({"id":test_ids.astype("int64"),"Personality":predictions})


submission.to_csv("submission(2).csv",index=False)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X_train,y_train)
rf.score(X_train,y_train)


preds=rf.predict(X_test)
print(classification_report(preds,y_test))


preds_proba = rf.predict_proba(X_test)
plt.figure(figsize=(8, 6))


sns.histplot(preds_proba, color='dodgerblue', kde=True, label='Predicted Probabilities', stat='density', bins=25, alpha=0.5)
sns.histplot(y_test, color='goldenrod', kde=True, label='Actual Labels', stat='density', bins=2, alpha=0.4)


plt.xlabel('Probability / Label')
plt.ylabel('Density')
plt.title('Distribution of Predicted Probabilities vs Actual Labels')
plt.legend()
plt.tight_layout()
plt.show()


y_proba = rf.predict_proba(X_test)[:, 1] 

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Compute AUC
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='dodgerblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()



from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stacked = StackingClassifier(
    estimators=[
        ('model1', model),
        ('model2',model),
    ],
    final_estimator=LogisticRegression()
)

stacked.fit(X_train, y_train)
y_pred = stacked.predict(X_test)



stacked.score(X_train,y_train)


print(classification_report(y_pred,y_test))


preds_proba = stacked.predict_proba(X_test)
plt.figure(figsize=(8, 6))


sns.histplot(preds_proba, color='dodgerblue', kde=True, label='Predicted Probabilities', stat='density', bins=25, alpha=0.5)
sns.histplot(y_test, color='goldenrod', kde=True, label='Actual Labels', stat='density', bins=2, alpha=0.4)


plt.xlabel('Probability / Label')
plt.ylabel('Density')
plt.title('Distribution of Predicted Probabilities vs Actual Labels')
plt.legend()
plt.tight_layout()
plt.show()


y_proba = stacked.predict_proba(X_test)[:, 1] 

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Compute AUC
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='dodgerblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()



predictions=stacked.predict(features_test)
predictions = pd.Series(predictions).apply(lambda x: "Extrovert" if x == 0 else "Introvert")
submission = pd.DataFrame({"id":test_ids.astype("int64"),"Personality":predictions})
submission.to_csv("submission(5).csv",index=False)


from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('model1', model),
        ('model2', rf)
    ],
    voting='soft'  
)

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)


voting_clf.score(X_train,y_train)


print(classification_report(y_pred,y_test))


preds_proba = voting_clf.predict_proba(X_test)
plt.figure(figsize=(8, 6))


sns.histplot(preds_proba, color='dodgerblue', kde=True, label='Predicted Probabilities', stat='density', bins=25, alpha=0.5)
sns.histplot(y_test, color='goldenrod', kde=True, label='Actual Labels', stat='density', bins=2, alpha=0.4)


plt.xlabel('Probability / Label')
plt.ylabel('Density')
plt.title('Distribution of Predicted Probabilities vs Actual Labels')
plt.legend()
plt.tight_layout()
plt.show()


y_proba = voting_clf.predict_proba(X_test)[:, 1] 

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Compute AUC
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='dodgerblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()



from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(scale_pos_weight=ratio)
lgbm.fit(X_train,y_train)
lgbm.score(X_train,y_train)


preds = lgbm.predict(X_test)
print(classification_report(preds,y_test))


preds_proba = lgbm.predict_proba(X_test)
plt.figure(figsize=(8, 6))


sns.histplot(preds_proba, color='dodgerblue', kde=True, label='Predicted Probabilities', stat='density', bins=25, alpha=0.5)
sns.histplot(y_test, color='goldenrod', kde=True, label='Actual Labels', stat='density', bins=2, alpha=0.4)


plt.xlabel('Probability / Label')
plt.ylabel('Density')
plt.title('Distribution of Predicted Probabilities vs Actual Labels')
plt.legend()
plt.tight_layout()
plt.show()


y_proba = lgbm.predict_proba(X_test)[:, 1] 

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Compute AUC
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='dodgerblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()



predictions=lgbm.predict(features_test)
predictions = pd.Series(predictions).apply(lambda x: "Extrovert" if x == 0 else "Introvert")
submission = pd.DataFrame({"id":test_ids.astype("int64"),"Personality":predictions})
submission.to_csv("submission(6).csv",index=False)


from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
stacking_classifier = StackingClassifier(
    estimators=[
        ('model1',rf),
        ('model2',lgbm),
        ('model3',model)
    ],
    final_estimator=GaussianNB()
)


stacking_classifier.fit(X_train,y_train)
stacking_classifier.score(X_train,y_train)


preds_proba = stacking_classifier.predict_proba(X_test)
plt.figure(figsize=(8, 6))


sns.histplot(preds_proba, color='dodgerblue', kde=True, label='Predicted Probabilities', stat='density', bins=25, alpha=0.5)
sns.histplot(y_test, color='goldenrod', kde=True, label='Actual Labels', stat='density', bins=2, alpha=0.4)


plt.xlabel('Probability / Label')
plt.ylabel('Density')
plt.title('Distribution of Predicted Probabilities vs Actual Labels')
plt.legend()
plt.tight_layout()
plt.show()


y_proba = stacking_classifier.predict_proba(X_test)[:, 1] 

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Compute AUC
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='dodgerblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()



predictions=stacking_classifier.predict(features_test)
predictions = pd.Series(predictions).apply(lambda x: "Extrovert" if x == 0 else "Introvert")
submission = pd.DataFrame({"id":test_ids.astype("int64"),"Personality":predictions})
submission.to_csv("submission(6).csv",index=False)


from catboost import CatBoostClassifier
cat = CatBoostClassifier(scale_pos_weight=ratio,verbose=0,)
cat.fit(X_train,y_train)
cat.score(X_train,y_train)


preds = cat.predict(X_test)
print(classification_report(preds,y_test))


predictions=cat.predict(features_test)
predictions = pd.Series(predictions).apply(lambda x: "Extrovert" if x == 0 else "Introvert")
submission = pd.DataFrame({"id":test_ids.astype("int64"),"Personality":predictions})
submission.to_csv("submission(8).csv",index=False)




