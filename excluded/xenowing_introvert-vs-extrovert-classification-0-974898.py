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


!pip install -U scikit-learn xgboost imbalanced-learn lightgbm shap


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_data=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train_data


test_data


train_data['Time_spent_Alone'].value_counts()


train_data['Stage_fear'].value_counts()


train_data['Social_event_attendance'].value_counts()


train_data['Going_outside'].value_counts()


train_data['Drained_after_socializing'].value_counts()


train_data["Friends_circle_size"].value_counts()	


train_data['Post_frequency'].value_counts()


train_data['Personality'].value_counts()


train_data.shape


train_data.dropna().shape


train_data=train_data.drop('id',axis=1)
test_data=test_data.drop('id',axis=1)


cols=train_data.columns
cols


categorical_cols=[col for col in cols if train_data[col].dtype=='object']
numerical_cols=[col for col in cols if train_data[col].dtype!='object']

categorical_cols,numerical_cols


train_data[categorical_cols]


train_data['Stage_fear']=train_data['Stage_fear'].replace('Yes',1)
train_data['Stage_fear']=train_data['Stage_fear'].replace('No',0)
train_data['Drained_after_socializing']=train_data['Drained_after_socializing'].replace('No',0)
train_data['Drained_after_socializing']=train_data['Drained_after_socializing'].replace('Yes',1)
train_data['Personality']=train_data['Personality'].replace('Extrovert',0)
train_data['Personality']=train_data['Personality'].replace('Introvert',1)


test_data['Stage_fear']=test_data['Stage_fear'].replace('Yes',1)
test_data['Stage_fear']=test_data['Stage_fear'].replace('No',0)
test_data['Drained_after_socializing']=test_data['Drained_after_socializing'].replace('No',0)
test_data['Drained_after_socializing']=test_data['Drained_after_socializing'].replace('Yes',1)


train_data['Stage_fear']=train_data['Stage_fear'].fillna(0)
train_data['Drained_after_socializing']=train_data['Drained_after_socializing'].fillna(0)
train_data['Personality']=train_data['Personality'].fillna(0)


test_data['Stage_fear']=test_data['Stage_fear'].fillna(0)
test_data['Drained_after_socializing']=test_data['Drained_after_socializing'].fillna(0)


train_data


numerical_cols


for i in numerical_cols:
    train_data[i]=train_data[i].fillna(0)
    test_data[i]=test_data[i].fillna(0)


train_data


test_data


X=train_data.drop('Personality',axis=1)
y=train_data['Personality']
X


y


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_test.shape,X_train.shape





from collections import Counter

counter=Counter(y_train)
scale_pos_weight=counter[0]/counter[1]
scale_pos_weight


from xgboost import XGBClassifier

xg_model=XGBClassifier(scale_pos_weight=3,use_label_encoder=False,eval_metric='logloss')
xg_model.fit(X_train,y_train)


from sklearn.metrics import classification_report

y_pred=xg_model.predict(X_test)

xgb_result=classification_report(y_pred,y_test)
print(xgb_result)


y_test_xg=xg_model.predict(test_data)
submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
submission['Personality']=y_test_xg
submission['Personality']=submission['Personality'].replace(1,'Introvert')
submission['Personality']=submission['Personality'].replace(0,'Extrovert')
submission.to_csv("Submission_2.csv",index=False)


from imblearn.over_sampling import SMOTE

smote=SMOTE(random_state=42)

X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)

print("Before SMOTE",y_train.value_counts())
print("After SMOTE",y_train_smote.value_counts())


smote_model=XGBClassifier(scale_pos_weight=1,use_label_encoder=False,eval_metric='logloss')
smote_model.fit(X_train_smote,y_train_smote)


y_pred_smote=smote_model.predict(X_test)
smote_result=classification_report(y_pred_smote,y_test)
print(smote_result)


param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],  # Controls complexity (regularization)
    'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight (hessian) needed in a child
}


from sklearn.model_selection import GridSearchCV

xgb = XGBClassifier(eval_metric='logloss')

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,
    scoring='f1',  # or 'roc_auc', 'accuracy', etc.
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)


grid_model=grid_search.best_estimator_
y_pred_grid=grid_model.predict(X_test)
print("Best Params:",grid_search.best_params_)
grid_result=classification_report(y_pred_grid,y_test)
print(grid_result)


y_test_grid=grid_model.predict(test_data)
submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
submission['Personality']=y_test_grid
submission['Personality']=submission['Personality'].replace(1,'Introvert')
submission['Personality']=submission['Personality'].replace(0,'Extrovert')
submission.to_csv("Submission_Grid_SearchCV.csv",index=False)


from xgboost import plot_importance

feature_model=XGBClassifier(eval_metric='logloss')
feature_model.fit(X_train,y_train)

#PLot top Features
plot_importance(feature_model,max_num_features=10)
plt.show()


X_train_drop=X_train.drop(['Stage_fear','Drained_after_socializing'],axis=1)
X_test_drop=X_test.drop(['Stage_fear','Drained_after_socializing'],axis=1)
feature_model.fit(X_train_drop,y_train)
y_pred_feature=feature_model.predict(X_test_drop)
feature_result=classification_report(y_pred_feature,y_test)
print(feature_result)


test_data_drop=test_data.drop(['Stage_fear','Drained_after_socializing'],axis=1)
y_test_feature=feature_model.predict(test_data_drop)
submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
submission['Personality']=y_test_feature
submission['Personality']=submission['Personality'].replace(1,'Introvert')
submission['Personality']=submission['Personality'].replace(0,'Extrovert')
submission.to_csv("Submission_Feature_importance.csv",index=False)


from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

cv_model=XGBClassifier(eval_metric='logloss',
                      verbosity=0)
skf=StratifiedKFold(n_splits=5,
                   shuffle=True,
                   random_state=42)


f1_scores=[]
for fold,(train_idx,val_idx) in enumerate(skf.split(X,y)):
    print(f"Fold:{fold+1}")
    
    X_train,X_val=X.iloc[train_idx],X.iloc[val_idx]
    y_train,y_val=y.iloc[train_idx],y.iloc[val_idx]
    
    cv_model.fit(X_train,y_train)
    y_pred=cv_model.predict(X_val)
    
    score=f1_score(y_val,y_pred,average='weighted')
    print(f"F1:{score:.4f}")
    
    f1_scores.append(score)
    print(classification_report(y_val,y_pred))
print(f"Average F1 Score across folds:{np.mean(f1_scores):.4f}")


y_test_cv=cv_model.predict(test_data)
submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
submission['Personality']=y_test_cv
submission['Personality']=submission['Personality'].replace(1,'Introvert')
submission['Personality']=submission['Personality'].replace(0,'Extrovert')
submission.to_csv("Submission_Cross_Validation.csv",index=False)


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

base_models=[
        ('xgb',XGBClassifier(eval_metric='logloss',verbosity=0)),
        ('lgbm',LGBMClassifier())
]

meta_model=LogisticRegression()

stacking_clf=StackingClassifier(estimators=base_models,
                               final_estimator=meta_model,
                               cv=3,
                               stack_method='predict_proba')

stacking_clf.fit(X_train,y_train)
y_pred=stacking_clf.predict(X_test)
stacking_result=classification_report(y_pred,y_test)
print(stacking_result)


y_test_stack=stacking_clf.predict(test_data)
submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
submission['Personality']=y_test_stack
submission['Personality']=submission['Personality'].replace(1,'Introvert')
submission['Personality']=submission['Personality'].replace(0,'Extrovert')
submission.to_csv("Submission_Ensemble_Method.csv",index=False)


import shap

explainer=shap.TreeExplainer(xg_model)
shap_values=explainer.shap_values(X_test)
shap.summary_plot(shap_values,X_test)


X_train_smote_final=X_train_smote.drop(['Stage_fear','Drained_after_socializing'],axis=1)
X_test_final=X_test.drop(['Stage_fear','Drained_after_socializing'],axis=1)
test_data_final=test_data.drop(['Stage_fear','Drained_after_socializing'],axis=1)
X_train_smote_final


## Base Models
xgb=XGBClassifier(eval_metric='logloss',
                  verbosity=0,
                  random_state=42,
                  colsample_bytree=0.8,
                  gamma=1,
                  learning_rate=0.1,
                  max_depth=3,
                  min_child_weight=1,
                  n_estimators=500,
                 subsample=0.6 
                 )
lgbm=LGBMClassifier(random_state=42)
meta_model=LogisticRegression(max_iter=1000,solver='liblinear')

stacking_clf_final=StackingClassifier(
    estimators=[('xgb',xgb),('lgbm',lgbm)],
    final_estimator=meta_model,
    stack_method='predict_proba',
    cv=3
)

param_grid={
    'lgbm__n_estimators':[100,200,300,400,500],
    'lgbm__learning_rate':[0.1,0.01,0.001,0.0001],
    'lgbm__max_depth':[3,5,7,9]
}

grid=GridSearchCV(estimator=stacking_clf_final,
                 param_grid=param_grid,
                 scoring='f1_weighted',
                 cv=3,
                 verbose=0,
                 n_jobs=-1)
grid.fit(X_train_smote_final,y_train_smote)


final_model=grid.best_estimator_
y_pred_final=final_model.predict(X_test_final)
final_result=classification_report(y_pred_final,y_test)
print(final_result)


y_test_final=final_model.predict(test_data_final)
submission['Personality']=y_test_final
submission['Personality']=submission['Personality'].replace(1,'Introvert')
submission['Personality']=submission['Personality'].replace(0,'Extrovert')
submission.to_csv("final_submission.csv",index=False)

