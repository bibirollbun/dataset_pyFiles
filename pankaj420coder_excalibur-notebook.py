import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
df=pd.read_csv("/kaggle/input/bank-full/bank-full.csv", sep=';')


X=train.drop(columns=['y'])
Y=train['y']


train.info()


train.isnull().sum()


train.duplicated().sum()




plt.figure(figsize=(8,5))
sns.countplot(x=train['y'])
plt.show()

## kind of imbalanced dataset as 0's are majority-----------OBSERVATION


train.columns


## seperating numerical and categorical features for better data visaulisation

cat_column=[col for col in train.columns if train[col].dtype=='object']
num_column=[col for col in train.columns if train[col].dtype=='int64']


print(f"Numerical Columns:{num_column}\n\nCategorical Columns:{cat_column}")


import warnings
warnings.filterwarnings('ignore')



## histplot and boxplot for num_column in loop

for col in num_column:
    plt.figure(figsize=(12,5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(train[col], kde=True)
    plt.title(f'Histogram of {col}')
    
    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot of {col}')
    
    plt.tight_layout()
    plt.show()


## countplot for cat_column in loop
for col in cat_column:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=train[col],hue=train['y'])
    plt.title(f'Countplot of {col}')
    plt.xticks(rotation=45)
    plt.show()


## heatmap and correlation check

plt.figure(figsize=(15, 10))
sns.heatmap(train.select_dtypes(include='number').corr(), annot=True)
plt.show()


cat_column , num_column


## checking the values of each categorical column

for i in cat_column:
    print(f"\n\nColumn: {i}")   
    print(train[i].value_counts())



train.info()


for col in ['default', 'housing', 'loan']:
    X[col] = X[col].map({"yes": 1, "no": 0})   



for col in ['default', 'housing', 'loan']:
    test[col] = test[col].map({"yes": 1, "no": 0})   



X.info()


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


## Models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



## train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


print(X_train.shape)
print(X_test.shape)



num = X.select_dtypes(include=['int64']).columns
cat = X.select_dtypes(include=['object']).columns


print(f"Numerical Columns:{num}\n\nCategorical Columns:{cat}")


## one hot encoder for rest object columns and standardscaler for int columns both under Column Transform

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])



pipeline= Pipeline([
    ("preprocessor",preprocessor),
    ("Random forest",RandomForestClassifier(n_estimators=100,
    max_depth=12,         
    max_features="sqrt",  
    n_jobs=-1,            
    random_state=42)),
])


pipeline.fit(X_train,y_train)


y_pred_proba= pipeline.predict_proba(X_test)


y_pred_proba


y_pred= pipeline.predict(X_test)


print("Performance Metrics(RandomForest):" )
print("--------"*4)

print("Accuracy Score",accuracy_score(y_test,y_pred))
print("ROC AUC Score" ,roc_auc_score(y_test,y_pred_proba[:,1]))
print("F1 Score",f1_score(y_test,y_pred))
print("Precision Score",precision_score(y_test,y_pred))
print("Recall Score",recall_score(y_test,y_pred))


pipeline1= Pipeline([
    ("preprocessor",preprocessor),
    ("XGB",XGBClassifier(n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric="logloss")),
])


pipeline1.fit(X_train,y_train)


y_pred1_proba= pipeline1.predict_proba(X_test)


y_pred1=pipeline1.predict(X_test)


print("Performance Metrics(XGBoost):" )
print("--------"*4)

print("Accuracy Score",accuracy_score(y_test,y_pred1))
print("ROC AUC Score" ,roc_auc_score(y_test,y_pred1_proba[:,1]))
print("F1 Score",f1_score(y_test,y_pred1))
print("Precision Score",precision_score(y_test,y_pred1))
print("Recall Score",recall_score(y_test,y_pred1))


pipeline2= Pipeline([
    ("preprocessor",preprocessor),
    ("LGBM",LGBMClassifier(n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42,
    n_jobs=-1)),
]) 


pipeline2.fit(X_train,y_train)


y_pred2=pipeline2.predict(X_test)


y_pred2_proba=pipeline2.predict_proba(X_test)


y_pred2=pipeline2.predict(X_test)


print("Performance Metrics(LGBM)")
print("--------"*6)

print("Accuracy Score",accuracy_score(y_test,y_pred2))
print("ROC AUC Score" ,roc_auc_score(y_test,y_pred2_proba[:,1]))
print("F1 Score",f1_score(y_test,y_pred2))
print("Precision Score",precision_score(y_test,y_pred2))
print("Recall Score",recall_score(y_test,y_pred2))


from sklearn.metrics import roc_curve,auc


## fpr= false positive rate
## tpr= true poisitve rate


# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba[:,1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

# XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred1_proba[:,1])
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# LightGBM
fpr_lgbm, tpr_lgbm, _ = roc_curve(y_test, y_pred2_proba[:,1])
roc_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)




plt.figure(figsize=(8,6))

plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot(fpr_lgbm, tpr_lgbm, label=f'LightGBM (AUC = {roc_auc_lgbm:.2f})')

plt.plot([0,1], [0,1], 'k--')  #dummy model
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(True)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()



pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression())
])

# Parameter grid for multiple models
param_grid = [
    {
        "clf": [LogisticRegression(max_iter=500,random_state=42)],
        'clf__penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga']
        
    },
    {
       
    },
    {
        "clf": [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [100,300, 500],
        'clf__max_features': ['auto', 'sqrt', 'log2'],
        'clf__max_depth' : [4,6,8],
        'clf__criterion' :['gini', 'entropy']
        
    },
    {
        "clf": [XGBClassifier(use_label_encoder=False, eval_metric='logloss',random_state=42)],
        'clf__n_estimators': [100, 200, 300, 400, 500],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'clf__max_depth': [3, 5, 6],
        'clf__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'clf__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
        
    },
    {
        "clf": [LGBMClassifier(random_state=42)],
        'clf__n_estimators': [100, 200, 300, 400, 500],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'clf__max_depth': [3, 4, 5, 6, 7],
        'clf__num_leaves': [20, 30, 40, 50, 60],
        'clf__boosting_type': ['gbdt', 'dart', 'goss']
        
    }
]


grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2)



grid.fit(X_train, y_train)


test_pred=pipeline2.predict(test)


test_pred


submission_df = pd.DataFrame({
    'id': test['id'], 
    'Prediction': test_pred
})


submission_df


submission_df.to_csv('Submission.csv',index=False)


df1=pd.read_csv('/kaggle/working/Submission.csv')


df1




