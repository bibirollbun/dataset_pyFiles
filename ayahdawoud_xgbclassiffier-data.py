import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn .preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train


test


train .isnull().sum()


test.isnull().sum()


train.info()


test.info()


train.describe()


test.describe()


for col in train.columns:
    print(f"\nColumn: {col}")
    
    print(train[col].value_counts())


num_cols=train.select_dtypes(exclude=['object']).columns.difference(['y'])
num_cols


cate_cols=train.select_dtypes(include=['object']).columns
cate_cols


num_col=test.select_dtypes(exclude=['object']).columns
num_col


cate_col=test.select_dtypes(include=['object']).columns
cate_col


train


plt.figure(figsize=(8,8))
sns.barplot(data=train,x='y',y='age',hue='marital',palette="flare")




for col in num_cols:
    if col!='y':
    
       plt.figure(figsize=(8,8))
       sns.histplot(data=train,x=col,kde=True,hue='y',stat='density')


sns.violinplot(data=train,x='age',color='#9b59b6')


sns.violinplot(data=train,x='balance')


sns.violinplot(data=train,x='campaign')


le=LabelEncoder()
for col in cate_cols:
    train[col]=le.fit_transform(train[col])


le=LabelEncoder()
for col in cate_col:
    test[col]=le.fit_transform(test[col])


train


test


correlation=train.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation,annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=14)
plt.show()


X=train.drop('y',axis=1)
X


y=train['y']
y


from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


X=train.drop('y',axis=1)
x_test=test.copy()
y=train['y']


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(x_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"== Fold {fold+1} ==")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        use_label_encoder=False,   
        eval_metric='auc'
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,   
        verbose=50                  #
    )
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(x_test)[:, 1] / kf.n_splits

# Local validation AUC
auc_score = roc_auc_score(y, oof_preds)
print(f"CV ROC AUC: {auc_score:.4f}")
#submission


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
x_train


x_test


y_train


y_test


import lightgbm as lgb
clf=lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
     max_depth=-1,
    random_state=42
)
clf.fit(x_train,y_train)


y_pred=clf.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score, classification_report
print('accuracy',accuracy_score(y_test,y_pred))
print(print(classification_report(y_test, y_pred)))





submission4=pd.DataFrame({'id':test['id'],'y':test_preds})
submission4.to_csv("submission_xgb.csv", index=False)

    
 



subm=pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
subm


submission4.head()

