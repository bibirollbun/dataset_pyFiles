import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/santander-customer-satisfaction/train.csv')
test = pd.read_csv('/kaggle/input/santander-customer-satisfaction/test.csv')


train.head()


train.shape


train.info()


train.isnull().sum()


test.head()


test.shape


test.info()


test.isnull().sum()


train['TARGET'].value_counts()


sns.countplot(x=train['TARGET'], palette='muted')
plt.title('0:Satisfied vs 1:Unsatisfied Distribution');


#Constant columns
nunique = train.nunique()
constant_columns = nunique[nunique == 1].index.tolist()


len(constant_columns)


train.drop(columns=constant_columns, inplace=True)
test.drop(columns=constant_columns, inplace=True)


#Duplicate columns
cols_to_check = train.drop(columns=['ID', 'TARGET'], errors='ignore')
duplicate_mask = cols_to_check.T.duplicated()
dups_cols = cols_to_check.columns[duplicate_mask].tolist()
print(f"Duplicate: {len(set(dups_cols))}")


train.drop(columns=dups_cols, inplace=True)
test.drop(columns=dups_cols, inplace=True)


# Var15 = age?
sns.FacetGrid(train, hue="TARGET").map(sns.kdeplot, "var15").add_legend();


train['var3'].value_counts()


#Missing values = -999999

train['var3'].replace(-999999, 2, inplace=True)
test['var3'].replace(-999999, 2, inplace=True)


train['var3'].value_counts()


x = train.drop(['TARGET', 'ID'], axis=1)
y = train['TARGET']


X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


models = {"Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
          "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
          "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='auc', random_state=42),
          "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)}
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_prob)
    results.append([name, auc])   
    print(f"{name} -> AUC: {auc:.5f}")


#XGBoost
best_model = XGBClassifier(n_estimators=100,max_depth=3,learning_rate=0.1,eval_metric='auc',random_state=42)
best_model.fit(x, y)


test_df=test.drop(['ID'], axis=1)


submission_preds = best_model.predict_proba(test_df)[:, 1]


submission = pd.DataFrame({"ID": test['ID'],"TARGET": submission_preds})
submission.to_csv('submission.csv', index=False)


import joblib

joblib.dump(best_model, 'xgb_model.pkl')

preprocessing_data = {'constant_columns': constant_columns,'dups_cols': dups_cols}
joblib.dump(preprocessing_data, 'preprocessing_data.pkl')

