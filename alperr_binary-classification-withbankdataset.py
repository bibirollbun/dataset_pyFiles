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


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


train.isnull().sum()


test.isnull().sum()


test.info()


train.info()


train.describe()


train.head()


categoric_features = train.select_dtypes(include='object').columns
for feature in categoric_features:
    print(train[feature].value_counts())
    print("\n")


categoric_features = train.select_dtypes(include='object').columns
for feature in categoric_features:
    print(test[feature].value_counts())
    print("\n")


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
train_encoded = train.copy()
test_encoded = test.copy()



#Binary Encoding for default housing loan
binary_cols = ['default', 'housing', 'loan']
encoders = {}

for col in binary_cols:
    le = LabelEncoder()
    train_encoded[col] = le.fit_transform(train[col])
    test_encoded[col]  = le.transform(test[col])
    encoders[col] = le   # her sütunun encoder'ını sakla

#Ordinal Encoding for education
education_order = {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': -1}
train_encoded['education'] = train['education'].map(education_order)
test_encoded['education'] = test['education'].map(education_order)

#One-Hot Encoding for marital, contact, poutcome 
ohe_cols = ['marital', 'contact', 'poutcome','job']
train_encoded = pd.get_dummies(train_encoded, columns=ohe_cols, prefix=ohe_cols)
test_encoded = pd.get_dummies(test_encoded, columns=ohe_cols, prefix=ohe_cols)

#Cyclic Encoding for month
month_mapping = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

train_encoded['month_numeric'] = train['month'].map(month_mapping)
test_encoded['month_numeric']  = test['month'].map(month_mapping)

train_encoded['month_sin'] = np.sin(2 * np.pi * train_encoded['month_numeric']/12)
train_encoded['month_cos'] = np.cos(2 * np.pi * train_encoded['month_numeric']/12)

test_encoded['month_sin'] = np.sin(2 * np.pi * test_encoded['month_numeric']/12)
test_encoded['month_cos'] = np.cos(2 * np.pi * test_encoded['month_numeric']/12)

train_encoded.drop('month_numeric', axis=1, inplace=True)
test_encoded.drop('month_numeric', axis=1, inplace=True)
train_encoded.drop('month', axis=1, inplace=True)
test_encoded.drop('month', axis=1, inplace=True)





bool_features = train_encoded.select_dtypes(include='bool').columns
for feature in bool_features:
    train_encoded[feature] = train_encoded[feature].astype(int)
    test_encoded[feature] = test_encoded[feature].astype(int)


train_encoded.info()


test_encoded.info()


train_encoded.head()


test_encoded.head()


train_encoded.drop('id', axis=1, inplace=True)
test_encoded.drop('id', axis=1, inplace=True)


from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve
from scipy.stats import randint, uniform


# ---- 1) X, y ----
X = train_encoded.drop('y', axis=1)
y = train_encoded['y']


# ---- 2) Train/Test ayır ----
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.02, random_state=42, stratify=y
)

# ---- 2b) Test'in bir kısmını validation olarak ayır ----
X_val, X_test_final, y_val, y_test_final = train_test_split(
    X_te, y_te, test_size=0.5, random_state=42, stratify=y_te
)

print("Train:", X_tr.shape, "Validation:", X_val.shape, "Test:", X_test_final.shape)







import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report, confusion_matrix
)





callbacks = [
    lgb.early_stopping(stopping_rounds=700, verbose=True),
    lgb.log_evaluation(period=10),
    lgb.reset_parameter(learning_rate=lambda it: max(0.001, 0.02*(0.9995**it)))
]

# ---- 4) Nihai model ----
model = LGBMClassifier(
    objective="binary",
    random_state=42,
    n_estimators=25000,      # yüksek tut; early stopping ile durur
    n_jobs=-1,
    num_leaves=63,
    max_depth=8,
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbose=1
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr,y_tr),(X_val, y_val)],
    eval_names=["train","val"],
    eval_metric="auc",
    callbacks=callbacks
)

print("Best iteration:", model.best_iteration_)

# ---- 5) Test performansı ----
y_pred_proba = model.predict_proba(X_test_final)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

print("Test ROC AUC :", roc_auc_score(y_test_final, y_pred_proba))
print("Accuracy     :", accuracy_score(y_test_final, y_pred))
print("\nClassification report:\n", classification_report(y_test_final, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test_final, y_pred))
from sklearn.model_selection import cross_val_score
#Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== TOP 10 FEATURE IMPORTANCE ===")
print(feature_importance.head(10))





test_ids = test['id'].copy()
predictions = model.predict_proba(test_encoded)[:, 1]  #  gerçekleşme olasılıkları
# Submission dosyasını oluştur
submission = pd.DataFrame({
        'id': test_ids,
        'prediction': predictions
    })
    
# ID'ye göre sırala (orijinal sıraya göre)
submission = submission.sort_values('id')
submission.to_csv('submission.csv', index=False)


submission.head()

