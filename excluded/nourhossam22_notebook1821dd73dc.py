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


import xgboost as xgb



train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


train.head()


train.info()


train.drop(columns=["id"], axis=1, inplace=True)
train.head()


train.duplicated().sum()


train.describe()


train['marital'].value_counts()
# train['poutcome'].value_counts()


train["housing"] = train["housing"].map({"yes": 1, "no": 0})
train["loan"] = train["loan"].map({"yes": 1, "no": 0})
train["default"] = train["default"].map({"yes": 1, "no": 0})
month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5,
             'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

train['month'] = train['month'].map(month_map)
train[["housing", "loan","default","month"]].head()



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


X = train.drop(columns=['y'],axis=1)
y = train['y']
X_train, X_vad, y_train, y_vad = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)



from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
categorical_cols = ["education", "poutcome", "housing", "loan", "contact"]
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
X_vad[categorical_cols] = encoder.transform(X_vad[categorical_cols])


cat_cols = ["job", "marital"]  
encoders = OneHotEncoder(drop="first", sparse_output=False)
train_encoded = encoders.fit_transform(X_train[cat_cols])
vad_encoded   = encoders.transform(X_vad[cat_cols])
encoded_cols = encoders.get_feature_names_out(cat_cols)
train_encoded = pd.DataFrame(train_encoded, columns=encoded_cols, index=X_train.index)
vad_encoded   = pd.DataFrame(vad_encoded, columns=encoded_cols, index=X_vad.index)
X_train = pd.concat([X_train.drop(cat_cols, axis=1), train_encoded], axis=1)
X_vad   = pd.concat([X_vad.drop(cat_cols, axis=1), vad_encoded], axis=1)

X_train.head()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_vad[num_cols] = scaler.transform(X_vad[num_cols])
final_features = X_train.columns



# log = LogisticRegression(max_iter=1000)
# log.fit(X_train, y_train)


xgb_model = xgb.XGBClassifier(
    n_estimators=500,       
    max_depth=5,           
    learning_rate=0.1,    
    subsample=0.8,         
    colsample_bytree=0.8,  
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'   
)

xgb_model.fit(X_train, y_train)

y_val_pred_proba = xgb_model.predict_proba(X_vad)[:,1]  
y_val_pred = xgb_model.predict(X_vad)                  


auc_score = roc_auc_score(y_vad, y_val_pred_proba)
print("Validation ROC AUC:", auc_score)


# y_pred_train = log.predict_proba(X_vad)[:, 1]


# acc_train = roc_auc_score(y_vad, y_pred_train)
# acc_train


ID=test['id']
ID


test.info()


test.drop(columns=["id"],axis=1,inplace=True)


test["housing"] = test["housing"].map({"yes": 1, "no": 0})
test["loan"] = test["loan"].map({"yes": 1, "no": 0})
test["default"] = test["default"].map({"yes": 1, "no": 0})
month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5,
             'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

test['month'] = test['month'].map(month_map)
test[["housing", "loan","default","month"]].head()



test_encoded_ord = pd.DataFrame(
    encoder.transform(test[categorical_cols]),
    columns=categorical_cols,
    index=test.index
)
test = test.drop(columns=categorical_cols).join(test_encoded_ord)

encoded = encoders.transform(test[cat_cols])
test_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=test.index)

test = pd.concat([test.drop(cat_cols, axis=1), test_encoded], axis=1)


X_test_scaled = test.copy()
X_test_scaled[num_cols] = scaler.transform(X_test_scaled[num_cols])

X_test_scaled = X_test_scaled[final_features]



# prediction = log.predict_proba(X_test_scaled)[:, 1] 
y_test_proba = xgb_model.predict_proba(X_test_scaled)[:,1]      


subm=pd.DataFrame({"id":ID
                  , "prediction":y_test_proba})
subm


subm.to_csv("submission.csv", index=False)


