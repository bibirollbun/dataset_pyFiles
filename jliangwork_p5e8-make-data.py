!pip install -U scikit-learn


pip install --upgrade category-encoders


import gc


# import stuff

import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split

# Feature Processing

from sklearn.preprocessing import TargetEncoder # what it sounds like
from sklearn.preprocessing import OrdinalEncoder # what it sounds like
from sklearn.preprocessing import StandardScaler # what it sounds like
from category_encoders import CountEncoder


# Date

import datetime # help me process dates
import calendar # what it sounds like


# download datasets

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
original = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep = ";")


## add date 

def process_date(row):

  month_dict = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}


  month = month_dict[row['month']]
  day = row['day']

  days_in_month = calendar.monthrange(2023, month)[1]
  if day > days_in_month:
    day = days_in_month

  newyear = datetime.date(2023, 1, 1).toordinal()
  date = datetime.date(2023, month, day)

  return date.toordinal() - newyear

train['date'] = train.apply(process_date, axis = 1)
test['date'] = test.apply(process_date, axis = 1)
original['date'] = original.apply(process_date, axis = 1)


# make sure order is correct

train_features = list(train.columns)
train_features.remove('id')

train = train[train_features]
original = original[train_features]


X = train.drop(columns = ['y'])
y = train['y'].copy()

X_test = test.drop(columns = ['id'])
X_test = X_test[list(X.columns)]

X_original = original.drop(columns = ['y'])
X_original = X_original[list(X.columns)]

y_original = original['y'].map({"no" : 0.0, "yes" : 1.0})


X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify = y, random_state = 1
)


del X, y
gc.collect()


X_train_processed = X_train.copy()
X_val_processed = X_val.copy()
X_test_processed = X_test.copy()
X_original_processed = X_original.copy()


del X_train, X_val, X_test

gc.collect()


from itertools import combinations


quantitative_features = [feature for feature in X_train_processed.columns if X_train_processed[feature].dtype == "int64"]

def create_features(X, quant_features):

    # features created are quantitative features as qual and then interaction of all quals
    
    for feature in quant_features:

        name = f"asqual_{feature}"
        X[name] = X[feature].astype(object)

    qualitative_features = [feature for feature in X.columns if X[feature].dtype == "object"]

    interactions = combinations(qualitative_features, 2)
    
    for a, b in interactions:

        name = f"interact_{a}_{b}"
        X[name] = X[a].astype(str) + '_' + X[b].astype(str)

    return X

X_train_processed = create_features(X_train_processed, quantitative_features)
X_val_processed = create_features(X_val_processed, quantitative_features)
X_test_processed = create_features(X_test_processed, quantitative_features)
X_original_processed = create_features(X_original_processed, quantitative_features)


X_train_processed.to_csv("X_train_processed", index = False)
X_val_processed.to_csv("X_val_processed", index = False)
X_test_processed.to_csv("X_test_processed", index = False)
X_original_processed.to_csv("X_original_processed", index = False)


X_train_A = pd.concat([X_original_processed, X_train_processed], axis = 0)
y_train_A = pd.concat([y_original, y_train], axis = 0)

X_train_B = X_train_processed.copy()
y_train_B = y_train.copy()

del X_train_processed
gc.collect()

X_original_A = X_original_processed.copy()[list(X_train_A.columns)]
X_original_B = X_original_processed.copy()[list(X_train_B.columns)]

y_original_A = y_original.copy()
y_original_B = y_original.copy()

del X_original_processed
gc.collect()


X_val_A = X_val_processed.copy()[list(X_train_A.columns)]
X_val_B = X_val_processed.copy()[list(X_train_A.columns)]

y_val_A = y_val.copy()
y_val_B = y_val.copy()

del X_val_processed
gc.collect()

X_test_A = X_test_processed.copy()[list(X_train_A.columns)]
X_test_B = X_test_processed.copy()[list(X_train_B.columns)]

del X_test_processed
gc.collect()


# add counts for features


for feature in [feature for feature in X_train_A.columns if X_train_A[feature].dtype == "object"]:
    
    counter = CountEncoder(handle_unknown = -1)
    name = f"count_{feature}"
    X_train_A[name] = counter.fit_transform(X_train_A[feature]).astype(float)
    X_val_A[name] = counter.transform(X_val_A[feature]).astype(float)
    X_test_A[name] = counter.transform(X_test_A[feature]).astype(float)


from sklearn.compose import ColumnTransformer


# now we target encode all the features (keep the default of replacing unseen values w/ mean)

target_encoded_features = [feature for feature in X_train_A.columns if X_train_A[feature].dtype == "object"]

process_cols = ColumnTransformer(
        transformers = [
            ('target', TargetEncoder(), target_encoded_features)
        ],
        verbose_feature_names_out = True,
        remainder = 'passthrough'
    )
    
X_train_A = process_cols.fit_transform(X_train_A, y_train_A)
X_train_A = pd.DataFrame(X_train_A, columns = process_cols.get_feature_names_out())

X_test_A = process_cols.transform(X_test_A)
X_test_A = pd.DataFrame(X_test_A, columns = process_cols.get_feature_names_out())

X_val_A = process_cols.transform(X_val_A)
X_val_A = pd.DataFrame(X_val_A, columns = process_cols.get_feature_names_out())


X_train_A.info()


# test the results using xgboost
# if nothing was messed up, the validation score should be decent

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

xgb_model = XGBClassifier(use_label_encoder=False,
                          n_estimators = 100,
                          learning_rate = 0.1,
                          eval_metric=['auc', 'error'],
                          objective = 'binary:logistic',
                          #early_stopping_rounds = 100,
                          max_depth = 0,
                          max_leaves = 41,
                          random_state = 1)

xgb_model.fit(X_train_A, y_train_A,
                  eval_set=[(X_train_A, y_train_A,), 
                            (X_val_A, y_val_A)],
                  verbose=10
             )
    
predictions = xgb_model.predict_proba(X_val_A)
roc_auc = roc_auc_score(y_val_A, predictions[:, 1])
    
print(f"ROC AUC Score: {roc_auc}")


# save X_train_A, y_train_A, X_val_A, y_val_A, X_test_A


X_train_A.to_csv("X_train_A.csv", index=False)
del X_train_A

y_train_A.to_csv("y_train_A.csv", index=False)
del y_train_A

X_val_A.to_csv("X_val_A.csv", index=False)
del X_val_A

y_val_A.to_csv("y_val_A.csv", index=False)
del y_val_A

X_test_A.to_csv("X_test_A.csv", index=False)
del X_test_A

gc.collect()


for feature in [feature for feature in X_train_B.columns if X_train_B[feature].dtype == "object"]:
    
    counter = CountEncoder(handle_unknown = -1)
    name = f"count_{feature}"
    X_train_B[name] = counter.fit_transform(X_train_B[feature]).astype(float)
    X_val_B[name] = counter.transform(X_val_B[feature]).astype(float)
    X_test_B[name] = counter.transform(X_test_B[feature]).astype(float)


for feature in [feature for feature in X_train_B.columns if X_train_B[feature].dtype == "object"]:
    
    encoder = TargetEncoder()
    name = f"origtarget_{feature}"
    encoder.fit(X_original_B[[feature]], y_original)
    
    X_train_B[name] = encoder.transform(X_train_B[[feature]]).astype(float).ravel()
    X_val_B[name] = encoder.transform(X_val_B[[feature]]).astype(float).ravel()
    X_test_B[name] = encoder.transform(X_test_B[[feature]]).astype(float).ravel()


# now we target encode all the features (keep the default of replacing unseen values w/ mean)

target_encoded_features = [feature for feature in X_train_B.columns if X_train_B[feature].dtype == "object"]

process_cols = ColumnTransformer(
        transformers = [
            ('traintarget', TargetEncoder(), target_encoded_features)
        ],
        verbose_feature_names_out = True,
        remainder = 'passthrough'
    )
    
X_train_B = process_cols.fit_transform(X_train_B, y_train_B)
X_train_B = pd.DataFrame(X_train_B, columns = process_cols.get_feature_names_out())

X_test_B = process_cols.transform(X_test_B)
X_test_B = pd.DataFrame(X_test_B, columns = process_cols.get_feature_names_out())

X_val_B = process_cols.transform(X_val_B)
X_val_B = pd.DataFrame(X_val_B, columns = process_cols.get_feature_names_out())


X_train_B.head()


# test the results using xgboost
# if nothing was messed up, the validation score should be decent

#from xgboost import XGBClassifier
#from sklearn.metrics import roc_auc_score

xgb_model = XGBClassifier(use_label_encoder=False,
                          n_estimators = 100,
                          learning_rate = 0.1,
                          eval_metric=['auc', 'error'],
                          objective = 'binary:logistic',
                          #early_stopping_rounds = 100,
                          max_depth = 0,
                          max_leaves = 41,
                          random_state = 1)

xgb_model.fit(X_train_B, y_train_B,
                  eval_set=[(X_train_B, y_train_B,), 
                            (X_val_B, y_val_B)],
                  verbose=10
             )
    
predictions = xgb_model.predict_proba(X_val_B)
roc_auc = roc_auc_score(y_val_B, predictions[:, 1])
    
print(f"ROC AUC Score: {roc_auc}")


# save X_train_B, y_train_B, X_val_B, y_val_B, X_test_B, y_original

X_train_B.to_csv("X_train_B.csv", index=False)
del X_train_B

y_train_B.to_csv("y_train_B.csv", index=False)
del y_train_B

X_val_B.to_csv("X_val_B.csv", index=False)
del X_val_B

y_val_B.to_csv("y_val_B.csv", index=False)
del y_val_B

X_test_B.to_csv("X_test_B.csv", index=False)
del X_test_B

y_original.to_csv("y_original.csv", index=False)
del y_original

gc.collect()

