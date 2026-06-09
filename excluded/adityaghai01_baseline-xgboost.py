import os

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")


import numpy as np
import pandas as pd


train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original_data = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
submission_data = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


print("train_data shape :",train_data.shape)
print("test_data shape :",test_data.shape)
print("original_data shape :",original_data.shape)
print("submission_data shape :",submission_data.shape)


train_data.head()


train_data = train_data.drop("id", axis=1)
test_data = test_data.drop("id", axis=1)
# cheat: merge original and competition dataset :D
train_data_new = pd.concat([train_data, original_data], ignore_index=True)
train_data_new = train_data_new.drop_duplicates()
print("shape of the data :",train_data_new.shape)


def pd_one_hot(dataset, col):
    one_hot = pd.get_dummies(dataset[col])
    dataset = dataset.drop(col,axis = 1)
    dataset = dataset.join(one_hot)
    return dataset


def transform_dataset(dataset, cat_cols: list[str]):
    # categorical to onehot
    for col in cat_cols:
        dataset = pd_one_hot(dataset, col)
    
    # normalize (no need for decision trees)
    
    return dataset


num_cols_train = list(train_data_new.select_dtypes(include='number')\
    .columns\
    .difference(['Fertilizer Name']))
cat_cols_train = list(train_data_new.select_dtypes(exclude='number')\
    .columns\
    .difference(['Fertilizer Name']))
num_cols_test = list(test_data.select_dtypes(include='number')\
    .columns)
cat_cols_test = list(test_data.select_dtypes(exclude='number')\
    .columns)

num_cols_train, cat_cols_train, num_cols_test, cat_cols_test


train_data_copy = train_data_new.copy()
test_data_copy = test_data.copy()

cat_cols = test_data.select_dtypes(exclude='number').columns.tolist()
feature_les = {col: LabelEncoder() for col in cat_cols}  # encoder for each categorical feature
target_le = LabelEncoder()


for col in cat_cols:
    train_data_copy[col] = feature_les[col].fit_transform(train_data_copy[col])
    test_data_copy[col] = feature_les[col].transform(test_data_copy[col])

train_data_copy['Fertilizer Name'] = target_le.fit_transform(train_data_copy['Fertilizer Name'])


from sklearn.preprocessing import label_binarize
import numpy as np


def map3_score(predicted_top3: np.ndarray,   # shape = (n_val, 3), dtype = object or int
               y_true_fold: np.ndarray,      # shape = (n_val,)
              ) -> float:
    """
    predicted_top3[i] is a length‐3 array of labels (strings/ints) that your model thinks
    are most likely for sample i, ordered from most confident 3rd most confident.
    y_true_fold[i] is the single true label for sample i.
    We give credit = 1/rank if the true label is at position 'rank' in that top‐3 list;
    otherwise 0. Then we average over all i.
    """
    print(type(predicted_top3), type(y_true_fold))
    
    n_val = y_true_fold.shape[0]
    total_score = 0.0

    for i in range(n_val):
        true_label = y_true_fold[i]
        top3_preds = predicted_top3[i].tolist()  # convert row to a Python list

        try:
            # .index(...) returns 0-based position. Add +1 to get 1-based rank.
            rank = top3_preds.index(true_label) + 1
            if rank <= 3:
                total_score += 1.0 / rank
            # If rank > 3, that cannot happen here, because top3_preds has exactly 3 items.
        except ValueError:
            # true_label not in top-3  score += 0
            pass

    return total_score / n_val


import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


X = train_data_copy.drop('Fertilizer Name',axis = 1)
y = train_data_copy["Fertilizer Name"]
# test = test_data_copy.copy()

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.1, stratify=y)


def xgboost_objective(trial):

    cv = StratifiedKFold(n_splits=5, shuffle=True)
    map3_scores = []

    for train_index, val_index in cv.split(X_train, y_train):
        # x_train_fold = X_train[train_index]
        # x_val_fold = X_train[val_index]
        # y_train_fold = y_train[train_index]
        # y_val_fold = y_train[val_index]

        x_train_fold = X_train.iloc[train_index]
        x_val_fold = X_train.iloc[val_index]
        
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]
        
        model = XGBClassifier()
        
        
        model.fit(x_train_fold, y_train_fold, eval_set=[(x_val_fold, y_val_fold)],
              early_stopping_rounds=50, verbose=False)

        pred_proba = model.predict_proba(x_val_fold)
        top3_index = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
        class_labs = model.classes_
        top3_labs = class_labs[top3_index]

        fold_map3 = map3_score(top3_labs, y_val_fold.to_numpy())
        map3_scores.append(fold_map3)
        mean_map3 = np.mean(map3_scores)

    return mean_map3


xgb_classifier = XGBClassifier(
    verbosity=0,
    objective='multi:softprob',
    enable_categorical=True,
    tree_method="gpu_hist",
    gpu_id=0, 
    predictor="gpu_predictor",
    n_jobs=-1
)

xgb_classifier.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
      early_stopping_rounds=50, verbose=False)


pred_proba = xgb_classifier.predict_proba(test_data_copy.to_numpy())


top3_index = np.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]
class_labs = xgb_classifier.classes_
top3_labs = class_labs[top3_index]


top3_result_strings = np.array(list(map(
    lambda x: ' '.join(target_le.inverse_transform(x)), top3_labs)))


top3_result_strings.shape, test_data.shape, submission_data.shape


submission = pd.DataFrame({
    'id': submission_data['id'].values,
    'Fertilizer Name': top3_result_strings
})
submission.to_csv('/kaggle/working/submission.csv',index=False)
submission

