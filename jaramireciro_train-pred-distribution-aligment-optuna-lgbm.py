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
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost.core')
warnings.filterwarnings("ignore", message="1 warning generated.")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#Data Wrangling Libraries
import numpy as np
import pandas as pd

#Data Processing Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#Model Metricrs

from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score


# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier


# Model Search
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.pipeline import Pipeline

#Plotting Libraries

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import math


train_df=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
train_df=train_df.set_index("id")#.sample(75000)
test_df=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
test_df=test_df.set_index("id")#.sample(25000)


train_df.head()


# -----------------------------
# 1. Combine Train and Test Data
# -----------------------------
train_df["__origin__"] = 0  # 0 = train
test_df["__origin__"] = 1   # 1 = test

df_train_data = pd.concat([train_df, test_df], ignore_index=True)

data_ohc = df_train_data.drop(columns=["__origin__","y"])
y = df_train_data["__origin__"]


def one_hot_encoding(data_ohc,le,ohc,categorical_cols):

    """
    This function goes through each columns and
    
    1. creates the one hot encoded based on the values of the selected column
    2. removes the selected column from the data.
    3. Renames the new columns
    4. Creates a data frame with the new columns
    5. Concatenate the data frame (with the new columns) to the original data frame.
    
    """
    for col in categorical_cols:
        dat = le.fit_transform(data_ohc[col]).astype(int)
        data_ohc = data_ohc.drop(col, axis=1)
        new_dat = ohc.fit_transform(dat.reshape(-1, 1))
        col_names = ['_'.join([col, str(x)]) for x in le.classes_]
        new_df = pd.DataFrame(new_dat, index=data_ohc.index, columns=col_names).astype(int)
        data_ohc = pd.concat([data_ohc, new_df], axis=1)

    return data_ohc


def one_hot_encoding_ordinal_columns(data_ohc, ordinal_cols, le, ohc):
    mappings = {}
    for col in ordinal_cols:
        dat = le.fit_transform(data_ohc[col]).astype(int)
        data_ohc = data_ohc.drop(col, axis=1)
        new_df = pd.DataFrame(dat, index=data_ohc.index, columns=[col])
        new_df = new_df.astype(int)
        data_ohc = pd.concat([data_ohc, new_df], axis=1)
        mappings[col] = dict(zip(range(len(le.classes_)), le.classes_))
    
    return data_ohc, mappings


def log1p_regularization(data_ohc,col_to_scale):
    for column in col_to_scale:
        data_ohc[column] = np.log1p(data_ohc[column])
    return data_ohc

# using scalers
import joblib  # or pickle

mm = MinMaxScaler()
s=StandardScaler()
pt = PowerTransformer(method='yeo-johnson') 


# Fit the scaler on training data
scaler = PowerTransformer(method='yeo-johnson')
#scaler.fit(data_ohc[non_categorical_cols])

# Save the fitted scaler to a file
joblib.dump(scaler, 'minmax_scaler.pkl')


# Load the saved scaler
#scaler = joblib.load('minmax_scaler.pkl')
# Transform the prediction data
#data_ohc_pred[col_to_scale] = scaler.transform(data_ohc_pred[col_to_scale])


mask = data_ohc.dtypes == 'object'
categorical_cols = data_ohc.columns[mask]
non_categorical_cols = data_ohc.select_dtypes(include=[np.number]).columns
ordinal_cols=["month"]
categorical_cols=list(set(categorical_cols)-set(ordinal_cols))
non_categorical_cols=list(set(non_categorical_cols))#-set(["y"]))


# One Hot Encoding Categorical Columns

le = LabelEncoder()
ohc = OneHotEncoder(sparse=False)
data_ohc =one_hot_encoding(data_ohc,le,ohc,categorical_cols)

data_ohc, mappings = one_hot_encoding_ordinal_columns(data_ohc, ordinal_cols, le, ohc)

#scaler = MinMaxScaler()
scaler.fit(data_ohc[non_categorical_cols])
data_ohc[non_categorical_cols] = scaler.transform(data_ohc[non_categorical_cols])

#data_ohc=log1p_regularization(data_ohc,non_categorical_cols)


data_ohc


X=data_ohc
clf = Pipeline([
    ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42))
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                  random_state=42, stratify=y)
clf.fit(X_train, y_train)

print("Validation accuracy (train vs test separation):", clf.score(X_val, y_val))





data_ohc_tain=train_df.drop(columns=["__origin__","y"])
data_ohc_tain=one_hot_encoding(data_ohc_tain,le,ohc,categorical_cols)
data_ohc_tain, mappings = one_hot_encoding_ordinal_columns(data_ohc_tain, ordinal_cols, le, ohc)
data_ohc_tain[non_categorical_cols] = scaler.transform(data_ohc_tain[non_categorical_cols])
data_ohc_tain


data_ohc_test=test_df.drop(columns=["__origin__"])
data_ohc_test=one_hot_encoding(data_ohc_test,le,ohc,categorical_cols)
data_ohc_test, mappings = one_hot_encoding_ordinal_columns(data_ohc_test, ordinal_cols, le, ohc)
data_ohc_test[non_categorical_cols] = scaler.transform(data_ohc_test[non_categorical_cols])
#data_ohc_test=log1p_regularization(data_ohc_test,non_categorical_cols)

data_ohc_test


# -----------------------------
# 4. Get probabilities (test-likeness)
# -----------------------------
train_probs = clf.predict_proba(data_ohc_tain)[:, 1]
test_probs = clf.predict_proba(data_ohc_test)[:, 1]


train_probs.shape


test_probs.shape


train_df["test_likeness"] = train_probs
test_df["test_likeness"] = test_probs



# -----------------------------
# 4. Get probabilities (test-likeness)
# -----------------------------

data_ohc_tain["test_likeness"] = train_probs
data_ohc_test["test_likeness"] = test_probs

# -----------------------------
# 5. Visualization
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(train_probs, bins=50, alpha=0.6, label="Training samples", density=True)
plt.hist(test_probs, bins=50, alpha=0.6, label="Test samples", density=True)
plt.axvline(0.3, color="red", linestyle="--", label="Threshold (0.3)")
plt.xlabel("Probability of being from TEST set (test-likeness)")
plt.ylabel("Density")
plt.title("Distribution of test-likeness scores")
plt.legend()
plt.show()



# -----------------------------
# 6. Filter or reweight
# -----------------------------

data_ohc_tain["y"]=train_df["y"]

threshold = 0.2375
filtered_train = data_ohc_tain[data_ohc_tain["test_likeness"] > threshold]



print(f"Original train size: {len(train_df)}, Filtered train size: {len(filtered_train)}")


# -----------------------------
# 7. Train your actual model
# -----------------------------
# Example using RandomForest with filtered data
X_filtered = filtered_train.drop(columns=["test_likeness","y"])  # drop helper cols
y_filtered = filtered_train["y"]  # <-- replace with your actual target col

print(y_filtered.shape)
print(X_filtered.shape)


def confusion_matrix_graph(y,y_pred):
    sns.set_context('talk')
    cm = confusion_matrix(y, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt='d')


def measure_error(y_true, y_pred,label, y_probs):
    return pd.Series({'accuracy':accuracy_score(y_true, y_pred),
                      'precision': precision_score(y_true, y_pred),
                      'recall': recall_score(y_true, y_pred),
                      'f1': f1_score(y_true, y_pred),
                       'auc_score' : roc_auc_score(y_true, y_probs),
                     },
                      name=label)


"""
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import numpy as np

# Define the model
best_xgb_param={'subsample': 0.8
                ,'reg_lambda': 2
                , 'reg_alpha': 1
                , 'n_estimators': 300
                , 'max_depth': 8
                , 'learning_rate': 0.1
                , 'gamma': 0.1
                , 'colsample_bytree': 0.6}

XGB_model = XGBClassifier(use_label_encoder=False
                      , eval_metric='auc'
                      , verbosity=0
                      ,**best_xgb_param)

# Run the search
XGB_model.fit(X_filtered, y_filtered)

#Fitting 3 folds for each of 50 candidates, totalling 150 fits
#Best params: {'subsample': 0.8, 'reg_lambda': 2, 'reg_alpha': 1, 'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1, 'gamma': 0.1, 'colsample_bytree': 0.6}
#Best ROC AUC: 0.9667851576509138
"""


"""
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


y_probs = XGB_model.predict_proba(X_filtered)[:, 1]  # Probabilities for the positive class

auc_score = roc_auc_score(y_filtered, y_probs)

fpr, tpr, thresholds = roc_curve(y_filtered, y_probs)


import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label='VotingClassifier')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
"""


"""
y_probs_pred = XGB_model.predict_proba(X_filtered)[:, 1]
y_pred=XGB_model.predict(X_filtered)
label="BaggingXCGBoost"
bag_xgb_classiffier_metrics=measure_error(y_filtered, y_pred, label,y_probs_pred)
bag_xgb_classiffier_metrics.name="BaggingXCGBoost"
bag_xgb_classiffier_metrics
"""


import optuna
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Split train/validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_filtered, y_filtered, test_size=0.2, random_state=42
)
""""""
def objective(trial):
    # Suggest hyperparameters
    param_grid = {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }

    model = lgb.LGBMClassifier(**param_grid, random_state=42)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[early_stopping(100), log_evaluation(0)]
    )

    # Predict on validation
    y_valid_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_valid_pred)

    return auc

# Run optimization
#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=30)

#print("Best AUC:", study.best_value)
#print("Best params:", study.best_params)

"""
Best AUC: 0.9700507644710196
Best params: {'learning_rate': 0.02903856251037579, 'num_leaves': 114, 'max_depth': 15, 'subsample': 0.69905185933801, 'colsample_bytree': 0.5397246745943016, 'min_child_samples': 23, 'reg_alpha': 1.3631147601168847, 'reg_lambda': 4.9743596953662745}

"""



# -----------------------------
# 4. Train Final Model with Best Params
# -----------------------------
best_params = study.best_trial.params
from lightgbm import log_evaluation


"""
best_params= {'learning_rate': 0.12294195115658733
              , 'num_leaves': 36
              , 'max_depth': 9
              , 'subsample': 0.6475738311300105
              , 'colsample_bytree': 0.9821505278723222
              , 'min_child_samples': 44
              , 'reg_alpha': 2.5289930631649122
              , 'reg_lambda': 1.8553965043740255}

"""
"""
best_params: {'learning_rate': 0.02903856251037579
             , 'num_leaves': 114
             , 'max_depth': 15
             , 'subsample': 0.69905185933801
             , 'colsample_bytree': 0.5397246745943016
             , 'min_child_samples': 23
             , 'reg_alpha': 1.3631147601168847
              , 'reg_lambda': 4.9743596953662745}
            
"""

best_params={'learning_rate': 0.06098508107342455
, 'num_leaves': 134, 'max_depth': 9
, 'subsample': 0.6094057179887311
, 'colsample_bytree': 0.5569072822258421
, 'min_child_samples': 34
, 'reg_alpha': 1.8994510601653842
, 'reg_lambda': 2.637953663803313}




best_params["n_estimators"] = 2000

final_model = lgb.LGBMClassifier(**best_params, random_state=42)

final_model.fit(
    X_filtered, y_filtered,
    eval_metric="auc",
    callbacks=[log_evaluation(100)]
)


y_probs_pred =  final_model.predict_proba(X_filtered)[:, 1]
y_pred = final_model.predict(X_filtered)

label = "LightGBM"
bag_xgb_classiffier_metrics = measure_error(y_filtered.values, y_pred, label, y_probs_pred)
bag_xgb_classiffier_metrics.name = "LightGBM"
bag_xgb_classiffier_metrics



df_test_data=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
df_test_data=df_test_data.set_index("id")
df_test_data


data_ohc_pred = df_test_data.copy()


#data_ohc_pred=imputing_data(data_ohc_pred,non_categorical_cols,imputer)
data_ohc_pred=one_hot_encoding(data_ohc_pred,le,ohc,categorical_cols)
data_ohc_pred, mappings = one_hot_encoding_ordinal_columns(data_ohc_pred, ordinal_cols, le, ohc)
data_ohc_pred[non_categorical_cols] = scaler.transform(data_ohc_pred[non_categorical_cols])
#data_ohc_pred=log1p_regularization(data_ohc_pred,non_categorical_cols)
y_pred=final_model.predict(data_ohc_pred)


y_pred=final_model.predict(data_ohc_pred)


data_ohc_pred


y_probs_pred =final_model.predict_proba(data_ohc_pred)[:, 1]  # Probabilities for the positive class
data_ohc_pred["y"]=y_probs_pred
results_probs_df=pd.DataFrame(data_ohc_pred["y"])
results_probs_df


results_probs_df.to_csv("results.csv")

