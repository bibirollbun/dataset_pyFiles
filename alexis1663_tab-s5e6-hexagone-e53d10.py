!pip -q install /kaggle/input/pytorchtabnet/pytorch_tabnet-4.1.0-py3-none-any.whl


from pytorch_tabnet.tab_model import TabNetClassifier
import torch


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import MinMaxScaler, RobustScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier

from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import xgboost as xgb
from xgboost import XGBClassifier

import category_encoders as ce

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

df1 = pd.read_csv('/kaggle/input/hex-s5e6/FertilizerPrediction1.csv')
df2 = pd.read_csv('/kaggle/input/hex-s5e6/FertilizerPrediction2.csv')

df_train = pd.concat([df1, df2, df_train], axis=0, ignore_index=True)

# Dans X, on va prendre toutes les colonnes sauf "id" qui est unique
FEATURES = [c for c in df_test.columns if c != "id"] 

CAT_COLS = ["Soil Type", "Crop Type"] # Données catégorielles
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS] # Données numériques
LABEL = "Fertilizer Name" # Le label ou le Y

ENCODER = "OE"

# Nombre de classes
CLASSES = 7

if ENCODER == "CAT":
    df_train[CAT_COLS] = df_train[CAT_COLS].fillna("None").astype("category")
    df_test[CAT_COLS] = df_test[CAT_COLS].fillna("None").astype("category")
elif ENCODER == "TE":
    enc = ce.TargetEncoder(cols=CAT_COLS)
    y_train = pd.Series(df_train[LABEL].tolist(), index=df_train[LABEL].index)
    enc.fit(df_train[FEATURES], y_train)
    df_train[FEATURES] = enc.transform(df_train[FEATURES], y_train)
    df_test[FEATURES] = enc.transform(df_test[FEATURES])
elif ENCODER == "OE":
    enc = OrdinalEncoder()
    enc.fit(df_train[CAT_COLS])
    df_train[CAT_COLS] = enc.transform(df_train[CAT_COLS])
    df_test[CAT_COLS] = enc.transform(df_test[CAT_COLS]) 

X = df_train[FEATURES]
y = df_train[LABEL]
X_test = df_test[FEATURES]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

"""
model = AdaBoostClassifier()
model.fit(X_train, y_train)

pred_val = model.predict(X_val)
pred = model.predict(X_test)
"""

"""
param_grid_reg =  {'num_leaves': [31], 'max_depth': [10], 'n_estimators':[1000], 'learning_rate':[0.3, 0.1, 0.03, 0.01]}
clf = LGBMClassifier(random_state=0, class_weight='balanced', force_col_wise=True)

gscv = GridSearchCV(estimator=clf, scoring="accuracy", cv=5, param_grid=param_grid_reg, verbose=5)
gscv.fit(X,y)

print("Score", gscv.best_score_)
print("Params", gscv.best_params_)
"""

# clf = LGBMClassifier(random_state=0, class_weight='balanced', force_col_wise=True)
"""
clf = XGBClassifier(
    random_state=42,
    tree_method='gpu_hist',
    n_estimators=2000,         # plus d’arbres, combinés avec un early_stopping dans fit()
    learning_rate=0.03,        # LR plus faible => convergence plus progressive
    max_depth=4,               # arbres moins profonds
    min_child_weight=7,        # plus d’échantillons requis par feuille
    gamma=0.5,                 # split plus sélectif
    subsample=0.7,             # moins d’échantillons par arbre
    colsample_bytree=0.6,      # moins de features par arbre
    reg_alpha=3.0,             # L1 plus fort
    reg_lambda=5.0,            # L2 plus fort
    objective='multi:softprob',
    num_class=len(encoder.classes_),
    use_label_encoder=False,
    eval_metric='mlogloss',
    scale_pos_weight=1,
    enable_categorical=True,
)
"""
clf = TabNetClassifier()
# Passez eval_set et early_stopping_rounds
clf.fit(
    X_train.values, y_train,
)

def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

pred_val = clf.predict_proba(X_val.values)

top_3_preds = np.argsort(pred_val, axis=1)[:, -3:][:, ::-1]  
actual = [[label] for label in y_val]

map3_score = mapk(actual, top_3_preds)
current_score = np.round(map3_score,6)

pred_test = clf.predict_proba(X_test.values)
top_3_preds = np.argsort(pred_test, axis=1)[:, -3:][:, ::-1]
top_3_labels = encoder.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

submission = pd.DataFrame({
    'id': sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})

submission.to_csv("submission.csv", index=False)
print(current_score)





































