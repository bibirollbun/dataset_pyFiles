# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import MinMaxScaler, RobustScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier

from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import xgboost as xgb

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

# Dans X, on va prendre toutes les colonnes sauf "id" qui est unique
FEATURES = [c for c in df_test.columns if c != "id"] 

CAT_COLS = ["Soil Type", "Crop Type"] # Données catégorielles
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS] # Données numériques
LABEL = "Fertilizer Name" # Le label ou le Y

ENCODER = "CAT"

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

X = df_train[FEATURES]
y = df_train[LABEL]
X_test = df_test[FEATURES]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Définition du modèle ---
clf = LGBMClassifier(
    random_state=0,
    class_weight='balanced',
    force_col_wise=True,
    n_estimators=1000,
    learning_rate=0.02,
    num_leaves=127,
    max_depth=12,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    min_child_samples=20,
    min_split_gain=0.1,
    boosting_type='goss'
)

# --- Stratified K-Fold CV ---
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]

    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)]
    )

    proba = clf.predict_proba(X_va)
    top3  = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
    actual = [[lab] for lab in y_va]
    score = mapk(actual, top3, k=3)
    print(f"Fold {fold} Map@3 : {score:.6f}")
    fold_scores.append(score)

print(f"\nMean Map@3 over {n_splits} folds: {np.mean(fold_scores):.6f}")

mean_map3 = np.mean(fold_scores)
current_score = mean_map3

# --- Entraînement final et soumission ---
clf.fit(X, y)  # on réentraîne sur tout l'ensemble

proba_test = clf.predict_proba(X_test)
top3_test = np.argsort(proba_test, axis=1)[:, -3:][:, ::-1]

# Reconvertir en nom de fertiliseur
le = LabelEncoder().fit(df_train[LABEL])
labels_test = le.inverse_transform(top3_test.ravel()).reshape(top3_test.shape)

submission = pd.DataFrame({
    'id': sub['id'],
    'Fertilizer Name': [' '.join(row) for row in labels_test]
})

submission.to_csv("submission.csv", index=False)
print("Fichier submission.csv généré.")  
print(current_score)




















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
















