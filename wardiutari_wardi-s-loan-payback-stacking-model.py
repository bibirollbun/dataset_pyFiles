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


df=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')



from sklearn.model_selection import train_test_split

X       = df.drop(columns=['id','loan_paid_back'])
y       = df['loan_paid_back']
id_base = df['id']

X_train_old, X_test_old, y_train, y_test,id_train, id_test = train_test_split(
    X, y, id_base, test_size=0.2, random_state=42
)

#print(y_train)
#print(id_train)


from itertools import combinations

def add_features(X):
    # kumpulkan kolom baru di dict
    new_cols = {}

    #changer numeric to categorical
    X['credit_band'] = pd.cut(X['credit_score'],
                               bins=[0, 600, 660, 720, 780, 1000],
                               labels=['subprime', 'near-prime', 'prime', 'super-prime', 'elite'])

    # numeric features
    base_numeric_features = X.select_dtypes(include=[np.number]).columns
    for col1, col2 in combinations(base_numeric_features, 2):
        new_cols[f'{col1}_x_{col2}'] = X[col1] * X[col2]
        new_cols[f'{col1}_d_{col2}'] = X[col1] / (X[col2] + 1e-10)

    # categorical features
    base_categorical_features = X.select_dtypes(include=['object', 'category']).columns
    for col1, col2 in (combinations(base_categorical_features, 2)):
        new_cols[f'{col1}__{col2}'] = X[col1].astype(str) + "_" + X[col2].astype(str)

    # combine
    X_new = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)

    return X_new

X_train = add_features(X_train_old)
X_test = add_features(X_test_old)

# Update categorical list with engineered categorical feature
feature_num = X_train.select_dtypes(include=[np.number]).columns
feature_cat = X_train.select_dtypes(include=['object', 'category']).columns

display(feature_num)
display(feature_cat)


# -----------------------------
# 4) Class imbalance handling
# -----------------------------
# Compute class weights for balanced learning (used by LightGBM)
from sklearn.utils.class_weight import compute_class_weight


classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

y_perc1 = y_train.value_counts(normalize=True)[1] * 100


# Preprocessing
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler,
    OneHotEncoder, OrdinalEncoder,
    FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models - Linear
from sklearn.linear_model import LogisticRegression, SGDClassifier, Ridge

# Models - Ensemble
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier,
    GradientBoostingClassifier, StackingClassifier,
    ExtraTreesRegressor, RandomForestRegressor,
    StackingRegressor
)

# Models - SVM
from sklearn.svm import SVC, LinearSVC

# Models - Tree & Naive Bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Models - Neural Network
from sklearn.neural_network import MLPClassifier

# Models - Gradient Boosting Libraries
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier

# Evaluation & CV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score


feature_cat_ord     = ['grade_subgrade']
target              = ['loan_paid_back']

# Manual Mapping grade_subgrade (A1–F5)
grade_order = [
    "A1","A2","A3","A4","A5",
    "B1","B2","B3","B4","B5",
    "C1","C2","C3","C4","C5",
    "D1","D2","D3","D4","D5",
    "E1","E2","E3","E4","E5",
    "F1","F2","F3","F4","F5"
]

# pipeline khusus untuk ordinal (hasilnya diskalakan ke 0–1)
ord_pipeline = Pipeline([
    ("ordinal", OrdinalEncoder(categories=[grade_order])),
    ("scaler", MinMaxScaler())   # ubah range 0–29 jadi 0–1
])

# transformer untuk tiap tipe fitur
preprocessor = ColumnTransformer(
    transformers=[
        ("number", StandardScaler(), feature_num),
        ("cat_nominal", OneHotEncoder(handle_unknown="ignore"), feature_cat),
        ("cat_ordinal", ord_pipeline, feature_cat_ord),
    ]
)


# BASE ESTIMATORS (dengan asumsi hyperparameter terbaik telah ditemukan)
base_estimators = [
    # 1. XGBoost    --> Optimize
    ('xgb', XGBClassifier(  
        subsample           = 0.6,
        n_estimators        = 300,
        max_depth           = 4,
        learning_rate       = 0.2,
        colsample_bytree    = 0.8,
        scale_pos_weight    = (100-y_perc1)/y_perc1,
        eval_metric='auc',
        random_state=42, 
        n_jobs=-1
    )),
    
    # 2. LightGBM   --> Optimize
    ('lgbm', LGBMClassifier(
        n_estimators=200, 
        num_leaves=40, 
        learning_rate=0.1, 
        random_state=42,
        class_weight=class_weight_dict,
        metric='auc',
        n_jobs=-1
    )),
    
    # 3. Random Forest
    ('rf', RandomForestClassifier(
        n_estimators=400, 
        max_depth=15, 
        min_samples_split=5,
        class_weight=class_weight_dict,
        random_state=42, 
        n_jobs=-1
    )),
    
    # 4. SGD Classifier
    ('sgd', SGDClassifier(
        penalty             = 'l1',
        max_iter            = 500,
        loss                = 'log_loss',
        learning_rate       = 'adaptive',
        eta0                = 0.001,
        alpha               = 1e-05,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1

    )),

    #5. MLP SCIKIT LEARN
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64),   # 2 hidden layer: 128 dan 64 neuron
        activation="relu",              # fungsi aktivasi
        solver="adam",                  # optimizer
        max_iter=500,                   # batas maksimum epoch (jarang sampai habis karena early stopping)
        early_stopping=True,            # aktifkan early stopping
        n_iter_no_change=10,            # berhenti jika 10 epoch berturut-turut tidak ada perbaikan
        random_state=42
    ))
]

# Meta learner (final estimator)
# FINAL ESTIMATOR (Meta-Learner)
final_estimator = LogisticRegression(
    solver='saga',      # Solver yang bagus untuk kasus besar/multinomial
    max_iter=5000,      # Tingkatkan iterasi untuk memastikan konvergensi
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)
# Stacking classifier
stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=final_estimator,
    n_jobs=-1,            # gunakan semua core CPU
    passthrough=False     # hanya gunakan prediksi base learners
)

# Pipeline dengan scaler
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ('stack', stack)
])

# Training
pipeline.fit(X_train, y_train)

# Evaluasi
print("Train accuracy:", pipeline.score(X_train, y_train))
print("Test accuracy:", pipeline.score(X_test, y_test))


import joblib

# Tentukan nama file yang akan digunakan (umumnya menggunakan ekstensi .joblib atau .pkl)
filename = '/kaggle/working/pipeline_klasifikasi_final.joblib' 

# 'pipeline' adalah objek Pipeline yang sudah Anda panggil .fit(X, y)
joblib.dump(pipeline, filename)

print(f"Pipeline berhasil disimpan ke: {filename}")


score = pipeline.score(X_test, y_test)

y_pred = pipeline.predict(X_test)

y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
# y_test = label asli (0/1)
# y_pred_proba = probabilitas prediksi kelas positif dari model
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("Pipeline score :", score)
print("MSE            :", mean_squared_error(y_test, y_pred))
print("R²             :", r2_score(y_test, y_pred))
print("ROC AUC Score  :", roc_auc)


#ROC AUC Score  : 0.921222335386582


df_sub  = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

X_sub_old    = df_sub.drop(columns=['id'])
X_sub = add_features(X_sub_old)
id_sub = df_sub['id']


y_sub_pred = pipeline.predict(X_sub)
y_sub_pred_proba =pipeline.predict_proba(X_sub)[:,1]


#Pred Proba Sumbission
submission_proba = pd.DataFrame({
    "id": id_sub,
    "loan_paid_back": y_sub_pred_proba
})
submission_proba.to_csv("/kaggle/working/submission_proba.csv", index=False)

#Pred Sumbission
submission = pd.DataFrame({
    "id": id_sub,
    "loan_paid_back": y_sub_pred
})
submission.to_csv("/kaggle/working/submission.csv", index=False)



