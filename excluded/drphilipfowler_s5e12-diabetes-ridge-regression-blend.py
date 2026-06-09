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


!pip install -q pytorch-tabnet
import os
import gc
import time
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import torch
import tensorflow as tf

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras import layers, models
from pytorch_tabnet.tab_model import TabNetClassifier

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("Libraries installed and imported successfully!")


IS_KAGGLE = True 

# Paths
if IS_KAGGLE:
    INPUT_DIR = '/kaggle/input/playground-series-s5e12' # Example path, adjust for your comp
    WORK_DIR = '/kaggle/working'
else:
    INPUT_DIR = 'data' # If uploading files manually to Colab root
    WORK_DIR = '.'

OOF_DIR = os.path.join(WORK_DIR, 'oof')
SUB_DIR = os.path.join(WORK_DIR, 'submissions')
os.makedirs(OOF_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

TARGET = 'diagnosed_diabetes'
ID = 'id'
SEED = 42
N_SPLITS = 5

print(f"Configuration set. Target: {TARGET}")




if not os.path.exists(os.path.join(INPUT_DIR, 'train.csv')):
    print(f"âš ï¸� WARNING: train.csv not found in {INPUT_DIR}. Please upload your data!")
else:
    train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
    sample_sub = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'))
    print(f"Data Loaded: Train {train_df.shape}, Test {test_df.shape}")

CAT_COLS = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
NUM_COLS = [c for c in train_df.columns if c not in CAT_COLS + [ID, TARGET]]

print(f"Categorical Features: {len(CAT_COLS)}")
print(f"Numerical Features: {len(NUM_COLS)}")


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
train_df['fold'] = -1

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[TARGET])):
    train_df.loc[val_idx, 'fold'] = fold

print("Folds generated successfully.")

def save_oof(name, oof_preds, test_preds):
    np.save(os.path.join(OOF_DIR, f'{name}_oof.npy'), oof_preds)
    np.save(os.path.join(OOF_DIR, f'{name}_test.npy'), test_preds)
    print(f"âœ… Saved {name} OOF & Test preds")


print("\n=== Training LightGBM ===")
oof_lgb = np.zeros(len(train_df))
test_lgb = np.zeros(len(test_df))

X_lgb = train_df.copy()
X_test_lgb = test_df.copy()
for col in CAT_COLS:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

for fold in range(N_SPLITS):
    tr_idx, val_idx = X_lgb['fold'] != fold, X_lgb['fold'] == fold
    X_tr, y_tr = X_lgb.loc[tr_idx].drop([ID, TARGET, 'fold'], axis=1), X_lgb.loc[tr_idx, TARGET]
    X_val, y_val = X_lgb.loc[val_idx].drop([ID, TARGET, 'fold'], axis=1), X_lgb.loc[val_idx, TARGET]
    
    model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=SEED, n_jobs=-1, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    oof_lgb[val_idx] = model.predict_proba(X_val)[:, 1]
    test_lgb += model.predict_proba(X_test_lgb.drop([ID], axis=1))[:, 1] / N_SPLITS
    print(f"Fold {fold} AUC: {roc_auc_score(y_val, oof_lgb[val_idx]):.5f}")

save_oof("lgbm_v1", oof_lgb, test_lgb)


print("\n=== Training XGBoost ===")
oof_xgb = np.zeros(len(train_df))
test_xgb = np.zeros(len(test_df))

for fold in range(N_SPLITS):
    tr_idx, val_idx = train_df['fold'] != fold, train_df['fold'] == fold
    X_tr = X_lgb.loc[tr_idx].drop([ID, TARGET, 'fold'], axis=1) 
    y_tr = train_df.loc[tr_idx, TARGET]
    X_val = X_lgb.loc[val_idx].drop([ID, TARGET, 'fold'], axis=1)
    y_val = train_df.loc[val_idx, TARGET]

    model = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6, 
        tree_method='hist', enable_categorical=True, 
        random_state=SEED, n_jobs=-1, eval_metric='auc',
        early_stopping_rounds=50
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    oof_xgb[val_idx] = model.predict_proba(X_val)[:, 1]
    test_xgb += model.predict_proba(X_test_lgb.drop([ID], axis=1))[:, 1] / N_SPLITS
    print(f"Fold {fold} AUC: {roc_auc_score(y_val, oof_xgb[val_idx]):.5f}")

save_oof("xgb_v1", oof_xgb, test_xgb)


print("\n=== Training CatBoost ===")
oof_cat = np.zeros(len(train_df))
test_cat = np.zeros(len(test_df))

X_cat = train_df.drop([ID, TARGET, 'fold'], axis=1).copy()
X_test_cat = test_df.drop([ID], axis=1).copy()
for col in CAT_COLS:
    X_cat[col] = X_cat[col].fillna("None").astype(str)
    X_test_cat[col] = X_test_cat[col].fillna("None").astype(str)

for fold in range(N_SPLITS):
    tr_idx, val_idx = train_df['fold'] != fold, train_df['fold'] == fold
    X_tr, y_tr = X_cat.loc[tr_idx], train_df.loc[tr_idx, TARGET]
    X_val, y_val = X_cat.loc[val_idx], train_df.loc[val_idx, TARGET]
    
    train_pool = Pool(X_tr, y_tr, cat_features=CAT_COLS)
    val_pool = Pool(X_val, y_val, cat_features=CAT_COLS)
    
    model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, eval_metric='AUC', random_seed=SEED, verbose=0)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
    
    oof_cat[val_idx] = model.predict_proba(X_val)[:, 1]
    test_cat += model.predict_proba(X_test_cat)[:, 1] / N_SPLITS
    print(f"Fold {fold} AUC: {roc_auc_score(y_val, oof_cat[val_idx]):.5f}")

save_oof("cat_v1", oof_cat, test_cat)


print("\n=== Training Logistic Regression ===")
oof_log = np.zeros(len(train_df))
test_log = np.zeros(len(test_df))

# Pipeline for Linear Model (Scaling + OHE)
preprocessor_linear = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), NUM_COLS),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), CAT_COLS)
])

X_lin = train_df.drop([ID, TARGET, 'fold'], axis=1)
X_test_lin = test_df.drop([ID], axis=1)

for fold in range(N_SPLITS):
    tr_idx, val_idx = train_df['fold'] != fold, train_df['fold'] == fold
    X_tr, y_tr = X_lin.loc[tr_idx], train_df.loc[tr_idx, TARGET]
    X_val, y_val = X_lin.loc[val_idx], train_df.loc[val_idx, TARGET]
    
    clf = Pipeline([('pre', preprocessor_linear), ('clf', LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1))])
    clf.fit(X_tr, y_tr)
    
    oof_log[val_idx] = clf.predict_proba(X_val)[:, 1]
    test_log += clf.predict_proba(X_test_lin)[:, 1] / N_SPLITS
    print(f"Fold {fold} AUC: {roc_auc_score(y_val, oof_log[val_idx]):.5f}")

save_oof("logistic_v1", oof_log, test_log)


print("\n=== Training ExtraTrees ===")
oof_et = np.zeros(len(train_df))
test_et = np.zeros(len(test_df))

preprocessor_tree = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), NUM_COLS),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')), 
                      ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), CAT_COLS)
])

for fold in range(N_SPLITS):
    tr_idx, val_idx = train_df['fold'] != fold, train_df['fold'] == fold
    X_tr, y_tr = X_lin.loc[tr_idx], train_df.loc[tr_idx, TARGET]
    X_val, y_val = X_lin.loc[val_idx], train_df.loc[val_idx, TARGET]
    
    clf = Pipeline([('pre', preprocessor_tree), ('clf', ExtraTreesClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=SEED))])
    clf.fit(X_tr, y_tr)
    
    oof_et[val_idx] = clf.predict_proba(X_val)[:, 1]
    test_et += clf.predict_proba(X_test_lin)[:, 1] / N_SPLITS
    print(f"Fold {fold} AUC: {roc_auc_score(y_val, oof_et[val_idx]):.5f}")

save_oof("et_v1", oof_et, test_et)


print("\n=== Training KNN (Scaled + OHE) ===")
# Reuse linear preprocessor as it does exactly what KNN needs
oof_knn = np.zeros(len(train_df))
test_knn = np.zeros(len(test_df))

for fold in range(N_SPLITS):
    tr_idx, val_idx = train_df['fold'] != fold, train_df['fold'] == fold
    X_tr, y_tr = X_lin.loc[tr_idx], train_df.loc[tr_idx, TARGET]
    X_val, y_val = X_lin.loc[val_idx], train_df.loc[val_idx, TARGET]
    
    clf = Pipeline([('pre', preprocessor_linear), ('clf', KNeighborsClassifier(n_neighbors=50, weights='distance', n_jobs=-1))])
    clf.fit(X_tr, y_tr)
    
    oof_knn[val_idx] = clf.predict_proba(X_val)[:, 1]
    test_knn += clf.predict_proba(X_test_lin)[:, 1] / N_SPLITS
    print(f"Fold {fold} AUC: {roc_auc_score(y_val, oof_knn[val_idx]):.5f}")

save_oof("knn_v1", oof_knn, test_knn)


print("\n=== Training Deep MLP ===")
X_mlp = train_df.copy()
X_test_mlp = test_df.copy()

scaler = StandardScaler()
X_mlp[NUM_COLS] = scaler.fit_transform(X_mlp[NUM_COLS].fillna(0))
X_test_mlp[NUM_COLS] = scaler.transform(X_test_mlp[NUM_COLS].fillna(0))

cat_dims = []
for col in CAT_COLS:
    le = LabelEncoder()
    full_list = pd.concat([X_mlp[col], X_test_mlp[col]]).astype(str)
    le.fit(full_list)
    X_mlp[col] = le.transform(X_mlp[col].astype(str))
    X_test_mlp[col] = le.transform(X_test_mlp[col].astype(str))
    cat_dims.append(len(le.classes_))

def build_mlp_model():
    num_inp = layers.Input(shape=(len(NUM_COLS),))
    embs = []
    cat_inps = []
    for dim in cat_dims:
        inp = layers.Input(shape=(1,))
        emb = layers.Embedding(dim, min(50, (dim+1)//2))(inp)
        emb = layers.Reshape(target_shape=(emb.shape[-1],))(emb)
        cat_inps.append(inp)
        embs.append(emb)
    x = layers.Concatenate()([num_inp] + embs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=[num_inp] + cat_inps, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

oof_mlp = np.zeros(len(train_df))
test_mlp = np.zeros(len(test_df))
test_inputs_mlp = [X_test_mlp[NUM_COLS].values] + [X_test_mlp[col].values for col in CAT_COLS]

for fold in range(N_SPLITS):
    tr_idx, val_idx = train_df['fold'] != fold, train_df['fold'] == fold
    X_tr = [X_mlp.loc[tr_idx, NUM_COLS].values] + [X_mlp.loc[tr_idx, col].values for col in CAT_COLS]
    y_tr = train_df.loc[tr_idx, TARGET].values
    X_val = [X_mlp.loc[val_idx, NUM_COLS].values] + [X_mlp.loc[val_idx, col].values for col in CAT_COLS]
    y_val = train_df.loc[val_idx, TARGET].values
    
    model = build_mlp_model()
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=15, batch_size=1024, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    
    oof_mlp[val_idx] = model.predict(X_val, verbose=0).flatten()
    test_mlp += model.predict(test_inputs_mlp, verbose=0).flatten() / N_SPLITS
    print(f"Fold {fold} AUC: {roc_auc_score(y_val, oof_mlp[val_idx]):.5f}")

save_oof("mlp_v1", oof_mlp, test_mlp)


print("\n=== Training TabNet ===")
oof_tab = np.zeros(len(train_df))
test_tab = np.zeros(len(test_df))

X_tab = X_mlp.drop([ID, TARGET, 'fold'], axis=1).values
X_test_tab = X_test_mlp.drop([ID], axis=1).values
y_full = train_df[TARGET].values

cat_idxs = [i for i, col in enumerate(X_mlp.drop([ID, TARGET, 'fold'], axis=1).columns) if col in CAT_COLS]

for fold in range(N_SPLITS):
    tr_idx = train_df['fold'] != fold
    val_idx = train_df['fold'] == fold
    X_tr, y_tr = X_tab[tr_idx], y_full[tr_idx]
    X_val, y_val = X_tab[val_idx], y_full[val_idx]
    
    clf = TabNetClassifier(
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 50, "gamma": 0.9},
        verbose=0, seed=SEED
    )
    clf.fit(X_train=X_tr, y_train=y_tr, eval_set=[(X_val, y_val)], eval_metric=['auc'],
            max_epochs=50, patience=10, batch_size=1024, virtual_batch_size=128, num_workers=0)
    
    oof_tab[val_idx] = clf.predict_proba(X_val)[:, 1]
    test_tab += clf.predict_proba(X_test_tab)[:, 1] / N_SPLITS
    print(f"Fold {fold} AUC: {roc_auc_score(y_val, oof_tab[val_idx]):.5f}")

save_oof("tabnet_v1", oof_tab, test_tab)


print("\n=== Ensemble Analysis ===")
oof_dict = {
    'lgbm': oof_lgb, 'xgb': oof_xgb, 'cat': oof_cat,
    'logistic': oof_log, 'extratrees': oof_et, 'knn': oof_knn,
    'mlp': oof_mlp, 'tabnet': oof_tab
}
oof_df = pd.DataFrame(oof_dict)

print("Correlation Matrix:")
corr = oof_df.corr()
print(corr)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".3f")
plt.title("OOF Correlation Matrix")
plt.show()


print("\n=== Training Final Ridge Blender ===")
X_meta = oof_df.values
X_meta_test = np.column_stack([
    test_lgb, test_xgb, test_cat, test_log, test_et, test_knn, test_mlp, test_tab
])
y_meta = train_df[TARGET].values

blender = Ridge(alpha=1.0)
blender.fit(X_meta, y_meta)

print("Blender Weights:")
weights = pd.Series(blender.coef_, index=oof_dict.keys())
print(weights.sort_values(ascending=False))

# Final 
final_oof = np.clip(blender.predict(X_meta), 0, 1)
final_test = np.clip(blender.predict(X_meta_test), 0, 1)

final_score = roc_auc_score(y_meta, final_oof)
print(f"\nğŸ�† FINAL STACKED OOF AUC: {final_score:.6f}")


sample_sub[TARGET] = final_test
sub_path = os.path.join(SUB_DIR, 'submission.csv')
sample_sub.to_csv(sub_path, index=False)
print(f"âœ… Submission saved to {sub_path}")

