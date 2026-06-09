# Imports, seed setting, and data loading
import os
import random
import numpy as np
import pandas as pd

# Set random seed for reproducibility
seed = 73
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# TensorFlow & Keras imports
import tensorflow as tf
tf.random.set_seed(seed)

# Tree–based model libraries
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Load datasets (adjust the paths if needed)
train = pd.read_csv("/kaggle/input/higgs-boson-detection-2025/train.csv")
test  = pd.read_csv("/kaggle/input/higgs-boson-detection-2025/test.csv")
sample_submission = pd.read_csv("/kaggle/input/higgs-boson-detection-2025/sample_submission.csv")
print("Train shape:", train.shape, "Test shape:", test.shape)


# TabM model using TensorFlow/Keras

# Identify feature columns (all except 'label')
features = [col for col in train.columns if col != 'label']
nunique_max = 50
cat_cols = []
num_cols = []
for col in features:
    if train[col].nunique() < nunique_max:
        cat_cols.append(col)
    else:
        num_cols.append(col)
        
# Preprocess numeric features: scale them
scaler = StandardScaler()
train_num = pd.DataFrame(scaler.fit_transform(train[num_cols]), columns=num_cols)
test_num  = pd.DataFrame(scaler.transform(test[num_cols]), columns=num_cols)

# Preprocess categorical features: convert to string then one–hot encode
train_cat = train[cat_cols].astype(str)
test_cat  = test[cat_cols].astype(str)
train_cat_dummies = pd.get_dummies(train_cat, prefix=cat_cols)
test_cat_dummies  = pd.get_dummies(test_cat, prefix=cat_cols)

# Align the one–hot encoded columns between train and test
train_cat_dummies, test_cat_dummies = train_cat_dummies.align(test_cat_dummies, join='outer', axis=1, fill_value=0)

# Combine numeric and categorical features
X_train_tabm = pd.concat([train_num, train_cat_dummies], axis=1)
X_test_tabm  = pd.concat([test_num, test_cat_dummies], axis=1)
y_train_tabm = train['label']

print("TabM training data shape:", X_train_tabm.shape)

# Build a simple MLP model using TensorFlow/Keras with the Functional API
def build_tabm_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss='mse',  # as in the original code
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

input_dim = X_train_tabm.shape[1]
print("Input dimension for TabM:", input_dim)

# Use KFold cross-validation to train and predict with the TabM model
kf = KFold(n_splits=10, random_state=seed, shuffle=True)
tabm_preds_folds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tabm)):
    print(f"\nTabM - Fold {fold}")
    # Ensure the data is converted to float32
    X_tr = X_train_tabm.iloc[train_idx].values.astype(np.float32)
    y_tr = y_train_tabm.iloc[train_idx].values.astype(np.float32)
    X_val = X_train_tabm.iloc[val_idx].values.astype(np.float32)
    y_val = y_train_tabm.iloc[val_idx].values.astype(np.float32)
    
    model = build_tabm_model(input_dim)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=15, batch_size=128, verbose=1)
    # Predict on the test set (convert test data to float32)
    preds = model.predict(X_test_tabm.values.astype(np.float32)).flatten()
    tabm_preds_folds.append(preds)

# Take the median of the predictions across folds
tabm_preds = np.median(np.stack(tabm_preds_folds, axis=0), axis=0)
tabm_sub = sample_submission.copy()
tabm_sub["Predicted"] = tabm_preds
tabm_sub["Id"] = tabm_sub["Id"].astype(np.int64).apply(lambda x: f"{float(x):.18e}")
tabm_sub.to_csv("TabM_submission.csv", index=False)
print("TabM submission saved as TabM_submission.csv")


# Cell 3: Tree–based models: XGB, LGBM, and CatBoost

X_train_tree = train.drop("label", axis=1)
y_train_tree = train["label"].astype(np.uint8)
X_test_tree  = test.copy()

X_train_tree.index = range(len(X_train_tree))
y_train_tree.index = range(len(X_train_tree))
X_test_tree.index  = range(len(X_test_tree))
n_splits = 10
cv = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

def score_metric(y_true, y_preds):
    return roc_auc_score(y_true, y_preds)

# Initialize arrays for out–of–fold and test predictions
oof_preds_xgb = np.zeros(len(X_train_tree))
test_preds_xgb = np.zeros(len(X_test_tree))
oof_preds_lgb = np.zeros(len(X_train_tree))
test_preds_lgb = np.zeros(len(X_test_tree))
oof_preds_cb  = np.zeros(len(X_train_tree))
test_preds_cb  = np.zeros(len(X_test_tree))

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_tree, y_train_tree)):
    print(f"\nFold {fold} - XGB")
    Xtr, Xval = X_train_tree.iloc[train_idx], X_train_tree.iloc[val_idx]
    ytr, yval = y_train_tree.iloc[train_idx], y_train_tree.iloc[val_idx]
    model_xgb = XGBClassifier(
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=6,
        reg_alpha=0.001,
        reg_lambda=0.001,
        colsample_bytree=0.35,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        early_stopping_rounds=100,
    )
    model_xgb.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=500)
    oof_preds_xgb[val_idx] = model_xgb.predict_proba(Xval)[:,1]
    test_preds_xgb += model_xgb.predict_proba(X_test_tree)[:,1] / n_splits

    print(f"\nFold {fold} - LGBM")
    model_lgb = LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=6,
        reg_alpha=0.001,
        reg_lambda=0.001,
        colsample_bytree=0.45,
        random_state=42,
    )
    model_lgb.fit(Xtr, ytr, eval_set=[(Xval, yval)], eval_metric='auc',
                  callbacks=[lgb.early_stopping(100, verbose=500)])
    oof_preds_lgb[val_idx] = model_lgb.predict_proba(Xval)[:,1]
    test_preds_lgb += model_lgb.predict_proba(X_test_tree)[:,1] / n_splits

    print(f"\nFold {fold} - CatBoost")
    model_cb = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=0.25,
        loss_function="Logloss",
        eval_metric="AUC",
        early_stopping_rounds=50,
        verbose=500,
        random_seed=42,
    )
    model_cb.fit(Xtr, ytr, eval_set=(Xval, yval))
    oof_preds_cb[val_idx] = model_cb.predict_proba(Xval)[:,1]
    test_preds_cb += model_cb.predict_proba(X_test_tree)[:,1] / n_splits

# Save individual submissions for each tree model
sub_xgb = sample_submission.copy()
sub_xgb["Predicted"] = test_preds_xgb
sub_xgb["Id"] = sub_xgb["Id"].astype(np.int64).apply(lambda x: f"{float(x):.18e}")
sub_xgb.to_csv("XGB_submission.csv", index=False)

sub_lgb = sample_submission.copy()
sub_lgb["Predicted"] = test_preds_lgb
sub_lgb["Id"] = sub_lgb["Id"].astype(np.int64).apply(lambda x: f"{float(x):.18e}")
sub_lgb.to_csv("LGBM_submission.csv", index=False)

sub_cb = sample_submission.copy()
sub_cb["Predicted"] = test_preds_cb
sub_cb["Id"] = sub_cb["Id"].astype(np.int64).apply(lambda x: f"{float(x):.18e}")
sub_cb.to_csv("CatBoost_submission.csv", index=False)

print("XGB, LGBM, and CatBoost submissions saved.")


# Final simple average ensemble submission (averaging XGB, LGBM, and CatBoost)
final_preds = (test_preds_xgb + test_preds_lgb + test_preds_cb) / 3.0
final_sub = sample_submission.copy()
final_sub["Predicted"] = final_preds
final_sub["Id"] = final_sub["Id"].astype(np.int64).apply(lambda x: f"{float(x):.18e}")
final_sub.to_csv("submission.csv", index=False)
print("Final ensemble submission saved as submission.csv")




