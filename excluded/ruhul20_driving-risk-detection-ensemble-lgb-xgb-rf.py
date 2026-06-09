import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/distracted-driving-risk-detection-challenge/kaggle_train.csv')

train


train.info()


train.isnull().sum()


train.describe()


train['risk_level'].value_counts()


plt.figure(figsize=(10, 6))
datalabel = sns.countplot(x='risk_level', data=train, palette='viridis')

for i in datalabel.containers:
    datalabel.bar_label(i)


plt.title('Distribution of risk_level')
plt.xlabel('risk_level')
plt.ylabel('Count')
plt.savefig('Distribution of risk_level.png')
plt.show()


# data = train.drop(columns=['label_source','observation_hour'], axis=1)
# columns = data.columns
# columns


from sklearn.metrics import  confusion_matrix, classification_report, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, roc_curve

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
import joblib

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler , LabelEncoder
from sklearn.pipeline import Pipeline


X = train.drop(['risk_level'], axis=1)

y = train['risk_level']


df = X.copy()
#df['readmitted'] = df['readmitted'].map({'No':0, '>30':})

label_encoder = LabelEncoder()

cat_cols = df.select_dtypes(include=['object','category']).columns  # pick categorical columns

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

df.head()


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# X_train.shape


numeric_features = list(df.select_dtypes(include=['int64', 'float64']).columns)


# standard_scaler = StandardScaler()
# X_sc = standard_scaler.fit_transform(df[numeric_features])

# X_sc
# X_train[numeric_features] = standard_scaler.fit_transform(X_train[numeric_features])

# X_test[numeric_features] = standard_scaler.transform(X_test[numeric_features])

# X_train.describe()


# ---------- Model parameters (tweak as desired) ----------
rf_params = dict(random_state=42, n_jobs=-1, n_estimators=800, verbose=0)

lgb_params = dict(
    n_estimators=1500, learning_rate=0.07, num_leaves=93, max_depth=10,
    colsample_bytree=0.975,
    random_state=42, n_jobs=-1, verbosity=-1
)

xgb_params = dict(
    tree_method="hist",
    max_depth=10, learning_rate=0.0669438, n_estimators=800,
    random_state=42, n_jobs=-1, verbosity=0
)

cat_params = dict(
    iterations=1500,
    learning_rate=0.07,
    depth=12,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=False,   # set to True/10 if you want CatBoost logs
    task_type='CPU'
)

# If you have categorical feature columns (names or indices), set here. Otherwise leave None.
cat_features_idx = None

# Ensemble weights (must sum to 1 ideally, but not required)
w_xgb = 0.5
w_lgb = 0.15
w_cat = 0.15
w_rf  = 0.2

# CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# ------------------ Sanity checks & class mapping ------------------
# Ensure df and y exist
if 'df' not in globals() or 'y' not in globals():
    raise RuntimeError("Please create variables `df` (features DataFrame) and `y` (labels Series) before running this script.")

# Allow y to be a pd.Series / numpy array
y_series = pd.Series(y).reset_index(drop=True)
if len(y_series) != len(df):
    raise RuntimeError("Length mismatch: `df` and `y` must have the same number of rows.")

classes = np.sort(y_series.unique())
n_classes = len(classes)
print("Detected classes:", classes, "| n_classes =", n_classes)

# Map labels to 0..n_classes-1 for consistent training/order
label_to_idx = {lab: i for i, lab in enumerate(classes)}
idx_to_label = {i: lab for lab, i in label_to_idx.items()}
y_idx = y_series.map(label_to_idx).values

# ------------------ Containers ------------------
oof_preds = np.zeros((len(df), n_classes), dtype=float)
fold_scores_auc = []
fold_scores_acc = []

print("\nStarting multiclass CV (ensemble of RF / LGB / XGB / CatBoost)...")

# ------------------ CV loop ------------------
for fold, (tr_idx, val_idx) in enumerate(skf.split(df, y_idx), start=1):
    print(f"\n--- Fold {fold}/{n_splits} ---")
    X_tr, X_val = df.iloc[tr_idx], df.iloc[val_idx]
    y_tr_idx, y_val_idx = y_idx[tr_idx], y_idx[val_idx]

    # instantiate fresh models
    rf_model = RandomForestClassifier(**rf_params)
    lgb_model = LGBMClassifier(**lgb_params)
    xgb_model = XGBClassifier(**xgb_params)
    cat_model = CatBoostClassifier(**cat_params)

    # 1) RandomForest
    rf_model.fit(X_tr, y_tr_idx)
    rf_val_proba = rf_model.predict_proba(X_val)

    # 2) LightGBM (multi-class)
    lgb_model.fit(
        X_tr, y_tr_idx,
        eval_set=[(X_val, y_val_idx)],
        eval_metric='multi_logloss',
        #verbose=0
    )
    lgb_val_proba = lgb_model.predict_proba(X_val)

    # 3) XGBoost (multi-class)
    xgb_model.fit(
        X_tr, y_tr_idx,
        eval_set=[(X_val, y_val_idx)],
        eval_metric='mlogloss',
        verbose=False
    )
    xgb_val_proba = xgb_model.predict_proba(X_val)

    # 4) CatBoost (multi-class)
    if cat_features_idx:
        cat_model.fit(
            X_tr, y_tr_idx,
            eval_set=(X_val, y_val_idx),
            cat_features=cat_features_idx,
            use_best_model=True,
            verbose=False
        )
    else:
        cat_model.fit(
            X_tr, y_tr_idx,
            eval_set=(X_val, y_val_idx),
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=False
        )
    cat_val_proba = cat_model.predict_proba(X_val)

    # ----- Weighted ensemble (probability vectors) -----
    val_pred_proba = (
        w_xgb * xgb_val_proba +
        w_lgb * lgb_val_proba +
        w_cat * cat_val_proba +
        w_rf  * rf_val_proba
    )

    # Save OOF probabilities
    oof_preds[val_idx, :] = val_pred_proba

    # Compute multiclass ROC-AUC (one-vs-rest macro)
    try:
        auc = roc_auc_score(y_val_idx, val_pred_proba, multi_class='ovr', average='macro')
    except Exception as e:
        auc = float('nan')
        print("Warning computing ROC-AUC for fold:", e)

    # Compute accuracy (argmax)
    val_pred_idx = np.argmax(val_pred_proba, axis=1)
    acc = accuracy_score(y_val_idx, val_pred_idx)

    fold_scores_auc.append(auc)
    fold_scores_acc.append(acc)

    print(f"Fold {fold} ROC-AUC (macro OVR): {auc:.6f} | Accuracy: {acc:.6f}")

    

# ------------------ CV Summary ------------------
print("\n=== CV Summary ===")
print("Fold AUCs      :", [round(s, 6) for s in fold_scores_auc])
print(f"Mean CV AUC    : {np.mean(fold_scores_auc):.6f} (+/- {np.std(fold_scores_auc):.6f})")
print("Fold Accuracies:", [round(a, 6) for a in fold_scores_acc])
print(f"Mean CV Acc    : {np.mean(fold_scores_acc):.6f} (+/- {np.std(fold_scores_acc):.6f})")

# OOF metrics
try:
    oof_auc = roc_auc_score(y_idx, oof_preds, multi_class='ovr', average='macro')
except Exception as e:
    oof_auc = float('nan')
    print("Warning computing OOF AUC:", e)
oof_pred_idx = np.argmax(oof_preds, axis=1)
oof_acc = accuracy_score(y_idx, oof_pred_idx)

print(f"OOF multiclass AUC: {oof_auc:.6f}")
print(f"OOF Accuracy      : {oof_acc:.6f}")

# Print classification report and confusion matrix (mapped back to original labels)
y_true_labels = [idx_to_label[i] for i in y_idx]
y_pred_labels = [idx_to_label[i] for i in oof_pred_idx]

print("\nClassification report (OOF):")
print(classification_report(y_true_labels, y_pred_labels, digits=4))

print("\nConfusion matrix (rows=true, cols=pred) with label order:", list(classes))
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=list(classes))
print(cm)

# Optionally save OOF predictions for analysis
if True:
    oof_df = pd.DataFrame(oof_preds, columns=[f"prob_{lab}" for lab in classes])
    oof_df['true_label'] = y_series.values
    oof_df['pred_label'] = [idx_to_label[i] for i in oof_pred_idx]
    oof_df.to_csv('oof_predictions.csv', index=False)
    print("\nSaved oof_predictions.csv (probs + true_label + pred_label).")


sample_sub = pd.read_csv('/kaggle/input/distracted-driving-risk-detection-challenge/kaggle_sample_submission.csv')

sample_sub


test = pd.read_csv('/kaggle/input/distracted-driving-risk-detection-challenge/kaggle_test.csv')

test


#cat_cols = test.select_dtypes(include=['object','category']).columns  # pick categorical columns

test[col] = label_encoder.fit_transform(test['label_source'].astype(str))

test.head()


test_preds_folds = []  # will store (n_test, n_classes) per fold if X_test exists

rf_test_proba = rf_model.predict_proba(test)
lgb_test_proba = lgb_model.predict_proba(test)
xgb_test_proba = xgb_model.predict_proba(test)
cat_test_proba = cat_model.predict_proba(test)

test_pred_fold = (
    w_xgb * xgb_test_proba +
    w_lgb * lgb_test_proba +
    w_cat * cat_test_proba +
    w_rf  * rf_test_proba
)
test_preds_folds.append(test_pred_fold)


# ------------------ Optional: Create Kaggle-style submission (updated for sample_submission with 'id' and 'risk_level') ------------------
if len(test_preds_folds) > 0:
    print("\nBuilding ensemble test predictions from fold outputs...")
    test_preds_avg = np.mean(np.stack(test_preds_folds, axis=0), axis=0)  # shape (n_test, n_classes)

    # We expect sample_submission.csv to have columns: ['id', 'risk_level']
    try:
        sample_submission = pd.read_csv('/kaggle/input/distracted-driving-risk-detection-challenge/kaggle_sample_submission.csv')
        if 'id' not in sample_submission.columns:
            raise RuntimeError("sample_submission.csv must contain an 'id' column.")

        # Predicted label index per test row (0..n_classes-1)
        pred_idx = np.argmax(test_preds_avg, axis=1)
        # Map back to original label values (e.g., 1,2,3,4 or strings)
        pred_labels = [idx_to_label[i] for i in pred_idx]

        # If sample_submission already has exactly the columns ['id','risk_level'], overwrite risk_level
        if 'risk_level' in sample_submission.columns:
            submission = sample_submission.copy()
            submission['risk_level'] = pred_labels
        else:
            # Otherwise create a new DataFrame with id and risk_level
            submission = pd.DataFrame({
                'id': sample_submission.iloc[:, 0],
                'risk_level': pred_labels
            })

        # Ensure index alignment length matches predicted rows
        if len(submission) != test_preds_avg.shape[0]:
            # If lengths mismatch, try to align by using X_test length (if available)
            if 'X_test' in globals() and len(X_test) == test_preds_avg.shape[0]:
                # Recreate submission using X_test order and sample_submission ids if they match length
                if len(sample_submission) == len(X_test):
                    submission = pd.DataFrame({
                        'id': sample_submission['id'],
                        'risk_level': pred_labels
                    })
                else:
                    raise RuntimeError("Length mismatch between sample_submission, X_test and model predictions. Check ordering.")
            else:
                raise RuntimeError("Length mismatch between sample_submission and model predictions. Check ordering.")

    except FileNotFoundError:
        # If sample_submission.csv isn't present, fallback to using test_ids or X_test indices
        print("sample_submission.csv not found — falling back to test_ids / X_test indices.")
        if 'test_ids' in globals():
            idxs = test_ids
        elif 'X_test' in globals():
            idxs = np.arange(len(X_test))
        else:
            raise RuntimeError("No sample_submission.csv and no test ids/X_test — cannot build submission.")

        pred_idx = np.argmax(test_preds_avg, axis=1)
        pred_labels = [idx_to_label[i] for i in pred_idx]
        submission = pd.DataFrame({'id': idxs, 'risk_level': pred_labels})

    # Cast risk_level to same dtype as sample_submission if possible (helpful if sample uses ints)
    try:
        # If the sample had numeric risk_level dtype, convert predictions accordingly
        if 'sample_submission' in locals() and sample_submission['risk_level'].dtype.kind in 'iu':
            submission['risk_level'] = submission['risk_level'].astype(int)
    except Exception:
        pass

    submission.to_csv('submission.csv', index=False)
    print("Saved submission.csv with columns:", list(submission.columns))
    print("First rows:\n", submission.head())
else:
    print("\nNo X_test/test predictions collected; skipping submission creation. "
          "Provide X_test and optionally sample_submission.csv to build one.")



submission

