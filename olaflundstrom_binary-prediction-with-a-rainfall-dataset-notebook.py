import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Model selection, evaluation, and preprocessing
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA

# Base models and ensemble methods
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Feature selection and calibration
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV

# Advanced meta-learners
from sklearn.neural_network import MLPClassifier

# Hyperparameter optimization
import optuna

# Blending weight optimization
from scipy.optimize import differential_evolution

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
sample_submission_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
display(train_df.head())


plt.figure(figsize=(6,4))
sns.countplot(x='rainfall', data=train_df)
plt.title("Target Distribution")
plt.show()


INCLUDE_POLY = True
features = [col for col in train_df.columns if col not in ['id', 'rainfall']]

steps = [('imputer', SimpleImputer(strategy='median'))]
if INCLUDE_POLY:
    steps.append(('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)))
steps.append(('scaler', StandardScaler()))
num_transformer = Pipeline(steps=steps)
preprocessor = ColumnTransformer(transformers=[('num', num_transformer, features)])

X_train_pre = preprocessor.fit_transform(train_df[features])
X_test_pre = preprocessor.transform(test_df[features])

if INCLUDE_POLY:
    poly_feature_names = preprocessor.named_transformers_['num'].named_steps['poly'].get_feature_names_out(features)
    final_feature_names = poly_feature_names
else:
    final_feature_names = features

X_train_proc = pd.DataFrame(X_train_pre, columns=final_feature_names)
X_test_proc = pd.DataFrame(X_test_pre, columns=final_feature_names)


xgb_fs = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
xgb_fs.fit(X_train_proc, train_df['rainfall'])
selector = SelectFromModel(xgb_fs, threshold='median', prefit=True)
X_train_sel = selector.transform(X_train_proc)
X_test_sel = selector.transform(X_test_proc)
print("Features before selection:", X_train_proc.shape[1], "after selection:", X_train_sel.shape[1])

APPLY_PCA = False  # Toggle PCA here if desired
if APPLY_PCA:
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_train_sel = pca.fit_transform(X_train_sel)
    X_test_sel = pca.transform(X_test_sel)
    print("After PCA, feature count:", X_train_sel.shape[1])


X = pd.DataFrame(X_train_sel)
y = train_df['rainfall']
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print("Training split:", X_tr.shape, "Validation split:", X_val.shape)


def objective_xgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE
    }
    model = xgb.XGBClassifier(**params)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=RANDOM_STATE)
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        X_cv_train, X_cv_test = X.iloc[train_idx], X.iloc[test_idx]
        y_cv_train, y_cv_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_cv_train, y_cv_train, verbose=False)
        y_pred = model.predict_proba(X_cv_test)[:,1]
        scores.append(roc_auc_score(y_cv_test, y_pred))
    return np.mean(scores)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=20)
print("XGBoost best params:", study_xgb.best_params)
best_xgb = xgb.XGBClassifier(**study_xgb.best_params, use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
best_xgb.fit(X, y)

def objective_lgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 15, 50),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'random_state': RANDOM_STATE
    }
    model = lgb.LGBMClassifier(**params)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=RANDOM_STATE)
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        X_cv_train, X_cv_test = X.iloc[train_idx], X.iloc[test_idx]
        y_cv_train, y_cv_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_cv_train, y_cv_train)  # Removed 'verbose'
        y_pred = model.predict_proba(X_cv_test)[:,1]
        scores.append(roc_auc_score(y_cv_test, y_pred))
    return np.mean(scores)

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=20)
print("LightGBM best params:", study_lgb.best_params)
best_lgb = lgb.LGBMClassifier(**study_lgb.best_params, random_state=RANDOM_STATE)
best_lgb.fit(X, y)


model_lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
model_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced')
model_xgb = best_xgb
model_lgb = best_lgb
model_cat = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
model_et = ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE)
model_svm_sigmoid = CalibratedClassifierCV(SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE), cv=3, method='sigmoid')
model_svm_isotonic = CalibratedClassifierCV(SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE), cv=3, method='isotonic')
model_knn_isotonic = CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=5), cv=3, method='isotonic')

base_models = {
    "LogisticRegression": model_lr,
    "RandomForest": CalibratedClassifierCV(model_rf, cv=3),
    "XGBoost": CalibratedClassifierCV(model_xgb, cv=3),
    "LightGBM": CalibratedClassifierCV(model_lgb, cv=3),
    "CatBoost": CalibratedClassifierCV(model_cat, cv=3),
    "ExtraTrees": CalibratedClassifierCV(model_et, cv=3),
    "SVM_sigmoid": model_svm_sigmoid,
    "SVM_isotonic": model_svm_isotonic,
    "KNN_isotonic": model_knn_isotonic
}

for name, model in base_models.items():
    scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc', n_jobs=-1)
    print(f"{name} ROC AUC: {scores.mean():.4f}")


estimators = [(name, model) for name, model in base_models.items()]

meta_mlp = MLPClassifier(hidden_layer_sizes=(50, 20), activation='relu', 
                         max_iter=1500, early_stopping=True, validation_fraction=0.2, random_state=RANDOM_STATE)
meta_lr = LogisticRegression(penalty='l2', solver='liblinear', random_state=RANDOM_STATE)
meta_rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)

stacking_mlp = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_mlp,
    cv=3,  # Reduced CV folds for speed
    n_jobs=-1,
    passthrough=True
)

stacking_lr = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_lr,
    cv=3,
    n_jobs=-1,
    passthrough=True
)

stacking_rf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_rf,
    cv=3,
    n_jobs=-1,
    passthrough=True
)

score_mlp = cross_val_score(stacking_mlp, X, y, cv=3, scoring='roc_auc', n_jobs=-1).mean()
score_lr = cross_val_score(stacking_lr, X, y, cv=3, scoring='roc_auc', n_jobs=-1).mean()
score_rf = cross_val_score(stacking_rf, X, y, cv=3, scoring='roc_auc', n_jobs=-1).mean()
print("Stacking MLP ROC AUC:", score_mlp)
print("Stacking LR ROC AUC:", score_lr)
print("Stacking RF ROC AUC:", score_rf)


def get_oof_predictions(model, X, y, cv):
    return cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:,1]

cv_fast = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=RANDOM_STATE)
oof_mlp = get_oof_predictions(stacking_mlp, X, y, cv_fast)
oof_lr = get_oof_predictions(stacking_lr, X, y, cv_fast)
oof_rf = get_oof_predictions(stacking_rf, X, y, cv_fast)

# Combine OOF predictions: each column corresponds to one stacking ensemble's predictions.
oof_stack = np.vstack([oof_mlp, oof_lr, oof_rf]).T

def blend_objective(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blended = np.dot(oof_stack, weights)
    return -roc_auc_score(y, blended)

bounds = [(0, 1)] * 3
result = differential_evolution(blend_objective, bounds, seed=RANDOM_STATE)
optimal_weights = result.x / np.sum(result.x)
print("Optimal blending weights:")
print(f"  Stacking MLP: {optimal_weights[0]:.4f}")
print(f"  Stacking LR:  {optimal_weights[1]:.4f}")
print(f"  Stacking RF:  {optimal_weights[2]:.4f}")

blended_cv_auc = roc_auc_score(y, np.dot(oof_stack, optimal_weights))
print("Blended Stacking Ensemble ROC AUC (OOF):", blended_cv_auc)


stacking_mlp.fit(X, y)
stacking_lr.fit(X, y)
stacking_rf.fit(X, y)

stacking_mlp_test = stacking_mlp.predict_proba(pd.DataFrame(X_test_sel))[:,1]
stacking_lr_test = stacking_lr.predict_proba(pd.DataFrame(X_test_sel))[:,1]
stacking_rf_test = stacking_rf.predict_proba(pd.DataFrame(X_test_sel))[:,1]

final_stack_preds = (optimal_weights[0] * stacking_mlp_test + 
                     optimal_weights[1] * stacking_lr_test + 
                     optimal_weights[2] * stacking_rf_test)


submission = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': final_stack_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")

