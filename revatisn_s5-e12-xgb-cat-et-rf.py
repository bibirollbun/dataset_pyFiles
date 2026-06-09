import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.linear_model import LogisticRegressionCV

TARGET = 'diagnosed_diabetes'
CATS = ['gender','ethnicity','education_level','income_level',
        'smoking_status','employment_status']
NUMS = ['age','alcohol_consumption_per_week','physical_activity_minutes_per_week',
        'diet_score','sleep_hours_per_day','screen_time_hours_per_day','bmi',
        'waist_to_hip_ratio','systolic_bp','diastolic_bp','heart_rate',
        'cholesterol_total','hdl_cholesterol','ldl_cholesterol','triglycerides',
        'family_history_diabetes','hypertension_history','cardiovascular_history']

# Assuming train and test are already loaded
X = train[CATS + NUMS]
y = train[TARGET]
X_test = test[CATS + NUMS]

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATS)
    ]
)

X_tr_p   = preprocess.fit_transform(X_tr)
X_val_p  = preprocess.transform(X_val)
X_test_p = preprocess.transform(X_test)

print("Training optimized stacking ensemble...")

# ==========================
# Model 1: XGBoost (optimized)
# ==========================
xgb_clf = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=4,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=4,
    reg_lambda=3.0,
    reg_alpha=1.0,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)

xgb_clf.fit(
    X_tr_p, y_tr,
    eval_set=[(X_val_p, y_val)],
    early_stopping_rounds=100,
    verbose=False
)

xgb_val_pred  = xgb_clf.predict_proba(X_val_p)[:, 1]
xgb_test_pred = xgb_clf.predict_proba(X_test_p)[:, 1]

# ==========================
# Model 2: CatBoost (replaces LightGBM) - handles categoricals natively
# ==========================
cat_clf = CatBoostClassifier(
    iterations=2000,
    depth=5,
    learning_rate=0.02,
    l2_leaf_reg=4.0,
    random_strength=0.3,
    bagging_temperature=0.8,
    border_count=128,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=False,
    thread_count=-1
)

cat_clf.fit(
    X_tr_p, y_tr,
    eval_set=(X_val_p, y_val),
    early_stopping_rounds=100,
    use_best_model=True
)

cat_val_pred  = cat_clf.predict_proba(X_val_p)[:, 1]
cat_test_pred = cat_clf.predict_proba(X_test_p)[:, 1]

# ==========================
# Model 3: ExtraTrees (replaces sklearn GBC) - fast, diverse tree ensemble
# ==========================
et_clf = ExtraTreesClassifier(
    n_estimators=1500,
    max_depth=8,
    min_samples_split=8,
    min_samples_leaf=4,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

et_clf.fit(X_tr_p, y_tr)
et_val_pred  = et_clf.predict_proba(X_val_p)[:, 1]
et_test_pred = et_clf.predict_proba(X_test_p)[:, 1]

# ==========================
# Model 4: RandomForest (additional diversity)
# ==========================
rf_clf = RandomForestClassifier(
    n_estimators=1200,
    max_depth=10,
    min_samples_split=12,
    min_samples_leaf=6,
    max_features=0.3,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

rf_clf.fit(X_tr_p, y_tr)
rf_val_pred  = rf_clf.predict_proba(X_val_p)[:, 1]
rf_test_pred = rf_clf.predict_proba(X_test_p)[:, 1]

# ==========================
# Stacking with LogisticRegressionCV meta-model
# ==========================
stack_X_val  = np.column_stack((xgb_val_pred, cat_val_pred, et_val_pred, rf_val_pred))
stack_X_test = np.column_stack((xgb_test_pred, cat_test_pred, et_test_pred, rf_test_pred))

meta_clf = LogisticRegressionCV(
    Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
    cv=5,
    scoring='roc_auc',
    penalty='elasticnet',
    l1_ratios=[0.1, 0.5, 0.9],
    solver='saga',
    max_iter=1000,
    n_jobs=-1,
    refit=True,
    random_state=42
)

meta_clf.fit(stack_X_val, y_val)
stack_val_pred  = meta_clf.predict_proba(stack_X_val)[:, 1]
stack_test_pred = meta_clf.predict_proba(stack_X_test)[:, 1]

# Results
auc_score = roc_auc_score(y_val, stack_val_pred)
print(f"Final Stacking Ensemble AUC: {auc_score:.6f}")

# Submission
sub = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': stack_test_pred
})

sub.to_csv('submission_xgb_cat_et_rf_stacking.csv', index=False)
print("Submission saved as 'submission_xgb_cat_et_rf_stacking.csv'")





