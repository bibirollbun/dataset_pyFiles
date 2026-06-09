!pip uninstall -y scikit-learn autogluon autogluon.tabular 
!pip install scikit-learn==1.4.2
!pip install autogluon.tabular==1.5.0
!pip install lightgbm catboost

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from autogluon.tabular import TabularPredictor
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import torch

print("GPU Available:", torch.cuda.is_available())


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

X = train_df.drop(columns=["diagnosed_diabetes"])
y = train_df["diagnosed_diabetes"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def feature_engineering(df):
    df = df.copy()
    df['bmi_age'] = df['bmi'] / (df['age'] + 1)
    df['age_squared'] = df['age'] ** 2
    df['waist_bmi_ratio'] = df['waist_to_hip_ratio'] / (df['bmi'] + 1)
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
    df['activity_per_age'] = df['physical_activity_minutes_per_week'] / (df['age'] + 1)
    return df

X_train = feature_engineering(X_train)
X_val = feature_engineering(X_val)
X_test = feature_engineering(test_df.copy())
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_val[col] = X_val[col].astype('category')
    X_test[col] = X_test[col].astype('category')


cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
num_cols = X_train.select_dtypes(exclude=['object', 'category']).columns


preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ('num', 'passthrough', num_cols)
])

rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)
rf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('rf', rf)
])

cal_rf = CalibratedClassifierCV(rf_pipeline, method='isotonic', cv=5)
cal_rf.fit(X_train, y_train)


lgb = LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=64,
    subsample=0.85,
    colsample_bytree=0.85,
    device='gpu',
    random_state=42
)

lgb.fit(X_train, y_train)


cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

cat = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    task_type='GPU',
    verbose=200,
    random_seed=42
)

cat.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_val, y_val),
    early_stopping_rounds=100,
    use_best_model=True
)


train_ag = X_train.copy()
train_ag["diagnosed_diabetes"] = y_train.values

predictor = TabularPredictor(
    label="diagnosed_diabetes",
    eval_metric="roc_auc"
).fit(
    train_data=train_ag,
    presets="best_quality",
    num_stack_levels=2,          # ğŸ”¥ critical
    num_bag_folds=5,             # ğŸ”¥ stability
    ag_args_fit={'num_gpus': 1},
    verbosity=2
)


pred_rf_val   = cal_rf.predict_proba(X_val)[:,1]
pred_lgb_val  = lgb.predict_proba(X_val)[:,1]
pred_cat_val  = cat.predict_proba(X_val)[:,1]
pred_auto_val = predictor.predict_proba(X_val)[predictor.class_labels[-1]]


best_auc = 0
best_w = None
step = 0.01

for w_rf in np.arange(0,1+step,step):
    for w_lgb in np.arange(0,1+step-w_rf,step):
        for w_cat in np.arange(0,1+step-w_rf-w_lgb,step):
            w_auto = 1 - w_rf - w_lgb - w_cat
            if w_auto < 0:
                continue

            ens = (
                w_rf*pred_rf_val +
                w_lgb*pred_lgb_val +
                w_cat*pred_cat_val +
                w_auto*pred_auto_val
            )

            auc = roc_auc_score(y_val, ens)
            if auc > best_auc:
                best_auc = auc
                best_w = (w_rf, w_lgb, w_cat, w_auto)

print("Best weights:", best_w)
print("Best AUC:", best_auc)


X_full = feature_engineering(X)
train_ag_full = X_full.copy()
train_ag_full["diagnosed_diabetes"] = y.values

predictor = TabularPredictor(
    label="diagnosed_diabetes",
    eval_metric="roc_auc"
).fit(
    train_data=train_ag_full,
    presets='best_quality',
    verbosity=2,
    num_stack_levels=1,
    ag_args_fit={'num_gpus': 1}
)

cal_rf.fit(X_full, y)


from sklearn.preprocessing import OrdinalEncoder

# 1ï¸�âƒ£ Define categorical columns (same as before)
cat_cols = ['gender','ethnicity','education_level','income_level','smoking_status','employment_status']
num_cols = [c for c in X_full.columns if c not in cat_cols]

# 2ï¸�âƒ£ Fit encoder on original train X
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(X[cat_cols])

# 3ï¸�âƒ£ Encode full dataset for LGBM and CatBoost
X_full_enc = X_full.copy()
X_full_enc[cat_cols] = encoder.transform(X_full[cat_cols])
X_full_enc[num_cols] = X_full_enc[num_cols].astype(float)

# 4ï¸�âƒ£ Encode test set
X_test_enc = X_test.copy()
X_test_enc[cat_cols] = encoder.transform(X_test[cat_cols])
X_test_enc[num_cols] = X_test_enc[num_cols].astype(float)

# 5ï¸�âƒ£ Train LightGBM on encoded full dataset
lgb.fit(X_full_enc, y)

# 6ï¸�âƒ£ Train CatBoost on encoded full dataset
cat.fit(X_full_enc, y)

# 7ï¸�âƒ£ Predict on test set
pred_rf_test   = cal_rf.predict_proba(X_test)[:,1]        # pipeline handles categorical automatically
pred_lgb_test  = lgb.predict_proba(X_test_enc)[:,1]
pred_cat_test  = cat.predict_proba(X_test_enc)[:,1]
pred_auto_test = predictor.predict_proba(X_test)[predictor.class_labels[-1]]  # AutoGluon already fitted

# 8ï¸�âƒ£ Ensemble with previously found best weights
w_rf, w_lgb, w_cat, w_auto = best_w
ensemble_test = (
    w_rf*pred_rf_test +
    w_lgb*pred_lgb_test +
    w_cat*pred_cat_test +
    w_auto*pred_auto_test
)



submission = pd.DataFrame({
    "id": test_df["id"],
    "diagnosed_diabetes": ensemble_test
})

submission.to_csv("submission_final.csv", index=False)
print("âœ… Submission saved as submission_final.csv")

