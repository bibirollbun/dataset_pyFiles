import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb
from cuml.preprocessing.TargetEncoder import TargetEncoder

import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


cat_cols = [ 'gender', 'ethnicity', 'education_level', 'income_level', 
            'smoking_status', 'employment_status', 'cardiovascular_history', 
            'hypertension_history', 'family_history_diabetes' ] 

num_cols = [ 'age', 'alcohol_consumption_per_week', 'bmi', 'cholesterol_total', 
            'diastolic_bp', 'diet_score', 'hdl_cholesterol', 'heart_rate', 'ldl_cholesterol', 
            'physical_activity_minutes_per_week', 'screen_time_hours_per_day', 'sleep_hours_per_day',
            'systolic_bp', 'triglycerides', 'waist_to_hip_ratio' ]


target = 'diagnosed_diabetes'


# Combine all data
test[target] = -1
combine = pd.concat([train, test], axis=0, ignore_index=True)


print(f"\nTrain: {train.shape}")
print(f"Test:  {test.shape}")

print(f"\nCombined data: {combine.shape}")


df=combine.copy()


#ğŸ§  Ratios & Risk Scores
# Blood pressure pulse
df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']

# Cholesterol ratios
df['chol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
df['ldl_hdl_ratio']  = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)

# Triglyceride to HDL ratio
df['tg_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-5)


# Central obesity flag
df['central_obesity'] = (df['waist_to_hip_ratio'] > 0.9).astype(int)


#ğŸ�ƒLifestyle & Behavioral Features
df['activity_per_day'] = df['physical_activity_minutes_per_week'] / 7

# Screen vs sleep imbalance
df['screen_sleep_ratio'] = df['screen_time_hours_per_day'] / (df['sleep_hours_per_day'] + 1e-5)

# Alcohol risk flag
df['heavy_alcohol'] = (
    df['alcohol_consumption_per_week'] >
    df['alcohol_consumption_per_week'].quantile(0.9)
).astype(int)


# Risk scores
df['cardio_risk_score'] = (
    df['hypertension_history'].astype(int) +
    df['cardiovascular_history'].astype(int) +
    (df['cholesterol_total'] > 240).astype(int)
)

df['lifestyle_risk_score'] = (
    (df['smoking_status'] == 'current').astype(int) +
    (df['physical_activity_minutes_per_week'] < 150).astype(int) +
    (df['sleep_hours_per_day'] < 6).astype(int)
)


# Interaction Features
df['bmi_x_activity'] = df['bmi'] * df['physical_activity_minutes_per_week']
df['age_x_bp'] = df['age'] * df['systolic_bp']
df['ldl_x_hdl'] = df['ldl_cholesterol'] * df['hdl_cholesterol']
df['chol_x_trig'] = df['cholesterol_total'] * df['triglycerides']
df['activity_x_age'] = df['physical_activity_minutes_per_week'] * df['age']


NEW_FEATURES = ['pulse_pressure', 'chol_hdl_ratio', 'ldl_hdl_ratio', 
                'tg_hdl_ratio', 'central_obesity', 'activity_per_day',
                'cardio_risk_score', 'lifestyle_risk_score', 'bmi_x_activity',
                'age_x_bp', 'ldl_x_hdl',
                'chol_x_trig', 'activity_x_age']
print(f"Created {len(NEW_FEATURES)} new features")


df.info()


CATS = cat_cols.copy()
NUMS = num_cols + [f for f in NEW_FEATURES]

# Create factorized versions of numerics
CATS_NUM = []
SIZES = {}

for c in NUMS:
    n = f"{c}_cat"
    CATS_NUM.append(n)
    df[n], _ = df[c].factorize()
    SIZES[n] = df[n].max() + 1
    df[n] = df[n].astype('int32')

print(f"Created {len(CATS_NUM)} categorical numeric features")


CATS_NUM


IMPORTANT_NUM_CATS = [
    'age_cat',
    'bmi_cat',
    'waist_to_hip_ratio_cat',
    'cardio_risk_score_cat',
    'central_obesity_cat'
]

IMPORTANT_CATS = [
    'gender',
    'ethnicity',
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history'
]


CATS_INTER = []

for num_cat in IMPORTANT_NUM_CATS:
    for cat in IMPORTANT_CATS:
        name = f"{num_cat}_{cat}"
        df[name] = df[num_cat].astype(str) + '_' + df[cat].astype(str)
        CATS_INTER.append(name)

print(f"Created {len(CATS_INTER)} numeric-categorical interactions")


# Count encoding
CE = []
ALL_CATS = CATS + CATS_NUM + CATS_INTER

print(f"\nCreating count encoding for {len(ALL_CATS)} categorical features...")
for i, c in enumerate(ALL_CATS):
    if i % 20 == 0:
        print(f"  Progress: {i}/{len(ALL_CATS)}")
    tmp = df.groupby(c)[target].count()
    tmp.name = f"CE_{c}"
    CE.append(f"CE_{c}")
    df = df.merge(tmp, on=c, how='left')

print(f"Created {len(CE)} count encodings")


# Split back
train = df.iloc[:len(train)].copy()
test = df.iloc[len(train):len(train) + len(test)].copy()

print(f"\nTrain: {train.shape}, Test: {test.shape}")


FOLDS = 8  # Increased from 7 for more stability
SEED = 42

params = {
    'objective':"binary:logistic",
    'eval_metric':"logloss",
    'tree_method':"gpu_hist",
    'random_state':42,
    "learning_rate": 0.14141008604431932,
    "max_depth": 4,
    "min_child_weight": 6,
    "subsample": 0.6043858586325254,
    "colsample_bytree": 0.9076381150343775,
    "gamma": 2.9365900809348835,
    "reg_lambda": 7.8511309801919715,
    "reg_alpha": 4.365923016193024,

}

print("\nModel Parameters:")
for k, v in params.items():
    print(f"  {k}: {v}")


FEATURES = NUMS + CATS + CATS_NUM + CATS_INTER + CE


oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print(f"\n{'='*25}")
    print(f"Fold {fold+1}/{FOLDS}")
    print('='*25)

    # -----------------------------
    # Split data
    # -----------------------------
    Xy_train = train.iloc[train_idx][FEATURES + [target]].copy()
    X_valid = train.iloc[val_idx][FEATURES].copy()
    y_valid = train.iloc[val_idx][target]
    X_test = test[FEATURES].copy()

    # -----------------------------
    # Target Encoding (OOF-safe)
    # -----------------------------
    TARGET_ENCODE_CATS = CATS_NUM + CATS_INTER
    print(f"Target encoding {len(TARGET_ENCODE_CATS)} features...")

    for c in TARGET_ENCODE_CATS:
        TE = TargetEncoder(
            n_folds=10,
            smooth=0,
            split_method='random',
            stat='mean'
        )

        Xy_train[c] = TE.fit_transform(
            Xy_train[[c]], Xy_train[target]
        ).astype('float32')

        X_valid[c] = TE.transform(
            X_valid[[c]]
        ).astype('float32')

        X_test[c] = TE.transform(
            X_test[[c]]
        ).astype('float32')

    # -----------------------------
    # Set categorical dtype
    # -----------------------------
    for c in CATS:
        Xy_train[c] = Xy_train[c].astype('category')
        X_valid[c] = X_valid[c].astype('category')
        X_test[c] = X_test[c].astype('category')

    # -----------------------------
    # Create DMatrix
    # -----------------------------
    dtrain = xgb.QuantileDMatrix(
        Xy_train[FEATURES],
        label=Xy_train[target],
        enable_categorical=True,
        max_bin=256
    )

    dval = xgb.DMatrix(
        X_valid,
        label=y_valid,
        enable_categorical=True
    )

    dtest = xgb.DMatrix(
        X_test,
        enable_categorical=True
    )

    # -----------------------------
    # Train
    # -----------------------------
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=12000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=350,
        verbose_eval=500
    )

    # -----------------------------
    # Predict
    # -----------------------------
    oof_preds[val_idx] = model.predict(
        dval, iteration_range=(0, model.best_iteration + 1)
    )

    test_preds += model.predict(
        dtest, iteration_range=(0, model.best_iteration + 1)
    ) / FOLDS

    # -----------------------------
    # Metrics
    # -----------------------------
    fold_auc = roc_auc_score(y_valid, oof_preds[val_idx])
    fold_scores.append(fold_auc)
    
    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")


print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)

overall_auc = roc_auc_score(train[target], oof_preds)

print(f"\nFold Scores:")
for i, score in enumerate(fold_scores, 1):
    print(f"  Fold {i}: {score:.5f}")

print(f"\nOverall OOF AUC: {overall_auc:.5f}")
print(f"Mean Fold AUC:  {np.mean(fold_scores):.5f}")
print(f"Std Fold AUC:   {np.std(fold_scores):.5f}")

print(f"New OOF:         {overall_auc:.5f}")
print(f"Expected LB:     {overall_auc + 0.00035:.5f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, FOLDS+1), fold_scores, color='steelblue', edgecolor='black')
axes[0].axhline(overall_auc, color='red', linestyle='--', label=f'Overall: {overall_auc:.5f}')
axes[0].set_xlabel('Fold')
axes[0].set_ylabel('AUC')
axes[0].set_title('Cross-Validation Fold Scores', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(oof_preds, bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Predicted Probability')
axes[1].set_ylabel('Frequency')
axes[1].set_title('OOF Prediction Distribution', fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Feature importance
fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=20, importance_type='gain', ax=ax)
plt.title("Top 20 Features (XGBoost)", fontweight='bold')
plt.tight_layout()
plt.show()


submission = pd.DataFrame({
    'id': test['id'],
    target: test_preds
})

submission.to_csv('submission.csv', index=False)

