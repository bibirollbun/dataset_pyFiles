import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import warnings
import gc

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']

SEED = 2025
TARGET = 'diagnosed_diabetes'

print("Libraries loaded!")
print(f"Optuna version: {optuna.__version__}")
print(f"XGBoost version: {xgb.__version__}")


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# Load external data for reference
try:
    original = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_012_health_indicators_BRFSS2015.csv')
    HAS_ORIGINAL = True
    print(f"Original dataset: {original.shape[0]:,} rows")
except:
    HAS_ORIGINAL = False
    print("External dataset not available")

print(f"\nTrain: {train.shape[0]:,} rows, {train.shape[1]} cols")
print(f"Test: {test.shape[0]:,} rows")
print(f"Target rate: {train[TARGET].mean():.1%}")

# Save IDs and drop
train_ids = train['id'].copy()
test_ids = test['id'].copy()
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


# Quick data overview
print("Data Types:")
print(train.dtypes.value_counts())
print(f"\nMissing values: {train.isnull().sum().sum()}")
train.head()


# Define column types
CATEGORICAL = ['gender', 'ethnicity', 'education_level', 'income_level', 
               'smoking_status', 'employment_status']
BINARY = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
NUMERICAL = [c for c in train.columns if c not in CATEGORICAL + BINARY + [TARGET]]

print(f"Numerical: {len(NUMERICAL)}, Categorical: {len(CATEGORICAL)}, Binary: {len(BINARY)}")


# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
target_counts = train[TARGET].value_counts()
axes[0].pie(target_counts.values, labels=['No Diabetes', 'Diabetes'], 
            autopct='%1.1f%%', colors=[COLORS[0], COLORS[3]],
            explode=(0, 0.05), shadow=True, startangle=90)
axes[0].set_title('Target Distribution', fontsize=14, fontweight='bold')

# Bar chart
bars = axes[1].bar(['No Diabetes (0)', 'Diabetes (1)'], target_counts.values, 
                   color=[COLORS[0], COLORS[3]], edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Target Class Counts', fontsize=14, fontweight='bold')
for bar, count in zip(bars, target_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, 
                 f'{count:,}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\nClass imbalance ratio: {target_counts[1]/target_counts[0]:.2f}")


# Numerical feature distributions
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

for i, col in enumerate(NUMERICAL):
    ax = axes[i]
    for target_val, color, label in [(0, COLORS[0], 'No Diabetes'), (1, COLORS[3], 'Diabetes')]:
        data = train[train[TARGET] == target_val][col]
        ax.hist(data, bins=30, alpha=0.6, color=color, label=label, density=True)
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle('Numerical Feature Distributions by Target', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 12))

# Encode categoricals temporarily for correlation
train_encoded = train.copy()
for col in CATEGORICAL:
    train_encoded[col] = LabelEncoder().fit_transform(train_encoded[col])

corr_matrix = train_encoded.corr()

# Mask upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdYlBu_r', 
            center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Top correlations with target
target_corr = corr_matrix[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print("\nTop 10 Features Correlated with Target:")
for feat, corr in target_corr.head(10).items():
    print(f"  {feat}: {corr:.4f}")


# Binary features impact
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, col in enumerate(BINARY):
    ax = axes[i]
    diabetes_rates = train.groupby(col)[TARGET].mean()
    bars = ax.bar(['No', 'Yes'], diabetes_rates.values, color=[COLORS[0], COLORS[3]], 
                  edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Diabetes Rate', fontsize=11)
    ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    for bar, rate in zip(bars, diabetes_rates.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{rate:.1%}', ha='center', fontsize=11, fontweight='bold')

plt.suptitle('Binary Features Impact on Diabetes Rate', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


def add_digit_features(df, cols, suffix='_dig'):
    """Extract digit features from numeric columns.
    
    Why this works: Synthetic data often has patterns in decimal places
    that can leak information about the target.
    """
    new_cols = []
    for col in cols:
        df[f'{col}{suffix}1'] = ((df[col] * 10) % 10).fillna(-1).astype('int8')
        new_cols.append(f'{col}{suffix}1')
    return new_cols

def add_bin_features(df, cols, q=5, suffix='_bin'):
    """Create binned categorical features from numeric.
    
    Why this works: Helps tree models find optimal split points
    and reduces noise in continuous features.
    """
    new_cols = []
    for col in cols:
        df[f'{col}{suffix}'], _ = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates='drop')
        new_cols.append(f'{col}{suffix}')
    return new_cols

def create_advanced_features(df):
    """Comprehensive domain-based feature engineering."""
    df = df.copy()
    
    # === LIPID PROFILE (Insulin Resistance Markers) ===
    df['non_hdl_chol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    df['total_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
    df['trig_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)  # Key T2D marker!
    df['trig_total_ratio'] = df['triglycerides'] / (df['cholesterol_total'] + 1)
    
    # === BLOOD PRESSURE METRICS ===
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']  # Arterial stiffness
    df['map'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)  # Mean arterial pressure
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
    
    # === BODY COMPOSITION ===
    df['bmi_risk'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 35, 100], labels=[0, 1, 2, 3, 4]).astype(float)
    df['abdominal_obesity'] = (df['waist_to_hip_ratio'] > 0.9).astype(int)  # Central obesity
    df['bmi_whr'] = df['bmi'] * df['waist_to_hip_ratio']  # Combined body metric
    
    # === LIFESTYLE RISK FACTORS ===
    df['sedentary'] = (df['physical_activity_minutes_per_week'] < 150).astype(int)  # WHO threshold
    df['high_screen'] = (df['screen_time_hours_per_day'] > 6).astype(int)
    df['poor_sleep'] = ((df['sleep_hours_per_day'] < 6) | (df['sleep_hours_per_day'] > 9)).astype(int)
    df['poor_diet'] = (df['diet_score'] < 5).astype(int)
    df['lifestyle_risk'] = df['sedentary'] + df['high_screen'] + df['poor_sleep'] + df['poor_diet']
    df['activity_per_bmi'] = df['physical_activity_minutes_per_week'] / (df['bmi'] + 1)
    
    # === AGE-BASED FEATURES ===
    df['age_decade'] = (df['age'] // 10).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(float)
    
    # === COMORBIDITY SCORE ===
    df['comorbidity'] = (df['family_history_diabetes'] + 
                         df['hypertension_history'] + 
                         df['cardiovascular_history'])
    
    # === FEATURE INTERACTIONS ===
    df['age_bmi'] = df['age'] * df['bmi']
    df['age_comorbid'] = df['age'] * df['comorbidity']
    df['bmi_bp'] = df['bmi'] * df['systolic_bp']
    df['age_bp'] = df['age'] * df['systolic_bp']
    df['bmi_trig'] = df['bmi'] * df['triglycerides']
    df['age_chol'] = df['age'] * df['cholesterol_total']
    
    # === METABOLIC SYNDROME SCORE ===
    df['metabolic_score'] = (df['bmi_risk'] + df['abdominal_obesity'] + 
                             (df['triglycerides'] > 150).astype(int) +
                             (df['systolic_bp'] > 130).astype(int))
    
    return df

# Apply feature engineering
train_fe = create_advanced_features(train)
test_fe = create_advanced_features(test)

new_features = [c for c in train_fe.columns if c not in train.columns]
print(f"Created {len(new_features)} domain features")


# Add digit features for key numeric columns
key_numerics = ['bmi', 'cholesterol_total', 'hdl_cholesterol', 'triglycerides', 
                'systolic_bp', 'waist_to_hip_ratio']
digit_cols = add_digit_features(train_fe, key_numerics)
_ = add_digit_features(test_fe, key_numerics)

# Add bin features
bin_cols = add_bin_features(train_fe, NUMERICAL, q=10)
_ = add_bin_features(test_fe, NUMERICAL, q=10)

print(f"Added {len(digit_cols)} digit features")
print(f"Added {len(bin_cols)} bin features")


# Label encode categoricals
cat_encoded = []
for col in CATEGORICAL:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col], test_fe[col]])
    le.fit(combined)
    train_fe[f'{col}_enc'] = le.transform(train_fe[col])
    test_fe[f'{col}_enc'] = le.transform(test_fe[col])
    cat_encoded.append(f'{col}_enc')

# Count encoding for categoricals
count_cols = []
for col in CATEGORICAL:
    counts = train_fe[col].value_counts()
    train_fe[f'{col}_cnt'] = train_fe[col].map(counts)
    test_fe[f'{col}_cnt'] = test_fe[col].map(counts).fillna(0)
    count_cols.append(f'{col}_cnt')

print(f"Added {len(cat_encoded)} label encoded + {len(count_cols)} count encoded")


# Final feature list
FEATURES = (
    NUMERICAL + BINARY + 
    cat_encoded + count_cols +
    new_features + digit_cols + bin_cols
)
FEATURES = [c for c in FEATURES if c in train_fe.columns and c != TARGET]

X = train_fe[FEATURES].values.astype(np.float32)
y = train_fe[TARGET].values
X_test = test_fe[FEATURES].values.astype(np.float32)

print(f"\nTotal features: {len(FEATURES)}")
print(f"X shape: {X.shape}")
print(f"X_test shape: {X_test.shape}")


# Optimized hyperparameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.008,
    'max_depth': 5,
    'subsample': 0.93,
    'colsample_bytree': 0.2,
    'reg_alpha': 2.0,
    'reg_lambda': 0.7,
    'min_child_weight': 5,
    'max_bin': 512,
    'n_estimators': 15000,
    'early_stopping_rounds': 300,
    'device': 'cuda',
    'tree_method': 'hist',
    'random_state': SEED
}

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.005,
    'max_depth': 4,
    'num_leaves': 31,
    'min_child_samples': 100,
    'subsample': 0.85,
    'colsample_bytree': 0.5,
    'reg_alpha': 0.3,
    'reg_lambda': 8.0,
    'max_bin': 200,
    'n_estimators': 15000,
    'device': 'gpu',
    'verbose': -1,
    'random_state': SEED
}

cat_params = {
    'iterations': 15000,
    'depth': 6,
    'learning_rate': 0.008,
    'l2_leaf_reg': 3.0,
    'border_count': 128,
    'task_type': 'GPU',
    'verbose': 0,
    'early_stopping_rounds': 300,
    'random_seed': SEED
}

print("Model configurations ready!")
print(f"\nXGBoost: {xgb_params['n_estimators']:,} max iterations, LR={xgb_params['learning_rate']}")
print(f"LightGBM: {lgb_params['n_estimators']:,} max iterations, LR={lgb_params['learning_rate']}")
print(f"CatBoost: {cat_params['iterations']:,} max iterations, LR={cat_params['learning_rate']}")


N_SPLITS = 5
N_REPEATS = 2

rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

# Initialize prediction arrays
oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

n_folds = N_SPLITS * N_REPEATS
fold_counts = np.zeros(len(X))

# Store fold scores for visualization
fold_scores = {'xgb': [], 'lgb': [], 'cat': []}

print(f"Training {n_folds} folds (RepeatedStratifiedKFold)...")
print("="*70)

for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    fold_counts[val_idx] += 1
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    xgb_pred_val = xgb_model.predict_proba(X_val)[:, 1]
    xgb_pred_test = xgb_model.predict_proba(X_test)[:, 1]
    oof_xgb[val_idx] += xgb_pred_val
    test_xgb += xgb_pred_test / n_folds
    xgb_score = roc_auc_score(y_val, xgb_pred_val)
    fold_scores['xgb'].append(xgb_score)
    
    # LightGBM
    lgb_model = LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    lgb_pred_val = lgb_model.predict_proba(X_val)[:, 1]
    lgb_pred_test = lgb_model.predict_proba(X_test)[:, 1]
    oof_lgb[val_idx] += lgb_pred_val
    test_lgb += lgb_pred_test / n_folds
    lgb_score = roc_auc_score(y_val, lgb_pred_val)
    fold_scores['lgb'].append(lgb_score)
    
    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    cat_pred_val = cat_model.predict_proba(X_val)[:, 1]
    cat_pred_test = cat_model.predict_proba(X_test)[:, 1]
    oof_cat[val_idx] += cat_pred_val
    test_cat += cat_pred_test / n_folds
    cat_score = roc_auc_score(y_val, cat_pred_val)
    fold_scores['cat'].append(cat_score)
    
    print(f"   XGB: {xgb_score:.5f} | LGB: {lgb_score:.5f} | CAT: {cat_score:.5f}")
    
    gc.collect()

# Average OOF predictions for repeated folds
oof_xgb = oof_xgb / fold_counts
oof_lgb = oof_lgb / fold_counts
oof_cat = oof_cat / fold_counts

print("\n" + "="*70)
print(f"\nFinal OOF Scores:")
print(f"  XGBoost OOF AUC: {roc_auc_score(y, oof_xgb):.5f} (mean: {np.mean(fold_scores['xgb']):.5f} ± {np.std(fold_scores['xgb']):.5f})")
print(f"  LightGBM OOF AUC: {roc_auc_score(y, oof_lgb):.5f} (mean: {np.mean(fold_scores['lgb']):.5f} ± {np.std(fold_scores['lgb']):.5f})")
print(f"  CatBoost OOF AUC: {roc_auc_score(y, oof_cat):.5f} (mean: {np.mean(fold_scores['cat']):.5f} ± {np.std(fold_scores['cat']):.5f})")


# Visualize fold scores
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Fold-by-fold comparison
ax = axes[0]
x = np.arange(1, n_folds + 1)
width = 0.25
ax.bar(x - width, fold_scores['xgb'], width, label='XGBoost', color=COLORS[0], alpha=0.8)
ax.bar(x, fold_scores['lgb'], width, label='LightGBM', color=COLORS[1], alpha=0.8)
ax.bar(x + width, fold_scores['cat'], width, label='CatBoost', color=COLORS[3], alpha=0.8)
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('AUC Score', fontsize=12)
ax.set_title('Model Performance by Fold', fontsize=14, fontweight='bold')
ax.legend()
ax.set_xticks(x)
ax.set_ylim(0.72, 0.735)

# Box plot comparison
ax = axes[1]
data = [fold_scores['xgb'], fold_scores['lgb'], fold_scores['cat']]
bp = ax.boxplot(data, labels=['XGBoost', 'LightGBM', 'CatBoost'], patch_artist=True)
for patch, color in zip(bp['boxes'], [COLORS[0], COLORS[1], COLORS[3]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('AUC Score', fontsize=12)
ax.set_title('Model Score Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


# ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

for name, oof, color in [('XGBoost', oof_xgb, COLORS[0]), 
                          ('LightGBM', oof_lgb, COLORS[1]), 
                          ('CatBoost', oof_cat, COLORS[3])]:
    fpr, tpr, _ = roc_curve(y, oof)
    auc = roc_auc_score(y, oof)
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{name} (AUC = {auc:.5f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Individual Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
plt.tight_layout()
plt.show()


def objective(trial):
    """Optuna objective function for weight optimization."""
    # Suggest weights (they will be normalized)
    w_xgb = trial.suggest_float('w_xgb', 0.1, 0.8)
    w_lgb = trial.suggest_float('w_lgb', 0.1, 0.8)
    w_cat = trial.suggest_float('w_cat', 0.05, 0.5)
    
    # Normalize weights to sum to 1
    total = w_xgb + w_lgb + w_cat
    w_xgb, w_lgb, w_cat = w_xgb/total, w_lgb/total, w_cat/total
    
    # Calculate blended prediction
    blend = w_xgb * oof_xgb + w_lgb * oof_lgb + w_cat * oof_cat
    
    return roc_auc_score(y, blend)

print("Running Optuna optimization for ensemble weights...\n")

# Create study with TPE sampler
sampler = TPESampler(seed=SEED)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=200, show_progress_bar=True)

# Get best weights
best_params = study.best_params
total = best_params['w_xgb'] + best_params['w_lgb'] + best_params['w_cat']
w_xgb_opt = best_params['w_xgb'] / total
w_lgb_opt = best_params['w_lgb'] / total
w_cat_opt = best_params['w_cat'] / total

print(f"\nOptuna Best Weights:")
print(f"  XGBoost: {w_xgb_opt:.4f}")
print(f"  LightGBM: {w_lgb_opt:.4f}")
print(f"  CatBoost: {w_cat_opt:.4f}")
print(f"\nOptuna Best OOF AUC: {study.best_value:.5f}")


# Also run grid search for comparison
print("Running Grid Search for comparison...\n")

best_score_grid = 0
best_weights_grid = None

for w_xgb in np.arange(0.2, 0.7, 0.05):
    for w_lgb in np.arange(0.1, 0.5, 0.05):
        w_cat = 1 - w_xgb - w_lgb
        if w_cat < 0.05 or w_cat > 0.6:
            continue
        
        blend = w_xgb * oof_xgb + w_lgb * oof_lgb + w_cat * oof_cat
        score = roc_auc_score(y, blend)
        
        if score > best_score_grid:
            best_score_grid = score
            best_weights_grid = (w_xgb, w_lgb, w_cat)

print(f"Grid Search Best Weights:")
print(f"  XGBoost: {best_weights_grid[0]:.2f}")
print(f"  LightGBM: {best_weights_grid[1]:.2f}")
print(f"  CatBoost: {best_weights_grid[2]:.2f}")
print(f"\nGrid Search Best OOF AUC: {best_score_grid:.5f}")


# Use Optuna weights if better, otherwise grid search
if study.best_value >= best_score_grid:
    w_xgb, w_lgb, w_cat = w_xgb_opt, w_lgb_opt, w_cat_opt
    print("Using Optuna optimized weights")
else:
    w_xgb, w_lgb, w_cat = best_weights_grid
    print("Using Grid Search weights")

# Create final blend
oof_final = w_xgb * oof_xgb + w_lgb * oof_lgb + w_cat * oof_cat
test_final = w_xgb * test_xgb + w_lgb * test_lgb + w_cat * test_cat

# Also compute simple average for comparison
oof_avg = (oof_xgb + oof_lgb + oof_cat) / 3
test_avg = (test_xgb + test_lgb + test_cat) / 3

print(f"\nFinal Results:")
print(f"  Optimized Blend OOF AUC: {roc_auc_score(y, oof_final):.5f}")
print(f"  Simple Average OOF AUC: {roc_auc_score(y, oof_avg):.5f}")


# Visualize final weights
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart of weights
ax = axes[0]
weights = [w_xgb, w_lgb, w_cat]
labels = [f'XGBoost\n{w_xgb:.1%}', f'LightGBM\n{w_lgb:.1%}', f'CatBoost\n{w_cat:.1%}']
colors_pie = [COLORS[0], COLORS[1], COLORS[3]]
explode = (0.05, 0, 0)
ax.pie(weights, labels=labels, colors=colors_pie, explode=explode,
       autopct='', shadow=True, startangle=90,
       textprops={'fontsize': 12, 'fontweight': 'bold'})
ax.set_title('Optimized Ensemble Weights', fontsize=14, fontweight='bold')

# Final ROC curve comparison
ax = axes[1]
fpr_blend, tpr_blend, _ = roc_curve(y, oof_final)
fpr_avg, tpr_avg, _ = roc_curve(y, oof_avg)
fpr_xgb, tpr_xgb, _ = roc_curve(y, oof_xgb)

ax.plot(fpr_xgb, tpr_xgb, color=COLORS[0], linewidth=2, alpha=0.5, 
        label=f'XGBoost (AUC={roc_auc_score(y, oof_xgb):.5f})')
ax.plot(fpr_avg, tpr_avg, color='gray', linewidth=2, linestyle='--',
        label=f'Simple Average (AUC={roc_auc_score(y, oof_avg):.5f})')
ax.plot(fpr_blend, tpr_blend, color=COLORS[4], linewidth=3,
        label=f'Optimized Blend (AUC={roc_auc_score(y, oof_final):.5f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Ensemble Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.show()


# Use best performing blend
if roc_auc_score(y, oof_final) > roc_auc_score(y, oof_avg):
    final_pred = test_final
    blend_type = "Optimized Weighted Blend"
else:
    final_pred = test_avg
    blend_type = "Simple Average Blend"

print(f"Using: {blend_type}")
print(f"\nPrediction Statistics:")
print(f"  Min: {final_pred.min():.4f}")
print(f"  Max: {final_pred.max():.4f}")
print(f"  Mean: {final_pred.mean():.4f}")
print(f"  Std: {final_pred.std():.4f}")


# Prediction distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Distribution comparison
ax = axes[0]
ax.hist(oof_final, bins=50, alpha=0.6, color=COLORS[0], label='OOF Predictions', density=True)
ax.hist(final_pred, bins=50, alpha=0.6, color=COLORS[3], label='Test Predictions', density=True)
ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prediction Distribution: Train vs Test', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)

# OOF predictions by actual class
ax = axes[1]
ax.hist(oof_final[y == 0], bins=50, alpha=0.6, color=COLORS[0], label='Actual: No Diabetes', density=True)
ax.hist(oof_final[y == 1], bins=50, alpha=0.6, color=COLORS[3], label='Actual: Diabetes', density=True)
ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold = 0.5')
ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('OOF Predictions by Actual Class', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)

plt.tight_layout()
plt.show()


# Create submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
submission[TARGET] = final_pred
submission.to_csv('submission.csv', index=False)

print("Submission saved!")
print(f"\nSubmission preview:")
submission.head(10)

