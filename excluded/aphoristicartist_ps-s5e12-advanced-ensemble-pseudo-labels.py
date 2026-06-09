import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import warnings
import gc

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Color schemes
COLORS = {
    'primary': '#6366f1',
    'secondary': '#8b5cf6',
    'accent': '#d946ef',
    'success': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'info': '#3b82f6',
    'dark': '#1f2937',
    'light': '#f3f4f6'
}
PALETTE = ['#6366f1', '#8b5cf6', '#d946ef', '#ec4899', '#f43f5e', '#10b981', '#3b82f6']

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titleweight'] = 'bold'

SEED = 2025
TARGET = 'diagnosed_diabetes'
N_SPLITS = 5
N_REPEATS = 2

print("Libraries loaded!")
print(f"Optuna: {optuna.__version__}")
print(f"XGBoost: {xgb.__version__}")


# Load competition data
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

print(f"Competition Train: {train.shape[0]:,} rows, {train.shape[1]} columns")
print(f"Competition Test: {test.shape[0]:,} rows")
print(f"\nTarget distribution: {train[TARGET].value_counts().to_dict()}")
print(f"Diabetes rate: {train[TARGET].mean():.1%}")


# Load external datasets
external_datasets = []

# Dataset 1: Diabetes Health Indicators (BRFSS 2015)
try:
    ext1 = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_012_health_indicators_BRFSS2015.csv')
    print(f"External 1 (BRFSS 2015): {ext1.shape[0]:,} rows")
    external_datasets.append(('brfss', ext1))
except:
    print("External dataset 1 not available")

# Dataset 2: Diabetes Prediction Dataset
try:
    ext2 = pd.read_csv('/kaggle/input/diabetes-prediction-dataset/diabetes_prediction_dataset.csv')
    print(f"External 2 (Prediction): {ext2.shape[0]:,} rows")
    external_datasets.append(('pred', ext2))
except:
    print("External dataset 2 not available")

# Dataset 3: Pima Indians
try:
    ext3 = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
    print(f"External 3 (Pima): {ext3.shape[0]:,} rows")
    external_datasets.append(('pima', ext3))
except:
    print("External dataset 3 not available")

print(f"\nTotal external datasets loaded: {len(external_datasets)}")


# Save IDs
train_ids = train['id'].copy()
test_ids = test['id'].copy()
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# Define column types based on competition data
CATEGORICAL = ['gender', 'ethnicity', 'education_level', 'income_level', 
               'smoking_status', 'employment_status']
BINARY = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
NUMERICAL = [c for c in train.columns if c not in CATEGORICAL + BINARY + [TARGET]]

print(f"\nColumn types:")
print(f"  Numerical: {len(NUMERICAL)} - {NUMERICAL[:5]}...")
print(f"  Categorical: {len(CATEGORICAL)}")
print(f"  Binary: {len(BINARY)}")


# Process and merge external data that matches our features
def process_brfss(df):
    """Process BRFSS dataset to match competition format."""
    df = df.copy()
    
    # Create target (binary: 0 = no diabetes, 1 = diabetes/prediabetes)
    df['target'] = (df['Diabetes_012'] > 0).astype(int)
    
    # Map available features
    processed = pd.DataFrame()
    processed['bmi'] = df['BMI']
    processed['age'] = df['Age'] * 5 + 20  # Approximate age from category
    processed['family_history_diabetes'] = 0  # Not available, use 0
    processed['hypertension_history'] = df['HighBP']
    processed['cardiovascular_history'] = df['HeartDiseaseorAttack']
    processed['physical_activity_minutes_per_week'] = df['PhysActivity'] * 150
    processed['cholesterol_total'] = df['HighChol'] * 100 + 150  # Approximate
    processed[TARGET] = df['target']
    processed['source'] = 'brfss'
    
    return processed

def process_prediction_dataset(df):
    """Process diabetes prediction dataset."""
    df = df.copy()
    
    processed = pd.DataFrame()
    processed['age'] = df['age']
    processed['bmi'] = df['bmi']
    processed['hypertension_history'] = df['hypertension']
    processed['cardiovascular_history'] = df['heart_disease']
    processed['smoking_status'] = df['smoking_history'].map({
        'never': 'Never Smoked', 'No Info': 'Never Smoked', 
        'current': 'Currently Smoke', 'former': 'Former Smoker',
        'ever': 'Former Smoker', 'not current': 'Former Smoker'
    }).fillna('Never Smoked')
    processed['gender'] = df['gender'].map({'Male': 'Male', 'Female': 'Female', 'Other': 'Other'}).fillna('Other')
    processed[TARGET] = df['diabetes']
    processed['source'] = 'pred'
    
    return processed

# Process external data
external_processed = []

for name, df in external_datasets:
    try:
        if name == 'brfss':
            processed = process_brfss(df)
        elif name == 'pred':
            processed = process_prediction_dataset(df)
        else:
            continue
        external_processed.append(processed)
        print(f"Processed {name}: {len(processed):,} rows")
    except Exception as e:
        print(f"Error processing {name}: {e}")

print(f"\nTotal external processed: {len(external_processed)} datasets")


# Create augmented training set
train['source'] = 'competition'

if external_processed:
    # Combine with external data
    all_columns = list(train.columns)
    
    for ext_df in external_processed:
        # Add missing columns with NaN
        for col in all_columns:
            if col not in ext_df.columns:
                ext_df[col] = np.nan
        # Reorder columns
        ext_df = ext_df[all_columns]
    
    # Sample external data to avoid overwhelming competition data
    external_sampled = []
    for ext_df in external_processed:
        sample_size = min(len(ext_df), 50000)  # Max 50k per external source
        sampled = ext_df.sample(n=sample_size, random_state=SEED)
        external_sampled.append(sampled)
        print(f"Sampled {len(sampled):,} from external")
    
    train_augmented = pd.concat([train] + external_sampled, ignore_index=True)
    print(f"\nAugmented train: {len(train_augmented):,} rows ({len(train):,} competition + {len(train_augmented)-len(train):,} external)")
else:
    train_augmented = train.copy()
    print("Using competition data only")

# Show source distribution
print(f"\nData sources:")
print(train_augmented['source'].value_counts())


# Target distribution by source
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Overall target distribution
ax = axes[0]
target_counts = train[TARGET].value_counts()
ax.pie(target_counts.values, labels=['No Diabetes', 'Diabetes'], 
       autopct='%1.1f%%', colors=[COLORS['success'], COLORS['danger']],
       explode=(0, 0.05), shadow=True, startangle=90,
       textprops={'fontsize': 12, 'fontweight': 'bold'})
ax.set_title('Competition Target Distribution', fontsize=14)

# Target by source
ax = axes[1]
source_target = train_augmented.groupby('source')[TARGET].mean().sort_values()
bars = ax.barh(source_target.index, source_target.values, color=PALETTE[:len(source_target)])
ax.set_xlabel('Diabetes Rate')
ax.set_title('Diabetes Rate by Data Source', fontsize=14)
ax.set_xlim(0, 1)
for bar, rate in zip(bars, source_target.values):
    ax.text(rate + 0.02, bar.get_y() + bar.get_height()/2, f'{rate:.1%}', va='center')

# Sample counts by source
ax = axes[2]
source_counts = train_augmented['source'].value_counts()
ax.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%',
       colors=PALETTE[:len(source_counts)], shadow=True)
ax.set_title('Data Source Distribution', fontsize=14)

plt.tight_layout()
plt.show()


# Feature distributions by target
fig, axes = plt.subplots(3, 5, figsize=(22, 14))
axes = axes.flatten()

for i, col in enumerate(NUMERICAL):
    ax = axes[i]
    
    # Plot for each target
    for target_val, color, label in [(0, COLORS['success'], 'No Diabetes'), 
                                      (1, COLORS['danger'], 'Diabetes')]:
        data = train[train[TARGET] == target_val][col].dropna()
        ax.hist(data, bins=40, alpha=0.6, color=color, label=label, density=True)
        ax.axvline(data.mean(), color=color, linestyle='--', linewidth=2, alpha=0.8)
    
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=9)
    if i == 0:
        ax.legend(fontsize=9)

for i in range(len(NUMERICAL), len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Numerical Features Distribution by Target (with mean lines)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# Correlation with target (exclude source column)
train_encoded = train.drop(columns=['source'], errors='ignore').copy()
for col in CATEGORICAL:
    if col in train_encoded.columns:
        train_encoded[col] = LabelEncoder().fit_transform(train_encoded[col].astype(str))

correlations = train_encoded.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(12, 10))
colors = [COLORS['danger'] if x > 0 else COLORS['info'] for x in correlations.values]
bars = ax.barh(correlations.index, correlations.values, color=colors, edgecolor='black', alpha=0.8)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Correlation with Diabetes', fontsize=12)
ax.set_title('Feature Correlation with Target', fontsize=14, fontweight='bold')

for bar, val in zip(bars, correlations.values):
    x_pos = val + 0.01 if val > 0 else val - 0.01
    ha = 'left' if val > 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', ha=ha, fontsize=9)

plt.tight_layout()
plt.show()


# Key feature interactions
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Age vs BMI
ax = axes[0, 0]
scatter = ax.scatter(train['age'], train['bmi'], c=train[TARGET], cmap='RdYlGn_r', alpha=0.3, s=5)
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_title('Age vs BMI (colored by diabetes)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax)

# Cholesterol vs Triglycerides
ax = axes[0, 1]
scatter = ax.scatter(train['cholesterol_total'], train['triglycerides'], c=train[TARGET], cmap='RdYlGn_r', alpha=0.3, s=5)
ax.set_xlabel('Total Cholesterol')
ax.set_ylabel('Triglycerides')
ax.set_title('Cholesterol vs Triglycerides', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax)

# Systolic vs Diastolic BP
ax = axes[0, 2]
scatter = ax.scatter(train['systolic_bp'], train['diastolic_bp'], c=train[TARGET], cmap='RdYlGn_r', alpha=0.3, s=5)
ax.set_xlabel('Systolic BP')
ax.set_ylabel('Diastolic BP')
ax.set_title('Blood Pressure Distribution', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax)

# Binary features impact
ax = axes[1, 0]
binary_rates = pd.DataFrame({
    'Feature': BINARY * 2,
    'Value': ['No'] * len(BINARY) + ['Yes'] * len(BINARY),
    'Rate': [train[train[col] == 0][TARGET].mean() for col in BINARY] + 
            [train[train[col] == 1][TARGET].mean() for col in BINARY]
})
sns.barplot(data=binary_rates, x='Feature', y='Rate', hue='Value', ax=ax, palette=[COLORS['success'], COLORS['danger']])
ax.set_xticklabels([b.replace('_', '\n') for b in BINARY], fontsize=9)
ax.set_ylabel('Diabetes Rate')
ax.set_title('Binary Features Impact', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)

# Age distribution by target
ax = axes[1, 1]
for target_val, color, label in [(0, COLORS['success'], 'No Diabetes'), (1, COLORS['danger'], 'Diabetes')]:
    data = train[train[TARGET] == target_val]['age']
    ax.hist(data, bins=30, alpha=0.6, color=color, label=label, density=True)
ax.set_xlabel('Age')
ax.set_title('Age Distribution by Target', fontsize=12, fontweight='bold')
ax.legend()

# BMI distribution by target
ax = axes[1, 2]
for target_val, color, label in [(0, COLORS['success'], 'No Diabetes'), (1, COLORS['danger'], 'Diabetes')]:
    data = train[train[TARGET] == target_val]['bmi']
    ax.hist(data, bins=30, alpha=0.6, color=color, label=label, density=True)
ax.set_xlabel('BMI')
ax.set_title('BMI Distribution by Target', fontsize=12, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.show()


class TargetEncoder:
    """Smoothed target encoding with regularization."""
    def __init__(self, smoothing=10, min_samples=1):
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.global_mean = None
        self.encodings = {}
        
    def fit(self, X, y):
        self.global_mean = y.mean()
        df = pd.DataFrame({'feature': X, 'target': y})
        agg = df.groupby('feature')['target'].agg(['mean', 'count'])
        
        # Smoothed encoding: (count * mean + smoothing * global_mean) / (count + smoothing)
        smoothed = (agg['count'] * agg['mean'] + self.smoothing * self.global_mean) / (agg['count'] + self.smoothing)
        self.encodings = smoothed.to_dict()
        return self
    
    def transform(self, X):
        return X.map(self.encodings).fillna(self.global_mean)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def create_advanced_features(df, is_train=True, encoders=None):
    """Comprehensive feature engineering."""
    df = df.copy()
    
    # ========== LIPID PROFILE ==========
    df['non_hdl_chol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    df['total_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
    df['trig_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)  # Key insulin resistance marker
    df['trig_total_ratio'] = df['triglycerides'] / (df['cholesterol_total'] + 1)
    df['atherogenic_index'] = np.log10(df['triglycerides'] / (df['hdl_cholesterol'] + 1) + 1)
    
    # ========== BLOOD PRESSURE ==========
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['map'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)  # Mean arterial pressure
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
    df['bp_risk'] = ((df['systolic_bp'] > 130) | (df['diastolic_bp'] > 80)).astype(int)
    
    # ========== BODY COMPOSITION ==========
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 35, 100], labels=[0, 1, 2, 3, 4]).astype(float)
    df['abdominal_obesity'] = (df['waist_to_hip_ratio'] > 0.9).astype(int)
    df['bmi_whr'] = df['bmi'] * df['waist_to_hip_ratio']
    df['bmi_squared'] = df['bmi'] ** 2
    df['bmi_log'] = np.log1p(df['bmi'])
    
    # ========== LIFESTYLE ==========
    df['sedentary'] = (df['physical_activity_minutes_per_week'] < 150).astype(int)
    df['very_active'] = (df['physical_activity_minutes_per_week'] > 300).astype(int)
    df['high_screen'] = (df['screen_time_hours_per_day'] > 6).astype(int)
    df['poor_sleep'] = ((df['sleep_hours_per_day'] < 6) | (df['sleep_hours_per_day'] > 9)).astype(int)
    df['poor_diet'] = (df['diet_score'] < 5).astype(int)
    df['lifestyle_risk'] = df['sedentary'] + df['high_screen'] + df['poor_sleep'] + df['poor_diet']
    df['activity_per_bmi'] = df['physical_activity_minutes_per_week'] / (df['bmi'] + 1)
    df['activity_log'] = np.log1p(df['physical_activity_minutes_per_week'])
    
    # ========== AGE FEATURES ==========
    df['age_decade'] = (df['age'] // 10).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 75, 100], labels=[0, 1, 2, 3, 4]).astype(float)
    df['age_squared'] = df['age'] ** 2
    df['age_log'] = np.log1p(df['age'])
    
    # ========== COMORBIDITIES ==========
    df['comorbidity_count'] = (df['family_history_diabetes'] + 
                               df['hypertension_history'] + 
                               df['cardiovascular_history'])
    df['high_risk'] = (df['comorbidity_count'] >= 2).astype(int)
    
    # ========== INTERACTIONS ==========
    df['age_bmi'] = df['age'] * df['bmi']
    df['age_comorbid'] = df['age'] * df['comorbidity_count']
    df['bmi_bp'] = df['bmi'] * df['systolic_bp']
    df['age_bp'] = df['age'] * df['systolic_bp']
    df['bmi_trig'] = df['bmi'] * df['triglycerides']
    df['age_chol'] = df['age'] * df['cholesterol_total']
    df['activity_sleep'] = df['physical_activity_minutes_per_week'] * df['sleep_hours_per_day']
    df['bmi_activity'] = df['bmi'] / (df['physical_activity_minutes_per_week'] + 1)
    
    # ========== METABOLIC SYNDROME SCORE ==========
    df['metabolic_score'] = (df['bmi_category'] + 
                             df['abdominal_obesity'] + 
                             (df['triglycerides'] > 150).astype(int) +
                             (df['hdl_cholesterol'] < 40).astype(int) +
                             df['bp_risk'])
    
    # ========== DIGIT FEATURES (for synthetic data) ==========
    for col in ['bmi', 'cholesterol_total', 'triglycerides', 'systolic_bp', 'hdl_cholesterol']:
        df[f'{col}_dig1'] = ((df[col] * 10) % 10).fillna(-1).astype('int8')
    
    # ========== QUANTILE FEATURES ==========
    for col in NUMERICAL:
        try:
            df[f'{col}_qbin'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
        except:
            df[f'{col}_qbin'] = pd.cut(df[col], bins=10, labels=False)
    
    return df

# Apply feature engineering
train_fe = create_advanced_features(train)
test_fe = create_advanced_features(test)

# Also create features for augmented data if available
if len(train_augmented) > len(train):
    train_aug_fe = create_advanced_features(train_augmented)
else:
    train_aug_fe = train_fe.copy()

new_features = [c for c in train_fe.columns if c not in train.columns]
print(f"Created {len(new_features)} new features")


# Target encoding for categorical features
target_encoders = {}

for col in CATEGORICAL:
    encoder = TargetEncoder(smoothing=20)
    train_fe[f'{col}_te'] = encoder.fit_transform(train_fe[col], train_fe[TARGET])
    test_fe[f'{col}_te'] = encoder.transform(test_fe[col])
    target_encoders[col] = encoder

# Label encoding
label_encoders = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col], test_fe[col]]).astype(str)
    le.fit(combined)
    train_fe[f'{col}_le'] = le.transform(train_fe[col].astype(str))
    test_fe[f'{col}_le'] = le.transform(test_fe[col].astype(str))
    label_encoders[col] = le

# Count encoding
for col in CATEGORICAL:
    counts = train_fe[col].value_counts()
    train_fe[f'{col}_cnt'] = train_fe[col].map(counts)
    test_fe[f'{col}_cnt'] = test_fe[col].map(counts).fillna(0)

print(f"Added target encoding, label encoding, and count encoding for {len(CATEGORICAL)} categorical features")


# Prepare final feature set
exclude_cols = CATEGORICAL + [TARGET, 'source']
FEATURES = [c for c in train_fe.columns if c not in exclude_cols]

X = train_fe[FEATURES].values.astype(np.float32)
y = train_fe[TARGET].values
X_test = test_fe[FEATURES].values.astype(np.float32)

# Handle any remaining NaN
X = np.nan_to_num(X, nan=-999)
X_test = np.nan_to_num(X_test, nan=-999)

print(f"Final features: {len(FEATURES)}")
print(f"X shape: {X.shape}")
print(f"X_test shape: {X_test.shape}")


# Model configurations - optimized hyperparameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.008,
    'max_depth': 6,
    'subsample': 0.9,
    'colsample_bytree': 0.3,
    'reg_alpha': 1.5,
    'reg_lambda': 1.0,
    'min_child_weight': 10,
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
    'learning_rate': 0.008,
    'max_depth': 5,
    'num_leaves': 40,
    'min_child_samples': 80,
    'subsample': 0.85,
    'colsample_bytree': 0.4,
    'reg_alpha': 0.5,
    'reg_lambda': 5.0,
    'max_bin': 255,
    'n_estimators': 15000,
    'device': 'gpu',
    'verbose': -1,
    'random_state': SEED
}

cat_params = {
    'iterations': 15000,
    'depth': 7,
    'learning_rate': 0.008,
    'l2_leaf_reg': 5.0,
    'border_count': 200,
    'task_type': 'GPU',
    'verbose': 0,
    'early_stopping_rounds': 300,
    'random_seed': SEED
}

hgb_params = {
    'learning_rate': 0.02,
    'max_iter': 5000,
    'max_depth': 6,
    'min_samples_leaf': 50,
    'l2_regularization': 1.0,
    'max_bins': 255,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 50,
    'random_state': SEED
}

et_params = {
    'n_estimators': 500,
    'max_depth': 15,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'n_jobs': -1,
    'random_state': SEED
}

print("Model configurations ready!")


# Training with RepeatedStratifiedKFold
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

# Initialize arrays
oof_preds = {name: np.zeros(len(X)) for name in ['xgb', 'lgb', 'cat', 'hgb', 'et']}
test_preds = {name: np.zeros(len(X_test)) for name in ['xgb', 'lgb', 'cat', 'hgb', 'et']}
fold_scores = {name: [] for name in ['xgb', 'lgb', 'cat', 'hgb', 'et']}

n_folds = N_SPLITS * N_REPEATS
fold_counts = np.zeros(len(X))

print(f"Training {n_folds} folds with 5 models...")
print("="*80)

for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    fold_counts[val_idx] += 1
    
    scores = []
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    oof_preds['xgb'][val_idx] += xgb_pred
    test_preds['xgb'] += xgb_model.predict_proba(X_test)[:, 1] / n_folds
    xgb_score = roc_auc_score(y_val, xgb_pred)
    fold_scores['xgb'].append(xgb_score)
    scores.append(f"XGB:{xgb_score:.5f}")
    
    # LightGBM
    lgb_model = LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
    oof_preds['lgb'][val_idx] += lgb_pred
    test_preds['lgb'] += lgb_model.predict_proba(X_test)[:, 1] / n_folds
    lgb_score = roc_auc_score(y_val, lgb_pred)
    fold_scores['lgb'].append(lgb_score)
    scores.append(f"LGB:{lgb_score:.5f}")
    
    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    cat_pred = cat_model.predict_proba(X_val)[:, 1]
    oof_preds['cat'][val_idx] += cat_pred
    test_preds['cat'] += cat_model.predict_proba(X_test)[:, 1] / n_folds
    cat_score = roc_auc_score(y_val, cat_pred)
    fold_scores['cat'].append(cat_score)
    scores.append(f"CAT:{cat_score:.5f}")
    
    # HistGradientBoosting
    hgb_model = HistGradientBoostingClassifier(**hgb_params)
    hgb_model.fit(X_train, y_train)
    hgb_pred = hgb_model.predict_proba(X_val)[:, 1]
    oof_preds['hgb'][val_idx] += hgb_pred
    test_preds['hgb'] += hgb_model.predict_proba(X_test)[:, 1] / n_folds
    hgb_score = roc_auc_score(y_val, hgb_pred)
    fold_scores['hgb'].append(hgb_score)
    scores.append(f"HGB:{hgb_score:.5f}")
    
    # ExtraTrees
    et_model = ExtraTreesClassifier(**et_params)
    et_model.fit(X_train, y_train)
    et_pred = et_model.predict_proba(X_val)[:, 1]
    oof_preds['et'][val_idx] += et_pred
    test_preds['et'] += et_model.predict_proba(X_test)[:, 1] / n_folds
    et_score = roc_auc_score(y_val, et_pred)
    fold_scores['et'].append(et_score)
    scores.append(f"ET:{et_score:.5f}")
    
    print(f"   {' | '.join(scores)}")
    gc.collect()

# Average OOF predictions
for name in oof_preds:
    oof_preds[name] /= fold_counts

print("\n" + "="*80)
print("\nFinal OOF Scores:")
for name in oof_preds:
    score = roc_auc_score(y, oof_preds[name])
    print(f"  {name.upper()}: {score:.5f} (mean fold: {np.mean(fold_scores[name]):.5f} ± {np.std(fold_scores[name]):.5f})")


# Visualize model performance
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Fold scores comparison
ax = axes[0]
x = np.arange(1, n_folds + 1)
for name, color in zip(['xgb', 'lgb', 'cat', 'hgb', 'et'], PALETTE):
    ax.plot(x, fold_scores[name], 'o-', color=color, label=name.upper(), linewidth=2, markersize=4)
ax.set_xlabel('Fold')
ax.set_ylabel('AUC')
ax.set_title('Fold-by-Fold Performance', fontsize=14, fontweight='bold')
ax.legend()

# Box plot
ax = axes[1]
bp = ax.boxplot([fold_scores[name] for name in ['xgb', 'lgb', 'cat', 'hgb', 'et']], 
                labels=['XGB', 'LGB', 'CAT', 'HGB', 'ET'], patch_artist=True)
for patch, color in zip(bp['boxes'], PALETTE[:5]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('AUC')
ax.set_title('Score Distribution', fontsize=14, fontweight='bold')

# ROC curves
ax = axes[2]
for name, color in zip(['xgb', 'lgb', 'cat', 'hgb', 'et'], PALETTE):
    fpr, tpr, _ = roc_curve(y, oof_preds[name])
    score = roc_auc_score(y, oof_preds[name])
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name.upper()} ({score:.5f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()


# Create initial ensemble prediction for pseudo labeling
initial_blend = (test_preds['xgb'] + test_preds['lgb'] + test_preds['cat']) / 3

# Find confident predictions
CONFIDENCE_THRESHOLD_HIGH = 0.85
CONFIDENCE_THRESHOLD_LOW = 0.15

confident_positive = initial_blend >= CONFIDENCE_THRESHOLD_HIGH
confident_negative = initial_blend <= CONFIDENCE_THRESHOLD_LOW
confident_mask = confident_positive | confident_negative

print(f"Test samples: {len(X_test):,}")
print(f"Confident positive (>{CONFIDENCE_THRESHOLD_HIGH}): {confident_positive.sum():,} ({confident_positive.mean():.1%})")
print(f"Confident negative (<{CONFIDENCE_THRESHOLD_LOW}): {confident_negative.sum():,} ({confident_negative.mean():.1%})")
print(f"Total confident: {confident_mask.sum():,} ({confident_mask.mean():.1%})")


# Create pseudo labels
pseudo_labels = (initial_blend >= 0.5).astype(int)

# Add confident samples to training
X_pseudo = X_test[confident_mask]
y_pseudo = pseudo_labels[confident_mask]

X_augmented = np.vstack([X, X_pseudo])
y_augmented = np.concatenate([y, y_pseudo])

print(f"\nOriginal training: {len(X):,}")
print(f"Pseudo samples added: {len(X_pseudo):,}")
print(f"Augmented training: {len(X_augmented):,}")
print(f"\nPseudo label distribution: {pd.Series(y_pseudo).value_counts().to_dict()}")


# Retrain with pseudo labels (single fold for speed)
print("Retraining models with pseudo labels...\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_pseudo = {name: np.zeros(len(X)) for name in ['xgb', 'lgb', 'cat']}
test_pseudo = {name: np.zeros(len(X_test)) for name in ['xgb', 'lgb', 'cat']}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}/5", end=" ")
    
    # Use all augmented data for training, but validate on original data only
    X_train_fold = np.vstack([X[train_idx], X_pseudo])
    y_train_fold = np.concatenate([y[train_idx], y_pseudo])
    X_val, y_val = X[val_idx], y[val_idx]
    
    scores = []
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val, y_val)], verbose=0)
    oof_pseudo['xgb'][val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    test_pseudo['xgb'] += xgb_model.predict_proba(X_test)[:, 1] / 5
    scores.append(f"XGB:{roc_auc_score(y_val, oof_pseudo['xgb'][val_idx]):.5f}")
    
    # LightGBM
    lgb_model = LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val, y_val)])
    oof_pseudo['lgb'][val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    test_pseudo['lgb'] += lgb_model.predict_proba(X_test)[:, 1] / 5
    scores.append(f"LGB:{roc_auc_score(y_val, oof_pseudo['lgb'][val_idx]):.5f}")
    
    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train_fold, y_train_fold, eval_set=(X_val, y_val))
    oof_pseudo['cat'][val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_pseudo['cat'] += cat_model.predict_proba(X_test)[:, 1] / 5
    scores.append(f"CAT:{roc_auc_score(y_val, oof_pseudo['cat'][val_idx]):.5f}")
    
    print(f"| {' | '.join(scores)}")
    gc.collect()

print("\nPseudo-label OOF Scores:")
for name in oof_pseudo:
    print(f"  {name.upper()}: {roc_auc_score(y, oof_pseudo[name]):.5f}")


# Combine all OOF predictions for stacking
oof_stack = np.column_stack([
    oof_preds['xgb'], oof_preds['lgb'], oof_preds['cat'], 
    oof_preds['hgb'], oof_preds['et'],
    oof_pseudo['xgb'], oof_pseudo['lgb'], oof_pseudo['cat']
])

test_stack = np.column_stack([
    test_preds['xgb'], test_preds['lgb'], test_preds['cat'],
    test_preds['hgb'], test_preds['et'],
    test_pseudo['xgb'], test_pseudo['lgb'], test_pseudo['cat']
])

print(f"Stacking features: {oof_stack.shape[1]}")

# Train Ridge meta-learner
meta_model = Ridge(alpha=1.0)
meta_oof = cross_val_predict(meta_model, oof_stack, y, cv=5, method='predict')
meta_model.fit(oof_stack, y)
meta_test = meta_model.predict(test_stack)

# Clip to valid probability range
meta_oof = np.clip(meta_oof, 0, 1)
meta_test = np.clip(meta_test, 0, 1)

print(f"Meta-learner OOF AUC: {roc_auc_score(y, meta_oof):.5f}")


# Optuna optimization for final blend weights
def objective(trial):
    w_xgb = trial.suggest_float('w_xgb', 0.1, 0.5)
    w_lgb = trial.suggest_float('w_lgb', 0.05, 0.4)
    w_cat = trial.suggest_float('w_cat', 0.05, 0.3)
    w_hgb = trial.suggest_float('w_hgb', 0.0, 0.2)
    w_et = trial.suggest_float('w_et', 0.0, 0.15)
    w_pseudo_xgb = trial.suggest_float('w_pseudo_xgb', 0.0, 0.3)
    w_pseudo_lgb = trial.suggest_float('w_pseudo_lgb', 0.0, 0.2)
    w_pseudo_cat = trial.suggest_float('w_pseudo_cat', 0.0, 0.15)
    w_meta = trial.suggest_float('w_meta', 0.0, 0.3)
    
    total = w_xgb + w_lgb + w_cat + w_hgb + w_et + w_pseudo_xgb + w_pseudo_lgb + w_pseudo_cat + w_meta
    
    blend = (
        w_xgb * oof_preds['xgb'] + 
        w_lgb * oof_preds['lgb'] + 
        w_cat * oof_preds['cat'] +
        w_hgb * oof_preds['hgb'] +
        w_et * oof_preds['et'] +
        w_pseudo_xgb * oof_pseudo['xgb'] +
        w_pseudo_lgb * oof_pseudo['lgb'] +
        w_pseudo_cat * oof_pseudo['cat'] +
        w_meta * meta_oof
    ) / total
    
    return roc_auc_score(y, blend)

print("Running Optuna optimization (300 trials)...\n")

sampler = TPESampler(seed=SEED)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=300, show_progress_bar=True)

print(f"\nBest OOF AUC: {study.best_value:.5f}")
print("\nOptimal Weights:")
for param, value in study.best_params.items():
    print(f"  {param}: {value:.4f}")


# Create final predictions with optimal weights
best = study.best_params
total = sum(best.values())

oof_final = (
    best['w_xgb'] * oof_preds['xgb'] + 
    best['w_lgb'] * oof_preds['lgb'] + 
    best['w_cat'] * oof_preds['cat'] +
    best['w_hgb'] * oof_preds['hgb'] +
    best['w_et'] * oof_preds['et'] +
    best['w_pseudo_xgb'] * oof_pseudo['xgb'] +
    best['w_pseudo_lgb'] * oof_pseudo['lgb'] +
    best['w_pseudo_cat'] * oof_pseudo['cat'] +
    best['w_meta'] * meta_oof
) / total

test_final = (
    best['w_xgb'] * test_preds['xgb'] + 
    best['w_lgb'] * test_preds['lgb'] + 
    best['w_cat'] * test_preds['cat'] +
    best['w_hgb'] * test_preds['hgb'] +
    best['w_et'] * test_preds['et'] +
    best['w_pseudo_xgb'] * test_pseudo['xgb'] +
    best['w_pseudo_lgb'] * test_pseudo['lgb'] +
    best['w_pseudo_cat'] * test_pseudo['cat'] +
    best['w_meta'] * meta_test
) / total

# Compare with simple average
oof_simple = (oof_preds['xgb'] + oof_preds['lgb'] + oof_preds['cat']) / 3

print(f"Optimized Blend OOF AUC: {roc_auc_score(y, oof_final):.5f}")
print(f"Simple Average OOF AUC: {roc_auc_score(y, oof_simple):.5f}")
print(f"Improvement: +{(roc_auc_score(y, oof_final) - roc_auc_score(y, oof_simple))*10000:.1f} (×10⁻⁴)")


# Final visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Weight distribution
ax = axes[0, 0]
weights = pd.Series(best).sort_values(ascending=True)
colors = [PALETTE[i % len(PALETTE)] for i in range(len(weights))]
ax.barh(weights.index, weights.values / total, color=colors)
ax.set_xlabel('Normalized Weight')
ax.set_title('Optimized Ensemble Weights', fontsize=14, fontweight='bold')

# Final ROC
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y, oof_final)
ax.plot(fpr, tpr, color=COLORS['primary'], linewidth=3, 
        label=f'Optimized Ensemble ({roc_auc_score(y, oof_final):.5f})')
fpr_simple, tpr_simple, _ = roc_curve(y, oof_simple)
ax.plot(fpr_simple, tpr_simple, color=COLORS['secondary'], linewidth=2, linestyle='--',
        label=f'Simple Average ({roc_auc_score(y, oof_simple):.5f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax.fill_between(fpr, tpr, alpha=0.2, color=COLORS['primary'])
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('Final ROC Curve', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')

# Prediction distribution
ax = axes[1, 0]
ax.hist(oof_final, bins=50, alpha=0.6, color=COLORS['primary'], label='OOF', density=True)
ax.hist(test_final, bins=50, alpha=0.6, color=COLORS['danger'], label='Test', density=True)
ax.set_xlabel('Predicted Probability')
ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
ax.legend()

# Precision-Recall curve
ax = axes[1, 1]
precision, recall, _ = precision_recall_curve(y, oof_final)
ap = average_precision_score(y, oof_final)
ax.plot(recall, precision, color=COLORS['accent'], linewidth=2, label=f'AP = {ap:.4f}')
ax.fill_between(recall, precision, alpha=0.2, color=COLORS['accent'])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.show()


# Create submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
submission[TARGET] = test_final
submission.to_csv('submission.csv', index=False)

print("Submission saved!")
print(f"\nPrediction Statistics:")
print(f"  Min: {test_final.min():.4f}")
print(f"  Max: {test_final.max():.4f}")
print(f"  Mean: {test_final.mean():.4f}")
print(f"  Std: {test_final.std():.4f}")
print(f"\nOOF AUC: {roc_auc_score(y, oof_final):.5f}")

submission.head(10)

