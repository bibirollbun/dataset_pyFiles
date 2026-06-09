import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
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

# Beautiful color palette
COLORS = {
    'primary': '#11998e',
    'secondary': '#38ef7d', 
    'accent1': '#667eea',
    'accent2': '#764ba2',
    'danger': '#f5576c',
    'warning': '#ffd93d',
    'info': '#4facfe',
    'dark': '#2d3436',
    'light': '#dfe6e9'
}
COLOR_LIST = ['#11998e', '#667eea', '#f5576c', '#ffd93d', '#4facfe', '#764ba2', '#38ef7d']

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

SEED = 2025
TARGET = 'diagnosed_diabetes'

print("Libraries loaded!")
print(f"Optuna: {optuna.__version__}")
print(f"XGBoost: {xgb.__version__}")


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

try:
    original = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_012_health_indicators_BRFSS2015.csv')
    HAS_ORIGINAL = True
    print(f"External dataset: {original.shape[0]:,} rows")
except:
    HAS_ORIGINAL = False

print(f"Train: {train.shape[0]:,} rows, {train.shape[1]} columns")
print(f"Test: {test.shape[0]:,} rows")
print(f"\nTarget distribution: {train[TARGET].value_counts().to_dict()}")
print(f"Diabetes rate: {train[TARGET].mean():.1%}")


# Save IDs
train_ids = train['id'].copy()
test_ids = test['id'].copy()
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# Define column types
CATEGORICAL = ['gender', 'ethnicity', 'education_level', 'income_level', 
               'smoking_status', 'employment_status']
BINARY = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
NUMERICAL = [c for c in train.columns if c not in CATEGORICAL + BINARY + [TARGET]]

print(f"\nColumn types:")
print(f"  Numerical: {len(NUMERICAL)}")
print(f"  Categorical: {len(CATEGORICAL)}")
print(f"  Binary: {len(BINARY)}")


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Pie chart
target_counts = train[TARGET].value_counts()
axes[0].pie(target_counts.values, labels=['No Diabetes', 'Diabetes'], 
            autopct='%1.1f%%', colors=[COLORS['primary'], COLORS['danger']],
            explode=(0, 0.05), shadow=True, startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[0].set_title('Target Distribution', fontsize=14, fontweight='bold')

# Bar chart
bars = axes[1].bar(['No Diabetes\n(0)', 'Diabetes\n(1)'], target_counts.values,
                   color=[COLORS['primary'], COLORS['danger']], edgecolor='black', linewidth=2)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Class Counts', fontsize=14, fontweight='bold')
for bar, count in zip(bars, target_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                 f'{count:,}', ha='center', fontsize=12, fontweight='bold')

# Imbalance ratio visualization
ratio = target_counts[1] / target_counts[0]
axes[2].barh(['Imbalance Ratio'], [ratio], color=COLORS['warning'], edgecolor='black', height=0.4)
axes[2].axvline(1.0, color='red', linestyle='--', linewidth=2, label='Balanced (1.0)')
axes[2].set_xlim(0, 2)
axes[2].set_title(f'Class Imbalance: {ratio:.2f}', fontsize=14, fontweight='bold')
axes[2].legend()

plt.tight_layout()
plt.show()


# Check missing values - use common columns only
common_cols = [c for c in train.columns if c in test.columns]
missing_train = train[common_cols].isnull().sum()
missing_test = test[common_cols].isnull().sum()

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(common_cols))
width = 0.35

ax.bar(x - width/2, missing_train.values, width, label='Train', color=COLORS['primary'], alpha=0.8)
ax.bar(x + width/2, missing_test.values, width, label='Test', color=COLORS['accent1'], alpha=0.8)

ax.set_ylabel('Missing Count')
ax.set_title('Missing Values by Feature', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(common_cols, rotation=45, ha='right')
ax.legend()

# Add text showing total missing
ax.text(0.02, 0.98, f'Total missing - Train: {missing_train.sum():,}, Test: {missing_test.sum():,}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()


n_cols = 5
n_rows = (len(NUMERICAL) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(NUMERICAL):
    ax = axes[i]
    
    # Plot distributions for each target class
    for target_val, color, label in [(0, COLORS['primary'], 'No Diabetes'), 
                                      (1, COLORS['danger'], 'Diabetes')]:
        data = train[train[TARGET] == target_val][col]
        ax.hist(data, bins=40, alpha=0.6, color=color, label=label, density=True)
    
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=9)
    
    # Add mean lines
    for target_val, color in [(0, COLORS['primary']), (1, COLORS['danger'])]:
        mean_val = train[train[TARGET] == target_val][col].mean()
        ax.axvline(mean_val, color=color, linestyle='--', linewidth=2, alpha=0.8)
    
    if i == 0:
        ax.legend(fontsize=9)

# Hide empty subplots
for i in range(len(NUMERICAL), len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Numerical Features Distribution by Target', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(CATEGORICAL):
    ax = axes[i]
    
    # Calculate diabetes rate per category
    rates = train.groupby(col)[TARGET].agg(['mean', 'count']).reset_index()
    rates.columns = [col, 'diabetes_rate', 'count']
    rates = rates.sort_values('diabetes_rate', ascending=True)
    
    # Create horizontal bar chart
    colors = plt.cm.RdYlGn_r(rates['diabetes_rate'])
    bars = ax.barh(rates[col].astype(str), rates['diabetes_rate'], color=colors, edgecolor='black')
    
    ax.axvline(train[TARGET].mean(), color='black', linestyle='--', linewidth=2, label='Overall Rate')
    ax.set_xlabel('Diabetes Rate')
    ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Add count labels
    for bar, count in zip(bars, rates['count']):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'n={count:,}', va='center', fontsize=9)

plt.suptitle('Diabetes Rate by Categorical Features', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, col in enumerate(BINARY):
    ax = axes[i]
    
    # Get diabetes rates
    rates = train.groupby(col)[TARGET].mean()
    counts = train.groupby(col)[TARGET].count()
    
    # Create stacked bar showing composition
    x = ['No', 'Yes']
    diabetes_counts = train.groupby([col, TARGET]).size().unstack(fill_value=0)
    
    ax.bar(x, diabetes_counts[0], color=COLORS['primary'], label='No Diabetes', edgecolor='black')
    ax.bar(x, diabetes_counts[1], bottom=diabetes_counts[0], color=COLORS['danger'], 
           label='Diabetes', edgecolor='black')
    
    # Add rate annotations
    for j, (rate, count) in enumerate(zip(rates.values, counts.values)):
        ax.text(j, count + 2000, f'{rate:.1%}', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel('Count')
    if i == 0:
        ax.legend(loc='upper right')

plt.suptitle('Binary Features: Composition & Diabetes Rate', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# Encode categoricals for correlation
train_encoded = train.copy()
for col in CATEGORICAL:
    train_encoded[col] = LabelEncoder().fit_transform(train_encoded[col])

corr_matrix = train_encoded.corr()

fig, ax = plt.subplots(figsize=(16, 14))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


target_corr = corr_matrix[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))

colors = [COLORS['danger'] if x > 0 else COLORS['primary'] for x in target_corr.values]
bars = ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor='black')

ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Correlation with Target', fontsize=12)
ax.set_title('Feature Correlation with Diabetes (Sorted by Absolute Value)', fontsize=14, fontweight='bold')

# Add value labels
for bar, val in zip(bars, target_corr.values):
    x_pos = val + 0.01 if val > 0 else val - 0.01
    ha = 'left' if val > 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
            va='center', ha=ha, fontsize=9)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot
ax = axes[0]
scatter = ax.scatter(train['age'], train['bmi'], c=train[TARGET], 
                     cmap='RdYlGn_r', alpha=0.3, s=10)
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('BMI', fontsize=12)
ax.set_title('Age vs BMI by Diabetes Status', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Diabetes')

# Heatmap of diabetes rates
ax = axes[1]
train['age_bin'] = pd.cut(train['age'], bins=10, labels=False)
train['bmi_bin'] = pd.cut(train['bmi'], bins=10, labels=False)
pivot = train.groupby(['age_bin', 'bmi_bin'])[TARGET].mean().unstack()

sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Diabetes Rate'})
ax.set_xlabel('BMI Bin', fontsize=12)
ax.set_ylabel('Age Bin', fontsize=12)
ax.set_title('Diabetes Rate by Age × BMI', fontsize=14, fontweight='bold')

train.drop(['age_bin', 'bmi_bin'], axis=1, inplace=True)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Systolic vs Diastolic
ax = axes[0]
for target_val, color, label in [(0, COLORS['primary'], 'No Diabetes'), 
                                  (1, COLORS['danger'], 'Diabetes')]:
    data = train[train[TARGET] == target_val]
    ax.scatter(data['systolic_bp'], data['diastolic_bp'], 
               c=color, alpha=0.2, s=5, label=label)
ax.set_xlabel('Systolic BP')
ax.set_ylabel('Diastolic BP')
ax.set_title('Blood Pressure Distribution', fontsize=12, fontweight='bold')
ax.legend()

# Systolic BP distribution by target
ax = axes[1]
train[train[TARGET] == 0]['systolic_bp'].hist(bins=40, alpha=0.6, color=COLORS['primary'], 
                                               label='No Diabetes', density=True, ax=ax)
train[train[TARGET] == 1]['systolic_bp'].hist(bins=40, alpha=0.6, color=COLORS['danger'], 
                                               label='Diabetes', density=True, ax=ax)
ax.set_xlabel('Systolic BP')
ax.set_title('Systolic BP Distribution by Target', fontsize=12, fontweight='bold')
ax.legend()

# Box plots
ax = axes[2]
bp_data = train[['systolic_bp', 'diastolic_bp', TARGET]].melt(id_vars=TARGET)
sns.boxplot(data=bp_data, x='variable', y='value', hue=TARGET, ax=ax,
            palette={0: COLORS['primary'], 1: COLORS['danger']})
ax.set_xlabel('')
ax.set_ylabel('mmHg')
ax.set_title('Blood Pressure by Target', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


lipid_cols = ['cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(lipid_cols):
    ax = axes[i]
    
    # Violin plot
    parts = ax.violinplot([train[train[TARGET] == 0][col].dropna(),
                           train[train[TARGET] == 1][col].dropna()],
                          positions=[0, 1], showmeans=True, showmedians=True)
    
    for pc, color in zip(parts['bodies'], [COLORS['primary'], COLORS['danger']]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Diabetes', 'Diabetes'])
    ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel('mg/dL')
    
    # Add mean values
    for j, target_val in enumerate([0, 1]):
        mean_val = train[train[TARGET] == target_val][col].mean()
        ax.text(j, ax.get_ylim()[1] * 0.95, f'Mean: {mean_val:.1f}', 
                ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Lipid Profile by Diabetes Status', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


lifestyle_cols = ['physical_activity_minutes_per_week', 'sleep_hours_per_day', 
                  'screen_time_hours_per_day', 'diet_score', 'alcohol_consumption_per_week']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(lifestyle_cols):
    ax = axes[i]
    
    # KDE plot
    for target_val, color, label in [(0, COLORS['primary'], 'No Diabetes'), 
                                      (1, COLORS['danger'], 'Diabetes')]:
        data = train[train[TARGET] == target_val][col].dropna()
        sns.kdeplot(data, ax=ax, color=color, label=label, linewidth=2, fill=True, alpha=0.3)
    
    ax.set_title(col.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

axes[5].set_visible(False)

plt.suptitle('Lifestyle Factors Distribution by Target', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

for i, col in enumerate(NUMERICAL):
    ax = axes[i]
    train[col].hist(bins=40, alpha=0.5, color=COLORS['primary'], label='Train', density=True, ax=ax)
    test[col].hist(bins=40, alpha=0.5, color=COLORS['accent1'], label='Test', density=True, ax=ax)
    ax.set_title(col, fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.legend(fontsize=8)

for i in range(len(NUMERICAL), len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Train vs Test Distribution Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


def add_digit_features(df, cols, suffix='_dig'):
    """Extract digit features - useful for synthetic data."""
    new_cols = []
    for col in cols:
        df[f'{col}{suffix}1'] = ((df[col] * 10) % 10).fillna(-1).astype('int8')
        new_cols.append(f'{col}{suffix}1')
    return new_cols

def add_bin_features(df, cols, q=5, suffix='_bin'):
    """Create quantile-based bins."""
    new_cols = []
    for col in cols:
        df[f'{col}{suffix}'], _ = pd.qcut(df[col], q=q, labels=False, retbins=True, duplicates='drop')
        new_cols.append(f'{col}{suffix}')
    return new_cols

def create_advanced_features(df):
    """Domain-based feature engineering."""
    df = df.copy()
    
    # Lipid ratios
    df['non_hdl_chol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    df['total_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
    df['trig_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)
    df['trig_total_ratio'] = df['triglycerides'] / (df['cholesterol_total'] + 1)
    
    # Blood pressure
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['map'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
    
    # Body composition
    df['bmi_risk'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 35, 100], labels=[0, 1, 2, 3, 4]).astype(float)
    df['abdominal_obesity'] = (df['waist_to_hip_ratio'] > 0.9).astype(int)
    df['bmi_whr'] = df['bmi'] * df['waist_to_hip_ratio']
    
    # Lifestyle
    df['sedentary'] = (df['physical_activity_minutes_per_week'] < 150).astype(int)
    df['high_screen'] = (df['screen_time_hours_per_day'] > 6).astype(int)
    df['poor_sleep'] = ((df['sleep_hours_per_day'] < 6) | (df['sleep_hours_per_day'] > 9)).astype(int)
    df['poor_diet'] = (df['diet_score'] < 5).astype(int)
    df['lifestyle_risk'] = df['sedentary'] + df['high_screen'] + df['poor_sleep'] + df['poor_diet']
    df['activity_per_bmi'] = df['physical_activity_minutes_per_week'] / (df['bmi'] + 1)
    
    # Age
    df['age_decade'] = (df['age'] // 10).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(float)
    
    # Comorbidities
    df['comorbidity'] = (df['family_history_diabetes'] + 
                         df['hypertension_history'] + 
                         df['cardiovascular_history'])
    
    # Interactions
    df['age_bmi'] = df['age'] * df['bmi']
    df['age_comorbid'] = df['age'] * df['comorbidity']
    df['bmi_bp'] = df['bmi'] * df['systolic_bp']
    df['age_bp'] = df['age'] * df['systolic_bp']
    df['bmi_trig'] = df['bmi'] * df['triglycerides']
    df['age_chol'] = df['age'] * df['cholesterol_total']
    
    # Metabolic score
    df['metabolic_score'] = (df['bmi_risk'] + df['abdominal_obesity'] + 
                             (df['triglycerides'] > 150).astype(int) +
                             (df['systolic_bp'] > 130).astype(int))
    
    return df

# Apply
train_fe = create_advanced_features(train)
test_fe = create_advanced_features(test)

new_features = [c for c in train_fe.columns if c not in train.columns]
print(f"Created {len(new_features)} domain features")


# Digit and bin features
key_numerics = ['bmi', 'cholesterol_total', 'hdl_cholesterol', 'triglycerides', 
                'systolic_bp', 'waist_to_hip_ratio']
digit_cols = add_digit_features(train_fe, key_numerics)
_ = add_digit_features(test_fe, key_numerics)

bin_cols = add_bin_features(train_fe, NUMERICAL, q=10)
_ = add_bin_features(test_fe, NUMERICAL, q=10)

print(f"Added {len(digit_cols)} digit features")
print(f"Added {len(bin_cols)} bin features")


# Encode categoricals
cat_encoded = []
for col in CATEGORICAL:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col], test_fe[col]])
    le.fit(combined)
    train_fe[f'{col}_enc'] = le.transform(train_fe[col])
    test_fe[f'{col}_enc'] = le.transform(test_fe[col])
    cat_encoded.append(f'{col}_enc')

# Count encoding
count_cols = []
for col in CATEGORICAL:
    counts = train_fe[col].value_counts()
    train_fe[f'{col}_cnt'] = train_fe[col].map(counts)
    test_fe[f'{col}_cnt'] = test_fe[col].map(counts).fillna(0)
    count_cols.append(f'{col}_cnt')

print(f"Added {len(cat_encoded)} label + {len(count_cols)} count encoded")


# Final features
FEATURES = (NUMERICAL + BINARY + cat_encoded + count_cols + new_features + digit_cols + bin_cols)
FEATURES = [c for c in FEATURES if c in train_fe.columns and c != TARGET]

X = train_fe[FEATURES].values.astype(np.float32)
y = train_fe[TARGET].values
X_test = test_fe[FEATURES].values.astype(np.float32)

print(f"\nTotal features: {len(FEATURES)}")
print(f"X: {X.shape}, X_test: {X_test.shape}")


# Model configurations
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'learning_rate': 0.008, 'max_depth': 5, 'subsample': 0.93,
    'colsample_bytree': 0.2, 'reg_alpha': 2.0, 'reg_lambda': 0.7,
    'min_child_weight': 5, 'max_bin': 512, 'n_estimators': 15000,
    'early_stopping_rounds': 300, 'device': 'cuda', 'tree_method': 'hist',
    'random_state': SEED
}

lgb_params = {
    'objective': 'binary', 'metric': 'auc',
    'learning_rate': 0.005, 'max_depth': 4, 'num_leaves': 31,
    'min_child_samples': 100, 'subsample': 0.85, 'colsample_bytree': 0.5,
    'reg_alpha': 0.3, 'reg_lambda': 8.0, 'max_bin': 200,
    'n_estimators': 15000, 'device': 'gpu', 'verbose': -1,
    'random_state': SEED
}

cat_params = {
    'iterations': 15000, 'depth': 6, 'learning_rate': 0.008,
    'l2_leaf_reg': 3.0, 'border_count': 128, 'task_type': 'GPU',
    'verbose': 0, 'early_stopping_rounds': 300, 'random_seed': SEED
}

print("Model configs ready!")


N_SPLITS = 5
N_REPEATS = 2

rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

n_folds = N_SPLITS * N_REPEATS
fold_counts = np.zeros(len(X))
fold_scores = {'xgb': [], 'lgb': [], 'cat': []}

print(f"Training {n_folds} folds...")
print("="*70)

for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    fold_counts[val_idx] += 1
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    oof_xgb[val_idx] += xgb_pred
    test_xgb += xgb_model.predict_proba(X_test)[:, 1] / n_folds
    xgb_score = roc_auc_score(y_val, xgb_pred)
    fold_scores['xgb'].append(xgb_score)
    
    # LightGBM
    lgb_model = LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
    oof_lgb[val_idx] += lgb_pred
    test_lgb += lgb_model.predict_proba(X_test)[:, 1] / n_folds
    lgb_score = roc_auc_score(y_val, lgb_pred)
    fold_scores['lgb'].append(lgb_score)
    
    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    cat_pred = cat_model.predict_proba(X_val)[:, 1]
    oof_cat[val_idx] += cat_pred
    test_cat += cat_model.predict_proba(X_test)[:, 1] / n_folds
    cat_score = roc_auc_score(y_val, cat_pred)
    fold_scores['cat'].append(cat_score)
    
    print(f"   XGB: {xgb_score:.5f} | LGB: {lgb_score:.5f} | CAT: {cat_score:.5f}")
    gc.collect()

oof_xgb /= fold_counts
oof_lgb /= fold_counts
oof_cat /= fold_counts

print("\n" + "="*70)
print(f"\nXGBoost OOF: {roc_auc_score(y, oof_xgb):.5f}")
print(f"LightGBM OOF: {roc_auc_score(y, oof_lgb):.5f}")
print(f"CatBoost OOF: {roc_auc_score(y, oof_cat):.5f}")


# Visualize model performance
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Fold scores
ax = axes[0]
x = np.arange(1, n_folds + 1)
ax.plot(x, fold_scores['xgb'], 'o-', color=COLORS['primary'], label='XGBoost', linewidth=2)
ax.plot(x, fold_scores['lgb'], 's-', color=COLORS['accent1'], label='LightGBM', linewidth=2)
ax.plot(x, fold_scores['cat'], '^-', color=COLORS['danger'], label='CatBoost', linewidth=2)
ax.set_xlabel('Fold')
ax.set_ylabel('AUC')
ax.set_title('Fold-by-Fold Performance', fontsize=12, fontweight='bold')
ax.legend()

# Box plot
ax = axes[1]
bp = ax.boxplot([fold_scores['xgb'], fold_scores['lgb'], fold_scores['cat']], 
                labels=['XGBoost', 'LightGBM', 'CatBoost'], patch_artist=True)
for patch, color in zip(bp['boxes'], [COLORS['primary'], COLORS['accent1'], COLORS['danger']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('AUC')
ax.set_title('Score Distribution', fontsize=12, fontweight='bold')

# ROC curves
ax = axes[2]
for name, oof, color in [('XGBoost', oof_xgb, COLORS['primary']),
                          ('LightGBM', oof_lgb, COLORS['accent1']),
                          ('CatBoost', oof_cat, COLORS['danger'])]:
    fpr, tpr, _ = roc_curve(y, oof)
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{name} ({roc_auc_score(y, oof):.5f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('ROC Curves', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()


def objective(trial):
    w_xgb = trial.suggest_float('w_xgb', 0.1, 0.8)
    w_lgb = trial.suggest_float('w_lgb', 0.1, 0.8)
    w_cat = trial.suggest_float('w_cat', 0.05, 0.5)
    
    total = w_xgb + w_lgb + w_cat
    w_xgb, w_lgb, w_cat = w_xgb/total, w_lgb/total, w_cat/total
    
    blend = w_xgb * oof_xgb + w_lgb * oof_lgb + w_cat * oof_cat
    return roc_auc_score(y, blend)

print("Running Optuna optimization...\n")

sampler = TPESampler(seed=SEED)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=200, show_progress_bar=True)

best_params = study.best_params
total = best_params['w_xgb'] + best_params['w_lgb'] + best_params['w_cat']
w_xgb = best_params['w_xgb'] / total
w_lgb = best_params['w_lgb'] / total
w_cat = best_params['w_cat'] / total

print(f"\nOptimal Weights:")
print(f"  XGBoost: {w_xgb:.4f}")
print(f"  LightGBM: {w_lgb:.4f}")
print(f"  CatBoost: {w_cat:.4f}")
print(f"\nBest AUC: {study.best_value:.5f}")


# Create final predictions
oof_final = w_xgb * oof_xgb + w_lgb * oof_lgb + w_cat * oof_cat
test_final = w_xgb * test_xgb + w_lgb * test_lgb + w_cat * test_cat

oof_avg = (oof_xgb + oof_lgb + oof_cat) / 3
test_avg = (test_xgb + test_lgb + test_cat) / 3

print(f"Optimized Blend: {roc_auc_score(y, oof_final):.5f}")
print(f"Simple Average: {roc_auc_score(y, oof_avg):.5f}")


# Final visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Weight pie chart
ax = axes[0]
ax.pie([w_xgb, w_lgb, w_cat], 
       labels=[f'XGB\n{w_xgb:.1%}', f'LGB\n{w_lgb:.1%}', f'CAT\n{w_cat:.1%}'],
       colors=[COLORS['primary'], COLORS['accent1'], COLORS['danger']],
       explode=(0.05, 0, 0), shadow=True, startangle=90,
       textprops={'fontsize': 12, 'fontweight': 'bold'})
ax.set_title('Ensemble Weights', fontsize=14, fontweight='bold')

# Prediction distributions
ax = axes[1]
ax.hist(oof_final, bins=50, alpha=0.6, color=COLORS['primary'], label='OOF', density=True)
ax.hist(test_final, bins=50, alpha=0.6, color=COLORS['danger'], label='Test', density=True)
ax.set_xlabel('Predicted Probability')
ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
ax.legend()

# Final ROC
ax = axes[2]
fpr, tpr, _ = roc_curve(y, oof_final)
ax.plot(fpr, tpr, color=COLORS['primary'], linewidth=3, 
        label=f'Ensemble ({roc_auc_score(y, oof_final):.5f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.fill_between(fpr, tpr, alpha=0.3, color=COLORS['primary'])
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('Final Ensemble ROC', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()


# Use best blend
if roc_auc_score(y, oof_final) > roc_auc_score(y, oof_avg):
    final_pred = test_final
    print("Using optimized blend")
else:
    final_pred = test_avg
    print("Using simple average")

submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
submission[TARGET] = final_pred
submission.to_csv('submission.csv', index=False)

print(f"\nSubmission saved!")
print(f"Range: [{final_pred.min():.4f}, {final_pred.max():.4f}]")
print(f"Mean: {final_pred.mean():.4f}")
submission.head()

