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


# ============================================================================
# IMPROVED EMPLOYEE ATTRITION PREDICTION - PRODUCTION VERSION
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                               ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV

# Imbalanced learning
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except:
    IMBLEARN_AVAILABLE = False

# Advanced Models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("="*80)
print("IMPROVED EMPLOYEE ATTRITION PREDICTION")
print("="*80)
print("\nOPTIMIZATIONS:")
print("  - Overfitting prevention")
print("  - SMOTE balancing" if IMBLEARN_AVAILABLE else "  - Class weights")
print("  - Feature selection")
print("  - Calibration")
print("  - Diverse ensemble")
print("="*80)

# ============================================================================
# DATA LOADING
# ============================================================================
print("\nSECTION 1: DATA LOADING")
print("-"*80)

train = pd.read_csv('/kaggle/input/tugas-1-sml-a-2025/train.csv')
test = pd.read_csv('/kaggle/input/tugas-1-sml-a-2025/test.csv')

print(f"Train: {train.shape}, Test: {test.shape}")

target_counts = train['Attrition'].value_counts()
print(f"\nClass 0: {target_counts[0]} ({target_counts[0]/len(train)*100:.1f}%)")
print(f"Class 1: {target_counts[1]} ({target_counts[1]/len(train)*100:.1f}%)")
print(f"Ratio: {target_counts[0]/target_counts[1]:.1f}:1")

# ============================================================================
# PREPROCESSING
# ============================================================================
print("\nSECTION 2: PREPROCESSING")
print("-"*80)

X = train.drop(['Attrition', 'id'], axis=1)
y = train['Attrition']
X_test = test.drop(['id'], axis=1)
test_ids = test['id'].copy()

# Remove constant columns
constant_cols = [col for col in X.columns if X[col].nunique() == 1]
if constant_cols:
    X = X.drop(constant_cols, axis=1)
    X_test = X_test.drop(constant_cols, axis=1)
    print(f"Removed: {constant_cols}")

# Encode categorical
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\nSECTION 3: FEATURE ENGINEERING")
print("-"*80)

def create_features(df):
    df = df.copy()
    df['ExperienceRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    df['CurrentRoleRatio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
    df['JobHoppingRate'] = df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1)
    df['ManagerStability'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    df['IncomePerYear'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    df['IsYoung'] = (df['Age'] < 30).astype(int)
    df['ShortTenure'] = (df['YearsAtCompany'] < 2).astype(int)
    df['TimeWithoutPromotion'] = (df['YearsSinceLastPromotion'] > 3).astype(int)
    df['LowJobLevel'] = (df['JobLevel'] <= 1).astype(int)
    df['LongCommute'] = (df['DistanceFromHome'] > 15).astype(int)
    df['PoorWorkLife'] = (df['WorkLifeBalance'] <= 2).astype(int)
    df['OverTime_Binary'] = df['OverTime']
    satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
    df['AvgSatisfaction'] = df[satisfaction_cols].mean(axis=1)
    df['LowSatisfaction'] = (df['AvgSatisfaction'] < 2).astype(int)
    df['HighRisk_Flag'] = ((df['OverTime_Binary'] == 1) & (df['JobSatisfaction'] <= 2) & (df['WorkLifeBalance'] <= 2)).astype(int)
    df['Career_Stagnation'] = ((df['YearsSinceLastPromotion'] > 5) & (df['YearsAtCompany'] > 10)).astype(int)
    df['Income_JobLevel'] = df['MonthlyIncome'] * df['JobLevel']
    df['Age_Experience'] = df['Age'] * df['TotalWorkingYears']
    df['Satisfaction_Performance'] = df['AvgSatisfaction'] * df['PerformanceRating']
    df['AttritionRiskScore'] = (df['LowSatisfaction'] * 2.5 + df['OverTime_Binary'] * 2.0 + df['LongCommute'] * 1.0 + 
                                df['TimeWithoutPromotion'] * 2.0 + df['ShortTenure'] * 1.5 + df['JobHoppingRate'] * 12 + 
                                df['PoorWorkLife'] * 1.5 + df['LowJobLevel'] * 0.5 + df['HighRisk_Flag'] * 3.0 + df['Career_Stagnation'] * 2.0)
    return df

X_eng = create_features(X)
X_test_eng = create_features(X_test)
print(f"Features: {X.shape[1]} -> {X_eng.shape[1]}")

# ============================================================================
# FEATURE SELECTION
# ============================================================================
print("\nSECTION 4: FEATURE SELECTION")
print("-"*80)

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_eng, y)

feat_imp = pd.DataFrame({
    'feature': X_eng.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 20:")
print(feat_imp.head(20).to_string(index=False))

N_FEAT = min(45, len(feat_imp))
selected = feat_imp.head(N_FEAT)['feature'].tolist()
X_sel = X_eng[selected].copy()
X_test_sel = X_test_eng[selected].copy()
print(f"\nSelected: {N_FEAT} features")

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
top = feat_imp.head(30)
ax.barh(range(len(top)), top['importance'], color='steelblue')
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top['feature'], fontsize=9)
ax.set_xlabel('Importance')
ax.set_title('Feature Importances', fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('01_features.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# TRAIN-VAL SPLIT
# ============================================================================
print("\nSECTION 5: TRAIN-VAL SPLIT")
print("-"*80)

X_train, X_val, y_train, y_val = train_test_split(
    X_sel, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)

# ============================================================================
# CLASS BALANCING
# ============================================================================
print("\nSECTION 6: CLASS BALANCING")
print("-"*80)

if IMBLEARN_AVAILABLE:
    smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)
    under = RandomUnderSampler(sampling_strategy=0.7, random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    X_train_bal, y_train_bal = under.fit_resample(X_train_bal, y_train_bal)
    X_train_bal_sc = scaler.fit_transform(X_train_bal)
    print(f"Balanced: {X_train.shape[0]} -> {X_train_bal.shape[0]}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(['No', 'Yes'], y_train.value_counts().sort_index(), color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Original', fontweight='bold')
    axes[1].bar(['No', 'Yes'], pd.Series(y_train_bal).value_counts().sort_index(), color=['#2ecc71', '#e74c3c'])
    axes[1].set_title('After SMOTE', fontweight='bold')
    plt.tight_layout()
    plt.savefig('02_balancing.png', dpi=300, bbox_inches='tight')
    plt.close()
else:
    X_train_bal, y_train_bal = X_train.copy(), y_train.copy()
    X_train_bal_sc = X_train_sc.copy()
    print("Using class_weight='balanced'")

# ============================================================================
# MODEL TRAINING
# ============================================================================
print("\nSECTION 7: MODEL TRAINING")
print("="*80)

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_model(name, model, use_scaled=False):
    print(f"\n{name}...")
    X_tr = X_train_bal_sc if (use_scaled and IMBLEARN_AVAILABLE) else X_train_sc if use_scaled else X_train_bal if IMBLEARN_AVAILABLE else X_train
    y_tr = y_train_bal if IMBLEARN_AVAILABLE else y_train
    X_v = X_val_sc if use_scaled else X_val
    
    model.fit(X_tr, y_tr)
    y_tr_pred = model.predict_proba(X_tr)[:, 1]
    y_v_pred = model.predict_proba(X_v)[:, 1]
    
    tr_score = roc_auc_score(y_tr, y_tr_pred)
    v_score = roc_auc_score(y_val, y_v_pred)
    
    try:
        cv_scores = cross_val_score(model, X_train_sc if use_scaled else X_train, y_train, 
                                    cv=cv, scoring='roc_auc', n_jobs=-1)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    except:
        cv_mean, cv_std = v_score, 0.0
    
    gap = tr_score - v_score
    status = 'GOOD' if gap < 0.05 else 'WARN' if gap < 0.10 else 'BAD'
    
    print(f"  Train: {tr_score:.4f}, Val: {v_score:.4f}, CV: {cv_mean:.4f}, Gap: {gap:.4f} ({status})")
    
    results[name] = {
        'model': model, 'train': tr_score, 'val': v_score, 'cv': cv_mean, 
        'cv_std': cv_std, 'gap': gap, 'pred': y_v_pred, 'scaled': use_scaled
    }

print("\nLINEAR MODELS:")
train_model("Logistic L2", LogisticRegression(max_iter=2000, C=0.05, penalty='l2', random_state=42, class_weight='balanced'), True)
train_model("Logistic L1", LogisticRegression(max_iter=2000, C=0.1, penalty='l1', solver='saga', random_state=42, class_weight='balanced'), True)

print("\nTREE MODELS:")
train_model("Random Forest", RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_split=30, min_samples_leaf=15, 
                                                     max_features='sqrt', random_state=42, n_jobs=-1, class_weight='balanced'), False)
train_model("Extra Trees", ExtraTreesClassifier(n_estimators=500, max_depth=7, min_samples_split=30, min_samples_leaf=15,
                                                 max_features='sqrt', random_state=42, n_jobs=-1, class_weight='balanced'), False)
train_model("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, learning_rate=0.03, max_depth=4,
                                                             min_samples_split=30, min_samples_leaf=15, subsample=0.7, random_state=42), False)

if XGBOOST_AVAILABLE:
    train_model("XGBoost", XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, min_child_weight=5,
                                         subsample=0.7, colsample_bytree=0.7, gamma=1, reg_alpha=0.5, reg_lambda=1.0,
                                         random_state=42, eval_metric='logloss', use_label_encoder=False), False)

if LIGHTGBM_AVAILABLE:
    train_model("LightGBM", LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, num_leaves=20,
                                           min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
                                           reg_alpha=0.5, reg_lambda=1.0, random_state=42, verbose=-1), False)

if CATBOOST_AVAILABLE:
    train_model("CatBoost", CatBoostClassifier(iterations=300, learning_rate=0.03, depth=5, l2_leaf_reg=3.0,
                                               random_state=42, verbose=False), False)

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n\nSECTION 8: COMPARISON")
print("="*80)

df_res = pd.DataFrame({
    'Model': list(results.keys()),
    'Train': [v['train'] for v in results.values()],
    'Val': [v['val'] for v in results.values()],
    'CV': [v['cv'] for v in results.values()],
    'Gap': [v['gap'] for v in results.values()]
}).sort_values('CV', ascending=False)

print("\n" + df_res.to_string(index=False))
print(f"\nBest CV: {df_res['CV'].max():.4f}, Best Val: {df_res['Val'].max():.4f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1
ax = axes[0, 0]
sorted_df = df_res.sort_values('CV')
colors = ['green' if x < 0.05 else 'orange' if x < 0.10 else 'red' for x in sorted_df['Gap']]
ax.barh(range(len(sorted_df)), sorted_df['CV'], color=colors)
ax.set_yticks(range(len(sorted_df)))
ax.set_yticklabels(sorted_df['Model'], fontsize=9)
ax.set_title('CV Scores', fontweight='bold')

# Plot 2
ax = axes[0, 1]
x = np.arange(len(df_res))
w = 0.35
ax.bar(x - w/2, df_res['Train'], w, label='Train', color='lightgreen')
ax.bar(x + w/2, df_res['Val'], w, label='Val', color='steelblue')
ax.set_xticks(x)
ax.set_xticklabels(df_res['Model'], rotation=45, ha='right', fontsize=8)
ax.set_title('Train vs Val', fontweight='bold')
ax.legend()

# Plot 3
ax = axes[1, 0]
colors = ['green' if x < 0.05 else 'orange' if x < 0.10 else 'red' for x in df_res['Gap']]
ax.barh(range(len(df_res)), df_res['Gap'], color=colors)
ax.set_yticks(range(len(df_res)))
ax.set_yticklabels(df_res['Model'], fontsize=9)
ax.set_title('Overfitting Gap', fontweight='bold')
ax.axvline(x=0.05, color='orange', linestyle='--')
ax.axvline(x=0.10, color='red', linestyle='--')

# Plot 4
ax = axes[1, 1]
ax.scatter(df_res['Gap'], df_res['CV'], s=100, alpha=0.6)
for _, row in df_res.iterrows():
    ax.annotate(row['Model'], (row['Gap'], row['CV']), fontsize=7, ha='right')
ax.set_xlabel('Gap')
ax.set_ylabel('CV')
ax.set_title('Generalization', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# CALIBRATION
# ============================================================================
print("\n\nSECTION 9: CALIBRATION")
print("="*80)

top3 = df_res.nsmallest(3, 'Gap')['Model'].tolist()
print(f"Calibrating: {top3}")

calib = {}
for name in top3:
    print(f"\n{name}...")
    r = results[name]
    X_tr = X_train_bal_sc if (r['scaled'] and IMBLEARN_AVAILABLE) else X_train_sc if r['scaled'] else X_train_bal if IMBLEARN_AVAILABLE else X_train
    y_tr = y_train_bal if IMBLEARN_AVAILABLE else y_train
    X_v = X_val_sc if r['scaled'] else X_val
    
    clf = CalibratedClassifierCV(r['model'], method='isotonic', cv=3)
    clf.fit(X_tr, y_tr)
    pred = clf.predict_proba(X_v)[:, 1]
    score = roc_auc_score(y_val, pred)
    
    print(f"  Original: {r['val']:.4f}, Calibrated: {score:.4f}, Gain: {score - r['val']:+.4f}")
    
    calib[f"{name} (Cal)"] = {'model': clf, 'val': score, 'pred': pred, 'scaled': r['scaled']}

# ============================================================================
# ENSEMBLE
# ============================================================================
print("\n\nSECTION 10: ENSEMBLE")
print("="*80)

linear = [n for n in results.keys() if 'Logistic' in n]
trees = [n for n in results.keys() if n not in linear]

best_lin = max(linear, key=lambda x: results[x]['cv'])
best_tree = max(trees, key=lambda x: results[x]['cv'])

print(f"Linear: {best_lin}, Tree: {best_tree}")

ens_models = [best_lin, best_tree]
for c in calib.keys():
    if c.replace(' (Cal)', '') in ens_models:
        ens_models.append(c)

print(f"Ensemble: {ens_models}")

preds, wts = [], []
for m in ens_models:
    if m in results:
        preds.append(results[m]['pred'])
        wts.append(results[m]['cv'])
    elif m in calib:
        preds.append(calib[m]['pred'])
        wts.append(calib[m]['val'])

wts = np.array(wts) / np.sum(wts)

pred_avg = np.mean(preds, axis=0)
pred_wt = np.average(preds, axis=0, weights=wts)

score_avg = roc_auc_score(y_val, pred_avg)
score_wt = roc_auc_score(y_val, pred_wt)

print(f"\nAverage: {score_avg:.4f}, Weighted: {score_wt:.4f}")

best = pred_wt if score_wt > score_avg else pred_avg
best_score = max(score_avg, score_wt)

print(f"Best: {best_score:.4f}, Gain: {best_score - df_res['Val'].max():+.4f}")

# ROC
fig, ax = plt.subplots(figsize=(12, 8))
for name in df_res.head(5)['Model'].tolist():
    pred = results[name]['pred']
    fpr, tpr, _ = roc_curve(y_val, pred)
    ax.plot(fpr, tpr, lw=2, label=f'{name} ({auc(fpr, tpr):.3f})')
fpr, tpr, _ = roc_curve(y_val, best)
ax.plot(fpr, tpr, lw=3, linestyle='--', color='red', label=f'Ensemble ({auc(fpr, tpr):.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('ROC Curves', fontweight='bold')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('04_roc.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# TEST PREDICTIONS
# ============================================================================
print("\n\nSECTION 11: TEST PREDICTIONS")
print("="*80)

test_preds = {}
for m in ens_models:
    if m in results:
        model, scaled = results[m]['model'], results[m]['scaled']
    elif m in calib:
        model, scaled = calib[m]['model'], calib[m]['scaled']
    else:
        continue
    
    X_t = scaler.transform(X_test_sel) if scaled else X_test_sel
    pred = model.predict_proba(X_t)[:, 1]
    test_preds[m] = pred
    print(f"{m}: {pred.mean():.4f}")

final = np.average(list(test_preds.values()), axis=0, weights=wts)
print(f"\nFinal: {final.mean():.4f}, Train: {y.mean():.4f}, Diff: {abs(final.mean() - y.mean()):.4f}")

# ============================================================================
# SUBMISSION
# ============================================================================
print("\n\nSECTION 12: SUBMISSION")
print("="*80)

sub = pd.DataFrame({'id': test_ids, 'Attrition': final})
sub.to_csv('submission.csv', index=False)
print("Created: submission.csv")
print(f"\nMean: {sub['Attrition'].mean():.4f}, Std: {sub['Attrition'].std():.4f}")
print(f"Min: {sub['Attrition'].min():.4f}, Max: {sub['Attrition'].max():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(sub['Attrition'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(x=sub['Attrition'].mean(), color='red', linestyle='--', label='Test')
axes[0].axvline(x=y.mean(), color='green', linestyle='--', label='Train')
axes[0].set_title('Distribution', fontweight='bold')
axes[0].legend()
axes[1].boxplot([sub['Attrition']])
axes[1].axhline(y=y.mean(), color='green', linestyle='--')
axes[1].set_title('Boxplot', fontweight='bold')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('05_submission.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + sub.head(20).to_string(index=False))

print("\n" + "="*80)
print("DONE!")
print("="*80)
print(f"CV: {df_res['CV'].max():.4f}, Val: {df_res['Val'].max():.4f}, Ensemble: {best_score:.4f}")
print("Files: submission.csv + 5 PNG visualizations")
print("="*80)

