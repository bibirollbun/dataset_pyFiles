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


import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 42
COMP_DIR = Path("/kaggle/input/playground-series-s5e3")
WORK_DIR = Path("/kaggle/working")
np.set_printoptions(suppress=True)
pd.set_option("display.max_columns", 100)



train = pd.read_csv(COMP_DIR / "train.csv")
test  = pd.read_csv(COMP_DIR / "test.csv")
sub   = pd.read_csv(COMP_DIR / "sample_submission.csv")

print(train.shape, test.shape, sub.shape)
train.head()


train_info = pd.DataFrame({
    "n_rows":[len(train)],
    "n_columns":[train.shape[1]],
    "target_guess":[ "rainfall" if "rainfall" in train.columns else sub.columns[1] ],
    "missing_total":[int(train.isna().sum().sum())]
})
display(train_info)

train.isna().sum().sort_values(ascending=False).head(20)


target_col = "rainfall" if "rainfall" in train.columns else sub.columns[1]
assert target_col in train.columns, f"Target column '{target_col}' not found!"

vc = train[target_col].value_counts().sort_index()
print("Target distribution:\n", vc, "\nRatio (positive):", float(train[target_col].mean()))

plt.figure()
vc.plot(kind="bar")
plt.title("Target distribution (0 = no rain, 1 = rain)")
plt.xlabel(target_col)
plt.ylabel("count")
plt.tight_layout()
plt.show()



numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
if target_col in numeric_cols:
    corr = train[numeric_cols].corr(numeric_only=True)[target_col].drop(labels=[target_col]).sort_values()
    corr.tail(12).plot(kind="barh", figsize=(6,4))
    plt.title("Top correlations with target")
    plt.xlabel("Pearson correlation")
    plt.tight_layout()
    plt.show()
    corr.sort_values(ascending=False).head(10)
else:
    print("Target not numeric; skipping correlation.")



# ensure consistent casing and pick target
df = train.copy()
df.columns = [c.strip().lower() for c in df.columns]
target_col = "rainfall"

# keep only needed columns and drop NaNs
df = df[["sunshine", "cloud", target_col]].replace([np.inf, -np.inf], np.nan).dropna()

plt.figure(figsize=(6,4))
for t, label in [(0, "no rain"), (1, "rain")]:
    sub = df[df[target_col] == t]
    plt.scatter(sub["sunshine"], sub["cloud"], alpha=0.6, s=20, label=label)

plt.title("Sunshine vs Cloud by rainfall")
plt.xlabel("Sunshine (hours)")
plt.ylabel("Cloud (%)")
plt.legend()
plt.tight_layout()
plt.show()



def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # light normalization; handle common typos safely
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)
    # optional typo fix if present
    if "temparature" in df.columns and "temperature" not in df.columns:
        df = df.rename(columns={"temparature": "temperature"})
    return df

def add_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "day" in df.columns:
        df["day_sin"] = np.sin(2*np.pi*df["day"]/365.0)
        df["day_cos"] = np.cos(2*np.pi*df["day"]/365.0)
    if "winddirection" in df.columns:
        df["winddir_sin"] = np.sin(2*np.pi*df["winddirection"]/360.0)
        df["winddir_cos"] = np.cos(2*np.pi*df["winddirection"]/360.0)
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    drop_cols = [c for c in ("id",) if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = add_cyclical(X)
    # NEW: protect against infs produced upstream
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


train = clean_columns(train)
test  = clean_columns(test)
target_col = target_col.lower()
display(train.head())


feature_cols = [c for c in train.columns if c not in (target_col, "id")]
X = make_features(train[feature_cols])
y = train[target_col].astype(int)

# Ensure test has same columns as X (after make_features)
X_test_raw = make_features(test)
X_test = X_test_raw.reindex(columns=X.columns, fill_value=0)

X.shape, X_test.shape



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

models = {
    "LogReg (balanced)": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))
    ]),
    "RandomForest (balanced)": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
        ))
    ]),
    "GradientBoosting": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])
}

def eval_model(name, model, X, y, cv):
    if hasattr(model, "predict_proba"):
        y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:,1]
    elif hasattr(model, "decision_function"):
        s = cross_val_predict(model, X, y, cv=cv, method="decision_function")
        y_prob = (s - s.min())/(s.max()-s.min() + 1e-9)
    else:
        y_prob = cross_val_predict(model, X, y, cv=cv, method="predict")
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "Model": name,
        "ROC_AUC": roc_auc_score(y, y_prob),
        "F1(0.5)": f1_score(y, y_pred),
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred)
    }

results = []
oof_probs = {}
for name, model in models.items():
    print(f"CV evaluating: {name}")
    if hasattr(model, "predict_proba"):
        y_prob = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:,1]
    elif hasattr(model, "decision_function"):
        s = cross_val_predict(model, X, y, cv=skf, method="decision_function")
        y_prob = (s - s.min())/(s.max()-s.min() + 1e-9)
    else:
        y_prob = cross_val_predict(model, X, y, cv=skf, method="predict")
    oof_probs[name] = y_prob
    y_pred = (y_prob >= 0.5).astype(int)
    results.append({
        "Model": name,
        "ROC_AUC": roc_auc_score(y, y_prob),
        "F1(0.5)": f1_score(y, y_pred),
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred)
    })

results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
results_df



rf = RandomForestClassifier(
    n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
ax = imp.head(15).iloc[::-1].plot(kind="barh", figsize=(7,4))
ax.set_title("Random Forest — top feature importances")
ax.set_xlabel("importance")
plt.tight_layout()
plt.show()
imp.head(20)



best_name = results_df.iloc[0]["Model"]
oof = oof_probs[best_name]

ths = np.linspace(0.2, 0.8, 61)
f1s = []
for t in ths:
    f1s.append(f1_score(y, (oof >= t).astype(int)))

best_idx = int(np.argmax(f1s))
best_thr = float(ths[best_idx])
best_f1  = float(f1s[best_idx])
best_name, best_thr, best_f1



# Load sample submission to get the correct format
sub = pd.read_csv(COMP_DIR / "sample_submission.csv")

# Prepare X_test from the test data with same preprocessing
test_cleaned = clean_columns(test)
test_features = make_features(test_cleaned)
X_test_aligned = test_features.reindex(columns=X.columns, fill_value=0)

# IMPORTANT: Make sure we're predicting for ALL rows in test/submission
print(f"Test data shape: {test.shape}")
print(f"X_test shape: {X_test_aligned.shape}")
print(f"Submission shape: {sub.shape}")

# Verify they match
assert len(X_test_aligned) == len(sub), f"Mismatch: X_test has {len(X_test_aligned)} rows but submission needs {len(sub)} rows"

# Now fit final model and predict
final_model = models[best_name]
final_model.fit(X, y)

# Predict probabilities for test
if hasattr(final_model, "predict_proba"):
    test_prob = final_model.predict_proba(X_test_aligned)[:, 1]
elif hasattr(final_model, "decision_function"):
    s = final_model.decision_function(X_test_aligned)
    test_prob = (s - s.min()) / (s.max() - s.min() + 1e-9)
else:
    test_prob = final_model.predict(X_test_aligned).astype(float)

print(f"Predicted {len(test_prob)} probabilities")

# Make two submissions:
# 1) probabilities (often better for AUC/logloss metrics)
sub_proba = sub.copy()
target_sub_col = sub.columns[1]  # use the sample_submission target column name
sub_proba[target_sub_col] = test_prob
proba_path = WORK_DIR / "submission_proba.csv"
sub_proba.to_csv(proba_path, index=False)

# 2) labels using best F1 threshold
sub_label = sub.copy()
sub_label[target_sub_col] = (test_prob >= best_thr).astype(int)
label_path = WORK_DIR / "submission_labels.csv"
sub_label.to_csv(label_path, index=False)

print("Wrote:", proba_path, "and", label_path)
display(sub_proba.head())
display(sub_label.head())


# ADDITIONAL MODELS TESTED - PROJECT CHECKPOINT 2 



print("TASK 1: EXPLORING MULTIPLE PREDICTIVE METHODS")


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Expand your models dictionary with additional methods

expanded_models = {
    # Your existing models (FIXED)
    "Logistic Regression": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", 
                                   random_state=RANDOM_STATE, solver='liblinear'))
    ]),
    "Random Forest": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", 
                                       random_state=RANDOM_STATE, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ]),
    
    # NEW: Additional classification methods
    "Support Vector Machine": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, class_weight="balanced", kernel='rbf', random_state=RANDOM_STATE))
    ]),
    "K-Nearest Neighbors": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=15, weights='distance'))
    ]),
    "Naive Bayes": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", GaussianNB())
    ]),
    "Neural Network": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, 
                              early_stopping=True, random_state=RANDOM_STATE))
    ]),
    
    # Ensemble methods
    "Voting Ensemble": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("vote", VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', 
                                              random_state=RANDOM_STATE, n_jobs=-1)),
                ('gb', GradientBoostingClassifier(random_state=RANDOM_STATE)),
                ('lr', Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', 
                                              random_state=RANDOM_STATE, solver='liblinear'))
                ]))
            ],
            voting='soft',
            n_jobs=-1
        ))
    ]),
    "Stacking Ensemble": Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("stack", StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', 
                                              random_state=RANDOM_STATE, n_jobs=-1)),
                ('gb', GradientBoostingClassifier(random_state=RANDOM_STATE))
            ],
            final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, solver='liblinear'),
            cv=5,
            n_jobs=-1
        ))
    ])
}
    
    

# Evaluate all methods
all_results = []
all_oof_probs = {}

for name, model in expanded_models.items():
    print(f"\nEvaluating: {name}")
    try:
        # Get out-of-fold predictions
        if hasattr(model.named_steps[list(model.named_steps.keys())[-1]], "predict_proba") or \
           hasattr(model, "predict_proba"):
            y_prob = cross_val_predict(model, X, y, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
        elif hasattr(model.named_steps[list(model.named_steps.keys())[-1]], "decision_function"):
            s = cross_val_predict(model, X, y, cv=skf, method="decision_function", n_jobs=-1)
            y_prob = (s - s.min()) / (s.max() - s.min() + 1e-9)
        else:
            y_prob = cross_val_predict(model, X, y, cv=skf, method="predict", n_jobs=-1).astype(float)
        
        all_oof_probs[name] = y_prob
        y_pred = (y_prob >= 0.5).astype(int)
        
        all_results.append({
            'Model': name,
            'ROC_AUC': roc_auc_score(y, y_prob),
            'F1 (0.5)': f1_score(y, y_pred),
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred),
            'Recall': recall_score(y, y_pred)
        })
        print(f"   ROC_AUC: {roc_auc_score(y, y_prob):.4f}")
    except Exception as e:
        print(f"   Error: {e}")
        continue

# Create comprehensive results table
all_results_df = pd.DataFrame(all_results).sort_values('ROC_AUC', ascending=False).reset_index(drop=True)


print("COMPREHENSIVE MODEL COMPARISON")

display(all_results_df)

# Visualization: Compare all models
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: ROC-AUC comparison
ax1 = axes[0]
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(all_results_df))]
ax1.barh(all_results_df['Model'], all_results_df['ROC_AUC'], color=colors)
ax1.set_xlabel('ROC-AUC Score', fontsize=12)
ax1.set_title('Model Comparison: ROC-AUC', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, v in enumerate(all_results_df['ROC_AUC']):
    ax1.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)

# Plot 2: F1 Score comparison
ax2 = axes[1]
ax2.barh(all_results_df['Model'], all_results_df['F1 (0.5)'], color=colors)
ax2.set_xlabel('F1 Score', fontsize=12)
ax2.set_title('Model Comparison: F1 Score', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, v in enumerate(all_results_df['F1 (0.5)']):
    ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.show()

# Key findings summary

print("KEY FINDINGS")

print(f"Total methods tested: {len(all_results_df)}")
print(f"Best performing model: {all_results_df.iloc[0]['Model']}")
print(f"Best ROC-AUC: {all_results_df.iloc[0]['ROC_AUC']:.4f}")
print(f"Performance range: {all_results_df['ROC_AUC'].min():.4f} - {all_results_df['ROC_AUC'].max():.4f}")
print(f"Ensemble methods rank: ", end="")
ensemble_models = all_results_df[all_results_df['Model'].str.contains('Ensemble|Voting|Stacking')]
print(f"Top {len(ensemble_models[ensemble_models['ROC_AUC'] >= all_results_df['ROC_AUC'].quantile(0.75)])} "
      f"in top quartile")



# PROJECT CHECKPOINT 2: ANALYZE RESULTS AND IDENTIFY ISSUES



print("ANALYZING RESULTS AND IDENTIFYING ISSUES")

issues_identified = []

# ISSUE 1: Class Imbalance

print("CLASS IMBALANCE")

class_0_count = (y == 0).sum()
class_1_count = (y == 1).sum()
class_0_pct = (y == 0).mean() * 100
class_1_pct = (y == 1).mean() * 100
imbalance_ratio = class_0_count / class_1_count

print(f"Class 0 (no rain):  {class_0_count:,} samples ({class_0_pct:.1f}%)")
print(f"Class 1 (rain):     {class_1_count:,} samples ({class_1_pct:.1f}%)")
print(f"Imbalance ratio:    {imbalance_ratio:.2f}:1")
print(f"\n   IMPACT:")
print(f"   Naive 'always predict majority' achieves {max(class_0_pct, class_1_pct):.1f}% accuracy")
print(f"   Models may be biased toward majority class")
print(f"   Metrics like accuracy can be misleading")
print(f"\nMITIGATION APPLIED:")
print(f"   Used class_weight='balanced' in tree models")
print(f"   Using ROC-AUC and F1 score (better for imbalanced data)")

issues_identified.append({
    'Issue': 'Class Imbalance',
    'Severity': 'High',
    'Impact': f'{imbalance_ratio:.1f}:1 ratio affects model bias',
    'Mitigation': 'class_weight=balanced, appropriate metrics'
})

# ISSUE 2: Model Performance Variance (Overfitting Detection)

print("MODEL VARIANCE & OVERFITTING RISK")

print("Testing cross-validation stability for top 3 models...\n")

variance_analysis = []
for name in all_results_df.head(3)['Model']:
    model = expanded_models[name]
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    variance_analysis.append({
        'Model': name,
        'Mean CV Score': cv_scores.mean(),
        'Std Dev': cv_scores.std(),
        'Min': cv_scores.min(),
        'Max': cv_scores.max()
    })
    
    print(f"{name}:")
    print(f"   Mean ROC-AUC: {cv_scores.mean():.4f}")
    print(f"   Std Dev:      {cv_scores.std():.4f}")
    print(f"   Range:        [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
    
    if cv_scores.std() > 0.02:
        print(f"   High variance detected - potential overfitting risk")
    else:
        print(f"   Stable performance across folds")
    print()

variance_df = pd.DataFrame(variance_analysis)

issues_identified.append({
    'Issue': 'Model Variance',
    'Severity': 'Medium',
    'Impact': f'Std dev up to {variance_df["Std Dev"].max():.4f}',
    'Mitigation': 'Cross-validation, regularization'
})

# ISSUE 3: Feature Correlation (Multicollinearity)

print("MULTICOLLINEARITY IN FEATURES")


# Calculate correlation matrix
numeric_cols = X.select_dtypes(include=[np.number]).columns
corr_matrix = X[numeric_cols].corr().abs()

# Find highly correlated pairs
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = []

for column in upper_tri.columns:
    for row in upper_tri.index:
        corr_val = upper_tri.loc[row, column]
        if not pd.isna(corr_val) and corr_val > 0.9:
            high_corr_pairs.append((row, column, corr_val))

print(f"Found {len(high_corr_pairs)} feature pairs with correlation > 0.9:\n")
if high_corr_pairs:
    for feat1, feat2, corr_val in high_corr_pairs[:10]:  # Show top 10
        print(f"   {feat1:20s} ↔ {feat2:20s}  :  {corr_val:.3f}")
    print(f"\n  IMPACT:")
    print(f"   Redundant information (features are nearly identical)")
    print(f"   Unstable coefficient estimates in linear models")
    print(f"   Inflated feature importance in tree models")
    
    issues_identified.append({
        'Issue': 'Multicollinearity',
        'Severity': 'Medium',
        'Impact': f'{len(high_corr_pairs)} highly correlated pairs',
        'Mitigation': 'Feature selection, PCA, remove redundant features'
    })
else:
    print("   No severe multicollinearity detected (all correlations < 0.9)")

# ISSUE 4: Suboptimal Decision Threshold

print("ISSUE 4: SUBOPTIMAL DECISION THRESHOLD")


# Use best model's OOF predictions
best_model_name = all_results_df.iloc[0]['Model']
best_oof_probs = all_oof_probs[best_model_name]

# Calculate F1 at different thresholds
thresholds = np.linspace(0.1, 0.9, 81)
f1_scores = [f1_score(y, (best_oof_probs >= t).astype(int)) for t in thresholds]
best_threshold_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_threshold_idx]
f1_at_default = f1_score(y, (best_oof_probs >= 0.5).astype(int))
f1_at_optimal = f1_scores[best_threshold_idx]

print(f"Model: {best_model_name}")
print(f"F1 Score at default threshold (0.5):   {f1_at_default:.4f}")
print(f"F1 Score at optimal threshold ({optimal_threshold:.2f}): {f1_at_optimal:.4f}")
print(f"\n  IMPACT:")
print(f"   Missing {((f1_at_optimal - f1_at_default) / f1_at_default * 100):.1f}% potential F1 improvement")
print(f"   Default 0.5 threshold assumes balanced classes and equal costs, for imbalanced data, threshold should be adjusted")

# Plot threshold vs F1
plt.figure(figsize=(10, 5))
plt.plot(thresholds, f1_scores, linewidth=2, label='F1 Score')
plt.axvline(x=0.5, color='red', linestyle='--', label=f'Default (0.5): F1={f1_at_default:.3f}')
plt.axvline(x=optimal_threshold, color='green', linestyle='--', 
            label=f'Optimal ({optimal_threshold:.2f}): F1={f1_at_optimal:.3f}')
plt.xlabel('Classification Threshold', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.title('F1 Score vs Classification Threshold', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

issues_identified.append({
    'Issue': 'Suboptimal Threshold',
    'Severity': 'High',
    'Impact': f'+{((f1_at_optimal - f1_at_default) / f1_at_default * 100):.1f}% F1 gain available',
    'Mitigation': 'Threshold optimization via grid search'
})

# ISSUE 5: Limited Feature Engineering

print("LIMITED FEATURE ENGINEERING")


print(f"Current feature count: {X.shape[1]}")
print(f"\nMissing potential domain-specific features:")

missing_features = []
if 'temperature' in X.columns and 'dewpoint' in X.columns:
    print("   • Dew point depression (temperature - dewpoint)")
    print("     → Key meteorological indicator for precipitation")
    missing_features.append("Dew point depression")

if 'sunshine' in X.columns and 'cloud' in X.columns:
    print("   • Sunshine/Cloud interaction (ratio, product)")
    print("     → Captures inverse relationship between sun and clouds")
    missing_features.append("Sunshine-Cloud interaction")

if 'humidity' in X.columns and 'cloud' in X.columns:
    print("   • Humidity × Cloud product")
    print("     → Combined moisture indicators")
    missing_features.append("Humidity-Cloud interaction")

if 'day' in X.columns:
    print("   • Seasonal indicators (is_summer, is_winter)")
    print("     → Categorical seasonal patterns")
    missing_features.append("Seasonal features")

print(f"\n  IMPACT:")
print(f"   Models cannot capture important non-linear relationships")
print(f"   Domain knowledge not fully utilized")
print(f"   Potential performance gains left on table")

issues_identified.append({
    'Issue': 'Limited Features',
    'Severity': 'High',
    'Impact': f'{len(missing_features)} key feature types missing',
    'Mitigation': 'Domain-specific feature engineering'
})

# ISSUE 6: Feature Importance Analysis

print("FEATURE IMPORTANCE DISTRIBUTION")


# Use Random Forest for feature importance
rf_temp = RandomForestClassifier(n_estimators=200, class_weight='balanced', 
                                  random_state=RANDOM_STATE, n_jobs=-1)
rf_temp.fit(X.fillna(X.median()), y)
feature_importances = pd.Series(rf_temp.feature_importances_, index=X.columns).sort_values(ascending=False)

print("Top 10 most important features:")
for i, (feat, imp) in enumerate(feature_importances.head(10).items(), 1):
    print(f"   {i:2d}. {feat:25s} : {imp:.4f}")

bottom_5_contribution = feature_importances.tail(5).sum()
print(f"\nBottom 5 features contribute only: {bottom_5_contribution:.4f} ({bottom_5_contribution*100:.2f}%)")
print(f"\n  IMPACT:")
print(f"   Low-importance features may add noise")
print(f"   Increased computational cost without benefit")
print(f"   Opportunity for dimensionality reduction")

# Plot feature importance
plt.figure(figsize=(10, 6))
feature_importances.head(15).iloc[::-1].plot(kind='barh', color='steelblue')
plt.xlabel('Importance', fontsize=12)
plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

issues_identified.append({
    'Issue': 'Feature Importance Skew',
    'Severity': 'Low',
    'Impact': f'Bottom 25% features contribute <{feature_importances.tail(len(feature_importances)//4).sum()*100:.1f}%',
    'Mitigation': 'Feature selection, remove low-importance features'
})

# Summary Table

print("SUMMARY")

issues_summary_df = pd.DataFrame(issues_identified)
display(issues_summary_df)





print("PROPOSING AND TESTING IMPROVEMENTS")


improvements_results = []

# Store baseline for comparison
baseline_model_name = all_results_df.iloc[0]['Model']
baseline_model = expanded_models[baseline_model_name]
baseline_roc_auc = all_results_df.iloc[0]['ROC_AUC']
baseline_f1 = all_results_df.iloc[0]['F1 (0.5)']

print(f"\nBaseline Model: {baseline_model_name}")
print(f"Baseline ROC-AUC: {baseline_roc_auc:.4f}")
print(f"Baseline F1:      {baseline_f1:.4f}")


# IMPROVEMENT 1: Threshold Optimization


print("IMPROVEMENT 1: THRESHOLD OPTIMIZATION")

print("Rationale: Default 0.5 threshold is suboptimal for imbalanced data")
print("Method:    Grid search over thresholds [0.1, 0.9] to maximize F1 score")

# Use OOF predictions from baseline model
baseline_oof_probs = all_oof_probs[baseline_model_name]
f1_default = f1_score(y, (baseline_oof_probs >= 0.5).astype(int))

# Find optimal threshold
thresholds = np.linspace(0.1, 0.9, 81)
f1_scores = [f1_score(y, (baseline_oof_probs >= t).astype(int)) for t in thresholds]
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
f1_optimal = f1_scores[optimal_idx]

print(f"\nResults:")
print(f"   Default threshold (0.5): F1 = {f1_default:.4f}")
print(f"   Optimal threshold ({optimal_threshold:.2f}): F1 = {f1_optimal:.4f}")
print(f"   Improvement: +{(f1_optimal - f1_default):.4f} (+{((f1_optimal - f1_default) / f1_default * 100):.1f}%)")

improvements_results.append({
    'Improvement': 'Threshold Optimization',
    'Baseline': f1_default,
    'Improved': f1_optimal,
    'Gain': f1_optimal - f1_default,
    'Gain %': ((f1_optimal - f1_default) / f1_default * 100),
    'Metric': 'F1 Score'
})


# IMPROVEMENT 2: Feature Engineering


print("IMPROVEMENT 2: DOMAIN-SPECIFIC FEATURE ENGINEERING")

print("Rationale: Meteorological relationships not captured by raw features")
print("Method:    Add dew point depression, sunshine-cloud interactions")

# Create enhanced features
X_enhanced = X.copy()
new_features_added = []

if 'temperature' in X.columns and 'dewpoint' in X.columns:
    X_enhanced['dew_depression'] = X['temperature'] - X['dewpoint']
    new_features_added.append('dew_depression')
    print("\n✓ Added: Dew point depression (temp - dewpoint)")

if 'sunshine' in X.columns and 'cloud' in X.columns:
    X_enhanced['sun_cloud_ratio'] = X['sunshine'] / (X['cloud'] + 0.1)
    X_enhanced['sun_cloud_product'] = X['sunshine'] * X['cloud']
    new_features_added.extend(['sun_cloud_ratio', 'sun_cloud_product'])
    print("✓ Added: Sunshine-Cloud ratio and product")

if 'humidity' in X.columns and 'cloud' in X.columns:
    X_enhanced['humidity_cloud'] = X['humidity'] * X['cloud']
    new_features_added.append('humidity_cloud')
    print("✓ Added: Humidity-Cloud interaction")

if 'day' in X.columns:
    X_enhanced['is_summer'] = ((X['day'] >= 172) & (X['day'] <= 264)).astype(int)
    X_enhanced['is_winter'] = ((X['day'] <= 79) | (X['day'] >= 355)).astype(int)
    new_features_added.extend(['is_summer', 'is_winter'])
    print("✓ Added: Seasonal indicators")

# Handle infinities
X_enhanced = X_enhanced.replace([np.inf, -np.inf], np.nan)

print(f"\nFeature count: {X.shape[1]} → {X_enhanced.shape[1]} (+{len(new_features_added)} features)")

# Test with same model architecture
if "Random Forest" in baseline_model_name or "Gradient" in baseline_model_name:
    test_model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight='balanced', 
                                       random_state=RANDOM_STATE, n_jobs=-1))
    ])
else:
    test_model = baseline_model

enhanced_scores = cross_val_score(test_model, X_enhanced, y, cv=skf, scoring='roc_auc', n_jobs=-1)
enhanced_roc_auc = enhanced_scores.mean()

print(f"\nResults:")
print(f"   Baseline ROC-AUC:  {baseline_roc_auc:.4f}")
print(f"   Enhanced ROC-AUC:  {enhanced_roc_auc:.4f}")
print(f"   Improvement: +{(enhanced_roc_auc - baseline_roc_auc):.4f} (+{((enhanced_roc_auc - baseline_roc_auc) / baseline_roc_auc * 100):.1f}%)")

improvements_results.append({
    'Improvement': 'Feature Engineering',
    'Baseline': baseline_roc_auc,
    'Improved': enhanced_roc_auc,
    'Gain': enhanced_roc_auc - baseline_roc_auc,
    'Gain %': ((enhanced_roc_auc - baseline_roc_auc) / baseline_roc_auc * 100),
    'Metric': 'ROC-AUC'
})


# IMPROVEMENT 3: Hyperparameter Tuning


print("IMPROVEMENT 3: HYPERPARAMETER TUNING")

print("Rationale: Default hyperparameters are rarely optimal")
print("Method:    RandomizedSearchCV on Random Forest with 20 iterations")

from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [10, 15, 20, None],
    'clf__min_samples_split': [5, 10, 20],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__max_features': ['sqrt', 'log2', None]
}

tuning_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1))
])

print("\nSearching hyperparameter space... ")
random_search = RandomizedSearchCV(
    tuning_pipeline, 
    param_distributions, 
    n_iter=20, 
    cv=5, 
    scoring='roc_auc',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)
random_search.fit(X_enhanced, y)

tuned_roc_auc = random_search.best_score_

print(f"\nBest hyperparameters found:")
for param, value in random_search.best_params_.items():
    print(f"   {param}: {value}")

print(f"\nResults:")
print(f"   Before tuning: {baseline_roc_auc:.4f}")
print(f"   After tuning:  {tuned_roc_auc:.4f}")
print(f"   Improvement: +{(tuned_roc_auc - baseline_roc_auc):.4f} (+{((tuned_roc_auc - baseline_roc_auc) / baseline_roc_auc * 100):.1f}%)")

improvements_results.append({
    'Improvement': 'Hyperparameter Tuning',
    'Baseline': baseline_roc_auc,
    'Improved': tuned_roc_auc,
    'Gain': tuned_roc_auc - baseline_roc_auc,
    'Gain %': ((tuned_roc_auc - baseline_roc_auc) / baseline_roc_auc * 100),
    'Metric': 'ROC-AUC'
})


# IMPROVEMENT 4: Advanced Ensemble (Voting)


print("IMPROVEMENT 4: VOTING ENSEMBLE")

print("Rationale: Combine diverse models to reduce variance and improve robustness")
print("Method:    Soft voting of Random Forest + Gradient Boosting + Logistic Regression")

voting_ensemble = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("vote", VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, 
                                          class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, 
                                              random_state=RANDOM_STATE)),
            ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', 
                                      random_state=RANDOM_STATE))
        ],
        voting='soft',
        n_jobs=-1
    ))
])

ensemble_scores = cross_val_score(voting_ensemble, X_enhanced, y, cv=skf, scoring='roc_auc', n_jobs=-1)
ensemble_roc_auc = ensemble_scores.mean()

print(f"\nResults:")
print(f"   Best single model: {baseline_roc_auc:.4f}")
print(f"   Voting ensemble:   {ensemble_roc_auc:.4f}")
print(f"   Improvement: +{(ensemble_roc_auc - baseline_roc_auc):.4f} (+{((ensemble_roc_auc - baseline_roc_auc) / baseline_roc_auc * 100):.1f}%)")

improvements_results.append({
    'Improvement': 'Voting Ensemble',
    'Baseline': baseline_roc_auc,
    'Improved': ensemble_roc_auc,
    'Gain': ensemble_roc_auc - baseline_roc_auc,
    'Gain %': ((ensemble_roc_auc - baseline_roc_auc) / baseline_roc_auc * 100),
    'Metric': 'ROC-AUC'
})

# Summary of improvements

print("IMPROVEMENTS SUMMARY")

improvements_df = pd.DataFrame(improvements_results)
display(improvements_df)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Absolute gains
ax1 = axes[0]
colors = ['#e74c3c' if g < 0 else '#2ecc71' for g in improvements_df['Gain']]
ax1.barh(improvements_df['Improvement'], improvements_df['Gain'], color=colors)
ax1.set_xlabel('Absolute Gain', fontsize=12)
ax1.set_title('Improvement Impact (Absolute)', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.axvline(x=0, color='black', linewidth=0.8)
for i, v in enumerate(improvements_df['Gain']):
    ax1.text(v + 0.001, i, f'{v:+.4f}', va='center', fontsize=10)

# Plot 2: Percentage gains
ax2 = axes[1]
colors2 = ['#e74c3c' if g < 0 else '#3498db' for g in improvements_df['Gain %']]
ax2.barh(improvements_df['Improvement'], improvements_df['Gain %'], color=colors2)
ax2.set_xlabel('Percentage Gain (%)', fontsize=12)
ax2.set_title('Improvement Impact (Percentage)', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.axvline(x=0, color='black', linewidth=0.8)
for i, v in enumerate(improvements_df['Gain %']):
    ax2.text(v + 0.2, i, f'{v:+.1f}%', va='center', fontsize=10)

plt.tight_layout()
plt.show()





print("TASK 4: PROGRESSIVE IMPROVEMENT RESULTS")


# Build progressive enhancement pipeline
progressive_results = []

# Step 1: Original Baseline
print("\nStep 1: Baseline Model")
step1_model = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(n_estimators=200, class_weight='balanced', 
                                   random_state=RANDOM_STATE, n_jobs=-1))
])
step1_prob = cross_val_predict(step1_model, X, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
step1_pred_default = (step1_prob >= 0.5).astype(int)

progressive_results.append({
    'Stage': '1. Baseline',
    'Features': X.shape[1],
    'ROC_AUC': roc_auc_score(y, step1_prob),
    'F1_Score': f1_score(y, step1_pred_default),
    'Accuracy': accuracy_score(y, step1_pred_default),
    'Precision': precision_score(y, step1_pred_default),
    'Recall': recall_score(y, step1_pred_default)
})

# Step 2: + Threshold Optimization
print("Step 2: + Threshold Optimization")
step2_pred_optimized = (step1_prob >= optimal_threshold).astype(int)

progressive_results.append({
    'Stage': '2. + Threshold Opt',
    'Features': X.shape[1],
    'ROC_AUC': roc_auc_score(y, step1_prob),
    'F1_Score': f1_score(y, step2_pred_optimized),
    'Accuracy': accuracy_score(y, step2_pred_optimized),
    'Precision': precision_score(y, step2_pred_optimized),
    'Recall': recall_score(y, step2_pred_optimized)
})

# Step 3: + Feature Engineering
print("Step 3: + Feature Engineering")
step3_model = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(n_estimators=200, class_weight='balanced', 
                                   random_state=RANDOM_STATE, n_jobs=-1))
])
step3_prob = cross_val_predict(step3_model, X_enhanced, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
step3_pred = (step3_prob >= optimal_threshold).astype(int)

progressive_results.append({
    'Stage': '3. + Feature Eng',
    'Features': X_enhanced.shape[1],
    'ROC_AUC': roc_auc_score(y, step3_prob),
    'F1_Score': f1_score(y, step3_pred),
    'Accuracy': accuracy_score(y, step3_pred),
    'Precision': precision_score(y, step3_pred),
    'Recall': recall_score(y, step3_pred)
})

# Step 4: + Hyperparameter Tuning
print("Step 4: + Hyperparameter Tuning")
step4_model = random_search.best_estimator_
step4_prob = cross_val_predict(step4_model, X_enhanced, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
step4_pred = (step4_prob >= optimal_threshold).astype(int)

progressive_results.append({
    'Stage': '4. + Hyperparam',
    'Features': X_enhanced.shape[1],
    'ROC_AUC': roc_auc_score(y, step4_prob),
    'F1_Score': f1_score(y, step4_pred),
    'Accuracy': accuracy_score(y, step4_pred),
    'Precision': precision_score(y, step4_pred),
    'Recall': recall_score(y, step4_pred)
})

# Step 5: + Ensemble
print("Step 5: + Voting Ensemble")
step5_prob = cross_val_predict(voting_ensemble, X_enhanced, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
step5_pred = (step5_prob >= optimal_threshold).astype(int)

progressive_results.append({
    'Stage': '5. + Ensemble',
    'Features': X_enhanced.shape[1],
    'ROC_AUC': roc_auc_score(y, step5_prob),
    'F1_Score': f1_score(y, step5_pred),
    'Accuracy': accuracy_score(y, step5_pred),
    'Precision': precision_score(y, step5_pred),
    'Recall': recall_score(y, step5_pred)
})

# Create DataFrame
progressive_df = pd.DataFrame(progressive_results)

# Calculate cumulative gains
progressive_df['ROC_AUC_Gain'] = progressive_df['ROC_AUC'] - progressive_df.iloc[0]['ROC_AUC']
progressive_df['F1_Gain'] = progressive_df['F1_Score'] - progressive_df.iloc[0]['F1_Score']

print("\n" + "="*80)
print("PROGRESSIVE ENHANCEMENT RESULTS")
print("="*80)
display(progressive_df)

# Visualization: Progressive improvement
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: ROC-AUC progression
ax1 = axes[0, 0]
stages = range(len(progressive_df))
ax1.plot(stages, progressive_df['ROC_AUC'], 'o-', linewidth=2.5, markersize=10, color='#3498db')
ax1.set_xticks(stages)
ax1.set_xticklabels([s.split('.')[1].strip() if '.' in s else s for s in progressive_df['Stage']], 
                     rotation=45, ha='right')
ax1.set_ylabel('ROC-AUC Score', fontsize=12)
ax1.set_title('Progressive Improvement: ROC-AUC', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
for i, v in enumerate(progressive_df['ROC_AUC']):
    ax1.text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

# Plot 2: F1 Score progression
ax2 = axes[0, 1]
ax2.plot(stages, progressive_df['F1_Score'], 'o-', linewidth=2.5, markersize=10, color='#2ecc71')
ax2.set_xticks(stages)
ax2.set_xticklabels([s.split('.')[1].strip() if '.' in s else s for s in progressive_df['Stage']], 
                     rotation=45, ha='right')
ax2.set_ylabel('F1 Score', fontsize=12)
ax2.set_title('Progressive Improvement: F1 Score', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
for i, v in enumerate(progressive_df['F1_Score']):
    ax2.text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

# Plot 3: Cumulative gains (ROC-AUC)
ax3 = axes[1, 0]
colors3 = ['#95a5a6' if i == 0 else '#3498db' for i in range(len(progressive_df))]
ax3.bar(stages, progressive_df['ROC_AUC_Gain'], color=colors3, alpha=0.8)
ax3.set_xticks(stages)
ax3.set_xticklabels([s.split('.')[1].strip() if '.' in s else s for s in progressive_df['Stage']], 
                     rotation=45, ha='right')
ax3.set_ylabel('Cumulative Gain', fontsize=12)
ax3.set_title('Cumulative ROC-AUC Gain from Baseline', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(progressive_df['ROC_AUC_Gain']):
    if v != 0:
        ax3.text(i, v + 0.001, f'+{v:.4f}', ha='center', fontsize=10, fontweight='bold')

# Plot 4: Cumulative gains (F1)
ax4 = axes[1, 1]
colors4 = ['#95a5a6' if i == 0 else '#2ecc71' for i in range(len(progressive_df))]
ax4.bar(stages, progressive_df['F1_Gain'], color=colors4, alpha=0.8)
ax4.set_xticks(stages)
ax4.set_xticklabels([s.split('.')[1].strip() if '.' in s else s for s in progressive_df['Stage']], 
                     rotation=45, ha='right')
ax4.set_ylabel('Cumulative Gain', fontsize=12)
ax4.set_title('Cumulative F1 Score Gain from Baseline', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(progressive_df['F1_Gain']):
    if v != 0:
        ax4.text(i, v + 0.002, f'+{v:.4f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('progressive_improvements.png', dpi=300, bbox_inches='tight')
plt.show()

# Final summary statistics

print("FINAL RESULTS SUMMARY")

baseline_metrics = progressive_df.iloc[0]
final_metrics = progressive_df.iloc[-1]

print(f"\nBASELINE (Stage 1):")
print(f"   ROC-AUC:   {baseline_metrics['ROC_AUC']:.4f}")
print(f"   F1 Score:  {baseline_metrics['F1_Score']:.4f}")
print(f"   Accuracy:  {baseline_metrics['Accuracy']:.4f}")
print(f"   Precision: {baseline_metrics['Precision']:.4f}")
print(f"   Recall:    {baseline_metrics['Recall']:.4f}")

print(f"\nFINAL MODEL (Stage 5):")
print(f"   ROC-AUC:   {final_metrics['ROC_AUC']:.4f}  (+{final_metrics['ROC_AUC_Gain']:.4f}, +{(final_metrics['ROC_AUC_Gain']/baseline_metrics['ROC_AUC']*100):.1f}%)")
print(f"   F1 Score:  {final_metrics['F1_Score']:.4f}  (+{final_metrics['F1_Gain']:.4f}, +{(final_metrics['F1_Gain']/baseline_metrics['F1_Score']*100):.1f}%)")
print(f"   Accuracy:  {final_metrics['Accuracy']:.4f}  (+{(final_metrics['Accuracy']-baseline_metrics['Accuracy']):.4f})")
print(f"   Precision: {final_metrics['Precision']:.4f}  (+{(final_metrics['Precision']-baseline_metrics['Precision']):.4f})")
print(f"   Recall:    {final_metrics['Recall']:.4f}  (+{(final_metrics['Recall']-baseline_metrics['Recall']):.4f})")


print("TOTAL IMPROVEMENT ACHIEVED:")
print(f"   ROC-AUC improved by {(final_metrics['ROC_AUC_Gain']/baseline_metrics['ROC_AUC']*100):.1f}%")
print(f"   F1 Score improved by {(final_metrics['F1_Gain']/baseline_metrics['F1_Score']*100):.1f}%")




