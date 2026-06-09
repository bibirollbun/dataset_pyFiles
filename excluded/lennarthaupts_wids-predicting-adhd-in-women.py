import numpy as np 
import pandas as pd 
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LassoCV
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import scipy
from scipy.stats import kstest, ttest_ind, ks_2samp, mannwhitneyu, mode
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


SEED = 42
REPEATS = 5
FOLDS = 5

train_q = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx")
train_c = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx")
test_q = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
test_c = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")

train_combined = pd.merge(train_q, train_c, on="participant_id", how="left").set_index("participant_id")
test_combined = pd.merge(test_q, test_c, on="participant_id", how="left").set_index("participant_id")

labels = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx").set_index("participant_id")
assert all(train_combined.index == labels.index), "Label IDs don't match train IDs"


# Drop columns 
drop_cols = [
    "Basic_Demos_Study_Site", "MRI_Track_Scan_Location", "PreInt_Demos_Fam_Child_Ethnicity",
    "PreInt_Demos_Fam_Child_Race", 'Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Occ'
]
train_combined.drop(drop_cols, axis=1, inplace=True)
test_combined.drop(drop_cols, axis=1, inplace=True)

# Standardize features
scaler = StandardScaler()
train_combined = pd.DataFrame(
    scaler.fit_transform(train_combined), columns=train_combined.columns, index=train_combined.index
)
test_combined = pd.DataFrame(
    scaler.transform(test_combined), columns=test_combined.columns, index=test_combined.index
)

# Impute missing values using IterativeImputer with Lasso
imputer = IterativeImputer(estimator=LassoCV(random_state=SEED), max_iter=5, random_state=SEED)
train_combined[:] = imputer.fit_transform(train_combined)
test_combined[:] = imputer.transform(test_combined)

# Retrieve targets
y_adhd = labels["ADHD_Outcome"]
y_sex = labels["Sex_F"]
# ADHD&Sex Combinations to stratify on
combinations = labels["ADHD_Outcome"].astype(str) + labels["Sex_F"].astype(str)


features_sex = [
       'EHQ_EHQ_Total', 'ColorVision_CV_Score', 'APQ_P_APQ_P_CP',
       'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_INV', 'APQ_P_APQ_P_OPD',
       'APQ_P_APQ_P_PM', 'APQ_P_APQ_P_PP', 'SDQ_SDQ_Conduct_Problems',
       'SDQ_SDQ_Difficulties_Total', 'SDQ_SDQ_Emotional_Problems',
       'SDQ_SDQ_Externalizing', 'SDQ_SDQ_Generating_Impact',
       'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Internalizing',
       'SDQ_SDQ_Peer_Problems', 'SDQ_SDQ_Prosocial', 'MRI_Track_Age_at_Scan',
       'Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P2_Edu'
]

features_adhd = [
       'EHQ_EHQ_Total', 'ColorVision_CV_Score', 'APQ_P_APQ_P_CP',
       'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_INV', 'APQ_P_APQ_P_OPD',
       'APQ_P_APQ_P_PM', 'APQ_P_APQ_P_PP', 'SDQ_SDQ_Conduct_Problems',
       'SDQ_SDQ_Difficulties_Total', 'SDQ_SDQ_Emotional_Problems',
       'SDQ_SDQ_Externalizing', 'SDQ_SDQ_Generating_Impact',
       'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Internalizing',
       'SDQ_SDQ_Peer_Problems', 'SDQ_SDQ_Prosocial', 'MRI_Track_Age_at_Scan',
       'Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P2_Edu', 'sex_proba',
       'I_APQ_P_APQ_P_INV', 'I_APQ_P_APQ_P_PP', 'I_SDQ_SDQ_Hyperactivity',
       'I_MRI_Track_Age_at_Scan', 'I_SDQ_SDQ_Generating_Impact'
]

# Features to be interacted with predicted probability of Sex_F = 1
interactions = [
    "APQ_P_APQ_P_INV", "APQ_P_APQ_P_PP", "SDQ_SDQ_Hyperactivity", 
    "MRI_Track_Age_at_Scan", "SDQ_SDQ_Generating_Impact"
]


def eval_metrics(y_true, y_pred, weights, label="None", thresh=0.5):
    """Evaluate predictions using Brier Score and F1 Score."""
    brier = brier_score_loss(y_true, y_pred)
    f1 = f1_score(y_true, (y_pred > thresh).astype(int), sample_weight=weights)
    print(f"{label} -> Brier Score: {brier:.4f}, F1: {f1:.4f}")
    return brier, f1

# store oof brier and f1
scores_sex = []
scores_adhd = []

# store oof predictions for diagnostics and threshold optimization
sex_oof = np.zeros(len(y_sex))
adhd_oof = np.zeros(len(y_adhd))

# classification thresholds
t_sex = 0.3
t_adhd = 0.4

# Repeated Stratified K-Fold
rskf = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)
# skf for LogisticRegressionCV
skf = StratifiedKFold(n_splits=FOLDS)

params_1 = {
    "penalty":"l1", 
    "Cs": 10, 
    "cv":skf, 
    "fit_intercept":True, 
    "scoring": "f1", 
    "random_state": SEED, 
    "solver": "saga"
}

params_2 = {
    "penalty":"l1", 
    "Cs": 10, 
    "cv":skf, 
    "fit_intercept":True, 
    "scoring": "f1", 
    "random_state": SEED, 
    "solver": "saga"
}

model_1 = LogisticRegressionCV(**params_1)
model_2 = LogisticRegressionCV(**params_2)

for fold, (train_idx, val_idx) in enumerate(rskf.split(train_combined, combinations), 1):
    print(f"\n=== Fold {fold} ===")

    # Split data
    X_train, X_val = train_combined.iloc[train_idx], train_combined.iloc[val_idx]
    y_train_adhd, y_val_adhd = y_adhd.iloc[train_idx], y_adhd.iloc[val_idx]
    y_train_sex, y_val_sex = y_sex.iloc[train_idx], y_sex.iloc[val_idx]
    # 2x weight for Sex_F == 1 and ADHD_Outcome == 1 (as mentioned in competition evaluation)
    weights_train = np.where(combinations.iloc[train_idx]=="11", 2, 1)
    weights = np.where(combinations.iloc[val_idx]=="11", 2, 1)

    # ----------------
    # Sex_F prediction
    # ----------------
    # Model 1
    model_1.fit(X_train[features_sex], y_train_sex, sample_weight=weights_train)
    sex_train = model_1.predict_proba(X_train[features_sex])[:, 1]
    sex_val = model_1.predict_proba(X_val[features_sex])[:, 1]
    sex_oof[val_idx] += sex_val / REPEATS

    sex_brier, sex_f1 = eval_metrics(y_val_sex, sex_val, weights, "Sex_F", thresh=t_sex)
    scores_sex.append((sex_brier, sex_f1))

    # ----------------
    # Outcome_ADHD prediction
    # ----------------
    # Add predicted proba from previous model
    X_train["sex_proba"] = sex_train
    X_val["sex_proba"] = sex_val

    # adding interactions between predicted sex and other features
    for interaction in interactions:
        X_train[f"I_{interaction}"] = X_train[interaction] * X_train["sex_proba"]
        X_val[f"I_{interaction}"] = X_val[interaction] * X_val["sex_proba"]

    # Logistic Regression with L1 penalty
    model_2.fit(X_train[features_adhd], y_train_adhd, sample_weight=weights_train)
    
    adhd_val = model_2.predict_proba(X_val[features_adhd])[:, 1]
    adhd_oof[val_idx] += adhd_val / REPEATS
    
    adhd_brier, adhd_f1 = eval_metrics(y_val_adhd, adhd_val, weights, "Outcome ADHD", thresh=t_adhd)
    scores_adhd.append((adhd_brier, adhd_f1))

print(f"\n=== CV Results ===")
print(f"Sex Mean Brier Score: {np.mean([s[0] for s in scores_sex]):.4f}")
print(f"Sex Mean F1: {np.mean([s[1] for s in scores_sex]):.4f}")
print(f"ADHD Mean Brier Score: {np.mean([s[0] for s in scores_adhd]):.4f}")
print(f"ADHD Mean F1: {np.mean([s[1] for s in scores_adhd]):.4f}")


weights = ((y_adhd == 1) & (y_sex == 1)) + 1
# Compute F1 scores and find the best threshold for sex_oof
thresholds = np.linspace(0, 1, 100)
sex_scores = []
for t in tqdm(thresholds, desc="Sex Thresholds"):
    tmp_pred = np.where(sex_oof > t, 1, 0)
    tmp_score = f1_score(y_sex, tmp_pred, sample_weight=weights)
    sex_scores.append(tmp_score)
best_sex_threshold = thresholds[np.argmax(sex_scores)]
best_sex_score = max(sex_scores)

# Compute F1 scores and find the best threshold for adhd_oof
adhd_scores = []
for t in tqdm(thresholds, desc="ADHD Thresholds"):
    tmp_pred = np.where(adhd_oof > t, 1, 0)
    tmp_score = f1_score(y_adhd, tmp_pred, sample_weight=weights)
    adhd_scores.append(tmp_score)
best_adhd_threshold = thresholds[np.argmax(adhd_scores)]
best_adhd_score = max(adhd_scores)

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

# Plot F1 scores for sex_oof
axs[0, 0].plot(thresholds, sex_scores, label='F1 Score', color='blue')
axs[0, 0].scatter(best_sex_threshold, best_sex_score, color='red', label=f'Best: {best_sex_score:.3f} (Threshold: {best_sex_threshold:.2f})')
axs[0, 0].set_title('F1 Scores vs Thresholds (Sex)')
axs[0, 0].set_xlabel('Threshold')
axs[0, 0].set_ylabel('F1 Score')
axs[0, 0].legend()

# Plot histogram of sex_oof
axs[0, 1].hist(sex_oof, bins=30, color='skyblue', edgecolor='black')
axs[0, 1].set_title('Distribution of sex_oof')
axs[0, 1].set_xlabel('Probability')
axs[0, 1].set_ylabel('Frequency')

# Plot F1 scores for adhd_oof
axs[1, 0].plot(thresholds, adhd_scores, label='F1 Score', color='orange')
axs[1, 0].scatter(best_adhd_threshold, best_adhd_score, color='red', label=f'Best: {best_adhd_score:.3f} (Threshold: {best_adhd_threshold:.2f})')
axs[1, 0].set_title('F1 Scores vs Thresholds (ADHD)')
axs[1, 0].set_xlabel('Threshold')
axs[1, 0].set_ylabel('F1 Score')
axs[1, 0].legend()

# Plot histogram of adhd_oof
axs[1, 1].hist(adhd_oof, bins=30, color='lightgreen', edgecolor='black')
axs[1, 1].set_title('Distribution of adhd_oof')
axs[1, 1].set_xlabel('Probability')
axs[1, 1].set_ylabel('Frequency')

plt.suptitle('Threshold Analysis and Distributions', fontsize=16)
plt.show()


# Final models and predictions
model_1.fit(train_combined[features_sex], y_sex, sample_weight=weights)

sex_proba_train = model_1.predict_proba(train_combined[features_sex])[:,1]
sex_proba_test = model_1.predict_proba(test_combined[features_sex])[:,1]

train_combined["sex_proba"] = sex_proba_train
test_combined["sex_proba"] = sex_proba_test

for interaction in interactions:
    train_combined[f"I_{interaction}"] = train_combined["sex_proba"] * train_combined[interaction]
    test_combined[f"I_{interaction}"] = test_combined["sex_proba"] * test_combined[interaction]

model_2.fit(train_combined[features_adhd], y_adhd, sample_weight=weights)

adhd_proba_test = model_2.predict_proba(test_combined[features_adhd])[:,1]
# Show most important features for model 2 
coeffs_2 = pd.DataFrame({"feature": features_adhd, "coeff": model_2.coef_[0]})
coeffs_2.sort_values(by="coeff", key=abs, ascending=False)[:15]


# Plotting distributions with improved visuals
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot for Sex predictions
ax[0].hist(sex_proba_test, bins=10, alpha=0.5, color='blue', label='Sex Test')
ax[0].hist(sex_oof, bins=10, alpha=0.5, color='orange', label='Sex OOF')
ax[0].set_title('Sex Predictions Distribution')
ax[0].set_xlabel('Predicted Probability')
ax[0].set_ylabel('Frequency')
ax[0].legend()

# Plot for ADHD predictions
ax[1].hist(adhd_proba_test, bins=10, alpha=0.5, color='green', label='ADHD Test')
ax[1].hist(adhd_oof, bins=10, alpha=0.5, color='red', label='ADHD OOF')
ax[1].set_title('ADHD Predictions Distribution')
ax[1].set_xlabel('Predicted Probability')
ax[1].set_ylabel('Frequency')
ax[1].legend()

plt.tight_layout()
plt.show()

# Statistical test to compare distributions
sex_test_result = ks_2samp(sex_proba_test, sex_oof)
adhd_test_result = ks_2samp(adhd_proba_test, adhd_oof)
sex_mwu_result = mannwhitneyu(sex_proba_test, sex_oof)
adhd_mwu_result = mannwhitneyu(adhd_proba_test, adhd_oof)

print("Kolmogorov-Smirnov Test and MannWhitneyU Results:")
print(f"Sex KS Test vs. OOF: Statistic={sex_test_result.statistic:.4f}, p-value={sex_test_result.pvalue:.4f}")
print(f"Sex MWU Test vs. OOF: Statistic={sex_mwu_result.statistic:.4f}, p-value={sex_mwu_result.pvalue:.4f}")
print(f"ADHD KS Test vs. OOF: Statistic={adhd_test_result.statistic:.4f}, p-value={adhd_test_result.pvalue:.4f}")
print(f"ADHD MWU Test vs. OOF: Statistic={adhd_mwu_result.statistic:.4f}, p-value={adhd_mwu_result.pvalue:.4f}")

# Submission
submission = pd.read_excel("/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx")
submission["ADHD_Outcome"] = np.where(adhd_proba_test > best_adhd_threshold, 1, 0)
submission["Sex_F"] = np.where(sex_proba_test > best_sex_threshold, 1, 0)
# Compare share of predicted labels at thresholds between OOF and Test
print(f"Share ADHD OOF: {np.mean(np.where(adhd_oof > best_adhd_threshold, 1, 0)):.4f} - Share ADHD Test: {submission.ADHD_Outcome.mean():.4f}")
print(f"Share Sex_F OOF: {np.mean(np.where(sex_oof > best_sex_threshold, 1, 0)):.4f} - Share Sex_F Test: {submission.Sex_F.mean():.4f}")


submission.to_csv("submission.csv", index=False)




