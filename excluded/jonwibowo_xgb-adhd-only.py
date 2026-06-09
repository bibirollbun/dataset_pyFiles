!pip -q install neuroHarmonize neuroCombat scikit-learn 


#data wrangling
import pandas as pd 
import numpy as np 
import re
from tqdm import tqdm
import optuna

#viz
import seaborn as sns
import matplotlib.pyplot as plt

#connectome 
from neuroHarmonize import harmonizationLearn

#ML 
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split, cross_validate, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, make_scorer, brier_score_loss
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

#stats 
from scipy.stats import kstest, ttest_ind, ks_2samp, mannwhitneyu, mode


# train data 
train_cat = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx")
train_num = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx")
train_conn = pd.read_csv("/kaggle/input/widsdatathon2025/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv")

# test data
test_cat = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")
test_num = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
test_conn = pd.read_csv("/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")

# solutions 
y = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx")
y.drop(columns = 'participant_id', inplace = True)


# merging the two datasets 
combined_cat = pd.concat([train_cat, test_cat], axis = 0)
combined_num = pd.concat([train_num, test_num], axis = 0)
combined_conn = pd.concat([train_conn, test_conn], axis = 0)


## CATEGORICAL DATA ##


sns.histplot(train_cat['Basic_Demos_Study_Site'], color = 'blue', label = 'Train', alpha = 0.5, binwidth = 1)
sns.histplot(test_cat['Basic_Demos_Study_Site'], color = 'orange', label = 'Test', alpha = 0.5, binwidth = 1)

plt.legend()
plt.xlabel('Basic Demos Study Site')
plt.ylabel('Count')
plt.title('Lack of Overlap in Train vs. Test Distribution of Study Site')
plt.show()


sns.histplot(combined_cat['Barratt_Barratt_P1_Occ'], binwidth = 1)
sns.histplot(combined_cat['Barratt_Barratt_P2_Occ'], binwidth = 1)


# Creating a composite score of social status based on available parent data
def calculate_social_status(row):
    # Check if data for parent 1 is available
    p1_available = pd.notna(row['Barratt_Barratt_P1_Edu']) and pd.notna(row['Barratt_Barratt_P1_Occ'])
    p1_score = row['Barratt_Barratt_P1_Edu'] + row['Barratt_Barratt_P1_Occ'] if p1_available else None
    
    # Check if data for parent 2 is available
    p2_available = pd.notna(row['Barratt_Barratt_P2_Edu']) and pd.notna(row['Barratt_Barratt_P2_Occ'])
    p2_score = row['Barratt_Barratt_P2_Edu'] + row['Barratt_Barratt_P2_Occ'] if p2_available else None
    
    # Calculate score based on available data
    if p1_available and p2_available:
        # Both parents' data available - average them
        return (p1_score + p2_score) / 2
    elif p1_available:
        # Only parent 1's data available
        return p1_score
    elif p2_available:
        # Only parent 2's data available
        return p2_score
    else:
        # No data available for either parent
        return np.nan

# Apply the function to create the social status score
combined_cat['social_status_score'] = combined_cat.apply(calculate_social_status, axis=1)

# Plot the distribution
sns.histplot(combined_cat['social_status_score'].dropna())


# missing indicator for parental 2 occupation, wonder if it's because it's a single parent household 
combined_cat['missing_p2_occ'] = combined_cat['Barratt_Barratt_P2_Occ'].isna().astype(int)


#drop participant id 
df = combined_conn.drop(columns = 'participant_id')

#store original col names 
original_column_names = df.columns.tolist()

#convert df to numpy array
con_array = df.values

# create dataframe and store the covariate
covars = pd.DataFrame()
covars['SITE'] = combined_cat['MRI_Track_Scan_Location']

#run harmonization
my_model, my_data_adj = harmonizationLearn(con_array, covars)

#get old dataframe
combined_conn = pd.DataFrame(my_data_adj, columns = original_column_names)


# Extract row and column indices from column names once
pattern = r'(\d+)throw_(\d+)thcolumn'
indices = []
rows = []
cols = []
col_names = []

for col in combined_conn.columns:
    if col != 'participant_id':
        match = re.match(pattern, col)
        if match:
            row, col_idx = int(match.group(1)), int(match.group(2))
            rows.append(row)
            cols.append(col_idx)
            col_names.append(col)

# Convert to participant ID index if available
if 'participant_id' in combined_conn.columns:
    result_df = pd.DataFrame(index=con_df['participant_id'])
else:
    result_df = pd.DataFrame(index=range(len(combined_conn)))

# Create a 3D matrix (participants × 200 × 200)
all_matrices = np.zeros((len(combined_conn), 200, 200))

# Fill the matrices in one vectorized operation
values = combined_conn[col_names].values  # Extract all values at once
for i, (r, c) in enumerate(zip(rows, cols)):
    all_matrices[:, r, c] = values[:, i]

# Make all matrices symmetric
for i in range(len(combined_conn)):
    all_matrices[i] = all_matrices[i] + all_matrices[i].T - np.diag(np.diag(all_matrices[i]))

# Compute connection strengths for all participants at once
connection_strengths = all_matrices.sum(axis=1)

# Convert to DataFrame
column_names = [f"connection_strength_{i+1}" for i in range(200)]
connection_strength_df = pd.DataFrame(connection_strengths, index=result_df.index, columns=column_names)

print(connection_strength_df.head())


# combining the data for modelling 
combined_num.reset_index(drop = True, inplace = True)
combined_cat.reset_index(drop = True, inplace = True)
connection_strength_df.reset_index(drop = True, inplace = True)


le = LabelEncoder()
combined_cat['PreInt_Demos_Fam_Child_Ethnicity'] = le.fit_transform(combined_cat['PreInt_Demos_Fam_Child_Ethnicity'])
combined_cat['PreInt_Demos_Fam_Child_Race'] = le.fit_transform(combined_cat['PreInt_Demos_Fam_Child_Race'])


# dropping unwanted columns
combined_cat.drop(columns = ["participant_id", "Basic_Demos_Enroll_Year", "Basic_Demos_Study_Site", "MRI_Track_Scan_Location", 'Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Edu', 'Barratt_Barratt_P2_Occ', 'PreInt_Demos_Fam_Child_Ethnicity', 'PreInt_Demos_Fam_Child_Race'], inplace = True)


combined_num.drop(columns = ['participant_id'], inplace = True)


# combining the 3 datasets 
all_feat = pd.concat([combined_cat, combined_num, connection_strength_df], axis = 1)


# split to Kaggle's train and test 
train = all_feat.iloc[:len(train_num), :] # train set
test = all_feat.iloc[len(train_num):, :] # test set 


best_params = {'booster': 'dart',
 'lambda': 0.2501250080557255,
 'alpha': 2.7486508427405283e-07,
 'subsample': 0.6845636891960641,
 'colsample_bytree': 0.3627904953672119,
 'n_estimators': 398,
 'max_depth': 3,
 'min_child_weight': 7,
 'eta': 0.03282706635280254,
 'gamma': 0.019343805913819682,
 'grow_policy': 'depthwise',
 'sample_type': 'weighted',
 'normalize_type': 'forest',
 'rate_drop': 0.0018972596578811524,
 'skip_drop': 0.0002610489338482561}


labels = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx").set_index("participant_id")
train.set_index(labels.index, drop = True, inplace = True)

y_sex = y['Sex_F']
y_adhd = y['ADHD_Outcome']

features = train.columns
features_adhd = train.columns
feature_sex = train.drop(columns = ['social_status_score', 'missing_p2_occ']).columns


# h/t to Lennard Haupts for this portion of the code

SEED = 42
REPEATS = 5
FOLDS = 5


assert all(train.index == labels.index), "Label IDs don't match train IDs"

combinations = labels["ADHD_Outcome"].astype(str) + labels["Sex_F"].astype(str)

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
t_sex = 0.2
t_adhd = 0.35

# Repeated Stratified K-Fold
rskf = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)
# skf for LogisticRegressionCV
skf = StratifiedKFold(n_splits=FOLDS)

params_1 =  {'booster': 'dart',
    'lambda': 9.684264461272669e-06,
    'alpha': 1.4088489370221833e-06,
    'subsample': 0.8016161406674585,
    'colsample_bytree': 0.8120047856579565,
    'n_estimators': 226,
    'max_depth': 7,
    'min_child_weight': 9,
    'eta': 0.013292290783237888,
    'gamma': 0.9646906380999847,
    'grow_policy': 'depthwise',
    'sample_type': 'weighted',
    'normalize_type': 'forest',
    'rate_drop': 0.3911369013366205,
    'skip_drop': 1.1975595482978298e-07}

params_2 = {'booster': 'dart',
 'lambda': 0.2501250080557255,
 'alpha': 2.7486508427405283e-07,
 'subsample': 0.6845636891960641,
 'colsample_bytree': 0.3627904953672119,
 'n_estimators': 398,
 'max_depth': 3,
 'min_child_weight': 7,
 'eta': 0.03282706635280254,
 'gamma': 0.019343805913819682,
 'grow_policy': 'depthwise',
 'sample_type': 'weighted',
 'normalize_type': 'forest',
 'rate_drop': 0.0018972596578811524,
 'skip_drop': 0.0002610489338482561}

model_1 = XGBClassifier(
    tree_method = 'exact', 
    **params_1,
    random_state=42
)
model_2 = XGBClassifier(
    tree_method = 'exact', 
    **params_2,
    random_state=42
)

for fold, (train_idx, val_idx) in enumerate(rskf.split(train, combinations), 1):
    print(f"\n=== Fold {fold} ===")

    # Split data
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train_adhd, y_val_adhd = y_adhd.iloc[train_idx], y_adhd.iloc[val_idx]
    y_train_sex, y_val_sex = y_sex.iloc[train_idx], y_sex.iloc[val_idx]
    # 2x weight for Sex_F == 1 and ADHD_Outcome == 1 (as mentioned in competition evaluation)
    weights_train = np.where(combinations.iloc[train_idx]=="11", 2, 1)
    weights = np.where(combinations.iloc[val_idx]=="11", 2, 1)

    # ----------------
    # Sex_F prediction
    # ----------------
    # Model 1
    model_1.fit(X_train[feature_sex], y_train_sex, sample_weight=weights_train)
    sex_train = model_1.predict_proba(X_train[feature_sex])[:, 1]
    sex_val = model_1.predict_proba(X_val[feature_sex])[:, 1]
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
    # for interaction in interactions:
    #     X_train[f"I_{interaction}"] = X_train[interaction] * X_train["sex_proba"]
    #     X_val[f"I_{interaction}"] = X_val[interaction] * X_val["sex_proba"]

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


def objective(trial):
    # Common parameters for both models
    params_base = {
        'booster': trial.suggest_categorical('booster', ['dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 1, 7),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise']),
        'sample_type': trial.suggest_categorical('sample_type', ['weighted']),
        'normalize_type': trial.suggest_categorical('normalize_type', ['forest']),
        'rate_drop': trial.suggest_float('rate_drop', 1e-8, 1.0, log=True),
        'skip_drop': trial.suggest_float('skip_drop', 1e-8, 1.0, log=True)
    }
    
    # If you want different parameters for each model, you can modify this approach
    params_1 = params_base.copy()
    #params_2 = params_base.copy()
    
    # Definition of fixed parameters from your code
    SEED = 42
    FOLDS = 5  # For quick optimization, you might want to use fewer folds
    REPEATS = 2  # Similarly, fewer repeats for optimization
    
    # Thresholds can also be optimized
    t_sex = trial.suggest_float('t_sex', 0.1, 0.5)
    #t_adhd = trial.suggest_float('t_adhd', 0.1, 0.5)
    
    # Store scores
    scores_sex = []
    #scores_adhd = []
    
    # Out of fold predictions
    sex_oof = np.zeros(len(y_sex))
    #adhd_oof = np.zeros(len(y_adhd))
    
    # Cross-validation setup
    rskf = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)
    
    # Models initialization
    model_1 = XGBClassifier(tree_method='exact', **params_1, random_state=SEED)
    #model_2 = XGBClassifier(tree_method='exact', **params_2, random_state=SEED)
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(rskf.split(train, combinations), 1):
        # Split data
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train_adhd, y_val_adhd = y_adhd.iloc[train_idx], y_adhd.iloc[val_idx]
        y_train_sex, y_val_sex = y_sex.iloc[train_idx], y_sex.iloc[val_idx]
        
        # Weights for instances (2x for Sex_F==1 and ADHD_Outcome==1)
        weights_train = np.where(combinations.iloc[train_idx]=="11", 2, 1)
        weights = np.where(combinations.iloc[val_idx]=="11", 2, 1)
        
        # Sex_F prediction
        model_1.fit(X_train[feature_sex], y_train_sex, sample_weight=weights_train)
        sex_train = model_1.predict_proba(X_train[feature_sex])[:, 1]
        sex_val = model_1.predict_proba(X_val[feature_sex])[:, 1]
        sex_oof[val_idx] += sex_val / REPEATS
        
        sex_brier = brier_score_loss(y_val_sex, sex_val)
        sex_f1 = f1_score(y_val_sex, (sex_val > t_sex).astype(int), sample_weight=weights)
        scores_sex.append((sex_brier, sex_f1))
        
        # # ADHD_Outcome prediction
        # X_train_copy = X_train.copy()
        # X_val_copy = X_val.copy()
        
        # X_train_copy["sex_proba"] = sex_train
        # X_val_copy["sex_proba"] = sex_val
        
        # # Optionally add interaction features here if needed
        
        # model_2.fit(X_train_copy[features_adhd], y_train_adhd, sample_weight=weights_train)
        # adhd_val = model_2.predict_proba(X_val_copy[features_adhd])[:, 1]
        # adhd_oof[val_idx] += adhd_val / REPEATS
        
        # adhd_brier = brier_score_loss(y_val_adhd, adhd_val)
        # adhd_f1 = f1_score(y_val_adhd, (adhd_val > t_adhd).astype(int), sample_weight=weights)
        # scores_adhd.append((adhd_brier, adhd_f1))
    
    # Calculate mean scores
    mean_sex_brier = np.mean([s[0] for s in scores_sex])
    mean_sex_f1 = np.mean([s[1] for s in scores_sex])
    # mean_adhd_brier = np.mean([s[0] for s in scores_adhd])
    # mean_adhd_f1 = np.mean([s[1] for s in scores_adhd])
    
    # You can use a combined metric as objective
    # For example, weighted sum of Brier scores (lower is better)
    score = mean_sex_f1 - mean_sex_brier
    
    # Log metrics for this trial
    trial.set_user_attr('sex_brier', mean_sex_brier)
    trial.set_user_attr('sex_f1', mean_sex_f1)
    # trial.set_user_attr('adhd_brier', mean_adhd_brier)
    # trial.set_user_attr('adhd_f1', mean_adhd_f1)
    
    return score  # Lower is better

# Create and run study
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

# # Print best parameters
# print("Best trial:")
# trial = study.best_trial
# print(f"  Value: {trial.value}")
# print("  Params: ")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")

# # Print best metrics
# print(f"  Sex Brier: {trial.user_attrs['sex_brier']}")
# print(f"  Sex F1: {trial.user_attrs['sex_f1']}")
# # print(f"  ADHD Brier: {trial.user_attrs['adhd_brier']}")
# # print(f"  ADHD F1: {trial.user_attrs['adhd_f1']}")


weights = np.where(combinations.loc[train.index]=="11", 2, 1)
model_1.fit(train[feature_sex], y_sex, sample_weight=weights)

sex_proba_train = model_1.predict_proba(train[feature_sex])[:,1]
sex_proba_test = model_1.predict_proba(test[feature_sex])[:,1]

train["sex_proba"] = sex_proba_train
test["sex_proba"] = sex_proba_test

features_adhd = train.columns

model_2.fit(train[features_adhd], y_adhd, sample_weight=weights)
adhd_proba_test = model_2.predict_proba(test[features_adhd])[:,1]


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
#submission["Sex_F"] = 1
# Compare share of predicted labels at thresholds between OOF and Test
print(f"Share ADHD OOF: {np.mean(np.where(adhd_oof > best_adhd_threshold, 1, 0)):.4f} - Share ADHD Test: {submission.ADHD_Outcome.mean():.4f}")
print(f"Share Sex_F OOF: {np.mean(np.where(sex_oof > best_sex_threshold, 1, 0)):.4f} - Share Sex_F Test: {submission.Sex_F.mean():.4f}")


print(submission['ADHD_Outcome'].value_counts())
print(submission['Sex_F'].value_counts())


submission.to_csv('submission.csv', index = False)

