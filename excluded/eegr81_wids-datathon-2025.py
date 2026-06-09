# I specifically would like to thank these two notebooks as well!
# https://www.kaggle.com/code/thiagomantuani/wids-2025-baseline
# https://www.kaggle.com/code/lennarthaupts/wids-predicting-adhd-in-women


import numpy as np 
import pandas as pd 
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LassoCV
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')
import scipy
from scipy.stats import kstest, ttest_ind, ks_2samp, mannwhitneyu, mode
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time


train_quantitative = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
train_categorical = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
test_quantitative = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
test_categorical = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")

train_combined = pd.merge(train_quantitative, train_categorical, on="participant_id", how="left").set_index("participant_id")
test_combined = pd.merge(test_quantitative, test_categorical, on="participant_id", how="left").set_index("participant_id")

train_solutions = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx").set_index("participant_id")

train_combined = train_combined.sort_values(by=['participant_id'])
train_solutions = train_solutions.sort_values(by=['participant_id'])

assert all(train_combined.index == train_solutions.index), "Label IDs don't match train IDs"



# Drop columns 
drop_cols = [
    "Basic_Demos_Study_Site", "MRI_Track_Scan_Location", "PreInt_Demos_Fam_Child_Ethnicity",
    "PreInt_Demos_Fam_Child_Race", 'Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Occ'
]
train_combined.drop(drop_cols, axis=1, inplace=True, errors = 'ignore' )
test_combined.drop(drop_cols, axis=1, inplace=True, errors = 'ignore')

# Columns in both test and train data
common_columns = list(set(test_combined.columns) & set(train_combined.columns))

# Re-align test and train datasets
test_combined = test_combined[common_columns]
train_combined = train_combined[common_columns]

test_combined = test_combined[train_combined.columns]

# Standardize features
scaler = StandardScaler()
train_combined = pd.DataFrame(
    scaler.fit_transform(train_combined), columns=train_combined.columns, index=train_combined.index
)
test_combined = pd.DataFrame(
    scaler.transform(test_combined), columns=test_combined.columns, index=test_combined.index
)


SEED = 42
REPEATS = 5
FOLDS = 5

# Replace missing values using IterativeImputer with Lasso
imputer = IterativeImputer(estimator=LassoCV(random_state=SEED), max_iter=5, random_state=SEED)
train_combined[:] = imputer.fit_transform(train_combined)
test_combined[:] = imputer.transform(test_combined)



# Get targets
y_adhd = train_solutions["ADHD_Outcome"]
y_sex = train_solutions["Sex_F"]
# ADHD & Sex Combinations to stratify on
combinations = train_solutions["ADHD_Outcome"].astype(str) + train_solutions["Sex_F"].astype(str)


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

# Feature Engineering for f_prob
features_to_interact_sex = [
    'SDQ_SDQ_Prosocial', 'SDQ_SDQ_Emotional_Problems', 'ColorVision_CV_Score', 
    'SDQ_SDQ_Externalizing', 'SDQ_SDQ_Hyperactivity' 
]

# Create basic interaction terms
for i in range(len(features_to_interact_sex)):
    for j in range(i + 1, len(features_to_interact_sex)):
        feature1 = features_to_interact_sex[i]
        feature2 = features_to_interact_sex[j]
        interaction_name = f'{feature1}_x_{feature2}'

        train_combined[interaction_name] = train_combined[feature1] * train_combined[feature2]
        test_combined[interaction_name] = test_combined[feature1] * test_combined[feature2]
        

# Update features_sex to include interaction terms
features_sex_extended = features_sex + list(train_combined.columns[train_combined.columns.str.startswith(tuple(features_to_interact_sex)) & train_combined.columns.str.contains('_x_')])

features_adhd = [
       'EHQ_EHQ_Total', 'ColorVision_CV_Score', 'APQ_P_APQ_P_CP',
       'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_INV', 'APQ_P_APQ_P_OPD',
       'APQ_P_APQ_P_PM', 'APQ_P_APQ_P_PP', 'SDQ_SDQ_Conduct_Problems',
       'SDQ_SDQ_Difficulties_Total', 'SDQ_SDQ_Emotional_Problems',
       'SDQ_SDQ_Externalizing', 'SDQ_SDQ_Generating_Impact',
       'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Internalizing',
       'SDQ_SDQ_Peer_Problems', 'SDQ_SDQ_Prosocial', 'MRI_Track_Age_at_Scan',
       'Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P2_Edu', 'f_prob',
       'F_Interaction_APQ_P_APQ_P_INV', 'F_Interaction_APQ_P_APQ_P_PP', 'F_Interaction_SDQ_SDQ_Hyperactivity',
       'F_Interaction_MRI_Track_Age_at_Scan', 'F_Interaction_SDQ_SDQ_Generating_Impact'
]

# Features to be interacted for adhd
interactions = [
    "APQ_P_APQ_P_INV", "APQ_P_APQ_P_PP", "SDQ_SDQ_Hyperactivity", 
    "MRI_Track_Age_at_Scan", "SDQ_SDQ_Generating_Impact", "SDQ_SDQ_Externalizing", 
    "SDQ_SDQ_Peer_Problems", "SDQ_SDQ_Prosocial", "EHQ_EHQ_Total", "SDQ_SDQ_Difficulties_Total",
    'SDQ_SDQ_Conduct_Problems'
]


train_solutions['ADHD_Outcome'].value_counts().plot(kind='bar', color='blue')
plt.title('ADHD Outcome')
plt.xlabel('Outcome (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()


# Correlation heatmap
def plot_correlation_heatmap(df, title="Feature Correlations"):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.show()


# Distribution of numerical features
def plot_numerical_distributions(df, numerical_columns):
    for col in numerical_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30, color='green')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()


def plot_categorical_features(df, categorical_columns):
    for col in categorical_columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, data=df, palette="viridis")
        plt.title(f"Count Plot of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


def plot_boxplots_by_group(df, numerical_columns, group_column="ADHD_Outcome"):
    for col in numerical_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=group_column, y=col, data=df, palette="coolwarm")
        plt.title(f"{col} Distribution by {group_column}")
        plt.xlabel(group_column)
        plt.ylabel(col)
        plt.show()


if __name__ == "__main__":
    combined = train_combined.copy()
    combined['ADHD_Outcome'] = train_solutions['ADHD_Outcome']

    # Plot correlation heatmap
    plot_correlation_heatmap(combined, title="Correlation Heatmap of Features")

    # Plot numerical feature distributions
    numerical_columns = combined.select_dtypes(include=['number']).columns
    plot_numerical_distributions(combined, numerical_columns)

    # Plot categorical feature distributions
    categorical_columns = combined.select_dtypes(include=['object', 'category']).columns
    plot_categorical_features(combined, categorical_columns)

    # # Plot boxplots of numerical features grouped by ADHD outcome
    # plot_boxplots_by_group(combined, numerical_columns, group_column="ADHD_Outcome")


model_1 = None
model_2 = None

def eval_metrics(y_true, y_pred, weights, label="None", thresh=0.5):
    
    f1 = f1_score(y_true, (y_pred > thresh).astype(int), sample_weight=weights)
    print(f"{label} -> F1: {f1:.4f}")
    return f1

def add_interaction_terms(X, interactions, base_feature):
    
    for interaction in interactions:
        X[f"F_Interaction_{interaction}"] = X[interaction] * X[base_feature]
    return X

def train_and_evaluate_fold(
    fold, train_idx, val_idx, train_data, y_sex, y_adhd, combinations, 
    model_1, model_2, features_sex, features_adhd, interactions, t_sex, t_adhd, 
    sex_oof, adhd_oof, scores_sex, scores_adhd
):
    print(f"\n Fold {fold}")
    
    # Split data
    X_train, X_val = train_data.iloc[train_idx], train_data.iloc[val_idx]
    y_train_sex, y_val_sex = y_sex.iloc[train_idx], y_sex.iloc[val_idx]
    y_train_adhd, y_val_adhd = y_adhd.iloc[train_idx], y_adhd.iloc[val_idx]

    # Set weights
    weights_train = np.where(combinations.iloc[train_idx] == "11", 2, 1)
    weights_val = np.where(combinations.iloc[val_idx] == "11", 2, 1)

    # Train and evaluate Sex_F model
    model_1.fit(X_train[features_sex], y_train_sex, sample_weight=weights_train)
    sex_train = model_1.predict_proba(X_train[features_sex])[:, 1]
    sex_val = model_1.predict_proba(X_val[features_sex])[:, 1]
    sex_oof[val_idx] += sex_val / REPEATS

    # Evaluate Sex_F
    sex_f1 = eval_metrics(y_val_sex, sex_val, weights_val, "Sex_F", thresh=t_sex)
    scores_sex.append(sex_f1)

    # Add f_prob to data
    X_train["f_prob"] = sex_train
    X_val["f_prob"] = sex_val

    # Add interaction terms for ADHD model
    X_train = add_interaction_terms(X_train, interactions, base_feature="f_prob")
    X_val = add_interaction_terms(X_val, interactions, base_feature="f_prob")

    # Train and evaluate ADHD model
    model_2.fit(X_train[features_adhd], y_train_adhd, sample_weight=weights_train)
    adhd_val = model_2.predict_proba(X_val[features_adhd])[:, 1]
    adhd_oof[val_idx] += adhd_val / REPEATS

    # Evaluate ADHD
    adhd_f1 = eval_metrics(y_val_adhd, adhd_val, weights_val, "Outcome ADHD", thresh=t_adhd)
    scores_adhd.append(adhd_f1)

# Main training
def train_models(
    train_data, y_sex, y_adhd, combinations, features_sex, features_adhd, interactions, 
    params_1, params_2, t_sex, t_adhd, folds=FOLDS, repeats=REPEATS, seed=SEED
):

    global model_1, model_2
    
    scores_sex = []
    scores_adhd = []
    sex_oof = np.zeros(len(y_sex))
    adhd_oof = np.zeros(len(y_adhd))

    # Models
    model_1 = LogisticRegressionCV(**params_1)
    model_2 = LogisticRegressionCV(**params_2)

    # K-Fold Splits
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)

    

    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(rskf.split(train_data, combinations), 1):
        train_and_evaluate_fold(
            fold, train_idx, val_idx, train_data, y_sex, y_adhd, combinations,
            model_1, model_2, features_sex, features_adhd, interactions, t_sex, t_adhd,
            sex_oof, adhd_oof, scores_sex, scores_adhd
        )


    return scores_sex, scores_adhd, sex_oof, adhd_oof


if __name__ == "__main__":
    # Define parameters for models
    params_1 = {
        "penalty": "l1",
        "Cs": [10],
        "cv": StratifiedKFold(n_splits=FOLDS),
        "fit_intercept": True,
        "scoring": "f1",
        "random_state": SEED,
        "solver": "saga"
    }

    params_2 = {
        "penalty": "l1",
        "Cs": [10],
        "cv": StratifiedKFold(n_splits=FOLDS),
        "fit_intercept": True,
        "scoring": "f1",
        "random_state": SEED,
        "solver": "saga"
    }

    # Train models
    startbigmodel_time = time.time()
    scores_sex, scores_adhd, sex_oof, adhd_oof = train_models(
        train_combined, y_sex, y_adhd, combinations, features_sex, features_adhd, 
        interactions, params_1, params_2, t_sex=0.3, t_adhd=0.4
    )


def find_best_threshold(oof_predictions, true_labels, weights, thresholds, desc="Threshold Optimization"):
    
    scores = []
    for t in tqdm(thresholds, desc=desc):
        tmp_pred = np.where(oof_predictions > t, 1, 0)
        tmp_score = f1_score(true_labels, tmp_pred, sample_weight=weights)
        scores.append(tmp_score)
    best_threshold = thresholds[np.argmax(scores)]
    best_score = max(scores)
    return best_threshold, best_score, scores


if __name__ == "__main__":
    # weightage
    weights = ((y_adhd == 1) & (y_sex == 1)) + 1
    thresholds = np.linspace(0, 1, 100)
    
    # Find the best threshold for Sex_F
    best_sex_threshold, best_sex_score, sex_scores = find_best_threshold(
        sex_oof, y_sex, weights, thresholds, desc="Sex Thresholds"
    )
    
    # Find the best threshold for ADHD
    best_adhd_threshold, best_adhd_score, adhd_scores = find_best_threshold(
        adhd_oof, y_adhd, weights, thresholds, desc="ADHD Thresholds"
    )


model_1.fit(train_combined[features_sex], y_sex, sample_weight=weights)

start_time_pred = time.time()

f_prob_train = model_1.predict_proba(train_combined[features_sex])[:,1]
f_prob_test = model_1.predict_proba(test_combined[features_sex])[:,1]

train_combined["f_prob"] = f_prob_train
test_combined["f_prob"] = f_prob_test

for interaction in interactions:
    train_combined[f"F_Interaction_{interaction}"] = train_combined["f_prob"] * train_combined[interaction]
    test_combined[f"F_Interaction_{interaction}"] = test_combined["f_prob"] * test_combined[interaction]

model_2.fit(train_combined[features_adhd], y_adhd, sample_weight=weights)


end_time_train = time.time()
total_training_time = end_time_train - startbigmodel_time
print(f"Total Training Time: {total_training_time:.2f} seconds")

adhd_proba_test = model_2.predict_proba(test_combined[features_adhd])[:,1]



submission = pd.read_excel("/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx")
submission["ADHD_Outcome"] = np.where(adhd_proba_test > best_adhd_threshold, 1, 0)
submission["Sex_F"] = np.where(f_prob_test > best_sex_threshold, 1, 0)

submission.to_csv("submission.csv", index=False)

prediction_time = time.time() - start_time_pred
print(f"Prediction Time: {prediction_time:.4f} seconds")


import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(coefficients, feature_names, title, top_n=10):
    
    #Create a DataFrame for feature importance
    coeffs_df = pd.DataFrame({"Feature": feature_names, "Importance": coefficients})
    coeffs_df["AbsImportance"] = coeffs_df["Importance"].abs()
    coeffs_df = coeffs_df.sort_values(by="AbsImportance", ascending=False).head(top_n)
    
    # Plot the top features
    plt.figure(figsize=(10, 6))
    plt.barh(coeffs_df["Feature"], coeffs_df["Importance"], color='skyblue')
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

# ADHD Features
adhd_coefficients = model_2.coef_[0] 
plot_feature_importance(adhd_coefficients, features_adhd, title="Top ADHD Features", top_n=10)

# Sex Features
sex_coefficients = model_1.coef_[0] 
plot_feature_importance(sex_coefficients, features_sex, title="Top Sex Features", top_n=10)


def get_top_features(coefficients, feature_names, top_n=5):
    coeffs_df = pd.DataFrame({"Feature": feature_names, "Importance": coefficients})
    coeffs_df["AbsImportance"] = coeffs_df["Importance"].abs()
    coeffs_df = coeffs_df.sort_values(by="AbsImportance", ascending=False).head(top_n)
    return coeffs_df["Feature"].tolist()

# Get top 5 features for ADHD and Sex
top_features_adhd = get_top_features(adhd_coefficients, features_adhd, top_n=5)
top_features_sex = get_top_features(sex_coefficients, features_sex, top_n=5)

print("Top 5 Features for ADHD:", top_features_adhd)
print("Top 5 Features for Sex:", top_features_sex)


X_train_simplified_sex = train_combined[top_features_sex]
X_train_simplified_adhd = train_combined[top_features_adhd]


simplified_model_sex = LogisticRegression(penalty="l1", solver="saga", random_state=42)
simplified_model_adhd = LogisticRegression(penalty="l1", solver="saga", random_state=42)

# Cross-validation
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

#store oof
sex_oof = np.zeros(len(y_sex))
adhd_oof = np.zeros(len(y_adhd)) 


sex_scores = []
adhd_scores = []


start_time_train = time.time()


for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_simplified_sex, y_sex), 1):
    print(f"\n Fold {fold} ")
    
    X_train_sex_fold, X_val_sex_fold = X_train_simplified_sex.iloc[train_idx], X_train_simplified_sex.iloc[val_idx]
    y_train_sex_fold, y_val_sex_fold = y_sex.iloc[train_idx], y_sex.iloc[val_idx]
    
    X_train_adhd_fold, X_val_adhd_fold = X_train_simplified_adhd.iloc[train_idx], X_train_simplified_adhd.iloc[val_idx]
    y_train_adhd_fold, y_val_adhd_fold = y_adhd.iloc[train_idx], y_adhd.iloc[val_idx]
    
    simplified_model_sex.fit(X_train_sex_fold, y_train_sex_fold)
    sex_val_proba = simplified_model_sex.predict_proba(X_val_sex_fold)[:, 1]
    sex_oof[val_idx] = sex_val_proba 
    
    simplified_model_adhd.fit(X_train_adhd_fold, y_train_adhd_fold)
    adhd_val_proba = simplified_model_adhd.predict_proba(X_val_adhd_fold)[:, 1]
    adhd_oof[val_idx] = adhd_val_proba  
    
    sex_f1 = f1_score(y_val_sex_fold, (sex_val_proba > 0.5).astype(int))
    adhd_f1 = f1_score(y_val_adhd_fold, (adhd_val_proba > 0.5).astype(int))
    
    print(f"Sex Model Fold F1: {sex_f1:.4f}")
    print(f"ADHD Model Fold F1: {adhd_f1:.4f}")
    
    sex_scores.append(sex_f1)
    adhd_scores.append(adhd_f1)

training_time = time.time() - start_time_train
print(f"\nTraining Time for Simplified Models with Cross-Validation: {training_time:.4f} seconds")

# Find Best Thresholds
thresholds = np.linspace(0, 1, 100)


best_sex_threshold = 0.5
best_sex_f1 = 0
for t in thresholds:
    sex_preds = (sex_oof > t).astype(int)
    f1 = f1_score(y_sex, sex_preds)
    if f1 > best_sex_f1:
        best_sex_f1 = f1
        best_sex_threshold = t

print(f"Best Threshold for Sex: {best_sex_threshold:.2f}, Best F1: {best_sex_f1:.4f}")


best_adhd_threshold = 0.5
best_adhd_f1 = 0
for t in thresholds:
    adhd_preds = (adhd_oof > t).astype(int)
    f1 = f1_score(y_adhd, adhd_preds)
    if f1 > best_adhd_f1:
        best_adhd_f1 = f1
        best_adhd_threshold = t

print(f"Best Threshold for ADHD: {best_adhd_threshold:.2f}, Best F1: {best_adhd_f1:.4f}")

# Predict for test_combined
test_combined_simplified_sex = test_combined[top_features_sex]
test_combined_simplified_adhd = test_combined[top_features_adhd]


start_time_pred = time.time()

f_prob_test = simplified_model_sex.predict_proba(test_combined_simplified_sex)[:, 1]
adhd_proba_test = simplified_model_adhd.predict_proba(test_combined_simplified_adhd)[:, 1]

prediction_time = time.time() - start_time_pred
print(f"Prediction Time for Simplified Models: {prediction_time:.4f} seconds")

# Create Submission File
submission = pd.read_excel("/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx")
submission["Sex_F"] = np.where(f_prob_test > best_sex_threshold, 1, 0)
submission["ADHD_Outcome"] = np.where(adhd_proba_test > best_adhd_threshold, 1, 0)

submission.to_csv("submission_simplified.csv", index=False)
print("Submission file submission_simplified.csv has been created.")

