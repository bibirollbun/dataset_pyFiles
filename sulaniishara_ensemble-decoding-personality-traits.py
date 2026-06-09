import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier, plot_importance as xgb_plot_importance
from lightgbm import LGBMClassifier, plot_importance as lgbm_plot_importance
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import optuna
from optuna.samplers import TPESampler
import time

warnings.filterwarnings("ignore")

palette = sns.color_palette("PRGn", 10)
sns.set_palette(palette)
sns.set_style("whitegrid", {
    'grid.color': '.7',
    'grid.linestyle': ':',
    'grid.linewidth': 0.7
})


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
original_df = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv")

print("Loaded Datasets:")


print("\nOriginal Data Info:")
original_df.info()
print("\nTrain Data Info:")
train_df.info()
print("\nTest Data Info:")
test_df.info()


def missing_values_summary(df, dataset_name="Dataset"):
    missing_count = df.isnull().sum()
    missing_pct = 100 * missing_count / len(df)
    data_types = df.dtypes
    summary_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %": missing_pct.round(2),
        "Dtype": data_types
    }).sort_values(by="Missing %", ascending=False)
    
    print(f"\nMissing Values Report â€” {dataset_name}")
    print(f"Total missing entries: {missing_count.sum()}\n")
    display(summary_df)
    return summary_df

def check_duplicates(df, dataset_name="Dataset"):
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate Rows Report â€” {dataset_name}")
    print(f"Total duplicate rows: {dup_count}\n")
    
    if dup_count > 0:
        print("Sample duplicate rows (first 5):")
        display(df[df.duplicated()].head())
    return dup_count

train_missing_summary = missing_values_summary(train_df, "Train Dataset")
train_duplicates = check_duplicates(train_df, "Train Dataset")

test_missing_summary = missing_values_summary(test_df, "Test Dataset")
test_duplicates = check_duplicates(test_df, "Test Dataset")

original_missing_summary = missing_values_summary(original_df, "Original Dataset")
original_duplicates = check_duplicates(original_df, "Original Dataset")


def target_distribution_summary(df, target_col='Personality'):
    counts = df[target_col].value_counts(dropna=False)
    percent = round(100 * counts / counts.sum(), 2)
    summary_df = pd.DataFrame({'Count': counts, 'Percentage (%)': percent})
    return summary_df

def print_target_comparison(df1, name1, df2, name2, target_col='Personality'):
    dist1 = target_distribution_summary(df1, target_col)
    dist2 = target_distribution_summary(df2, target_col)
    print(f"\nTarget Distribution - {name1}")
    display(dist1)
    print(f"\nTarget Distribution - {name2}")
    display(dist2)

print_target_comparison(train_df, 'Train Data', original_df, 'Original Data', target_col='Personality')


print("\nOriginal Data Describe (Numerical):")
display(original_df.describe().T)
print("\nTrain Data Describe (Numerical):")
display(train_df.describe().T)
print("\nTest Data Describe (Numerical):")
display(test_df.describe().T)

print("\nOriginal Data Describe (Categorical):")
display(original_df.describe(include=['object', 'category']).T)
print("\nTrain Data Describe (Categorical):")
display(train_df.describe(include=['object', 'category']).T)
print("\nTest Data Describe (Categorical):")
display(test_df.describe(include=['object', 'category']).T)


def numerical_describe_by_group(df, group_col='Personality'):
    print(f"\nDescriptive Statistics Grouped by '{group_col}'")
    display(df.groupby(group_col).describe().transpose())

numerical_describe_by_group(train_df, group_col='Personality')
numerical_describe_by_group(original_df, group_col='Personality')


def categorical_describe_by_group(df, group_col='Personality'):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.to_list()
    
    print(f"\nDescriptive Statistics of Categorical Features Grouped by '{group_col}':")
    desc = df.groupby(group_col)[cat_cols].describe().transpose()
    display(desc)

categorical_describe_by_group(train_df, group_col='Personality')
categorical_describe_by_group(original_df, group_col='Personality')


def missing_by_group(df, group_col='Personality'):
    group_counts = df[group_col].value_counts(dropna=False)
    summary = {}
    dtypes = df.dtypes
    for group in group_counts.index:
        group_df = df[df[group_col] == group]
        total = len(group_df)
        missing_count = group_df.isnull().sum()
        missing_pct = 100 * missing_count / total
        summary[group] = pd.DataFrame({
            'Missing Count': missing_count,
            'Missing %': missing_pct.round(2),
            'Dtype': dtypes
        })
    return summary

result = missing_by_group(train_df, group_col='Personality')
for personality_type, table in result.items():
    print(f"\nMissing Value Summary for Personality: {personality_type}")
    display(table[table['Missing Count'] > 0])


def correlation_summary(df, dataset_name):
    numeric_df = df.select_dtypes(include=['number'])
    
    print(f"\nCorrelation Matrix - {dataset_name}\n")
    corr_matrix = numeric_df.corr(method='pearson')
    display(corr_matrix.round(3))
    return corr_matrix

train_corr = correlation_summary(train_df, "Train Dataset")
original_corr = correlation_summary(original_df, "Original Dataset")


def compare_correlation(df1, name1, df2, name2):
    # Drop 'id' if exists and select numeric columns
    corr1 = df1.drop(columns=['id'], errors='ignore').select_dtypes(include=['number']).corr()
    corr2 = df2.drop(columns=['id'], errors='ignore').select_dtypes(include=['number']).corr()
    diff = (corr1 - corr2).abs()
    
    print(f"\nDifference in correlations: {name1} vs {name2}\n")
    display(diff.round(3))
    return diff

corr_diff = compare_correlation(train_df, "Train Dataset", original_df, "Original Dataset")


def missingness_correlation(df, dataset_name):
    numeric_df = df.select_dtypes(include=['number'])
    
    # Boolean DataFrame: True if missing
    missing_bool = numeric_df.isnull()
    
    print(f"\nMissing Value Correlation Matrix - {dataset_name}\n"
          "(Correlation between missing/not missing indicators per feature pair)")
    
    corr_missing = missing_bool.corr(method='pearson')
    display(corr_missing.round(3))
    
    return corr_missing

train_missing_corr = missingness_correlation(train_df, "Train Dataset")


original_df = (
    original_df
    .rename(columns={'Personality': 'match_p'})
    .drop_duplicates([
        'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
        'Going_outside', 'Drained_after_socializing',
        'Friends_circle_size', 'Post_frequency'
    ])
)

merge_cols = [col for col in original_df.columns if col != 'match_p']
train_df = train_df.merge(original_df, how='left', on=merge_cols)
test_df = test_df.merge(original_df, how='left', on=merge_cols)

# After Merging
print("\nNull values after merge (train):")
display(train_df.isnull().sum().to_frame("Missing Values"))
print("\nNull values after merge (test):")
display(test_df.isnull().sum().to_frame("Missing Values"))

print("\ntrain_df info:")
train_df.info()
print("\ntest_df info:")
test_df.info()


train_ID = train_df.pop('id')
test_ID = test_df.pop('id')
y_train = train_df.pop('Personality').map({'Extrovert': 1, 'Introvert': 0}).values
ntrain = train_df.shape[0]
all_data = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
all_data.drop(columns='Personality', inplace=True, errors='ignore')

# After Preparing all_data
print("\nSample of combined data:")
display(all_data.head())

print("\nall_data info:")
all_data.info()

target_dist = pd.Series(y_train).value_counts(normalize=True) * 100
print("\nTarget Variable Distribution (%):")
for cls, pct in target_dist.items():
    print(f"  - Class {cls}: {pct:.2f}%")


def fill_missing_by_quantile_group(df, group_source_col, target_col, quantiles=[0, 0.25, 0.5, 0.75, 1.0], labels=None):
    if labels is None:
        labels = [f'Q{i+1}' for i in range(len(quantiles) - 1)]
    bin_col = f'{group_source_col}_bin'
    df[bin_col] = pd.qcut(df[group_source_col], q=quantiles, labels=labels)
    df[target_col] = df[target_col].fillna(df.groupby(bin_col)[target_col].transform('median'))
    df.drop(columns=[bin_col], inplace=True)
    return df


# Time_spent_Alone
for source in ['Social_event_attendance', 'Going_outside']:
    all_data = fill_missing_by_quantile_group(all_data, source, 'Time_spent_Alone')
print("Filled Time_spent_Alone missing values")
print(all_data['Time_spent_Alone'].isnull().value_counts(), "\n")

# Social_event_attendance
for source in ['Going_outside', 'Friends_circle_size', 'Post_frequency']:
    all_data = fill_missing_by_quantile_group(all_data, source, 'Social_event_attendance')
print("Filled Social_event_attendance missing values")
print(all_data['Social_event_attendance'].isnull().value_counts(), "\n")

# Friends_circle_size
for source in ['Post_frequency', 'Going_outside', 'Social_event_attendance']:
    all_data = fill_missing_by_quantile_group(all_data, source, 'Friends_circle_size')
print("Filled Friends_circle_size missing values")
print(all_data['Friends_circle_size'].isnull().value_counts(), "\n")

# Post_frequency
all_data = fill_missing_by_quantile_group(all_data, 'Friends_circle_size', 'Post_frequency')
all_data = fill_missing_by_quantile_group(all_data, 'Time_spent_Alone', 'Post_frequency')
print("Filled Post_frequency missing values")
print(all_data['Post_frequency'].isnull().value_counts(), "\n")

# Going_outside (final pass)
for source in ['Friends_circle_size', 'Post_frequency']:
    all_data = fill_missing_by_quantile_group(all_data, source, 'Going_outside')
print("Final pass on Going_outside")
print(all_data['Going_outside'].isnull().value_counts(), "\n")

# Final fill for categorical columns
all_data.fillna({
    'Stage_fear': 'Unknown',
    'Drained_after_socializing': 'Unknown'
}, inplace=True)
print("Filled missing categorical values")
print(all_data[['Stage_fear', 'Drained_after_socializing']].isnull().sum(), "\n")

# Verify There Are No Missing Values
print("Data Overview after Imputation:")
all_data.info()


all_data = pd.get_dummies(all_data, columns=[
    'Stage_fear', 'Drained_after_socializing', 'match_p'
], prefix=['Stage', 'Drained', 'match'])

print("One-hot encoded categorical columns")
print("Final Columns:")
print(all_data.columns.tolist())

# Verify completenessâ€”should be no missing values and all columns are now numeric or bool
print("\nData Overview after One-Hot Encoding:")
all_data.info()



def final_data_summary(df):
    total_rows = df.shape[0]
    summary = pd.DataFrame({
        'Feature': df.columns,
        'Count': df.count().values,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum() / total_rows * 100).round(2).values,
        'Dtype': [str(dtype) for dtype in df.dtypes]
    })
    summary = summary[['Feature', 'Count', 'Missing Count', 'Missing %', 'Dtype']]
    return summary

print("\nFinal Data Summary Check:")
final_summary_df = final_data_summary(all_data)
display(final_summary_df)

# Assert no missing values remain
assert all_data.isnull().sum().sum() == 0, "There are still missing values!"

print("\nNo missing values detected. Data is ready for modeling.")


X_train = all_data[:ntrain]
X_test = all_data[ntrain:]
X = X_train
y = y_train


class_1 = y.sum()
class_0 = len(y) - class_1
scale_pos_weight = class_0 / class_1


# Best Trial Parameters (use after Optuna)
best_params_dict = {
    'XGBoost': {
        'max_depth': 10, 
        'learning_rate': 0.013683607181209666, 
        'n_estimators': 735,
        'subsample': 0.8526047335850097, 
        'colsample_bytree': 0.7839342871434789,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42, 'verbosity': 0, 'n_jobs': -1
    },
    'CatBoost': {
        'iterations': 894, 
        'depth': 6, 
        'learning_rate': 0.015254285834997475,
        'class_weights': [scale_pos_weight, 1],
        'random_seed': 42, 'verbose': 0
    },
    'LightGBM_gbdt': {
        'boosting_type': 'gbdt', 
        'num_leaves': 48, 
        'learning_rate': 0.014034705559998232,
        'n_estimators': 696, 
        'subsample': 0.7586519441655896, 
        'colsample_bytree': 0.82266052103882,
        'class_weight': {0: scale_pos_weight, 1: 1},
        'random_state': 42, 'verbosity': -1
    },
    'LightGBM_goss': {
        'boosting_type': 'goss', 
        'num_leaves': 56, 
        'learning_rate': 0.02046015361791542,
        'n_estimators': 750, 
        'subsample': 0.9276519441655896, 
        'colsample_bytree': 0.7537907009597088,
        'class_weight': {0: scale_pos_weight, 1: 1},
        'random_state': 42, 'verbosity': -1
    },
    'HistGB': {
        'max_iter': 300, 
        'max_depth': 8, 
        'learning_rate': 0.0201942082243779,
        'min_samples_leaf': 20, 
        'class_weight': 'balanced',
        'random_state': 42
    }
}


# Instantiate base learners
xgb = XGBClassifier(**best_params_dict['XGBoost'])
cat = CatBoostClassifier(**best_params_dict['CatBoost'])
lgbm_gbdt = LGBMClassifier(**best_params_dict['LightGBM_gbdt'])
lgbm_goss = LGBMClassifier(**best_params_dict['LightGBM_goss'])
hgb = HistGradientBoostingClassifier(**best_params_dict['HistGB'])

base_models = [
    ('xgb', xgb), ('cat', cat),
    ('lgbm_gbdt', lgbm_gbdt),
    ('lgbm_goss', lgbm_goss),
    ('hgb', hgb)
]


def get_oof_predictions_detailed(models, X, y, X_test, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros((X.shape[0], len(models)))
    test_preds = np.zeros((X_test.shape[0], len(models)))
    rows = []

    for idx, (name, model) in enumerate(models):
        print(f"\nTraining base model: {name}")
        test_fold_preds, fold_val_acc, fold_train_acc = [], [], []
        start = time.time()

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model.fit(X_tr, y_tr)

            train_acc = accuracy_score(y_tr, model.predict(X_tr))
            val_acc = accuracy_score(y_val, model.predict(X_val))
            fold_train_acc.append(train_acc)
            fold_val_acc.append(val_acc)
            print(f"  Fold {fold+1} - Train: {train_acc:.4f} | Val: {val_acc:.4f}")

            oof_preds[val_idx, idx] = model.predict_proba(X_val)[:, 1]
            test_fold_preds.append(model.predict_proba(X_test)[:, 1])

        end = time.time()
        test_preds[:, idx] = np.mean(test_fold_preds, axis=0)

        row = {
            'Model': name,
            **{f'Fold{i+1}': v for i, v in enumerate(fold_val_acc)},
            'Mean': np.mean(fold_val_acc),
            'Std': np.std(fold_val_acc)
        }
        rows.append(row)

        print(f"{name} - Train Mean: {np.mean(fold_train_acc):.4f} Â± {np.std(fold_train_acc):.4f} | "
              f"Val Mean: {np.mean(fold_val_acc):.4f} Â± {np.std(fold_val_acc):.4f} | Time: {end-start:.2f}s")

    summary_df = pd.DataFrame(rows)
    return oof_preds, test_preds, summary_df

oof_preds, test_preds, perf_summary = get_oof_predictions_detailed(base_models, X, y, X_test)
print("\nBase Learner Performance Summary (Validation Accuracies):")
display(perf_summary)


# Best Parameters (use after Optuna)
best_meta_params = {'C': 3.1566, 'penalty': 'l1'}
solver = 'liblinear' if best_meta_params['penalty'] == 'l1' else 'lbfgs'
meta_model = LogisticRegression(**best_meta_params, solver=solver, max_iter=2000)
meta_model.fit(oof_preds, y)


X_tr, X_val, y_tr, y_val, oof_tr, oof_val = train_test_split(
    X, y, oof_preds, test_size=0.2, stratify=y, random_state=42
)
meta_val = LogisticRegression(**best_meta_params, solver=solver, max_iter=2000)
meta_val.fit(oof_tr, y_tr)
stacking_probs = meta_val.predict_proba(oof_val)[:, 1]


best_thresh, best_score = 0.5, 0
for t in np.arange(0.4, 0.6, 0.01):
    acc = accuracy_score(y_val, (stacking_probs >= t).astype(int))
    if acc > best_score:
        best_score, best_thresh = acc, t
print(f"\nBest Threshold for Stacking: {best_thresh:.2f} | Validation Accuracy: {best_score:.4f}")


voting = VotingClassifier(estimators=base_models, voting='soft')
voting.fit(X_tr, y_tr)
voting_probs = voting.predict_proba(X_val)[:, 1]
voting_preds = (voting_probs >= best_thresh).astype(int)
stacking_preds = (stacking_probs >= best_thresh).astype(int)

soft_val_acc = accuracy_score(y_val, voting_preds)
stack_val_acc = accuracy_score(y_val, stacking_preds)

print("\nEnsemble Comparison:")
print(f"Soft Voting Accuracy: {soft_val_acc:.4f}")
print(f"Stacking Accuracy: {stack_val_acc:.4f}")
print(f"Improvement (Stacking - Voting): {stack_val_acc - soft_val_acc:+.4f}")


def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['Introvert', 'Extrovert'], cmap="PRGn")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

evaluate_model("Soft Voting", y_val, voting_preds)
evaluate_model("Stacking", y_val, stacking_preds)


if stack_val_acc >= soft_val_acc:
    best_method = "Stacking"
    print(f"\nUsing Stacking Ensemble for Final Submission (Val Acc: {stack_val_acc:.4f})")
    final_test_probs = meta_model.predict_proba(test_preds)[:, 1]
else:
    best_method = "Soft Voting"
    print(f"\nUsing Soft Voting Ensemble for Final Submission (Val Acc: {soft_val_acc:.4f})")
    voting.fit(X, y)
    final_test_probs = voting.predict_proba(X_test)[:, 1]

final_preds = (final_test_probs >= best_thresh).astype(int)
submission = pd.DataFrame({'id': test_ID, 'Personality': final_preds})
submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
submission_file = f"submission_{best_method.lower()}.csv"
submission.to_csv(submission_file, index=False)

print(f"\nFinal Submission saved as '{submission_file}' using {best_method} Ensemble!")


print("\nTest Set Prediction Distribution:")
print(submission['Personality'].value_counts())
sns.countplot(data=submission, x='Personality', palette='PRGn')
plt.title(f"Test Set Personality Distribution ({best_method})")
plt.show()


# Final Output
print("\nWorkflow completed successfully!")
display(submission.head())

