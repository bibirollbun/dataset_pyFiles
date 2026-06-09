import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

widsdatathon2025_path = "/kaggle/input/widsdatathon2025"

# training dataset
quantitative_train = pd.read_excel(f'{widsdatathon2025_path}/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx', index_col='participant_id')
categorical_train = pd.read_excel(f'{widsdatathon2025_path}/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx', index_col='participant_id')
fmri_train = pd.read_csv(f'{widsdatathon2025_path}/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv', index_col='participant_id')
target_train = pd.read_excel(f'{widsdatathon2025_path}/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx', index_col='participant_id')

target_train.sort_index(inplace=True)
quantitative_train.sort_index(inplace=True)
categorical_train.sort_index(inplace=True)
fmri_train.sort_index(inplace=True)

target_train_classes = []
for idx in range(quantitative_train.shape[0]):
  participant_id = quantitative_train.index[idx]
  class_label = 2 * target_train.loc[participant_id]['ADHD_Outcome'] + target_train.loc[participant_id]['Sex_F']
  target_train_classes.append(class_label)

target_train_classes = pd.DataFrame(target_train_classes, columns=['class_label'], index=quantitative_train.index)

# testing dataset
quantitative_test = pd.read_excel(f'{widsdatathon2025_path}/TEST/TEST_QUANTITATIVE_METADATA.xlsx', index_col='participant_id')
categorical_test = pd.read_excel(f'{widsdatathon2025_path}/TEST/TEST_CATEGORICAL.xlsx', index_col='participant_id')
fmri_test = pd.read_csv(f'{widsdatathon2025_path}/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv', index_col='participant_id')

# total
quantitative_total = pd.concat([quantitative_train, quantitative_test])
categorical_total = pd.concat([categorical_train, categorical_test])
fmri_total = pd.concat([fmri_train, fmri_test])

def get_missing_value_ratio(df):
  val = (df.isna().sum() / df.shape[0]) * 100
  val = val[val > 0]
  return val.sort_values(ascending=True)

def plot_missing_values_ratio(df, title):
  plt.figure(figsize=(12, 8))
  sns.barplot(x=df.index, y=df.values, palette="rocket", hue=df.index, legend=False)
  plt.xticks(rotation=90)
  plt.xlabel('Columns')
  plt.ylabel('Percentage of Missing Values')
  plt.title(title)
  plt.tight_layout()
  plt.show()

missing_q_t = get_missing_value_ratio(quantitative_train)
missing_c_t = get_missing_value_ratio(categorical_train)
missing_q_te = get_missing_value_ratio(quantitative_test)
missing_c_te = get_missing_value_ratio(categorical_test)
missing_c_total = get_missing_value_ratio(categorical_total)
missing_q_total = get_missing_value_ratio(quantitative_total)

print('Missing values ratio on quantitative train dataset:')
print(missing_q_t)
print('Missing values ratio on categorical train dataset:')
print(missing_c_t)
print('Missing values ratio on quantitative test dataset:')
print(missing_q_te)
print('Missing values ratio on categorical test dataset:')
print(missing_c_te)
print('Missing values ratio on quantitative total dataset:')
print(missing_q_total)
print('Missing values ratio on categorical total dataset:')
print(missing_c_total)

plot_missing_values_ratio(missing_q_t, 'Missing values ratio on quantitative train dataset')
plot_missing_values_ratio(missing_c_t, 'Missing values ratio on categorical train dataset')
plot_missing_values_ratio(missing_q_te, 'Missing values ratio on quantitative test dataset')
plot_missing_values_ratio(missing_c_te, 'Missing values ratio on categorical test dataset')
plot_missing_values_ratio(missing_c_total, 'Missing values ratio on categorical total dataset')
plot_missing_values_ratio(missing_q_total, 'Missing values ratio on quantitative total dataset')

numerical_cols = quantitative_total.columns
print('Numerical cols:', numerical_cols)

categorical_cols = categorical_total.columns
print('Categorical cols:', categorical_cols)


quantitative_total[numerical_cols].describe().T


# Plot Data for each attribute
for col in numerical_cols:
    sns.histplot(x=quantitative_total[col], data=quantitative_total, color="teal")
    plt.show(block=True)


# Checking for outliers
for col in numerical_cols:
    sns.boxplot(x=quantitative_total[col], data=quantitative_total, color="indianred")
    plt.show(block=True)


def check_outliers(df, numerical_cols, low_threshold=0.1, up_threshold=0.9):
    outlier_cols = []
    for col in numerical_cols:
        q1 = df[col].quantile(low_threshold)
        q3 = df[col].quantile(up_threshold)
        interquantile = q3 - q1
        up_limit = q3 + 1.5 * interquantile
        low_limit = q1 - 1.5 * interquantile
        if df[(df[col] > up_limit) | (df[col] < low_limit)].any(axis=None):
            outlier_cols.append(col)
    if not outlier_cols:
        print("There is no outliers")
    return outlier_cols


outlier_cols = check_outliers(quantitative_total, numerical_cols)
outlier_cols


sns.heatmap(quantitative_total[numerical_cols].corr(), annot=True, linewidths=0.5,)


for col in categorical_cols:
    # Get unique values and their counts
    unique_values = categorical_total[col].unique()
    value_counts = categorical_total[col].value_counts()

    # Reindex value_counts to include all unique values
    value_counts = value_counts.reindex(unique_values, fill_value=0)

    sns.barplot(x=unique_values, y=value_counts, palette="rocket", hue=unique_values, legend=False).set(title=col)
    plt.show(block=True)


# Get ADHD_Outcome counts and plot
sns.barplot(x = target_train["ADHD_Outcome"].unique(), y = target_train["ADHD_Outcome"].value_counts()/target_train['ADHD_Outcome'].shape[0], palette="viridis", hue=target_train["ADHD_Outcome"].unique(), legend=False).set(title="ADHD_Outcome")


# Get Sex_F counts and plot
sns.barplot(x = target_train["Sex_F"].unique(), y = target_train["Sex_F"].value_counts()/target_train['Sex_F'].shape[0], palette="viridis", hue=target_train["Sex_F"].unique(), legend=False).set(title="Sex_F")


# Get class labels counts and plot
sns.barplot(x = target_train_classes["class_label"].unique(), y = target_train_classes["class_label"].value_counts()/target_train_classes['class_label'].shape[0], palette="viridis", hue=target_train_classes["class_label"].unique(), legend=False).set(title="class_label")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, make_scorer, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from statsmodels.stats.weightstats import ztest
import joblib

import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression

# --- Configuration ---
submission_path = "stacked_binary_submission.csv"
RNG = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
SCORER = make_scorer(f1_score)

widsdatathon2025_path = "/kaggle/input/widsdatathon2025"

# --- Data Loading ---
quant_train = pd.read_excel(f"{widsdatathon2025_path}/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
cat_train   = pd.read_excel(f"{widsdatathon2025_path}/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
fmri_train  = pd.read_csv(f"{widsdatathon2025_path}/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
target_train= pd.read_excel(f"{widsdatathon2025_path}/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
quant_test  = pd.read_excel(f"{widsdatathon2025_path}/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
cat_test    = pd.read_excel(f"{widsdatathon2025_path}/TEST/TEST_CATEGORICAL.xlsx")
fmri_test   = pd.read_csv(f"{widsdatathon2025_path}/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")

train_df = (target_train
            .merge(quant_train, on='participant_id')
            .merge(cat_train, on='participant_id')
            .merge(fmri_train, on='participant_id'))
test_df = (quant_test
           .merge(cat_test, on='participant_id')
           .merge(fmri_test, on='participant_id'))

# --- Feature Lists ---
quant_cols_all = quant_train.columns.drop('participant_id').tolist()
cat_cols = cat_train.columns.drop('participant_id').tolist()
fmri_cols = fmri_train.columns.drop('participant_id').tolist()

# --- Label-Specific Z-test Selection ---
sig_adhd = []
for feat in quant_cols_all:
    grp0 = train_df.loc[train_df['ADHD_Outcome']==0, feat].dropna()
    grp1 = train_df.loc[train_df['ADHD_Outcome']==1, feat].dropna()
    if len(grp0)>1 and len(grp1)>1:
        _, p = ztest(grp0, grp1)
        if p<0.05: sig_adhd.append(feat)
print(f"ADHD Z-test kept {len(sig_adhd)}/{len(quant_cols_all)} features")

# Final per-task feature sets
quant_cols_adhd = sig_adhd
all_feats_adhd = quant_cols_adhd + cat_cols + fmri_cols
all_feats = quant_cols_all + cat_cols + fmri_cols

# --- Preprocessor Factory ---
def make_preprocessor(quant_cols):
    num_pipe = Pipeline([
        ('log', FunctionTransformer(np.log1p, validate=False)),
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    fmri_pipe = Pipeline([
        ('var', VarianceThreshold(0.01)),
        ('pca', PCA(n_components=0.95, random_state=RNG))
    ])
    return ColumnTransformer([
        ('num', num_pipe, quant_cols),
        ('cat', cat_pipe, cat_cols),
        ('fmri', fmri_pipe, fmri_cols)
    ], remainder='drop', n_jobs=-1)

# Preprocessors for each task
pre_adhd = make_preprocessor(quant_cols_adhd)

def make_mlp(hidden, alpha, lr):
    return MLPClassifier(early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, hidden_layer_sizes=hidden, alpha=alpha,
                         learning_rate_init=lr, max_iter=500,
                         random_state=RNG)

# --- Estimators ---
adhd_estimators = [
    ('mlp', ImbPipeline([('pre', pre_adhd),('smote',SMOTE(random_state=RNG)),('clf',make_mlp((100,50,25),0.01,0.001))])),
    ('xgb', ImbPipeline([('pre', pre_adhd),('smote',SMOTE(random_state=RNG)),('clf',XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, verbosity=0,
        objective='binary:logistic', eval_metric='logloss', random_state=RNG
    ))])),
    ('lgbm', ImbPipeline([('pre', pre_adhd),('smote',SMOTE(random_state=RNG)),('clf',LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=64,
        objective='binary', random_state=RNG, verbosity=-1
    ))]))
]

stack_adhd = StackingClassifier(
    estimators=adhd_estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=CV, n_jobs=-1, stack_method='predict_proba'
)

submission = pd.DataFrame({'participant_id': test_df['participant_id']})

stack_adhd.fit(train_df[all_feats_adhd], train_df['ADHD_Outcome'])
submission['ADHD_Outcome'] = stack_adhd.predict(test_df[all_feats_adhd])

# Doing predictions for sex label

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
pre_sex = make_preprocessor(quant_cols_all)

# --- Balanced LogisticRegression + RandomizedSearchCV ---
param_dist = {
    'clf__C':        [1],
    'clf__penalty':  ['l2'],
}

logreg_bal = Pipeline([
    ('pre', pre_sex),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        solver='saga',
        random_state=RNG
    ))
])

search = RandomizedSearchCV(
    logreg_bal,
    param_distributions=param_dist,
    n_iter=40,
    cv=CV,
    scoring=SCORER,
    n_jobs=-1,
    random_state=RNG,
    verbose=1
)

search.fit(train_df[all_feats], train_df['Sex_F'])
print("Best params:", search.best_params_)
print("Tuned LogReg F1 CV:", search.best_score_)

# --- Out-of-Fold Metrics ---
oof = cross_val_predict(search.best_estimator_, train_df[all_feats], train_df['Sex_F'], cv=CV, n_jobs=-1)
print("\nOOF classification report:\n", classification_report(train_df['Sex_F'], oof))
print("OOF confusion matrix:\n", confusion_matrix(train_df['Sex_F'], oof))

# 1) Get true out-of-fold probabilities for class 1
oof_probs = cross_val_predict(
    search.best_estimator_,         # the tuned pipeline
    train_df[all_feats],            # features
    train_df['Sex_F'],              # target
    cv=CV,
    method='predict_proba',
    n_jobs=-1
)[:, 1]

# 2) Tune threshold on those OOF probabilities
prec, recall, thresholds = precision_recall_curve(train_df['Sex_F'], oof_probs)
f1_scores = 2 * prec * recall / (prec + recall)
best_idx = np.nanargmax(f1_scores)
best_thr = thresholds[best_idx]
print(f"Proper OOF‐based optimal threshold: {best_thr:.3f}, OOF F1: {f1_scores[best_idx]:.3f}")

# 3) Evaluate OOF with that threshold
oof_preds = (oof_probs >= best_thr).astype(int)
print("\nOOF classification report at tuned threshold:\n",
      classification_report(train_df['Sex_F'], oof_preds))
print("OOF confusion matrix:\n", confusion_matrix(train_df['Sex_F'], oof_preds))

# 4) Finally, apply to test set
test_probs = search.best_estimator_.predict_proba(test_df[all_feats])[:, 1]
submission2 = pd.DataFrame({
    'participant_id': test_df['participant_id'],
    'Sex_F':          (test_probs >= best_thr).astype(int)
})

submission = submission.merge(submission2, on='participant_id')
submission.to_csv(submission_path, index=False)

