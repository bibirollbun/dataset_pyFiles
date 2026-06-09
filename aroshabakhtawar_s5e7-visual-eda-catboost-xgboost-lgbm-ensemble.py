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



# ğŸ“š Importing data manipulation libraries
import pandas as pd  # for loading and working with tabular data (train/test datasets)
import numpy as np   # for numerical computations, arrays, and random number generation

# ğŸ“Š Importing visualization libraries
import matplotlib.pyplot as plt  # for custom and fine-grained visualizations
import seaborn as sns            # high-level interface for drawing attractive plots

# ğŸ§  Sklearn modules for preprocessing, validation, and evaluation
from sklearn.model_selection import StratifiedKFold  # for cross-validation with balanced classes
from sklearn.preprocessing import StandardScaler     # for scaling numerical features
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report  # model evaluation

# âš ï¸� To suppress unnecessary warnings from libraries
import warnings
warnings.filterwarnings('ignore')  # hides warnings to keep output clean

# ğŸ”� Set a random seed for reproducibility
RANDOM_STATE = 42  # ensures that data splits and results are consistent across runs

# ğŸ�¨ Set default styles for plots
plt.style.use('seaborn-darkgrid')  # apply a clean visual style to matplotlib plots
sns.set_palette('viridis')         # set a beautiful color palette for seaborn plots

# âœ… Quick check to confirm the setup was successful
print("âœ… Libraries imported and environment set.")




# ğŸ“‚ Load the training and test datasets using pandas
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")  # training data includes features + target
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")    # test data includes features only

# ğŸ“Œ Display the first 5 rows of the training dataset
print("ğŸ”� First 5 rows of the training dataset:")
display(train.head())

# ğŸ§¾ View structure and data types in training data
print("\nğŸ§¾ Info about training dataset:")
train.info()

# ğŸ”¢ Shape of the datasets â€” how many rows and columns?
print(f"\nğŸ“� Training data shape: {train.shape}")
print(f"ğŸ“� Test data shape: {test.shape}")

# â�“ Check if any missing values exist in the training or test set
print("\nâ�— Missing values in training data:")
print(train.isnull().sum().sum())  # total missing values in train
print("â�— Missing values in test data:")
print(test.isnull().sum().sum())   # total missing values in test


# ğŸ�·ï¸� First, create a copy to avoid touching the original
df = train.copy()

# ğŸ”� Convert target variable 'Personality' to binary numeric for modeling
# Introvert = 0, Extrovert = 1
df['target'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})

# ğŸ�¯ Plot the distribution of the target classes (0 = Introvert, 1 = Extrovert)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='target')
plt.title("Target Class Distribution")
plt.xlabel("Target (0 = Introvert, 1 = Extrovert)")
plt.ylabel("Count")
plt.xticks([0, 1], ['Introvert', 'Extrovert'])
plt.show()

# ğŸ§¾ Show class distribution in a more descriptive way

# Count and percentage of each class
class_counts = df['target'].value_counts()
class_percentages = df['target'].value_counts(normalize=True) * 100

# Mapping class label to readable string
labels_map = {0: 'Introvert', 1: 'Extrovert'}

# Display as a table
print("ğŸ“Š Personality Class Distribution:\n")
for label in class_counts.index:
    print(f"ğŸ”¹ {labels_map[label]}: {class_counts[label]} samples ({class_percentages[label]:.2f}%)")


# ğŸ“Š Visualize distributions of a few numerical features (sample)
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[col], kde=True, bins=30, color='teal')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
plt.tight_layout()
plt.show()




# âœ… **What this block does:**

# * Converts `Personality` to a usable numeric `target`
# * Shows the balance between introverts and extroverts
# * Plots **distributions of 5 numeric features** to explore spread, skewness, or outliers




# ğŸ§© Copy the original datasets to avoid modifying the raw ones
train_df = train.copy()
test_df = test.copy()

# ğŸ�¯ Convert target to binary: 'Introvert' = 0, 'Extrovert' = 1
train_df['target'] = train_df['Personality'].map({'Introvert': 0, 'Extrovert': 1})

# ğŸ§¼ Drop columns not needed for modeling
train_df = train_df.drop(columns=['id', 'Personality'])
test_ids = test_df['id']  # Save test IDs for submission
test_df = test_df.drop(columns=['id'])

# ğŸ› ï¸� Combine train and test for unified preprocessing
test_df['target'] = np.nan  # Add dummy target for test to match columns
full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# ğŸ”� Encode categorical columns: convert 'Yes'/'No' to 1/0
cat_map = {'Yes': 1, 'No': 0}
full_df['Stage_fear'] = full_df['Stage_fear'].map(cat_map)
full_df['Drained_after_socializing'] = full_df['Drained_after_socializing'].map(cat_map)

# ğŸ§© Fill missing values
# Categorical â†’ fill with mode (most common value)
full_df['Stage_fear'].fillna(full_df['Stage_fear'].mode()[0], inplace=True)
full_df['Drained_after_socializing'].fillna(full_df['Drained_after_socializing'].mode()[0], inplace=True)

# Numeric â†’ fill with median (robust to outliers)
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']

for col in num_cols:
    full_df[col].fillna(full_df[col].median(), inplace=True)

print("âœ… Missing values handled and categorical columns encoded.")



full_df.tail()



# ğŸ”� Split back into training and test sets
processed_train = full_df[full_df['target'].notna()].copy()
processed_test = full_df[full_df['target'].isna()].drop(columns=['target']).copy()

# ğŸ�¯ Define input features and target
X_train = processed_train.drop(columns=['target'])
y_train = processed_train['target'].astype(int)  # convert to int for modeling
X_test = processed_test.copy()

# ğŸ“Š Correlation heatmap to show feature interdependencies
plt.figure(figsize=(10, 8))
corr_matrix = X_train.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("ğŸ”� Feature Correlation Heatmap")
plt.show()

# ğŸ“ˆ Violin plots: Compare distributions by target class
import matplotlib.gridspec as gridspec

plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 3)

for idx, col in enumerate(X_train.columns):
    ax = plt.subplot(gs[idx])
    sns.violinplot(x=y_train, y=X_train[col], palette='Set2')
    ax.set_title(f"{col} by Personality")
    ax.set_xlabel("Personality (0 = Introvert, 1 = Extrovert)")

plt.tight_layout()
plt.show()



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import numpy as np

# ğŸ”§ Set number of folds
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# ğŸ§  Prepare storage for OOF and test predictions
oof_preds_cat = np.zeros(len(X_train))
test_preds_cat = np.zeros(len(X_test))
fold_scores = []

# ğŸ§ª Start Cross-Validation
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nğŸ“‚CatBoost Fold {fold + 1}")

    # ğŸ“¤ Split data
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # ğŸ§  Initialize CatBoost
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50
    )

    # ğŸ�‹ï¸� Train model
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

    # ğŸ�¯ Predict
    oof_preds_cat[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds_cat += model.predict_proba(X_test)[:, 1] / N_SPLITS

    # ğŸ“ˆ Evaluate
    score = roc_auc_score(y_val, oof_preds_cat[val_idx])
    fold_scores.append(score)
    print(f"âœ…CatBoot Fold {fold + 1} AUC: {score:.4f}")

# ğŸ“Š Final AUC on full training set
overall_auc = roc_auc_score(y_train, oof_preds_cat)
print(f"\nğŸ�¯ CatBoost Overall OOF AUC: {overall_auc:.4f}")



from xgboost import XGBClassifier

# ğŸ§  Setup for XGB predictions
oof_preds_xgb = np.zeros(len(X_train))
test_preds_xgb = np.zeros(len(X_test))

# ğŸ”� Train XGB using Stratified K-Fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nğŸ“‚ XGB Fold {fold + 1}")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model_xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42
    )

    model_xgb.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=50,
                  verbose=False)

    oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / N_SPLITS

    score = roc_auc_score(y_val, oof_preds_xgb[val_idx])
    print(f"âœ… XGB Fold {fold + 1} AUC: {score:.4f}")

# ğŸ�¯ XGB Overall OOF AUC
overall_xgb_auc = roc_auc_score(y_train, oof_preds_xgb)
print(f"\nğŸ�¯ XGB Overall OOF AUC: {overall_xgb_auc:.4f}")



from lightgbm import LGBMClassifier

# ğŸ§  Setup for LGBM predictions
oof_preds_lgbm = np.zeros(len(X_train))
test_preds_lgbm = np.zeros(len(X_test))

# ğŸ”� Train LightGBM using Stratified K-Fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nğŸ“‚ LGBM Fold {fold + 1}")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model_lgbm = LGBMClassifier(
        n_estimators=1000,
        verbosity = -1,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        metric='auc'
    )

    # ğŸ�‹ï¸� Train without early stopping
    model_lgbm.fit(X_tr, y_tr)

    oof_preds_lgbm[val_idx] = model_lgbm.predict_proba(X_val)[:, 1]
    test_preds_lgbm += model_lgbm.predict_proba(X_test)[:, 1] / N_SPLITS

    score = roc_auc_score(y_val, oof_preds_lgbm[val_idx])
    print(f"âœ… LGBM Fold {fold + 1} AUC: {score:.4f}")

# ğŸ�¯ LGBM Overall OOF AUC
overall_lgbm_auc = roc_auc_score(y_train, oof_preds_lgbm)
print(f"\nğŸ�¯ LGBM Overall OOF AUC: {overall_lgbm_auc:.4f}")



from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ğŸ“¦ Stack base model predictions into dataframe
df_meta_train = pd.DataFrame({
    'cat': oof_preds_cat,
    'xgb': oof_preds_xgb,
    'lgbm': oof_preds_lgbm
})

df_meta_test = pd.DataFrame({
    'cat': test_preds_cat,
    'xgb': test_preds_xgb,
    'lgbm': test_preds_lgbm
})

# â�• Add derived features
def enrich_meta(df):
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    df['max'] = df.max(axis=1)
    df['min'] = df.min(axis=1)
    df['range'] = df['max'] - df['min']
    return df

X_meta_train = enrich_meta(df_meta_train.copy())
X_meta_test = enrich_meta(df_meta_test.copy())

# (Optional) â�• Add original numerical features to meta-train/test
# Only add if scaling/handling is done!
X_meta_train = pd.concat([X_meta_train, X_train.reset_index(drop=True)], axis=1)
X_meta_test = pd.concat([X_meta_test, X_test.reset_index(drop=True)], axis=1)

# ğŸ§  Train CatBoost stacker
meta_model = CatBoostClassifier(
    iterations=700,
    learning_rate=0.03,
    depth=6,
    random_state=42,
    verbose=0
)

meta_model.fit(X_meta_train, y_train)

# ğŸ”® Predict final test probabilities
final_test_preds = meta_model.predict_proba(X_meta_test)[:, 1]

# ğŸ�¯ Evaluate Stacked AUC
stacked_auc = roc_auc_score(y_train, meta_model.predict_proba(X_meta_train)[:, 1])
print(f"\nğŸ”¥ Final Stacked AUC (w/ engineered features): {stacked_auc:.4f}")



from sklearn.metrics import accuracy_score
import numpy as np

best_acc = 0
best_thresh = 0.5

# Try thresholds from 0.3 to 0.7
for thresh in np.arange(0.3, 0.71, 0.01):
    preds = (meta_model.predict_proba(X_meta_train)[:, 1] >= thresh).astype(int)
    acc = accuracy_score(y_train, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"âœ… Best Accuracy: {best_acc:.4f} at threshold: {best_thresh:.2f}")



# ğŸ”� Visualize prediction distribution
plt.figure(figsize=(8, 4))
sns.histplot(final_test_preds, bins=50, kde=True, color='cornflowerblue')
plt.title("ğŸ“Š Final Stacked Predictions Distribution (CatBoost)", fontsize=14)
plt.xlabel("Probability of Extrovert")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()



# ğŸ”� Convert final probabilities into labels
final_preds_binary = (final_test_preds >= 0.33).astype(int)

# ğŸ”� Map labels to personality classes
label_map = {0: 'Introvert', 1: 'Extrovert'}
final_labels = pd.Series(final_preds_binary).map(label_map)

# ğŸ“¤ Create submission dataframe
submission = pd.DataFrame({
    'id': test_ids,
    'Personality': final_labels
})

# ğŸ’¾ Save the file
submission.to_csv("submission.csv", index=False)
print("âœ… Final submission.csv generated!")



submission




