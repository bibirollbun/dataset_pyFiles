import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# change if you put the csv files somewhere else
DATA_DIR = Path('/kaggle/input/playground-series-s5e7')

train_path = DATA_DIR / 'train.csv'
test_path  = DATA_DIR / 'test.csv'

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

print(f"train shape: {train.shape} | test shape: {test.shape}")
train.head()

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


train.info()
print('\nMissingâ€‘value ratio (%):')
(train.isna().mean()*100).sort_values(ascending=False)


# â•”â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•—
# â•‘   Balance data: drop rows that are BOTH Extrovert & have NAs   â•‘
# â•šâ•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# Count rows to be removed
extro_missing_mask = (train['Personality'] == 'Extrovert') & train.isna().any(axis=1)
n_remove           = extro_missing_mask.sum()

print(f"Dropping {n_remove} Extrovert rows with at least one missing value...")

# Drop them in-place
train = train.loc[~extro_missing_mask].reset_index(drop=True)

print(f"New class counts:\n{train['Personality'].value_counts()}")


num_cols = train.select_dtypes(include=['int64','float64']).columns.drop('id', errors='ignore')
fig, axes = plt.subplots(len(num_cols), 1, figsize=(6, 3*len(num_cols)))
for ax, col in zip(axes, num_cols):
    sns.histplot(train[col], ax=ax, kde=True)
    ax.set_title(col)
plt.tight_layout()


num_cols = train.select_dtypes(include=['int64','float64']).columns.drop('id', errors='ignore')
fig, axes = plt.subplots(len(num_cols), 1, figsize=(6, 3*len(num_cols)))
for ax, col in zip(axes, num_cols):
    sns.histplot(train[col], ax=ax, kde=True)
    ax.set_title(col)
plt.tight_layout()


sns.countplot(x='Personality', data=train)
plt.title('Target distribution')
plt.show()


# â•”â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•—
# â•‘      COMPARATIVE EDA â€“ INTROVERT  vs.  EXTROVERT            â•‘
# â•šâ•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Split once for reuse
intro  = train[train['Personality'] == 'Introvert']
extro  = train[train['Personality'] == 'Extrovert']

# 1ï¸�âƒ£  Class balance
plt.figure(figsize=(4,3))
sns.countplot(x='Personality', data=train, palette='pastel')
plt.title('Class distribution')
plt.show()

# 2ï¸�âƒ£  Numeric features sideâ€‘byâ€‘side
num_cols = train.select_dtypes(include=['int64','float64']).columns.drop(['id'], errors='ignore')

for col in num_cols:
    fig, axes = plt.subplots(1, 2, figsize=(8,3), sharex=True, sharey=True)
    sns.histplot(intro[col], ax=axes[0], kde=True, color='#5E81AC')
    axes[0].set_title(f'Introvert â€“ {col}')
    sns.histplot(extro[col], ax=axes[1], kde=True, color='#A3BE8C')
    axes[1].set_title(f'Extrovert â€“ {col}')
    plt.tight_layout()
    plt.show()

# 3ï¸�âƒ£  Numeric feature means & stds (tidy table)
summary = (
    train.groupby('Personality')[num_cols]
         .agg(['mean', 'std'])
         .T                               # flip for readability
)
summary.index.set_names(['feature', 'stat'], inplace=True)   # â†� fix
display(summary.head(10))

# 4ï¸�âƒ£  Categorical features â€“ stacked bars
cat_cols = train.select_dtypes(include='object').columns.drop(['Personality','id'], errors='ignore')

for col in cat_cols:
    ct = pd.crosstab(train[col], train['Personality'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, figsize=(6,3), color=['#5E81AC','#A3BE8C'])
    plt.ylabel('% within category')
    plt.title(col)
    plt.legend(title='Personality', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



corr = train[num_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Numeric feature correlation')
plt.show()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns to train OR test DataFrame and return it."""
    
    # 1ï¸�âƒ£  Binary flags for Yes/No questions
    df['Stage_fear_bin']  = (df['Stage_fear']              == 'Yes').astype(int)
    df['Drained_bin']     = (df['Drained_after_socializing'] == 'Yes').astype(int)
    
    # 2ï¸�âƒ£  Socialâ€‘anxiety score (0,1,2)
    df['Social_anxiety_score'] = df['Stage_fear_bin'] + df['Drained_bin']
    
    # 3ï¸�âƒ£  Total outward activity
    social_cols = ['Social_event_attendance', 'Going_outside', 'Post_frequency']
    df['Total_social_activity'] = df[social_cols].sum(axis=1)
    
    # 4ï¸�âƒ£  Aloneâ€‘time ratios
    df['Alone_ratio']      = df['Time_spent_Alone'] / (df['Total_social_activity'] + 1)
    df['Alone_per_friend'] = df['Time_spent_Alone'] / (df['Friends_circle_size'] + 1)
    
    # 5ï¸�âƒ£  Rowâ€‘wise missing count
    df['missing_count'] = df.isna().sum(axis=1)
    
    return df

# Apply to both datasets
train = add_features(train)
test  = add_features(test)




num_cols1 = train.select_dtypes(include=['int64','float64']).columns.drop('id', errors='ignore')

orr = train[num_cols1].corr()
plt.figure(figsize=(8,6))
sns.heatmap(orr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Numeric feature correlation after Feature Engineering')
plt.show()


train.head()


# âœ±âœ± PREâ€‘PROCESSING & MODEL PIPELINE  âœ±âœ±
# --------------------------------------
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute       import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble      import RandomForestRegressor, GradientBoostingClassifier
from sklearn.compose       import ColumnTransformer
from sklearn.pipeline      import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics       import accuracy_score, classification_report

# ----------------------------------------------------------
X = train.drop(columns=['Personality'])   # keep 'id' for now
y = train['Personality']

# Separate dtypes first (they may still include 'id')
num_all  = X.select_dtypes(include=['int64', 'float64']).columns
cat_all  = X.select_dtypes(include=['object']).columns

# Remove 'id' from those lists if present
numeric_features     = [c for c in num_all if c != 'id']
categorical_features = [c for c in cat_all if c != 'id']

# 1ï¸�âƒ£  NUMERIC PIPELINE  (RF imputer + Standard scaler)
numeric_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        max_iter=10,
        initial_strategy='median',
        random_state=42
    )),
    ('scaler', StandardScaler())
])

# 2ï¸�âƒ£  CATEGORICAL PIPELINE  (fill with "unknown")
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# 3ï¸�âƒ£  COLUMNWISE COMPOSITION  â�œ drop 'id' inside the transformer
preprocess = ColumnTransformer(
    transformers=[
        ('drop_id', 'drop',            ['id']),     # <â”€â”€ here
        ('num',    numeric_transformer, numeric_features),
        ('cat',    categorical_transformer, categorical_features)
    ])

# 4ï¸�âƒ£  MODEL
model = GradientBoostingClassifier(random_state=42)

clf = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model',       model)
])

# quick holdâ€‘out sanity check
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf.fit(X_train, y_train)
preds_valid = clf.predict(X_valid)
print("Validation accuracy:", accuracy_score(y_valid, preds_valid))
# Align indices (train_test_split keeps original indices)
Xv   = X_valid.reset_index(drop=True)
yv   = y_valid.reset_index(drop=True)

mis_mask = preds_valid != yv
mis_df   = Xv.loc[mis_mask].copy()

mis_df['true_label'] = yv[mis_mask].values
mis_df['pred_label'] = preds_valid[mis_mask]

# Optional: keep the row index from the original dataset as a column
# mis_df['orig_index'] = y_valid.index[mis_mask]

outfile = "misclassified_valid_samples.csv"
mis_df.to_csv(outfile, index=False)
print(f"ğŸ“�  Saved {len(mis_df)} misâ€‘classified validation rows â†’ {outfile}")

print("\nClassification report:\n", classification_report(y_valid, preds_valid))



cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print('CV accuracy mean Â± std: %.4f Â± %.4f' % (cv_scores.mean(), cv_scores.std()))


clf.fit(X, y)
test_preds = clf.predict(test)

submission = pd.DataFrame({
    'id': test['id'],
    'Personality': test_preds
})
submission.to_csv('submission.csv', index=False)
print('Saved submission.csv')




