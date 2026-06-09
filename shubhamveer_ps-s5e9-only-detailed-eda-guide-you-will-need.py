
# === Imports & Setup ===
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to use seaborn if available (optional aesthetics)
try:
    import seaborn as sns
    sns.set_theme(context="notebook", style="whitegrid")
    HAS_SNS = True
except Exception:
    HAS_SNS = False

# Plotting defaults
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

DATA_DIR = '/kaggle/input/playground-series-s5e9'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
TARGET = 'BeatsPerMinute'

RANDOM_STATE = 42

print(f"Seaborn available: {HAS_SNS}")
print(f"Train path: {TRAIN_PATH}\nTest path: {TEST_PATH}")




train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print('train.shape:', train.shape, '| test.shape:', test.shape)
display(train.head())
display(test.head())




# Dtypes
print('Train dtypes:')
print(train.dtypes)

# Missingness
train_na = train.isna().sum().sort_values(ascending=False)
test_na = test.isna().sum().sort_values(ascending=False)

print('\nTop missing (train):')
display(train_na.head(20).to_frame('missing_count'))

print('\nTop missing (test):')
display(test_na.head(20).to_frame('missing_count'))

# Basic stats
display(train.describe().T)




y = train[TARGET]

# Histogram
plt.figure()
plt.hist(y, bins=50)
plt.title('Target Distribution: BeatsPerMinute')
plt.xlabel('BeatsPerMinute')
plt.ylabel('Count')
plt.show()

# Boxplot (outliers)
plt.figure()
plt.boxplot(y, vert=True, labels=['BeatsPerMinute'])
plt.title('Target Boxplot: BeatsPerMinute')
plt.ylabel('BPM')
plt.show()

print('Target summary:')
display(y.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).to_frame('BPM'))




# Separate features (exclude id and target)
num_features = [c for c in train.columns if c not in ['id', TARGET]]
id_col = 'id'

print('Numeric features:', num_features)

def numeric_histograms(df, cols, ncols=3, bins=40, title='Numeric Feature Distributions'):
    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*3.5))
    axes = axes.ravel()

    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=bins)
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
    # Hide any unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    fig.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

def scatter_against_target(df, cols, target, ncols=3):
    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*3.5))
    axes = axes.ravel()

    for i, col in enumerate(cols):
        axes[i].scatter(df[col], df[target], s=5, alpha=0.5)
        axes[i].set_title(f'{col} vs {target}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()




numeric_histograms(train, num_features, ncols=3, bins=40, title='Numeric Feature Distributions (Train)')




scatter_against_target(train, num_features, TARGET, ncols=3)




corr = train[num_features + [TARGET]].corr(numeric_only=True)

# Show top correlations with target
tgt_corr = corr[TARGET].drop(TARGET).sort_values(ascending=False)
print('Top positive correlations with target:')
display(tgt_corr.head(10).to_frame('corr'))
print('Top negative correlations with target:')
display(tgt_corr.tail(10).to_frame('corr'))

# Heatmap (matplotlib imshow)
plt.figure(figsize=(9,7))
im = plt.imshow(corr.values, aspect='auto', interpolation='nearest')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()




def binned_target_plot(df, feature, target, bins=10):
    s = pd.qcut(df[feature], q=bins, duplicates='drop')
    tmp = df.groupby(s)[target].agg(['count','mean','median','std']).reset_index()
    tmp['bin'] = tmp[feature].astype(str)
    plt.figure(figsize=(8,4))
    plt.plot(range(len(tmp)), tmp['mean'], marker='o')
    plt.xticks(range(len(tmp)), tmp['bin'], rotation=45, ha='right')
    plt.xlabel(f'{feature} (quantile bins)')
    plt.ylabel(f'Mean {target}')
    plt.title(f'{target} vs binned {feature} (mean)')
    plt.tight_layout()
    plt.show()
    return tmp

binned_summaries = {}
for col in num_features:
    print(f'\n=== {col} ===')
    try:
        summary = binned_target_plot(train, col, TARGET, bins=10)
        binned_summaries[col] = summary
        display(summary.head())
    except Exception as e:
        print('Skipped due to error:', e)




# Boxplots for numeric features
ncols = 3
n = len(num_features)
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*3.5))
axes = axes.ravel()

for i, col in enumerate(num_features):
    axes[i].boxplot(train[col].dropna(), vert=True)
    axes[i].set_title(col)
    axes[i].set_ylabel(col)
for j in range(i+1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.show()

# Example: log transform candidates (positive-only)
log_candidates = [c for c in num_features if (train[c] > 0).all()]
print('Log-transform candidates (all > 0):', log_candidates[:10])

# Visual check for a couple of features
for col in log_candidates[:4]:
    plt.figure()
    plt.hist(np.log1p(train[col].values), bins=50)
    plt.title(f'log1p({col}) distribution')
    plt.xlabel(f'log1p({col})')
    plt.ylabel('Count')
    plt.show()




def train_test_compare(train, test, cols, ncols=3, bins=40):
    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*3.5))
    axes = axes.ravel()

    for i, col in enumerate(cols):
        axes[i].hist(train[col].dropna(), bins=bins, alpha=0.5, label='train')
        axes[i].hist(test[col].dropna(), bins=bins, alpha=0.5, label='test')
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].legend()
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

common_cols = [c for c in test.columns if c in train.columns and c not in [TARGET]]
train_test_compare(train, test, common_cols, ncols=3, bins=40)




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

X = train.drop(columns=[TARGET, 'id'], errors='ignore')
y = train[TARGET]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_valid_s = scaler.transform(X_valid)

model = Ridge(max_iter=1000, alpha=1.0, random_state=RANDOM_STATE)
model.fit(X_train_s, y_train)
preds = model.predict(X_valid_s)

mae = mean_absolute_error(y_valid, preds)
print(f'Validation MAE (Ridge + StandardScaler): {mae:.4f}')



full_X = train.drop(columns=[TARGET, 'id'], errors='ignore')
full_y = train[TARGET]
scaler_full = StandardScaler().fit(full_X)
full_model = Ridge(max_iter=1000, alpha=1.0, random_state=RANDOM_STATE).fit(scaler_full.transform(full_X), full_y)

test_X = test.drop(columns=['id'], errors='ignore')
test_pred = full_model.predict(scaler_full.transform(test_X))

sub = pd.DataFrame({'id': test['id'], TARGET: test_pred})
sub_path = 'submission.csv'
sub.to_csv(sub_path, index=False)
print('Saved:', sub_path)
sub.head()





