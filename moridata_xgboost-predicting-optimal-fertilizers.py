# ===== Suppress Warnings =====
import warnings
warnings.filterwarnings("ignore")

# ===== Core Libraries =====
import numpy as np
import pandas as pd
from collections import Counter

# ===== Visualization Libraries =====
import matplotlib.pyplot as plt   # Plotting
import seaborn as sns             # Enhanced plotting styles

# ===== Preprocessing =====
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# ===== Modeling =====
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier

# ===== Evaluation and Cross-Validation =====
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


# Add 'id' to the second dataset starting from len(df1)
original.insert(0, 'id', range(len(test), len(test) + len(original)))


original_copy = original.copy()
for _ in range(6):
    original = pd.concat([original, original_copy], axis=0)


# Add a 'dataset' column to track source
train['dataset'] = 'train'
test['dataset'] = 'test'

original['dataset'] = 'train'

# Combine train and test datasets for unified preprocessing
df = pd.concat([test, original, train ], axis=0).reset_index(drop=True)


train


original


test


df


df.shape


# 2. Preview the first 5 rows to verify columns and 'dataset' marker
df.head()


df.info()


df.describe()


train.isnull().sum()


# ===== Visualize Distribution of Numerical Features =====

# Make sure 'train_df' is defined as the training subset (750,000 rows)
# For example:
train_df = df[df['dataset'] == 'train'].copy()

num_feats = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

for col in num_feats:
    plt.figure(figsize=(6, 4))
    sns.histplot(train_df[col], kde=True, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Print descriptive statistics
    print(f'\nğŸ“Š Descriptive Stats for {col}:\n')
    print(train_df[col].describe(), '\n' + '-'*40)


# ===== Visualize Distribution of Categorical Features =====

# Assume `train_df` is already defined as:
# train_df = df[df['dataset'] == 'train'].copy()

cat_feats = ['Soil Type', 'Crop Type']

for col in cat_feats:
    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=train_df,
        x=col,
        order=train_df[col].value_counts().index,
        palette='Set2',
        edgecolor='black'
    )
    plt.title(f'{col} Distribution', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print(f'\nğŸ“Š Proportion of Each Category in "{col}":\n')
    print(train_df[col].value_counts(normalize=True).round(3), '\n' + '-'*40)


# ===== Target Variable Distribution =====

# Ensure 'train_df' is defined as the training subset (750,000 rows)
# For example:
# train_df = df[df['dataset'] == 'train'].copy()

plt.figure(figsize=(8, 5))
sns.countplot(
    data=train_df,
    x='Fertilizer Name',
    palette='coolwarm',
    edgecolor='black',
    order=train_df['Fertilizer Name'].value_counts().index
)
plt.title('Distribution of Fertilizer Name (All 7 Classes)', fontsize=14)
plt.xlabel('Fertilizer Name', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Display normalized value counts (as proportions)
print("\nğŸ“Š Fertilizer Name Value Counts (Proportions):")
print(train_df['Fertilizer Name']
      .value_counts(normalize=True)
      .round(3))


# ===== Categorical Feature Distributions by Fertilizer Name =====

# Ensure 'train_df' is defined as the training subset:
# train_df = df[df['dataset'] == 'train'].copy()

cat_feats = ['Soil Type', 'Crop Type']

for col in cat_feats:
    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=train_df,
        x=col,
        hue='Fertilizer Name',
        palette='Set1',
        edgecolor='black'
    )
    plt.title(f'{col} by Fertilizer Name', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Fertilizer Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Print proportions of fertilizer distribution within each category
    print(f'\nğŸ“Š Proportions of Fertilizer within "{col}":\n')
    prop_table = train_df.groupby(col)['Fertilizer Name'].value_counts(normalize=True).unstack().round(3)
    print(prop_table, '\n' + '-'*50)


# ===== Numerical Feature Distributions by Fertilizer Name (Boxplots) =====

# Ensure 'train_df' is defined as the training subset:
# train_df = df[df['dataset'] == 'train'].copy()

numeric_feats = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

for col in numeric_feats:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=train_df, x='Fertilizer Name', y=col, palette='Set3')
    plt.title(f'{col} by Fertilizer Name', fontsize=14)
    plt.xlabel('Fertilizer Name', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ===== Correlation Matrix for Numerical Features =====

# Ensure 'train_df' is defined as the training subset:
# train_df = df[df['dataset'] == 'train'].copy()

num_feats = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

plt.figure(figsize=(6, 4))
sns.heatmap(
    train_df[num_feats].corr(),
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidths=0.5,
    square=True,
    cbar_kws={'shrink': 0.75}
)
plt.title('ğŸ”— Correlation Matrix of Numerical Features', fontsize=14)
plt.tight_layout()
plt.show()


###################


# Handle Missing Values

train.isnull().sum()


# Categorical Encoding

# Numerical and Categorical columns
num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
cat_cols = ['Soil Type', 'Crop Type']

# # One-Hot Encoding for Categorical Features
# df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

#Label Encode the Target
le = LabelEncoder()
train_mask = df['dataset'] == 'train'
df.loc[train_mask, 'Fertilizer Name'] = le.fit_transform(df.loc[train_mask, 'Fertilizer Name'])


# Separate train and test datasets
train_df = df[df['dataset'] == 'train'].drop(columns=['dataset'], errors='ignore')
test_df  = df[df['dataset'] == 'test'].drop(columns =['dataset'], errors='ignore')


# Drop unnecessary columns from both datasets
train_df = train_df.drop(columns=['id'], errors='ignore')
test_df = test_df.drop(columns=['Fertilizer Name'], errors='ignore')



# Separate features and target
X = train_df.drop(['Fertilizer Name'], axis=1)
y = train_df['Fertilizer Name']

y = y.astype(int)

#Split Strategy
#K-Fold Cross-Validation


cat_cols = [
    col for col in X.select_dtypes(include=['object', 'category']).columns
    if col != "Fertilizer Name"
]


# ========== Evaluate with MAP@3 ==========
def mapk(actual, predicted, k=3):
    """Compute mean average precision at k (MAP@k)."""
    def apk(a, p, k):
        score = 0.0
        for i in range(min(k, len(p))):
            if p[i] == a:
                score += 1.0 / (i + 1)
                break  # only the first correct prediction counts
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# # Optionally encode categorical columns if needed
X_enc = X.copy()
for col in cat_cols:
    X_enc[col] = X_enc[col].astype("category").cat.codes
    test_df[col] = test_df[col].astype("category").cat.codes
    
    

# 1. Compute class_weights globally (can also be per fold)
# counter_full = Counter(y)
# max_count_full = max(counter_full.values())
# class_weights_full = {cls: max_count_full / count for cls, count in counter_full.items()}

# 2. Stratified CV
kfold = StratifiedKFold(n_splits=13, shuffle=True, random_state=42)
fold_accuracies = []
oof_preds = np.zeros((X.shape[0], len(np.unique(y))))

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_enc, y), 1):
    print(f"\n================ Fold {fold} ================")

    X_tr, X_va = X_enc.iloc[train_idx], X_enc.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    # 3. Compute per-instance weights
    counter_fold = Counter(y_tr)
    max_count_fold = max(counter_fold.values())
    sample_weights = y_tr.map(lambda cls: max_count_fold / counter_fold[cls])

    # 4. Instantiate XGBoost model
    XGB_model = XGBClassifier(
        objective='multi:softprob',
    num_class=len(np.unique(y)),
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    max_bin=128,
    colsample_bytree=0.3,
    colsample_bylevel=1,
    colsample_bynode=1,
    tree_method='gpu_hist',            # or 'gpu_hist' if using GPU
    random_state=42,
    eval_metric='mlogloss',
    device='cuda',                 # ensure XGBoost >= 1.7 for 'device' param
    enable_categorical=True,       # needs pandas category dtype
    n_estimators=5000,
    early_stopping_rounds=150,
    use_label_encoder=False        # recommended to disable for newer versions

    )

    # 5. Fit with early stopping
    XGB_model.fit(
        X_tr,
        y_tr,
        sample_weight=sample_weights,
        eval_set=[(X_va, y_va)],
        # early_stopping_rounds=150,
        verbose=200,
    )

    val_labels = XGB_model.predict(X_va)
    val_probas = XGB_model.predict_proba(X_va)

    oof_preds[val_idx] = val_probas
    acc = accuracy_score(y_va, val_labels)
    fold_accuracies.append(acc)
    print(f"âœ… Fold {fold} Accuracy: {acc:.4f}")

# 6. Final CV metrics
print("\nğŸ�¯ Mean CV Accuracy:", np.mean(fold_accuracies))
print("ğŸ“ˆ Std CV Accuracy:", np.std(fold_accuracies))

# Get Top-3 predicted class indices
top3_preds = np.argsort(oof_preds, axis=1)[:, ::-1][:, :3]

# Calculate MAP@3
map3_score = mapk(y.values, top3_preds, k=3)
print(f"\nğŸ“Š Mean Average Precision @3 (MAP@3): {map3_score:.5f}")


# Assume: `model` is already trained using CatBoost
# Prepare test features by dropping the 'id' column if it exists
test_features = test_df.drop(columns=['id'], errors='ignore')

# Predict class probabilities
# probs = model.predict_proba(test_features)
probs = XGB_model.predict_proba(test_features)


# # Get top 3 predictions per sample
top_3_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # Top 3 indices, reversed (high to low)

# Decode labels
top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

# Build submission dataframe
submission = pd.DataFrame({
    'id': test_df['id'],  # if 'id' exists in test_df
    'Fertilizer Name': [' '.join(preds) for preds in top_3_labels]
})

# Save submission
submission.to_csv('submission.csv', index=False)


submission

