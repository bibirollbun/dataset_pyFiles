import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


class CFG:
    train_path = "/kaggle/input/playground-series-s5e6/train.csv"
    test_path = "/kaggle/input/playground-series-s5e6/test.csv"

    n_folds=5
    target = 'Fertilizer Name'
    seed=42


train = pd.read_csv(CFG.train_path, index_col="id")
test = pd.read_csv(CFG.test_path, index_col="id")


train.head()


test.head()


train.info()


test.info()


cat_cols = train.select_dtypes(include="object").columns.tolist()
print(cat_cols)


cat_cols.remove(CFG.target)
print(cat_cols)


train[cat_cols] = train[cat_cols].astype(str).astype("category")
test[cat_cols] = test[cat_cols].astype(str).astype("category")


numerical_data = train.select_dtypes(include="number").columns.tolist()
print(numerical_data)


# Define palettes
color_comb1 = sns.color_palette("husl", len(numerical_data))
color_comb2 = sns.color_palette("bright", len(numerical_data))

# Display palettes visually
plt.figure(figsize=(10, 1))
sns.palplot(color_comb1)
plt.title("Color Combination 1: husl")
plt.show()

plt.figure(figsize=(10, 1))
sns.palplot(color_comb2)
plt.title("Color Combination 2: bright")
plt.show()


sns.set_theme(style="whitegrid", palette="Set2")  # Elegant style

for i, feature in enumerate(numerical_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    color1 = color_comb1[i%len(numerical_data)]
    color2 = color_comb2[i%len(numerical_data)]
    # Histogram
    sns.histplot(train[feature], bins=20, color=color1, edgecolor="black", ax=axes[0])
    axes[0].set_title(f"Histogram of {feature}", fontsize=13)
    axes[0].legend()
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Frequency")

    # Boxplot
    sns.boxplot(x=train[feature], ax=axes[1], color=color2)
    axes[1].set_title(f"Boxplot of {feature}", fontsize=13)
    axes[1].set_xlabel(feature)

    plt.suptitle(f"Distribution of {feature}", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()



croptype_fertilizer = pd.crosstab(train['Crop Type'], train['Fertilizer Name'])

plt.figure(figsize=(12, 8))
sns.heatmap(croptype_fertilizer, fmt="d", annot=True, cmap="YlGnBu", linewidths=0.5)
plt.title("Frequency of Fertilizer Use Across Crop Types", fontsize=14, fontweight='bold')
plt.xlabel("Fertilizer Name", fontsize=12)
plt.ylabel("Crop Type", fontsize=12)
plt.xticks(rotation=45, color="black", ha="right")
plt.yticks(rotation=0, color="navy")

plt.tight_layout()
plt.show()



soiltype_fertilizer = pd.crosstab(train['Soil Type'], train['Fertilizer Name'])

plt.figure(figsize=(12, 8))
sns.heatmap(soiltype_fertilizer, fmt="d", annot=True, cmap="YlOrBr", linewidths=0.5)
plt.title("Frequency of Fertilizer Use Across Crop Types", fontsize=14, fontweight='bold')
plt.xlabel("Fertilizer Name", fontsize=12)
plt.ylabel("Soil Type", fontsize=12)
plt.xticks(rotation=45, color="black", ha="right")
plt.yticks(rotation=0, color="black")

plt.tight_layout()
plt.show()



complete_data = pd.concat([train, test])


complete_data[cat_cols] = complete_data[cat_cols].astype(str).astype("category")


complete_data.head()


complete_data.tail()


label_encoders = {}
category_mapping = {}

for col in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    complete_data[col] = le.fit_transform(complete_data[col].astype(str))
    label_encoders[col] = le
    category_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))



category_mapping


label_encoders


train = complete_data.iloc[:len(train)] 
test = complete_data.iloc[len(train):]


test = test.drop(['Fertilizer Name'],axis=1)


print(train.shape)
print(test.shape)


X = train.drop(['Fertilizer Name'],axis=1)
y = train['Fertilizer Name']


le = LabelEncoder()
y = le.fit_transform(y)


# Mapping from original class labels to encoded integers
y_transform = dict(zip(le.classes_, range(len(le.classes_))))



y_transform


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def xgb_objective(trial):
    accs = []
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 900, 1100),  # around 1000
            'max_depth': trial.suggest_int('max_depth', 6, 8),  # around 7
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.06),  # around 0.056
            'subsample': trial.suggest_float('subsample', 0.54, 0.58),  # around 0.5605
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.54, 0.58),  # around 0.5594
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 7),  # around 6
            'gamma': trial.suggest_float('gamma', 0.3, 0.4),  # around 0.358
            'reg_alpha': trial.suggest_float('reg_alpha', 0.9, 1.05),  # around 0.9747
            'reg_lambda': trial.suggest_float('reg_lambda', 0.68, 0.75),  # around 0.706
            'objective': 'multi:softprob',
            'num_class': 7,
            'eval_metric': 'mlogloss',
            'verbosity': 0,
            'n_jobs': -1,
            'tree_method': 'hist'  # faster training, optional
        }

        model = XGBClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        accs.append(accuracy_score(y_valid, preds))

    return np.mean(accs)



from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import numpy as np

# Best parameters from Optuna
best_params = {
    'n_estimators': 1061,
    'max_depth': 8,
    'learning_rate': 0.05056999107789834,
    'subsample': 0.5624515589730706,
    'colsample_bytree': 0.5654657637330991,
    'min_child_weight': 6,
    'gamma': 0.33627307148271546,
    'reg_alpha': 1.041251860511138,
    'reg_lambda': 0.687773700877449,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': 7,
    'verbosity': 0,
    'n_jobs': -1,
    'tree_method': 'hist'
}

# Your MAP@3 function
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]
        score = 0.0
        num_hits = 0.0
        for i, pred in enumerate(p):
            if pred == a and pred not in p[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_map3_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ”� Fold {fold+1}")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    model = XGBClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)

    y_pred_probs = model.predict_proba(X_valid)
    top3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]
    map3_score = mapk(y_valid.tolist(), [list(p) for p in top3_preds], k=3)

    fold_map3_scores.append(map3_score)
    print(f"ğŸ“ˆ MAP@3 Score (Fold {fold+1}): {map3_score:.5f}")

# Final average
avg_map3 = np.mean(fold_map3_scores)
print(f"\nâœ… Average MAP@3 across all folds: {avg_map3:.5f}")



test


# 1. Predict probabilities
test_pred_probs = model.predict_proba(test)


# 2. Get top-3 predicted class indices
top3_indices = np.argsort(test_pred_probs, axis=1)[:, -3:][:, ::-1]

# 3. Ensure labels are strings
fertilizer_labels = [str(label) for label in model.classes_]


top3_indices


# 4. Map indices to fertilizer names
top3_names = [[fertilizer_labels[i] for i in row] for row in top3_indices]


# 5. Create space-separated strings
top3_strings = [' '.join(row) for row in top3_names]


# 1. Reverse the label mapping
id_to_name = {v: k for k, v in y_transform.items()}

# 2. Convert '4 5 0' style prediction strings to fertilizer names
top3_strings_named = []
for row in top3_strings:
    indices = [int(i) for i in row.split()]
    names = [id_to_name[i] for i in indices]
    top3_strings_named.append(' '.join(names))

# 3. Create submission DataFrame using test.index
submission = pd.DataFrame({
    'id': test.index,
    'Fertilizer Name': top3_strings_named
})

# 4. Save to CSV
submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv with label names generated!")








