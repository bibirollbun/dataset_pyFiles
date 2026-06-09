import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
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


complete_data = pd.concat([train, test])


label_encoders = {}
category_mapping = {}

for col in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    complete_data[col] = le.fit_transform(complete_data[col].astype(str))
    label_encoders[col] = le
    category_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))



category_mapping


train = complete_data.iloc[:len(train)] 
test = complete_data.iloc[len(train):]


test = test.drop(['Fertilizer Name'],axis=1)


# Select only the first 75,000 rows
X = train.drop(['Fertilizer Name'], axis=1)
y = train['Fertilizer Name']



le = LabelEncoder()
y = le.fit_transform(y)
# Mapping from original class labels to encoded integers
y_transform = dict(zip(le.classes_, range(len(le.classes_))))
y_transform


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


params = {
    "boosting_type": "gbdt",
    "device": "gpu",
    "colsample_bytree": 0.436,
    "learning_rate": 0.0165,
    "max_depth": 12,
    "min_child_samples": 67,
    "n_estimators": 10000,
    "n_jobs": -1,
    "num_leaves": 243,
    "random_state": 42,
    "reg_alpha": 6.38,
    "reg_lambda": 9.39,
    "subsample": 0.799,
    "verbose": -1
}

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
    
fold_map3_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ”� Fold {fold+1}")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_pred_probs = model.predict_proba(X_valid)
    top3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]
    map3_score = mapk(y_valid.tolist(), [list(p) for p in top3_preds], k=3)

    fold_map3_scores.append(map3_score)
    print(f"ğŸ“ˆ MAP@3 Score (Fold {fold+1}): {map3_score:.5f}")

# Final average
avg_map3 = np.mean(fold_map3_scores)
print(f"\nâœ… Average MAP@3 across all folds: {avg_map3:.5f}")


test_pred_probs = model.predict_proba(test)


# 2. Get top-3 predicted class indices
top3_indices = np.argsort(test_pred_probs, axis=1)[:, -3:][:, ::-1]

# 3. Ensure labels are strings
fertilizer_labels = [str(label) for label in model.classes_]


top3_indices


# 4. Map indices to fertilizer names
top3_names = [[fertilizer_labels[i] for i in row] for row in top3_indices]


# Create space-separated strings
top3_strings = [' '.join(row) for row in top3_names]
# 1. Reverse the label mapping
id_to_name = {v: k for k, v in y_transform.items()}


top3_strings_named = []
for row in top3_strings:
    indices = [int(i) for i in row.split()]
    names = [id_to_name[i] for i in indices]
    top3_strings_named.append(' '.join(names))

#  Create submission DataFrame using test.index
submission = pd.DataFrame({
    'id': test.index,
    'Fertilizer Name': top3_strings_named
})


submission.head()


# 4. Save to CSV
submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv with label names generated!")




