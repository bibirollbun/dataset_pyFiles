# ─── Data Manipulation ─────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd

# ─── Data Visualization ───────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Preprocessing & Feature Engineering ────────────────────────────────────
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.decomposition import PCA
from sklearn.compose       import ColumnTransformer

# ─── Model Selection & Pipeline ──────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline        import Pipeline

# ─── Classifiers ─────────────────────────────────────────────────────────────
from sklearn.tree     import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.head()


test.head()


(train['Fertilizer Name'].value_counts(normalize=True) * 100).round(2)


# make 1×3 axes instead of 1×2
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

sns.violinplot(
    data=train, 
    x="Fertilizer Name", 
    y="Temparature", 
    ax=axes[0],
    inner="quartile",
    color="forestgreen"
)
axes[0].set_title("Temperature by Fertilizer")

sns.violinplot(
    data=train, 
    x="Fertilizer Name", 
    y="Humidity", 
    ax=axes[1],
    inner="quartile",
    color="skyblue"
)
axes[1].set_title("Humidity by Fertilizer")

sns.violinplot(
    data=train, 
    x="Fertilizer Name", 
    y="Moisture", 
    ax=axes[2],
    inner="quartile",
    color="sienna"
)
axes[2].set_title("Moisture by Fertilizer")

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 7))

sns.violinplot(
    data=train, 
    x="Fertilizer Name", 
    y="Nitrogen", 
    ax=axes[0],
    inner="quartile",
    color="olive"
)
axes[0].set_title("Nitrogen by Fertilizer")

sns.violinplot(
    data=train, 
    x="Fertilizer Name", 
    y="Potassium", 
    ax=axes[1],
    inner="quartile",
    color="teal"
)
axes[1].set_title("Potassium by Fertilizer")

sns.violinplot(
    data=train, 
    x="Fertilizer Name", 
    y="Phosphorous", 
    ax=axes[2],
    inner="quartile",
    color="peru"
)
axes[2].set_title("Phosphorous by Fertilizer")

plt.tight_layout()
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Soil Type counts by fertilizer
sns.countplot(
    data=train,
    x="Soil Type",
    hue="Fertilizer Name",
    ax=ax1
)
ax1.set_title("Soil Type Distribution by Fertilizer")
ax1.tick_params(axis='x', rotation=45)

ax1.legend(
    title='Fertilizer Name',
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

# Crop Type counts by fertilizer
sns.countplot(
    data=train,
    x="Crop Type",
    hue="Fertilizer Name",
    ax=ax2
)
ax2.set_title("Crop Type Distribution by Fertilizer")
ax2.tick_params(axis='x', rotation=45)

ax2.legend(
    title='Fertilizer Name',
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.tight_layout()
plt.show()


from typing import List, Any

def apk(actual: List[Any], predicted: List[Any], k: int = 3) -> float:
    """
    Computes the average precision at k between two lists:
      actual    - list of ground-truth items (order doesn’t matter)
      predicted - list of predicted items (ordered by confidence)
      k         - maximum number of predicted elements to consider

    Returns:
      score in [0.0, 1.0]
    """
    if not actual:
        return 0.0

    # truncate predictions to top k
    pred = predicted[:k]

    score = 0.0
    hits  = 0.0

    for i, p in enumerate(pred, start=1):
        if p in actual and p not in pred[: i-1]:
            hits += 1.0
            score += hits / i

    return score / min(len(actual), k)


def mapk(actuals: List[List[Any]], predictions: List[List[Any]], k: int = 3) -> float:
    """
    Computes the mean average precision at k over all samples.
      actuals     - list of lists of ground-truth items
      predictions - list of lists of predicted items
      k           - as in apk

    Returns:
      mean apk over all samples
    """
    return sum(apk(a, p, k) for a, p in zip(actuals, predictions)) / len(actuals)

def selecting_top_3(arr): 
    n = arr.shape[0]
    out = np.zeros((n, 3))
    for i in range(0, n):
        out[i, ] = arr[i,].argsort()[::-1][:3]
    return out.astype('int32')



X = train.drop(columns=["Fertilizer Name"])
y = train["Fertilizer Name"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

cat_cols = ["Soil Type", "Crop Type"]
num_cols = [c for c in X.columns if c not in cat_cols]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(),num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)


et_pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])


fold_scores, test_preds = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), start=1):
    # Split
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # Fit
    et_pipeline.fit(X_train, y_train)

    # Predict top-3
    proba = et_pipeline.predict_proba(X_val)
    top3 = [list(le.inverse_transform(np.argsort(p)[::-1][:3])) for p in proba]
    actuals = [[lbl] for lbl in le.inverse_transform(y_val)]

    # Score
    score = mapk(actuals, top3, k=3)
    fold_scores.append(score)
    print(f"Fold {fold} MAP@3: {score:.4f}")

    test_preds.append(et_pipeline.predict_proba(test))

print(f"\nAverage MAP@3 over {len(fold_scores)} folds: {np.mean(fold_scores):.4f}")


# 1) Aggregate your list of probability arrays into one mean array
pred_agg = np.mean(np.stack(test_preds, axis=0), axis=0)

# 2) Pick the top-3 class indices for each row
test_pred = selecting_top_3(pred_agg).astype(np.int32)
#    now test_pred.shape == (n_samples, 3)

# 3) Flatten to 1-D, invert labels, and reshape back
flat_preds = test_pred.ravel()  
flat_names = le.inverse_transform(flat_preds)  
top_3_predictions = flat_names.reshape(test_pred.shape)
#    shape: (n_samples, 3) of string fertilizer names

# 4) Load the sample submission (id is in the index), assign your predictions
submission = pd.read_csv(
    '../input/playground-series-s5e6/sample_submission.csv', 
    index_col=0
)
submission['Fertilizer Name'] = [' '.join(row) for row in top_3_predictions]

# 5) Write out your submission
submission.to_csv('submission.csv')
submission.head(10)

