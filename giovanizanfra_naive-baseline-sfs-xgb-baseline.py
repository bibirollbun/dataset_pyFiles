## 1. Imports and Configuration

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from xgboost import XGBClassifier


# File paths
TRAIN_PATH = "/kaggle/input/playground-series-s5e6/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e6/test.csv"
OUTPUT_PATH = "submission.csv"

# Numerical and categorical feature names (as they appear in the CSV)
NUMERICAL_COLS   = [
    "Temparature",  # Note: original CSV has the typo "Temparature"
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
]
CATEGORICAL_COLS = ["Soil Type", "Crop Type"]
TARGET_COL       = "Fertilizer Name"



import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from xgboost import XGBClassifier

train_path = "/kaggle/input/playground-series-s5e6/train.csv"
sample_path = "/kaggle/input/playground-series-s5e6/sample_submission.csv"

train_df = pd.read_csv(train_path)
sample_df = pd.read_csv(sample_path)

NUMERICAL_COLS = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
]
CATEGORICAL_COLS = ["Soil Type", "Crop Type"]
TARGET_COL = "Fertilizer Name"


# 2.1 Load raw data
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# 2.2 Label-encode the target column in train
label_encoder = LabelEncoder()
train_df[TARGET_COL] = label_encoder.fit_transform(train_df[TARGET_COL])

# 2.3 One-Hot Encode categorical features in train
ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
ohe_train_arr = ohe.fit_transform(train_df[CATEGORICAL_COLS])
ohe_train_cols = ohe.get_feature_names_out(CATEGORICAL_COLS)

# Build a DataFrame of OHE columns for train
ohe_train_df = pd.DataFrame(
    ohe_train_arr,
    columns=ohe_train_cols,
    index=train_df.index
).astype("category")

# Concatenate numerical + OHE columns for train
train_numeric = train_df[NUMERICAL_COLS].reset_index(drop=True)
train_ohe      = ohe_train_df.reset_index(drop=True)
train_processed = pd.concat([train_df[["id", TARGET_COL]].reset_index(drop=True),
                             train_numeric, train_ohe], axis=1)

# Prepare X_train and y_train
X_train = train_processed.drop(columns=["id", TARGET_COL])
y_train = train_processed[TARGET_COL].astype(int)

# 2.4 One-Hot Encode categorical features in test (use same encoder)
ohe_test_arr = ohe.transform(test_df[CATEGORICAL_COLS])
ohe_test_df  = pd.DataFrame(
    ohe_test_arr,
    columns=ohe.get_feature_names_out(CATEGORICAL_COLS),
    index=test_df.index
).astype("category")

# Concatenate numerical + OHE columns for test
test_numeric = test_df[NUMERICAL_COLS].reset_index(drop=True)
test_ohe     = ohe_test_df.reset_index(drop=True)
test_processed = pd.concat([test_df[["id"]].reset_index(drop=True),
                            test_numeric, test_ohe], axis=1)

# Prepare X_test
X_test = test_processed.drop(columns=["id"])



# # 3.1 Define custom MAP@3 scorer for use in SFS
def mapk_scorer(estimator, X_val, y_val, k=3):
    """
    Uses estimator.predict_proba to compute MAP@k.
    y_val contains integer-encoded true labels.
    """
    probas = estimator.predict_proba(X_val)
    topk   = np.argsort(probas, axis=1)[:, -k:][:, ::-1]  # shape: (n_samples, k)
    scores = []
    for i, true_label in enumerate(y_val):
        preds = topk[i]
        score = 0.0
        hits  = 0
        seen  = set()
        for rank, p in enumerate(preds):
            if p == true_label and p not in seen:
                hits += 1
                score += hits / (rank + 1)
                seen.add(p)
        scores.append(score / 1.0)  # each actual list has length=1
    return np.mean(scores)

# # 3.2 Take a 10% random subsample for faster SFS
# X_sub = X_train.sample(frac=0.10, random_state=42)
# y_sub = y_train.loc[X_sub.index]

# # 3.3 Define a lightweight XGBClassifier for selection
# xgb_fast = XGBClassifier(
#     max_depth=4,
#     n_estimators=100,
#     learning_rate=0.2,
#     objective="multi:softprob",
#     use_label_encoder=False,
#     eval_metric="mlogloss",
#     random_state=42,
#     tree_method="hist",
#     enable_categorical=True,
# )

# # 3.4 Configure SFS to select the top-22 features (forward, no floating)
# sfs = SFS(
#     estimator=xgb_fast,
#     k_features=22,       # number of features to select
#     forward=True,
#     floating=False,
#     scoring=mapk_scorer,
#     cv=3,                # 3-fold cross-validation for speed
#     n_jobs=-1,
#     verbose=2,
# )

# # 3.5 Run the feature selector on the subsampled data
# sfs = sfs.fit(X_sub, y_sub)

# # 3.6 Extract the order of feature addition and corresponding MAP@3
# selected_order = []
# prev_set = set()
# for k in sorted(sfs.subsets_.keys()):
#     cur_set = set(sfs.subsets_[k]["feature_idx"])
#     added   = cur_set - prev_set
#     if len(added) == 1:
#         feat_idx = added.pop()
#         feat_name = X_train.columns[feat_idx]
#         avg_score = sfs.subsets_[k]["avg_score"]
#         selected_order.append((k, feat_name, avg_score))
#     prev_set = cur_set

# # 3.7 Display the selection order and MAP@3
# print("Feature | Order → Feature Name         | MAP@3")
# print("--------|-------------------------------|--------")
# for k, feat, score in selected_order:
#     print(f"{k:>2d}      → {feat:<30s}   {score:.5f}")


from sklearn.model_selection import train_test_split

# 4.1 Select only the top feature from SFS (here: "Moisture")
TOP_FEATURE = ["Moisture", "Phosphorous", "Potassium", "Soil Type_Black", "Nitrogen", "Soil Type_Sandy", "Crop Type_Sugarcane", "Temparature", "Crop Type_Oil seeds", "Crop Type_Cotton"]
X_train_top = X_train[TOP_FEATURE]
X_test_top  = X_test[TOP_FEATURE]

# # 4.2 Split out a validation set for early stopping
# X_tr, X_val, y_tr, y_val = train_test_split(
#     X_train_top,
#     y_train,
#     test_size=0.2,
#     random_state=42,
#     stratify=y_train
# )

# # 4.3 Define and train the final XGBClassifier with early stopping
# final_model = XGBClassifier(
#     max_depth=12,
#     colsample_bytree=0.467,
#     subsample=0.86,
#     n_estimators=4000,
#     learning_rate=0.03,
#     gamma=0.26,
#     max_delta_step=4,
#     reg_alpha=2.7,
#     reg_lambda=1.4,
#     early_stopping_rounds=100,
#     objective="multi:softprob",
#     random_state=13,
#     enable_categorical=True,
#     tree_method="hist",
#     device="cuda"
# )

# final_model.fit(
#     X_tr,
#     y_tr,
#     eval_set=[(X_val, y_val)],
#     verbose=50
# )

# # 4.4 Predict top-3 labels for each test row
# proba    = final_model.predict_proba(X_test_top)                       # shape: (n_test, n_classes)
# top3idx  = np.argsort(proba, axis=1)[:, -3:][:, ::-1]                  # indices of top-3 probabilities
# top3labs = label_encoder.inverse_transform(top3idx.ravel()).reshape(top3idx.shape)

# # 4.5 Build submission DataFrame
# submission_df = pd.DataFrame({
#     "id": test_df["id"],
#     "Fertilizer Name": [" ".join(labels) for labels in top3labs]
# })

# # 4.6 Save to CSV
# submission_df.to_csv(OUTPUT_PATH, index=False)
# print(f"✅ Submission saved to `{OUTPUT_PATH}`")



# 5. Out-of-Fold Predictions & Final Test Predictions

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

# Prepare OOF container (n_samples × 3 top predictions)
n_samples = X_train_top.shape[0]
oof_preds = np.zeros((n_samples, 3), dtype=int)

# 5.1 Stratified 5-Fold CV for OOF predictions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_top, y_train), start=1):
    print(f"Fold {fold}/5")
    
    X_tr = X_train_top.iloc[train_idx]
    y_tr = y_train.iloc[train_idx]
    X_va = X_train_top.iloc[val_idx]
    
    model = XGBClassifier(
        max_depth=12,
        colsample_bytree=0.467,
        subsample=0.86,
        n_estimators=4000,
        learning_rate=0.03,
        gamma=0.26,
        max_delta_step=4,
        reg_alpha=2.7,
        reg_lambda=1.4,
        objective="multi:softprob",
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=13,
        enable_categorical=True,
        tree_method='gpu_hist',  # ou 'hist' para CPU
        predictor='gpu_predictor',
        device="cuda"
    )
    
    model.fit(X_tr, y_tr, verbose=False)
    proba_va = model.predict_proba(X_va)
    top3_va = np.argsort(proba_va, axis=1)[:, -3:][:, ::-1]
    oof_preds[val_idx, :] = top3_va

# 5.2 Compute OOF MAP@3
def mapk(actual, predicted, k=3):
    scores = []
    for true_label, preds in zip(actual, predicted):
        hits = 0
        score = 0.0
        seen = set()
        for i, p in enumerate(preds[:k]):
            if p == true_label and p not in seen:
                hits += 1
                score += hits / (i + 1)
                seen.add(p)
        scores.append(score / 1.0)
    return np.mean(scores)

oof_score = mapk(y_train.values, oof_preds)
print(f"OOF MAP@3: {oof_score:.5f}")

# 5.3 Retrain on full training set and predict on test set
final_model = XGBClassifier(
    max_depth=12,
    colsample_bytree=0.467,
    subsample=0.86,
    n_estimators=4000,
    learning_rate=0.03,
    gamma=0.26,
    max_delta_step=4,
    reg_alpha=2.7,
    reg_lambda=1.4,
    objective="multi:softprob",
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=13,
    enable_categorical=True,
    tree_method='gpu_hist',  # ou 'hist' para CPU
    predictor='gpu_predictor',
    device="cuda"
)

final_model.fit(X_train_top, y_train, verbose=False)
proba_test = final_model.predict_proba(X_test_top)
top3_test = np.argsort(proba_test, axis=1)[:, -3:][:, ::-1]

# Decode integer labels back to fertilizer names
test_top1 = label_encoder.inverse_transform(top3_test[:, 0])
test_top2 = label_encoder.inverse_transform(top3_test[:, 1])
test_top3 = label_encoder.inverse_transform(top3_test[:, 2])

submission_df = pd.DataFrame({
    "id": test_df["id"],
    "Fertilizer Name": [
        f"{a} {b} {c}" for a, b, c in zip(test_top1, test_top2, test_top3)
    ]
})

submission_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Submission saved to `{OUTPUT_PATH}`")


import pandas as pd
from sklearn.metrics import confusion_matrix

oof_preds = pd.read_csv("/kaggle/input/naivebaseline-sfs-xgb-baseline-oof-predictions/NaiveBaseline _SFS_XGB baseline_oof_predictions.csv")
df = oof_preds.merge(train_df[["Fertilizer Name", "id"]], on="id")
df["Fertilizer Name_y"] = label_encoder.inverse_transform(df["Fertilizer Name_y"])
# True labels and predicted labels
y_pred = df["Fertilizer Name_x"].apply(lambda x: x.split(" ")[0])
y_true = df["Fertilizer Name_y"]

# Get unique labels in sorted order (for consistent matrix)
labels = sorted(list(set(y_true) | set(y_pred)))

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Convert to DataFrame for readability
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df

