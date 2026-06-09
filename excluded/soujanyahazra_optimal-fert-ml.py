# Final Fertilizer Prediction Model (XGBoost + LightGBM + Logistic Ensemble)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# === Load Data ===
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
external = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")

# === Feature Engineering ===
def add_features(df):
    total = df["Nitrogen"] + df["Phosphorous"] + df["Potassium"]
    df["N_ratio"] = df["Nitrogen"] / total
    df["P_ratio"] = df["Phosphorous"] / total
    df["K_ratio"] = df["Potassium"] / total
    df["NPK_std"] = df[["N_ratio", "P_ratio", "K_ratio"]].std(axis=1)
    df["is_balanced"] = (df["NPK_std"] < 0.05).astype(int)
    df["SoilCrop"] = df["Soil Type"] + "_" + df["Crop Type"]
    return df

train = add_features(train)
test = add_features(test)
external = add_features(external)

# Drop 'id' if present in external and align columns
external.drop(columns=["id"], errors="ignore", inplace=True)
valid_cols = [col for col in train.columns if col != 'id']
external = external[[col for col in valid_cols if col in external.columns]]

# === Combine Datasets ===
combined = pd.concat([train.drop(columns=["id"]), external], ignore_index=True)

# === Encode Categorical ===
cat_cols = ["Soil Type", "Crop Type", "SoilCrop"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    test[col] = le.transform(test[col])
    encoders[col] = le

fert_le = LabelEncoder()
combined["Fert"] = fert_le.fit_transform(combined["Fertilizer Name"])
inv_fert = {i: str(f) for i, f in enumerate(fert_le.classes_)}

# === Features & Targets ===
features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
            'Nitrogen', 'Phosphorous', 'Potassium', 'N_ratio', 'P_ratio', 'K_ratio',
            'NPK_std', 'is_balanced', 'SoilCrop']
X = combined[features]
y = combined["Fert"]
X_test = test[features]

# === CV & Meta Ensemble ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores, val_xgb, val_lgb, val_y = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    model_xgb = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=6,
                              use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_lgb = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, objective='multiclass',
                                   num_class=len(fert_le.classes_), random_state=42)
    model_xgb.fit(X_tr, y_tr)
    model_lgb.fit(X_tr, y_tr)

    xgb_pred = model_xgb.predict_proba(X_val)
    lgb_pred = model_lgb.predict_proba(X_val)
    val_xgb.append(xgb_pred)
    val_lgb.append(lgb_pred)
    val_y.append(y_val_fold)

    stack_train = np.hstack([xgb_pred, lgb_pred])
    meta_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, multi_class='multinomial'))
    meta_model.fit(stack_train, y_val_fold)
    final_val_pred = meta_model.predict_proba(stack_train)

    def map3(y_true, y_prob):
        y_true = [[x] for x in y_true]
        y_pred = np.argsort(y_prob, axis=1)[:, -3:][:, ::-1].tolist()
        def ap3(y_t, y_p):
            score, hits = 0.0, 0.0
            for i, p in enumerate(y_p[:3]):
                if p in y_t and p not in y_p[:i]:
                    hits += 1
                    score += hits / (i + 1)
            return score
        return np.mean([ap3(t, p) for t, p in zip(y_true, y_pred)])

    score = map3(y_val_fold.tolist(), final_val_pred)
    scores.append(score)
    print(f"Fold {fold+1} MAP@3: {score:.5f}")

# === Retrain on All Data ===
final_xgb = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=6,
                          use_label_encoder=False, eval_metric='mlogloss', random_state=42)
final_lgb = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05,
                               objective='multiclass', num_class=len(fert_le.classes_), random_state=42)
final_xgb.fit(X, y)
final_lgb.fit(X, y)

meta_all = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, multi_class='multinomial'))
meta_all.fit(np.hstack([np.vstack(val_xgb), np.vstack(val_lgb)]), np.concatenate(val_y))

# === Predict Test Set ===
test_stack = np.hstack([final_xgb.predict_proba(X_test), final_lgb.predict_proba(X_test)])
final_test_probs = meta_all.predict_proba(test_stack)

# === Generate Submission ===
top_3 = np.argsort(final_test_probs, axis=1)[:, -3:][:, ::-1]
final_predictions = []
for row in top_3:
    preds = []
    for i in row:
        fert = inv_fert.get(i, "Unknown")
        if isinstance(fert, str) and fert not in preds:
            preds.append(fert)
    while len(preds) < 3:
        preds.append(preds[0])
    final_predictions.append(" ".join(preds[:3]))

submission = pd.DataFrame({
    "id": test["id"].values,
    "Fertilizer Name": final_predictions
})

submission.to_csv("submission.csv", index=False)
print("\nFinal Mean CV MAP@3: {:.5f} ± {:.5f}".format(np.mean(scores), np.std(scores)))
submission.head()


