import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def mapk(y_true, y_score, k=3):
    top_k_preds = np.argsort(y_score, axis=1)[:, ::-1][:, :k]
    relevance = np.zeros_like(top_k_preds)
    for i, true in enumerate(y_true.argmax(axis=1)):
        if true in top_k_preds[i]:
            relevance[i, np.where(top_k_preds[i] == true)[0][0]] = 1
    scores = []
    for rel in relevance:
        score = 0
        num_hits = 0
        for i, val in enumerate(rel):
            if val:
                num_hits += 1
                score += num_hits / (i + 1)
        scores.append(score / min(k, 1))
    return np.mean(scores)

train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_target = LabelEncoder()


train["Soil Type"] = le_soil.fit_transform(train["Soil Type"])
train["Crop Type"] = le_crop.fit_transform(train["Crop Type"])
train["Fertilizer Label"] = le_target.fit_transform(train["Fertilizer Name"])

test["Soil Type"] = le_soil.transform(test["Soil Type"])
test["Crop Type"] = le_crop.transform(test["Crop Type"])


feature_columns = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
                   'Nitrogen', 'Potassium', 'Phosphorous']
X = train[feature_columns]
y = train["Fertilizer Label"]
X_test = test[feature_columns]


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

blend_test_probs = np.zeros((len(X_test), len(np.unique(y))))
fold_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\n===== Fold {fold+1} =====")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    xgb_model = XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=16,
                              random_state=42, eval_metric='mlogloss'
                              ,subsample=0.686,colsample_bytree=0.25,min_child_weight=5,reg_alpha=0.11,reg_lambda=1.1,
                              )
    lgb_model = LGBMClassifier(n_estimators=800, learning_rate=0.05, max_depth=16,
                               random_state=42
                               ,subsample=0.686,colsample_bytree=0.25,min_child_samples=30,num_leaves=128,boosting_type='gbdt',
                               verbosity=-1)

    xgb_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train,)

    xgb_valid_probs = xgb_model.predict_proba(X_valid)
    lgb_valid_probs = lgb_model.predict_proba(X_valid)
    blend_valid_probs = 0.8175 * xgb_valid_probs + 0.1825 * lgb_valid_probs

    true_labels = np.zeros_like(blend_valid_probs)
    true_labels[np.arange(len(y_valid)), y_valid.values] = 1

    fold_score = mapk(true_labels, blend_valid_probs, k=3)
    fold_scores.append(fold_score)
    print(f"平均 MAP@3 Score: {np.mean(fold_scores):.8f}")


    xgb_test_probs = xgb_model.predict_proba(X_test)
    lgb_test_probs = lgb_model.predict_proba(X_test)
    blend_test_probs += 0.8175 * xgb_test_probs + 0.1825 * lgb_test_probs


blend_test_probs /= skf.n_splits

top3_indices = np.argsort(blend_test_probs, axis=1)[:, ::-1][:, :3]
top3_labels = le_target.inverse_transform(top3_indices.ravel()).reshape(top3_indices.shape)
predictions = [" ".join(row) for row in top3_labels]

submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": predictions
})
submission.to_csv("sample_submission.csv", index=False)

print("\nsample_submission finish")
print(f"平均 MAP@3 Score: {np.mean(fold_scores):.8f}")


