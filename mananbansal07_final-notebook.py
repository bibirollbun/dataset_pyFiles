import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



train = pd.read_csv("/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/train.csv")
test  = pd.read_csv("/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/test.csv")
lookup = pd.read_csv("/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/feature_lookup.csv")



TARGET = "relationship_probability"
ID = "ID"

cat_feats = lookup[lookup["type"]=="categorical"]["feature_code"].tolist()
num_feats = lookup[lookup["type"]=="numeric"]["feature_code"].tolist()

cat_feats = [c for c in cat_feats if c in train.columns]
num_feats = [n for n in num_feats if n in train.columns]

print("Categorical:", cat_feats)
print("Numeric:", num_feats)



plt.figure(figsize=(6,4))
sns.histplot(train[TARGET], kde=True)
plt.title("Target Distribution")
plt.show()



# Pick top correlated numeric features
corrs = train[num_feats].corrwith(train[TARGET]).abs()
top10 = corrs.sort_values(ascending=False).head(10).index.tolist()
top10



poly = PolynomialFeatures(
    degree=2,
    include_bias=False,
    interaction_only=True     # ONLY interactions, no squares
)

poly_train = poly.fit_transform(train[top10])
poly_test  = poly.transform(test[top10])

poly_cols = poly.get_feature_names_out(top10)
poly_cols_clean = [c for c in poly_cols if c not in top10]

poly_train_df = pd.DataFrame(poly_train, columns=poly_cols)[poly_cols_clean]
poly_test_df  = pd.DataFrame(poly_test,  columns=poly_cols)[poly_cols_clean]

train_poly = pd.concat([train, poly_train_df], axis=1)
test_poly  = pd.concat([test,  poly_test_df], axis=1)

num_feats_poly = num_feats + poly_cols_clean



preprocessor_poly = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_feats_poly),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
    ]
)

preprocessor_poly.fit(train_poly[cat_feats + num_feats_poly])



param_grid = [
    {"alpha": 0.0001, "l1_ratio": 0.7},
    {"alpha": 0.0002, "l1_ratio": 0.8},
    {"alpha": 0.0005, "l1_ratio": 0.5},
    {"alpha": 0.0008, "l1_ratio": 0.6},
]

def comp_score(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    rmse_n = max(0, 1 - rmse/100)
    mae_n  = max(0, 1 - mae/100)
    r2_n   = max(0, min(1, r2))

    return (0.4*rmse_n + 0.3*mae_n + 0.3*r2_n) * 100

best_score = -999
best_params = None
Xp = train_poly[cat_feats + num_feats_poly]
Xp_trans = preprocessor_poly.transform(Xp)

for params in param_grid:
    print("Testing:", params)
    en = ElasticNet(
        alpha=params["alpha"],
        l1_ratio=params["l1_ratio"],
        max_iter=5000
    )

    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
        Xp_trans, train_poly[TARGET], test_size=0.2, random_state=42
    )

    en.fit(X_train_s, y_train_s)
    preds = en.predict(X_val_s)
    score = comp_score(y_val_s, preds)
    print("Score:", score)

    if score > best_score:
        best_score = score
        best_params = params

print("\nBest params:", best_params, "Best score:", best_score)



X_full = preprocessor_poly.transform(train_poly[cat_feats + num_feats_poly])
y_full = train_poly[TARGET]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(train_poly))
models = []

for fold, (tr, val) in enumerate(kf.split(X_full)):
    print("FOLD", fold+1)
    en = ElasticNet(
        alpha=best_params["alpha"],
        l1_ratio=best_params["l1_ratio"],
        max_iter=5000
    )
    en.fit(X_full[tr], y_full.iloc[tr])
    pred = en.predict(X_full[val])

    oof[val] = pred
    models.append(en)

print("\nOOF Score:", comp_score(y_full, oof))



# ----------------------------------------------
# DOUBLE ELASTICNET OOF TRAINING
# ----------------------------------------------

X_full = preprocessor_poly.transform(train_poly[cat_feats + num_feats_poly])
y_full = train_poly[TARGET]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(train_poly))

models_A = []
models_B = []

fold_num = 1
for tr_idx, val_idx in kf.split(X_full):
    print(f"\nFOLD {fold_num}")
    fold_num += 1

    X_tr, X_val = X_full[tr_idx], X_full[val_idx]
    y_tr, y_val = y_full.iloc[tr_idx], y_full.iloc[val_idx]
    
    # ----------------------------------------------
    # MODEL A (strong + general)
    # ----------------------------------------------
    en_A = ElasticNet(
        alpha=0.0003,
        l1_ratio=0.6,
        max_iter=5000,
        random_state=42
    )
    en_A.fit(X_tr, y_tr)
    pred_A = en_A.predict(X_val)
    models_A.append(en_A)

    # ----------------------------------------------
    # MODEL B (slightly different regularization)
    # ----------------------------------------------
    en_B = ElasticNet(
        alpha=0.00045,
        l1_ratio=0.5,
        max_iter=5000,
        random_state=42
    )
    en_B.fit(X_tr, y_tr)
    pred_B = en_B.predict(X_val)
    models_B.append(en_B)
    
    # ----------------------------------------------
    # BLEND predictions for this fold
    # ----------------------------------------------
    fold_pred = 0.6 * pred_A + 0.4 * pred_B
    oof[val_idx] = fold_pred

# Final OOF score
print("\n=======================================")
print("DOUBLE ENET OOF Score:", comp_score(y_full, oof))
print("OOF RMSE:", mean_squared_error(y_full, oof, squared=False))
print("OOF MAE:", mean_absolute_error(y_full, oof))
print("OOF R2:", r2_score(y_full, oof))
print("=======================================\n")



# Transform test data
X_test_trans = preprocessor_poly.transform(test_poly[cat_feats + num_feats_poly])

# ElasticNet A predictions
test_preds_A = np.mean([model.predict(X_test_trans) for model in models_A], axis=0)

# ElasticNet B predictions
test_preds_B = np.mean([model.predict(X_test_trans) for model in models_B], axis=0)

# Blend final predictions
final_preds = 0.6 * test_preds_A + 0.4 * test_preds_B



final_smooth = (final_preds - final_preds.mean()) / final_preds.std()
final_smooth = final_smooth * train[TARGET].std()
final_smooth = final_smooth + train[TARGET].mean()
final_smooth = np.clip(final_smooth, 0, 100)



submission = pd.DataFrame({
    "ID": test["ID"],
    "relationship_probability": final_smooth
})

submission.to_csv("submission.csv", index=False)
submission.head()



import joblib

joblib.dump(poly, "poly_transformer.pkl")
joblib.dump(preprocessor_poly, "preprocessor.pkl")
joblib.dump(models, "elasticnet_models.pkl")  # 5-model ensemble


