# ========================
# 0. å®‰è£�ä¾�è³´ï¼ˆå�ªéœ€åŸ·è¡Œä¸€æ¬¡ï¼‰
# ========================
!pip install -q pytorch-tabnet optuna lightgbm

import pandas as pd
import numpy as np
import itertools
import warnings
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score, accuracy_score, f1_score, recall_score, precision_score
from lightgbm import LGBMRegressor
import optuna
from pytorch_tabnet.tab_model import TabNetRegressor

warnings.filterwarnings("ignore")

# ========================
# 1. è³‡æ–™è¼‰å…¥ï¼‹ç‰¹å¾µå·¥ç¨‹
# ========================
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

def add_feature_cross_terms(df, features):
    df = df.copy()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            df[f"{features[i]}_x_{features[j]}"] = df[features[i]] * df[features[j]]
    return df

def add_interaction_features(df, features):
    df_new = df.copy()
    for f1, f2 in itertools.combinations(features, 2):
        df_new[f"{f1}_plus_{f2}"]  = df_new[f1] + df_new[f2]
        df_new[f"{f1}_minus_{f2}"] = df_new[f1] - df_new[f2]
        df_new[f"{f2}_minus_{f1}"] = df_new[f2] - df_new[f1]
        df_new[f"{f1}_div_{f2}"]   = df_new[f1] / (df_new[f2] + 1e-5)
        df_new[f"{f2}_div_{f1}"]   = df_new[f2] / (df_new[f1] + 1e-5)
    return df_new

def add_statistical_features(df, features):
    df_new = df.copy()
    df_new["row_mean"]   = df[features].mean(axis=1)
    df_new["row_std"]    = df[features].std(axis=1)
    df_new["row_max"]    = df[features].max(axis=1)
    df_new["row_min"]    = df[features].min(axis=1)
    df_new["row_median"] = df[features].median(axis=1)
    return df_new

for func in (add_feature_cross_terms, add_interaction_features, add_statistical_features):
    train = func(train, numerical_features)
    test  = func(test, numerical_features)

le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex']  = le.transform(test['Sex'])

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_train = poly.fit_transform(train[numerical_features])
poly_test  = poly.transform(test[numerical_features])
poly_cols  = poly.get_feature_names_out(numerical_features)

train = pd.concat([train.reset_index(drop=True), pd.DataFrame(poly_train, columns=poly_cols)], axis=1)
test  = pd.concat([test.reset_index(drop=True), pd.DataFrame(poly_test,  columns=poly_cols)], axis=1)

X      = train.drop(columns=["id", "Calories"])
y      = np.log1p(train["Calories"])
X_test = test.drop(columns=["id"])

corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
X      = X.drop(columns=to_drop)
X_test = X_test.drop(columns=to_drop)

lgb = LGBMRegressor().fit(X, y)
imp = pd.Series(lgb.feature_importances_, index=X.columns)
top100 = imp.sort_values(ascending=False).head(100).index.tolist()
X      = X[top100]
X_test = X_test[top100]

scaler       = StandardScaler()
X_scaled     = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ========================
# 2. Optunaè‡ªå‹•TabNetå�ƒæ•¸æ�œå°‹
# ========================
def tabnet_objective(trial):
    n_d = trial.suggest_int("n_d", 8, 64, step=8)
    n_a = trial.suggest_int("n_a", 8, 64, step=8)
    n_steps = trial.suggest_int("n_steps", 3, 10)
    gamma = trial.suggest_float("gamma", 1.0, 2.0)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-2, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    oof = np.zeros(len(X_scaled))
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(X_scaled):
        X_tr, y_tr = X_scaled[tr_idx], y.iloc[tr_idx].values.reshape(-1,1)
        X_val, y_val = X_scaled[val_idx], y.iloc[val_idx].values.reshape(-1,1)
        model = TabNetRegressor(
            n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
            lambda_sparse=lambda_sparse, optimizer_params=dict(lr=learning_rate),
            verbose=0, seed=42
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric=['rmse'],
            patience=15, batch_size=batch_size, max_epochs=60
        )
        oof[val_idx] = model.predict(X_val).ravel()
    score = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof)))
    print(f"[TabNet Trial] n_d={n_d}, n_a={n_a}, n_steps={n_steps}, gamma={gamma:.3f}, "
          f"lambda_sparse={lambda_sparse:.6f}, lr={learning_rate:.5f}, "
          f"batch_size={batch_size} -> RMSLE: {score:.5f}")
    return score

tabnet_study = optuna.create_study(direction="minimize")
tabnet_study.optimize(tabnet_objective, n_trials=10)
tabnet_params = tabnet_study.best_trial.params
print("Best TabNet params:", tabnet_params)

# ========================
# 3. TabNet 5æŠ˜è¨“ç·´+losså…¨è¨˜éŒ„+OOF erroråŒ¯å‡º+è©•åˆ†
# ========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
tabnet_oof = np.zeros(len(X_scaled))
tabnet_preds = np.zeros(len(X_test_scaled))
tabnet_train_losses = []
tabnet_val_losses = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
    X_tr, y_tr = X_scaled[tr_idx], y.iloc[tr_idx].values.reshape(-1,1)
    X_val, y_val = X_scaled[val_idx], y.iloc[val_idx].values.reshape(-1,1)
    model = TabNetRegressor(
        n_d=tabnet_params["n_d"], n_a=tabnet_params["n_a"], n_steps=tabnet_params["n_steps"],
        gamma=tabnet_params["gamma"], lambda_sparse=tabnet_params["lambda_sparse"],
        optimizer_params=dict(lr=tabnet_params["learning_rate"]),
        verbose=0, seed=42
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=['rmse'],
        patience=20,
        batch_size=tabnet_params["batch_size"],
        max_epochs=100,
        loss_fn='mse'
    )
    tabnet_train_losses.append(model.history['loss'])
    tabnet_val_losses.append(model.history['val_0_rmse'])
    tabnet_oof[val_idx] = model.predict(X_val).ravel()
    tabnet_preds += model.predict(X_test_scaled).ravel() / kf.n_splits

# ========================
# 4. æŒ‡æ¨™ã€�lossã€�å��å·®/æ–¹å·®ã€�åˆ†é¡�è©•åˆ†ã€�éŒ¯èª¤æ¨£æœ¬åŒ¯å‡º
# ========================
y_true = np.expm1(y)
y_tabnet = np.expm1(tabnet_oof)

# (1) å›�æ­¸æŒ‡æ¨™
rmsle = np.sqrt(mean_squared_log_error(y_true, y_tabnet))
mae = mean_absolute_error(y_true, y_tabnet)
r2 = r2_score(y_true, y_tabnet)
print(f"\nTabNet OOF å›�æ­¸è©•åˆ†ï¼š\nRMSLE: {rmsle:.5f}\nMAE: {mae:.3f}\nRÂ²: {r2:.4f}")

# (2) Loss å�¯è¦–åŒ–
plt.figure(figsize=(10,4))
for i, (train_l, val_l) in enumerate(zip(tabnet_train_losses, tabnet_val_losses)):
    plt.plot(train_l, label=f"TabNet Fold {i+1} - Train", linestyle='--')
    plt.plot(val_l, label=f"TabNet Fold {i+1} - Val")
plt.title("TabNet Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

# (3) Loss çµ±è¨ˆ
print("Train loss (mean/std/min/max by fold):",
      np.mean([np.mean(l) for l in tabnet_train_losses]),
      np.std([np.mean(l) for l in tabnet_train_losses]),
      np.min([np.min(l) for l in tabnet_train_losses]),
      np.max([np.max(l) for l in tabnet_train_losses]))
print("Val loss (mean/std/min/max by fold):",
      np.mean([np.mean(l) for l in tabnet_val_losses]),
      np.std([np.mean(l) for l in tabnet_val_losses]),
      np.min([np.min(l) for l in tabnet_val_losses]),
      np.max([np.max(l) for l in tabnet_val_losses]))

# (4) å��å·®/æ–¹å·®è¨ºæ–·
if r2 < 0.5:
    print("â�— High Bias (Underfitting): Try deeper models or better features.")
elif r2 > 0.95 and rmsle > 0.1:
    print("â�— High Variance (Overfitting): Try more regularization or collect more data.")
else:
    print("âœ… Model Generalizes Well.")

# (5) åˆ†é¡�å�‹è©•åˆ†ï¼ˆbinningï¼‰
bins = np.quantile(y_true, [0, 1/3, 2/3, 1])
labels = [0, 1, 2]
y_cls_true = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True)
y_cls_pred = pd.cut(y_tabnet, bins=bins, labels=labels, include_lowest=True)
print("Accuracy :", accuracy_score(y_cls_true, y_cls_pred))
print("F1 Score :", f1_score(y_cls_true, y_cls_pred, average='macro'))
print("Precision:", precision_score(y_cls_true, y_cls_pred, average='macro'))
print("Recall   :", recall_score(y_cls_true, y_cls_pred, average='macro'))

# (6) åŒ¯å‡º TabNet éŒ¯èª¤ OOF æ¨£æœ¬
error_df = train.copy()
error_df["Pred_TabNet"] = y_tabnet
error_df["Actual"] = y_true
error_df["Abs_Error"] = np.abs(error_df["Pred_TabNet"] - error_df["Actual"])
error_df["Abs_Percent_Error"] = error_df["Abs_Error"] / (error_df["Actual"] + 1e-6)
wrong_pred_df = error_df[error_df["Abs_Percent_Error"] > 0.1]
cols_to_save = list(X.columns) + ["Pred_TabNet", "Actual", "Abs_Error", "Abs_Percent_Error"]
wrong_pred_df[cols_to_save].to_csv("tabnet_oof_error_cases.csv", index=False)
print(f"ğŸ”� Output all high-error OOF samples to tabnet_oof_error_cases.csvï¼Œæ•¸é‡�: {len(wrong_pred_df)} ç­†")

# (7) æ��äº¤çµ�æ�œ
final_preds = np.clip(tabnet_preds, 1, 314)
submission["Calories"] = np.expm1(final_preds)
submission.to_csv("final_tabnet_submission.csv", index=False)
print("\nâœ… Final submission saved as final_tabnet_submission.csv")



