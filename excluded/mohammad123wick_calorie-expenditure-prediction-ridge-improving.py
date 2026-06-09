!pip install ydf koolbox scikit-learn==1.5.2 && pip install --no-deps scikeras


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor , Pool
from xgboost import XGBRegressor
from koolbox import Trainer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import joblib
import shutil
import optuna
import json
import glob
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib



warnings.filterwarnings("ignore")


class CFG:
    train_path = "/kaggle/input/playground-series-s5e5/train.csv"
    test_path = "/kaggle/input/playground-series-s5e5/test.csv"
    sample_sub_path = "/kaggle/input/playground-series-s5e5/sample_submission.csv"
    
    original_path = "/kaggle/input/calories-burnt-prediction/calories.csv"

    metric = root_mean_squared_error
    target = "Calories"
    n_folds = 5
    seed = 42

    cv = KFold(n_splits=n_folds, random_state=seed, shuffle=True)

    run_optuna = True
    n_optuna_trials = 250


train = pd.read_csv(CFG.train_path, index_col="id")
test = pd.read_csv(CFG.test_path, index_col="id")

train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

X = train.drop(CFG.target, axis=1)
y = np.log1p(train[CFG.target])
X_test = test


original = pd.read_csv(CFG.original_path, index_col="User_ID")
original["Gender"] = original["Gender"].map({"male": 0, "female": 1})
original = original.rename(columns={"Gender": "Sex"})

X_original = original.drop(CFG.target, axis=1)
y_original = np.log1p(original[CFG.target])


mutual_info = mutual_info_regression(X, y, random_state=CFG.seed)

mutual_info = pd.Series(mutual_info)
mutual_info.index = X.columns
mutual_info = pd.DataFrame(mutual_info.sort_values(ascending=False), columns=['Mutual Information'])
mutual_info.style.bar(subset=['Mutual Information'], cmap='RdYlGn')


mutual_info = mutual_info_regression(X_original, y_original, random_state=CFG.seed)

mutual_info = pd.Series(mutual_info)
mutual_info.index = X_original.columns
mutual_info = pd.DataFrame(mutual_info.sort_values(ascending=False), columns=['Mutual Information'])
mutual_info.style.bar(subset=['Mutual Information'], cmap='RdYlGn')


sns.set_style("white")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

corr_train = train.corr()
mask_train = np.triu(np.ones_like(corr_train, dtype=bool), k=1)
sns.heatmap(
    data=corr_train,
    annot=True,
    fmt='.2f',
    mask=mask_train,
    square=True,
    cmap='coolwarm',
    cbar_kws={'shrink': .7, 'format': '%.2f'},   
    annot_kws={'size': 8},
    center=0,
    ax=axes[0]
)
axes[0].set_title('Train')
axes[0].tick_params(axis='both', which='major', labelsize=8)

corr_orig = original.corr()
mask_orig = np.triu(np.ones_like(corr_orig, dtype=bool), k=1)
sns.heatmap(
    data=corr_orig,
    annot=True,
    fmt='.2f',
    mask=mask_orig,
    square=True,
    cmap='coolwarm',
    cbar_kws={'shrink': .7, 'format': '%.2f'},   
    annot_kws={'size': 8},
    center=0,
    ax=axes[1]
)
axes[1].set_title('Original')
axes[1].tick_params(axis='both', which='major', labelsize=8)

plt.tight_layout()
plt.show()


histgb_params = {
    "l2_regularization": 10.412017522533768,
    "learning_rate": 0.011702680619474444,
    "max_depth": 59,
    "max_features": 0.30616140080552673,
    "max_iter": 4454,
    "max_leaf_nodes": 385,
    "min_samples_leaf": 50,
    "random_state": 42
}

lgbm_params = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.8213924491907012,
    "learning_rate": 0.059976685297931195,
    "min_child_samples": 10,
    "min_child_weight": 0.5425237767880097,
    "n_estimators": 50000,
    "n_jobs": -1,
    "num_leaves": 89,
    "random_state": 42,
    "reg_alpha": 2.0325709613371545,
    "reg_lambda": 87.27971117911044,
    "subsample": 0.6452823633939004,
    "verbose": -1
}

lgbm_goss_params = {
    "boosting_type": "goss",
    "colsample_bytree": 0.9068724002629094,
    "learning_rate": 0.06459027654473874,
    "min_child_samples": 39,
    "min_child_weight": 0.5337673729810578,
    "n_estimators": 50000,
    "n_jobs": -1,
    "num_leaves": 13,
    "random_state": 42,
    "reg_alpha": 1.603969498256519,
    "reg_lambda": 10.806488455621444,
    "subsample": 0.5966412222358356,
    "verbose": -1
}

xgb_params = {
    "colsample_bylevel": 0.8606487417581108,
    "colsample_bynode": 0.9410596660335436,
    "colsample_bytree": 0.9407540036296737,
    "early_stopping_rounds": 100,
    "eval_metric": "rmse",
    "gamma": 0.023260595738991977,
    "learning_rate": 0.03669372905801298,
    "max_depth": 11,
    "max_leaves": 51,
    "min_child_weight": 96,
    "n_estimators": 50000,
    "n_jobs": -1,
    "random_state": 42,
    "reg_alpha": 2.953205886504917,
    "reg_lambda": 67.64147033446291,
    "subsample": 0.6973241930754311,
    "verbosity": 0
}

cb_params = {
    "border_count": 88,
    "colsample_bylevel": 0.7903437608890396,
    "depth": 8,
    "eval_metric": "RMSE",
    "iterations": 50000,
    "l2_leaf_reg": 6.065104074215131,
    "learning_rate": 0.030946464122148992,
    "min_child_samples": 138,
    "random_state": 42,
    "random_strength": 0.035251008593976785,
    "verbose": False
}


scores = {}
oof_preds = {}
test_preds = {}


histgb_trainer = Trainer(
    HistGradientBoostingRegressor(**histgb_params),
    cv=CFG.cv,
    metric=CFG.metric,
    task="regression"
)

histgb_trainer.fit(X, y, extra_X=X_original, extra_y=y_original)

scores["HistGB"] = histgb_trainer.fold_scores
oof_preds["HistGB"] = histgb_trainer.oof_preds
test_preds["HistGB"] = histgb_trainer.predict(X_test)


lgbm_trainer = Trainer(
    LGBMRegressor(**lgbm_params),
    cv=CFG.cv,
    metric=CFG.metric,
    use_early_stopping=True,
    task="regression"
)

fit_args = {
    "eval_metric": "rmse",
    "callbacks": [
        log_evaluation(period=1000), 
        early_stopping(stopping_rounds=100)
    ]
}

lgbm_trainer.fit(X, y, fit_args=fit_args, extra_X=X_original, extra_y=y_original)

scores["LightGBM (gbdt)"] = lgbm_trainer.fold_scores
oof_preds["LightGBM (gbdt)"] = lgbm_trainer.oof_preds
test_preds["LightGBM (gbdt)"] = lgbm_trainer.predict(X_test)


lgbm_goss_trainer = Trainer(
    LGBMRegressor(**lgbm_goss_params),
    cv=CFG.cv,
    metric=CFG.metric,
    use_early_stopping=True,
    task="regression"
)

fit_args = {
    "eval_metric": "rmse",
    "callbacks": [
        log_evaluation(period=1000), 
        early_stopping(stopping_rounds=100)
    ]
}

lgbm_goss_trainer.fit(X, y, fit_args=fit_args, extra_X=X_original, extra_y=y_original)

scores["LightGBM (goss)"] = lgbm_goss_trainer.fold_scores
oof_preds["LightGBM (goss)"] = lgbm_goss_trainer.oof_preds
test_preds["LightGBM (goss)"] = lgbm_goss_trainer.predict(X_test)


xgb_trainer = Trainer(
    XGBRegressor(**xgb_params),
    cv=CFG.cv,
    metric=CFG.metric,
    use_early_stopping=True,
    task="regression"
)

fit_args = {
    "verbose": 1000
}

xgb_trainer.fit(X, y, fit_args=fit_args, extra_X=X_original, extra_y=y_original)

scores["XGBoost"] = xgb_trainer.fold_scores
oof_preds["XGBoost"] = xgb_trainer.oof_preds
test_preds["XGBoost"] = xgb_trainer.predict(X_test)


cb_trainer = Trainer(
    CatBoostRegressor(**cb_params),
    cv=CFG.cv,
    metric=CFG.metric,
    use_early_stopping=True,
    task="regression"
)

fit_args = {
    "verbose": 1000,
    "early_stopping_rounds": 100,
    "use_best_model": True
}

cb_trainer.fit(X, y, fit_args=fit_args, extra_X=X_original, extra_y=y_original)

scores["CatBoost"] = cb_trainer.fold_scores
oof_preds["CatBoost"] = cb_trainer.oof_preds
test_preds["CatBoost"] = cb_trainer.predict(X_test)


oof_preds_files = glob.glob(f'/kaggle/input/s05e05-calorie-expenditure-prediction-automl/*_oof_preds_*.pkl')
test_preds_files = glob.glob(f'/kaggle/input/s05e05-calorie-expenditure-prediction-automl/*_test_preds_*.pkl')

ag_oof_preds = np.log1p(joblib.load(oof_preds_files[0]))
ag_test_preds = np.log1p(joblib.load(test_preds_files[0]))

ag_scores = []
split = KFold(n_splits=CFG.n_folds, random_state=CFG.seed, shuffle=True).split(X, y)
for _, val_idx in split:
    y_val = y[val_idx]
    y_preds = ag_oof_preds[val_idx]   
    score = root_mean_squared_error(y_preds, y_val)
    ag_scores.append(score)
    
oof_preds["AutoGluon"], test_preds["AutoGluon"], scores["AutoGluon"] = ag_oof_preds, ag_test_preds, ag_scores


def plot_weights(weights, title):
    sorted_indices = np.argsort(weights[0])[::-1]
    sorted_coeffs = np.array(weights[0])[sorted_indices]
    sorted_model_names = np.array(list(oof_preds.keys()))[sorted_indices]

    plt.figure(figsize=(10, weights.shape[1] * 0.5))
    ax = sns.barplot(x=sorted_coeffs, y=sorted_model_names, palette="RdYlGn_r")

    for i, (value, name) in enumerate(zip(sorted_coeffs, sorted_model_names)):
        if value >= 0:
            ax.text(value, i, f"{value:.3f}", va="center", ha="left", color="black")
        else:
            ax.text(value, i, f"{value:.3f}", va="center", ha="right", color="black")

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.1 * abs(xlim[0]), xlim[1] + 0.1 * abs(xlim[1]))

    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


X = pd.DataFrame(oof_preds)
X_test = pd.DataFrame(test_preds)


joblib.dump(X, "oof_preds.pkl")
joblib.dump(X_test, "test_preds.pkl")


def objective(trial):    
    params = {
        "random_state": CFG.seed,
        "alpha": trial.suggest_float("alpha", 0, 10),
        "tol": trial.suggest_float("tol", 1e-6, 1e-2)
    }
    
    trainer = Trainer(
        Ridge(**params),
        cv=CFG.cv,
        metric=CFG.metric,
        task="regression",
        verbose=False
    )
    trainer.fit(X, y)
    
    return np.mean(trainer.fold_scores)

if CFG.run_optuna:
    sampler = optuna.samplers.TPESampler(seed=CFG.seed, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=CFG.n_optuna_trials, n_jobs=-1, catch=(ValueError,))
    best_params = study.best_params

    ridge_params = {
        "random_state": CFG.seed,
        "alpha": best_params["alpha"],
        "tol": best_params["tol"]
    }
else:
    ridge_params = {
        "random_state": CFG.seed
    }


print(json.dumps(ridge_params, indent=2))


ridge_trainer = Trainer(
    Ridge(**ridge_params),
    cv=CFG.cv,
    metric=CFG.metric,
    task="regression"
)

ridge_trainer.fit(X, y)

scores["Ridge (ensemble)"] = ridge_trainer.fold_scores
ridge_test_preds = np.expm1(ridge_trainer.predict(X_test))


ridge_coeffs = np.zeros((1, X.shape[1]))
for m in ridge_trainer.estimators:
    ridge_coeffs += m.coef_
ridge_coeffs = ridge_coeffs / len(ridge_trainer.estimators)

plot_weights(ridge_coeffs, "Ridge Coefficients")


sub = pd.read_csv(CFG.sample_sub_path)
sub[CFG.target] = ridge_test_preds
sub.to_csv(f"sub_ridge_{np.mean(scores['Ridge (ensemble)']):.6f}.csv", index=False)
sub.head()


scores = pd.DataFrame(scores)
mean_scores = scores.mean().sort_values(ascending=True)
order = scores.mean().sort_values(ascending=True).index.tolist()

min_score = mean_scores.min()
max_score = mean_scores.max()
padding = (max_score - min_score) * 0.5
lower_limit = min_score - padding
upper_limit = max_score + padding

fig, axs = plt.subplots(1, 2, figsize=(15, scores.shape[1] * 0.3))

boxplot = sns.boxplot(data=scores, order=order, ax=axs[0], orient="h", color="grey")
axs[0].set_title(f"Fold {CFG.metric.__name__}")
axs[0].set_xlabel("")
axs[0].set_ylabel("")

barplot = sns.barplot(x=mean_scores.values, y=mean_scores.index, ax=axs[1], color="grey")
axs[1].set_title(f"Average {CFG.metric.__name__}")
axs[1].set_xlabel("")
axs[1].set_xlim(left=lower_limit, right=upper_limit)
axs[1].set_ylabel("")

for i, (score, model) in enumerate(zip(mean_scores.values, mean_scores.index)):
    color = "cyan" if "ensemble" in model.lower() else "grey"
    barplot.patches[i].set_facecolor(color)
    boxplot.patches[i].set_facecolor(color)
    barplot.text(score, i, round(score, 6), va="center")

plt.tight_layout()
plt.show()


shutil.rmtree("catboost_info", ignore_errors=True)


# ðŸ”§ Add interaction & polynomial features
def add_features(df):
    df["Weight_Duration"] = df["Weight"] * df["Duration"]
    df["HeartRate_Duration"] = df["Heart_Rate"] * df["Duration"]
    df["Weight_HeartRate"] = df["Weight"] * df["Heart_Rate"]
    df["Height_Duration"] = df["Height"] * df["Duration"]
    
    df["Duration_squared"] = df["Duration"] ** 2
    df["HeartRate_squared"] = df["Heart_Rate"] ** 2
    df["BodyTemp_squared"] = df["Body_Temp"] ** 2

    df["log_Duration"] = np.log1p(df["Duration"])
    df["sqrt_Duration"] = np.sqrt(df["Duration"])
    return df

# Apply to all datasets
X = add_features(X)
X_test = add_features(X_test)
X_original = add_features(X_original)





ridge_features = X.columns  # includes both original and new features

scaler = StandardScaler()
X_ridge = scaler.fit_transform(X[ridge_features])
X_test_ridge = scaler.transform(X_test[ridge_features])
X_original_ridge = scaler.transform(X_original[ridge_features])


oof_preds_lgb2 = np.zeros(len(X))
test_preds_lgb2 = np.zeros(len(X_test))

cv = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=2025)  # different seed

for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y[val_idx]

    model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.015,
        n_estimators=10000,
        max_depth=10,
        num_leaves=128,
        subsample=0.7,
        colsample_bytree=0.8,
        random_state=2025
    )

    model.fit(
    X_train_fold, y_train_fold,
    eval_set=[(X_val_fold, y_val_fold)],
    eval_metric="rmse",
    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(0)]
)

    oof_preds_lgb2[val_idx] = model.predict(X_val_fold)
    test_preds_lgb2 += model.predict(X_test) / CFG.n_folds

print("LGBM Variant 2 CV RMSLE:", root_mean_squared_error(y, oof_preds_lgb2))




oof_preds_xgb2 = np.zeros(len(X))
test_preds_xgb2 = np.zeros(len(X_test))

cv = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=2027)  # different seed

for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y[val_idx]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.02,
        n_estimators=10000,
        max_depth=9,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=1,
        reg_lambda=2,
        random_state=2027,
        verbosity=0
    )

    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        early_stopping_rounds=100,
        verbose=False
    )

    oof_preds_xgb2[val_idx] = model.predict(X_val_fold)
    test_preds_xgb2 += model.predict(X_test) / CFG.n_folds

print("XGBoost Variant 2 CV RMSLE:", root_mean_squared_error(y, oof_preds_xgb2))





# Save LightGBM v2
joblib.dump(oof_preds_lgb2, "oof_preds_lgb2.pkl")
joblib.dump(test_preds_lgb2, "test_preds_lgb2.pkl")

# Save XGBoost v2
joblib.dump(oof_preds_xgb2, "oof_preds_xgb2.pkl")
joblib.dump(test_preds_xgb2, "test_preds_xgb2.pkl")





oof_preds_cat2 = np.zeros(len(X))
test_preds_cat2 = np.zeros(len(X_test))

cv = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=2029)

for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y[val_idx]

    model = CatBoostRegressor(
        iterations=10000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6,
        random_seed=2029,
        loss_function='RMSE',
        eval_metric='RMSE',
        verbose=0,
        early_stopping_rounds=100
    )

    model.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold)
    )

    oof_preds_cat2[val_idx] = model.predict(X_val_fold)
    test_preds_cat2 += model.predict(X_test) / CFG.n_folds

print("CatBoost Variant 2 CV RMSLE:", root_mean_squared_error(y, oof_preds_cat2))



joblib.dump(oof_preds_cat2, "oof_preds_cat2.pkl")
joblib.dump(test_preds_cat2, "test_preds_cat2.pkl")



oof_preds_keras = np.zeros(len(X_ridge))
test_preds_keras = np.zeros(len(X_test_ridge))

cv = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=2033)

for fold, (train_idx, val_idx) in enumerate(cv.split(X_ridge)):
    X_train_fold = X_ridge[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X_ridge[val_idx]
    y_val_fold = y[val_idx]

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_ridge.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=64, verbose=0, validation_data=(X_val_fold, y_val_fold))

    oof_preds_keras[val_idx] = model.predict(X_val_fold).reshape(-1)
    test_preds_keras += model.predict(X_test_ridge).reshape(-1) / CFG.n_folds


print("Keras MLP CV RMSLE:", root_mean_squared_error(y, oof_preds_keras))




joblib.dump(oof_preds_keras, "oof_preds_keras.pkl")
joblib.dump(test_preds_keras, "test_preds_keras.pkl")





# Ridge or base Ridge variant
# oof_preds_ridge = joblib.load("...")
# test_preds_ridge = joblib.load("...")

# LightGBM variants
oof_preds_lgb2 = joblib.load("oof_preds_lgb2.pkl")
test_preds_lgb2 = joblib.load("test_preds_lgb2.pkl")

# XGBoost variants
oof_preds_xgb2 = joblib.load("oof_preds_xgb2.pkl")
test_preds_xgb2 = joblib.load("test_preds_xgb2.pkl")

# CatBoost variants
oof_preds_cat2 = joblib.load("oof_preds_cat2.pkl")
test_preds_cat2 = joblib.load("test_preds_cat2.pkl")

# Keras MLP
oof_preds_keras = joblib.load("oof_preds_keras.pkl")
test_preds_keras = joblib.load("test_preds_keras.pkl")



# Stack into meta-training and meta-test matrices
X_stack = np.vstack([
    oof_preds_lgb2,
    oof_preds_xgb2,
    oof_preds_cat2,
    oof_preds_keras,
]).T  # shape: (n_samples, n_models)

X_stack_test = np.vstack([
    test_preds_lgb2,
    test_preds_xgb2,
    test_preds_cat2,
    test_preds_keras,
]).T



cv = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=42)

oof_preds_stack = np.zeros(len(y))
final_preds = np.zeros(len(X_stack_test))

for fold, (train_idx, val_idx) in enumerate(cv.split(X_stack)):
    X_tr, y_tr = X_stack[train_idx], y[train_idx]
    X_val, y_val = X_stack[val_idx], y[val_idx]

    meta_model = Ridge(alpha=1.0, random_state=42)
    meta_model.fit(X_tr, y_tr)

    oof_preds_stack[val_idx] = meta_model.predict(X_val)
    final_preds += meta_model.predict(X_stack_test) / CFG.n_folds

print("Stacked Ensemble CV RMSLE:", root_mean_squared_error(y, oof_preds_stack))



# Detect if y is log-transformed by checking its max range
if y.max() < 20:  # likely log1p scale
    y_true = np.expm1(y)
    y_pred = np.expm1(oof_preds_stack)
    print("Detected log1p scale â€” plotting in original scale.")
else:
    y_true = y
    y_pred = oof_preds_stack
    print("Detected raw scale â€” plotting without transformation.")

# ðŸ“ˆ Actual vs Predicted
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.4)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Ridge Meta-Model: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# ðŸ“Š Residuals Plot
residuals = y_true - y_pred

plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=40, kde=True)
plt.title("Ridge Meta-Model: Residuals Distribution")
plt.xlabel("Prediction Error")
plt.grid(True)
plt.tight_layout()
plt.show()



submission = pd.read_csv(CFG.sample_sub_path)
submission["Calories"] = np.expm1(final_preds)  # if you used log1p earlier
submission.to_csv("submission_stacked.csv", index=False)


