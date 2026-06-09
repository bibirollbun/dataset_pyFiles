import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


# Initial Exploration
df.info()


df.describe().T


# Select numeric columns
num_cols = df.columns.drop(["id", "BeatsPerMinute"])


# Histograms for numeric features
plt.figure(figsize=(12,6))
df[num_cols].hist(bins = 20 ,figsize = (15, 15), color="mediumorchid", edgecolor="purple", linewidth=1.2)
plt.suptitle("Histograms of the Numeric Columns", fontsize=30, fontweight="bold")
plt.show()


# Target variable distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["BeatsPerMinute"], kde=True, color="mediumorchid", edgecolor="purple", linewidth=1.2)
plt.title("Distribution of Beats Per Minute", fontweight = "bold" , fontsize=30)
plt.ylabel("Frequency")
plt.xlabel("Beats Per Minute")
plt.show()


plt.figure(figsize=(10, 5))

# Compute correlation matrix
corr = df.corr()
target_corr = corr['BeatsPerMinute'].drop('BeatsPerMinute').abs().sort_values(ascending=False)

# Plot heatmap
sns.heatmap(corr, annot=True, cmap="Purples", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# Feature Engineering credit to @albab12

# Interaction features 
def feature_eng(df):

    df = df.copy()

    # Features
    df["Rhythm_Audio"] = df["RhythmScore"] * df["AudioLoudness"]
    df['Rhythm_Audio_Interaction'] = df['RhythmScore'] * df['AudioLoudness']
    df['Vocal_Acoustic_Ratio'] = df['VocalContent'] / (df['AcousticQuality'] + 1e-6)
    df['Energy_Mood_Product'] = df['Energy'] * df['MoodScore']
    df['Instrumental_Live_Interaction'] = df['InstrumentalScore'] * df['LivePerformanceLikelihood']

    # Log Transform
    #shift = abs(df["TrackDurationMs"].min()) + 1
    #df["log_TrackDuration"] = np.log1p(df["TrackDurationMs"] + shift)
        
    # Polynomial features for top correlated features
    #top_3_features = target_corr.head(3).index.tolist()
    #for feature in top_3_features:
        #df[f'{feature}_squared'] = df[feature] ** 2
        #df[f'{feature}_sqrt'] = np.sqrt(np.abs(df[feature]))

    # Binning Features
    #df["TrackLengthBin"] = pd.qcut(df["TrackDurationMs"], q=10, labels=False)
    #df["EnergyBin"] = pd.qcut(df["Energy"], q=10, labels=False)

    return df

df = feature_eng(df)
test = feature_eng(test)



df.corr()["BeatsPerMinute"].drop('BeatsPerMinute').abs().sort_values(ascending=False)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = df.drop(columns=["BeatsPerMinute", "id", "AudioLoudness",
                     "AcousticQuality","InstrumentalScore"])

y = df["BeatsPerMinute"]

X_t = test[X.columns]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(X_t)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import optuna


# ------------------------------
# Custom CV with Early Stopping
# ------------------------------
#def cv_rmse(model, X, y, fit_params=None):
#    kf = KFold(n_splits=3, shuffle=True, random_state=42)
#    rmses = []
#
#    for train_idx, valid_idx in kf.split(X):
#        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
#        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
#
#        model.fit(
#            X_train, y_train,
#            eval_set=[(X_valid, y_valid)],
#            eval_metric="rmse",
#            early_stopping_rounds=100,
#            verbose=False,
#            **(fit_params if fit_params else {})
#        )
#
#        preds = model.predict(X_valid)
#        rmse = mean_squared_error(y_valid, preds, squared=False)
#        rmses.append(rmse)
#
#    return np.mean(rmses)


# ------------------------------
# Objective function for CatBoost
# ------------------------------
#def objective_cat(trial):
#    params = {
#        "depth": trial.suggest_int("depth", 4, 10),
#        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
#        "n_estimators": 5000,
#        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
#        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#        "bootstrap_type": "Bernoulli",
#        "random_seed": 42,
#        "task_type": "GPU",  # âš¡ GPU
#        "verbose":0
#    }
#    model = CatBoostRegressor(**params)
#    return -cv_rmse(model, X_cat, y)


# ------------------------------
# Objective function for LGBM
# ------------------------------
#def objective_lgbm(trial):
#    params = {
#        "num_leaves": trial.suggest_int("num_leaves", 31, 150),
#        "max_depth": trial.suggest_int("max_depth", 3, 12),
#        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
#        "n_estimators": 5000,  # high number; early stopping will cut
#        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
#        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
#        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
#        "random_state": 42,
#        "objective": "regression",
#        "metric": "rmse",
#        "verbose":0
#    }
#    model = LGBMRegressor(**params)
#    return -cv_rmse(model, X_lgbm, y)


# ------------------------------
# Objective function for XGBoost
# ------------------------------
#def objective_xgb(trial):
#    params = {
#        "max_depth": trial.suggest_int("max_depth", 3, 12),
#        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
#        "n_estimators": 5000,
#        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
#        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#        "reg_alpha": trial.suggest_float("reg_alpha", 0, 0.5),
#        "reg_lambda": trial.suggest_float("reg_lambda", 0, 0.5),
#        "random_state": 42,
#        "tree_method": "hist",
#        "device": "cuda"   # âš¡ GPU
#    }
#    model = XGBRegressor(**params)
#    return -cv_rmse(model, X_xgb, y)


# ------------------------------
# Run separate studies
# ------------------------------
#results = {}

#for name, objective in [
#    ("LightGBM", objective_lgbm),
#    ("XGBoost", objective_xgb),
#    ("CatBoost", objective_cat),
#]:
#    print(f"ğŸ”� Tuning {name}...")
#    study = optuna.create_study(direction="maximize")
#    study.optimize(objective, n_trials=30, n_jobs=1)  # increase n_trials if time allows
#    results[name] = {
#        "best_params": study.best_params,
#        "best_rmse": -study.best_value
#    }



best_xgb_params = {
    'max_depth': 5,
    'learning_rate': 0.01638151846983357,
    'subsample': 0.6458860803573412,
    'colsample_bytree': 0.955703610068519,
    'reg_alpha': 0.3633949566393569,
    'reg_lambda': 0.048850451780827975,
    'n_estimators': 5000,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cuda'
}

best_lgbm_params = {
    'learning_rate': 0.01138151846983357, 
    'n_estimators': 1000, 
    'subsample': 0.638151846983357, 
    'num_leaves': 30,
    'reg_lambda': 0.48850451780827975, 
    'reg_alpha': 0.1, 
    'min_child_samples': 50, 
    'max_depth': 15, 
    'random_state': 294,
    'colsample_bytree': 1.0,
    "verbose":-1
}



best_cat_params = {
    'learning_rate': 0.007367892233447686,
    'bagging_temperature': 2.0, 
    'border_count': 64, 
    'depth': 11, 
    'grow_policy': 'Depthwise', 
    'l2_leaf_reg': 1,
    'min_data_in_leaf': 10, 
    'random_state': 42,
    'random_strength': 0.1,
    'verbose':0
}

best_lgbm = LGBMRegressor(**best_lgbm_params)
best_xgb  = XGBRegressor(**best_xgb_params)
best_cat  = CatBoostRegressor(**best_cat_params)


# Fit models first (on small sample for speed)
X_sm = X.sample(frac=0.1, random_state=42)
y_sm = y.loc[X_sm.index]

best_lgbm.fit(X_sm, y_sm)
best_xgb.fit(X_sm, y_sm)
best_cat.fit(X_sm, y_sm)

# Feature Importances
fi_lgbm = pd.Series(best_lgbm.feature_importances_, index=X.columns, name="LGBM")
fi_xgb  = pd.Series(best_xgb.feature_importances_, index=X.columns, name="XGB")
fi_cat  = pd.Series(best_cat.get_feature_importance(), index=X.columns, name="CAT")

fi_all = pd.concat([fi_lgbm, fi_xgb, fi_cat], axis=1).fillna(0)
fi_all["Mean"] = fi_all.mean(axis=1)
fi_norm = fi_all.div(fi_all.sum(axis=0), axis=1) * 100
top_feats = fi_norm.sort_values("Mean", ascending=False)
print("Best Features across models:\n", top_feats)

# Correlation Matrix
corr = X[top_feats.index].corr()

plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")
sns.heatmap(
    corr, 
    cmap=sns.color_palette("coolwarm", as_cmap=True),  # modern gradient
    center=0, 
    annot=True,  # show correlation values
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Correlation"}
)
plt.title("Correlation Heatmap (Top Features)", fontsize=16, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Feature Importances Side-by-Side
top_feats_long = fi_norm.loc[top_feats.index, ["LGBM", "XGB", "CAT"]]
top_feats_long = top_feats_long.reset_index().melt(
    id_vars="index", 
    value_vars=["LGBM", "XGB", "CAT"],
    var_name="Model", 
    value_name="Importance"
)

plt.figure(figsize=(12, 7))
sns.set_palette("Set2")  # soft, professional palette
sns.barplot(
    data=top_feats_long, 
    x="Importance", 
    y="index", 
    hue="Model", 
    edgecolor="black"
)
plt.title("Feature Importances Across Models", fontsize=16, fontweight="bold")
plt.xlabel("Importance (%)", fontsize=13)
plt.ylabel("Feature", fontsize=13)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.legend(title="Model", fontsize=11, title_fontsize=12)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()


# OOF predictions generator
def get_oof_preds(model, folds=5, random_state=42):
    """
    Returns: oof_preds (np.array), models_trained(list)
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X))
    models = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_tr, y_tr)
        oof[val_idx] = model_clone.predict(X_val)
        models.append(model_clone)

        rmse = mean_squared_error(y_val, oof[val_idx], squared=False)
        print(f"Fold {fold + 1} RMSE: {rmse:.5f}")

    full_rmse = mean_squared_error(y, oof, squared=False)
    print("Full OOF RMSE:", full_rmse)
    return oof, models

print("\nGenerating OOF preds for CatBoost")
oof_cat, cat_models = get_oof_preds(best_cat)

print("\nGenerating OOF preds for LightGBM")
oof_lgbm, lgbm_models = get_oof_preds(best_lgbm)

print("\nGenerating OOF preds for XGBoost")
oof_xgb, xgb_models = get_oof_preds(best_xgb)


# Meta-model training (stacking)
meta_X = pd.DataFrame({
    "cat": oof_cat,
    "lgbm": oof_lgbm
})
meta_y = y.values

meta_model = Ridge(alpha=1.0, random_state=42)
meta_model.fit(meta_X, meta_y)
oof_stack = meta_model.predict(meta_X)
print("Stacked OOF RMSE:", mean_squared_error(meta_y, oof_stack, squared=False))


# Base model predictions on the test set
pred_lgbm_test = best_lgbm.predict(test_scaled)
pred_cat_test  = best_cat.predict(test_scaled)

# Combine into meta-model features
meta_X_test = pd.DataFrame({
    "cat": pred_cat_test,
    "lgbm": pred_lgbm_test
})

# Meta-model (stacked) predictions
stacked_preds = meta_model.predict(meta_X_test)

meta_submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": stacked_preds
})

meta_submission.to_csv("submission_stack_oof.csv", index=False)

print("Saved: submission_stack_oof.csv")


estimators = [
    ("XGBoost", best_xgb),
    ("LGBM", best_lgbm),
    ("CatBoost", best_cat)
]

stack_with_ridge = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    passthrough=False,
    cv=5,
    n_jobs=-1
)
stack_with_ridge.fit(X_scaled, y)
stack_pred = stack_with_ridge.predict(X_test)

rmse = np.sqrt(mean_squared_error(stack_pred,y_test))
print(f"Stacked: {rmse}")


submission_preds = stack_with_ridge.predict(test_scaled)

stack_submit = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": submission_preds
})

stack_submit.to_csv("submission_stack_reg.csv", index=False)

print("Saved: submission_stack_reg.csv")

