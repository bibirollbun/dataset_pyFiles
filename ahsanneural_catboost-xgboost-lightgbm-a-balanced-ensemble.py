import pandas as pd
import numpy as np

train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

train.head()


train.info()


train["diagnosed_diabetes"].value_counts(normalize=True)


cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level',
            'smoking_status', 'employment_status']

cat_cols


for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")


target = "diagnosed_diabetes"
id_col = "id"

X = train.drop(columns=[target, id_col])
y = train[target]
X_test = test.drop(columns=[id_col])


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n===== Fold {fold+1} =====")

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    train_ds = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
    valid_ds = lgb.Dataset(X_val, y_val, categorical_feature=cat_cols)

    model = lgb.train(
        params,
        train_ds,
        valid_sets=[train_ds, valid_ds],
        num_boost_round=5000,
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(200)
        ]
    )

    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / 5

cv_auc = roc_auc_score(y, oof)
print("\nOverall CV AUC:", cv_auc)


# Pulse pressure (blood pressure spread)
train["pulse_pressure"] = train["systolic_bp"] - train["diastolic_bp"]
test["pulse_pressure"] = test["systolic_bp"] - test["diastolic_bp"]

# Cholesterol ratios
train["cholesterol_ratio"] = train["cholesterol_total"] / (train["hdl_cholesterol"] + 1)
test["cholesterol_ratio"] = test["cholesterol_total"] / (test["hdl_cholesterol"] + 1)

train["ldl_hdl_ratio"] = train["ldl_cholesterol"] / (train["hdl_cholesterol"] + 1)
test["ldl_hdl_ratio"] = test["ldl_cholesterol"] / (test["hdl_cholesterol"] + 1)

train["tg_hdl_ratio"] = train["triglycerides"] / (train["hdl_cholesterol"] + 1)
test["tg_hdl_ratio"] = test["triglycerides"] / (test["hdl_cholesterol"] + 1)


train["activity_per_bmi"] = train["physical_activity_minutes_per_week"] / (train["bmi"] + 1)
test["activity_per_bmi"] = test["physical_activity_minutes_per_week"] / (test["bmi"] + 1)

train["screen_sleep_ratio"] = train["screen_time_hours_per_day"] / (train["sleep_hours_per_day"] + 1)
test["screen_sleep_ratio"] = test["screen_time_hours_per_day"] / (test["sleep_hours_per_day"] + 1)


bins = [0, 30, 45, 60, 80, 120]
labels = ["young", "adult", "mid_age", "senior", "elder"]

train["age_group"] = pd.cut(train["age"], bins=bins, labels=labels)
test["age_group"] = pd.cut(test["age"], bins=bins, labels=labels)


cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level',
            'smoking_status', 'employment_status', 'age_group']

for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")


target = "diagnosed_diabetes"
id_col = "id"

X = train.drop(columns=[target, id_col])
y = train[target]
X_test = test.drop(columns=[id_col])


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n===== Fold {fold+1} =====")

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    train_ds = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
    valid_ds = lgb.Dataset(X_val, y_val, categorical_feature=cat_cols)

    model = lgb.train(
        params,
        train_ds,
        valid_sets=[train_ds, valid_ds],
        num_boost_round=5000,
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(200)
        ]
    )

    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / 5

cv_auc = roc_auc_score(y, oof)
print("\nOverall CV AUC:", cv_auc)


lgb_preds = test_preds


from catboost import CatBoostClassifier, Pool


train_pool = Pool(X, y, cat_features=cat_cols)
test_pool = Pool(X_test, cat_features=cat_cols)


cat_model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    iterations=5000,
    random_seed=42,
    od_type="Iter",
    od_wait=200,
    verbose=200
)


cat_model.fit(train_pool, eval_set=train_pool)


cat_preds = cat_model.predict_proba(test_pool)[:, 1]

sub = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": cat_preds
})

sub.to_csv("submission_catboost.csv", index=False)
print("Saved submission_catboost.csv")


from IPython.display import FileLink
FileLink('submission_catboost.csv')


import xgboost as xgb


X_xgb = X.copy()
X_test_xgb = X_test.copy()

for col in cat_cols:
    X_xgb[col] = X_xgb[col].cat.codes
    X_test_xgb[col] = X_test_xgb[col].cat.codes


dtrain = xgb.DMatrix(X_xgb, label=y)
dtest = xgb.DMatrix(X_test_xgb)


# 3. XGBoost parameters
# -----------------------------------------
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.03,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "seed": 42
}

# -----------------------------------------
# 4. Train XGBoost with visible logs
# -----------------------------------------
watchlist = [(dtrain, "train")]

xgb_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=2000,
    evals=watchlist,
    verbose_eval=200   # print progress every 200 rounds
)

# -----------------------------------------
# 5. Predict on test set
# -----------------------------------------
xgb_preds = xgb_model.predict(dtest)

print("XGBoost predictions generated!")


final_preds = (
    0.50 * cat_preds +
    0.30 * xgb_preds +
    0.20 * lgb_preds
)

sub = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": final_preds
})

sub.to_csv("submission_blend.csv", index=False)
print("Saved submission_blend.csv")


from IPython.display import FileLink
FileLink('submission_blend.csv')


import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display, HTML

# ============================================================
# 1. Collect model metrics
# ============================================================

metrics = {
    "Model": ["LightGBM", "CatBoost (no CV)", "XGBoost"],
    "Train AUC / CV AUC": [
        cv_auc,      # LightGBM CV AUC
        None,        # CatBoost CV not available
        0.8296       # XGBoost train AUC (from logs)
    ]
}

metrics_df = pd.DataFrame(metrics)

# ============================================================
# 2. Plot model performance comparison
# ============================================================

fig_auc = px.bar(
    metrics_df,
    x="Model",
    y="Train AUC / CV AUC",
    title="Model Performance Comparison (Train/CV AUC)",
    text="Train AUC / CV AUC",
    color="Model"
)

fig_auc.update_traces(texttemplate='%{text}', textposition='outside')
fig_auc.update_layout(yaxis=dict(range=[0.60, 0.90]))

fig_auc.show()

# ============================================================
# 3. Combine predictions into a comparison table
# ============================================================

comparison_df = pd.DataFrame({
    "id": test["id"],
    "LightGBM": lgb_preds,
    "CatBoost": cat_preds,
    "XGBoost": xgb_preds,
    "Blend": final_preds
})

display(HTML("<h2>Prediction Comparison Table</h2>"))
display(comparison_df.head(20))

# ============================================================
# 4. Plot prediction distributions
# ============================================================

fig_dist = go.Figure()

fig_dist.add_trace(go.Histogram(x=lgb_preds, name="LightGBM", opacity=0.5))
fig_dist.add_trace(go.Histogram(x=cat_preds, name="CatBoost", opacity=0.5))
fig_dist.add_trace(go.Histogram(x=xgb_preds, name="XGBoost", opacity=0.5))
fig_dist.add_trace(go.Histogram(x=final_preds, name="Blend", opacity=0.5))

fig_dist.update_layout(
    title="Prediction Distribution Comparison",
    barmode="overlay",
    xaxis_title="Predicted Probability",
    yaxis_title="Count"
)

fig_dist.show()

# ============================================================
# 5. Correlation heatmap between model predictions
# ============================================================

corr = comparison_df[["LightGBM", "CatBoost", "XGBoost", "Blend"]].corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    title="Correlation Between Model Predictions",
    color_continuous_scale="Blues"
)

fig_corr.show()


# âœ… Blend 2: More balanced ensemble
final_preds = (
    0.40 * cat_preds +
    0.30 * xgb_preds +
    0.30 * lgb_preds
)

# âœ… Create submission file
sub = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": final_preds
})

sub.to_csv("submission_blend.csv", index=False)
print("Saved submission_blend.csv")


from IPython.display import FileLink
FileLink('submission_blend.csv')

