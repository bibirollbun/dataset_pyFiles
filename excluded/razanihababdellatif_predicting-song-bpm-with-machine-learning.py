
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))





#  Imports & Global Settings

# import os, sys, glob, math, gc, warnings, textwrap
# warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.model_selection import KFold, train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import Ridge

# # Optional GBMs
# try:
#     from xgboost import XGBRegressor
#     XGB_AVAILABLE = True
# except Exception as e:
#     print("XGBoost not available:", e)
#     XGB_AVAILABLE = False

# try:
#     import lightgbm as lgb
#     from lightgbm import LGBMRegressor
#     LGB_AVAILABLE = True
# except Exception as e:
#     print("LightGBM not available:", e)
#     LGB_AVAILABLE = False

# try:
#     from catboost import CatBoostRegressor, Pool
#     CAT_AVAILABLE = True
# except Exception as e:
#     print("CatBoost not available:", e)
#     CAT_AVAILABLE = False

# RANDOM_STATE = 42
# np.random.seed(RANDOM_STATE)

# # Display options
# pd.set_option("display.max_columns", 200)
# pd.set_option("display.width", 200)



#train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
#test  = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


#print("\nShapes: train =", train.shape, ", test =", test.shape)
#display(train.head())


# # Detect ID column
# def detect_id_column(df):
#     candidates = ["ID", "Id", "id"]
#     for c in candidates:
#         if c in df.columns:
#             return c
#     # fallback: first column if it's integer-like and unique enough
#     first = df.columns[0]
#     return first
#
# ID_COL = detect_id_column(test)
# print("Detected ID column:", ID_COL)
#
# # Detect target column (case-insensitive search for 'BeatsPerMinute')
# def detect_target_column(df):
#     exact = "BeatsPerMinute"
#     if exact in df.columns:
#         return exact
#     lowered = {c.lower(): c for c in df.columns}
#     if "beatsperminute" in lowered:
#         return lowered["beatsperminute"]
#     raise KeyError("Target column 'BeatsPerMinute' not found in training data.")
#
# TARGET = detect_target_column(train)
# print("Detected TARGET column:", TARGET)
#
# # Separate features and target
# y = train[TARGET]
# X = train.drop(columns=[TARGET])
#
# # Ensure test has same feature columns (union might differ if leaks exist)
# # Generally, Kaggle Playground is aligned; we just check and align columns later via transformers.
# test_ids = test[ID_COL].copy()



# # Identify feature types
# num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
# cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
#
# print(f"Numeric columns ({len(num_cols)}):", num_cols[:15], "..." if len(num_cols)>15 else "")
# print(f"Categorical columns ({len(cat_cols)}):", cat_cols[:15], "..." if len(cat_cols)>15 else "")
#
# # Missing values overview
# def missing_table(df):
#     total = df.isnull().sum()
#     percent = (total / len(df)) * 100
#     mt = pd.DataFrame({"missing_count": total, "missing_%": percent})
#     mt = mt[mt["missing_count"] > 0].sort_values(by="missing_%", ascending=False)
#     return mt
#
# print("\nMissing values in TRAIN:")
# mt_train = missing_table(train)
# display(mt_train.head(20))
#
# print("\nMissing values in TEST:")
# mt_test = missing_table(test)
# display(mt_test.head(20))
#
# # Target distribution
# fig = plt.figure(figsize=(6,4))
# plt.hist(y.values, bins=30, edgecolor='black')
# plt.title("Target Distribution: BeatsPerMinute")
# plt.xlabel("BPM")
# plt.ylabel("Frequency")
# plt.show()
#
# # Simple correlations (numeric-only)
# if len(num_cols) > 0:
#     corr_with_target = train[num_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
#     display(pd.DataFrame({"corr_with_BPM": corr_with_target}).head(20))
#
#     # Plot top 20 absolute correlations
#     top_corr = corr_with_target.abs().sort_values(ascending=False).head(20)
#     fig = plt.figure(figsize=(6,6))
#     plt.barh(top_corr.index[::-1], top_corr.values[::-1])
#     plt.title("Top 20 |corr| with BPM (numeric features)")
#     plt.xlabel("|Correlation|")
#     plt.ylabel("Feature")
#     plt.tight_layout()
#     plt.show()
# else:
#     print("No numeric features to correlate with the target.")



# from sklearn.utils.validation import check_is_fitted
#
# # Preprocessor for linear model (scale numerics + OHE cats)
# numeric_transformer_linear = Pipeline(steps=[
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler())
# ])
#
# categorical_transformer_ohe = Pipeline(steps=[
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))
# ])
#
# preprocessor_linear = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer_linear, num_cols),
#         ("cat", categorical_transformer_ohe, cat_cols)
#     ],
#     remainder="drop"
# )
#
# # Preprocessor for tree models (no scaling + OHE cats)
# numeric_transformer_tree = Pipeline(steps=[
#     ("imputer", SimpleImputer(strategy="median"))
# ])
#
# preprocessor_tree = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer_tree, num_cols),
#         ("cat", categorical_transformer_ohe, cat_cols)
#     ],
#     remainder="drop"
# )




# def cv_train_predict(model, preprocessor, X, y, X_test, n_splits=5, model_name="model", use_early_stopping=False):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
#     oof = np.zeros(len(X))
#     preds_test = np.zeros(len(X_test))
#     fold_scores = []
#     feature_names_ = None
#     importances = []
#
#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
#         X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
#         y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
#
#         # Fit preprocessor on training fold
#         preprocessor.fit(X_tr)
#         X_tr_t = preprocessor.transform(X_tr)
#         X_va_t = preprocessor.transform(X_va)
#         X_test_t = preprocessor.transform(X_test)
#
#         # Fit model (with optional early stopping)
#         if use_early_stopping and hasattr(model, "fit"):
#             if "XGBRegressor" in model.__class__.__name__:
#                 model.set_params(random_state=RANDOM_STATE)
#                 model.fit(
#                     X_tr_t, y_tr,
#                     eval_set=[(X_va_t, y_va)],
#                     verbose=False,
#                     early_stopping_rounds=200
#                 )
#             elif "LGBMRegressor" in model.__class__.__name__:
#                 model.set_params(random_state=RANDOM_STATE)
#                 model.fit(
#                     X_tr_t, y_tr,
#                     eval_set=[(X_va_t, y_va)],
#                     callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
#                 )
#             else:
#                 model.fit(X_tr_t, y_tr)
#         else:
#             model.fit(X_tr_t, y_tr)
#
#         # OOF and test predictions
#         oof[va_idx] = model.predict(X_va_t)
#         preds_test += model.predict(X_test_t) / n_splits
#
#         # Track RMSE
#         score = rmse(y_va, oof[va_idx])
#         fold_scores.append(score)
#         print(f"[{model_name}] Fold {fold}: RMSE = {score:.5f}")
#
#         # Feature names after transformation
#         try:
#             feature_names_ = preprocessor.get_feature_names_out()
#         except Exception:
#             feature_names_ = None
#
#         # Feature importances if available
#         if hasattr(model, "feature_importances_"):
#             importances.append(model.feature_importances_)
#
#         gc.collect()
#
#     cv_score = rmse(y, oof)
#     print(f"[{model_name}] CV RMSE (OOF): {cv_score:.5f}")
#     print(f"Fold scores: {np.round(fold_scores, 5)} => mean {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
#
#     # Aggregate importances across folds
#     if importances and feature_names_ is not None:
#         imp = np.mean(np.vstack(importances), axis=0)
#         importance_df = pd.DataFrame({"feature": feature_names_, "importance": imp})
#         importance_df = importance_df.sort_values("importance", ascending=False)
#     else:
#         importance_df = None
#
#     return oof, preds_test, importance_df, fold_scores



# ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
#
# ridge_pipe = Pipeline(steps=[
#     ("pre", preprocessor_linear),
#     ("model", ridge)
# ])
#
# # We pass model inside pipeline so the CV helper is adapted:
# # For simplicity, we'll mimic by calling cv with the underlying parts.
# ridge_oof, ridge_test, ridge_imp, ridge_folds = cv_train_predict(
#     model=ridge,
#     preprocessor=preprocessor_linear,
#     X=X, y=y, X_test=test,
#     n_splits=5, model_name="Ridge", use_early_stopping=False
# )




# if XGB_AVAILABLE:
#     xgb = XGBRegressor(
#         n_estimators=5000,
#         max_depth=8,
#         learning_rate=0.03,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         reg_alpha=0.0,
#         reg_lambda=1.0,
#         tree_method="hist",
#         n_jobs=-1,
#         random_state=RANDOM_STATE
#     )
#
#     xgb_oof, xgb_test, xgb_imp, xgb_folds = cv_train_predict(
#         model=xgb,
#         preprocessor=preprocessor_tree,
#         X=X, y=y, X_test=test,
#         n_splits=5, model_name="XGBoost", use_early_stopping=True
#     )
# else:
#     print("Skipping XGBoost â€” not available in this environment.")
#     xgb_oof = xgb_test = xgb_imp = xgb_folds = None



# if LGB_AVAILABLE:
#     lgbm = LGBMRegressor(
#         n_estimators=5000,
#         learning_rate=0.03,
#         num_leaves=63,
#         max_depth=-1,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         reg_alpha=0.0,
#         reg_lambda=1.0,
#         random_state=RANDOM_STATE,
#         n_jobs=-1
#     )
#
#     lgb_oof, lgb_test, lgb_imp, lgb_folds = cv_train_predict(
#         model=lgbm,
#         preprocessor=preprocessor_tree,
#         X=X, y=y, X_test=test,
#         n_splits=5, model_name="LightGBM", use_early_stopping=True
#     )
# else:
#     print("Skipping LightGBM â€” not available in this environment.")
#     lgb_oof = lgb_test = lgb_imp = lgb_folds = None



# if CAT_AVAILABLE:
#     # Prepare CatBoost-specific datasets
#     # Convert object/bool to category dtype for safety
#     X_cb = X.copy()
#     test_cb = test.copy()
#     for c in cat_cols:
#         X_cb[c] = X_cb[c].astype("category")
#         test_cb[c] = test_cb[c].astype("category")
#
#     # Impute numerics (median) + categoricals (most_frequent)
#     num_imputer = SimpleImputer(strategy="median")
#     cat_imputer = SimpleImputer(strategy="most_frequent")
#
#     X_num = pd.DataFrame(num_imputer.fit_transform(X_cb[num_cols]), columns=num_cols, index=X_cb.index) if num_cols else pd.DataFrame(index=X_cb.index)
#     X_cat = pd.DataFrame(cat_imputer.fit_transform(X_cb[cat_cols]), columns=cat_cols, index=X_cb.index) if cat_cols else pd.DataFrame(index=X_cb.index)
#
#     T_num = pd.DataFrame(num_imputer.transform(test_cb[num_cols]), columns=num_cols, index=test_cb.index) if num_cols else pd.DataFrame(index=test_cb.index)
#     T_cat = pd.DataFrame(cat_imputer.transform(test_cb[cat_cols]), columns=cat_cols, index=test_cb.index) if cat_cols else pd.DataFrame(index=test_cb.index)
#
#     X_cb2 = pd.concat([X_num, X_cat], axis=1)
#     T_cb2 = pd.concat([T_num, T_cat], axis=1)
#
#     # Cat feature indices in the new combined frame
#     cat_feature_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
#
#     # CV loop for CatBoost
#     kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
#     cb_oof = np.zeros(len(X_cb2))
#     cb_test = np.zeros(len(T_cb2))
#     cb_scores = []
#
#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb2, y), 1):
#         X_tr, X_va = X_cb2.iloc[tr_idx], X_cb2.iloc[va_idx]
#         y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
#
#         train_pool = Pool(X_tr, y_tr, cat_features=cat_feature_indices)
#         valid_pool = Pool(X_va, y_va, cat_features=cat_feature_indices)
#         test_pool  = Pool(T_cb2, cat_features=cat_feature_indices)
#
#         cb = CatBoostRegressor(
#             iterations=10000,
#             learning_rate=0.03,
#             depth=8,
#             loss_function="RMSE",
#             eval_metric="RMSE",
#             random_seed=RANDOM_STATE,
#             od_type="Iter",
#             od_wait=200,
#             verbose=False
#         )
#         cb.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=False)
#
#         cb_oof[va_idx] = cb.predict(valid_pool)
#         cb_test += cb.predict(test_pool) / 5
#
#         score = rmse(y_va, cb_oof[va_idx])
#         cb_scores.append(score)
#         print(f"[CatBoost] Fold {fold}: RMSE = {score:.5f}")
#
#     cb_cv = rmse(y, cb_oof)
#     print(f"[CatBoost] CV RMSE (OOF): {cb_cv:.5f}")
#     print(f"Fold scores: {np.round(cb_scores, 5)} => mean {np.mean(cb_scores):.5f} +/- {np.std(cb_scores):.5f}")
# else:
#     print("Skipping CatBoost â€” not available in this environment.")
#     cb_oof = cb_test = None




# def plot_top_importances(importance_df, title="Feature Importance (Top 25)", top_k=25):
#     if importance_df is None or importance_df.empty:
#         print("No importance data to plot.")
#         return
#     top = importance_df.head(top_k)
#     fig = plt.figure(figsize=(6,6))
#     plt.barh(top["feature"][::-1], top["importance"].values[::-1])
#     plt.title(title)
#     plt.xlabel("Importance")
#     plt.ylabel("Feature")
#     plt.tight_layout()
#     plt.show()
#
# if XGB_AVAILABLE and 'xgb_imp' in globals() and xgb_imp is not None:
#     plot_top_importances(xgb_imp, title="XGBoost Feature Importance (Top 25)")
#
# if LGB_AVAILABLE and 'lgb_imp' in globals() and lgb_imp is not None:
#     plot_top_importances(lgb_imp, title="LightGBM Feature Importance (Top 25)")



# blend_candidates = []
# blend_names = []
# blend_oofs = []
# blend_tests = []
# blend_scores = []
#
# # Collect models that actually ran
# if 'ridge_oof' in globals() and ridge_oof is not None:
#     blend_candidates.append(("Ridge", ridge_oof, ridge_test))
#     blend_names.append("Ridge")
#     blend_oofs.append(ridge_oof)
#     blend_tests.append(ridge_test)
#     # Estimate per-fold mean as approximation
#     blend_scores.append(np.mean(ridge_folds))
#
# if XGB_AVAILABLE and 'xgb_oof' in globals() and xgb_oof is not None:
#     blend_candidates.append(("XGB", xgb_oof, xgb_test))
#     blend_names.append("XGB")
#     blend_oofs.append(xgb_oof)
#     blend_tests.append(xgb_test)
#     blend_scores.append(np.mean(xgb_folds))
#
# if LGB_AVAILABLE and 'lgb_oof' in globals() and lgb_oof is not None:
#     blend_candidates.append(("LGB", lgb_oof, lgb_test))
#     blend_names.append("LGB")
#     blend_oofs.append(lgb_oof)
#     blend_tests.append(lgb_test)
#     blend_scores.append(np.mean(lgb_folds))
#
# if CAT_AVAILABLE and 'cb_oof' in globals() and cb_oof is not None:
#     blend_candidates.append(("CatBoost", cb_oof, cb_test))
#     blend_names.append("CatBoost")
#     blend_oofs.append(cb_oof)
#     blend_tests.append(cb_test)
#     # Approximate fold score from cb_scores mean
#     blend_scores.append(np.mean(cb_scores))
#
# if len(blend_candidates) == 0:
#     raise RuntimeError("No models ran successfully. Check earlier cells.")
#
# # Compute weights inversely proportional to CV error (use a small epsilon to avoid div-by-zero)
# eps = 1e-6
# inv_errors = np.array([1.0 / (s + eps) for s in blend_scores])
# weights = inv_errors / inv_errors.sum()
#
# print("Models in blend:", blend_names)
# print("Approx. CV RMSEs:", np.round(blend_scores, 6))
# print("Blend weights:", np.round(weights, 4))
#
# # Weighted OOF and test predictions
# blend_oof_pred = np.average(np.column_stack(blend_oofs), axis=1, weights=weights)
# blend_test_pred = np.average(np.column_stack(blend_tests), axis=1, weights=weights)
#
# blend_cv = rmse(y, blend_oof_pred)
# print(f"[Blend] CV RMSE (OOF): {blend_cv:.5f}")



# # --- Use blend predictions instead of undefined "model" ---
# # Final predictions come from the blending step
# test_preds = blend_test_pred
#
# # --- Save submission ---
# submission_path = '/kaggle/working/submission.csv'
#
# submission = pd.DataFrame({
#     ID_COL: test_ids,          # use the correct detected ID column
#     'BeatsPerMinute': test_preds
# })
#
# submission.to_csv(submission_path, index=False)
#
# print(f"âœ… Submission saved to {submission_path}")
# display(submission.head())
#
# # --- Basic validation before upload ---
# # 1. Check header names
# expected_cols = ['ID', 'BeatsPerMinute']
# if list(submission.columns) == expected_cols:
#     print("âœ… Column names correct")
# else:
#     print("âš ï¸� Column names incorrect:", submission.columns.tolist())
#
# # 2. Check row count matches test set
# if len(submission) == len(test):
#     print("âœ… Row count matches test set")
# else:
#     print(f"âš ï¸� Row count mismatch: {len(submission)} rows vs {len(test)} in test")
#
# # 3. Simulate Kaggle check (file must exist and have 2 columns)
# import os
# if os.path.exists(submission_path) and submission.shape[1] == 2:
#     print("âœ… File ready to enter competition")
# else:
#     print("â�Œ File not ready")



#import shutil

# Remove the catboost_info folder if it exists
#shutil.rmtree("catboost_info", ignore_errors=True)



# if CAT_AVAILABLE:
#     kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
#     cb_oof = np.zeros(len(X_cb2))
#     cb_test = np.zeros(len(T_cb2))
#     cb_folds = []
#
#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb2, y), 1):
#         X_tr, X_va = X_cb2.iloc[tr_idx], X_cb2.iloc[va_idx]
#         y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
#
#         train_pool = Pool(X_tr, y_tr, cat_features=cat_feature_indices)
#         valid_pool = Pool(X_va, y_va, cat_features=cat_feature_indices)
#
#         model_cb = CatBoostRegressor(
#             iterations=5000,
#             learning_rate=0.03,
#             depth=8,
#             loss_function="RMSE",
#             eval_metric="RMSE",
#             random_state=RANDOM_STATE,
#             verbose=False
#         )
#
#         model_cb.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=200, verbose=False)
#
#         cb_oof[va_idx] = model_cb.predict(X_va)
#         cb_test += model_cb.predict(T_cb2) / kf.n_splits
#
#         score = rmse(y_va, cb_oof[va_idx])
#         cb_folds.append(score)
#         print(f"[CatBoost] Fold {fold}: RMSE = {score:.5f}")
#
#     cb_score = rmse(y, cb_oof)
#     print(f"[CatBoost] CV RMSE (OOF): {cb_score:.5f}")
# else:
#     cb_oof = cb_test = cb_folds = None



# # Collect OOF and test predictions from available models
# blend_train = []
# blend_test = []
#
# if ridge_oof is not None: 
#     blend_train.append(ridge_oof)
#     blend_test.append(ridge_test)
#
# if xgb_oof is not None: 
#     blend_train.append(xgb_oof)
#     blend_test.append(xgb_test)
#
# if lgb_oof is not None: 
#     blend_train.append(lgb_oof)
#     blend_test.append(lgb_test)
#
# if cb_oof is not None: 
#     blend_train.append(cb_oof)
#     blend_test.append(cb_test)
#
# # Stack predictions
# blend_train = np.vstack(blend_train).T
# blend_test = np.vstack(blend_test).T
#
# # Meta-learner (Ridge on top of model predictions)
# meta = Ridge(alpha=1.0, random_state=RANDOM_STATE)
# meta.fit(blend_train, y)
# final_preds = meta.predict(blend_test)
#
#
# # Save submission
# submission = pd.DataFrame({ID_COL: test_ids, TARGET: final_preds})
# submission.to_csv("submission.csv", index=False)
# print("âœ… submission.csv saved")
# submission.head()



# import os
# import gc
# import math
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import Ridge, RidgeCV
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# import lightgbm as lgb
# import xgboost as xgb

# RANDOM_STATE = 42
# TARGET = "BeatsPerMinute"
# ID = "id"

# # --------------------------
# # Utility functions
# # --------------------------
# def rmse(y_true, y_pred):
#     return math.sqrt(mean_squared_error(y_true, y_pred))

# def add_features(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     eps = 1e-6
#     if "TrackDurationMs" in df.columns:
#         df["TrackDurationMin"] = df["TrackDurationMs"] / 60000.0
#         df["TrackDurationSec"] = df["TrackDurationMs"] / 1000.0
#         df["LogDuration"] = np.log1p(df["TrackDurationMs"])
#     for a, b, name in [
#         ("RhythmScore","Energy","Rhythm_Energy"),
#         ("AcousticQuality","VocalContent","Acoustic_Vocal"),
#         ("MoodScore","LivePerformanceLikelihood","Mood_Live")]:
#         if a in df.columns and b in df.columns:
#             df[name] = df[a] * df[b]
#     return df

# def cv_train_predict(model, preprocessor, X, y, X_test, n_splits=5, model_name="model", use_early_stopping=False):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
#     oof = np.zeros(len(X))
#     preds_test = np.zeros(len(X_test))
#     fold_scores = []
#     for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
#         X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
#         y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
#         preprocessor.fit(X_tr)
#         X_tr_t = preprocessor.transform(X_tr)
#         X_va_t = preprocessor.transform(X_va)
#         X_test_t = preprocessor.transform(X_test)
#         if use_early_stopping:
#             cls_name = model.__class__.__name__
#             if "XGB" in cls_name:
#                 model.set_params(random_state=RANDOM_STATE)
#                 model.fit(X_tr_t, y_tr, eval_set=[(X_va_t, y_va)], 
#                           early_stopping_rounds=200, verbose=False)
#             elif "LGBM" in cls_name:
#                 model.set_params(random_state=RANDOM_STATE)
#                 model.fit(X_tr_t, y_tr, eval_set=[(X_va_t, y_va)],
#                           callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
#             else:
#                 model.fit(X_tr_t, y_tr)
#         else:
#             model.fit(X_tr_t, y_tr)
#         pred_val = model.predict(X_va_t)
#         oof[va_idx] = pred_val
#         preds_test += model.predict(X_test_t) / n_splits
#         score = rmse(y_va, pred_val)
#         fold_scores.append(score)
#         print(f"[{model_name}] Fold {fold} RMSE: {score:.5f}")
#         gc.collect()
#     cv_score = rmse(y, oof)
#     print(f"[{model_name}] CV RMSE (OOF): {cv_score:.5f}")
#     return oof, preds_test, fold_scores

# # --------------------------
# # Main
# # --------------------------
# def main():
#     train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
#     test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

#     X = train.drop(columns=[TARGET])
#     y = train[TARGET]
#     test_ids = test[ID]

#     X = add_features(X)
#     test_fe = add_features(test)

#     numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
#     X_num = X[numeric_cols].reset_index(drop=True)
#     test_num = test_fe[numeric_cols].reset_index(drop=True)

#     preprocessor = Pipeline([
#         ("imputer", SimpleImputer(strategy="median"))
#     ])

#     # --------------------------
#     # Models
#     # --------------------------
#     ridge = Ridge(alpha=1.0)

#     lgbm = lgb.LGBMRegressor(
#         num_leaves=127,
#         learning_rate=0.005,
#         n_estimators=5000,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         reg_alpha=0.5,
#         reg_lambda=0.5,
#         random_state=RANDOM_STATE,
#         n_jobs=-1
#     )

#     xgbr = xgb.XGBRegressor(
#         n_estimators=4000,
#         learning_rate=0.01,
#         max_depth=10,
#         subsample=0.7,
#         colsample_bytree=0.7,
#         reg_alpha=0.5,
#         reg_lambda=0.5,
#         tree_method="hist",  # efficient for large datasets
#         random_state=RANDOM_STATE,
#         n_jobs=-1
#     )

#     # --------------------------
#     # Training
#     # --------------------------
#     print("\n--- Ridge ---")
#     ridge_oof, ridge_test, ridge_folds = cv_train_predict(ridge, preprocessor, X_num, y, test_num, model_name="Ridge")

#     print("\n--- LightGBM ---")
#     lgb_oof, lgb_test, lgb_folds = cv_train_predict(lgbm, preprocessor, X_num, y, test_num, model_name="LightGBM", use_early_stopping=True)

#     print("\n--- XGBoost ---")
#     xgb_oof, xgb_test, xgb_folds = cv_train_predict(xgbr, preprocessor, X_num, y, test_num, model_name="XGBoost", use_early_stopping=True)

#     # Ensemble by inverse RMSE
#     cv_scores = np.array([rmse(y, ridge_oof), rmse(y, lgb_oof), rmse(y, xgb_oof)])
#     names = ["Ridge", "LGB", "XGB"]
#     print("CV RMSEs:", dict(zip(names, cv_scores)))
#     weights = 1 / (cv_scores + 1e-6)
#     weights = weights / weights.sum()
#     print("Ensemble weights:", dict(zip(names, weights.round(3))))

#     ensemble_test = ridge_test * weights[0] + lgb_test * weights[1] + xgb_test * weights[2]

#     # Stacking
#     oof_stack = np.vstack([ridge_oof, lgb_oof, xgb_oof]).T
#     test_stack = np.vstack([ridge_test, lgb_test, xgb_test]).T
#     meta = RidgeCV(alphas=[0.1, 1.0, 10.0])
#     meta.fit(oof_stack, y)
#     meta_pred = meta.predict(test_stack)
#     meta_oof_pred = meta.predict(oof_stack)
#     meta_cv = rmse(y, meta_oof_pred)
#     print("Meta CV RMSE:", meta_cv)

#     final_preds = meta_pred if meta_cv < np.min(cv_scores) else ensemble_test

#     sub = pd.DataFrame({ID: test_ids, TARGET: final_preds})
#     sub.to_csv("submission.csv", index=False)
#     print("\nâœ… Saved submission.csv")
#     print(sub.head())

# if __name__ == "__main__":
#     main()




# # Kaggle â€” BPM Prediction (Improved LightGBM with Stratified K-Fold)
# # Improvements:
# # - Target binning for StratifiedKFold (better distribution across folds)
# # - Wider Optuna search space
# # - Early stopping + higher n_estimators
# # - Final training with tuned params
# # - Shows submission.head()

# import os, gc, warnings
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import mean_squared_error

# import lightgbm as lgb
# import optuna

# warnings.filterwarnings("ignore")


# # ## Config & helpers

# TARGET = "BeatsPerMinute"
# N_SPLITS = 5
# RANDOM_STATE = 42
# N_TRIALS = 60
# SEED = RANDOM_STATE

# def rmse(y_true, y_pred):
#     return np.sqrt(mean_squared_error(y_true, y_pred))


# # ## Load data

# DATA_DIR = "/kaggle/input/playground-series-s5e9"

# train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
# test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# print("train shape:", train.shape, " test shape:", test.shape)


# train_x = train.drop([TARGET, "id"], axis=1)
# train_y = train[TARGET].copy()
# test_x = test.drop(["id"], axis=1) if "id" in test.columns else test.copy()

# # Align columns
# common_cols = [c for c in train_x.columns if c in test_x.columns]
# train_x = train_x[common_cols]
# test_x = test_x[common_cols]

# # Drop constant cols
# nzv = train_x.columns[train_x.nunique() <= 1].tolist()
# if nzv:
#     train_x.drop(columns=nzv, inplace=True)
#     test_x.drop(columns=nzv, inplace=True)

# X, X_test = train_x, test_x
# print("Number of features:", X.shape[1])


# # ## Stratified bins for regression target

# # Bin target into 10 quantiles for stratification
# bins = pd.qcut(train_y, q=10, duplicates="drop", labels=False)


# # ## Optuna hyperparameter tuning

# def objective(trial):
#     params = {
#         "boosting_type": "gbdt",
#         "objective": "regression",
#         "metric": "rmse",
#         "verbosity": -1,
#         "random_state": SEED,
#         "num_leaves": trial.suggest_int("num_leaves", 16, 512),
#         "max_depth": trial.suggest_int("max_depth", 3, 18),
#         "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.15, log=True),
#         "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
#         "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
#     }

#     cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
#     scores = []

#     for tr_idx, val_idx in cv.split(X, bins):
#         X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
#         y_tr, y_val = train_y.iloc[tr_idx], train_y.iloc[val_idx]

#         model = lgb.LGBMRegressor(n_estimators=5000, **params)
#         model.fit(
#             X_tr, y_tr,
#             eval_set=[(X_val, y_val)],
#             callbacks=[lgb.early_stopping(100, verbose=False)]
#         )

#         preds = model.predict(X_val, num_iteration=model.best_iteration_)
#         scores.append(rmse(y_val, preds))

#     return np.mean(scores)

# # Run Optuna
# study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
# study.optimize(objective, n_trials=N_TRIALS)

# print("Best params:", study.best_params)
# print("Best RMSE:", study.best_value)

# best_params = study.best_params
# best_params.update({
#     "objective": "regression",
#     "metric": "rmse",
#     "verbosity": -1,
#     "random_state": SEED
# })


# # ## Final training with Stratified K-Fold

# skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# oof_preds = np.zeros(len(X))
# test_preds = np.zeros(len(X_test))

# for fold, (tr_idx, val_idx) in enumerate(skf.split(X, bins)):
#     print(f"Fold {fold+1}")
#     X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
#     y_tr, y_val = train_y.iloc[tr_idx], train_y.iloc[val_idx]

#     model = lgb.LGBMRegressor(n_estimators=8000, **best_params)
#     model.fit(
#         X_tr, y_tr,
#         eval_set=[(X_val, y_val)],
#         callbacks=[lgb.early_stopping(200), lgb.log_evaluation(300)]
#     )

#     val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
#     oof_preds[val_idx] = val_pred
#     print(f"  Fold RMSE: {rmse(y_val, val_pred):.5f}")

#     test_preds += model.predict(X_test, num_iteration=model.best_iteration_) / N_SPLITS

#     del model, X_tr, X_val, y_tr, y_val, val_pred
#     gc.collect()

# print("OOF RMSE:", rmse(train_y, oof_preds))


# # ## Submission

# submission = pd.DataFrame({
#     "id": test["id"] if "id" in test.columns else np.arange(len(test)),
#     "BeatsPerMinute": test_preds
# })

# submission.to_csv("submission.csv", index=False)
# print("âœ… submission.csv saved")

# print("\nsubmission.head():")
# print(submission.head())




import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ==========================
# Load data
# ==========================
train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df  = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

TARGET = "BeatsPerMinute"
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

# ==========================
# PCA + IsolationForest (Outlier removal)
# ==========================
num_features = train_df.drop(columns=["id", TARGET]).select_dtypes("number").columns

scaler = StandardScaler()
scaled = scaler.fit_transform(train_df[num_features])

pca = PCA(n_components=3, random_state=42)
pca_features = pca.fit_transform(scaled)

iso = IsolationForest(contamination=0.06, random_state=42)
flags = iso.fit_predict(pca_features)

clean_train = train_df[flags == 1].reset_index(drop=True)

print("Original rows:", len(train_df))
print("After outlier removal:", len(clean_train))
print("Outliers removed:", (flags == -1).sum())

# ==========================
# Train / Test split
# ==========================
X = clean_train.drop(columns=["id", TARGET])
y = clean_train[TARGET]
X_test = test_df.drop(columns=["id"])

# ==========================
# Cross-validation setup
# ==========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
models = []

# ==========================
# Advanced LightGBM Training
# ==========================
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n--- Fold {fold} (Advanced) ---")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=10000,      # large cap, let early stopping decide
        learning_rate=0.01,      # slow but accurate
        num_leaves=127,          # deeper trees
        max_depth=-1,            # no limit
        subsample=0.75,          # row sampling
        colsample_bytree=0.75,   # feature sampling
        bagging_freq=1,          # apply bagging each iteration
        min_child_samples=20,    # minimum data per leaf
        min_split_gain=0.01,     # only split if gain is meaningful
        lambda_l1=1.0,           # L1 regularization
        lambda_l2=1.0,           # L2 regularization
        random_state=42 + fold,
        n_jobs=-1
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(300), lgb.log_evaluation(200)]
    )
    
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / kf.n_splits
    models.append(model)
    
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    r2 = r2_score(y_val, val_pred)
    print(f"Fold {fold}: RMSE={rmse:.4f}, RÂ²={r2:.4f}")

# ==========================
# Final OOF metrics
# ==========================
final_rmse = np.sqrt(mean_squared_error(y, oof_preds))
final_mae = mean_absolute_error(y, oof_preds)
final_r2 = r2_score(y, oof_preds)

print("\n=== Final OOF Results ===")
print("RMSE:", round(final_rmse, 4))
print("MAE :", round(final_mae, 4))
print("RÂ²  :", round(final_r2, 4))

# ==========================
# Feature Importance
# ==========================
importances = np.mean([m.feature_importances_ for m in models], axis=0)
feat_imp = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)

print("\nTop 10 important features:")
print(feat_imp.head(10))

# ==========================
# Create Submission
# ==========================
submission = pd.DataFrame({"id": test_df["id"], TARGET: test_preds})
submission.to_csv("submission.csv", index=False)

print("\nâœ… Advanced submission saved as 'submission.csv'")
print(submission.head())


