# import pandas as pd
# import numpy as np
# from scipy.stats import rankdata
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import roc_auc_score

# # Load data
# train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
# test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

# # Define columns
# target_col = "loan_paid_back"
# id_col = "id"

# # Separate features and target
# X = train.drop(columns=[target_col])
# y = train[target_col]

# # Handle missing values
# for c in X.columns:
#     if X[c].dtype in ['int64', 'float64']:
#         X[c] = X[c].fillna(X[c].median())
#         test[c] = test[c].fillna(X[c].median())
#     else:
#         X[c] = X[c].fillna(X[c].mode()[0])
#         test[c] = test[c].fillna(X[c].mode()[0])

# # Label Encoding for categorical columns
# for c in X.select_dtypes(include='object').columns:
#     le = LabelEncoder()
#     le.fit(pd.concat([X[c], test[c]], axis=0))
#     X[c] = le.transform(X[c])
#     test[c] = le.transform(test[c])

# # Split train-validation
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X.drop(columns=[id_col]), y, test_size=0.15, random_state=42
# )

# # Bhai jaan models

# # 1. XGBoost
# xgb_model = XGBClassifier(
#     n_estimators=400,
#     learning_rate=0.05,
#     max_depth=5,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     eval_metric='auc',
#     random_state=42
# )
# xgb_model.fit(X_train, y_train)
# xgb_pred = xgb_model.predict_proba(X_valid)[:, 1]

# # 2. CatBoost
# cat_model = CatBoostClassifier(
#     iterations=400,
#     learning_rate=0.05,
#     depth=5,
#     eval_metric='AUC',
#     random_state=42,
#     verbose=False
# )
# cat_model.fit(X_train, y_train)
# cat_pred = cat_model.predict_proba(X_valid)[:, 1]

# # 3. Random Forest
# rf_model = RandomForestClassifier(
#     n_estimators=400,
#     max_depth=5,
#     random_state=42
# )
# rf_model.fit(X_train, y_train)
# rf_pred = rf_model.predict_proba(X_valid)[:, 1]

# # 4. Decision Tree
# dt_model = DecisionTreeClassifier(
#     max_depth=5,
#     random_state=42
# )
# dt_model.fit(X_train, y_train)
# dt_pred = dt_model.predict_proba(X_valid)[:, 1]

# # Dilbar Ensemble

# # val_pred = (xgb_pred + cat_pred + rf_pred + dt_pred) / 4

# # print(f"Validation AUC (4-model ensemble): {roc_auc_score(y_valid, val_pred):.5f}")

# # Weighted Ensemble
# # w1, w2 = 0.6, 0.4  # Adjust weights based on validation scores
# # val_pred = (xgb_pred * w1 + cat_pred * w2)
# # print(f"Validation AUC: {roc_auc_score(y_valid, val_pred):.5f}")


# # Rankdata
# # val_pred = (rankdata(xgb_pred) + rankdata(cat_pred)) / 2
# # val_pred = val_pred / val_pred.max()  # normalize to [0,1]
# # print(f"Validation AUC: {roc_auc_score(y_valid, val_pred):.5f}")

# # Stacking (Meta-model Ensemble)
# stack_X = np.column_stack([xgb_pred + cat_pred + rf_pred + dt_pred])
# meta_model = LogisticRegression()
# meta_model.fit(stack_X, y_valid)
# val_pred = meta_model.predict_proba(stack_X)[:, 1]
# print(f"Validation AUC: {roc_auc_score(y_valid, val_pred):.5f}")

# # Final Test Prediction

# final_pred_prob = (
#     xgb_model.predict_proba(test.drop(columns=[id_col]))[:, 1] +
#     cat_model.predict_proba(test.drop(columns=[id_col]))[:, 1] +
#     rf_model.predict_proba(test.drop(columns=[id_col]))[:, 1] +
#     dt_model.predict_proba(test.drop(columns=[id_col]))[:, 1]
# ) / 4

# # Save Submission

# submission = pd.DataFrame({
#     id_col: test[id_col],
#     target_col: final_pred_prob  # val_pred
# })
# submission.to_csv("submission1.csv", index=False)
# print("submission_4model_ensemble.csv saved successfully!")


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import roc_auc_score
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

# # Load data
# train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
# test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

# # Define columns
# target_col = "loan_paid_back"
# id_col = "id"

# # Separate features and target
# X = train.drop(columns=[target_col])
# y = train[target_col]

# # Handle missing values
# for c in X.columns:
#     if X[c].dtype in ['int64', 'float64']:
#         X[c] = X[c].fillna(X[c].median())
#         test[c] = test[c].fillna(X[c].median())
#     else:
#         X[c] = X[c].fillna(X[c].mode()[0])
#         test[c] = test[c].fillna(X[c].mode()[0])

# # Label Encoding for categorical columns
# for c in X.select_dtypes(include='object').columns:
#     le = LabelEncoder()
#     le.fit(pd.concat([X[c], test[c]], axis=0))
#     X[c] = le.transform(X[c])
#     test[c] = le.transform(test[c])

# # Split train-validation
# X_train, X_valid, y_train, y_valid = train_test_split(
#     X.drop(columns=[id_col]), y, test_size=0.20, random_state=42
# )

# # ------------------ MODELS ------------------

# preds = []

# # 1. XGBoost (default)
# xgb1 = XGBClassifier(
#     n_estimators=600, learning_rate=0.05, max_depth=8,
#     subsample=0.8, colsample_bytree=0.8, eval_metric='auc', random_state=42
# )
# xgb1.fit(X_train, y_train)
# preds.append(xgb1.predict_proba(X_valid)[:, 1])

# # 2. XGBoost (variant - deeper)
# xgb2 = XGBClassifier(
#     n_estimators=800, learning_rate=0.03, max_depth=12,
#     subsample=0.9, colsample_bytree=0.7, eval_metric='auc', random_state=42
# )
# xgb2.fit(X_train, y_train)
# preds.append(xgb2.predict_proba(X_valid)[:, 1])

# # 3. CatBoost (default)
# cat1 = CatBoostClassifier(
#     iterations=600, learning_rate=0.05, depth=8,
#     eval_metric='AUC', random_state=42, verbose=False
# )
# cat1.fit(X_train, y_train)
# preds.append(cat1.predict_proba(X_valid)[:, 1])

# # 4. CatBoost (variant - deeper + slower)
# cat2 = CatBoostClassifier(
#     iterations=800, learning_rate=0.03, depth=10,
#     eval_metric='AUC', random_state=42, verbose=False
# )
# cat2.fit(X_train, y_train)
# preds.append(cat2.predict_proba(X_valid)[:, 1])

# # 5. Random Forest (default)
# rf1 = RandomForestClassifier(
#     n_estimators=400, max_depth=5, random_state=42, n_jobs=-1
# )
# rf1.fit(X_train, y_train)
# preds.append(rf1.predict_proba(X_valid)[:, 1])

# # 6. Random Forest (variant - deeper)
# rf2 = RandomForestClassifier(
#     n_estimators=600, max_depth=10, random_state=42, n_jobs=-1
# )
# rf2.fit(X_train, y_train)
# preds.append(rf2.predict_proba(X_valid)[:, 1])

# # 7. Random Forest (variant - shallow + many estimators)
# rf3 = RandomForestClassifier(
#     n_estimators=1000, max_depth=4, random_state=42, n_jobs=-1
# )
# rf3.fit(X_train, y_train)
# preds.append(rf3.predict_proba(X_valid)[:, 1])

# # 8. Decision Tree (default)
# dt1 = DecisionTreeClassifier(max_depth=5, random_state=42)
# dt1.fit(X_train, y_train)
# preds.append(dt1.predict_proba(X_valid)[:, 1])

# # 9. Decision Tree (variant - deeper)
# dt2 = DecisionTreeClassifier(max_depth=8, random_state=42)
# dt2.fit(X_train, y_train)
# preds.append(dt2.predict_proba(X_valid)[:, 1])

# # 10. Decision Tree (variant - shallow)
# dt3 = DecisionTreeClassifier(max_depth=3, random_state=42)
# dt3.fit(X_train, y_train)
# preds.append(dt3.predict_proba(X_valid)[:, 1])

# # ------------------ ENSEMBLE ------------------

# val_pred = np.mean(preds, axis=0)
# print(f"Validation AUC (10-model ensemble): {roc_auc_score(y_valid, val_pred):.5f}")

# # ------------------ TEST PREDICTIONS ------------------

# test_preds = []
# for model in [xgb1, xgb2, cat1, cat2, rf1, rf2, rf3, dt1, dt2, dt3]:
#     test_preds.append(model.predict_proba(test.drop(columns=[id_col]))[:, 1])

# final_pred_prob = np.mean(test_preds, axis=0)

# # ------------------ SAVE ------------------

# submission = pd.DataFrame({
#     id_col: test[id_col],
#     target_col: final_pred_prob
# })
# submission.to_csv("submission.csv", index=False)
# print("submission_10model_ensemble.csv saved successfully!")


import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import rankdata

# CONFIG
old_folder = Path("/kaggle/input/03-november-2025-ps-s5e11")  # folder containing your submissions
output_path = Path("/kaggle/working/submission.csv")

target_col = "loan_paid_back"
id_col = "id"
use_rank_blending = False  # set True for rank averaging (more robust)

# LOAD ALL FILES
submission_files = list(old_folder.glob("*.csv"))
print(f"Found {len(submission_files)} submission files.")

# Preload all files efficiently
dfs = []
for f in submission_files:
    try:
        df = pd.read_csv(f, usecols=[id_col, target_col])
        df = df.set_index(id_col)
        dfs.append(df)
    except Exception as e:
        print(f"Skipping {f.name} due to error: {e}")

# VALIDATION
if not dfs:
    raise ValueError("No valid submission files loaded!")

# Combine all models efficiently
blend_df = pd.concat(dfs, axis=1)
blend_df.columns = [f"model_{i+1}" for i in range(len(dfs))]

# BLENDING
if use_rank_blending:
    # Rank averaging (robust to outliers)
    rank_blend = blend_df.apply(lambda x: rankdata(x) / len(x))
    blend_df["final_pred"] = rank_blend.mean(axis=1)
else:
    # Simple mean
    blend_df["final_pred"] = blend_df.mean(axis=1)

# OUTPUT
final_submission = blend_df[["final_pred"]].reset_index()
final_submission.columns = [id_col, target_col]
final_submission.to_csv(output_path, index=False)

print(f"Final ensemble saved as: {output_path}")

