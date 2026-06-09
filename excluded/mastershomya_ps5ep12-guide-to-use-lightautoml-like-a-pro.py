!pip install lightautoml


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.metrics import roc_auc_score

SEED = 42

# ============================================
# 1. Load Data
# ============================================

df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

df_tr = df_train.drop(columns=["id"]).copy()
df_te = df_test.drop(columns=["id"]).copy()

df_tr["diagnosed_diabetes"] = df_tr["diagnosed_diabetes"].astype(int)

# ============================================
# 2. Basic ordinal mappings (We will not do any "one hot encoding" we just need to do ordinal encoding)
# ============================================

edu_map = {'No formal':0,'Highschool':1,'Graduate':2,'Postgraduate':3}
income_map = {'Low':0,'Lower-Middle':1,'Middle':2,'Upper-Middle':3,'High':4}
smoke_map = {'Never':0,'Former':1,'Current':2}

for col, mp in zip(
    ["education_level","income_level","smoking_status"],
    [edu_map, income_map, smoke_map]
):
    df_tr[col] = df_tr[col].map(mp)
    df_te[col] = df_te[col].map(mp)

# ============================================
# 3. Prepare training and test datasets
# ============================================

y = df_tr["diagnosed_diabetes"]
X = df_tr.drop(columns=["diagnosed_diabetes"])
X_test = df_te.copy()
test_ids = df_test["id"]

train_data = pd.concat([X, y], axis=1)

# ============================================
# 4. LightAutoML Model
# ============================================

task = Task("binary")

automl = TabularAutoML(
    task=task,
    timeout=3600,        # 1 hour (increase if you want even stronger but will take more time)
    cpu_limit=4,
    reader_params={
        "n_jobs": 4,
        "cv": 5,         # INTERNAL CV=5 - increase if you want even stronger model
        "random_state": SEED
    }
)

print("Training the LightAutoML model...")
oof_pred = automl.fit_predict(
    train_data,
    roles={"target": "diagnosed_diabetes"}
)

# Evaluate full OOF AUC (internal CV)
print("====================================")
print("Internal CV AUC:", roc_auc_score(y, oof_pred.data[:,0]))
print("====================================")

# ============================================
# 5. Predict on Test
# ============================================

print("Predicting test...")
test_pred = automl.predict(X_test).data[:, 0]

df_sub = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": test_pred
})

df_sub.to_csv("submission_lightautoml.csv", index=False)
print("Saved submission_lightautoml.csv")


print("============= MODEL ARCHITECTURE =============")
print(automl.create_model_str_desc())


# Access Level 0 (It is a list of pipelines)
level_0 = automl.levels[0]

print(f"Type of Level 0: {type(level_0)}") # its just a sanity check

print("\n============= HYPERPARAMETERS =============")
if isinstance(level_0, list):
    pipelines_to_check = level_0
else:
    pipelines_to_check = level_0.pipes.values()

for pipe in pipelines_to_check:
    # Each 'pipe' holds algorithms
    if hasattr(pipe, 'ml_algos'):
        for algo in pipe.ml_algos:
            print(f"\n -> Model: {algo.name}")
            
            # Try to get params from the first fold's model
            try:
                # The actual model object (LGBM/CatBoost) is inside .models[0]
                model_obj = algo.models[0]
                
                # Different libraries store params differently
                if hasattr(model_obj, 'get_params'):
                    params = model_obj.get_params()
                elif hasattr(model_obj, 'params'): # Common for LightGBM
                    params = model_obj.params
                else:
                    params = {"Info": "Could not extract params directly"}

                # Print interesting ones
                print("   Key Params:")
                relevant_keys = ['learning_rate', 'num_leaves', 'depth', 'iterations', 
                                 'n_estimators', 'reg_lambda', 'reg_alpha', 'min_child_samples']
                
                for k, v in params.items():
                    if k in relevant_keys:
                        print(f"     • {k}: {v}")
            except Exception as e:
                print(f"   (Could not extract params: {e})")


feat_importance = automl.get_feature_scores()
print("Actual Columns:", feat_importance.columns) 
plt.figure(figsize=(10, 6))
sns.barplot(
    data=feat_importance.head(15), 
    x='Importance',
    y='Feature'
)
plt.title("Top 15 Features used by LightAutoML")
plt.show()


# from lightautoml.automl.presets.tabular_presets import TabularAutoML
# from lightautoml.tasks import Task
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score

# SEED = 42
# NFOLDS = 5

# # =====================================================
# # 1. Prepare RAW data for LAMA (NO manual preprocessing)
# # =====================================================

# df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
# df_test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# df_tr = df_train.drop(columns=["id"])
# df_tr['diagnosed_diabetes'] = df_tr['diagnosed_diabetes'].astype(int)

# df_te = df_test.drop(columns=["id"])

# edu_map = {
#     'No formal': 0,
#     'Highschool': 1,
#     'Graduate': 2,
#     'Postgraduate': 3
# }

# income_map = {
#     'Low': 0,
#     'Lower-Middle': 1,
#     'Middle': 2,
#     'Upper-Middle': 3,
#     'High': 4
# }

# smoke_map = {
#     'Never': 0,
#     'Former': 1,
#     'Current': 2
# }

# df_tr['education_level'] = df_tr['education_level'].map(edu_map)
# df_tr['income_level'] = df_tr['income_level'].map(income_map)
# df_tr['smoking_status'] = df_tr['smoking_status'].map(smoke_map)

# df_te['education_level'] = df_te['education_level'].map(edu_map)
# df_te['income_level'] = df_te['income_level'].map(income_map)
# df_te['smoking_status'] = df_te['smoking_status'].map(smoke_map)

# y = df_tr["diagnosed_diabetes"]
# X = df_tr.drop(columns=["diagnosed_diabetes"])
# test_ids = df_test["id"]
# X_test = df_te.copy()

# # =====================================================
# # 2. OOF + TEST containers
# # =====================================================

# oof_preds = np.zeros(len(X))
# test_preds = np.zeros(len(X_test))

# skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

# # =====================================================
# # 3. LightAutoML OOF Loop
# # =====================================================

# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#     print(f"\n===== FOLD {fold+1} / {NFOLDS} =====")
    
#     X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     # Setup LAMA Task
#     task = Task('binary')
    
#     automl = TabularAutoML(
#         task=task,
#         timeout=1800,          # 30 min per fold
#         cpu_limit=4,
#         reader_params={
#             'n_jobs': 4,
#             'cv': 3,           # VERY IMPORTANT: disable internal CV here 
#             'random_state': SEED
#         }
#     )
    
#     train_data = pd.concat([X_tr, y_tr], axis=1)
#     oof_fold = automl.fit_predict(
#         train_data,
#         roles={'target': 'diagnosed_diabetes'}
#     )
    
#     # Predict validation fold
#     val_pred = automl.predict(X_val).data[:, 0]
#     oof_preds[val_idx] = val_pred
    
#     # Predict test (average)
#     test_pred_fold = automl.predict(X_test).data[:, 0]
#     test_preds += test_pred_fold / NFOLDS

# # =====================================================
# # 4. Evaluate OOF
# # =====================================================

# print("\n===================================================")
# print("Final LightAutoML OOF AUC:", roc_auc_score(y, oof_preds))
# print("===================================================\n")

# # =====================================================
# # 5. Save OOF + Test styled like your other models
# # =====================================================

# df_oof = pd.DataFrame({
#     "id": df_train["id"],
#     "diagnosed_diabetes": y,
#     "lightautoml_pred": oof_preds
# })
# df_oof.to_csv("oof_lightautoml.csv", index=False)

# df_test_out = pd.DataFrame({
#     "id": test_ids,
#     "lightautoml_pred": test_preds
# })
# df_test_out.to_csv("test_lightautoml.csv", index=False)

# print("Saved oof_lightautoml.csv and test_lightautoml.csv!")

