import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from sklearn.pipeline import Pipeline 
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from scipy.optimize import minimize

# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


df_diabetes = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col='id')
ori = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')


#----------------------------------------------------------------------------------------------------------------------------
# Function for columns definition
def grab_col_names(df, target=None, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if df[col].dtype in ["O", "category", "bool"]]
    num_but_cat = [col for col in df.columns 
                   if df[col].nunique() < cat_th and df[col].dtype in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns 
                   if df[col].nunique() > car_th and df[col].dtype in ["O", "category"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    if target:
        for col_list in [cat_cols, num_cols, cat_but_car, num_but_cat]:
            if target in col_list:
                col_list.remove(target)
    cat_cols = [col for col in cat_cols if col not in num_but_cat]
    print("-" * 20)
    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    print("-" * 20)
    print('cat_cols:\n',cat_cols)
    print('num_cols:\n',num_cols)
    print('cat_but_car:\n',cat_but_car)
    print('num_but_cat:\n',num_but_cat)
    print("-" * 20)    
    return cat_cols, num_cols, cat_but_car, num_but_cat

'''cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df=df, target='target')'''
#----------------------------------------------------------------------------------------------------------------------------
# Function for optimized weights for blend
def auc_ensemble(weights):
    # Normalización defensiva (por estabilidad numérica)
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    pred = np.dot(blend_train, weights)
    
    # Negativo porque minimize minimiza
    return -roc_auc_score(y_train_meta, pred)
#----------------------------------------------------------------------------------------------------------------------------
# Function for submission files
def make_submission(predictions, filename):
    submission = pd.DataFrame({
        "id": df_test.index,
        "accident_risk": predictions
    })
    submission.to_csv(filename, index=False)
    return submission.head()


cols_extra_en_ori = set(ori.columns) - set(df_diabetes.columns)
cols_extra_en_ori


df_final = pd.concat(
    [df_diabetes, ori[df_diabetes.columns]],
    axis=0,
    ignore_index=True
)

df_final.shape


def features_proxy(df):
    # Age-based risk
    df["age_risk_flag"] = (df["age"] >= 35).astype(int)
    
    # BMI risk
    df["bmi_overweight_flag"] = (df["bmi"] >= 25).astype(int)
    df["bmi_obese_flag"] = (df["bmi"] >= 30).astype(int)
    
    # Hipertensión proxy
    df["bp_risk_flag"] = (
        (df["systolic_bp"] >= 130) |
        (df["diastolic_bp"] >= 80)
    ).astype(int)
    
    # Dislipidemia proxy
    df["lipid_risk_flag"] = (
        (df["ldl_cholesterol"] >= 130) |
        (df["triglycerides"] >= 150) |
        (df["hdl_cholesterol"] < 40)
    ).astype(int)
    
    # lifestyle
    df["low_activity_flag"] = (
        df["physical_activity_minutes_per_week"] < 150
    ).astype(int)
    df["poor_diet_flag"] = (df["diet_score"] < 5).astype(int)
    
    # diabetes_risk_score
    risk_features = [
        "age_risk_flag",
        "bmi_overweight_flag",
        "bp_risk_flag",
        "lipid_risk_flag",
        "low_activity_flag",
        "family_history_diabetes",
    ]
    df["metabolic_risk_score"] = df[risk_features].sum(axis=1)
    
    # Some aditional features
    df["age_bmi_interaction"] = df["age"] * df["bmi"]
    df["bmi_activity_ratio"] = df["bmi"] / (df["physical_activity_minutes_per_week"] + 1)
    
    return df


df_final = features_proxy(df_final)
df_test = features_proxy(df_test)

df_final.shape


Target = 'diagnosed_diabetes'
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df=df_final, target=Target)


X = df_final.drop([Target], axis=1)
y = df_final[Target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69, stratify=y)


ordinal_features = ["education_level", "income_level"]

ordinal_categories = [
    ["No formal", "Highschool", "Graduate", "Postgraduate"],
    ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"]
]

nominal_features = [
    "gender",
    "ethnicity",
    "smoking_status",
    "employment_status"
]

numeric_scaled = ["alcohol_consumption_per_week"]

preprocessor = ColumnTransformer(
    transformers=[
        ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal_features),
        ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_features),
        ("num", RobustScaler(), num_cols),
        ("num_scaled", StandardScaler(), numeric_scaled)
    ],
    remainder="passthrough"
)


# XGBoost
xgboost = XGBClassifier(
    random_state=69,
    n_estimators=10000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.5,
    gamma=0.01,
    reg_lambda=1.0,
    reg_alpha=0.3,
    n_jobs=-1,
    device="cuda",
    eval_metric="auc",
    objective="binary:logistic"
)

# LightGBM
lightgbm = LGBMClassifier(
    random_state=69,
    n_estimators=10000,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_samples=50,
    n_jobs=-1,
    verbose=-1,
    device="gpu",
    objective="binary",
    metric="auc"
)

# CatBoost
catboost = CatBoostClassifier(
    random_state=69,
    iterations=10000,
    learning_rate=0.01,
    depth=6,
    l2_leaf_reg=3,
    bagging_temperature=1,
    task_type="CPU", 
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=0
)


# XGBOOST
XGB_pipeline = Pipeline([
    ('preprocessor', preprocessor),  
    ('model', xgboost)
    ])

# LGBM
LGBM_pipeline = Pipeline([
    ('preprocessor', preprocessor),  
    ('model', lightgbm)
    ])

#CATBOOST
CAT_pipeline = Pipeline([
    ('preprocessor', preprocessor),  
    ('model', catboost)
    ])



# pipelines = {
#     "XGBoost": XGB_pipeline,
#     "LightGBM": LGBM_pipeline,
#     "CatBoost": CAT_pipeline
# }

# results = {}

# for name, pipe in pipelines.items():
#     print(50*'*')
#     print('Model: ',name)
#     pipe.fit(X_train, y_train)
#     y_pred = pipe.predict_proba(X_test)[:, 1]
#     results[name] = roc_auc_score(y_test, y_pred)

# for model, score in results.items():
#     print(f"{model}: ROC AUC = {score:.4f}")



# # === 1. Split interno para stacking (SIN leakage) ===
# X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(
#     X_train, y_train,
#     test_size=0.25,
#     random_state=69,
#     stratify=y_train
# )

# # === 2. Entrenar modelos base ===
# pipelines_base = {}

# for name, pipe in pipelines.items():
#     pipe_clone = clone(pipe)
#     pipe_clone.fit(X_train_base, y_train_base)
#     pipelines_base[name] = pipe_clone

# # === 3. Meta-features ===
# blend_train = np.column_stack([
#     pipe.predict_proba(X_train_meta)[:, 1]
#     for pipe in pipelines_base.values()
# ])

# blend_test = np.column_stack([
#     pipe.predict_proba(X_test)[:, 1]
#     for pipe in pipelines_base.values()
# ])

# # === 4. Entrenar meta-modelo ===
# meta_model = LogisticRegression(
#     solver="lbfgs",
#     max_iter=1000
# )
# meta_model.fit(blend_train, y_train_meta)

# # === 5. Evaluación final ===
# final_preds = meta_model.predict_proba(blend_test)[:, 1]
# roc_blend = roc_auc_score(y_test, final_preds)
# print(f"ROC AUC STACKING: {roc_blend:.4f}")

# # === 6. Comparación contra modelos individuales ===
# for name, pipe in pipelines_base.items():
#     preds = pipe.predict_proba(X_test)[:, 1]
#     print(f"{name} ROC AUC: {roc_auc_score(y_test, preds):.4f}")

# # === 7. Comparación contra Blend simple(promedio) ===
# blend_simple = np.mean(blend_test, axis=1)
# print("ROC AUC BLEND SIMPLE:", roc_auc_score(y_test, blend_simple))


# n_models = blend_train.shape[1]

# constraints = ({'type': 'eq', 'fun': lambda w: 1 - np.sum(w)})
# bounds = [(0, 1)] * n_models
# init = np.ones(n_models) / n_models

# res = minimize(
#     auc_ensemble,
#     init,
#     bounds=bounds,
#     constraints=constraints,
#     method="SLSQP"
# )

# best_weights = res.x / res.x.sum()

# print("Pesos óptimos:", best_weights)

# print('Apply optimal weights to the test')
# blend_weighted = np.dot(blend_test, best_weights)
# roc_weighted = roc_auc_score(y_test, blend_weighted)
# print(f"ROC AUC BLEND PONDERADO: {roc_weighted:.4f}")

# print('*'*50)
# print('FINAL COMPARATION')
# print("ROC AUC BLEND SIMPLE  :", roc_auc_score(y_test, blend_simple))
# print("ROC AUC BLEND PESOS   :", roc_weighted)
# print("ROC AUC STACKING LR   :", roc_blend)
# print('*'*50)


# pipelines_final = {}

# for name, pipe in pipelines.items():
#     pipe_final = clone(pipe)
#     pipe_final.fit(X_train, y_train)
#     pipelines_final[name] = pipe_final

# blend_test_final = np.column_stack([
#     pipe.predict_proba(df_test)[:, 1]
#     for pipe in pipelines_final.values()
# ])

# stacking_test_preds = meta_model.predict_proba(blend_test_final)[:, 1]



# submissions = {}

# # Individual pipelines predictions
# for name, pipe in pipelines_base.items():
#     submissions[name] = pipe.predict_proba(df_test)[:, 1]

# make_submission(submissions["XGBoost"],   "ori_new_fea_submission_xgboost.csv")
# make_submission(submissions["LightGBM"],  "ori_new_fea_submission_lightgbm.csv")
# make_submission(submissions["CatBoost"],  "ori_new_fea_submission_catboost.csv")

# # Simple blend(Mean)
# blend_simple_test = np.mean(
#     np.column_stack(list(submissions.values())),
#     axis=1
# )
# make_submission(blend_simple_test, "ori_new_fea_submission_blend_simple.csv")

# # Optimized weights blend
# blend_weighted_test = np.dot(
#     np.column_stack(list(submissions.values())),
#     best_weights
# )
# make_submission(blend_weighted_test, "ori_new_fea_submission_blend_weighted.csv")  

# # Optimized weights blend
# make_submission(stacking_test_preds, "ori_new_fea_submission_stacking.csv")


