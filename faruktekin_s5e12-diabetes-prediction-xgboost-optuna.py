#to-do

#outlier check
#shap analizi
#new features
#data augmentation ?


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


pd.set_option("display.max_columns", 999)


df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")


df.head()


df.info()


df.describe().T


def add_health_features(df):  
    #HDL ranges vary from gender to gender, therefore we need to make different ranges for each gender
    hdl_conditions = [
        (df["gender"] == "Male") & (df["hdl_cholesterol"] < 40),
        (df["gender"] == "Female") & (df["hdl_cholesterol"] < 50),

        (df["hdl_cholesterol"] >= 60),
    
    ]
    choices = ["low_hdl_high_risk", "low_hdl_high_risk", "high_protective"]
    df["hdl_cat"] = np.select(condlist = hdl_conditions, choicelist=choices, default = "normal").astype("object")


     # LDL has general range for everyone. We dont need to specify the gender
    ldl_ranges = [0,100,130,160,190,np.inf]
    ldl_labels = ["ideal", "near_ideal", "limit_range", "high", "very_high"]
    df["ldl_cat"] = pd.cut(df["ldl_cholesterol"], bins = ldl_ranges, labels = ldl_labels).astype("object")


    # Cholesterol total has general range for everyone. We dont need to speficy the gender
    total_chl_ranges = [0,200,240,np.inf]
    total_chl_labels = ["ideal", "limit_range", "high"]
    df["cholesterol_total_cat"] = pd.cut(df["cholesterol_total"], bins = total_chl_ranges, labels = total_chl_labels).astype("object")


    triglycerides_ranges= [0,150,200,500,np.inf]
    triglycerides_labels = ["normal", "limit_range", "high", "very_high"]
    df["triglycerides_cat"] = pd.cut(df["triglycerides"], bins = triglycerides_ranges, labels = triglycerides_labels).astype("object")


    systolic_ranges = [0, 90, 120, 130, 140, np.inf]
    systolic_labels = ["hypotension", "normal", "elevated", "hypertension_stage_1", "hypertension_stage_2"]
    df["systolic_bp_cat"] = pd.cut(df["systolic_bp"], bins = systolic_ranges, labels = systolic_labels).astype("object")


    diastolic_bp_ranges= [0,60,80,90,np.inf]
    diastolic_bp_labels = ["hypotension","normal", "hypertension_1st_deg", "hypertension_2nd_deg"]
    df["diastolic_bp_cat"] = pd.cut(df["diastolic_bp"], bins = diastolic_bp_ranges, labels = diastolic_bp_labels).astype("object")

    rank_map = {
        "hypotension": 0,
        "normal": 0,
        "elevated":1,
        "hypertension_stage_1":2,
        "hypertension_stage_2":3
    }

    df["systolic_rank"] = df["systolic_bp_cat"].map(rank_map).astype(int)

    df["tg_hdl_ratio"] = (df["triglycerides"]) / (df["hdl_cholesterol"])

    return df


def add_health_indicators(df):
    bmi_bins = [0, 18.5, 25, 30, 35, 40, np.inf]
    bmi_labels = [0, 1, 2, 3, 4, 5]
    df["bmi_class"] = pd.cut(df["bmi"], bins  = bmi_bins, labels = bmi_labels).astype(int)

    df["whr_risk"] = np.where(
        ((df["gender"] == "Male") & (df["waist_to_hip_ratio"] > 0.90)) | ((df["gender"] == "Female") & (df["waist_to_hip_ratio"] > 0.85)),1,0)

    df["metabolic_syndrome"] = (
        (df["whr_risk"] == 1).astype(int) 
        + (df["triglycerides"] >= 150).astype(int)  
        + (df["hdl_cat"] == "low_hdl_high_risk").astype(int)  
        + ((df["systolic_bp"] >= 130) | (df["diastolic_bp"] >= 85) | (df["hypertension_history"] == 1)).astype(int)  
    )
    df["metabolic_syndrome_binary"] = (df["metabolic_syndrome"] >= 3).astype(int)

    df["lipid_risk_score"] = (
    (df["ldl_cat"].isin(["high","very_high"])).astype(int)
    + (df["cholesterol_total_cat"] == "high").astype(int)
    + (df["triglycerides_cat"].isin(["high","very_high"])).astype(int)
    + (df["hdl_cat"] == "low_hdl_high_risk").astype(int)
    )
    
    df["rate_pressure_product"] = df["systolic_bp"] * df["heart_rate"]

    df["activity_per_age"] = df["physical_activity_minutes_per_week"] / df["age"]
    
    return df


def add_features(df):
    df = add_health_features(df)
    df = add_health_indicators(df)
    return df


df = add_features(df)


df.head(2)


from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer,RobustScaler

from sklearn.linear_model import LogisticRegressionCV


y = df["diagnosed_diabetes"]
X = df.drop(["diagnosed_diabetes", "id"], axis = 1)


df.head(2)


nominal_cols = ["gender", "ethnicity"]
binary_cols = ["family_history_diabetes", "hypertension_history", "cardiovascular_history", "whr_risk", "metabolic_syndrome_binary"]
ordinal_cols = [col for col in X.columns if (col not in binary_cols) & (col not in nominal_cols) & (X[col].dtype == "object")]

cat_cols = nominal_cols + binary_cols + ordinal_cols

num_cols = [col for col in X.columns if (col not in cat_cols) & (X[col].dtype != "object")]


ordinal_order = {
    'education_level': ['No formal','Highschool','Graduate','Postgraduate'],
    'income_level': ['Low','Lower-Middle','Middle','Upper-Middle','High'],
    'smoking_status': ['Never','Former','Current'],
    'employment_status': ['Student','Unemployed','Employed','Retired'],

    'hdl_cat': ['low_hdl_high_risk', 'normal', 'high_protective'],
    'ldl_cat': ['ideal','near_ideal','limit_range','high','very_high'],
    'cholesterol_total_cat': ['ideal','limit_range','high'],
    'triglycerides_cat': ['normal','limit_range','high','very_high'],
    'systolic_bp_cat': ['hypotension','normal','elevated','hypertension_stage_1','hypertension_stage_2'],
    'diastolic_bp_cat': ['hypotension','normal','hypertension_1st_deg','hypertension_2nd_deg'],
}


print(f"Our data has {df.shape[0]} values")
for col in num_cols:
    iqr = sp.stats.iqr(df[col])
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    print(f"{col} has {df[(df[col] < lower) | (df[col] > upper)].shape[0]} outlier values\n")


skew_values = df[num_cols].skew().sort_values(ascending = False)
log_cols = ["activity_per_age", "physical_activity_minutes_per_week"]
skew_values
# lets apply log transformation to physical_activity_minutes_per_week column since its skew value bigger than 1


from xgboost import XGBClassifier


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y ,random_state = 41)


'''numeric_log_pipeline = Pipeline(steps = [
    ("log", FunctionTransformer(func = np.log1p, feature_names_out = "one-to-one")),
    ("standard", StandardScaler())
])

numeric_standardization = StandardScaler()'''

nominal_encoder = OneHotEncoder(handle_unknown = "ignore", sparse_output = False)

ordinal_encoder = OrdinalEncoder(categories = [ordinal_order[c] for c in ordinal_cols], 
                                 handle_unknown = "use_encoded_value", unknown_value = -1)

preprocessor_xgb = ColumnTransformer(transformers=[
    #("num_log", numeric_log_pipeline, log_col),
    #("num_plain", numeric_standardization, num_cols),
    ("nominal_encode", nominal_encoder, nominal_cols),
    ("ordinal_encode", ordinal_encoder, ordinal_cols)
], remainder = "passthrough")


X_train_processed = preprocessor_xgb.fit_transform(X_train)
X_test_processed = preprocessor_xgb.transform(X_test)
feature_names = preprocessor_xgb.get_feature_names_out()


X_train_processed_df = pd.DataFrame(X_train_processed, columns = feature_names, index=X_train.index)
X_test_processed_df = pd.DataFrame(X_test_processed, columns= feature_names, index =X_test.index)
X_train_processed_df.head(3)


import optuna
import cupy as cp

X_gpu = cp.asarray(X_train_processed_df.values)
y_gpu = cp.asarray(y_train.values)


pos = (y_train == 1).sum()
neg = (y_train == 0).sum()


def objective(trial):
    param = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 41,
        "tree_method": "hist",
        "device": "cuda",
        
        "max_depth": trial.suggest_int("max_depth", 3,10), 
        "n_estimators": trial.suggest_int("n_estimators", 100,2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
    }

    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    scores = []
    for tr, va in skf.split(
        cp.asnumpy(X_gpu), cp.asnumpy(y_gpu)
    ):
        pos = (y_train==1).sum()
        neg = (y_train==0).sum()
        model = XGBClassifier(**param, scale_pos_weight = neg/pos)
        model.fit(X_gpu[tr], y_gpu[tr])

        p = model.predict_proba(X_gpu[va])[:, 1]
        scores.append(roc_auc_score(cp.asnumpy(y_gpu[va]), cp.asnumpy(p)))

    return float(cp.asarray(scores).mean())


study = optuna.create_study(study_name = "xgboost_study_gpu11", direction = "maximize")
study.optimize(objective, n_trials = 20, show_progress_bar = True, n_jobs = 1)


best_params = study.best_params
print(f"\nBest parameters: {best_params}")

fixed_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cuda",
    "random_state": 41,
}

pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
fixed_params["scale_pos_weight"] = neg / pos

final_xgb_model = XGBClassifier(
    **fixed_params,
    **best_params
)



final_xgb_model.fit(X_gpu, y_gpu)


X_test_gpu = cp.asarray(X_test_processed_df.values)
optimized_xgb_preds = final_xgb_model.predict(X_test_gpu)

print(confusion_matrix(y_test, optimized_xgb_preds))
print(classification_report(y_test, optimized_xgb_preds))

p = final_xgb_model.predict_proba(X_test_gpu)[:,1]
print(f"roc-auc score: {roc_auc_score(y_test, p)}")


test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


test_df.head(3)


test_x = test_df.drop("id", axis = 1)


test_x = add_features(test_x)


print(test_x.shape[1])
test_x.head(3)


test_x_processed = preprocessor_xgb.transform(test_x)
test_x_processed_df = pd.DataFrame(test_x_processed, columns = feature_names, index=test_x.index)


test_pred_xgb = final_xgb_model.predict(cp.asarray(test_x_processed_df.values))


submission_xgb = pd.DataFrame({
    "id": test_df["id"],
    "diagnosed_diabetes":test_pred_xgb
})

submission_xgb.to_csv("submission_xgb.csv", index = False)

