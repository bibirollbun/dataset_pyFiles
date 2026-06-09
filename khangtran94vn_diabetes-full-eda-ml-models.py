import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
import optuna

# !pip install --quiet autogluon

warnings.filterwarnings(
    "ignore",
    message="use_inf_as_na option is deprecated",
    category=FutureWarning)

warnings.filterwarnings(
    "ignore",
    message="The default of observed=False is deprecated",
    category=FutureWarning)

warnings.filterwarnings(
    "ignore",
    message="When grouping with a length-1 list-like",
    category=FutureWarning)

warnings.simplefilter(action='ignore', category=FutureWarning)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv',index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col ='id')
train


target = 'diagnosed_diabetes'
train.describe(include='all').T


import eda_utility_scripts
import inspect

functions_list = inspect.getmembers(eda_utility_scripts, inspect.isfunction)
for func_name, func_obj in functions_list:
    # Skip private functions starting with _ if you want
    if not func_name.startswith('_'):
        sig = inspect.signature(func_obj)
        print(f"{func_name}{sig}")
        print()


from eda_utility_scripts import *
mutual_information_classif(train, 'diagnosed_diabetes')


num_cols, cat_cols = overview(train)


plot_corr_heatmaps(train,figsize = (25,10))


compare_numeric_distribution(train, test, num_cols, bins=30)


compare_categorical_distribution(train, test, cat_cols)


plot_num_vs_target(train,'diagnosed_diabetes')


plot_cat_vs_target(train,'diagnosed_diabetes')


def bmi_standards(df):
    """
    Classify BMI according to WHO standards and add an ordinal categorical column 'bmi_who'.
    
    Order:
    Underweight < Normal Weight < Overweight < Obesity
    """
    
    conditions = [
        df['bmi'] < 18.5,
        (df['bmi'] >= 18.5) & (df['bmi'] < 25.0),
        (df['bmi'] >= 25.0) & (df['bmi'] < 30.0),
        df['bmi'] >= 30.0]
    
    categories = [
        'Underweight',
        'Normal Weight',
        'Overweight',
        'Obesity']
    
    df['bmi_standard'] = np.select(conditions, categories, default=np.nan)
    
    # Define ordered categorical type
    bmi_cat_type = pd.CategoricalDtype(
        categories=categories,
        ordered=True)
    
    # Convert to ordinal categorical
    df['bmi_standard'] = df['bmi_standard'].astype(bmi_cat_type)
    
    return df

train = bmi_standards(train)
test = bmi_standards(test)
plot_cat_vs_target(train[['bmi_standard',target]], target, max_unique=20)


def wth_standard(df):
    """
    Classify waist-to-hip ratio (WHR) risk based on WHO standards.
    Treat 'Other' as 'Female'.
    """
    
    # Convert to string first to avoid categorical replace warning
    gender = df['gender'].astype(str)
    
    # Replace 'Other' with 'Female'
    gender = gender.replace({'Other': 'Female'})
    
    whr = df['waist_to_hip_ratio']
    
    conditions = [
        # Low Risk
        ((gender == 'Male') & (whr < 0.90)) |
        ((gender == 'Female') & (whr < 0.80)),
        
        # Moderate Risk
        ((gender == 'Male') & (whr >= 0.90) & (whr < 1.00)) |
        ((gender == 'Female') & (whr >= 0.80) & (whr < 0.85)),
        
        # High Risk
        ((gender == 'Male') & (whr >= 1.00)) |
        ((gender == 'Female') & (whr >= 0.85))
    ]
    
    categories = ['Low Risk', 'Moderate Risk', 'High Risk']
    
    df['wth_standard'] = np.select(conditions, categories, default=np.nan)
    
    # Ordered categorical
    wth_cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
    df['wth_standard'] = df['wth_standard'].astype(wth_cat_type)
    
    return df


train = wth_standard(train)
test = wth_standard(test)
plot_cat_vs_target(train[['wth_standard',target]], target, max_unique=20)


train['Risk_Weight_Combination'] = train['bmi_standard'].astype(str) + ' - ' + train['wth_standard'].astype(str)
test['Risk_Weight_Combination'] = test['bmi_standard'].astype(str) + ' - ' + test['wth_standard'].astype(str)
bmi_levels = train['bmi_standard'].cat.categories
wth_levels = train['wth_standard'].cat.categories

risk_weight_categories = [
    f"{bmi} - {wth}"
    for bmi in bmi_levels
    for wth in wth_levels]

# Convert to ordered categorical
train['Risk_Weight_Combination'] = pd.Categorical(
    train['Risk_Weight_Combination'],
    categories=risk_weight_categories)

test['Risk_Weight_Combination'] = pd.Categorical(
    test['Risk_Weight_Combination'],
    categories=risk_weight_categories)


plot_cat_vs_target_v2(train[['Risk_Weight_Combination',target]], target, max_unique=20)


EDA_stacked_depression_v2(
    train,
    'Risk_Weight_Combination',
    'family_history_diabetes',
    target)


def bin_pressure(df):
    conditions = [
        (df['systolic_bp'] >= 180) | (df['diastolic_bp'] >= 110),
        (df['systolic_bp'] >= 160) | (df['diastolic_bp'] >= 100),
        (df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90),
        (df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 85),
        (df['systolic_bp'] < 130) & (df['diastolic_bp'] < 85)]

    choices = [
        'Hypertension Grade 3 (Severe)',
        'Hypertension Grade 2 (Moderate)',
        'Hypertension Grade 1 (Mild)',
        'Normal-High',
        'Normal']

    df['bp_category'] = np.select(conditions, choices, default=np.nan)

    # Define ordered categorical
    bp_order = [
        'Normal',
        'Normal-High',
        'Hypertension Grade 1 (Mild)',
        'Hypertension Grade 2 (Moderate)',
        'Hypertension Grade 3 (Severe)']
    df['bp_category'] = pd.Categorical(df['bp_category'], categories=bp_order, ordered=True)

    return df

train = bin_pressure(train)
test = bin_pressure(test)
plot_cat_vs_target_v2(train[['bp_category',target]], target, max_unique=20)


fig, ax = plt.subplots(1, 2, figsize=(16, 6))

EDA_stacked_depression_v2(
    train,
    'bp_category',
    'family_history_diabetes',
    target,
    ax=ax[1])

EDA_stacked_depression_v2(
    train,
    'bp_category',
    'hypertension_history',
    target,
    ax=ax[0])

plt.tight_layout()
plt.show()


train['age_risk'] = train['age'].apply(lambda x: 'Over 50' if x > 50 else 'Below 50')
test['age_risk'] = test['age'].apply(lambda x: 'Over 50' if x > 50 else 'Below 50')
plot_cat_vs_target(train[['age_risk',target]], target, max_unique=20)


fig, ax = plt.subplots(1, 2, figsize=(16, 6))

EDA_stacked_depression_v2(
    train,
    'age_risk','bp_category',
    target,
    ax=ax[0])

EDA_stacked_depression_v2(
    train,
    'age_risk','family_history_diabetes',
    target,
    ax=ax[1])

plt.tight_layout()
plt.show()


train['physical_activity_bin'] = train['physical_activity_minutes_per_week'].apply(
        lambda x: 'Low activity' if x < 70 else 'High activity')
test['physical_activity_bin'] = test['physical_activity_minutes_per_week'].apply(
        lambda x: 'Low activity' if x < 70 else 'High activity')
plot_cat_vs_target(train[['physical_activity_bin',target]], target, max_unique=20)


fig, ax = plt.subplots(1, 2, figsize=(16, 6))

EDA_stacked_depression_v2(
    train,
    'physical_activity_bin','bp_category',
    target,
    ax=ax[0])

EDA_stacked_depression_v2(
    train,
    'physical_activity_bin','family_history_diabetes',
    target,
    ax=ax[1])

plt.tight_layout()
plt.show()


train['diet_score_bin'] = train['diet_score'].apply(lambda x: 'Healthier' if x > 6 else 'Not healthier')
test['diet_score_bin'] = test['diet_score'].apply(lambda x: 'Healthier' if x > 6 else 'Not healthier')
plot_cat_vs_target(train[['diet_score_bin',target]], target, max_unique=20)


fig, ax = plt.subplots(1, 2, figsize=(16, 6))

EDA_stacked_depression_v2(
    train,
    'diet_score_bin','bp_category',
    target,
    ax=ax[0])

EDA_stacked_depression_v2(
    train,
    'diet_score_bin','family_history_diabetes',
    target,
    ax=ax[1])

plt.tight_layout()
plt.show()


def calculate_ratio(df):
    df["tg_hdl_risk"] = df["triglycerides"] / df["hdl_cholesterol"].replace(0, np.nan)
    df["tc_hdl_risk"] = df["cholesterol_total"] / df["hdl_cholesterol"].replace(0, np.nan)
    df["tg_hdl_risk"] = pd.cut(
        df["tg_hdl_risk"],
        bins=[-np.inf, 2, 3, np.inf],
        labels=["low_risk", "moderate_risk", "high_insulin_resistance"])

    df["tc_hdl_risk"] = pd.cut(
        df["tc_hdl_risk"],
        bins=[-np.inf, 4, np.inf],
        labels=["optimal", "increased_cardiometabolic_risk"])
    # Total Cholesterol
    df["total_cholesterol_class"] = pd.cut(
        df["cholesterol_total"],
        bins=[-float("inf"), 199, 239, float("inf")],
        labels=["desirable", "borderline_high", "high"])

    # LDL Cholesterol
    df["ldl_class"] = pd.cut(
        df["ldl_cholesterol"],
        bins=[-float("inf"), 99, 129, 159, float("inf")],
        labels=["optimal", "near_optimal", "borderline_high", "high"])

    # HDL Cholesterol
    df["hdl_class"] = pd.cut(
        df["hdl_cholesterol"],
        bins=[-float("inf"), 39, 59, float("inf")],
        labels=["low", "normal", "protective"])

    # Triglycerides
    df["triglycerides_class"] = pd.cut(
    df["triglycerides"],
    bins=[-float("inf"), 149, 199, float("inf")],
    labels=["normal", "borderline_high", "high"])    
    return df
    
train = calculate_ratio(train)
test = calculate_ratio(test)
plot_cat_vs_target(train[['tg_hdl_risk', 'tc_hdl_risk',
       'total_cholesterol_class', 'ldl_class', 'hdl_class',
       'triglycerides_class',target]], target, max_unique=20)


for col in train.select_dtypes(include='object').columns:
    train[col] = pd.Categorical(train[col])

for col in train.columns:
    if col in test.columns:
        test[col] = test[col].astype(train[col].dtype)

cat_features = ['alcohol_consumption_per_week','gender', 'ethnicity', 'education_level','income_level', 'smoking_status', 'employment_status','family_history_diabetes', 
                'hypertension_history','cardiovascular_history', 'bmi_standard','wth_standard', 'Risk_Weight_Combination', 'bp_category', 'age_risk','physical_activity_bin', 
                'diet_score_bin', 'tg_hdl_risk', 'tc_hdl_risk','total_cholesterol_class', 'ldl_class', 'hdl_class','triglycerides_class']

neg = np.sum(train[target] == 0)
pos = np.sum(train[target] == 1)
scale_pos_weight = neg / pos
scale_pos_weight


# def convert_categorize(df):  
#     ### Define order
#     alcohol_order = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#     education_order = ['No formal', 'Highschool', 'Graduate', 'Postgraduate']
#     income_order = ['Low','Lower-Middle','Middle','Upper-Middle','High']
#     physical_order = ['Low activity','High activity']
#     diet_order = ['Not healthier', 'Healthier']

#     ### Convert columns to ordered categorical types if they exist
#     df['alcohol_consumption_per_week'] = pd.Categorical(df['alcohol_consumption_per_week'], categories=alcohol_order, ordered=True)
#     df['education_level'] = pd.Categorical(df['education_level'], categories=education_order, ordered=True)
#     df['income_level'] = pd.Categorical(df['income_level'], categories=income_order, ordered=True)
#     df['family_history_diabetes'] = pd.Categorical(df['family_history_diabetes'], categories=[0,1], ordered=True)
#     df['hypertension_history'] = pd.Categorical(df['hypertension_history'], categories=[0,1], ordered=True)
#     df['cardiovascular_history'] = pd.Categorical(df['cardiovascular_history'], categories=[0,1], ordered=True)
#     df['physical_activity_bin'] = pd.Categorical(df['physical_activity_bin'], categories=physical_order, ordered=True)
#     df['diet_score_bin'] = pd.Categorical(df['diet_score_bin'], categories=diet_order, ordered=True)
#     return df


X = train.drop('diagnosed_diabetes',axis=1)
y = train['diagnosed_diabetes']

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, roc_auc_score

X_train, X_val, y_train, y_val = train_test_split(X,y, stratify = y, random_state=42)


# import xgboost as xgb
# from sklearn.metrics import roc_auc_score

# # ----------------------------
# # Prepare DMatrix
# # ----------------------------
# dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
# dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

# # ----------------------------
# # Define objective
# # ----------------------------
# def objective(trial):
#     param = {
#         'objective': 'binary:logistic',
#         'tree_method': 'hist',
#         'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'gamma': trial.suggest_float('gamma', 0, 5),
#         'alpha': trial.suggest_float('alpha', 0, 5),
#         'lambda': trial.suggest_float('lambda', 0, 5),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#         'scale_pos_weight': scale_pos_weight,
#         'enable_categorical': True,
#         'n_jobs': -1,
#         'eval_metric': 'auc'
#     }

#     # Tune num_boost_round (max rounds) with Optuna
#     num_boost_round = trial.suggest_int('num_boost_round', 100, 2000)

#     bst = xgb.train(
#         params=param,
#         dtrain=dtrain,
#         num_boost_round=num_boost_round,
#         evals=[(dval, 'validation')],
#         early_stopping_rounds=10,
#         verbose_eval=False
#     )

#     y_pred = bst.predict(dval)
#     auc = roc_auc_score(y_val, y_pred)
#     return auc

# # ----------------------------
# # Run Optuna study with 30 trials
# # ----------------------------
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)

# # ----------------------------+
# # Results
# # ----------------------------
# print("Best ROC-AUC:", study.best_value)
# print("Best params:", study.best_params)


import xgboost as xgb
xgb_params = {
    "booster": "gbtree",
    "objective": "binary:logistic",
    "max_depth": 9,
    "eta": 0.02635673354762255,
    "subsample": 0.7423700822026071,
    "colsample_bytree": 0.823473067977625,
    "gamma": 2.0751379302908814,
    "alpha": 1.001557798454784,
    "lambda": 2.396617965393215,
    "min_child_weight": 9,
    "scale_pos_weight": scale_pos_weight,   # <<< ADDED
    "eval_metric": "auc",
    "seed": 42,
    "tree_method": "hist",
    "max_cat_to_onehot": 4}

num_boost_round = 982

# DMatrix
dtrain = xgb.DMatrix(
    X_train,
    label=y_train,
    enable_categorical=True)

dvalid = xgb.DMatrix(
    X_val,
    label=y_val,
    enable_categorical=True)

# Train model
xgb_model = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtrain, "train"), (dvalid, "valid")],
    early_stopping_rounds=50,
    verbose_eval=50)


# from lightgbm import LGBMClassifier, early_stopping
# from sklearn.metrics import roc_auc_score


# # ----------------------------
# # Define objective for Optuna
# # ----------------------------
# def objective(trial):
#     model = LGBMClassifier(
#         objective='binary',
#         boosting_type="gbdt",
#         num_leaves=trial.suggest_int('num_leaves', 16, 256),
#         max_depth=trial.suggest_int('max_depth', 3, 12),
#         learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         min_child_samples=trial.suggest_int('min_child_samples', 5, 100),
#         subsample=trial.suggest_float('subsample', 0.6, 1.0),
#         colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         reg_alpha=trial.suggest_float('reg_alpha', 0.0, 5.0),
#         reg_lambda=trial.suggest_float('reg_lambda', 0.0, 5.0),
#         scale_pos_weight=4.5,
#         n_estimators=trial.suggest_int('num_boost_round', 100, 2000),
#         n_jobs=-1, verbosity=-1 
#     )

#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         eval_metric='auc',
#         categorical_feature=cat_features,
#         callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
#     )

#     y_pred = model.predict_proba(X_val)[:, 1]
#     auc = roc_auc_score(y_val, y_pred)
#     return auc

# # ----------------------------
# # Run Optuna study
# # ----------------------------
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)

# # ----------------------------
# # Results
# # ----------------------------
# print("Best ROC-AUC:", study.best_value)
# print("Best params:", study.best_params)


from lightgbm import LGBMClassifier, early_stopping
lgbm_model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    num_leaves=237,
    max_depth=12,
    learning_rate=0.02982684285126942,
    min_child_samples=73,
    subsample=0.6013380258811998,
    colsample_bytree=0.7590322600822524,
    reg_alpha=1.673711227849413,
    reg_lambda=3.0913284163692367,
    scale_pos_weight=scale_pos_weight,   # <<< ADDED
    n_estimators=1980,
    random_state=42,
    n_jobs=-1,
    verbosity=-1)

lgbm_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    categorical_feature=cat_features,
    callbacks=[early_stopping(stopping_rounds=50, verbose=True)])


# from catboost import CatBoostClassifier, Pool
# from sklearn.metrics import roc_auc_score

# def objective(trial):
#     params = {
#         'iterations': trial.suggest_int('iterations', 100, 2000),  # num_boost_round
#         'depth': trial.suggest_int('depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
#         'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
#         'border_count': trial.suggest_int('border_count', 32, 255),
#         'scale_pos_weight': scale_pos_weight,
#         'task_type': 'CPU',
#         'random_seed': 42,
#         'logging_level': 'Silent'
#     }

#     model = CatBoostClassifier(**params)
    
#     train_pool = Pool(X_train, y_train, cat_features=cat_features)
#     val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
#     model.fit(
#         train_pool,
#         eval_set=val_pool,
#         early_stopping_rounds=10,
#         verbose=False
#     )
    
#     y_pred = model.predict_proba(X_val)[:, 1]
#     auc = roc_auc_score(y_val, y_pred)
#     return auc  # Optuna maximizes by default

# # ----------------------------
# # Run Optuna study
# # ----------------------------
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)

# # ----------------------------
# # Best results
# # ----------------------------
# print("Best ROC-AUC:", study.best_value)
# print("Best params:", study.best_params)


from catboost import CatBoostClassifier

catboost_model = CatBoostClassifier(
    iterations=1222,
    depth=6,
    learning_rate=0.1535302058477091,
    l2_leaf_reg=4.931265799487626,
    bagging_temperature=0.13558051553927536,
    border_count=217,
    loss_function="Logloss",
    eval_metric="AUC",
    scale_pos_weight=scale_pos_weight,   # <<< ADDED
    random_seed=42,
    verbose=50)

catboost_model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_features,
    use_best_model=True,
    early_stopping_rounds=50)


models = {
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model,
    "CatBoost": catboost_model
}

y_val_pred_proba = {}
y_val_pred = {}

# Create DMatrix once for XGBoost
dval = xgb.DMatrix(X_val, enable_categorical=True)

for name, model in models.items():
    if name == "XGBoost":
        # Native Booster → predict returns probabilities directly
        y_val_pred_proba[name] = model.predict(dval)
    else:
        # sklearn-style models
        y_val_pred_proba[name] = model.predict_proba(X_val)[:, 1]

    y_val_pred[name] = (y_val_pred_proba[name] >= 0.5).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, preds) in zip(axes, y_val_pred.items()):
    cm = confusion_matrix(y_val, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(name)

plt.tight_layout()
plt.show()


for name in models.keys():
    print(f"\n{name} — Classification Report")
    print(classification_report(y_val, y_val_pred[name], digits=4))


from sklearn.metrics import roc_curve, auc as sk_auc

plt.figure(figsize=(8, 6))

for name, probs in y_val_pred_proba.items():
    fpr, tpr, _ = roc_curve(y_val, probs)
    roc_auc = sk_auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(20, 6))

xgb_importance = xgb_model.get_score(importance_type="gain")

xgb_df = (
    pd.DataFrame({
        "feature": xgb_importance.keys(),
        "importance": xgb_importance.values()
    })
    .sort_values("importance", ascending=False)
    # .head(20)
)

axes[0].barh(xgb_df["feature"], xgb_df["importance"])
axes[0].set_title("XGBoost Feature Importance (gain)")
axes[0].invert_yaxis()
lgb_df = (
    pd.DataFrame({
        "feature": X_train.columns,
        "importance": lgbm_model.feature_importances_
    })
    .sort_values("importance", ascending=False)
    # .head(20)
)

axes[1].barh(lgb_df["feature"], lgb_df["importance"])
axes[1].set_title("LightGBM Feature Importance")
axes[1].invert_yaxis()
cat_df = (
    pd.DataFrame({
        "feature": X_train.columns,
        "importance": catboost_model.get_feature_importance()
    })
    .sort_values("importance", ascending=False)
    # .head(20)
)

axes[2].barh(cat_df["feature"], cat_df["importance"])
axes[2].set_title("CatBoost Feature Importance")
axes[2].invert_yaxis()

plt.tight_layout()
plt.show()


from sklearn.model_selection import StratifiedKFold

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_preds = {
    "XGBoost": np.zeros(len(X_train)),
    "LightGBM": np.zeros(len(X_train)),
    "CatBoost": np.zeros(len(X_train))}

oof_scores = {
    "XGBoost": [],
    "LightGBM": [],
    "CatBoost": []}


for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n===== Fold {fold}/{n_splits} =====")

    X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)

    xgb_model_cv = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=5000,      # large on purpose
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False)

    preds = xgb_model_cv.predict(dval)
    auc = roc_auc_score(y_va, preds)

    oof_preds["XGBoost"][val_idx] = preds
    oof_scores["XGBoost"].append(auc)

    print(
        f"XGBoost Fold AUC: {auc:.5f} | "
        f"Best iteration: {xgb_model_cv.best_iteration}")


for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n===== Fold {fold}/{n_splits} =====")

    X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    lgbm_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_leaves": 237,
        "max_depth": 12,
        "learning_rate": 0.02982684285126942,
        "min_child_samples": 73,
        "subsample": 0.6013380258811998,
        "colsample_bytree": 0.7590322600822524,
        "reg_alpha": 1.673711227849413,
        "reg_lambda": 3.0913284163692367,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "scale_pos_weight":scale_pos_weight}
    
    
    lgbm_model_cv = LGBMClassifier(
        **lgbm_params,
        n_estimators=5000)
    
    lgbm_model_cv.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[early_stopping(50, verbose=False)])
    
    preds = lgbm_model_cv.predict_proba(X_va)[:, 1]
    fold_auc = roc_auc_score(y_va, preds)
    
    oof_preds["LightGBM"][val_idx] = preds
    oof_scores["LightGBM"].append(fold_auc)
    
    print(
        f"LightGBM Fold AUC: {fold_auc:.5f} | "
        f"Best iteration: {lgbm_model_cv.best_iteration_}")


for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n===== Fold {fold}/{n_splits} =====")

    X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
    cat_params = {
        "iterations": 1222,
        "depth": 6,
        "learning_rate": 0.1535302058477091,
        "l2_leaf_reg": 4.931265799487626,
        "bagging_temperature": 0.13558051553927536,
        "border_count": 217,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": False,
        "scale_pos_weight":scale_pos_weight}
    
    cat_model_cv = CatBoostClassifier(**cat_params)
    
    cat_model_cv.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        cat_features=cat_features,
        use_best_model=True,
        early_stopping_rounds=50)
    
    preds = cat_model_cv.predict_proba(X_va)[:, 1]
    fold_auc = roc_auc_score(y_va, preds)
    
    oof_preds["CatBoost"][val_idx] = preds
    oof_scores["CatBoost"].append(fold_auc)
    
    print(
        f"CatBoost Fold AUC: {fold_auc:.5f} | "
        f"Best iteration: {cat_model_cv.get_best_iteration()}")


oof_preds


oof_scores


cv_summary = pd.DataFrame({
    model: {
        "mean_auc": np.mean(scores),
        "std_auc": np.std(scores)
    }
    for model, scores in oof_scores.items()
}).T

cv_summary


final_model = catboost_model
submit = test.copy()
submit['diagnosed_diabetes'] = final_model.predict_proba(submit)[:, 1]
submit = submit.reset_index()
submit = submit[['id','diagnosed_diabetes']]
submit


submit.to_csv('submission.csv',index=False)
print('Export submission file COMPLETED')


# from autogluon.tabular import TabularPredictor

# # Combine X_train and y_train into a single DataFrame as AutoGluon expects
# train_data = X_train.copy()
# train_data['diagnosed_diabetes'] = y_train

# val_data = X_val.copy()
# val_data['diagnosed_diabetes'] = y_val

# # Define the label column
# label = 'diagnosed_diabetes'

# # Create and train the predictor
# predictor = TabularPredictor(label=label, eval_metric='roc_auc').fit(
#     train_data, 
#     time_limit=1800,  # max training time in seconds
#     presets='medium_quality'  # prioritizes accuracy over speed
# )

# leaderboard = predictor.leaderboard(val_data, silent=False)
# print(leaderboard)



# # Evaluate on validation set
# performance = predictor.evaluate(val_data)
# print(performance)


# all_models = predictor.get_model_names()
# print(all_models)


# # Evaluate on validation set
# performance = predictor.evaluate(val_data)
# print(performance)


# y_pred_specific = predictor.predict(X_val, model='LightGBM_BAG_L1')

