import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)


df1 = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv", skiprows = 0,header = 0)
df1.head()


df_clean = df1.copy(deep = True)


df_clean.shape


df_clean.drop(columns = ['id'], inplace = True)


df_clean.duplicated().sum()


df_clean.columns


df_clean.isnull().sum()


df_clean.hist(bins = 100, figsize = (14,14))
plt.show()


df_clean.info()


import numpy as np

def feature_engg(df_in):
    df = df_in.copy()
    cols = set(df.columns)

    # ---------------- 1. CORE CARDIOMETABOLIC SIGNALS ----------------

    # Central obesity (strong predictor)
    if {'waist_to_hip_ratio'}.issubset(cols):
        df['fe_central_obesity'] = df['waist_to_hip_ratio']

    # Pulse pressure (vascular stiffness)
    if {'systolic_bp', 'diastolic_bp'}.issubset(cols):
        df['fe_pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']

        # Mean arterial pressure (overall BP load)
        df['fe_map'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3.0

    # ---------------- 2. LIPID RISK RATIOS (VERY HIGH SIGNAL) ----------------

    if {'cholesterol_total', 'hdl_cholesterol'}.issubset(cols):
        df['fe_tc_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-3)

    if {'triglycerides', 'hdl_cholesterol'}.issubset(cols):
        df['fe_tg_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-3)

    # ---------------- 3. BODY COMPOSITION ----------------

    if {'bmi'}.issubset(cols):
        df['fe_bmi'] = df['bmi']

    # ---------------- 4. LIFESTYLE CONTINUOUS SIGNALS ----------------
    # (no binarization, preserve ranking)

    if {'physical_activity_minutes_per_week', 'screen_time_hours_per_day'}.issubset(cols):
        df['fe_activity_screen_ratio'] = (
            df['physical_activity_minutes_per_week'] /
            (df['screen_time_hours_per_day'] * 60 + 1e-3)
        )

    if {'sleep_hours_per_day'}.issubset(cols):
        df['fe_sleep_hours'] = df['sleep_hours_per_day']

    if {'alcohol_consumption_per_week'}.issubset(cols):
        df['fe_alcohol'] = df['alcohol_consumption_per_week']

    # ---------------- 5. HISTORY FLAGS (KEEP AS-IS) ----------------

    for c in [
        'family_history_diabetes',
        'hypertension_history',
        'cardiovascular_history'
    ]:
        if c in cols:
            df[c] = df[c].astype(int)

    return df



# #shorten and stratify
# from sklearn.model_selection import train_test_split

# x_num = num_cols
# y = df_clean['diagnosed_diabetes']

# x_tr,x_test,y_tr,y_test = train_test_split(x_num,y, test_size = 0.5, stratify = y, random_state=11)


# from sklearn.feature_selection import mutual_info_regression

# #TAKES A WHILE TO RUN, TAKE SS AFTER FIRST RUN AND COMMENT THEREAFTER

# mi_scores = mutual_info_regression(x_tr,y_tr, random_state = 42) #random_state used cause scores are calculated by knn estimator of entropy
# mi_series = pd.Series(mi_scores, index = x_tr.columns).sort_values(ascending = False)
# mi_series


num_cols = df_clean.select_dtypes(include  = ['int64','float64']).drop(columns = ['diagnosed_diabetes'])
cat_cols = df_clean.select_dtypes(include = ['object'])
cat_cols.head()


num_cols.head()


onehot_cols = ['ethnicity','gender','smoking_status','employment_status']
ordinal_cols = ['education_level','income_level']


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

cat_col_preprocess = ColumnTransformer(

    transformers=[
        ('onehot',Pipeline([
            ('impute', SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')),
            ('encoder',OneHotEncoder(drop='first', sparse_output = False, handle_unknown = 'ignore'))
        ]), onehot_cols),

        ('ordinal_education',Pipeline([
            ('imputer',SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')),
            ('encoder',OrdinalEncoder(categories = [['No formal', 'Highschool', 'Graduate', 'Postgraduate']]))
        ]), ['education_level']),

        ('ordinal_income',Pipeline([
            ('imputer',SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')),
            ('encoder', OrdinalEncoder(categories=[['Low','Lower-Middle','Middle','Upper-Middle','High']]))
        ]), ['income_level']),

    ], remainder = 'passthrough'
)



from sklearn.preprocessing import StandardScaler

cont_num_cols = num_cols.drop(columns = ['family_history_diabetes','cardiovascular_history'])
binary_num_cols = num_cols.drop(columns = cont_num_cols.columns)

num_col_preprocess = ColumnTransformer(
    transformers = [
        ('continuous',Pipeline([
            ('imputer', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
            ('scaling', StandardScaler())
        ]), cont_num_cols.columns),

        ('binary',Pipeline([
            ('imputer',SimpleImputer(missing_values = np.nan, strategy = 'most_frequent'))
        ]), binary_num_cols.columns)
    ], remainder = 'passthrough'
)


# Use full cleaned data for models that do CV
x = df_clean.drop(columns = ['diagnosed_diabetes'])
y = df_clean['diagnosed_diabetes']


x = feature_engg(x)
x.head()


def preprocessor(df_initial = None, is_test = 0, num_pre = None, cat_pre = None):

  if num_pre is None:
      num_pre = num_col_preprocess
  if cat_pre is None:
      cat_pre = cat_col_preprocess

  num_cols = df_initial.select_dtypes(include=['int64','float64'])
  cat_cols = df_initial.select_dtypes(include = ['object'])

  #IMP
  for i in ['id','diagnosed_diabetes']:
    if i in num_cols.columns:                # num_cols.columns is an Index of strings
      num_cols = num_cols.drop(columns = [i])

  onehot_cols = ['ethnicity','gender','smoking_status','employment_status']
  ordinal_cols = ['education_level','income_level']

  if is_test == 0:
    print('TRAIN')
    cat_col_preprocessed = cat_pre.fit_transform(cat_cols)
    num_col_preprocessed = num_pre.fit_transform(num_cols)
  else:
    print('TEST')
    cat_col_preprocessed = cat_pre.transform(cat_cols)
    num_col_preprocessed = num_pre.transform(num_cols)

  cat_features = cat_pre.get_feature_names_out(input_features = cat_cols.columns)
  cat_col_df = pd.DataFrame(cat_col_preprocessed, columns = cat_features, index = cat_cols.index)

  num_feature = num_pre.get_feature_names_out(input_features = num_cols.columns)
  num_col_df = pd.DataFrame(num_col_preprocessed, columns = num_feature, index = num_cols.index)

  df_final = pd.concat([num_col_df,cat_col_df], axis = 1)

  return df_final


train_df_preprocessed_full = preprocessor(x, is_test=0)
print("Full preprocessed train shape:", train_df_preprocessed_full.shape)





x_train = train_df_preprocessed_full.copy()
y_train = df_clean['diagnosed_diabetes']


x_train.head()


!pip install LightGBM


# from lightgbm import LGBMClassifier
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score

# lgbm = LGBMClassifier(random_state = 11, eval_metric = 'auc')
# skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 11)
# scores = []

# y_probs = np.zeros(len(y_train)) #storing the predicted probabilities during cross val

# for train_idx,val_idx in skf.split(x_train,y_train):
#   x_train_rows, x_val_rows = x_train.iloc[train_idx], x_train.iloc[val_idx]
#   y_train_rows, y_val_rows = y_train.iloc[train_idx], y_train.iloc[val_idx]

#   lgbm.fit(x_train_rows,y_train_rows)

#   print(lgbm.classes_)

#   preds = lgbm.predict_proba(x_val_rows)[:,1]
#   y_probs[val_idx] = preds #storing the predictions for plotting precision recall curve

#   metrics_score = roc_auc_score(y_val_rows, preds)
#   scores.append(metrics_score)



!pip install optuna
!pip install optuna-integration[lightgbm]




# LightGBM 4.6.0 + GPU + Optuna + Multi-Seed CV

import numpy as np
import optuna

from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 11
N_OPTUNA_TRIALS = 25
OPTUNA_FOLDS = 3
FINAL_FOLDS = 10
SEEDS = [11, 21, 42]

# FAST OPTUNA OBJECTIVE (COARSE SEARCH)

def objective(trial):

    params = {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 32, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 200),
        "subsample": trial.suggest_float("subsample", 0.8, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),

        "max_depth": -1,
        "objective": "binary",

        # GPU
        "device": "gpu",
        "max_bin": 255,

        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }

    skf = StratifiedKFold(
        n_splits=OPTUNA_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        X_tr, X_val = x_train.iloc[tr_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model = LGBMClassifier(**params)

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[early_stopping(50)]
        )

        preds = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, preds)
        fold_aucs.append(fold_auc)

        # Aggressive pruning
        trial.report(np.mean(fold_aucs), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_aucs)

# RUN OPTUNA

pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=1
)

study = optuna.create_study(
    direction="maximize",
    pruner=pruner
)

study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

print("\nBest Optuna CV AUC:", study.best_value)
print("Best Params:", study.best_params)

best_params = study.best_params

# FINAL MULTI-SEED STRATIFIED K-FOLD (OOF PROBS)
oof_preds = np.zeros(len(y_train))
oof_scores = []

for seed in SEEDS:
    print(f"\n===== Training with seed {seed} =====")

    model = LGBMClassifier(
        **best_params,
        n_estimators=2000,
        max_depth=-1,
        objective="binary",
        device="gpu",
        max_bin=255,
        random_state=seed,
        n_jobs=-1
    )

    skf = StratifiedKFold(
        n_splits=FINAL_FOLDS,
        shuffle=True,
        random_state=seed
    )

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y_train), 1):
        X_tr, X_val = x_train.iloc[tr_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[early_stopping(100)]
        )

        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] += preds

        auc = roc_auc_score(y_val, preds)
        oof_scores.append(auc)

        print(f"Seed {seed} | Fold {fold} AUC: {auc:.5f}")


oof_preds /= len(SEEDS)

print("\nMean OOF AUC (all folds, all seeds):", np.mean(oof_scores))




FINAL_LGB_PARAMS = {
    "learning_rate": 0.022801444921783764,
    "num_leaves": 40,
    "min_child_samples": 113,
    "subsample": 0.9139779037566488,
    "colsample_bytree": 0.8511551960500995,
    "reg_alpha": 4.9801966287723465,
    "reg_lambda": 0.5848124477760118,

    "n_estimators": 2000,
    "max_depth": -1,
    "objective": "binary",

    "device": "gpu",
    "max_bin": 255,

    "random_state": 11,
    "n_jobs": -1
}



import lightgbm as lgb
lgb.__version__


from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, threshold = precision_recall_curve(y_train, oof_preds)
ap = average_precision_score(y_train, oof_preds)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR Curve (AP = {ap:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precisionâ€“Recall Curve (OOF, Multi-seed)")
plt.legend()
plt.grid(True)
plt.show()



pr_df = pd.DataFrame({'threshold': np.append(threshold,1.0),
                      'precision':precision,
                      'recall':recall
                      })
pr_df.head()


target_recall = 0.95
eligible_rows = pr_df[pr_df['recall'] > target_recall]

best_rows = eligible_rows.sort_values('precision', ascending = False)
best_rows.head()


opt_threshold = best_rows['threshold'].iloc[0]
opt_threshold


pred_opt = (oof_preds > opt_threshold).astype(int)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
ids = test_df['id']
test_df_fe = feature_engg(test_df)
test_df_preprocessed_final = preprocessor(test_df_fe, is_test = 1)
test_df_preprocessed_final.head()


FINAL_LGB_PARAMS = {
    # ðŸ”¹ Optuna-tuned params
    "learning_rate": 0.022801444921783764,
    "num_leaves": 40,
    "min_child_samples": 113,
    "subsample": 0.9139779037566488,
    "colsample_bytree": 0.8511551960500995,
    "reg_alpha": 4.9801966287723465,
    "reg_lambda": 0.5848124477760118,

    "n_estimators": 2000,
    "max_depth": -1,
    "objective": "binary",

    "device": "gpu",
    "max_bin": 255,
    "n_jobs": -1
}



from lightgbm import LGBMClassifier, early_stopping
import numpy as np

SEEDS = [11, 21, 42]

test_preds = np.zeros(len(test_df_preprocessed_final))

for seed in SEEDS:
    print(f"Training final model with seed {seed}")

    model = LGBMClassifier(
        **FINAL_LGB_PARAMS,
        random_state=seed   # override seed
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train)],
        eval_metric="auc",
        callbacks=[early_stopping(100)]
    )

    test_preds += model.predict_proba(test_df_preprocessed_final)[:, 1]

test_preds /= len(SEEDS)



test_preds


sub_df = pd.DataFrame({'id':ids, 'diagnosed_diabetes':test_preds})
sub_df.head()


sub_df.to_csv("submissionVF.csv", index = False)




