# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
df_train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",df_train.shape)
df_train.head()
df_test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", df_test.shape )


from lifelines import KaplanMeierFitter

def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
    
df_train["y"] = transform_survival_probability(
    df_train, time_col = 'efs_time', event_col = 'efs'
)

plt.hist(df_train.loc[df_train.efs == 1,"y"],bins = 100,label = "efs=1, Yes Event")
plt.hist(df_train.loc[df_train.efs == 0,"y"],bins = 100,label = "efs=0, Maybe Event")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


df_train['efs_time_new'] = df_train['efs_time'].copy()
df_train.loc[df_train['efs'] == 0, 'efs_time_new'] *= -1


df_train.columns


list_not_features = ['ID', 'efs', 'efs_time', 'y', 'efs_time_new']
list_features = [c for c in df_train.columns if not c in list_not_features]
list_categorical_features = []

for c in list_features:
    if df_train[c].dtype == "object":
        list_categorical_features.append(c)
        df_train[c] = df_train[c].fillna("NAN")
        df_test[c] = df_test[c].fillna("NAN")

df_train[list_categorical_features] = df_train[list_categorical_features].astype('category')
df_test[list_categorical_features] = df_test[list_categorical_features].astype('category')


df_train.info()


import matplotlib.pyplot as plt 
plt.rcParams['figure.dpi'] = 200 
import seaborn as sns 
import math 
from scipy import stats 
from scipy.stats import norm 

plt.figure(figsize = (20, 10), facecolor = "white")

sns.heatmap(
    df_train.isnull(), vmin = 0, vmax = 1
)

plt.show()


plt.figure(figsize = (20, 4), facecolor = "white")

sns.heatmap(
    df_test.isnull(), vmin = 0, vmax = 1
)

plt.show()


def summary_numerical_dist(df_data, col, q_min, q_max):
    
    fig = plt.figure(figsize = (8, 4), facecolor = "white")

    layout_plot = (2, 2)
    num_subplot = 4
    axes = [None for _ in range(num_subplot)]
    list_shape_subplot = [[(0, 0), (0, 1), (1, 0), (1, 1)], [1, 1, 1, 1], [1, 1, 1, 1]]
    for i in range(num_subplot):
        axes[i] = plt.subplot2grid(
            layout_plot, list_shape_subplot[0][i],
            rowspan = list_shape_subplot[1][i],
            colspan = list_shape_subplot[2][i]
        )

    sns.histplot(data = df_data, x = col, kde = True, ax = axes[0])
    stats.probplot(x = df_data[col], dist = stats.norm, plot = axes[1])
    sns.boxplot(data = df_data, x = col, ax = axes[2])
    pts = df_data[col].quantile(q = np.arange(q_min, q_max, 0.01))
    sns.lineplot(x = pts.index, y = pts, ax = axes[3])
    axes[3].grid(True)

    list_title = ["Histogram", "QQ plot", "Boxplot", "Outlier"]
    for i in range(num_subplot):
        axes[i].set_title(list_title[i])
    plt.suptitle(f"Distribution of: {col}", fontsize = 15)
    plt.tight_layout()
    plt.show()
    
def summary_categorical_dist(df_data, col):
    
    fig = plt.figure(figsize = (8, 4), facecolor = "white")

    layout_plot = (1, 2)
    num_subplot = 2
    axes = [None for _ in range(num_subplot)]
    list_shape_subplot = [[(0, 0), (0, 1)], [1, 1], [1, 1]]
    for i in range(num_subplot):
        axes[i] = plt.subplot2grid(
            layout_plot, list_shape_subplot[0][i],
            rowspan = list_shape_subplot[1][i],
            colspan = list_shape_subplot[2][i]
        )
    
    count = df_data[col].value_counts().sort_index()
    
    sns.countplot(data = df_data, y = col, order = count.index, ax = axes[0])
    axes[1].pie(data = df_data, x = count, labels = count.index, autopct = '%1.1f%%', startangle = 90)
    
    list_title = ["Counts", "Proportions"]
    for i in range(num_subplot):
        axes[i].set_title(list_title[i])
    plt.suptitle(f"Distribution of: {col}", fontsize = 15)
    plt.tight_layout()
    plt.show()


display(df_train.describe().round(3).T)


display(df_train.describe(include = ['object', 'bool', 'category']).T)


summary_categorical_dist(df_train, 'efs')


summary_numerical_dist(df_train, 'efs_time', .95, 1)


summary_numerical_dist(df_train, 'y', .95, 1)


summary_numerical_dist(df_train, 'efs_time_new', .95, 1)


summary_numerical_dist(df_train.loc[df_train['efs'] == 0], 'efs_time_new', 0, .05)


summary_numerical_dist(df_train.loc[df_train['efs'] == 1], 'efs_time_new', .95, 1)


summary_categorical_dist(df_train, 'dri_score')


plt.figure(figsize = (6, 6), facecolor = "white")

sns.boxplot(
    data = df_train, 
    x = 'y', y = 'dri_score',
    orient = 'h'
)

plt.show()


plt.figure(figsize = (6, 6), facecolor = "white")

sns.boxplot(
    data = df_train, 
    x = 'efs_time_new', y = 'dri_score',
    orient = 'h'
)

plt.show()


summary_categorical_dist(df_train, 'conditioning_intensity')


plt.figure(figsize = (6, 3), facecolor = "white")

sns.boxplot(
    data = df_train, 
    x = 'y', y = 'conditioning_intensity',
    orient = 'h'
)

plt.show()


plt.figure(figsize = (6, 3), facecolor = "white")

sns.boxplot(
    data = df_train, 
    x = 'efs_time_new', y = 'conditioning_intensity',
    orient = 'h'
)

plt.show()


summary_numerical_dist(df_train, 'karnofsky_score', .95, 1)


summary_categorical_dist(df_train, 'karnofsky_score')


plt.figure(figsize = (6, 3), facecolor = "white")

sns.boxplot(
    data = df_train, 
    x = 'y', y = 'karnofsky_score',
    orient = 'h'
)

plt.show()


plt.figure(figsize = (6, 3), facecolor = "white")

sns.boxplot(
    data = df_train, 
    x = 'efs_time_new', y = 'karnofsky_score',
    orient = 'h'
)

plt.show()


df_crosstab_cyto_score = pd.crosstab(
    df_train['cyto_score'], df_train['cyto_score_detail'],
    normalize = 'index'
)
display(df_crosstab_cyto_score)


plt.figure(figsize = (6, 4), facecolor = "white")

sns.heatmap(
    data = df_crosstab_cyto_score, 
    annot = True,
    cmap = 'Blues',
    fmt = ".2f"
)

plt.show()


plt.figure(figsize = (20, 10), facecolor = "white")

sns.heatmap(
    data = df_train.corr(numeric_only = True),
    cmap = "vlag",
    vmin = -1, vmax = 1,
    linecolor = "white", linewidth = 0.5,
    annot = True,
    fmt = ".2f"
)

plt.title('Correlation Heatmap')
plt.show()


hct_ci_mapping = {
    "arrhythmia": {"No": 0, "Not done": 0, "Yes": 1},  
    "cardiac": {"No": 0, "Not done": 0, "Yes": 1}, 
    "diabetes": {"No": 0, "Not done": 0, "Yes": 1},  
    "hepatic_mild": {"No": 0, "Not done": 0, "Yes": 1},
    "hepatic_severe": {"No": 0, "Not done": 0, "Yes": 3},
    "psych_disturb": {"No": 0, "Not done": 0, "Yes": 1}, 
    "obesity": {"No": 0, "Not done": 0, "Yes": 1}, 
    "rheum_issue": {"No": 0, "Not done": 0, "Yes": 2},
    "peptic_ulcer": {"No": 0, "Not done": 0, "Yes": 2},  
    "renal_issue": {"No": 0, "Not done": 0, "Yes": 2}, 
    "prior_tumor": {"No": 0, "Not done": 0, "Yes": 3}, 
    "pulm_moderate": {"No": 0, "Not done": 0, "Yes": 2}, 
    "pulm_severe": {"No": 0, "Not done": 0, "Yes": 3},  
}
def calculate_hct_ci_score(row, mapping):
        score = 0
    
        if "hepatic_severe" in row and row["hepatic_severe"] == "Yes":
            score += mapping["hepatic_severe"]["Yes"]
        elif "hepatic_mild" in row and row["hepatic_mild"] == "Yes":
            score += mapping["hepatic_mild"]["Yes"]
        if "pulm_moderate" in row and row["pulm_moderate"] == "Yes":
            score += mapping["pulm_moderate"]["Yes"]
        elif "pulm_severe" in row and row["pulm_severe"] == "Yes":
            score += mapping["pulm_severe"]["Yes"]
    
        for condition, mapping_values in mapping.items():
            if condition not in ["hepatic_mild", "hepatic_severe","pulm_moderate", "pulm_severe"] and condition in row:
                score += mapping_values.get(row[condition], 0)
    
        return score

def cat2num(df):
    df['conditioning_intensity'] = df['conditioning_intensity'].map({
    'NMA': 1, 
    'RIC': 2,
    'MAC': 3,
    'TBD': None,
    'No drugs reported': None,
    'N/A, F(pre-TED) not submitted': None})
    
    df['tbi_status'] = df['tbi_status'].map({
    'No TBI': 0, 
    'TBI +- Other, <=cGy': 1,
    'TBI +- Other, -cGy, fractionated': 2,
    'TBI + Cy +- Other': 3,
    'TBI +- Other, -cGy, single': 4,
    'TBI +- Other, >cGy': 5,
    'TBI +- Other, unknown dose': None})
    
    df['dri_score'] = df['dri_score'].map({
    'Low': 1, 
    'Intermediate': 2,
    'Intermediate - TED AML case <missing cytogenetics': 3,
    'High': 4,
    'High - TED AML case <missing cytogenetics': 5,
    'Very High': 6,
    'N/A - pediatric': -3,
    'N/A - non-malignant indication': -1,
    'TBD cytogenetics': -2,
    'N/A - disease not classifiable': -4,
    'Missing disease status': 0})
    
    df['cyto_score'] = df['cyto_score'].map({
    'Poor': 4,
    'Normal': 3,
    'Intermediate': 2,
    'Favorable': 1,
    'TBD': -1,
    'Other': -2,
    'Not tested': None})
    
    df['cyto_score_detail'] = df['cyto_score_detail'].map({
    'Poor': 3, 
    'Intermediate': 2,
    'Favorable': 1,
    'TBD': -1,
    'Not tested': None})
    
    return df
def fill_hla_combined_low(row):
    if np.isnan(row['hla_combined_low']): 
        components = [
            row['hla_match_drb1_low'], row['hla_match_dqb1_low'], 
            row['hla_match_a_low'], row['hla_match_b_low'], row['hla_match_c_low']
        ]
        if all([not np.isnan(x) for x in components]):
            return sum(components)
        else:
            if not np.isnan(row['hla_low_res_8']) and not np.isnan(row['hla_match_dqb1_low']):
                return row['hla_low_res_8'] + row['hla_match_dqb1_low']
            elif not np.isnan(row['hla_low_res_6']): 
                components_6 = [
                    row['hla_match_dqb1_low'], row['hla_match_c_low']
                ]
                if all([not np.isnan(x) for x in components_6]):
                    return row['hla_low_res_6'] + sum(components_6)
                else: 
                    return sum([x for x in components if not np.isnan(x)])
    return row['hla_combined_low'] 
def add_features(df):
    df["hct_ci_score"] = df.apply(lambda row: calculate_hct_ci_score(row, hct_ci_mapping), axis=1)
    df['donor_recipient_age_diff'] = abs(df['donor_age'] - df['age_at_hct'])
    df = cat2num(df)
    df['hla_combined_low'] = df['hla_low_res_10']
    df['hla_combined_low'] = df.apply(fill_hla_combined_low, axis=1)
    df['hla_match_ratio'] = (df['hla_high_res_8'] + df['hla_low_res_8']) / 16
    df['years_since_2000'] = df['year_hct'] - 2000
    df['null_count'] = df.isnull().sum(axis=1)
    df['ci_score_danger'] = df['hct_ci_score'].apply(lambda x: 2 if x >= 3 else 1 if x >= 1 else 0)
    return df

df_train = add_features(df_train)
df_test = add_features(df_test)


combined = pd.concat([df_train,df_test],axis=0,ignore_index=True)

print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in list_features:

    if c in list_categorical_features:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
        
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")

df_train = combined.iloc[:len(df_train)].copy()
df_test = combined.iloc[len(df_train):].reset_index(drop=True).copy()


list_features += ["hct_ci_score", 'donor_recipient_age_diff', "hla_combined_low", "hla_match_ratio", 
             "years_since_2000", "null_count","ci_score_danger"]
df_train.head()


from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import numpy as np

FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(df_train))
pred_efs = np.zeros(len(df_test))

for i, (train_index, test_index) in enumerate(kf.split(df_train, df_train["efs"])):
    print("#" * 25)
    print(f"### Fold {i+1}")
    print("#" * 25)
    
    x_train = df_train.loc[train_index, list_features].copy()
    y_train = df_train.loc[train_index, "efs"]
    x_valid = df_train.loc[test_index, list_features].copy()
    y_valid = df_train.loc[test_index, "efs"]
    x_test = df_test[list_features].copy()

    model_xgb = XGBClassifier(
        device="cuda",
        max_depth=3,  
        colsample_bytree=0.7129400756425178, 
        subsample=0.8185881823156917, 
        n_estimators=20000, 
        learning_rate=0.04425768131771064,  
        eval_metric="auc", 
        early_stopping_rounds=50, 
        objective='binary:logistic',
        scale_pos_weight=1.5379160847615545,  
        min_child_weight=4,
        enable_categorical=True,
        gamma=3.1330719334577584
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=100
    )

    oof_xgb[test_index] = (model_xgb.predict_proba(x_valid)[:, 1] > 0.5).astype(int)
    pred_efs += model_xgb.predict_proba(x_test)[:, 1]

pred_efs = (pred_efs / FOLDS > 0.5).astype(int)

accuracy = accuracy_score(df_train["efs"], oof_xgb)
f1 = f1_score(df_train["efs"], oof_xgb)
roc_auc = roc_auc_score(df_train["efs"], oof_xgb)
precision = precision_score(df_train["efs"], oof_xgb)
recall = recall_score(df_train["efs"], oof_xgb)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(df_train))
pred_xgb = np.zeros(len(df_test))

for i, (train_index, test_index) in enumerate(kf.split(df_train)):
    print("#" * 25)
    print(f"### Fold {i+1}")
    print("#" * 25)
    
    x_train = df_train.loc[train_index, list_features].copy()
    y_train = df_train.loc[train_index, "y"]
    x_valid = df_train.loc[test_index, list_features].copy()
    y_valid = df_train.loc[test_index, "y"]
    x_test = df_test[list_features].copy()
    
    model_xgb = XGBRegressor(
        device="cpu",
        max_depth=5,  
        colsample_bytree=0.4309907360736148, 
        subsample=0.6727848987288046, 
        n_estimators=10_000,  
        learning_rate=0.03509792076095853, 
        eval_metric="mae",
        early_stopping_rounds=25,
        objective='reg:logistic',
        enable_categorical=True,
        min_child_weight=10,
        reg_alpha=2.950200470036872, 
        reg_lambda=1.484334590329492,
        gamma=0.008314053362236895
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=100 
    )

    oof_xgb[test_index] = model_xgb.predict(x_valid)
    pred_xgb += model_xgb.predict(x_test)

pred_xgb /= FOLDS

mae = mean_absolute_error(df_train["y"], oof_xgb)
mse = mean_squared_error(df_train["y"], oof_xgb)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")


feature_importance = model_xgb.feature_importances_
importance_df = pd.DataFrame({
    "Feature": list_features,  
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis() 
plt.show()


from catboost import CatBoostRegressor, CatBoostClassifier
import catboost
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_cat = np.zeros(len(df_train))
pred_cat = np.zeros(len(df_test))

cat_params = {
    'depth': 6, 
    'learning_rate': 0.04699005545173896, 
    'l2_leaf_reg': 6.853082507365295, 
    'colsample_bylevel': 0.9312642681213008, 
    'min_data_in_leaf': 14, 
    'grow_policy': 'Depthwise', 
    'bootstrap_type': 'Bernoulli', 
    'iterations': 1727
}

for i, (train_index, test_index) in enumerate(kf.split(df_train)):
    print("#" * 25)
    print(f"### Fold {i+1}")
    print("#" * 25)

    x_train = df_train.loc[train_index, list_features].copy()
    y_train = df_train.loc[train_index, "y"]
    x_valid = df_train.loc[test_index, list_features].copy()
    y_valid = df_train.loc[test_index, "y"]
    x_test = df_test[list_features].copy()

    model_cat = CatBoostRegressor(
        **cat_params,
        cat_features=list_categorical_features,
        task_type="CPU",  
        eval_metric='MAE',
        early_stopping_rounds=100,
        random_seed=42,
        verbose=100
    )
    
    model_cat.fit(
        x_train,
        y_train,
        eval_set=(x_valid, y_valid),
    )

    oof_cat[test_index] = model_cat.predict(x_valid)
    pred_cat += model_cat.predict(x_test)

pred_cat /= FOLDS

mae = mean_absolute_error(df_train["y"], oof_cat)
mse = mean_squared_error(df_train["y"], oof_cat)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")


feature_importance = model_cat.get_feature_importance()
importance_df = pd.DataFrame({
    "Feature": list_features, 
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CatBoost Feature Importance")
plt.gca().invert_yaxis() 
plt.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

best_rmse = float("inf")
best_weight = None
true_y = df_train["y"].values

for w in np.linspace(0, 1, 101):
    blended_oof = w * oof_xgb + (1 - w) * oof_cat
    rmse = np.sqrt(mean_squared_error(true_y, blended_oof))
    if rmse < best_rmse:
        best_rmse = rmse
        best_weight = w

oof_ensemble = best_weight * oof_xgb + (1 - best_weight) * oof_cat
test_preds_ensemble = best_weight * pred_xgb + (1 - best_weight) * pred_cat

mae = mean_absolute_error(true_y, oof_ensemble)
mse = mean_squared_error(true_y, oof_ensemble)
rmse = np.sqrt(mse)

print(f"\nBest XGBoost Weight: {best_weight:.4f}")
print(f"Ensemble MAE: {mae:.4f}")
print(f"Ensemble MSE: {mse:.4f}")
print(f"Ensemble RMSE: {rmse:.4f}")

