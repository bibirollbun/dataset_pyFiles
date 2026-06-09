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


import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split,StratifiedKFold
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from scipy.stats import skew
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy.stats import chi2,ks_2samp
import plotly.express as exp
import plotly.io as pio
from typing import Tuple
import seaborn as sb
import joblib


path = "/kaggle/input/playground-series-s5e12/train.csv"
df = pd.read_csv(path)
df.head()


df.info()


test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
test_id = test["id"]
test.head()


df.drop("id",axis=1,inplace=True)
test.drop("id",axis=1,inplace=True)


class FeatureDriftSummary:
    def __init__(
        self,
        ks_pvalue_threshold: float = 0.05,
        psi_threshold_moderate: float = 0.1,
        psi_threshold_high: float = 0.2,
        min_category_count: int = 20
    ):
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.psi_threshold_moderate = psi_threshold_moderate
        self.psi_threshold_high = psi_threshold_high
        self.min_category_count = min_category_count

    @staticmethod
    def _calculate_psi(train: pd.Series, test: pd.Series) -> float:
        train_dist = train.value_counts(normalize=True)
        test_dist = test.value_counts(normalize=True)

        categories = train_dist.index.union(test_dist.index)
        train_dist = train_dist.reindex(categories, fill_value=1e-6)
        test_dist = test_dist.reindex(categories, fill_value=1e-6)

        return float(np.sum((train_dist - test_dist) * np.log(train_dist / test_dist)))

    def generate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> pd.DataFrame:

        rows = []
        common_columns = train_df.columns.intersection(test_df.columns)

        for col in common_columns:
            train_col = train_df[col]
            test_col = test_df[col]

            train_count = train_col.notna().sum()
            test_count = test_col.notna().sum()

            if pd.api.types.is_numeric_dtype(train_col):
                stat, p_value = ks_2samp(
                    train_col.dropna(),
                    test_col.dropna()
                )

                drift = p_value < self.ks_pvalue_threshold
                severity = "high" if drift else "none"

                rows.append({
                    "feature": col,
                    "feature_type": "numerical",
                    "test_used": "KS",
                    "statistic": stat,
                    "p_value": p_value,
                    "drift_detected": drift,
                    "severity": severity,
                    "train_rows": train_count,
                    "test_rows": test_count
                })

            else:
                train_str = train_col.astype(str)
                test_str = test_col.astype(str)

                valid_categories = (
                    train_str.value_counts()
                    .loc[lambda x: x >= self.min_category_count]
                    .index
                )

                train_filtered = train_str[train_str.isin(valid_categories)]
                test_filtered = test_str[test_str.isin(valid_categories)]

                psi = self._calculate_psi(train_filtered, test_filtered)

                if psi >= self.psi_threshold_high:
                    severity = "high"
                elif psi >= self.psi_threshold_moderate:
                    severity = "moderate"
                else:
                    severity = "none"

                rows.append({
                    "feature": col,
                    "feature_type": "categorical",
                    "test_used": "PSI",
                    "statistic": psi,
                    "p_value": np.nan,
                    "drift_detected": psi >= self.psi_threshold_moderate,
                    "severity": severity,
                    "train_rows": train_count,
                    "test_rows": test_count
                })

        summary_df = pd.DataFrame(rows)
        return summary_df.sort_values(
            by=["drift_detected", "severity"],
            ascending=[False, False]
        ).reset_index(drop=True)



drift_analyzer = FeatureDriftSummary()

summary_table = drift_analyzer.generate(df, test)
summary_table = summary_table[(summary_table['drift_detected']==True) & (summary_table['severity']=='high')]
print("Statistical Comparison Between Test and Train")
print(summary_table)


cat_cols = df.select_dtypes(include='O').columns.tolist()
cat_cols


num_cols = [col for col in df.columns.tolist() if col not in cat_cols and col != 'diagnosed_diabetes']
num_cols


imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer.fit_transform(df[cat_cols])
test[cat_cols] = imputer.fit_transform(test[cat_cols])
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
    test[col] = test[col].fillna(test[col].median())


def transform(data):
    d1 = data.copy()
    d1['age_bin'] = pd.cut(d1['age'],bins=10,labels=[0,1,2,3,4,5,6,7,8,9]).astype(int)
    #bmi
    #d1['bmi_bin'] = pd.cut(d1['bmi'],bins=10,labels=[0,1,2,3,4,5,6,7,8,9]).astype(int)
    mean_bmi = d1.groupby(['gender','age_bin'])['bmi'].transform('mean')
    d1['bmi_mean'] = d1['bmi'] - mean_bmi
    #sleep hours per day
    #d1['sleep_hours_per_day_bin'] = pd.cut(d1['sleep_hours_per_day'],bins=10,labels=[0,1,2,3,4,5,6,7,8,9]).astype(int)
    mean_sleep = d1.groupby(['gender','age_bin'])['sleep_hours_per_day'].transform('mean')
    d1['sleep_hours_per_day_mean'] = d1['sleep_hours_per_day'] - mean_sleep
    
    #waist to heap ratio
    d1['waist_to_hip_ratio_mean'] = d1.groupby(['gender','age_bin'])['waist_to_hip_ratio'].transform('mean')
    d1['waist_to_hip_ratio_cohort_deviation'] = d1['waist_to_hip_ratio'] - d1['waist_to_hip_ratio_mean']

    #physical_activity_minutes_per_week
    d1['physical_activity_minutes_per_week_bin'] = pd.cut(d1['physical_activity_minutes_per_week'],bins=4,labels=[0,1,2,3]).astype(int)
    
    #diabetic dislipedimia
    d1['diabetic_dislipidemia'] = d1['triglycerides'] - d1['hdl_cholesterol']
    
    ## remove earlier features   
    #d1.drop(['age','bmi','sleep_hours_per_day','waist_to_hip_ratio'],axis=1,inplace=True)
    #d1.drop(['waist_to_hip_ratio_mean','physical_activity_minutes_per_week','triglycerides','hdl_cholesterol'],axis=1,inplace=True)

    return d1


#monotone_constraints = {'age': 1, 'family_history_diabetes': 1, 'age_bin': 1, 'waist_to_hip_ratio_mean': 1, 'physical_activity_minutes_per_week_bin': -1} # {'family_history_diabetes': 1, 'age_bin': 1, 'physical_activity_minutes_per_week_bin': -1}
monotone_constraints = {'age': 1, 'family_history_diabetes': 1}


def detect_and_remove_anomalies(
    df: pd.DataFrame,
    chi2_alpha: float = 0.997,
    support_fraction: float = 0.75,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Detects multivariate anomalies using Robust Mahalanobis Distance
    and removes anomalous rows.

    Returns:
        - cleaned_df: dataframe with anomalies removed
        - anomaly_flags: boolean Series indicating anomalous rows
    """

    if not 0 < chi2_alpha < 1:
        raise ValueError("chi2_alpha must be between 0 and 1")

    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns available for Mahalanobis detection")

    X = df[numeric_cols].copy()
    X = X.fillna(X.median())

    mcd = MinCovDet(
        support_fraction=support_fraction,
        random_state=random_state
    ).fit(X)

    mahalanobis_sq = mcd.mahalanobis(X)
    threshold = chi2.ppf(chi2_alpha, df=len(numeric_cols))

    anomaly_flags = pd.Series(mahalanobis_sq > threshold, index=df.index)
    cleaned_df = df.loc[~anomaly_flags].reset_index(drop=True)

    return cleaned_df, anomaly_flags



#df_org = df.copy()
#print(f"Length of original dataset :{len(df)}")
#df, df_anomalies = detect_and_remove_anomalies(df)
#df_anomalies = df_anomalies[df_anomalies==True]
#print(f"Length of cleaned dataset:{len(df)}")
#print(f"Percentage of anomalies:{len(df_anomalies)*100/len(df_org):.2f}%")


#df = transform(data=df)
#test = transform(data=test)


df.columns


cat_cols = df.select_dtypes(include='O').columns.tolist()
cat_cols


target = df['diagnosed_diabetes']
df.drop('diagnosed_diabetes',axis=1,inplace=True)


best_params={'n_estimators': 4000, 
             'booster': 'gbtree', 
             'subsample': 1,             #0.8191622554413194,
             'colsample_bytree': 0.9187461194496291, 
             'min_child_weight': 22, 
             'reg_lambda': 1.6804064946327641,
             'scale_pos_weight': 1, #1.423202565354993, 
             'max_depth': 8, 
             'eta': 0.3,           #0.3572116306024745,
             'gamma': 0.0,           #0.0005239774831900048,
             'max_bin':512,
             'grow_policy': 'depthwise'}


best_params['tree_method'] = 'hist'
best_params['device'] = 'gpu'
best_params['eval_metric'] = 'auc'
best_params['monotone_constraints'] = monotone_constraints
best_params['verbosity'] = 2


%%time
stkfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

oof_pred_xgb = np.zeros(len(df))
test_pred_xgb = np.zeros(len(test))
thresh = np.linspace(0.05,0.95,100)
for col in cat_cols:
    df[col] = df[col].astype('category')
    test[col] = test[col].astype('category')
    
for idx,(train_id,val_id) in enumerate(stkfold.split(df,target)):
    print(f"Kfold_XGBoost #{idx}")
    x_train,x_val = df.iloc[train_id],df.iloc[val_id]
    y_train,y_val = target.iloc[train_id],target.iloc[val_id]
    train_data = xgb.DMatrix(data=x_train,label=y_train,enable_categorical=True)
    valid_data = xgb.DMatrix(data=x_val,label=y_val,enable_categorical=True)
    dtest = xgb.DMatrix(data=test,enable_categorical=True)

    model_xgboost = xgb.train(best_params,train_data,evals=[(valid_data,"validation")],early_stopping_rounds=20,num_boost_round=50)
    

    oof_pred_xgb[val_id] = model_xgboost.predict(valid_data)
    test_pred_xgb += model_xgboost.predict(dtest)/ stkfold.n_splits


print("OOF XGBoost ROC AUC SCORE:", roc_auc_score(target,oof_pred_xgb))



cat_cols = df.select_dtypes(include='category').columns.tolist()
cat_cols


num_cols = df.select_dtypes(include = ['int64','float64']).columns.tolist()
num_cols


%%time
stkfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

oof_pred_ctb = np.zeros(len(df))
test_pred_ctb = np.zeros(len(test))


for idx,(train_id,val_id) in enumerate(stkfold.split(df,target)):
    print(f"Kfold #{idx}")
    x_train,x_val = df.iloc[train_id],df.iloc[val_id]
    y_train,y_val = target.iloc[train_id],target.iloc[val_id]
    train_data = Pool(x_train,y_train,cat_features=cat_cols)
    valid_data = Pool(x_val,y_val,cat_features=cat_cols)

    model_catboost = CatBoostClassifier(
        iterations=4000,
        learning_rate=0.02,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        l2_leaf_reg=5,
        bagging_temperature= 0.3,
        random_strength=1.5,
        od_wait=80,
        task_type="GPU",
        devices = '0:1',
        #monotone_constraints=monotone_constraints,
        verbose=False
    )
    model_catboost.fit(train_data, eval_set=valid_data, use_best_model=True)

    oof_pred_ctb[val_id] = model_catboost.predict_proba(x_val)[:, 1]
    test_pred_ctb += model_catboost.predict_proba(test)[:, 1] / stkfold.n_splits
print("OOF ROC AUC SCORE:", roc_auc_score(target, oof_pred_ctb))


%%time
opt_auc = 0.0
opt_weights=(0.0,0.0) 

for w_xgb in np.linspace(0.1, 0.9,100):
        w_cb = 1.0 - w_xgb
                      
        # Determine the optimum combination
        predicted = (w_xgb * oof_pred_xgb ) + (w_cb * oof_pred_ctb) 
        auc = roc_auc_score(target,predicted)
        
        if auc > opt_auc:
            opt_auc = auc
            opt_weights = (w_xgb,w_cb)

print(f"Maximum ROC_AUC_Score: {opt_auc:.5f}")
print(f"Optimum Priorities -> XGBoost: {opt_weights[0]:.2f}, CATboost: {opt_weights[1]:.2f}")


test.isnull().sum()


submission = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
submission.columns


submission["id"]= test_id 
submission["diagnosed_diabetes"] = opt_weights[1] * test_pred_ctb +  opt_weights[0] * test_pred_xgb    #0.97 * test_pred_xgb #- 0.025 * test_pred_ctb
submission.head()


submission.to_csv('submission.csv',index=False)

