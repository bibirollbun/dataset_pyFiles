# !git clone https://github.com/yunsuxiaozi/Yunbase.git
# !pip download -r Yunbase/requirements.txt -q 

# !pip install -q --requirement /kaggle/working/Yunbase/requirements.txt --no-index --find-links file:. -q
# !cp /kaggle/input/yunbase-py-v21/yunbase_pipeline_v2.py /kaggle/working/yunbase_pipeline.py


import os
import pandas as pd#read csv,parquet
import numpy as np#for scientific computation of matrices
from  lightgbm import LGBMRegressor,LGBMClassifier,log_evaluation,early_stopping
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer#word2vec feature
from sklearn.preprocessing import LabelEncoder
import gc
from sklearn.base import BaseEstimator, TransformerMixin, clone, ClassifierMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings#avoid some negligible errors
#The filterwarnings () method is used to set warning filters, which can control the output method and level of warning information.
warnings.filterwarnings('ignore')
import random#provide some function to generate random_seed.
#set random seed,to make sure model can be recurrented.
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm 
from copy import deepcopy
import datetime
from datetime import datetime, timedelta
from sklearn.metrics import precision_recall_curve, auc


def check_nan_val(x):
    if f"{x}" == "None" or f"{x}" == "nan":
        return True
    return False


def seed_everything(seed):
    np.random.seed(seed)#numpy's random seed
    random.seed(seed)#python built-in random seed
    
seed_everything(seed=2025)


mode = "train"
path = f'/kaggle/input/phems-hackathon-early-sepsis-prediction/{mode}ing_data/'
train_df = pd.read_csv(f"{path}/SepsisLabel_{mode}.csv").drop_duplicates()
train_df['measurement_datetime_day'] = train_df['measurement_datetime'].fillna('None').apply(lambda x:x[:10])
train_df['measurement_datetime'] = pd.to_datetime(train_df['measurement_datetime'])

prev_train_df = pd.read_parquet("/kaggle/input/phems-anhlt11-drug-route-health-record/train_drug_and_route_data_process_before_date.parquet")
inday_train_df = pd.read_parquet("/kaggle/input/phems-anhlt11-drug-route-health-record/train_drug_and_route_data_process_in_day.parquet")
inday_train_df["drug_mearsure_same_day"] = inday_train_df["drug_datetime_hourly"].apply(lambda x: not check_nan_val(x))


mode = "test"
path = f'/kaggle/input/phems-hackathon-early-sepsis-prediction/{mode}ing_data/'
test_df = pd.read_csv(f"{path}/SepsisLabel_{mode}.csv").drop_duplicates()
test_df['measurement_datetime_day'] = test_df['measurement_datetime'].fillna('None').apply(lambda x:x[:10])
test_df['measurement_datetime'] = pd.to_datetime(test_df['measurement_datetime'])

prev_test_df = pd.read_parquet(f"/kaggle/input/phems-anhlt11-drug-route-health-record/{mode}_drug_and_route_data_process_before_date.parquet")
inday_test_df = pd.read_parquet(f"/kaggle/input/phems-anhlt11-drug-route-health-record/{mode}_drug_and_route_data_process_in_day.parquet")
inday_test_df["drug_mearsure_same_day"] = inday_test_df["drug_datetime_hourly"].apply(lambda x: not check_nan_val(x))


def process_meds_feature_recent(df, mm_df, recent_days=3):
    feat_df = df[['person_id', 'measurement_datetime']]
    columns = [
       'Systolic blood pressure', 'Diastolic blood pressure',
       'Body temperature', 'Respiratory rate', 'Heart rate',
       'Measurement of oxygen saturation at periphery',
       'Oxygen/Gas total [Pure volume fraction] Inhaled gas'
    ]
    
    mm_df = mm_df.sort_values(by=['measurement_datetime'], ascending=False).groupby('person_id').agg({
        'measurement_datetime': lambda x: list(x),
        'Systolic blood pressure': lambda x: list(x),
        'Diastolic blood pressure': lambda x: list(x),
        'Body temperature': lambda x: list(x),
        'Respiratory rate': lambda x: list(x),
        'Heart rate': lambda x: list(x),
        'Measurement of oxygen saturation at periphery': lambda x: list(x),
        'Oxygen/Gas total [Pure volume fraction] Inhaled gas': lambda x: list(x)
    }).reset_index().rename(columns={"measurement_datetime": "measurement_meds_datetime"})
    
    feat_df = feat_df.merge(mm_df, on=["person_id"], how="left")
    val_map_mean_list = {col: [] for col in columns if col != "measurement_meds_datetime"}
    val_map_min_list = {col: [] for col in columns if col != "measurement_meds_datetime"}
    val_map_max_list = {col: [] for col in columns if col != "measurement_meds_datetime"}
    for idx, row in tqdm(feat_df.iterrows()):
        measurement_datetime = row["measurement_datetime"]
        measurement_meds_datetime = row["measurement_meds_datetime"]
    
        col_map = {col: row[col] for col in columns if col != "measurement_meds_datetime"}
        if check_nan_val(measurement_datetime) or check_nan_val(measurement_meds_datetime):
            for col in col_map:
                val_map_mean_list[col].append(-9999)
                val_map_min_list[col].append(-9999)
                val_map_max_list[col].append(-9999)
            continue
            
        indices = []
        for i, val in enumerate(measurement_meds_datetime):
            diff_hours = (measurement_datetime - val).total_seconds() / 3600
            if diff_hours > 0 and diff_hours <= recent_days * 24:
                indices.append(i)
                
        for col in col_map:
            val = []
            for i in indices:
                if not check_nan_val(col_map[col][i]):
                    val.append(col_map[col][i])
                    
            min_val = -9999 if len(val) == 0 else min(val)
            max_val = -9999 if len(val) == 0 else max(val)
            mean_val = -9999 if len(val) == 0 else sum(val) / len(val)
            val_map_min_list[col].append(min_val)
            val_map_max_list[col].append(max_val)
            val_map_mean_list[col].append(mean_val)
    
    for col in val_map_min_list:
        feat_df = feat_df.drop(col, axis=1)
        
    for col in val_map_min_list:
        feat_df[f"{col}_min_recent_a{recent_days}"] = val_map_min_list[col]
        feat_df[f"{col}_max_recent_a{recent_days}"] = val_map_max_list[col]
        feat_df[f"{col}_mean_recent_a{recent_days}"] = val_map_mean_list[col]

    feat_df = feat_df.drop('measurement_meds_datetime', axis=1).drop_duplicates(['person_id', 'measurement_datetime'])
    feat_df = df.merge(feat_df, on=["person_id", "measurement_datetime"], how="left")

    for col in val_map_min_list:
        feat_df[f"{col}_min_recent_a{recent_days}"] = feat_df[f"{col}_min_recent_a{recent_days}"].fillna(-9999)
        feat_df[f"{col}_max_recent_a{recent_days}"] = feat_df[f"{col}_max_recent_a{recent_days}"].fillna(-9999)
        feat_df[f"{col}_mean_recent_a{recent_days}"] = feat_df[f"{col}_mean_recent_a{recent_days}"].fillna(-9999)
    return feat_df

mm_train_df = pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/measurement_meds_train.csv')
mm_train_df['measurement_datetime'] = pd.to_datetime(mm_train_df['measurement_datetime'])

mm_test_df = pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/measurement_meds_test.csv')
mm_test_df['measurement_datetime'] = pd.to_datetime(mm_test_df['measurement_datetime'])

train_df = process_meds_feature_recent(train_df, mm_train_df, recent_days=7)
train_df = process_meds_feature_recent(train_df, mm_train_df, recent_days=30)

test_df = process_meds_feature_recent(test_df, mm_test_df, recent_days=7)
test_df = process_meds_feature_recent(test_df, mm_test_df, recent_days=30)


def process_obs_feature_recent(df, mm_df, recent_days=3):
    feat_df = df[['person_id', 'measurement_datetime']]
    columns = [
        'Left pupil Diameter Auto',
        'Right pupil Diameter Auto',
        'Glasgow coma scale', 
        'Capillary refill [Time]', 
        'Pulse',
        'Arterial pulse pressure',
        'Right pupil Pupillary response',
        'Left pupil Pupillary response'
    ]
    
    mm_df = mm_df.sort_values(by=['measurement_datetime'], ascending=False).groupby('person_id').agg({
        'measurement_datetime': lambda x: list(x),
        'Left pupil Diameter Auto': lambda x: list(x),
        'Right pupil Diameter Auto': lambda x: list(x),
        'Glasgow coma scale': lambda x: list(x), 
        'Capillary refill [Time]': lambda x: list(x), 
        'Pulse': lambda x: list(x),
        'Arterial pulse pressure': lambda x: list(x),
        'Right pupil Pupillary response': lambda x: list(x),
        'Left pupil Pupillary response': lambda x: list(x)
    }).reset_index().rename(columns={"measurement_datetime": "measurement_meds_datetime"})
    
    feat_df = feat_df.merge(mm_df, on=["person_id"], how="left")
    val_map_mean_list = {col: [] for col in [
        'Left pupil Diameter Auto',
        'Right pupil Diameter Auto',
        'Glasgow coma scale'] if col != "measurement_meds_datetime"}
    val_map_min_list = {col: [] for col in [
        'Left pupil Diameter Auto',
        'Right pupil Diameter Auto',
        'Glasgow coma scale'] if col != "measurement_meds_datetime"}
    val_map_max_list = {col: [] for col in [
        'Left pupil Diameter Auto',
        'Right pupil Diameter Auto',
        'Glasgow coma scale'] if col != "measurement_meds_datetime"}

    other_map_list = {col: [] for col in [
        'Capillary refill [Time]', 
        'Pulse',
        'Arterial pulse pressure',
        'Right pupil Pupillary response',
        'Left pupil Pupillary response'] if col != "measurement_meds_datetime"}
    
    for idx, row in tqdm(feat_df.iterrows()):
        measurement_datetime = row["measurement_datetime"]
        measurement_meds_datetime = row["measurement_meds_datetime"]
    
        col_map = {col: row[col] for col in columns if col != "measurement_meds_datetime"}
        if check_nan_val(measurement_datetime) or check_nan_val(measurement_meds_datetime):
            for col in val_map_mean_list:
                val_map_mean_list[col].append(-9999)
                val_map_min_list[col].append(-9999)
                val_map_max_list[col].append(-9999)

            for col in other_map_list:
                other_map_list[col].append("None")
            continue
            
        indices = []
        for i, val in enumerate(measurement_meds_datetime):
            diff_hours = (measurement_datetime - val).total_seconds() / 3600
            if diff_hours > 0 and diff_hours <= recent_days * 24:
                indices.append(i)
                
        for col in col_map:
            val = []
            for i in indices:
                if not check_nan_val(col_map[col][i]):
                    val.append(col_map[col][i])

            if col in val_map_mean_list:
                min_val = -9999 if len(val) == 0 else min(val)
                max_val = -9999 if len(val) == 0 else max(val)
                mean_val = -9999 if len(val) == 0 else sum(val) / len(val)
                val_map_min_list[col].append(min_val)
                val_map_max_list[col].append(max_val)
                val_map_mean_list[col].append(mean_val)
            else:
                val = "None" if len(val) == 0 else " ".join(val)
                other_map_list[col].append(val)

    for col in val_map_min_list:
        feat_df = feat_df.drop(col, axis=1)

    for col in other_map_list:
        feat_df = feat_df.drop(col, axis=1)
        
    for col in val_map_min_list:
        feat_df[f"{col}_min_recent_a{recent_days}"] = val_map_min_list[col]
        feat_df[f"{col}_max_recent_a{recent_days}"] = val_map_max_list[col]
        feat_df[f"{col}_mean_recent_a{recent_days}"] = val_map_mean_list[col]

    for col in other_map_list:
        feat_df[f"{col}_combine_recent_a{recent_days}"] = other_map_list[col]

    feat_df = feat_df.drop('measurement_meds_datetime', axis=1).drop_duplicates(['person_id', 'measurement_datetime'])
    feat_df = df.merge(feat_df, on=["person_id", "measurement_datetime"], how="left")

    for col in val_map_max_list:
        feat_df[f"{col}_min_recent_a{recent_days}"] = feat_df[f"{col}_min_recent_a{recent_days}"].fillna(-9999)
        feat_df[f"{col}_max_recent_a{recent_days}"] = feat_df[f"{col}_max_recent_a{recent_days}"].fillna(-9999)
        feat_df[f"{col}_mean_recent_a{recent_days}"] = feat_df[f"{col}_mean_recent_a{recent_days}"].fillna(-9999)
    
    for col in other_map_list:
        feat_df[f"{col}_combine_recent_a{recent_days}"] = feat_df[f"{col}_combine_recent_a{recent_days}"].fillna("None")
        
    return feat_df

mm_train_df = pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/measurement_observation_train.csv')
mm_train_df['measurement_datetime'] = pd.to_datetime(mm_train_df['measurement_datetime'])

mm_test_df = pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/measurement_observation_test.csv')
mm_test_df['measurement_datetime'] = pd.to_datetime(mm_test_df['measurement_datetime'])

train_df = process_obs_feature_recent(train_df, mm_train_df, recent_days=7)
train_df = process_obs_feature_recent(train_df, mm_train_df, recent_days=30)

test_df = process_obs_feature_recent(test_df, mm_test_df, recent_days=7)
test_df = process_obs_feature_recent(test_df, mm_test_df, recent_days=30)


train_df.to_parquet("/kaggle/working/train_features_v3.parquet")
test_df.to_parquet("/kaggle/working/test_features_v3.parquet")
























































































