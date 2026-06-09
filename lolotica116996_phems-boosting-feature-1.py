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


demo_train_df = pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/testing_data/person_demographics_episode_test.csv')
demo_test_df = pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/person_demographics_episode_train.csv')

demo_df = pd.concat([demo_train_df, demo_test_df]).reset_index(drop = True)
le = LabelEncoder()
demo_df['gender'] = le.fit_transform(demo_df['gender'])
demo_df = demo_df[['person_id', 'birth_datetime', 'gender']]
demo_df = demo_df.sort_values(by=['birth_datetime', 'gender'], key=lambda col: col.isna())
demo_df = demo_df[['person_id', 'birth_datetime', 'gender']].drop_duplicates(subset = ['person_id']).reset_index(drop = True)

train_df = train_df.merge(demo_df, how='left', on = ['person_id'])
test_df = test_df.merge(demo_df, how='left', on = ['person_id'])


train_df['dow']=train_df['measurement_datetime'].dt.dayofweek
train_df['doy']=train_df['measurement_datetime'].dt.dayofyear
train_df['hour']=train_df['measurement_datetime'].dt.hour

test_df['dow']=test_df['measurement_datetime'].dt.dayofweek
test_df['doy']=test_df['measurement_datetime'].dt.dayofyear
test_df['hour']=test_df['measurement_datetime'].dt.hour


def split_time(datetime):
    try:
        return str(datetime).split(" ")[1]
    except: return ""
        
def create_datetime_feature(df):
    df['curr_date'] = df['measurement_datetime'].apply(lambda x: str(x).split(" ")[0])
    df['curr_date'] = pd.to_datetime(df['curr_date'])
    df['birth_datetime'] = pd.to_datetime(df['birth_datetime'])
    
    df["age_in_months"] = ((df["curr_date"].dt.year - df["birth_datetime"].dt.year) * 12) + (df["curr_date"].dt.month - df["birth_datetime"].dt.month)
    df["age_in_months"] -= (df["curr_date"].dt.day < df["birth_datetime"].dt.day).astype(int)
    df['year'] = df['curr_date'].dt.year
    df['month'] = df['curr_date'].dt.month
    df['day'] = df['curr_date'].dt.day
    
    df['time'] = df['measurement_datetime'].apply(split_time)
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.hour

    del df['birth_datetime']
    del df['curr_date']
    return df
    
train_df = create_datetime_feature(train_df)
test_df = create_datetime_feature(test_df)


train_df['born_year'] = train_df['year'] - train_df['age_in_months'] / 12
test_df['born_year'] = test_df['year'] - test_df['age_in_months'] / 12

train_df['is_overnight'] = ((train_df['time'] >= 23) | (train_df['time'] <= 4)).astype(np.int32)
test_df['is_overnight'] = ((test_df['time'] >= 23) | (test_df['time'] <= 4)).astype(np.int32)


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
            if diff_hours > 0 and diff_hours <= 3 * 24:
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

train_df = process_meds_feature_recent(train_df, mm_train_df, recent_days=3)
train_df = process_meds_feature_recent(train_df, mm_train_df, recent_days=7)
train_df = process_meds_feature_recent(train_df, mm_train_df, recent_days=30)

test_df = process_meds_feature_recent(test_df, mm_test_df, recent_days=3)
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
            if diff_hours > 0 and diff_hours <= 3 * 24:
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

train_df = process_obs_feature_recent(train_df, mm_train_df, recent_days=3)
train_df = process_obs_feature_recent(train_df, mm_train_df, recent_days=7)
train_df = process_obs_feature_recent(train_df, mm_train_df, recent_days=30)

test_df = process_obs_feature_recent(test_df, mm_test_df, recent_days=3)
test_df = process_obs_feature_recent(test_df, mm_test_df, recent_days=7)
test_df = process_obs_feature_recent(test_df, mm_test_df, recent_days=30)


def extend_list_list(val):
    res = []
    for x in val:
        res.extend(x)
    return res

def get_hour_range(x):
    if x >= 23 or x <= 4:
        return 0

    if x >= 5 or x <= 11: 
        return 1

    if x >= 12 or x <= 18: 
        return 2

    if x >= 19 or x <= 22: 
        return 3


def extract_recent_drug_route_feature_df(train_df, prev_train_df, num_recent_days=3):
    drug_concept_id_vals = []
    route_concept_id_vals = []
    drug_route_concept_id_vals = []

    drug_concept_id_with_period_vals = []
    route_concept_id_with_period_vals = []
    drug_route_concept_id_with_period_vals = []
    
    for idx, row in tqdm(prev_train_df[["person_id", "measurement_datetime", "drug_date_hourly", "drug_datetime_hourly", "drug_concept_id", "route_concept_id"]].iterrows()):
        measurement_datetime = row["measurement_datetime"]
        drug_datetime_hourly = row["drug_datetime_hourly"]
        drug_date_hourly = row["drug_date_hourly"]
        drug_concept_id = row["drug_concept_id"]
        route_concept_id = row["route_concept_id"]
    
        if check_nan_val(measurement_datetime) or check_nan_val(drug_date_hourly):
            drug_concept_id_vals.append("None")
            route_concept_id_vals.append("None")
            drug_route_concept_id_vals.append("None")

            drug_concept_id_with_period_vals.append("None")
            route_concept_id_with_period_vals.append("None")
            drug_route_concept_id_with_period_vals.append("None")
            continue
    
        indices = []
        for i, val in enumerate(drug_date_hourly):
            diff_days = (measurement_datetime.date() - datetime.strptime(val, "%Y-%m-%d").date()).days
            if diff_days > 0 and diff_days < num_recent_days:
                indices.append(i)
    
        drug_concept_id = [drug_concept_id[i] for i in indices]
        route_concept_id = [route_concept_id[i] for i in indices]
        drug_datetime_hourly = [drug_datetime_hourly[i] for i in indices]
    
        drug_concept_id = extend_list_list(drug_concept_id)
        route_concept_id = extend_list_list(route_concept_id)
        drug_route_concept_id = [f"{x1}-{x2}" for x1, x2 in zip(drug_concept_id, route_concept_id)]
        drug_datetime_hourly = extend_list_list(drug_datetime_hourly)
        drug_datetime_hourly = [get_hour_range(i) for i in drug_datetime_hourly]
    
        drug_concept_with_hour_range_id = " ".join([f"{x1}-{x2}" for x1, x2 in zip(drug_concept_id, drug_datetime_hourly)])
        route_concept_with_hour_range_id = " ".join([f"{x1}-{x2}" for x1, x2 in zip(route_concept_id, drug_datetime_hourly)])
        drug_route_concept_with_hour_range_id = " ".join([f"{x1}-{x2}" for x1, x2 in zip(drug_route_concept_id, drug_datetime_hourly)])
    
        drug_concept_id = " ".join(drug_concept_id)
        route_concept_id = " ".join(route_concept_id)
        drug_route_concept_id = " ".join(drug_route_concept_id)
        
        drug_concept_id_vals.append(drug_concept_id)
        route_concept_id_vals.append(route_concept_id)
        drug_route_concept_id_vals.append(drug_route_concept_id)
    
        drug_concept_id_with_period_vals.append(drug_concept_with_hour_range_id)
        route_concept_id_with_period_vals.append(route_concept_with_hour_range_id)
        drug_route_concept_id_with_period_vals.append(drug_route_concept_with_hour_range_id)
    
    feat_df = prev_train_df[["person_id", "measurement_datetime"]]
    feat_df[f"drug_concept_id_recent_a{num_recent_days}"] = drug_concept_id_vals
    feat_df[f"route_concept_id_recent_a{num_recent_days}"] = route_concept_id_vals
    feat_df[f"drug_route_concept_id_recent_a{num_recent_days}"] = drug_route_concept_id_vals

    feat_df[f"drug_concept_id_with_period_recent_a{num_recent_days}"] = drug_concept_id_with_period_vals
    feat_df[f"route_concept_id_with_period_recent_a{num_recent_days}"] = route_concept_id_with_period_vals
    feat_df[f"drug_route_concept_id_with_period_recent_a{num_recent_days}"] = drug_route_concept_id_with_period_vals
    
    feat_df = train_df.merge(feat_df, on=["person_id", "measurement_datetime"], how="left")
    for col in [f"drug_concept_id_recent_a{num_recent_days}",
                 f"route_concept_id_recent_a{num_recent_days}", 
                 f"drug_route_concept_id_recent_a{num_recent_days}",
                 f"drug_concept_id_with_period_recent_a{num_recent_days}",
                 f"route_concept_id_with_period_recent_a{num_recent_days}", 
                 f"drug_route_concept_id_with_period_recent_a{num_recent_days}"]:
        feat_df[col] = feat_df[col].fillna('None')
    return feat_df

def extract_recent_drug_route_in_day_feature_df(train_df, inday_train_df):
    drug_concept_id_vals = []
    route_concept_id_vals = []
    drug_route_concept_id_vals = []

    drug_concept_id_with_period_vals = []
    route_concept_id_with_period_vals = []
    drug_route_concept_id_with_period_vals = []
    
    for idx, row in tqdm(inday_train_df[["person_id", "measurement_datetime", "drug_datetime_hourly", "drug_concept_id", "route_concept_id"]].iterrows()):
        measurement_datetime = row["measurement_datetime"]
        drug_datetime_hourly = row["drug_datetime_hourly"]
        drug_concept_id = row["drug_concept_id"]
        route_concept_id = row["route_concept_id"]
    
        if check_nan_val(measurement_datetime) or check_nan_val(drug_datetime_hourly):
            drug_concept_id_vals.append("None")
            route_concept_id_vals.append("None")
            drug_route_concept_id_vals.append("None")

            drug_concept_id_with_period_vals.append("None")
            route_concept_id_with_period_vals.append("None")
            drug_route_concept_id_with_period_vals.append("None")
            continue
    
        drug_date_hourly = [x.astype('datetime64[ms]').astype(datetime).hour for x in drug_datetime_hourly]
        drug_date_hourly = [get_hour_range(i) for i in drug_date_hourly]
        
        indices = []
        for i, val in enumerate(drug_datetime_hourly):
            diff_hours = (measurement_datetime - val).total_seconds() / 3600
            if diff_hours > 0:
                indices.append(i)
    
        drug_date_hourly = [drug_date_hourly[i] for i in indices]
        drug_concept_id = [drug_concept_id[i] for i in indices]
        route_concept_id = [route_concept_id[i] for i in indices]
    
        drug_concept_id = extend_list_list(drug_concept_id)
        route_concept_id = extend_list_list(route_concept_id)
        drug_route_concept_id = [f"{x1}-{x2}" for x1, x2 in zip(drug_concept_id, route_concept_id)]
    
        drug_concept_with_hour_range_id = " ".join([f"{x1}-{x2}" for x1, x2 in zip(drug_concept_id, drug_date_hourly)])
        route_concept_with_hour_range_id = " ".join([f"{x1}-{x2}" for x1, x2 in zip(route_concept_id, drug_date_hourly)])
        drug_route_concept_with_hour_range_id = " ".join([f"{x1}-{x2}" for x1, x2 in zip(drug_route_concept_id, drug_date_hourly)])
    
        drug_concept_id = " ".join(drug_concept_id)
        route_concept_id = " ".join(route_concept_id)
        drug_route_concept_id = " ".join(drug_route_concept_id)
        
        drug_concept_id_vals.append(drug_concept_id)
        route_concept_id_vals.append(route_concept_id)
        drug_route_concept_id_vals.append(drug_route_concept_id)
    
        drug_concept_id_with_period_vals.append(drug_concept_with_hour_range_id)
        route_concept_id_with_period_vals.append(route_concept_with_hour_range_id)
        drug_route_concept_id_with_period_vals.append(drug_route_concept_with_hour_range_id)
    
    feat_df = inday_train_df[["person_id", "measurement_datetime"]]
    feat_df[f"drug_concept_id_in_day"] = drug_concept_id_vals
    feat_df[f"route_concept_id_in_day"] = route_concept_id_vals
    feat_df[f"drug_route_concept_id_in_day"] = drug_route_concept_id_vals
    
    feat_df[f"drug_concept_id_with_period_in_day"] = drug_concept_id_with_period_vals
    feat_df[f"route_concept_id_with_period_in_day"] = route_concept_id_with_period_vals
    feat_df[f"drug_route_concept_id_with_period_in_day"] = drug_route_concept_id_with_period_vals
    
    feat_df = train_df.merge(feat_df, on=["person_id", "measurement_datetime"], how="left")
    for col in [f"drug_concept_id_in_day", 
                 f"route_concept_id_in_day", 
                 f"drug_route_concept_id_in_day",
                 f"drug_concept_id_with_period_in_day", 
                 f"route_concept_id_with_period_in_day", 
                 f"drug_route_concept_id_with_period_in_day"]:
        feat_df[col] = feat_df[col].fillna('None')
    return feat_df


# def extract_past_hour_drug_route_features(train_df, inday_train_df, past_hour_cons = 3):
#     hourly_drug_features = []
#     hourly_route_features = []
#     max_drug_used_per_hour_most_features = []
#     max_drug_count_used_per_hour_most_features = []
#     max_route_used_per_hour_most_features = []
#     max_route_count_used_per_hour_most_features = []

#     res_df = inday_train_df[['person_id','measurement_datetime']]        
#     for idx, row in tqdm(inday_train_df.iterrows()):
#         drug_concept_id = row["drug_concept_id"]
#         route_concept_id = row["route_concept_id"]
#         drug_datetime_hourly = row["drug_datetime_hourly"]
#         measurement_datetime = row["measurement_datetime"]
    
#         hourly_drug_count = [0 for _ in range(len(drug_concept_index))]
#         hourly_route_count = [0 for _ in range(len(route_concept_index))]

#         max_drug_c = -1
#         max_route_c = -1
#         max_drug = None
#         max_route = None
        
#         if check_nan_val(drug_datetime_hourly):
#             hourly_drug_features.append(hourly_drug_count)
#             hourly_route_features.append(hourly_route_count)
#             max_drug_used_per_hour_most_features.append(max_drug)
#             max_drug_count_used_per_hour_most_features.append(max_drug_c)
#             max_route_used_per_hour_most_features.append(max_route)
#             max_route_count_used_per_hour_most_features.append(max_route_c)
#             continue
    
#         for drugs, routes, drug_date in zip(drug_concept_id, route_concept_id, drug_datetime_hourly):
#             if check_nan_val(drugs) or check_nan_val(routes):
#                 continue
    
#             diff_hour = (measurement_datetime - drug_date).total_seconds() / 3600

#             if diff_hour < past_hour_cons:
#                 drugs = [x if not check_nan_val(x) else "Unknown" for x in drugs]
#                 routes = [x if not check_nan_val(x) else "Unknown" for x in routes]
                
#                 for drug in drugs:
#                     hourly_drug_count[drug_concept_index[drug]] += 1
    
#                 for route in routes:
#                     hourly_route_count[route_concept_index[route]] += 1
#         hourly_drug_features.append(hourly_drug_count)
#         hourly_route_features.append(hourly_route_count) 

#     hourly_drug_features = np.array(hourly_drug_features, dtype=np.int32)
#     hourly_route_features = np.array(hourly_route_features, dtype=np.int32)

#     for x, i in drug_concept_index.items():
#         res_df[f"times_drug_{x}_used_in_last_{past_hour_cons}_hour"] = hourly_drug_features[:, i]   

#     for x, i in route_concept_index.items():
#         res_df[f"times_route_{x}_used_in_last_{past_hour_cons}_hour"] = hourly_route_features[:, i]

#     res_df = train_df.merge(res_df, on=["person_id", "measurement_datetime"], how="left")
#     for col in [f"drug_concept_id_in_day", 
#                  f"route_concept_id_in_day", 
#                  f"drug_route_concept_id_in_day",
#                  f"drug_concept_id_with_period_in_day", 
#                  f"route_concept_id_with_period_in_day", 
#                  f"drug_route_concept_id_with_period_in_day"]:
#         res_df[col] = res_df[col].fillna('None')
#     return res_df


train_df = extract_recent_drug_route_in_day_feature_df(train_df, inday_train_df)
train_df = extract_recent_drug_route_feature_df(train_df, prev_train_df, num_recent_days=3)
train_df = extract_recent_drug_route_feature_df(train_df, prev_train_df, num_recent_days=7)
train_df = extract_recent_drug_route_feature_df(train_df, prev_train_df, num_recent_days=30)
train_df['drug_mearsure_same_day'] = inday_train_df['drug_mearsure_same_day']


test_df = extract_recent_drug_route_in_day_feature_df(test_df, inday_test_df)
test_df = extract_recent_drug_route_feature_df(test_df, prev_test_df, num_recent_days=3)
test_df = extract_recent_drug_route_feature_df(test_df, prev_test_df, num_recent_days=7)
test_df = extract_recent_drug_route_feature_df(test_df, prev_test_df, num_recent_days=30)
test_df['drug_mearsure_same_day'] = inday_test_df['drug_mearsure_same_day']


train_df.to_parquet("/kaggle/working/train_v11.parquet")
test_df.to_parquet("/kaggle/working/test_v11.parquet")


print(1)
























































































