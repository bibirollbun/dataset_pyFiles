%%capture
!pip uninstall pyarrow -y
!pip install pyarrow==18.0.0

!pip uninstall numpy -y
!pip install numpy==1.26.4

!pip uninstall async-timeout
!pip install async-timeout==4.0.2

!pip uninstall google-api-core
!pip install google-api-core==2.10.2

!pip uninstall tensorflow==2.17.1
!pip install tensorflow==2.17.0


%%capture
# Yunbase download
!git clone https://github.com/yunsuxiaozi/Yunbase.git
!pip download -r Yunbase/requirements.txt


# Importing dependency files
source_file_path = '/kaggle/working/Yunbase/baseline.py'
target_file_path = '/kaggle/working/baseline.py'
with open(source_file_path, 'r', encoding='utf-8') as file:
    content = file.read()
with open(target_file_path, 'w', encoding='utf-8') as file:
    file.write(content)


# Installing requirements
!pip install -q --requirement /kaggle/working/Yunbase/requirements.txt  \
--no-index --find-links file:.


# Importing libraries

# Dependent class
from baseline import Yunbase

# For making and handling DataFrame 
import pandas as pd

# For calculation, peculiar in Python
import numpy as np

# Machine learning model LGBM
from  lightgbm import LGBMRegressor,LGBMClassifier,log_evaluation,early_stopping

# TFIDF for text handling
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# For avoiding some errors
import warnings

# For setting "warning filters" to control the output method and level of warning
warnings.filterwarnings('ignore')

# For setting randomness
import random

# For setting random seed, to ensure model can be recurrent
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
seed_everything(seed=2025)

# For processing date and time
from datetime import datetime, timedelta

# For progress bar
from tqdm import tqdm


# Method to calculate tomorrow's date (original author yunsuxiaozi)
def plus_one_day(string_date):
    if string_date=='None':
        return 'None'
    x = datetime.strptime(string_date, '%Y-%m-%d')
    x = x + timedelta(days=1)
    x = x.strftime('%Y-%m-%d')
    return x


# The features are from only two files: SepsisLabel file and drugsexposure file
# When you include other file such as measurement meds and measurement lab, the final private score is decreased in my environment.
# The original author is yunsuxiaozi

def get_features(mode='train'):
    path = f'/kaggle/input/phems-hackathon-early-sepsis-prediction/{mode}ing_data/'

    # Read SepsisLabel file
    feats = pd.read_csv(path + f"SepsisLabel_{mode}.csv").drop_duplicates()
    
    # Extract date from SepsisLabel file
    feats['measurement_datetime_day'] = feats['measurement_datetime'].fillna('None').apply(lambda x:x[:10])

    # Read drugsexposure file
    drug = pd.read_csv(path+f"drugsexposure_{mode}.csv")

    # Extract date from drugsexposure file
    drug['measurement_datetime_day']=drug['drug_datetime_hourly'].fillna('None').apply(lambda x:x[:10])
    
    for col in ['drug_concept_id', 'route_concept_id']:

        # Fill the missing value with None string
        drug[col] = drug[col].fillna('None').astype(str)

        # Create two dataframe, each containing drug_concept_id and route_concept_id
        group_df = drug.groupby(['person_id', 'measurement_datetime_day'])[col].apply(lambda x:" ".join(x)).reset_index()
        
        # Merge two dataframe to feats dataframe on person_id and measurement_datetime_day ground.
        feats = feats.merge(group_df, on=['person_id','measurement_datetime_day'], how='left')

        # Fill the missing value with None string
        feats[col] = feats[col].fillna('None')

    # Shift measurement_datetime_day one day forward
    drug['measurement_datetime_day'] = drug['measurement_datetime_day'].apply(lambda x: plus_one_day(x))
    
    for col in ['drug_concept_id','route_concept_id']:

        # Create two dataframe, each containing drug_concept_id and route_concept_id with new name
        group_df = drug.groupby(['person_id','measurement_datetime_day'])[col].apply(lambda x:" ".join(x)).reset_index().rename(columns={col:col+"_previous_day"})
        
        # Merge two dataframe to feats dataframe on person_id and measurement_datetime_day ground.
        feats = feats.merge(group_df, on=['person_id','measurement_datetime_day'], how='left')

        # Fill the missing value with None string
        feats[col+"_previous_day"] = feats[col+"_previous_day"].fillna('None')

    # Create date features
    feats['measurement_datetime']=pd.to_datetime(feats['measurement_datetime'])
    feats["year"] = feats["measurement_datetime"].dt.year
    feats["month"] = feats["measurement_datetime"].dt.month
    feats["day"] = feats["measurement_datetime"].dt.day
    feats['dayofweek']=feats['measurement_datetime'].dt.dayofweek
    feats['dayofyear']=feats['measurement_datetime'].dt.dayofyear
    feats['hour']=feats['measurement_datetime'].dt.hour
    feats.drop(['measurement_datetime'],axis=1,inplace=True)
    
    return feats

train = get_features(mode='train')
test = get_features(mode='test')


train['weight']=train['SepsisLabel'] + 10

# Text feature
# Original author is yunsuxiaozi
text_cols=['drug_concept_id','route_concept_id',
           'drug_concept_id_previous_day','route_concept_id_previous_day',
          ]
word2vec_models=[]
for t_col in text_cols:
    word2vec_models.append(  \
        (TfidfVectorizer(analyzer='word',max_features=50,ngram_range=(1,1)),t_col,'tfidf')  \
        )


train.isnull().sum()


dup = train[train.duplicated()]
len(dup)


test.isnull().sum()


dup = test[test.duplicated()]
len(dup)





train.head()


train.info()


train.describe().T


test.head()


test.info()


test.describe().T


# Parameters for LGBM model
lgb_params={
  'boosting_type':'gbdt','class_weight':None,'colsample_bytree':0.7488447863498244,
  'importance_type':'gain','learning_rate':0.06893681524443476,'max_depth':-1,'min_child_samples':15,
  'min_child_weight':0.001,'min_split_gain':0.0,'n_estimators':150,'n_jobs':-1,'num_leaves':19,
  'objective':None,'random_state':2025,'reg_alpha':6.321244942789797,'reg_lambda':0.44170411438999,
  'subsample':0.9748276894890832,'subsample_for_bin':200000,'extra_trees':True,'metric':'auc',
  'verbose':-1   
}


# Yunbase Framework + LGBM model
yunbase=Yunbase(num_folds=5,
                n_repeats=2,
                objective='binary',
                models=[(LGBMClassifier(**lgb_params),'lgb')],
                metric='pr_auc',
                num_classes=2,
                word2vec_models=word2vec_models,
                early_stop=200,
                group_col='person_id',
                target_col='SepsisLabel',
                use_high_corr_feat=True,
                use_eval_metric=False,
               )


yunbase.fit(train)


test_preds=yunbase.predict(test)


sub=pd.read_csv('/kaggle/input/phems-hackathon-early-sepsis-prediction/SepsisLabel_sample_submission.csv')
sub['SepsisLabel']=test_preds
sub.to_csv('submission.csv', index=False)


sub




