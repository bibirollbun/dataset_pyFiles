!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)



train["y"] = train.efs_time.values
mx = train.loc[train.efs==1,"efs_time"].max()
mn = train.loc[train.efs==0,"efs_time"].min()
train.loc[train.efs==0,"y"] = train.loc[train.efs==0,"y"] + mx - mn
train.y = train.y.rank()
train.loc[train.efs==0,"y"] += len(train)//2
train.y = train.y / train.y.max()


RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


combined = pd.concat([train,test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

# LABEL ENCODE CATEGORICAL FEATURES
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:

    # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
        
    # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost
print("Using XGBoost version",xgboost.__version__)


import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):
    print("#" * 25)
    print(f"### Fold {i+1}")
    print("#" * 25)

    x_train = train.loc[train_index, FEATURES].copy()
    y_train = train.loc[train_index, "y"]
    x_valid = train.loc[test_index, FEATURES].copy()
    y_valid = train.loc[test_index, "y"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        tree_method="hist",  # Use "hist" instead of "gpu_hist"
        device="cuda",  # Change to "cpu" if no GPU is available
        max_depth=3,
        colsample_bytree=0.5,
        subsample=0.8,
        n_estimators=10_000,
        learning_rate=0.1,
        eval_metric="mae",
        early_stopping_rounds=25,
        objective='reg:logistic',
        enable_categorical=True,
        min_child_weight=5
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=100
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS



from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = -oof_xgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost =",m)


from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import KFold
import catboost
print("Using CatBoost version",catboost.__version__)


import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))

# Check if GPU is available for CatBoost
try:
    from catboost.utils import get_gpu_device_count
    USE_GPU = get_gpu_device_count() > 0
except:
    USE_GPU = False

device_type = "GPU" if USE_GPU else "CPU"

for i, (train_index, test_index) in enumerate(kf.split(train)):
    print("#" * 25)
    print(f"### Fold {i+1} (Device: {device_type})")
    print("#" * 25)

    x_train = train.loc[train_index, FEATURES].copy()
    y_train = train.loc[train_index, "y"]
    x_valid = train.loc[test_index, FEATURES].copy()
    y_valid = train.loc[test_index, "y"]
    x_test = test[FEATURES].copy()

    model_cat = CatBoostRegressor(
        task_type=device_type,  # Auto-switch between GPU and CPU
        iterations=10000,
        learning_rate=0.1,
        depth=6,
        eval_metric="MAE",
        early_stopping_rounds=25
    )

    model_cat.fit(
        x_train, y_train,
        eval_set=(x_valid, y_valid),
        cat_features=CATS,
        verbose=100
    )

    # INFER OOF
    oof_cat[test_index] = model_cat.predict(x_valid)
    # INFER TEST
    pred_cat += model_cat.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat /= FOLDS


from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = -oof_cat
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost =",m)


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding
from tensorflow.keras.layers import Concatenate, BatchNormalization
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold

print('TF Version',tf.__version__)


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
        CATS.append(c)
    elif not "age" in c:
        train[c] = train[c].astype("str")
        test[c] = test[c].astype("str")
        CATS.append(c)
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


CAT_SIZE = []
CAT_EMB = []
NUMS = []

combined = pd.concat([train,test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

print("We LABEL ENCODE the CATEGORICAL FEATURES: ")

for c in FEATURES:
    if c in CATS:
        # LABEL ENCODE
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        #combined[c] = combined[c].astype("category")

        n = combined[c].nunique()
        mn = combined[c].min()
        mx = combined[c].max()
        print(f'{c} has ({n}) unique values')

        CAT_SIZE.append(mx+1) 
        CAT_EMB.append( int(np.ceil( np.sqrt(mx+1))) ) 
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
            
        m = combined[c].mean()
        s = combined[c].std()
        combined[c] = (combined[c]-m)/s
        combined[c] = combined[c].fillna(0)
        
        NUMS.append(c)
        
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


EPOCHS = 4
LRS = [0.01]*2 + [0.001]*1 + [0.0001]*1

def lrfn(epoch):
    return LRS[epoch]

rng = [i for i in range(EPOCHS)]
lr_y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, lr_y, '-o')
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}". \
        format(lr_y[0], max(lr_y), lr_y[-1]))
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.show()

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = False)


def build_model():
    
    # CATEGORICAL FEATURES
    x_input_cats = Input(shape=(len(CATS),))
    embs = []
    for j in range(len(CATS)):
        e = tf.keras.layers.Embedding(CAT_SIZE[j],CAT_EMB[j])
        x = e(x_input_cats[:,j])
        x = tf.keras.layers.Flatten()(x)
        embs.append(x)
        
    # NUMERICAL FEATURES
    x_input_nums = Input(shape=(len(NUMS),))
    
    # COMBINE
    x = tf.keras.layers.Concatenate(axis=-1)(embs+[x_input_nums]) 
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    
    model = Model(inputs=[x_input_cats,x_input_nums], outputs=x)
    
    return model


REPEATS = 3
FOLDS = 5
kf = KFold(n_splits=FOLDS, random_state=42, shuffle=True)

oof_nn = np.zeros( len(train) )
pred_nn = np.zeros( len(test) )
for r in range(REPEATS):
    VERBOSE = r==0
    print("#"*25)
    print(f"### REPEAT {r+1} ###")
    print("#"*25)
        
    for i, (train_index, test_index) in enumerate(kf.split(train)):
        
        X_train_cats = train.loc[train_index,CATS].values
        X_train_nums = train.loc[train_index,NUMS].values
        y_train = train.loc[train_index,"y"].values
        y_train2 = train.loc[train_index,"efs"].values
        X_valid_cats = train.loc[test_index,CATS].values
        X_valid_nums = train.loc[test_index,NUMS].values
        y_valid = train.loc[test_index,"y"].values
        y_valid2 = train.loc[test_index,"efs"].values
        
        X_test_cats = test[CATS].values
        X_test_nums = test[NUMS].values

        if VERBOSE:
            print(" ","#"*25)
            print(" ",f"### Fold {i+1} ###")
            print(" ","#"*25)
         # TRAIN MODEL
        K.clear_session()
        model = build_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
                      loss="mean_squared_error",  
                     )
        v = 2 if VERBOSE else 0
        model.fit([X_train_cats,X_train_nums], [y_train], 
                  validation_data = ([X_valid_cats,X_valid_nums], [y_valid]),
                  callbacks = [lr_callback],
                  batch_size=512, epochs=EPOCHS, verbose=v)
        #model.save_weights(f'{directory}/NN_f{i}_r{r}.weights.h5')
        
        # INFER OOF
        oof_nn[test_index] += model.predict([X_valid_cats,X_valid_nums], verbose=v, batch_size=512).flatten()
        # INFER TEST
        pred_nn += model.predict([X_test_cats,X_test_nums], verbose=v, batch_size=512).flatten()
oof_nn /= REPEATS
pred_nn /= (FOLDS*REPEATS)


from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_nn
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for NN =",m)


%%time

!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import polars as pl
import pandas as pd

from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

import optuna
import os
from colorama import Fore

from tqdm import tqdm
from IPython.display import clear_output
from lifelines import KaplanMeierFitter
pd.options.display.max_columns = None
from lifelines.utils import concordance_index


import lightgbm as lgb
from lightgbm import early_stopping  
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.model_selection import *
from sklearn.metrics import *

SEED = 114514
n_splits = 10


sp = '/kaggle/input/abdbase/AbdML/main.py'
tp = '/kaggle/working/main.py'

with open(sp, 'r', encoding='utf-8') as file:
    content = file.read()
with open(tp, 'w', encoding='utf-8') as file:
    file.write(content)

from main import AbdBase

train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
sample = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
train_solution = train[['ID','efs','efs_time','race_group']].copy()

cat_c = ['dri_score','psych_disturb', 'cyto_score', 'diabetes', 'tbi_status', 'arrhythmia', 'graft_type', 'vent_hist',
 'renal_issue','pulm_severe', 'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab', 'prod_type',
 'cyto_score_detail', 'conditioning_intensity', 'ethnicity', 'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match',
 'hepatic_severe', 'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match', 'race_group',
 'hepatic_mild', 'tce_div_match', 'donor_related', 'melphalan_dose', 'cardiac','pulm_moderate']

def update(df):
    
    global cat_c

    for c in cat_c:
        df[c] = df[c].fillna('None').astype('category')

    j_ch=',[]{}:"\\<'
    for ch in j_ch:
        for c in cat_c:
            df[c] = df[c].apply(lambda x:str(x).replace(ch,''))
                
    return df

train = update(train)
test = update(test)

def transform_survival_probability(df, time_col='efs_time', event_col='efs'):

    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], event_observed=df[event_col])
    survival_probabilities = kmf.survival_function_at_times(df[time_col]).values.flatten()
    return survival_probabilities

def update_target_with_survival_probabilities(df, time_col='efs_time', event_col='efs',cen=0.15):

    race_group = sorted(df['race_group'].unique())
    survival_probs_dict = {}
    for race in race_group:
        race_df = df[df['race_group'] == race]
        survival_probs_dict[race] = transform_survival_probability(race_df, time_col, event_col)
    for race in race_group:
        df.loc[df['race_group'] == race, 'target'] = survival_probs_dict[race]
    df.loc[df[event_col] == 0, 'target'] -= cen
    
    return df

train = update_target_with_survival_probabilities(train, time_col='efs_time', event_col='efs',cen=0.10)

f_fe = [
    'year_hct', 'dri_score_High', 'comorbidity_score', 'conditioning_intensity_None', 
    'karnofsky_score', 'donor_age', 'age_at_hct', 'mrd_hct_None', 
    'cyto_score_detail_Poor', 'dri_score_Intermediate', 'conditioning_intensity_RIC', 
    'cyto_score_Poor', 'hla_match_a_high', 'prim_disease_hct_ALL', 
    'gvhd_proph_FK+ MMF +- others', 
    'dri_score_High - TED AML case missing cytogenetics', 'sex_match_F-M', 
    'pulm_severe_Yes', 'cmv_status_-/+', 'hla_nmdp_6', 'cardiac_Yes', 
    'race_group_Black or African-American', 'sex_match_M-M', 'prim_disease_hct_AML', 
    'mrd_hct_Negative', 'donor_related_Related', 'hla_match_a_low', 
    'cyto_score_detail_None', 'cyto_score_Favorable', 'sex_match_M-F', 
    'arrhythmia_No', 'prior_tumor_No', 'in_vivo_tcd_Yes', 
    'race_group_More than one race', 'sex_match_F-F', 'hla_match_drb1_high', 
    'donor_related_Unrelated', 'tbi_status_No TBI', 'cyto_score_detail_Favorable', 
    'pulm_severe_No', 'tce_imm_match_None', 'mrd_hct_Positive', 
    'prim_disease_hct_MDS', 'diabetes_Yes', 'cmv_status_+/-', 
    'gvhd_proph_FKalone', 'prior_tumor_Not done', 'melphalan_dose_MEL', 
    'diabetes_No', 'arrhythmia_None', 'gvhd_proph_Cyclophosphamide +- others', 
    'hla_low_res_8', 'gvhd_proph_CSA + MMF +- others(not FK)', 'hepatic_severe_No', 
    'hla_low_res_6', 'graft_type_Bone marrow', 'cmv_status_+/+', 
    'prim_disease_hct_IEA', 'hla_match_dqb1_high', 'hla_match_dqb1_low', 
    'hla_match_b_low', 'dri_score_N/A - pediatric', 'dri_score_TBD cytogenetics', 
    'conditioning_intensity_MAC', 'obesity_No', 'tce_match_None', 
    'in_vivo_tcd_None', 'race_group_White', 'tce_div_match_None', 
    'hla_high_res_10', 'prod_type_BM', 'prim_disease_hct_IIS', 
    'hla_match_c_high', 'hla_match_c_low', 'prod_type_PB', 'hla_low_res_10', 
    'cyto_score_None', 'cmv_status_-/-', 'prior_tumor_Yes', 
    'conditioning_intensity_NMA', 'arrhythmia_Not done', 'cardiac_None', 
    'tce_imm_match_G/G', 'prim_disease_hct_NHL', 'cyto_score_detail_Not tested', 
    'dri_score_Low', 'ethnicity_Not Hispanic or Latino', 'hla_match_b_high', 
    'race_group_Asian', 'melphalan_dose_N/A Mel not given', 'hepatic_mild_None', 
    'psych_disturb_No', 'tbi_status_TBI +- Other =cGy', 
    'cyto_score_detail_Intermediate', 'in_vivo_tcd_No', 'conditioning_intensity_TBD', 
    'hla_match_drb1_low', 'graft_type_Peripheral blood', 'hla_high_res_8', 
    'hla_high_res_6', 'prim_disease_hct_HIS', 'cyto_score_Intermediate', 
    'cyto_score_TBD', 'donor_related_Multiple donor (non-UCB)', 
    'pulm_moderate_Yes', 'tce_imm_match_P/P', 'tbi_status_TBI +- Other >cGy', 
    'vent_hist_Yes', 'tbi_status_TBI + Cy +- Other', 'tce_div_match_HvG non-permissive', 
    'cyto_score_detail_TBD', 'gvhd_proph_Cyclophosphamide alone', 
    'tce_div_match_Permissive mismatched', 'obesity_None', 'tce_match_Permissive', 
    'pulm_severe_None', 'rheum_issue_Yes', 'tce_div_match_GvH non-permissive', 
    'cardiac_No', 'dri_score_Very high', 'diabetes_Not done', 'rituximab_None', 
    'tce_match_GvH non-permissive', 'tce_imm_match_H/H', 'gvhd_proph_None', 
    'prim_disease_hct_SAA', 'rituximab_No', 'vent_hist_No', 'hepatic_severe_Yes', 
    'tce_imm_match_G/B', 'pulm_moderate_No', 'vent_hist_None', 
    'gvhd_proph_TDEPLETION alone', 'dri_score_N/A - non-malignant indication', 
    'race_group_Native Hawaiian or other Pacific Islander', 'prim_disease_hct_PCD', 
    'rheum_issue_Not done', 'cyto_score_Other', 'dri_score_None', 'ethnicity_None', 
    'dri_score_Intermediate - TED AML case missing cytogenetics', 
    'cmv_status_None', 'melphalan_dose_None', 'gvhd_proph_FK+ MTX +- others(not MMF)', 
    'psych_disturb_Yes', 'ethnicity_Hispanic or Latino', 'pulm_severe_Not done', 
    'renal_issue_None', 'peptic_ulcer_No', 'donor_related_None', 
    'prim_disease_hct_AI', 'tbi_status_TBI +- Other -cGy unknown dose', 
    'hepatic_severe_Not done', 'peptic_ulcer_None', 
    'tce_div_match_Bi-directional non-permissive', 'renal_issue_No', 
    'arrhythmia_Yes', 'tce_match_Fully matched', 'pulm_moderate_None', 
    'rituximab_Yes'
]

def c_index_score(modeloff, model_name, weights=None):
    y_true = train_solution 
    y_pred = train_solution[["ID"]].copy()

    if isinstance(modeloff, (list, tuple, np.ndarray)) and all(isinstance(m, np.ndarray) for m in modeloff):
        if weights is None:
            weights = [1] * len(modeloff)
        
        assert len(modeloff) == len(weights), "The number of models must match the number of weights."
        
        combined_modeloff = sum(weight * model for weight, model in zip(weights, modeloff))
        y_pred["prediction"] = combined_modeloff
    else:
        y_pred["prediction"] = modeloff

    c_index = base.CIBMTR_score(y_true.copy(), y_pred.copy(), "ID")
    print(Fore.YELLOW + f"The Score of {model_name} is: {c_index:.4f}")
    
ohe_cols = {'cat_c': cat_c}
target_cols = {'cat_c': cat_c, 'target_col': 'target'}

base = AbdBase(train_data=train, test_data=test, target_column='target',gpu=False,
                 problem_type="regression", metric="mae", seed=SEED,ohe_fe=ohe_cols,
                 n_splits=10,early_stop=True,num_classes=0,cat_features=None,
                 fold_type='RKF')

base.X_train = base.X_train[f_fe]
base.X_test = base.X_test[f_fe]


import numpy as np
import random
PL = {'n_estimators': 1796, 'learning_rate': 0.023860332525675564, 'max_depth': 10, 'num_leaves': 57,
      'min_child_samples': 99, 'min_child_weight': 13.462669062341009, 'subsample': 0.8076926662942767,
      'colsample_bytree': 0.6291958123820834, 'reg_alpha': 0.11086139650561047, 'reg_lambda': 2.517250995062926}

LIGHT_MODEL = base.Train_ML(PL,'LGBM',e_stop=200)  
c_index_score(LIGHT_MODEL[0],'LIGHT')



oof_bgm = np.random.rand(len(LIGHT_MODEL[0]))
print(oof_bgm)


pred_bgm = np.random.rand(len(LIGHT_MODEL[1]))
print(pred_bgm)


print(-pred_nn)


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = -oof_xgb -oof_cat -oof_nn -oof_bgm
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Ensemble =",m)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = -pred_xgb -pred_cat -pred_nn -pred_bgm
sub.to_csv("submission.csv",index=False) 
print("Sub shape:",sub.shape)
sub.head()




