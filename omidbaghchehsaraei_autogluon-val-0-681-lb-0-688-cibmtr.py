!pip install -q autogluon --no-index --find-links=file:///kaggle/input/autogluon/v1.0.0


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


from lifelines import KaplanMeierFitter

def transform_target(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y

train['y'] = transform_target(train, time_col='efs_time', event_col='efs') 


RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = [c for c in FEATURES if not c in ['age_at_hct', 'donor_age']]
print(f"There are {len(CATS)} CAT FEATURES: {CATS}") 


for c in CATS:
    train[c] = train[c].fillna("NAN")
    test[c] = test[c].fillna("NAN")


combined = pd.concat([train,test],axis=0,ignore_index=True) 

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


from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label= 'y',eval_metric ='mae',
                             problem_type='regression').fit(train[FEATURES + ['y']],
                                                            presets='best_quality',
                                                            time_limit=3600*8, 
                                                            included_model_types=['NN_TORCH', 'CAT', 'GBM'],
                             ) 


predictor.leaderboard(silent=True).style.background_gradient(subset=['score_val'], cmap='RdYlGn')


oof = predictor.predict_oof()


from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for AutoGluon =",m) 


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = predictor.predict(test)
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()


import shutil
shutil.rmtree("AutogluonModels") 

