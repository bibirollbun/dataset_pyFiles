# Huge thanks to Chris Deotte
!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


# ## update install of tabpfn
!mkdir -p /root/.cache/tabpfn/
!cp /kaggle/usr/lib/tabpfn_and_extensions/tabpfn-v2-regressor.ckpt /root/.cache/tabpfn/tabpfn-v2-regressor.ckpt

# ## update install of tabpfn_extensions
!cp -r /kaggle/usr/lib/tabpfn_and_extensions/tabpfn-extensions/src/tabpfn_extensions/ tabpfn_extensions
!mkdir -p tabpfn_extensions/hpo/hpo_models/
!cp /kaggle/usr/lib/tabpfn_and_extensions/tabpfn-v2-regressor.ckpt tabpfn_extensions/hpo/hpo_models/tabpfn-v2-regressor.ckpt
!cp /kaggle/usr/lib/tabpfn_and_extensions/tabpfn-v2-regressor-09gpqh39.ckpt tabpfn_extensions/hpo/hpo_models/tabpfn-v2-regressor-09gpqh39.ckpt
!cp /kaggle/usr/lib/tabpfn_and_extensions/tabpfn-v2-regressor-2noar4o2.ckpt tabpfn_extensions/hpo/hpo_models/tabpfn-v2-regressor-2noar4o2.ckpt
!cp /kaggle/usr/lib/tabpfn_and_extensions/tabpfn-v2-regressor-wyl4o83o.ckpt tabpfn_extensions/hpo/hpo_models/tabpfn-v2-regressor-wyl4o83o.ckpt
!cp /kaggle/usr/lib/tabpfn_and_extensions/tabpfn-v2-regressor-5wof9ojf.ckpt tabpfn_extensions/hpo/hpo_models/tabpfn-v2-regressor-5wof9ojf.ckpt


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print("Train shape:",train.shape)
train.head()


# create target
from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter

def create_km_target(data, n_splits=5, random_state=42):
    """
    Uses Kaplan-Meier
    """
    kmf = KaplanMeierFitter()
    oof_preds = np.zeros(len(data))
    
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(data)):
        train_data = data.iloc[train_idx]
        valid_data = data.iloc[valid_idx]

        kmf.fit(durations=train_data['efs_time'], event_observed=train_data['efs'])
        oof_preds[valid_idx] = kmf.survival_function_at_times(
            valid_data['efs_time']
        ).values

    data['y'] = oof_preds
    return data


def create_na_target(data, n_splits=5, random_state=42):
    """
    Uses Nelson-Aalen
    """
    naf = NelsonAalenFitter()
    oof_preds = np.zeros(len(data))

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(data)):
        train_data = data.iloc[train_idx]
        valid_data = data.iloc[valid_idx]

        naf.fit(durations=train_data['efs_time'], event_observed=train_data['efs'])
        survival_vals = np.exp(-naf.cumulative_hazard_at_times(valid_data['efs_time']).values)
        oof_preds[valid_idx] = survival_vals

    data['y_na'] = oof_preds
    return data

train = create_km_target(train)
train = create_na_target(train)


RMV = ["ID","efs","efs_time","y","y_na"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("missing")
        test[c] = test[c].fillna("missing")
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


combined = pd.concat([train,test],axis=0,ignore_index=True)

for c in FEATURES:
    if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
    if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_tabpfn_kap= np.zeros(len(train))
pred_tabpfn_kap = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()
    tabpfn_kap = TabPFNRegressor(random_state=42,ignore_pretraining_limits=True,device="cuda")
    tabpfn_kap.fit(x_train, y_train)

    # INFER OOF
    oof_tabpfn_kap[test_index] = tabpfn_kap.predict(x_valid)
    # INFER TEST
    pred_tabpfn_kap += tabpfn_kap.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_tabpfn_kap /= FOLDS


from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_tabpfn_kap
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Tabpfn KaplanMeier =",m)


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_tabpfn_aal= np.zeros(len(train))
pred_tabpfn_aal = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y_na"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y_na"]
    x_test = test[FEATURES].copy()
    tabpfn_aal = TabPFNRegressor(random_state=42,ignore_pretraining_limits=True,device="cuda")
    tabpfn_aal.fit(x_train, y_train)

    # INFER OOF
    oof_tabpfn_aal[test_index] = tabpfn_aal.predict(x_valid)
    # INFER TEST
    pred_tabpfn_aal += tabpfn_aal.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_tabpfn_aal /= FOLDS


from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_tabpfn_aal
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Tabpfn AAlon =",m)


from scipy.stats import rankdata 

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = rankdata(oof_tabpfn_kap) + rankdata(oof_tabpfn_aal)
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for TabFn target Ensemble =",m)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = rankdata(pred_tabpfn_kap) + rankdata(pred_tabpfn_aal)

sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()

