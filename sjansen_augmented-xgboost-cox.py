!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import sys
sys.path.insert(1, '/kaggle/input/adapted-stratified-concordance-index')
from metric_sjoerd import score_new


import os
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb

import pandas as pd
import pandas.api.types
import numpy as np
from lifelines.utils import concordance_index


test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
data_dictionary = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv")

print("Test shape:", test.shape )
print("Train shape:",train.shape)
train.head()


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
        combined[c] = combined[c].astype("float64")
        combined[c] = combined[c].astype("category")

    # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")

train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


# SURVIVAL COX NEEDS THIS TARGET (TO DIGEST EFS AND EFS_TIME)
train["efs_time2"] = train.efs_time.copy()
train.loc[train.efs==0,"efs_time2"] *= -1


#Augmentation

def simple_aug(sample):
    sample["race_group"] = 1
    #rnd_sample["race_group"].astype("category")
    sample["ethnicity"] = 0
    #rnd_sample["ethnicity"].astype("category")
    return sample

def noise_aug(sample, NUMs, stds):
    for std, num in zip(NUMs, stds):
      #add Gaussian noise
      sample[num] += (std*np.random.randn(len(sample))).astype(int)
    return sample

def augment(total_train_data, train_ids, number_samples = 5000, aug_func = simple_aug):
    #take only those datapoints that appear in those indices (i.e. in the training part of the fold)
    train_data_allowed = total_train_data.loc[train_ids].copy()
    NUMS = list(set(FEATURES).difference(CATS))
    #take a sample of them for augmentation
    sample = train_data_allowed.sample(number_samples, random_state=42)
    #apply augmentation
    aug_sample = aug_func(sample)
    return aug_sample

def augment_sub(total_train_data, train_ids, sub_groups=[0], number_samples = 5000, aug_func = simple_aug):
    #take only those datapoints that appear in those indices (i.e. in the training part of the fold)
    train_data_allowed = total_train_data.loc[train_ids].copy()
    #further filter to take only from certain race groups
    filtered_train = train_data_allowed[train_data_allowed["race_group"].isin(sub_groups)]
    NUMS = list(set(FEATURES).difference(CATS))
    #take a sample of them for augmentation
    sample = filtered_train.sample(number_samples, random_state=42)

    #apply augmentation
    aug_sample = aug_func(sample)
    return aug_sample



stds = []
NUMS = list(set(FEATURES).difference(CATS))
for num in NUMS:
  stds.append(np.std(train[num]))
print(augment(total_train_data = train, train_ids = [1,2,3,4,5,6,7], number_samples = 5, aug_func = (lambda x: noise_aug(x, stds, NUMS))).loc[:, NUMS])


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_xgb_cox = np.zeros(len(train))
pred_xgb_cox = np.zeros(len(test))



for i, (train_index, test_index) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    #Add given number of augmentated data points to FOLDS-1 training folds
    aug = augment_sub(total_train_data = train, sub_groups = [2,5], train_ids = train_index, number_samples = 5000, aug_func = simple_aug)

    x_train = train.loc[train_index,FEATURES].copy()
    x_aug = aug.loc[:, FEATURES].copy()
    x_new = pd.concat([x_train, x_aug])

    y_train = train.loc[train_index,"efs_time2"].copy()
    y_aug = aug.loc[:, "efs_time2"].copy()
    y_new = pd.concat([y_train, y_aug])

    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"efs_time2"].copy()
    x_test = test[FEATURES].copy()

    model_xgb_cox = XGBRegressor(
        device="cuda",
        max_depth=3,
        colsample_bytree=0.5,
        subsample=0.8,
        n_estimators=2000,
        learning_rate=0.02,
        enable_categorical=True,
        min_child_weight=80,
        objective='survival:cox',
        eval_metric='cox-nloglik',
    )
    model_xgb_cox.fit(
        x_new, y_new,
        eval_set=[(x_valid, y_valid)],
        verbose=500
    )

    # INFER OOF
    oof_xgb_cox[test_index] = model_xgb_cox.predict(x_valid)
    # INFER TEST
    pred_xgb_cox += model_xgb_cox.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb_cox /= FOLDS


from scipy.stats import rankdata
y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = rankdata(oof_xgb_cox)
m_new = score_new(y_true.copy(), y_pred.copy(), "ID")
print(f"\nC-indices for XGBoost Survival:Cox =",m_new[0], f"\nMean of C-indices for XGBoost Survival:Cox =",m_new[1], f"\nStd of C-indices for XGBoost Survival:Cox =",m_new[2], f"\nOverall CV for XGBoost Survival:Cox =", m_new[1]-m_new[2])


feature_importance = model_xgb_cox.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES,  # Replace FEATURES with your list of feature names
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Survival:Cox Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()

