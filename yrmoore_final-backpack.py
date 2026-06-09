#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering with Fast cuDF-Pandas!
# One of the most powerful feature engineering techniques is groupby(COL1)[COL2].agg(STAT). This is where we group by COL1 column and aggregate (i.e. compute) a statistic STAT over another column COL2. This is the underlying method to compute target encoding and count encoding. By computing raw statistics and inputting them into our model, our model can do more than only receiving TE or CE. This notebook illustrates creating 50 engineered features, but we can create hundreds more and improve CV score and LB score!
# 
# When our dataset has millions of rows like Kaggle's Backpack competition, then groupby operations take time to compute. The fastest way to compute a groupby aggregation is to use GPU with [RAPIDS cuDF-Pandas][1] library.
# 
# There are two ways to use [RAPIDS cuDF][2]. We can write cuDF code which looks just like Pandas code and starts with import cudf. Or we can write normal Pandas code with import pandas but before that we add the cell magic command %load_ext cudf.pandas. By adding this magic command all calls to Pandas afterward will use [RAPIDS cuDF][2] behind the scenes taking advantage of the massive speed boost of GPU!
# 
# Alternatively, we can use [cuDF-Polars][3]. To use [cuDF-Polars][3], we write Polars code with lazy frame. Then the final call includes .collect(engine="gpu") which will run all previous Polars code behind the scenes with [RAPIDS cuDF][2].
# 
# [1]: https://rapids.ai/cudf-pandas/
# [2]: https://docs.rapids.ai/install/
# [3]: https://rapids.ai/polars-gpu-engine/

# # RAPIDS v25.02
# [RAPIDS v25.02][1] was just released Feb 15, 2025! Instructions on installing RAPIDS is [here][1]. On Kaggle, the easiest way to pip install new libraries is to do it once in a Utility Script notebook. Then whenever we attach the Utility Script notebook to another Kaggle notebook, the second Kaggle notebook immediately gets the benefit of the pip installed libraries. We created a [RAPIDS 25.02][1] Utility Script [here][2], and we attach it to the notebook you are reading. Therefore the notebook you are reading can import RAPIDS v25.02 without needing to pip install!
# 
# [1]: https://docs.rapids.ai/install/
# [2]: https://www.kaggle.com/code/cdeotte/rapids-cudf-25-02-cuml-25-02

# # GPU Acceleration
# We activate [cuDF-Pandas][1] with the magic command %load_ext cudf.pandas below. Afterward, all calls to Pandas will use fast GPU [RAPIDS cuDF][2] behind the scenes! Since we attached Utility Script notebook [here][3] to the notebook you are reading, we will be using the new [RAPIDS v25.02][2]!
# 
# [1]: https://rapids.ai/cudf-pandas/
# [2]: https://docs.rapids.ai/install/
# [3]: https://www.kaggle.com/code/cdeotte/rapids-cudf-25-02-cuml-25-02

# In[1]:



import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)

VER=1

# # Load Data
# We load train, train extra, and test data. The combined train data has 4 million rows! This means we do not need to fear overfitting train. We can make hundreds/thousands of new features and every time our CV improves our LB will improve too!

# In[2]:


import kagglehub

path = kagglehub.competition_download('playground-series-s5e2')

train = pd.read_csv(f"{path}/train.csv")
print("Train shape", train.shape )
train.head()

# In[3]:


train2 = pd.read_csv(f"{path}/training_extra.csv")
print("Extra Train shape", train2.shape )
train2.head()

# In[4]:


train = pd.concat([train,train2],axis=0,ignore_index=True)
print("Combined Train shape", train.shape)

# In[5]:


test = pd.read_csv(f"{path}/test.csv")
print("Test shape", test.shape )
test.head()

# # Feature Engineer Columns
# We will engineer 8 new columns by combining existing columns.

# In[6]:


CATS = list(train.columns[1:-2])
print(f"There are {len(CATS)} categorical columns:")
print( CATS )
print(f"There are 1 numerical column:")
print( ["Weight Capacity (kg)"] )

# In[7]:

COMBO = []
for i, c in enumerate(CATS):
    combine = pd.concat([train[c], test[c]], axis=0)
    combine, _ = pd.factorize(combine)
    train[c] = combine[:len(train)]
    test[c] = combine[len(train):]

    n = f"{c}_wc"
    train[n] = train[c] * 100 + train["Weight Capacity (kg)"]
    test[n] = test[c] * 100 + test["Weight Capacity (kg)"]
    COMBO.append(n)

    for j, x in enumerate(CATS):
        if i < j:  # Avoid duplicates (A * B same as B * A)
            combine_x = pd.concat([train[x], test[x]], axis=0)
            combine_x, _ = pd.factorize(combine_x)  # Factorize x the same way
            train[x] = combine_x[:len(train)]
            test[x] = combine_x[len(train):]

            b = f"{c}_{x}_mul"
            train[b] = train[c] * train[x]  # Multiply factorized values
            test[b] = test[c] * test[x]
            COMBO.append(b)

# In[8]:


FEATURES = CATS + ["Weight Capacity (kg)"] + COMBO
print(f"We now have {len(FEATURES)} columns:")
print( FEATURES )

# # XGBoost with Feature Engineer GroupBy
# We train XGBoost with nested folds. We use the inner nested fold to create new features that aggregate the target price. And we use the outer fold to create new features that do not aggregate the target price. In each k fold loop, we engineer new features using the advanced feature engineering technique groupby(COL1)[COL2].agg(STAT). Since we are using [RAPIDS cuDF-Pandas][1], these groupby computations will run fast on GPU! And we will train our model quickly on GPU using XGBoost!
# 
# [1]: https://rapids.ai/cudf-pandas/

# In[9]:


from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import xgboost as xgb
print(f"XGBoost version",xgb.__version__)

# In[10]:


# STATISTICS TO AGGEGATE FOR OUR FEATURE GROUPS
STATS = ["mean","std","count","nunique","median","min","max","skew"]
STATS2 = ["mean","std"]

# In[11]:

def dynamic_m(n):
    return 5 if n < 100 else 10 if n < 1000 else 20

FOLDS=7
kf=KFold(n_splits=FOLDS,shuffle=True,random_state=42)
oof=np.zeros((len(train)))
pred=np.zeros((len(test)))

for i,(train_index,test_index) in enumerate(kf.split(train)):
    print(f"### OUTER Fold {i+1} ###")
    X_train=train.loc[train_index,FEATURES+['Price']].reset_index(drop=True).copy()
    y_train=train.loc[train_index,'Price']
    X_valid=train.loc[test_index,FEATURES].reset_index(drop=True).copy()
    y_valid=train.loc[test_index,'Price']
    X_test=test[FEATURES].reset_index(drop=True).copy()
    
    kf2=KFold(n_splits=FOLDS,shuffle=True,random_state=42)
    for j,(train_index2,test_index2) in enumerate(kf2.split(X_train)):
        print(f" ## INNER Fold {j+1} (outer fold {i+1}) ##")
        X_train2=X_train.loc[train_index2,FEATURES+['Price']].copy()
        X_valid2=X_train.loc[test_index2,FEATURES].copy()
        
        col="Weight Capacity (kg)"
        global_mean=X_train2["Price"].mean()
        tmp=X_train2.groupby(col).Price.agg(STATS)
        tmp["count"]=X_train2.groupby(col).Price.count()
        tmp["m"] = tmp["count"].map(dynamic_m)
        for s in STATS:
            weight=tmp["count"]/(tmp["count"]+tmp["m"])
            tmp[f"TE1_wc_{s}"]=weight*tmp[s]+(1-weight)*global_mean
        tmp=tmp[[f"TE1_wc_{s}" for s in STATS]]
        X_valid2=X_valid2.merge(tmp,on=col,how="left")
        for c in tmp.columns:
            X_train.loc[test_index2,c]=X_valid2[c].values
        
        for col in COMBO:
            global_mean=X_train2["Price"].mean()
            tmp=X_train2.groupby(col).Price.agg(STATS2)
            tmp["count"]=X_train2.groupby(col).Price.count()
            tmp["m"] = tmp["count"].map(dynamic_m)
            for s in STATS2:
                weight=tmp["count"]/(tmp["count"]+tmp["m"])
                tmp[f"TE2_{col}_{s}"]=weight*tmp[s]+(1-weight)*global_mean
            tmp=tmp[[f"TE2_{col}_{s}" for s in STATS2]]
            X_valid2=X_valid2.merge(tmp,on=col,how="left")
            for c in tmp.columns:
                X_train.loc[test_index2,c]=X_valid2[c].values

    col="Weight Capacity (kg)"
    global_mean=X_train["Price"].mean()
    tmp=X_train.groupby(col).Price.agg(STATS)
    tmp["count"]=X_train.groupby(col).Price.count()
    tmp["m"] = tmp["count"].map(dynamic_m)
    for s in STATS:
        weight=tmp["count"]/(tmp["count"]+tmp["m"])
        tmp[f"TE1_wc_{s}"]=weight*tmp[s]+(1-weight)*global_mean
    tmp=tmp[[f"TE1_wc_{s}" for s in STATS]]
    X_valid=X_valid.merge(tmp,on=col,how="left")
    X_test=X_test.merge(tmp,on=col,how="left")

    for col in COMBO:
        global_mean=X_train["Price"].mean()
        tmp=X_train.groupby(col).Price.agg(STATS2)
        tmp["count"]=X_train.groupby(col).Price.count()
        tmp["m"] = tmp["count"].map(dynamic_m)
        for s in STATS2:
            weight=tmp["count"]/(tmp["count"]+tmp["m"])
            tmp[f"TE2_{col}_{s}"]=weight*tmp[s]+(1-weight)*global_mean
        tmp=tmp[[f"TE2_{col}_{s}" for s in STATS2]]
        X_valid=X_valid.merge(tmp,on=col,how="left")
        X_test=X_test.merge(tmp,on=col,how="left")

    for col in CATS:
        col2="Weight Capacity (kg)"
        global_mean=X_train[col2].mean()
        tmp=X_train.groupby(col)[col2].agg(STATS2)
        tmp["count"]=X_train.groupby(col)[col2].count()
        tmp["m"] = tmp["count"].map(dynamic_m)
        for s in STATS2:
            weight=tmp["count"]/(tmp["count"]+tmp["m"])
            tmp[f"FE3_{col}_wc_{s}"]=weight*tmp[s]+(1-weight)*global_mean
        tmp=tmp[[f"FE3_{col}_wc_{s}" for s in STATS2]]
        X_train=X_train.merge(tmp,on=col,how="left")
        X_valid=X_valid.merge(tmp,on=col,how="left")
        X_test=X_test.merge(tmp,on=col,how="left")

    # CONVERT TO CATS SO XGBOOST RECOGNIZES THEM
    X_train[CATS] = X_train[CATS].astype("category")
    X_valid[CATS] = X_valid[CATS].astype("category")
    X_test[CATS] = X_test[CATS].astype("category")

    # DROP PRICE THAT WAS USED FOR TARGET ENCODING
    X_train = X_train.drop(['Price'],axis=1)

    # BUILD MODEL
    model = XGBRegressor(
        device="cuda",
        max_depth=6,  
        colsample_bytree=0.5, 
        subsample=0.8,  
        n_estimators=1000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=10,
        early_stopping_rounds=100,
    )
    
    # TRAIN MODEL
    COLS = X_train.columns
    model.fit(
        X_train[COLS], y_train,
        eval_set=[(X_valid[COLS], y_valid)],  
        verbose=300,
    )

    # PREDICT OOF AND TEST
    oof[test_index] = model.predict(X_valid[COLS])
    pred += model.predict(X_test[COLS])

pred /= FOLDS




# # Overall CV Score
# Below we display overall cv score and save oof predictions to disk so we can use them later to assist finding ensemble weights with our other models.

# In[12]:


# COMPUTE OVERALL CV SCORE
true = train.Price.values
s = np.sqrt(np.mean( (oof-true)**2.0 ) )
print(f"=> Overall CV Score = {s}")

# In[13]:


# SAVE OOF TO DISK FOR ENSEMBLES
np.save(f"oof_v{VER}",oof)
print("Saved oof to disk")

# # Feature Names
# Below we list all our engineered features. We are using 57 features in total!

# In[14]:


print(f"\nIn total, we used {len(COLS)} features, Wow!\n")
print( list(COLS) )

# # XGB Feature Importance
# Here is XGBoost feature importance sorted by gain.

# In[15]:


import xgboost as xgb
fig, ax = plt.subplots(figsize=(10, 20))
xgb.plot_importance(model, max_num_features=100, importance_type='gain',ax=ax)
plt.title("Top 100 Feature Importances (XGBoost)")
plt.show()

# # Make Submission CSV
# We save our test predictions to submission.csv and plot our predictions. 

# In[18]:

sub = pd.read_csv(f"{path}/sample_submission.csv")
sub.Price = pred
sub.to_csv(f"submission_{s}.csv",index=False)
sub.head()

# In[19]:


plt.figure(figsize=(6,4))
plt.hist(sub.Price,bins=100)
plt.title("Test Predictions")
plt.show()
pd.read_csv(f"submission_{s}.csv")



s


pd.read_csv(f"submission_{s}.csv")

