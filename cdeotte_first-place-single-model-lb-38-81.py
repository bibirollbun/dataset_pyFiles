%load_ext cudf.pandas

import numpy as np, pandas as pd, gc
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)

VER=1


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
print("Train shape", train.shape )
train.head()


train2 = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
print("Extra Train shape", train2.shape )
train2.head()


train = pd.concat([train,train2],axis=0,ignore_index=True)
print("Combined Train shape", train.shape)


test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
print("Test shape", test.shape )
test.head()


CATS = list(train.columns[1:-2])
print(f"There are {len(CATS)} categorical columns:")
print( CATS )
print(f"There are 1 numerical column:")
print( ["Weight Capacity (kg)"] )


COMBO = ["NaNs"]
train["NaNs"] = np.float32(0)
test["NaNs"] = np.float32(0)

for i,c in enumerate(CATS):

    # NEW FEATURE - ENCODE ALL NAN AS ONE BASE-2 FEATURE
    train["NaNs"] += train[c].isna()*2**i
    test["NaNs"] += test[c].isna()*2**i

    # NEW FEATURE - COMBINE EACH COLUMN'S NAN WITH WEIGHT CAPACITY
    n = f"{c}_nan_wc"
    train[n] = train[c].isna()*100 + train["Weight Capacity (kg)"]
    test[n] = test[c].isna()*100 + test["Weight Capacity (kg)"]
    COMBO.append(n)
    
    combine = pd.concat([train[c],test[c]],axis=0)
    combine,_ = pd.factorize(combine)
    train[c] = combine[:len(train)].astype("float32")
    test[c] = combine[len(train):].astype("float32")
    n = f"{c}_wc"
    train[n] = train[c]*100 + train["Weight Capacity (kg)"]
    test[n] = test[c]*100 + test["Weight Capacity (kg)"]
    COMBO.append(n)


# NEW FEATURE - BIN WEIGHT CAPACITY USING ROUNDING
for k in range(7,10):
    n = f"round{k}"
    train[n] = train["Weight Capacity (kg)"].round(k)
    test[n] = test["Weight Capacity (kg)"].round(k)
    COMBO.append(n)


# NEW FEATURE - ORIGINAL DATASET PRICE
NEW_COLS = []
orig = pd.read_csv("/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")
tmp = orig.groupby("Weight Capacity (kg)").Price.mean()
tmp.name = "orig_price"
train = train.merge(tmp, on="Weight Capacity (kg)", how="left")
test = test.merge(tmp, on="Weight Capacity (kg)", how="left")
NEW_COLS.append("orig_price")


# NEW FEATURE - ORIGINAL DATASET PRICE FROM ROUNDED WEIGHT CAPACITY 
for k in range(7,10):
    n = f"round{k}"
    orig[n] = orig["Weight Capacity (kg)"].round(k)
    tmp = orig.groupby(n).Price.mean()
    tmp.name = f"orig_price_r{k}"
    train = train.merge(tmp, on=n, how="left")
    test = test.merge(tmp, on=n, how="left")
    NEW_COLS.append(f"orig_price_r{k}")


# NEW FEATURE - DIGIT EXTRACTION FROM WEIGHT CAPACITY
for k in range(1,10):
    train[f'digit{k}'] = ((train['Weight Capacity (kg)'] * 10**k) % 10).fillna(-1).astype("int8")
    test[f'digit{k}'] = ((test['Weight Capacity (kg)'] * 10**k) % 10).fillna(-1).astype("int8")
DIGITS = [f"digit{k}" for k in range(1,10)]


# NEW FEATURE - COMBINATIONS OF DIGITS 
for i in range(4):
    for j in range(i+1,5):
        n = f"digit_{i+1}_{j+1}"
        train[n] = ((train[f'digit{i+1}']+1)*11 + train[f'digit{j+1}']+1).astype("int8")
        test[n] = ((test[f'digit{i+1}']+1)*11 + test[f'digit{j+1}']+1).astype("int8")
        COMBO.append(n)


# NEW FEATURE - COMBINATIONS OF CATS
PAIRS = []
for i,c1 in enumerate(CATS[:-1]):
    for j,c2 in enumerate(CATS[i+1:]):
        n = f"{c1}_{c2}"
        m1 = train[c1].max()+1
        m2 = train[c2].max()+1
        train[n] = ((train[c1]+1 + (train[c2]+1)/(m2+1))*(m2+1)).astype("int8")
        test[n] = ((test[c1]+1 + (test[c2]+1)/(m2+1))*(m2+1)).astype("int8")
        COMBO.append(n)
        PAIRS.append(n)


print(f"New Train shape:", train.shape )
train.head()


FEATURES = CATS + ["Weight Capacity (kg)"] + COMBO + DIGITS + NEW_COLS
print(f"We now have {len(FEATURES)} columns:")
print( FEATURES )


from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import xgboost as xgb
print(f"XGBoost version",xgb.__version__)


# STATISTICS TO AGGEGATE FOR OUR FEATURE GROUPS
STATS = ["mean","std","count","nunique","median","min","max","skew"]
STATS2 = ["mean"]


# QUANTILES AND HISTOGRAM BINS TO AGGREGATE
BINS=10
QUANTILES = [5,10,40,45,55,60,90,95]
def make_histogram(prices, bins=BINS, range_min=15, range_max=150):
    hist, _ = np.histogram(prices, bins=bins, range=(range_min, range_max))
    return hist


%%time

FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros((len(train)))
pred = np.zeros((len(test)))

# OUTER K FOLD
for i, (train_index, test_index) in enumerate(kf.split(train)):
    print(f"### OUTER Fold {i+1} ###")

    X_train = train.loc[train_index,FEATURES+['Price']].reset_index(drop=True).copy()
    y_train = train.loc[train_index,'Price']

    X_valid = train.loc[test_index,FEATURES].reset_index(drop=True).copy()
    y_valid = train.loc[test_index,'Price']

    X_test = test[FEATURES].reset_index(drop=True).copy()

    # INNER K FOLD (TO PREVENT LEAKAGE WHEN USING PRICE)
    kf2 = KFold(n_splits=FOLDS, shuffle=True, random_state=42)   
    for j, (train_index2, test_index2) in enumerate(kf2.split(X_train)):
        print(f" ## INNER Fold {j+1} (outer fold {i+1}) ##")

        X_train2 = X_train.loc[train_index2,FEATURES+['Price']].copy()
        X_valid2 = X_train.loc[test_index2,FEATURES].copy()

        ### FEATURE SET 1 (uses price) ###
        col = "Weight Capacity (kg)"
        tmp = X_train2.groupby(col).Price.agg(STATS)
        tmp.columns = [f"TE1_wc_{s}" for s in STATS]
        X_valid2 = X_valid2.merge(tmp, on=col, how="left")
        for c in tmp.columns:
            X_train.loc[test_index2,c] = X_valid2[c].values.astype("float32")

        ### FEATURE SET 2 (uses price) ###
        for col in COMBO:
            tmp = X_train2.groupby(col).Price.agg(STATS2)
            tmp.columns = [f"TE2_{col}_{s}" for s in STATS2]
            X_valid2 = X_valid2.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                X_train.loc[test_index2,c] = X_valid2[c].values.astype("float32")

        # AGGREGATE QUANTILES (uses price)
        for k in QUANTILES:
            result = X_train2.groupby('Weight Capacity (kg)').agg({'Price': lambda x: x.quantile(k/100)})
            result.columns = [f"quantile_{k}"]
            X_valid2 = X_valid2.merge(result, on="Weight Capacity (kg)", how="left")
            X_train.loc[test_index2,f"quantile_{k}"] = X_valid2[f"quantile_{k}"].values.astype("float32")

        # AGGREGATE HISTOGRAMS (uses price)
        tmp = X_train2.loc[~X_train2.orig_price.isna()].groupby("Weight Capacity (kg)")[["Price"]].agg("count")
        tmp.columns = ['ct']
        X_train3 = X_train2.merge(tmp.loc[tmp['ct']>1],on="Weight Capacity (kg)",how="left")
        X_train3 = X_train3.loc[~X_train3['ct'].isna()]
        result = X_train3.groupby("Weight Capacity (kg)")["Price"].apply(make_histogram)
        result = result.to_frame()['Price'].apply(pd.Series)
        result.columns = [f"histogram_{x}" for x in range(BINS)]
        X_valid2 = X_valid2.merge(result, on="Weight Capacity (kg)", how="left")
        for c in [f"histogram_{x}" for x in range(BINS)]:
            X_train.loc[test_index2,c] = X_valid2[c].values.astype("float32")
            
        del result, X_train3, tmp
        del X_train2, X_valid2
        gc.collect()

    ### FEATURE SET 1 (uses price) ###
    col = "Weight Capacity (kg)"
    tmp = X_train.groupby(col).Price.agg(STATS)
    tmp.columns = [f"TE1_wc_{s}" for s in STATS]
    tmp = tmp.astype("float32")
    X_valid = X_valid.merge(tmp, on=col, how="left")
    X_test = X_test.merge(tmp, on=col, how="left")

    ### FEATURE SET 2 (uses price) ###
    for col in COMBO:
        tmp = X_train.groupby(col).Price.agg(STATS2)
        tmp.columns = [f"TE2_{col}_{s}" for s in STATS2]
        tmp = tmp.astype("float32")
        X_valid = X_valid.merge(tmp, on=col, how="left")
        X_test = X_test.merge(tmp, on=col, how="left")

    # AGGREGATE QUANTILES (uses price)
    for k in QUANTILES:
        result = X_train.groupby('Weight Capacity (kg)').agg({'Price': lambda x: x.quantile(k/100)})
        result.columns = [f"quantile_{k}"]
        result = result.astype("float32")
        X_valid = X_valid.merge(result, on="Weight Capacity (kg)", how="left")
        X_test = X_test.merge(result, on="Weight Capacity (kg)", how="left")

    # AGGREGATE HISTOGRAMS (uses price)
    tmp = X_train.loc[~X_train.orig_price.isna()].groupby("Weight Capacity (kg)")[["Price"]].agg("count")
    tmp.columns = ['ct']
    X_train3 = X_train.merge(tmp.loc[tmp['ct']>1],on="Weight Capacity (kg)",how="left")
    X_train3 = X_train3.loc[~X_train3['ct'].isna()]
    result = X_train3.groupby("Weight Capacity (kg)")["Price"].apply(make_histogram)
    result = result.to_frame()['Price'].apply(pd.Series)
    result.columns = [f"histogram_{x}" for x in range(BINS)]
    result = result.astype("float32")
    X_valid = X_valid.merge(result, on="Weight Capacity (kg)", how="left")
    X_test = X_test.merge(result, on="Weight Capacity (kg)", how="left")
    del result, X_train3, tmp

    # COUNT PER NUNIQUE
    X_train['TE1_wc_count_per_nunique'] = X_train['TE1_wc_count']/X_train['TE1_wc_nunique']
    X_valid['TE1_wc_count_per_nunique'] = X_valid['TE1_wc_count']/X_valid['TE1_wc_nunique']
    X_test['TE1_wc_count_per_nunique'] = X_test['TE1_wc_count']/X_test['TE1_wc_nunique']
    
    # STD PER COUNT
    X_train['TE1_wc_std_per_count'] = X_train['TE1_wc_std']/X_train['TE1_wc_count']
    X_valid['TE1_wc_std_per_count'] = X_valid['TE1_wc_std']/X_valid['TE1_wc_count']
    X_test['TE1_wc_std_per_count'] = X_test['TE1_wc_std']/X_test['TE1_wc_count']

    # CONVERT TO CATS SO XGBOOST RECOGNIZES THEM
    X_train[CATS+DIGITS] = X_train[CATS+DIGITS].astype("category")
    X_valid[CATS+DIGITS] = X_valid[CATS+DIGITS].astype("category")
    X_test[CATS+DIGITS] = X_test[CATS+DIGITS].astype("category")

    # DROP PRICE THAT WAS USED FOR TARGET ENCODING
    X_train = X_train.drop(['Price'],axis=1)

    # DROP NON-TE CAT PAIRS
    X_train = X_train.drop(PAIRS,axis=1)
    X_valid = X_valid.drop(PAIRS,axis=1)
    X_test = X_test.drop(PAIRS,axis=1)

    # BUILD MODEL
    model = XGBRegressor(
        device="cuda",
        max_depth=6,  
        colsample_bynode=0.3, 
        subsample=0.8,  
        n_estimators=50_000,  
        learning_rate=0.01,  
        enable_categorical=True,
        min_child_weight=10,
        early_stopping_rounds=500,
    )
    
    # TRAIN MODEL
    COLS = X_train.columns
    model.fit(
        X_train[COLS], y_train,
        eval_set=[(X_valid[COLS], y_valid)],  
        verbose=500,
    )

    # PREDICT OOF AND TEST
    oof[test_index] = model.predict(X_valid[COLS])
    pred += model.predict(X_test[COLS])

    # CLEAR MEMORY
    del X_train, X_valid, X_test
    del y_train, y_valid
    if i != FOLDS-1: del model
    gc.collect()

pred /= FOLDS


# COMPUTE OVERALL CV SCORE
true = train.Price.values
s = np.sqrt(np.mean( (oof-true)**2.0 ) )
print(f"=> Overall CV Score = {s}")


# SAVE OOF TO DISK FOR ENSEMBLES
np.save(f"oof_v{VER}",oof)
print("Saved oof to disk")


print(f"\nIn total, we used {len(COLS)} features, Wow!\n")
print( list(COLS) )


import xgboost as xgb
fig, ax = plt.subplots(figsize=(10, 20))
xgb.plot_importance(model, max_num_features=100, importance_type='gain',ax=ax)
plt.title("Top 100 Feature Importances (XGBoost)")
plt.show()


sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub.Price = pred
sub.to_csv(f"submission_v{VER}.csv",index=False)
sub.head()


plt.figure(figsize=(6,4))
plt.hist(sub.Price,bins=100)
plt.title("Test Predictions")
plt.show()

