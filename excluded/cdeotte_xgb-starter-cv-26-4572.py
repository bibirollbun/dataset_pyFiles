import pandas as pd, numpy as np

train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
print("Train shape:", train.shape )
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
test['BeatsPerMinute'] = -1
print("Test shape:", test.shape )
test.head()


orig = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Train.csv")
print("Original data shape:", orig.shape )
orig.head()


combine = pd.concat([train,test,orig],axis=0,ignore_index=True)
print("Combine shape:", combine.shape )
combine.head()


FEATURES = list( orig.columns[:-1] )
TARGET = orig.columns[-1]
print(f"Features: {FEATURES}, Target: ''{TARGET}''")


DIGITS = []
for c in ['Energy','MoodScore','AcousticQuality']:
    for k in range(1,10):
        n = f'{c}_d{k}'
        combine[n] = ((combine[c] * 10**k) % 10).fillna(-1).astype("int8")
        DIGITS.append(n)


ROUND = []
RR = [9,8]
for c in FEATURES:
    print(f"{c}, ",end="")
    for r in RR:
        n = f"{c}_r{r}"
        combine[n] = combine[c].round(r)
        ROUND.append(n)


train = combine.iloc[:len(train)]
test = combine.iloc[len(train):len(train)+len(test)]
orig = combine.iloc[-len(orig):]
print(f"Train shape: {train.shape}, Test shape: {test.shape}, Original data shape: {orig.shape}")


TE = []
print(f"Processing {len(FEATURES+ROUND)} features... ",end="")
for c in FEATURES+ROUND:
    tmp = orig.groupby(c)[TARGET].mean()
    n = f"TE0_{c}"
    print(f"{n}, ",end="")
    tmp.name = n
    train = train.merge(tmp, on=c, how='left')
    test = test.merge(tmp, on=c, how='left')
    TE.append(n)


from cuml.preprocessing import TargetEncoder
from sklearn.model_selection import KFold
import xgboost as xgb

print(f"XGBoost version {xgb.__version__}")


FOLDS = 7
SEED = 42

params = {
    "objective": "reg:squarederror",   
    "eval_metric": "rmse",             
    "learning_rate": 0.002,
    "max_depth": 5,                    
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": SEED,
    "device": "cuda",
}


oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {fold+1} ###")
    print("#"*25)

    X_train = train.iloc[train_idx][FEATURES+TE+ROUND+DIGITS].copy()
    y_train = train.iloc[train_idx][TARGET]
    
    X_valid = train.iloc[val_idx][FEATURES+TE+ROUND+DIGITS].copy()
    y_valid = train.iloc[val_idx][TARGET]
    X_test = test[FEATURES+TE+ROUND+DIGITS].copy()

    CC = FEATURES+ROUND
    print(f"Target encoding {len(CC)} features... ",end="")
    for i,c in enumerate(CC):
        if i%5==0: print(f"{i}, ",end="")
        n = f"TE_{c}"
        TE0 = TargetEncoder(n_folds=10, smooth=4, split_method='random', stat='mean')
        X_train[n] = TE0.fit_transform(X_train[c],y_train).astype('float32')
        X_valid[n] = TE0.transform(X_valid[c]).astype('float32')
        X_test[n] = TE0.transform(X_test[c]).astype('float32')            
    print()

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=False)
    dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=False)
    dtest  = xgb.DMatrix(X_test, enable_categorical=False)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=200,
        verbose_eval=200
    )

    oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / FOLDS


m = np.sqrt( np.mean( (oof_preds - train[TARGET].values)**2. ) )
print(f" Overall CV RMSE = {m}")


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 25))
xgb.plot_importance(model, max_num_features=100, importance_type='gain',ax=ax)
plt.title("Top 100 Feature Importances (XGBoost)")
plt.show()


sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
sub.BeatsPerMinute = test_preds
sub.to_csv("submission.csv",index=False)
sub.head()


test_preds_full = np.zeros(len(test))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {fold+1} ###")
    print("#"*25)

    # WE NOW USE 100% TRAIN HERE
    X_train = train[FEATURES+TE+ROUND+DIGITS].copy()
    y_train = train[TARGET]
    
    X_valid = train.iloc[val_idx][FEATURES+TE+ROUND+DIGITS].copy()
    y_valid = train.iloc[val_idx][TARGET]
    X_test = test[FEATURES+TE+ROUND+DIGITS].copy()

    CC = FEATURES+ROUND
    print(f"Target encoding {len(CC)} features... ",end="")
    for i,c in enumerate(CC):
        if i%5==0: print(f"{i}, ",end="")
        n = f"TE_{c}"
        TE0 = TargetEncoder(n_folds=10, smooth=4, split_method='random', stat='mean')
        X_train[n] = TE0.fit_transform(X_train[c],y_train).astype('float32')
        X_valid[n] = TE0.transform(X_valid[c]).astype('float32')
        X_test[n] = TE0.transform(X_test[c]).astype('float32')            
    print()

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=False)
    dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=False)
    dtest  = xgb.DMatrix(X_test, enable_categorical=False)

    # WE CHANGE SEED EACH FOLD AND USE FIXED 2440 ITERATIONS
    params['seed'] = fold
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2440,
        evals=[(dtrain, "train"), (dval, "valid")],
        verbose_eval=200
    )

    test_preds_full += model.predict(dtest) / FOLDS


sub.BeatsPerMinute = test_preds_full
sub.to_csv("submission_refit_full.csv",index=False)
sub.head()

