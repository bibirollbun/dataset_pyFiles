%load_ext cudf.pandas


import pandas as pd, numpy as np, os
PATH = '/kaggle/input/playground-series-s5e9/'
train = pd.read_csv(f'{PATH}train.csv').set_index('id')
print('Train Shape:', train.shape)
train.head()


test = pd.read_csv(f"{PATH}test.csv").set_index('id')
test['BeatsPerMinute'] = -1
print('Test Shape:', test.shape)
test.head()


ORIG_PATH = '/kaggle/input/bpm-prediction-challenge/'
orig_train = pd.read_csv(f"{ORIG_PATH}Train.csv")
print('orig_train Shape:', orig_train.shape)
orig_train.head()


combine = pd.concat([train,test,orig_train],axis=0)
print("Combined data shape", combine.shape )


CATS = []
NUMS = []
for c in combine.columns[:-1]:
    t = "CAT"
    if combine[c].dtype=='object':
        CATS.append(c)
    else:
        NUMS.append(c)
        t = "NUM"
    n = combine[c].nunique()
    na = combine[c].isna().sum()
    print(f"[{t}] {c} has {n} unique and {na} NA")
print("CATS:", CATS )
print("NUMS:", NUMS )


from cuml.preprocessing import TargetEncoder
from sklearn.model_selection import KFold
import xgboost as xgb

print(f"XGBoost version {xgb.__version__}")


FEATURES = NUMS
print(f"We have {len(FEATURES)} features.")

FOLDS = 14
SEED = 69

params = {
    "objective": "binary:logistic",  
    "eval_metric": "auc",           
    "learning_rate": 0.0635,
    "max_depth": 0,
    "subsample": 0.75,
    "colsample_bytree": 0.73,
    "seed": SEED,
    "device": "cuda",
    "grow_policy": "lossguide", 
    "max_leaves": 36,          
    "alpha": 2,
}


class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256*1024):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0 
        self.batch_size = batch_size
        self.batches = int( np.ceil( len(df) / self.batch_size ) )
        super().__init__()

    def reset(self):
        '''Reset the iterator'''
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data.'''
        if self.it == self.batches:
            return 0 # Return 0 when there's no more batch.
        
        a = self.it * self.batch_size
        b = min( (self.it + 1) * self.batch_size, len(self.df) )
        #dt = cudf.DataFrame(self.df.iloc[a:b])
        dt = self.df.iloc[a:b]
        input_data(data=dt[self.features], label=dt[self.target]) 
        self.it += 1
        return 1


# Training with XGBoost (Regression for BeatsPerMinute)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print("#" * 25)
    print(f"### Fold {fold+1} ###")
    print("#" * 25)

    # combine fold train + original smaller dataset
    Xy_train = train.iloc[train_idx][FEATURES + ['BeatsPerMinute']].copy()
    Xy_more = orig_train[FEATURES + ['BeatsPerMinute']]
    Xy_train = pd.concat([Xy_train, Xy_more], axis=0, ignore_index=True)

    X_valid = train.iloc[val_idx][FEATURES].copy()
    y_valid = train.iloc[val_idx]['BeatsPerMinute']
    X_test = test[FEATURES].copy()

    # Create DMatrix using IterLoader
    Xy_train_iter = IterLoadForDMatrix(Xy_train, FEATURES, 'BeatsPerMinute')
    dtrain = xgb.QuantileDMatrix(Xy_train_iter, enable_categorical=False, max_bin=256)
    dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=False)
    dtest  = xgb.DMatrix(X_test, enable_categorical=False)

    # Train model
    model = xgb.train(
        params={**params, "objective": "reg:squarederror", "eval_metric": "rmse"},
        dtrain=dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=200,
        verbose_eval=200
    )

    # Predict OOF and Test
    oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / FOLDS


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(train['BeatsPerMinute'], oof_preds))
mae = mean_absolute_error(train['BeatsPerMinute'], oof_preds)
r2 = r2_score(train['BeatsPerMinute'], oof_preds)

print(f"XGB with Original Data as rows CV RMSE = {rmse:.4f}")
print(f"XGB with Original Data as rows CV MAE  = {mae:.4f}")
print(f"XGB with Original Data as rows CV R²   = {r2:.4f}")


# Load sample submission
sub = pd.read_csv(f"{PATH}sample_submission.csv")

# Assign predictions
sub['BeatsPerMinute'] = test_preds

# Save submission file
sub.to_csv("submission.csv", index=False)
print('Submission shape:', sub.shape)
sub.head()

