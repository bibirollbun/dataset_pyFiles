%load_ext cudf.pandas


import pandas as pd, numpy as np, os

PATH = "/kaggle/input/playground-series-s5e8/"
train = pd.read_csv(f"{PATH}train.csv").set_index('id')
print("Train shape", train.shape )
train.head()


test = pd.read_csv(f"{PATH}test.csv").set_index('id')
test['y'] = -1
print("Test shape", test.shape )
test.head()


orig = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv",delimiter=";")
orig['y'] = orig.y.map({'yes':1,'no':0})
orig['id'] = (np.arange(len(orig))+1e6).astype('int')
orig = orig.set_index('id')
print("Original data shape", orig.shape )
orig.head()


def micro_fe(df, chap=False):
    
    df = df.copy()
    
    def f2(x):
        if x['default']=='no' and x['housing']=='no' and x['loan']=='no':
            return 21
        if x['default']=='no' and x['housing']=='no'\
        or x['default']=='no' and x['loan']=='no'\
        or x['housing']=='no' and x['loan']=='no':
            return 7
        if x['default']=='no' or x['housing']=='no' or x['loan']=='no':
            return 3
        return 0
        
    def f1(x):
        if x['education']=='unknown' and x['contact'] =='unknown' and x['poutcome']=='unknown':
            return 32
        if x['education']=='unknown' and x['contact'] =='unknown'\
        or x['education']=='unknown' and x['poutcome']=='unknown'\
        or x['contact']  =='unknown' and x['poutcome']=='unknown':
            return 8
        if x['education']=='unknown' or x['contact']=='unknown' or x['poutcome']=='unknown':
            return 4
        return 0

    def f3(x):
        if x['unknowns'] == 0 and x['many_no'] == 0: return 1
        if x['unknowns'] == 0 and x['many_no'] == 4: return 2
        if x['unknowns'] == 0 and x['many_no'] == 8: return 3
        if x['unknowns'] == 0 and x['many_no'] ==32: return 4
        if x['unknowns'] == 3 and x['many_no'] == 0: return 5
        if x['unknowns'] == 3 and x['many_no'] == 4: return 6
        if x['unknowns'] == 3 and x['many_no'] == 8: return 7
        if x['unknowns'] == 3 and x['many_no'] ==32: return 8
        if x['unknowns'] == 7 and x['many_no'] == 0: return 9
        if x['unknowns'] == 7 and x['many_no'] == 4: return 10
        if x['unknowns'] == 7 and x['many_no'] == 8: return 11
        if x['unknowns'] == 7 and x['many_no'] ==32: return 12
        if x['unknowns'] ==21 and x['many_no'] == 0: return 13
        if x['unknowns'] ==21 and x['many_no'] == 4: return 14
        if x['unknowns'] ==21 and x['many_no'] == 8: return 15
        if x['unknowns'] ==21 and x['many_no'] ==32: return 16
        return 0

    
    df['unknowns'] = df.apply(lambda x: f2(x), axis=1)
    df['many_no']  = df.apply(lambda x: f1(x), axis=1)
    df['unno']     = df.apply(lambda x: f3(x), axis=1)

    
    columns = df.columns.tolist()

    if 'y' in columns:   
        idxl = len(columns) - 1
        idxy = idxl - 3
        columns[idxl], columns[idxy] = columns[idxy], columns[idxl]
        df = df[columns]


    return df


%%time

train = micro_fe(train)
test  = micro_fe(test)
orig  = micro_fe(orig)


combine = pd.concat([train,test,orig],axis=0)
print("Combined data shape", combine.shape )
combine


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



CATS1 = []
SIZES = {}
for c in NUMS + CATS:
    n = c
    if c in NUMS: 
        n = f"{c}2"
        CATS1.append(n)
    combine[n],_ = combine[c].factorize()
    SIZES[n] = combine[n].max()+1

    combine[c] = combine[c].astype('int32')
    combine[n] = combine[n].astype('int32')

print("New CATS:", CATS1 )
print("Cardinality of all CATS:", SIZES )


from itertools import combinations

pairs = combinations(CATS + CATS1, 2)
new_cols = {}
CATS2 = []

for c1, c2 in pairs:
    s = SIZES[c1] * SIZES[c2]
    name = "_".join(sorted((c1, c2)))
    new_cols[name] = combine[c1] * SIZES[c2] + combine[c2]
    CATS2.append(name)
if new_cols:
    new_df = pd.DataFrame(new_cols)         
    combine = pd.concat([combine, new_df], axis=1) 

print(f"Created {len(CATS2)} new CAT columns")


CE = []
CC = CATS+CATS1+CATS2

print(f"Processing {len(CC)} columns... ",end="")
for i,c in enumerate(CC):
    if i%10==0: print(f"{i}, ",end="")
    tmp = combine.groupby(c).y.count()
    tmp = tmp.astype('int32')
    tmp.name = f"CE_{c}"
    CE.append( f"CE_{c}" )
    combine = combine.merge(tmp, on=c, how='left')
print()


train = combine.iloc[:len(train)]
test = combine.iloc[len(train):len(train)+len(test)]
orig = combine.iloc[-len(orig):]
del combine
print("Train shape", train.shape,"Test shape", test.shape,"Original shape", orig.shape )


from cuml.preprocessing import TargetEncoder
from sklearn.model_selection import KFold
import xgboost as xgb

print(f"XGBoost version {xgb.__version__}")


FEATURES = NUMS+CATS+CATS1+CATS2+CE
print(f"We have {len(FEATURES)} features.")

print('\n',FEATURES)

FOLDS = 7
SEED = 42

params = {
    "objective": "binary:logistic",  
    "eval_metric": "auc",           
    "learning_rate": 0.1,
    "max_depth": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "seed": SEED,
    "device": "cuda",
    "grow_policy": "lossguide", 
    "max_leaves": 32,          
    "alpha": 2.0,
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


oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {fold+1} ###")
    print("#"*25)

    Xy_train = train.iloc[train_idx][ FEATURES+['y'] ].copy()
    Xy_more = orig[ FEATURES+['y'] ]
    for k in range(1):
        Xy_train = pd.concat([Xy_train,Xy_more],axis=0,ignore_index=True)
    
    X_valid = train.iloc[val_idx][FEATURES].copy()
    y_valid = train.iloc[val_idx]['y']
    X_test = test[FEATURES].copy()

    CC = CATS1+CATS2
    print(f"Target encoding {len(CC)} features... ",end="")
    for i,c in enumerate(CC):
        if i%10==0: print(f"{i}, ",end="")
        TE0 = TargetEncoder(n_folds=10, smooth=0, split_method='random', stat='mean')
        Xy_train[c] = TE0.fit_transform(Xy_train[c],Xy_train['y']).astype('float32')
        X_valid[c] = TE0.transform(X_valid[c]).astype('float32')
        X_test[c] = TE0.transform(X_test[c]).astype('float32')
    print()

    Xy_train[CATS] = Xy_train[CATS].astype('category')
    X_valid[CATS] = X_valid[CATS].astype('category')
    X_test[CATS] = X_test[CATS].astype('category')

    Xy_train = IterLoadForDMatrix(Xy_train, FEATURES, 'y')
    dtrain = xgb.QuantileDMatrix(Xy_train, enable_categorical=True, max_bin=256)
    dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test, enable_categorical=True)

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


from sklearn.metrics import roc_auc_score

m = roc_auc_score(train.y, oof_preds)
print(f"XGB with Original Data as rows CV = {m}")


import xgboost as xgb
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))
xgb.plot_importance(model, max_num_features=20, importance_type='gain',ax=ax)
plt.title("Top 20 Feature Importances (XGBoost)")
plt.show()


TE_ORIG = []
CC = CATS+CATS1+CATS2

print(f"Processing {len(CC)} columns... ",end="")
for i,c in enumerate(CC):
    if i%10==0: print(f"{i}, ",end="")
    tmp = orig.groupby(c).y.mean()
    tmp = tmp.astype('float32')
    tmp.name = f"TE_ORIG_{c}"
    TE_ORIG.append( f"TE_ORIG_{c}" )
    train = train.merge(tmp, on=c, how='left')
    test = test.merge(tmp, on=c, how='left')
print()


FEATURES += TE_ORIG
print(f"We have {len(FEATURES)} features.")

FOLDS = 7
SEED = 42

params = {
    "objective": "binary:logistic",  
    "eval_metric": "auc",           
    "learning_rate": 0.1,
    "max_depth": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "seed": SEED,
    "device": "cuda",
    "grow_policy": "lossguide", 
    "max_leaves": 32,           
    "alpha": 2.0,
}


oof_preds2 = np.zeros(len(train))
test_preds2 = np.zeros(len(test))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {fold+1} ###")
    print("#"*25)

    Xy_train = train.iloc[train_idx][ FEATURES+['y'] ].copy()    
    X_valid = train.iloc[val_idx][FEATURES].copy()
    y_valid = train.iloc[val_idx]['y']
    X_test = test[FEATURES].copy()

    CC = CATS1+CATS2
    print(f"Target encoding {len(CC)} features... ",end="")
    for i,c in enumerate(CC):
        if i%10==0: print(f"{i}, ",end="")
        TE0 = TargetEncoder(n_folds=10, smooth=0, split_method='random', stat='mean')
        Xy_train[c] = TE0.fit_transform(Xy_train[c],Xy_train['y']).astype('float32')
        X_valid[c] = TE0.transform(X_valid[c]).astype('float32')
        X_test[c] = TE0.transform(X_test[c]).astype('float32')
    print()

    Xy_train[CATS] = Xy_train[CATS].astype('category')
    X_valid[CATS] = X_valid[CATS].astype('category')
    X_test[CATS] = X_test[CATS].astype('category')

    Xy_train = IterLoadForDMatrix(Xy_train, FEATURES, 'y')
    dtrain = xgb.QuantileDMatrix(Xy_train, enable_categorical=True, max_bin=256)
    dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test, enable_categorical=True)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=200,
        verbose_eval=200
    )

    oof_preds2[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    test_preds2 += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / FOLDS


from sklearn.metrics import roc_auc_score

m = roc_auc_score(train.y, oof_preds2)
print(f"XGB with Original Data as columns CV = {m}")


import xgboost as xgb
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))
xgb.plot_importance(model, max_num_features=20, importance_type='gain',ax=ax)
plt.title("Top 20 Feature Importances (XGBoost)")
plt.show()


m = roc_auc_score(train.y, oof_preds+oof_preds2)
print(f"Ensemble CV = {m}")


sub = pd.read_csv(f"{PATH}sample_submission.csv")
sub['y'] = (test_preds + test_preds2)/2.
sub.to_csv("submission.csv",index=False)
print('Submission shape',sub.shape)
sub.head()


plt.hist(sub.y,bins=100)
plt.title('Test Preds')
plt.ylim((0,10_000))
plt.show()



# Archive

# subm_A = pd.read_csv('/kaggle/input/21-august-2025-ps-s5e8/submission 0.97706.csv')
# subm_B = pd.read_csv('/kaggle/input/21-august-2025-ps-s5e8/submission 0.97707.csv') 
# subm_C = pd.read_csv('/kaggle/input/21-august-2025-ps-s5e8/submission 0.97708.csv')

# def straight_blend(df1,df2,df3, f, w=[0.33,0.33,0.34]):
#     df1['y'] = df1['y']*w[0] + w[1]*df2['y'] + w[2]*df3['y']
#     df1.to_csv(f, index=False)
#     display(df1)

# straight_blend(subm_A,subm_B,subm_C, 'submission 0.97709.csv', w=[0.33,0.33,0.34])

