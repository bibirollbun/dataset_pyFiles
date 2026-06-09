# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_squared_log_error
from xgboost import XGBClassifier, XGBRegressor
import warnings

warnings.filterwarnings('ignore')


# Configuration class, to help reduce redundancy in upcoming code
class CFG:
    train_path = '/kaggle/input/playground-series-s5e5/train.csv'
    test_path = '/kaggle/input/playground-series-s5e5/test.csv'
    sub_path = '/kaggle/input/playground-series-s5e5/sample_submission.csv'
    original_path = '/kaggle/input/calories-burnt-prediction/calories.csv'
    target = 'Calories'
    idx = 'id'
    n_splits = 5
    seed = 42


# Read the train, test and original data csv files into pandas DataFrames
train = pd.read_csv(CFG.train_path, index_col=CFG.idx)
test = pd.read_csv(CFG.test_path, index_col=CFG.idx)
original = pd.read_csv(CFG.original_path).drop('User_ID', axis=1).rename(columns={'Gender': 'Sex'})


# Display a sample of 10 data points from train
train.sample(10)


# Display a sample of 10 data points from test
test.sample(10)


# Display a sample of 10 data points from original
original.sample(10)


# Let's have a look at the shapes of the three datasets
print(f'Shape of train data: {train.shape}')
print(f'Shape of original data: {original.shape}')
print(f'Shape of test data: {test.shape}')


train.head()


# Check out the data types
display(train.dtypes.reset_index().rename(
    columns={'index': 'column', 0: 'dtype'}
))
print()
display(test.dtypes.reset_index().rename(
    columns={'index': 'column', 0: 'dtype'}
))
print()
display(original.dtypes.reset_index().rename(
    columns={'index': 'column', 0: 'dtype'}
))


features = [c for c in train.columns if c not in [CFG.target]]
num_features = [f for f in features if train[f].dtype != 'object']
cat_features = [f for f in features if f not in num_features]


class AdversarialValidation:
    def __init__(self, train, test, original, features, cat_features, num_features, target, params=None, paradigm='train_v_test', seed=99):
        self.train = train.copy()
        self.test = test.copy()
        self.original = original.copy()
        self.features = features
        self.cat_features = cat_features
        self.target = target
        self.seed = seed
        self.params = params or {
            'learning_rate': 0.05, 
            'max_depth': 4, 
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'objective': 'binary:logistic',
            'n_estimators': 100, 
            'gamma': 1, 
            'min_child_weight': 4,
            'verbosity': 0, 
            'enable_categorical': True,
            'eval_metric': 'logloss', 
            'early_stopping_rounds': 10,
            'tree_method': 'gpu_hist',
            'random_state': seed 
        }
        self.paradigm = 0  if paradigm == 'train_v_test' else 1

        if self.paradigm == 0:
            self.df1, self.df2 = self.train.copy(), self.test.copy()
        else:
            self.df1 = pd.concat([self.train, self.test], axis=0).sample(frac=1.0, random_state=self.seed)
            self.df2 = self.original.copy().drop(target, axis=1)

    def run(self):
        self.df1 = self.df1.drop(self.target, axis=1, errors='ignore')
        self.df1['cat_'] = 0
        self.df2['cat_'] = 1

        df = pd.concat([self.df1, self.df2], axis=0).sample(frac=1.0, random_state=self.seed)
        df_num = df[num_features+['cat_']]
        df_cat = df[self.cat_features].apply(lambda x: pd.factorize(x)[0])
        df = pd.concat([df_cat, df_num], axis=1)

        X = df.drop(columns=['cat_'])
        y = df['cat_']

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        fold_scores = []

        for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

            model = XGBClassifier(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            oof_preds = model.predict_proba(X_val)[:, 1]

            fold_score = roc_auc_score(y_val, oof_preds)
            fold_scores.append(fold_score)
            print(f'Fold {fold}: ROC-AUC score = {fold_score:4f}')

        print(f'\nAverage ROC-AUC score: {np.mean(fold_scores):.4f}Â±{np.std(fold_scores):.4f}')

        return fold_scores


av = AdversarialValidation(
    train=train, 
    test=test,
    original=original,
    features=features,
    cat_features=cat_features,
    num_features=num_features,
    target=CFG.target,
    paradigm='train_v_test',
    seed=CFG.seed
)
_ = av.run()


av = AdversarialValidation(
    train=train, 
    test=test,
    original=original,
    features=features,
    cat_features=cat_features,
    num_features=num_features,
    target=CFG.target,
    paradigm='comp_v_original',
    seed=CFG.seed
)
_ = av.run()


train = pd.read_csv(CFG.train_path, index_col=CFG.idx)
test = pd.read_csv(CFG.test_path, index_col=CFG.idx)
original = pd.read_csv(CFG.original_path).drop('User_ID', axis=1).rename(columns={'Gender': 'Sex'})

kf = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

features = [c for c in train.columns if c not in [CFG.target]]
num_features = [f for f in features if train[f].dtype != 'object']
cat_features = [f for f in features if f not in num_features]

train['is_original'] = False
test['is_original'] = False
original['is_original'] = True

features.append('is_original')
cat_features.append('is_original')

for f in cat_features:
    train[f] = train[f].astype('category')
    test[f] = test[f].astype('category')
    original[f] = original[f].astype('category')
    
combined_data = pd.concat([train, original], axis=0)
X = combined_data.drop(CFG.target, axis=1)
y = combined_data[CFG.target]
X_test = test[features]

eps = 1e-8
y = np.maximum(eps, y)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))
fold_scores = []

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'------------ Fold {fold + 1} ------------')
    X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    model = XGBRegressor(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=1000,
        objective='reg:squaredlogerror',  
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='gpu_hist',
        seed=CFG.seed,
        enable_categorical=True  
    )
    
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmsle',
        early_stopping_rounds=100,
        verbose=100
    )
    
    oof_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    oof_preds[val_idx] = oof_pred
    test_preds += test_pred / CFG.n_splits
    
    fold_score = np.sqrt(mean_squared_log_error(y_val, oof_pred))
    fold_scores.append(fold_score)
    print(f'Fold {fold + 1} RMSLE: {fold_score:.6f}\n')

overall_cv_score = np.sqrt(mean_squared_log_error(y, oof_preds))
print(f'Overall CV RMSLE: {overall_cv_score:.6f}')
print(f'{CFG.n_splits}-fold cross-validation mean RMSLE: {np.mean(fold_scores):.6f}Â±{np.std(fold_scores):.6f}')


sub = pd.read_csv(CFG.sub_path)
sub[CFG.target] = test_preds
sub.to_csv('submission_0_xgbwithoriginal.csv', index=False) # Scores 0.06492 on the LB


train = pd.read_csv(CFG.train_path, index_col=CFG.idx)
test = pd.read_csv(CFG.test_path, index_col=CFG.idx)
original = pd.read_csv(CFG.original_path).drop('User_ID', axis=1).rename(columns={'Gender': 'Sex'})

kf = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

features = [c for c in train.columns if c not in [CFG.target]]
num_features = [f for f in features if train[f].dtype != 'object']
cat_features = [f for f in features if f not in num_features]

train['is_original'] = False
test['is_original'] = False
original['is_original'] = True

features.append('is_original')
cat_features.append('is_original')

for f in cat_features:
    train[f] = train[f].astype('category')
    test[f] = test[f].astype('category')
    original[f] = original[f].astype('category')
    
X = train.drop(CFG.target, axis=1)
y = train[CFG.target]
X_test = test[features]

eps = 1e-8
y = np.maximum(eps, y)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))
fold_scores = []

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'------------ Fold {fold + 1} ------------')
    X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    model = XGBRegressor(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=1000,
        objective='reg:squaredlogerror',  
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='gpu_hist',
        seed=CFG.seed,
        enable_categorical=True  
    )
    
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmsle',
        early_stopping_rounds=100,
        verbose=100
    )
    
    oof_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    oof_preds[val_idx] = oof_pred
    test_preds += test_pred / CFG.n_splits
    
    fold_score = np.sqrt(mean_squared_log_error(y_val, oof_pred))
    fold_scores.append(fold_score)
    print(f'Fold {fold + 1} RMSLE: {fold_score:.6f}\n')

overall_cv_score = np.sqrt(mean_squared_log_error(y, oof_preds))
print(f'Overall CV RMSLE: {overall_cv_score:.6f}')
print(f'{CFG.n_splits}-fold cross-validation mean RMSLE: {np.mean(fold_scores):.6f}Â±{np.std(fold_scores):.6f}')


sub = pd.read_csv(CFG.sub_path)
sub[CFG.target] = test_preds
sub.to_csv('submission_1_xgbwithoutoriginal.csv', index=False) # Scores 0.06472 on the LB




