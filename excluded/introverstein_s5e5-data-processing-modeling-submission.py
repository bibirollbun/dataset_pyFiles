import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


class CFG:
    train_path = '/kaggle/input/playground-series-s5e5/train.csv'
    test_path = '/kaggle/input/playground-series-s5e5/test.csv'
    sub_path = '/kaggle/input/playground-series-s5e5/sample_submission.csv'
    target = 'Calories'
    n_splits = 5
    idx = 'id'
    seed = 42


train = pd.read_csv(CFG.train_path, index_col=CFG.idx)
test = pd.read_csv(CFG.test_path, index_col=CFG.idx)

print(f'Shape of training data: {train.shape}')
print(f'Shape of testing data: {test.shape}')


features = [f for f in test.columns if f != CFG.target]

for f in features:
    print(f'Feature: {f} | Data type: {train[f].dtype}')
    print(f'{train[f].nunique()} unique values in train set')
    print(f'{test[f].nunique()} unique values in test set.\n')


cat_features = ['Sex']
num_features = [f for f in features if f not in cat_features]


train.isnull().sum()


test.isnull().sum()


class LabelEncoderV0:
    def __init__(self, train, test, cat_features):
        self.train = train.copy()
        self.test = test.copy()
        self.cat_features = cat_features.copy()

    def run(self):
        train_size = len(self.train)
        combined = pd.concat([self.train, self.test], axis=0)
        
        for f in self.cat_features:
            le = LabelEncoder()
            combined[f] = le.fit_transform(combined[f])

        train_ = combined.iloc[:train_size]
        test_ = combined.iloc[train_size:].drop(CFG.target, axis=1, errors='ignore')

        return train_, test_


class NumericalImputer:
    def __init__(self, num_features, strategy='median'):
        self.num_features = num_features
        self.strategy = strategy
        self.feat_to_val = {}
        
    def fit(self, df):
        self.feat_to_val = {
            f: df[f].median() if self.strategy == 'median' else df[f].mean()
            for f in self.num_features
        }
        return self
        
    def transform(self, df):
        df = df.copy()
        for feat, val in self.feat_to_val.items():
            df[feat] = df[feat].fillna(val)
        return df
        
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


class ProcessData:
    """
    Prepare data for modeling.
    """
    def __init__(self, CFG, cat_features, num_features, one_hot_encoding=False):
        self.train = pd.read_csv(CFG.train_path, index_col=CFG.idx)
        self.test = pd.read_csv(CFG.test_path, index_col=CFG.idx)
        self.cat_features = cat_features
        self.num_features = num_features
        self.features = self.cat_features + self.num_features
        self.ordinal_features = []
        self.one_hot_encoding = one_hot_encoding
      

    def label_encode(self, cols_to_labelencode=None):
        if cols_to_labelencode is None:
            cols_to_labelencode = self.cat_features
            
        le = LabelEncoderV0(self.train, self.test, cols_to_labelencode)
        self.train, self.test = le.run()

    def typecast_categoricals(self, cols=None):
        if cols is None:
            cols = [f for f in self.cat_features if f not in self.ordinal_features]
            
        for f in cols:
            self.train[f] = self.train[f].astype(str)
            self.test[f] = self.test[f].astype(str)

    def typecast_columns(self, cols=None):
        if cols is None:
            cols = self.features

        for f in cols:
            if self.train[f].dtype == 'int64':
                self.train[f] = self.train[f].astype('int32')
                self.test[f] = self.test[f].astype('int32')
                
            elif self.train[f].dtype == 'float64':
                self.train[f] = self.train[f].astype('float32')
                self.test[f] = self.test[f].astype('float32')

    def one_hot_encode(self, cols_to_ohe=None):
        if cols_to_ohe is None:
            cols_to_ohe = self.cat_features

        # Need to type cast the columns to string data type for pd.get_dummies to work
        for c in cols_to_ohe:
            self.train[c] = self.train[c].astype(str)
            self.test[c] = self.test[c].astype(str)
            
        keep_cols = [col for col in self.features if col not in cols_to_ohe]
        self.cat_features = [f for f in self.cat_features if f not in cols_to_ohe]
        self.features = [f for f in self.features if f not in cols_to_ohe]
        
        train_dummies = pd.get_dummies(self.train[cols_to_ohe], prefix_sep='_', dtype=int)
        test_dummies = pd.get_dummies(self.test[cols_to_ohe], prefix_sep='_', dtype=int)
        
        self.train = pd.concat([self.train[keep_cols], train_dummies, self.train[CFG.target]], axis=1)
        self.test = pd.concat([self.test[keep_cols], test_dummies], axis=1)

        new_features = train_dummies.columns.tolist()
        self.features.extend(new_features)
        self.cat_features.extend(new_features)

    def run(self):
        self.label_encode(cols_to_labelencode=['Sex'])

        if self.one_hot_encoding:
            self.one_hot_encode()
            
        self.typecast_columns()
                
    def get_output(self):
        return self.train, self.test, self.features, self.cat_features, self.num_features


train = pd.read_csv(CFG.train_path, index_col=CFG.idx)
test = pd.read_csv(CFG.test_path, index_col=CFG.idx)

features = [f for f in test.columns if f != CFG.target]
cat_features = ['Sex']
num_features = [f for f in features if f not in cat_features]

pipeline = ProcessData(CFG, cat_features, num_features, one_hot_encoding=False)
pipeline.run()

train, test, features, cat_features, num_features = pipeline.get_output()

kf = KFold(
    n_splits=CFG.n_splits,
    shuffle=True,
    random_state=CFG.seed
)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_scores = []

X = train.drop(CFG.target, axis=1)
y = train[CFG.target]
X_test = test

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'------------ Fold {fold + 1} ------------')
    
    X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    X_train_cat = X_train[cat_features]
    X_val_cat = X_val[cat_features]
    X_test_cat = X_test[cat_features]
    
    poly = PolynomialFeatures(
        degree=2,
        interaction_only=False,
        include_bias=False
    )
    
    X_train_poly = poly.fit_transform(X_train[num_features])
    X_val_poly = poly.transform(X_val[num_features])
    X_test_poly = poly.transform(X_test[num_features])
    
    poly_feature_names = poly.get_feature_names_out(input_features=num_features)
    X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_feature_names, index=X_train.index)
    X_val_poly_df = pd.DataFrame(X_val_poly, columns=poly_feature_names, index=X_val.index)
    X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names, index=X_test.index)
    
    X_train = pd.concat([X_train_poly_df, X_train_cat], axis=1)
    X_val = pd.concat([X_val_poly_df, X_val_cat], axis=1)
    X_test = pd.concat([X_test_poly_df, X_test_cat], axis=1)
    
    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_val[num_features] = scaler.transform(X_val[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])
    
    model = LinearRegression()
    y_train_log = np.log1p(y_train)
    model.fit(X_train, y_train_log)
    oof_pred_log = model.predict(X_val)
    test_pred_log = model.predict(X_test)
    
    oof_pred = np.expm1(oof_pred_log)
    test_pred = np.expm1(test_pred_log)
    
    oof_preds[val_idx] = oof_pred
    test_preds += test_pred / CFG.n_splits
    fold_score = rmsle(y_val, oof_pred)
    fold_scores.append(fold_score)
    print(f'Fold score: {fold_score}\n')

overall_cv_score = rmsle(y, oof_preds)
print(f'Overall CV score: {overall_cv_score}')
print(f'{CFG.n_splits}-fold cross-validation score: {np.mean(fold_scores)}Â±{np.std(fold_scores)}')


sub = pd.read_csv(CFG.sub_path)
sub[CFG.target] = test_preds
sub.to_csv('submission_baseline_linearregression.csv', index=False)


train = pd.read_csv(CFG.train_path, index_col=CFG.idx)
test = pd.read_csv(CFG.test_path, index_col=CFG.idx)

features = [f for f in test.columns if f != CFG.target]
cat_features = ['Sex']
num_features = [f for f in features if f not in cat_features]

pipeline = ProcessData(CFG, cat_features, num_features)
pipeline.run()

train, test, features, cat_features, num_features = pipeline.get_output()

kf = KFold(
    n_splits=CFG.n_splits,
    shuffle=True,
    random_state=CFG.seed
)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_scores = []

X = train.drop(CFG.target, axis=1)
y = train[CFG.target]
X_test = test

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'------------ Fold {fold + 1} ------------')
    
    X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    y_train_log = np.log1p(y_train)
    
    model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function='RMSE',
        random_seed=CFG.seed,
        verbose=100,
        cat_features=cat_features,
        task_type='GPU'
    )
    model.fit(X_train, y_train_log)
    
    oof_pred_log = model.predict(X_val)
    test_pred_log = model.predict(X_test)
    
    oof_pred = np.expm1(oof_pred_log)
    test_pred = np.expm1(test_pred_log)
    
    oof_preds[val_idx] = oof_pred
    test_preds += test_pred / CFG.n_splits
    
    fold_score = rmsle(y_val, oof_pred)
    fold_scores.append(fold_score)
    print(f'Fold score: {fold_score}\n')

overall_cv_score = rmsle(y, oof_preds)
print(f'Overall CV score: {overall_cv_score}')
print(f'{CFG.n_splits}-fold cross-validation score: {np.mean(fold_scores)}Â±{np.std(fold_scores)}')


sub = pd.read_csv(CFG.sub_path)
sub[CFG.target] = test_preds
sub.to_csv('submission_baseline_catboost.csv', index=False)


train = pd.read_csv(CFG.train_path, index_col=CFG.idx)
test = pd.read_csv(CFG.test_path, index_col=CFG.idx)

features = [f for f in test.columns if f != CFG.target]
cat_features = ['Sex']
num_features = [f for f in features if f not in cat_features]

pipeline = ProcessData(CFG, cat_features, num_features)
pipeline.run()
train, test, features, cat_features, num_features = pipeline.get_output()

kf = KFold(
    n_splits=CFG.n_splits,
    shuffle=True,
    random_state=CFG.seed
)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_scores = []

X = train.drop(CFG.target, axis=1)
y = train[CFG.target]
X_test = test

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'------------ Fold {fold + 1} ------------')
    
    X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    model = XGBRegressor(
        objective='reg:squarederror',   
        max_depth=6,                   
        learning_rate=0.05,            
        n_estimators=1000,             
        random_state=CFG.seed,
        tree_method='gpu_hist',        
        eval_metric='rmse',            
        early_stopping_rounds=100,
        enable_categorical=True,
        verbosity=1
    )
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_train, y_train_log), (X_val, y_val_log)],  
        verbose=100                     
    )
    
    oof_pred_log = model.predict(X_val)
    test_pred_log = model.predict(X_test)
    
    oof_pred = np.expm1(oof_pred_log)
    test_pred = np.expm1(test_pred_log)
    
    oof_preds[val_idx] = oof_pred
    test_preds += test_pred / CFG.n_splits
    
    fold_score = rmsle(y_val, oof_pred)
    fold_scores.append(fold_score)
    print(f'Fold score: {fold_score}\n')

overall_cv_score = rmsle(y, oof_preds)
print(f'Overall CV score: {overall_cv_score}')
print(f'{CFG.n_splits}-fold cross-validation score: {np.mean(fold_scores)}Â±{np.std(fold_scores)}')


sub = pd.read_csv(CFG.sub_path)
sub[CFG.target] = test_preds
sub.to_csv('submission_baseline_xgb.csv', index=False)


train = pd.read_csv(CFG.train_path, index_col=CFG.idx)
test = pd.read_csv(CFG.test_path, index_col=CFG.idx)

features = [f for f in test.columns if f != CFG.target]
cat_features = ['Sex']
num_features = [f for f in features if f not in cat_features]

pipeline = ProcessData(CFG, cat_features, num_features)
pipeline.run()
train, test, features, cat_features, num_features = pipeline.get_output()

kf = KFold(
    n_splits=CFG.n_splits,
    shuffle=True,
    random_state=CFG.seed
)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_scores = []

X = train.drop(CFG.target, axis=1)
y = train[CFG.target]
X_test = test

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    
for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'------------ Fold {fold + 1} ------------')
    
    X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    cat_indices = [X.columns.get_loc(col) for col in cat_features if col in X.columns]
    
    model = LGBMRegressor(
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        max_depth=6,
        learning_rate=0.05,
        n_estimators=1000,
        num_leaves=63,
        random_state=CFG.seed,
        verbosity=-1,
        device='gpu',
        importance_type='gain',
        verbose=100
    )
    
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_train, y_train_log), (X_val, y_val_log)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=100)],
        categorical_feature=cat_indices
    )
    
    oof_pred_log = model.predict(X_val)
    test_pred_log = model.predict(X_test)
    
    oof_pred = np.expm1(oof_pred_log)
    test_pred = np.expm1(test_pred_log)
    
    oof_preds[val_idx] = oof_pred
    test_preds += test_pred / CFG.n_splits
    
    fold_score = rmsle(y_val, oof_pred)
    fold_scores.append(fold_score)
    print(f'Fold score: {fold_score}\n')
overall_cv_score = rmsle(y, oof_preds)
print(f'Overall CV score: {overall_cv_score}')
print(f'{CFG.n_splits}-fold cross-validation score: {np.mean(fold_scores)}Â±{np.std(fold_scores)}')


sub = pd.read_csv(CFG.sub_path)
sub[CFG.target] = test_preds
sub.to_csv('submission_baseline_lgbm_gbdt.csv', index=False)


train = pd.read_csv(CFG.train_path, index_col=CFG.idx)
test = pd.read_csv(CFG.test_path, index_col=CFG.idx)

features = [f for f in test.columns if f != CFG.target]
cat_features = ['Sex']
num_features = [f for f in features if f not in cat_features]

pipeline = ProcessData(CFG, cat_features, num_features)
pipeline.run()
train, test, features, cat_features, num_features = pipeline.get_output()

kf = KFold(
    n_splits=CFG.n_splits,
    shuffle=True,
    random_state=CFG.seed
)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_scores = []

X = train.drop(CFG.target, axis=1)
y = train[CFG.target]
X_test = test

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    
for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'------------ Fold {fold + 1} ------------')
    
    X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    cat_indices = [X.columns.get_loc(col) for col in cat_features if col in X.columns]
    
    model = LGBMRegressor(
        objective='regression',
        metric='rmse',
        boosting_type='goss',
        max_depth=6,
        learning_rate=0.05,
        n_estimators=1000,
        num_leaves=63,
        random_state=CFG.seed,
        verbosity=-1,
        device='gpu',
        importance_type='gain',
        verbose=100
    )
    
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_train, y_train_log), (X_val, y_val_log)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=100)],
        categorical_feature=cat_indices
    )
    
    oof_pred_log = model.predict(X_val)
    test_pred_log = model.predict(X_test)
    
    oof_pred = np.expm1(oof_pred_log)
    test_pred = np.expm1(test_pred_log)
    
    oof_preds[val_idx] = oof_pred
    test_preds += test_pred / CFG.n_splits
    
    fold_score = rmsle(y_val, oof_pred)
    fold_scores.append(fold_score)
    print(f'Fold score: {fold_score}\n')
overall_cv_score = rmsle(y, oof_preds)
print(f'Overall CV score: {overall_cv_score}')
print(f'{CFG.n_splits}-fold cross-validation score: {np.mean(fold_scores)}Â±{np.std(fold_scores)}')


sub = pd.read_csv(CFG.sub_path)
sub[CFG.target] = test_preds
sub.to_csv('submission_baseline_lgbm_goss.csv', index=False)




