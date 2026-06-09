import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb


RANDOM_SEED = 42
N_FOLDS = 5

# show files
for p, d, files in os.walk('/kaggle/input'):
    if files:
        print(p)
        break


def load_data():
    # common names used in playground series
    candidates = ['train.csv','test.csv','sample_submission.csv']
    base_dir = '/kaggle/input'
    train = None
    test = None
    sample = None
    # search recursively
    for root,dirs,files in os.walk(base_dir):
        for f in files:
            if f in candidates:
                path = os.path.join(root,f)
                if f=='train.csv': train = pd.read_csv(path)
                if f=='test.csv': test = pd.read_csv(path)
                if f=='sample_submission.csv': sample = pd.read_csv(path)
    if train is None:
        raise FileNotFoundError('train.csv not found under /kaggle/input')
    if test is None:
        raise FileNotFoundError('test.csv not found under /kaggle/input')
    if sample is None:
        sample = pd.DataFrame({'id': test['id']})
    return train, test, sample

train, test, sample = load_data()
print('train shape:', train.shape, 'test shape:', test.shape)


train.head()


test.head()


sample.head()


# train = train.drop('id', axis=1)


# train.shape


orig_list = []
pattern = '/kaggle/input/**/synthetic_road_accidents_*/*.csv'
# try pattern search
for root,dirs,files in os.walk('/kaggle/input'):
    for f in files:
        if re.search(r'synthetic_road_accidents_\d+k', f):
            p = os.path.join(root, f)
            print('found synthetic file', p)
            try:
                df = pd.read_csv(p)
                orig_list.append(df)
            except Exception:
                pass

if orig_list:
    orig = pd.concat(orig_list, axis=0, ignore_index=True)
    print('orig extra data shape:', orig.shape)

    # Ensure 'id' column exists in synthetic data
    if 'id' not in orig.columns:
        orig['id'] = np.arange(len(orig)) + train['id'].max() + 1
    # unify columns if possible

    # Now check for column compatibility
    if set(train.columns).issubset(set(orig.columns)):
        orig = orig[train.columns]
        train = pd.concat([train, orig], axis=0, ignore_index=True)
        print('train shape after augmentation:', train.shape)
    else:
        missing = set(train.columns) - set(orig.columns)
        print('Cannot merge, missing columns in synthetic data:', missing)


orig.head()


train.head()


# EDA: identify categorical vs numeric
TARGET = 'accident_risk'
cols = [c for c in train.columns if c not in ['id', TARGET]]
cat_cols = [c for c in cols if train[c].dtype == 'object' or train[c].nunique() < 50]
num_cols = [c for c in cols if c not in cat_cols]
print('num_cols', len(num_cols), 'cat_cols', len(cat_cols))


# simple feature engineering
def add_features(df):
    df = df.copy()
    # frequency encoding for categorical
    for c in cat_cols:
        freq = df[c].map(df[c].value_counts())
        df[c + '_freq'] = freq
    # basic numeric interactions
    if 'speed_limit' in df.columns and 'road_width' in df.columns:
        df['speed_x_width'] = df['speed_limit'] * df['road_width']
    # fill na
    df.fillna(-999, inplace=True)
    return df

all_df = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
all_df = add_features(all_df)


# split back
train_feats = all_df.iloc[:len(train)]
test_feats = all_df.iloc[len(train):]


# prepare arrays
X = train_feats[ [c for c in train_feats.columns if c!='id'] ]
y = train[TARGET].values
X_test = test_feats[ [c for c in test_feats.columns if c!='id'] ]


# standardize numeric columns - fit on train
scaler = StandardScaler()
# X[num_cols] = scaler.fit_transform(X[num_cols])
# X_test[num_cols] = scaler.transform(X_test[num_cols])
X.loc[:, num_cols] = scaler.fit_transform(X[num_cols])
X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])



# Cross-validation setup
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)


from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np
import pandas as pd


# 1ï¸� Identify categorical and numeric columns

cat_cols = X.select_dtypes(include=['object', 'category']).columns
num_cols = X.select_dtypes(exclude=['object', 'category']).columns


# 2ï¸� Define preprocessing for each type of column

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)


# 3ï¸� Define the model and pipeline

mlp = MLPRegressor(random_state=RANDOM_SEED)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', mlp)
])


# 4ï¸� Define hyperparameter grid for tuning

param_grid = {
    'model__hidden_layer_sizes': [(128, 64), (256, 128), (256, 128, 64)],
    'model__activation': ['relu', 'tanh'],
    'model__alpha': [1e-4, 1e-3, 1e-2],  # L2 regularization
    'model__learning_rate_init': [0.001, 0.01],
    'model__max_iter': [300, 500],
}

# Define scorer (since we want to minimize RMSE)
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)


# 5ï¸� K-Fold Cross-validation with tuning

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
oof_base = np.zeros(len(X))
preds_base = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"\nğŸ”¹ Fold {fold+1}/{N_FOLDS}")

    # Split data (use standard indexing since y may be numpy array)
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # Create GridSearchCV to tune parameters for this fold
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=rmse_scorer,
        cv=3,              # inner 3-fold CV
        n_jobs=-1,
        verbose=1
    )

    # Fit GridSearch
    grid.fit(X_tr, y_tr)
    print(f"âœ… Best params for fold {fold+1}: {grid.best_params_}")
    print(f"âœ… Best CV RMSE (inner): {-grid.best_score_:.5f}")

    # Best model from this fold
    best_model = grid.best_estimator_

    # Predict validation and test
    oof_base[va_idx] = best_model.predict(X_va)
    preds_base += best_model.predict(X_test) / N_FOLDS


# 6 Evaluate overall CV performance

base_cv = np.sqrt(mean_squared_error(y, oof_base))
print(f"\nğŸ�� Final Base CV RMSE: {base_cv:.5f}")



# 2) Train XGBoost on residuals (target - base_pred)
residuals = y - oof_base

# add base predictions as a feature
X_xgb = X.copy()
X_xgb['base_pred'] = oof_base
X_test_xgb = X_test.copy()
X_test_xgb['base_pred'] = preds_base

params = {
    'objective':'reg:squarederror',
    'learning_rate':0.05,
    'max_depth':6,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'seed':RANDOM_SEED,
    'n_estimators':10000
}

oof_xgb = np.zeros(len(X))
preds_xgb = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_xgb)):
    print('XGB fold', fold)
    dtrain = xgb.DMatrix(X_xgb.iloc[tr_idx], label=residuals[tr_idx])
    dval = xgb.DMatrix(X_xgb.iloc[va_idx], label=residuals[va_idx])
    watchlist = [(dtrain, 'train'), (dval, 'valid')]
    model = xgb.train(params, dtrain, num_boost_round=5000, evals=watchlist,
                      early_stopping_rounds=100, verbose_eval=100)
    oof_xgb[va_idx] = model.predict(xgb.DMatrix(X_xgb.iloc[va_idx]))
    preds_xgb += model.predict(xgb.DMatrix(X_test_xgb)) / N_FOLDS




