import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization
from sklearn.feature_selection import RFE

# Load datasets
app_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
app_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
credit_card = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
installments = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')

# ğŸ”� Debug: Check if datasets are loaded correctly
print("\nâœ… Checking Initial Data")
print("Shape of app_train:", app_train.shape)
print("Shape of app_test:", app_test.shape)

# Ensure TARGET column exists
if 'TARGET' not in app_train.columns:
    raise ValueError("â�Œ ERROR: 'TARGET' column is missing from app_train!")

# Label encode categorical features
for df in [app_train, app_test]:
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# FEATURE ENGINEERING - Aggregate statistics from bureau and credit data
def aggregate_features(df, group_var, agg_funcs):
    """Helper function to aggregate numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_agg = df.groupby(group_var)[numeric_cols].agg(agg_funcs)
    df_agg.columns = ["_".join(x) for x in df_agg.columns.ravel()]
    return df_agg.reset_index()

# Aggregate statistics from bureau, credit card, and installments
bureau_agg = aggregate_features(bureau, 'SK_ID_CURR', ['mean', 'sum', 'max', 'min'])
credit_card_agg = aggregate_features(credit_card, 'SK_ID_CURR', ['mean', 'sum', 'max', 'min'])
installments_agg = aggregate_features(installments, 'SK_ID_CURR', ['mean', 'sum', 'max', 'min'])

# Merge aggregated features into app_train and app_test
app_train = app_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
app_test = app_test.merge(bureau_agg, on='SK_ID_CURR', how='left')
app_train = app_train.merge(credit_card_agg, on='SK_ID_CURR', how='left')
app_test = app_test.merge(credit_card_agg, on='SK_ID_CURR', how='left')
app_train = app_train.merge(installments_agg, on='SK_ID_CURR', how='left')
app_test = app_test.merge(installments_agg, on='SK_ID_CURR', how='left')

# Define Features and Target
X = app_train.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
y = app_train['TARGET']
X_test = app_test.drop(columns=['SK_ID_CURR'], errors='ignore')

# Drop columns where all values are NaN
X.dropna(axis=1, how="all", inplace=True)

# Fill remaining missing values with median
X.fillna(X.median(), inplace=True)
X_test.fillna(X_test.median(), inplace=True)

# FEATURE SELECTION: Recursive Feature Elimination (RFE)
lgb_clf = lgb.LGBMClassifier(n_estimators=100)
selector = RFE(lgb_clf, n_features_to_select=150, step=10)
selector.fit(X, y)
X = X.loc[:, selector.support_]
X_test = X_test.loc[:, selector.support_]

# Splitting dataset for training and evaluation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# BAYESIAN OPTIMIZATION: LightGBM Hyperparameter Tuning
def lgb_eval(num_leaves, max_depth, feature_fraction, bagging_fraction, lambda_l1, lambda_l2):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': int(num_leaves),
        'max_depth': int(max_depth),
        'feature_fraction': max(min(feature_fraction, 1), 0),
        'bagging_fraction': max(min(bagging_fraction, 1), 0),
        'lambda_l1': max(lambda_l1, 0),
        'lambda_l2': max(lambda_l2, 0),
        'verbosity': -1
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    
    for train_idx, valid_idx in skf.split(X_train, y_train):
        train_data = lgb.Dataset(X_train.iloc[train_idx], label=y_train.iloc[train_idx])
        valid_data = lgb.Dataset(X_train.iloc[valid_idx], label=y_train.iloc[valid_idx])
        
        model = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=1000, callbacks=[lgb.early_stopping(100)])
        preds = model.predict(X_train.iloc[valid_idx])
        aucs.append(roc_auc_score(y_train.iloc[valid_idx], preds))
    
    return np.mean(aucs)

optimizer = BayesianOptimization(
    f=lgb_eval,
    pbounds={'num_leaves': (20, 100),
             'max_depth': (5, 15),
             'feature_fraction': (0.5, 1.0),
             'bagging_fraction': (0.5, 1.0),
             'lambda_l1': (0, 10),
             'lambda_l2': (0, 10)},
    random_state=42
)
optimizer.maximize(init_points=5, n_iter=15)

# Extract best hyperparameters
best_params = optimizer.max['params']
best_params['num_leaves'] = int(best_params['num_leaves'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params.update({'objective': 'binary', 'metric': 'auc', 'verbosity': -1})

# Train LightGBM Model with optimized hyperparameters
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
model = lgb.train(best_params, train_data, valid_sets=[valid_data], num_boost_round=1000, callbacks=[lgb.early_stopping(100)])

# Predict and Evaluate Model
y_pred = model.predict(X_valid)
auc_score = roc_auc_score(y_valid, y_pred)
print(f'\nğŸ�¯ Final Model AUC: {auc_score:.4f}')

# Create Submission File
test_preds = model.predict(X_test)
submission = pd.DataFrame({'SK_ID_CURR': app_test['SK_ID_CURR'], 'TARGET': test_preds})
submission.to_csv('submission.csv', index=False)


