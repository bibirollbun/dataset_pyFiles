# IMPORTS AND SETUP
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# LOAD DATA
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

new_train_df = train_df.copy()
new_test_df = test_df.copy()

print(f'Train shape: {train_df.shape}, Test shape: {test_df.shape}')



# ENCODING
personality_encoding = {'Extrovert': 0, 'Introvert': 1}
drained_encoding = {'No': 0, 'Yes': 1}
stage_encoding = {'No': 0, 'Yes': 1}

new_train_df['Drained_after_socializing_encoded'] = new_train_df['Drained_after_socializing'].map(drained_encoding)
new_test_df['Drained_after_socializing_encoded'] = new_test_df['Drained_after_socializing'].map(drained_encoding)

new_train_df['Stage_fear_encoded'] = new_train_df['Stage_fear'].map(stage_encoding)
new_test_df['Stage_fear_encoded'] = new_test_df['Stage_fear'].map(stage_encoding)

new_train_df['Personality_encoded'] = new_train_df['Personality'].map(personality_encoding)

# SIMPLE RATIO FEATURES
new_train_df['social_ratio'] = new_train_df['Social_event_attendance'] / (new_train_df['Time_spent_Alone'] + 1)
new_test_df['social_ratio'] = new_test_df['Social_event_attendance'] / (new_test_df['Time_spent_Alone'] + 1)

new_train_df['friend_post_ratio'] = new_train_df['Friends_circle_size'] / (new_train_df['Post_frequency'] + 1)
new_test_df['friend_post_ratio'] = new_test_df['Friends_circle_size'] / (new_test_df['Post_frequency'] + 1)



# FEATURE COMBINATIONS
def feature_combination(train, test, cols_to_combine, pair_size):
    for pair in pair_size:
        for cols in combinations(cols_to_combine, pair):
            new_col_name = '_'.join(cols)
            train[new_col_name] = train[cols[0]].astype(str)
            for col in cols[1:]:
                train[new_col_name] = train[new_col_name] + '_' + train[col].astype(str)
            
            test[new_col_name] = test[cols[0]].astype(str)
            for col in cols[1:]:
                test[new_col_name] = test[new_col_name] + '_' + test[col].astype(str)
    return train, test

cols_to_combine = train_df.columns.drop(labels=['id', 'Personality'])
new_train_df, new_test_df = feature_combination(new_train_df, new_test_df, cols_to_combine, pair_size=[2])

print(f'Train shape after combinations: {new_train_df.shape}')



# TARGET ENCODING
def target_encoding(train, test, cols_to_encode, target, cv, agg=['mean']):
    for col in cols_to_encode:
        for train_idx, valid_idx in cv.split(train, train[target]):
            X_tr = train.iloc[train_idx]
            X_val = train.iloc[valid_idx]
            
            for stat in agg:
                stat_result = X_tr.groupby(col)[target].agg(stat)
                name_new_cols = f'{col}_{stat}_te'
                train.loc[valid_idx, name_new_cols] = X_val[col].map(stat_result)
        
        for stat in agg:
            global_stat = train.groupby(col)[target].agg(stat)
            name_new_cols = f'{col}_{stat}_te'
            test[name_new_cols] = test[col].map(global_stat)
    
    train = train.drop(columns=cols_to_encode)
    test = test.drop(columns=cols_to_encode)
    return train, test

cat_cols = new_train_df.select_dtypes(include='object').drop(labels=['Stage_fear', 'Drained_after_socializing', 'Personality'], axis=1).columns
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

new_train_df, new_test_df = target_encoding(new_train_df, new_test_df, 
                                           cols_to_encode=cat_cols, 
                                           target='Personality_encoded', cv=skf, 
                                           agg=['mean'])



# AGGREGATION FEATURES
def aggregation(data, agg_list, cols_used, name_new_cols=''):
    for agg in agg_list:
        new_cols = f'{name_new_cols}{agg}'
        data.loc[:, new_cols] = data[cols_used].agg(agg, axis=1)
    return data

# Global aggregation
agg_list = ['mean', 'std', 'min', 'max', 'sum']
cols_used = new_test_df.drop(columns=['id', 'Stage_fear', 'Drained_after_socializing']).columns

new_train_df = aggregation(new_train_df, agg_list, cols_used)
new_test_df = aggregation(new_test_df, agg_list, cols_used)

# Base features aggregation
base_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
             'Friends_circle_size', 'Post_frequency', 'Drained_after_socializing_encoded', 'Stage_fear_encoded']

new_train_df = aggregation(new_train_df, agg_list, base_cols, name_new_cols='base_')
new_test_df = aggregation(new_test_df, agg_list, base_cols, name_new_cols='base_')

print(f'Final train shape: {new_train_df.shape}')



from sklearn.impute import SimpleImputer

# PREPARE DATA - Check which columns exist before dropping
x = new_train_df.drop(columns=['id', 'Personality', 'Personality_encoded', 'Drained_after_socializing', 'Stage_fear'])
y = new_train_df['Personality_encoded']

# Check which columns exist in test_df before dropping
cols_to_drop = ['id', 'Drained_after_socializing', 'Stage_fear']
existing_cols_to_drop = [col for col in cols_to_drop if col in new_test_df.columns]

# Drop only existing columns
if existing_cols_to_drop:
    new_test_df = new_test_df.drop(columns=existing_cols_to_drop)
    print(f"Dropped columns: {existing_cols_to_drop}")

# HANDLE MISSING VALUES
print(f"Missing values in train: {x.isnull().sum().sum()}")
print(f"Missing values in test: {new_test_df.isnull().sum().sum()}")

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
x_imputed = pd.DataFrame(imputer.fit_transform(x), columns=x.columns, index=x.index)
test_imputed = pd.DataFrame(imputer.transform(new_test_df), columns=new_test_df.columns, index=new_test_df.index)

print(f"After imputation - Missing values in train: {x_imputed.isnull().sum().sum()}")
print(f"After imputation - Missing values in test: {test_imputed.isnull().sum().sum()}")

# FEATURE SELECTION
selector = SelectKBest(f_classif, k=min(50, x_imputed.shape[1]))
x_selected = selector.fit_transform(x_imputed, y)
test_selected = selector.transform(test_imputed)

print(f'Selected {x_selected.shape[1]} features from {x_imputed.shape[1]}')



# IMPROVED XGBOOST PARAMETERS
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.03,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 2025,
    'verbosity': 0
}

# LIGHTGBM PARAMETERS
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'random_state': 2025,
    'verbosity': -1
}



# ENSEMBLE TRAINING
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(x_selected))
oof_lgb = np.zeros(len(x_selected))
test_xgb = []
test_lgb = []

for i, (train_index, val_index) in enumerate(skfold.split(x_selected, y)):
    x_train, x_val = x_selected[train_index], x_selected[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # XGBoost
    dtrain = xgb.DMatrix(x_train, y_train)
    dval = xgb.DMatrix(x_val, y_val)
    dtest = xgb.DMatrix(test_selected)
    
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=1000,
                         evals=[(dval, 'val')], early_stopping_rounds=50,
                         verbose_eval=False)
    
    oof_xgb[val_index] = xgb_model.predict(dval)
    test_xgb.append(xgb_model.predict(dtest))
    
    # LightGBM
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_val = lgb.Dataset(x_val, y_val)
    
    lgb_model = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_val],
                         num_boost_round=1000, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    oof_lgb[val_index] = lgb_model.predict(x_val)
    test_lgb.append(lgb_model.predict(test_selected))
    
    # Ensemble prediction
    ensemble_val = 0.6 * oof_xgb[val_index] + 0.4 * oof_lgb[val_index]
    val_acc = accuracy_score(y_val, (ensemble_val > 0.5).astype(int))
    
    print(f'Fold {i+1} - Ensemble Val Accuracy: {val_acc:.4f}')

# Final ensemble
oof_ensemble = 0.6 * oof_xgb + 0.4 * oof_lgb
test_ensemble = 0.6 * np.mean(test_xgb, axis=0) + 0.4 * np.mean(test_lgb, axis=0)

print(f'Overall Ensemble CV Accuracy: {accuracy_score(y, (oof_ensemble > 0.5).astype(int)):.4f}')



# THRESHOLD OPTIMIZATION
thresholds = np.arange(0.3, 0.7, 0.01)
best_acc = 0
best_threshold = 0.5

for thresh in thresholds:
    acc = accuracy_score(y, (oof_ensemble > thresh).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_threshold = thresh

print(f'Best threshold: {best_threshold:.3f}, Accuracy: {best_acc:.4f}')

# SUBMISSION
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission['Personality'] = (test_ensemble > best_threshold).astype(int)
submission['Personality'] = submission['Personality'].map({0: 'Extrovert', 1: 'Introvert'})

submission.to_csv('submission.csv', index=False)
print('Submission saved!')

# FEATURE IMPORTANCE
feature_names = [f'feature_{i}' for i in range(x_selected.shape[1])]
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_model.get_score(importance_type='gain').values()
}).sort_values('importance', ascending=False)

print(importance_df.head(10))





