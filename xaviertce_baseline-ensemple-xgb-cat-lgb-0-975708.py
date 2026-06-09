import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import catboost as cb


train = pd.read_csv(r'/kaggle/input/playground-series-s5e7/train.csv')


test = pd.read_csv(r'/kaggle/input/playground-series-s5e7/test.csv')


train.head()


train['Personality'] = train['Personality'].map({'Extrovert':0,'Introvert':1})


train.info()


FEATURES = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']


cat_col = ['Stage_fear', 'Drained_after_socializing']
for col in cat_col:
    train[col] = train[col].fillna('Missing').astype('category')
    test[col] = test[col].fillna('Missing').astype('category')


def NE_FE(df):
    df['social_activity_score'] = (
        0.5 * df['Social_event_attendance'].fillna(0) +
        0.3 * df['Going_outside'].fillna(0) +
        0.2 * df['Post_frequency'].fillna(0)
    )
    df['friend_post_density'] = df['Post_frequency'] / (df['Friends_circle_size'] + 1e-5)
    df['introversion_index'] = df['Time_spent_Alone'] / (df['Social_event_attendance'] + 1)
    df['interactiveness'] = (
        df['Friends_circle_size'].fillna(0) +
        df['Social_event_attendance'].fillna(0) +
        df['Post_frequency'].fillna(0)
    )
    df["friend_density_squared"] = df["friend_post_density"]**2
    
    return df


train = NE_FE(train)
test = NE_FE(test)





%%time
FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=1)
val_scores_xgb = []
train_scores_xgb = []
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))
for i, (train_index, test_index) in enumerate(kf.split(train,train['Personality'].values)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"Personality"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"Personality"]
    x_test = test[FEATURES].copy()

    xgb_params = {
                  'n_estimators': 1000, 
                  'eta': 0.017579962549289938, 
                  'alpha': 2.321837605269201, 
                  'subsample': 0.7882697285800268,
                  'colsample_bytree': 0.9490715823210952, 
                  'max_depth': 12, 
                  'min_child_weight': 6, 
                  'gamma': 1.1312486129566937, 
                  'max_bin': 78406,
                  'device': 'cpu',
                  'eval_metric': 'auc',
                  'random_state' : 42,
                  'enable_categorical':True
                 }
    model_xgb = XGBClassifier(**xgb_params)
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500 
    )

    # INFER OOF
    val_preds_proba = model_xgb.predict_proba(x_valid)[:, 1]
    # INFER TEST
    train_preds_proba = model_xgb.predict_proba(x_train)[:, 1]
    pred_xgb += model_xgb.predict_proba(test[FEATURES])[:, 1]

    val_scores_xgb.append(roc_auc_score(y_valid, val_preds_proba))
    train_scores_xgb.append(roc_auc_score(y_train, train_preds_proba))
    print(f'Fold {i}: train_scores - {train_scores_xgb[-1]:.5f} val_scores - {val_scores_xgb[-1]:.5f}')

pred_xgb /= FOLDS








##LGB


%%time
FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=1)
val_scores_lgb = []
train_scores_lgb = []
oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))
for i, (train_index, test_index) in enumerate(kf.split(train,train['Personality'].values)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"Personality"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"Personality"]
    x_test = test[FEATURES].copy()

    lgb_params = {
        'objective': 'binary',
        'metric': 'logloss',
        'boosting_type': 'gbdt',
        'device': 'cpu',
        'n_estimators': 1700, # High number, use early stopping
        'learning_rate': 0.01,
        'num_leaves': 31, # Adjust based on data complexity
        'max_depth': 5, # No limit
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(x_train, y_train,
              eval_set=[(x_valid, y_valid)],
              eval_metric='logloss',
              callbacks=[lgb.early_stopping(100, verbose=False)], # Stop if validation logloss doesn't improve for 100 rounds
              categorical_feature=cat_col) # Pass categorical feature nam

    # INFER OOF
    val_preds_proba = model_lgb.predict_proba(x_valid)[:, 1]
    # INFER TEST
    train_preds_proba = model_lgb.predict_proba(x_train)[:, 1]
    pred_lgb += model_lgb.predict_proba(test[FEATURES])[:, 1]

    val_scores_lgb.append(roc_auc_score(y_valid, val_preds_proba))
    train_scores_lgb.append(roc_auc_score(y_train, train_preds_proba))
    print(f'Fold {i}: train_scores - {train_scores_lgb[-1]:.5f} val_scores - {val_scores_lgb[-1]:.5f}')

pred_lgb /= FOLDS











%%time
FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=1)
val_scores_cat = []
train_scores_cat = []
oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))
for i, (train_index, test_index) in enumerate(kf.split(train,train['Personality'].values)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"Personality"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"Personality"]
    x_test = test[FEATURES].copy()

    model_cat = cb.CatBoostClassifier(
            iterations=1700, # High number, use early stopping
            learning_rate=0.02,
            loss_function='Logloss',
            eval_metric='Logloss',
            task_type='CPU',
            depth=7, # Adjust as needed
            l2_leaf_reg=3, # Regularization
            random_seed=42,
            verbose=0, # Suppress verbose output during training
            early_stopping_rounds=100,
            cat_features=[x_train.columns.get_loc(col) for col in cat_col if col in x_train.columns], # Pass indices or names
            # task_type="GPU", # Uncomment if you have a suitable GPU and installed CatBoost with GPU support
        )
    
    model_cat.fit(x_train, y_train,
                  eval_set=[(x_valid, y_valid)],
              use_best_model=True)

    # INFER OOF
    val_preds_proba = model_cat.predict_proba(x_valid)[:, 1]
    # INFER TEST
    train_preds_proba = model_cat.predict_proba(x_train)[:, 1]
    pred_cat += model_cat.predict_proba(test[FEATURES])[:, 1]

    val_scores_cat.append(roc_auc_score(y_valid, val_preds_proba))
    train_scores_cat.append(roc_auc_score(y_train, train_preds_proba))
    print(f'Fold {i}: train_scores - {train_scores_cat[-1]:.5f} val_scores - {val_scores_cat[-1]:.5f}')

pred_cat /= FOLDS





data_dict = {
    'XGB_VAL_score': val_scores_xgb,
    'LGB_VAL_score': val_scores_lgb,
    'CAT_VAL_score': val_scores_cat,
    'XGB_TRAIN_score': train_scores_xgb,
    'LGB_TRAIN_score': train_scores_lgb,
    'CAT_TRAIN_score': train_scores_cat,    
    
}
df = pd.DataFrame(data_dict)
df


ensample_pred = (pred_xgb + pred_lgb + pred_cat) / 3


sub = pd.read_csv(r'/kaggle/input/playground-series-s5e7/sample_submission.csv')


sub['Personality'] = np.where(ensample_pred >= 0.5, 1, 0)


sub['Personality'] = sub['Personality'].map({0:'Extrovert',1:'Introvert'})


sub.to_csv(r'submission.csv',index=False)







