%load_ext autoreload
%autoreload 2


import pandas as pd
import numpy as np
import seaborn as sns

import src.utils as utils

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)


dtypes = {
    'id': np.int64,
    'age': np.int16,
    'alcohol_consumption_per_week': np.int64,
    'physical_activity_minutes_per_week': np.int64,
    'diet_score': np.float64,
    'sleep_hours_per_day': np.float64,
    'screen_time_hours_per_day': np.float64,
    'bmi': np.float64,
    'waist_to_hip_ratio': np.float64,
    'systolic_bp': np.int64,
    'diastolic_bp': np.int64,
    'heart_rate': np.int64,
    'cholesterol_total': np.int64,
    'hdl_cholesterol': np.int64,
    'ldl_cholesterol': np.int64,
    'triglycerides': np.int64,
    'gender': pd.CategoricalDtype(),
    'ethnicity': pd.CategoricalDtype(),
    'education_level': pd.CategoricalDtype(),
    'income_level': pd.CategoricalDtype(),
    'smoking_status': pd.CategoricalDtype(),
    'employment_status': pd.CategoricalDtype(),
    'family_history_diabetes': np.int64,
    'hypertension_history': np.int64,
    'cardiovascular_history': np.int64,
    'diagnosed_diabetes': np.float64,
}

df_train = pd.read_csv('input/train.csv', dtype=dtypes) # type: ignore
df_test = pd.read_csv('input/test.csv', dtype=dtypes) # type: ignore


df_train.sample(5, random_state=3)


# skims data for more informations

utils.skim_data(df_train)


utils.skim_data(df_test)


X = df_train.drop(columns=['diagnosed_diabetes'])
y = df_train['diagnosed_diabetes']

# separate to train and validation datasets

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=29, stratify=y)


# find the best model

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV

import category_encoders as ce

from skopt import BayesSearchCV
from skopt.space import Real, Categorical

def sgd_proba(X_train, y_train, X_valid):
    # preprocessor
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    numeric_features = [col for col in numeric_features if col != 'id']
    ohe_features = ['gender', 'smoking_status']
    target_features = ['ethnicity', 'education_level', 'employment_status']
    ordinal_feature = ['income_level']

    numeric_pipeline = Pipeline(
        steps=[('step', StandardScaler())]
    )
    ohe_pipeline = Pipeline(
        steps=[('step', OneHotEncoder(handle_unknown='ignore'))]
    )
    target_pipeline = Pipeline(
        steps=[('step', ce.TargetEncoder())]
    )
    ordinal_pipeline = Pipeline(
        steps=[
            (
                'step',
                OrdinalEncoder(
                    categories=[['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']]
                )
            )
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('ohe', ohe_pipeline, ohe_features),
            ('tar', target_pipeline, target_features),
            ('ord', ordinal_pipeline, ordinal_feature),
        ],
        remainder='drop'
    )

    # main pipeline
    pipeline = Pipeline(
        steps=[
            ('preprocess', preprocessing),
            ('select', SelectFromModel(estimator=RandomForestClassifier(random_state=29))),
            ('rbf', RBFSampler(random_state=29)),
            (
                'classifier',
                CalibratedClassifierCV(
                    estimator=SGDClassifier(random_state=29, loss='log_loss'),
                    cv=5
                )
            )
        ]
    )

    # randomized search
    param_grid = {
        'select__threshold': Categorical(['0.5*median', '0.75*median']),
        'rbf__gamma': Real(1e-3, 1e+2, 'log-uniform'),
        'classifier__estimator__alpha': Real(1e-6, 1e-1, 'log-uniform'),
        'classifier__method': Categorical(['sigmoid', 'isotonic'])
    }
    custom_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=29)
    bayes_search = BayesSearchCV(
        estimator=pipeline,
        search_spaces=param_grid,
        scoring='roc_auc',
        n_jobs=4,
        cv=custom_cv,
        random_state=29,
        n_iter=30
    )
    start_time = utils.get_time()
    print(f'Fitting RandomizedSearch at {start_time}')
    bayes_search.fit(X_train, y_train)
    print(f'Best parameters: {bayes_search.best_params_}')
    best_model = bayes_search.best_estimator_
    timestamp = utils.get_time()
    utils.save_model(best_model, f'sgd_{timestamp}.joblib')
    print('Predict on valid dataset')
    df_valid_result = bayes_search.predict_proba(X_valid)
    df_submission = pd.DataFrame(
        {
            'id': X_valid['id'],
            'diagnosed_diabetes': df_valid_result[:, 1]
        }
    )
    timestamp = utils.get_time()
    df_submission.to_csv(f'input/sgd_valid_{timestamp}.csv', index=False)

sgd_proba(X_train, y_train, X_valid)


from lightgbm import LGBMClassifier

def lgbm_proba(X_train, y_train, X_valid):
    # preprocessor
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    numeric_features = [col for col in numeric_features if col != 'id']
    ohe_features = ['gender', 'smoking_status']
    target_features = ['ethnicity', 'education_level', 'employment_status']
    ordinal_feature = ['income_level']

    numeric_pipeline = Pipeline(
        steps=[('step', StandardScaler())]
    )
    ohe_pipeline = Pipeline(
        steps=[('step', OneHotEncoder(handle_unknown='ignore'))]
    )
    target_pipeline = Pipeline(
        steps=[('step', ce.TargetEncoder())]
    )
    ordinal_pipeline = Pipeline(
        steps=[
            (
                'step',
                OrdinalEncoder(
                    categories=[['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']]
                )
            )
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('ohe', ohe_pipeline, ohe_features),
            ('tar', target_pipeline, target_features),
            ('ord', ordinal_pipeline, ordinal_feature),
        ],
        remainder='drop'
    )

    # main pipeline
    pipeline = Pipeline(
        steps=[
            ('preprocess', preprocessing),
            ('classifier', LGBMClassifier(random_state=29))
        ]
    )

    # randomized search
    param_grid = {
        'classifier__n_estimators': [100, 200, 500],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__num_leaves': [20, 31, 40],
        'classifier__reg_alpha': [0.1, 0.5, 1.0],
        'classifier__reg_lambda': [0.1, 0.5, 1.0]
    }
    custom_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=29)
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        scoring='roc_auc',
        n_jobs=5,
        cv=custom_cv,
        random_state=29,
    )
    start_time = utils.get_time()
    print(f'Fitting RandomizedSearch at {start_time}')
    random_search.fit(X_train, y_train)
    print(f'Best parameters: {random_search.best_params_}')
    best_model = random_search.best_estimator_
    timestamp = utils.get_time()
    utils.save_model(best_model, f'lgbm_{timestamp}.joblib')
    print('Predict on valid dataset')
    df_valid_result = random_search.predict_proba(X_valid)
    df_submission = pd.DataFrame(
        {
            'id': X_valid['id'],
            'diagnosed_diabetes': df_valid_result[:, 1]
        }
    )
    timestamp = utils.get_time()
    df_submission.to_csv(f'input/lgbm_valid_{timestamp}.csv', index=False)

lgbm_proba(X_train, y_train, X_valid)


from sklearn.metrics import roc_auc_score

valid_preds_sgd = pd.read_csv('input/sgd_valid_2025_12_04_00_30_47.csv')['diagnosed_diabetes']
valid_preds_lgbm = pd.read_csv('input/lgbm_valid_2025_12_03_20_29_00.csv')['diagnosed_diabetes']
best_score = 0
best_weight = 0

for weight in np.arange(0.0, 1.01, 0.01):
    blended_score = (weight * valid_preds_sgd) + ((1 - weight) * valid_preds_lgbm)
    score = roc_auc_score(y_valid, blended_score)

    if score > best_score:
        best_score = score
        best_weight = weight

print(f"Local validation score for SGD only: {roc_auc_score(y_valid, valid_preds_sgd):.5f}")
print(f"Local validation score for LGBM only: {roc_auc_score(y_valid, valid_preds_lgbm):.5f}")
print("-" * 20)
print(f"Best blending weight for SGD model: {best_weight:.2f}")
print(f"Best blending weight for LGBM model: {1 - best_weight:.2f}")
print(f"Best blended ROC AUC score on holdout set: {best_score:.5f}")
print("-" * 50)


sgd_model = utils.load_model('models/sgd_2025_12_03_20_09_37.joblib')
lgbm_model = utils.load_model('models/lgbm_2025_12_03_20_28_58.joblib')

df_test_sgd_result = None
df_test_lgbm_result = None

if sgd_model is not None:
    print('Predict on test dataset using SGD')
    df_test_sgd_result = sgd_model.predict_proba(df_test)

if lgbm_model is not None:
    print('Predict on test dataset using LGBM')
    df_test_lgbm_result = lgbm_model.predict_proba(df_test)

final_blended_preds = (best_weight * df_test_sgd_result) + ((1 - best_weight) * df_test_lgbm_result)
df_submission = pd.DataFrame(
    {
        'id': df_test['id'],
        'diagnosed_diabetes': final_blended_preds[:, 1]
    }
)
timestamp = utils.get_time()
df_submission.to_csv(f'input/submission_blended_{timestamp}.csv', index=False)


from skopt.space import Integer
from catboost import CatBoostClassifier

def catboost_proba(X_train, y_train, X_test):
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    numeric_features = [col for col in numeric_features if col != 'id']
    ohe_features = ['gender', 'smoking_status']
    target_features = ['ethnicity', 'education_level', 'employment_status']
    ordinal_feature = ['income_level']
    categorical_features = ohe_features + target_features + ordinal_feature

    search_spaces = {
        'iterations': Integer(50, 200),
        'learning_rate': Real(1e-3, 0.1, 'log-uniform'),
        'depth': Integer(4, 10),
        'l2_leaf_reg': Real(1e-2, 1e1, 'log-uniform'),
        'border_count': Integer(32, 128),
        'bootstrap_type': Categorical(['Bayesian', 'MVS']),
        'random_strength': Real(1e-2, 1e0, 'log-uniform'),
    }

    cb_model = CatBoostClassifier(random_state=23,
                                  logging_level='Silent',
                                  cat_features=categorical_features)
    opt = BayesSearchCV(
        estimator=cb_model,
        search_spaces=search_spaces,
        n_iter=50,
        cv=3,
        scoring='roc_auc',
        verbose=0,
        n_jobs=4,
        random_state=29
    )

    start_time = utils.get_time()
    print(f'Fitting BayesSearch at {start_time}')
    opt.fit(X_train, y_train)
    print(f'Best parameters: {opt.best_params_}')
    best_model = opt.best_estimator_
    end_time = utils.get_time()
    utils.save_model(best_model, f'cbt_{end_time}.joblib')
    print('Predict on test dataset')
    df_test_result = opt.predict_proba(X_test)
    df_submission = pd.DataFrame(
        {
            'id': X_test['id'],
            'diagnosed_diabetes': df_test_result[:, 1]
        }
    )
    df_submission.to_csv(f'input/cbt_test_{end_time}.csv', index=False)

catboost_proba(X_train, y_train, df_test)


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def stacking_proba(X_train, y_train, X_test):
    # preprocessor
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    numeric_features = [col for col in numeric_features if col != 'id']
    ohe_features = ['gender', 'smoking_status']
    target_features = ['ethnicity', 'education_level', 'employment_status']
    ordinal_feature = ['income_level']
    numeric_pipeline = Pipeline(
        steps=[('step', StandardScaler())]
    )
    ohe_pipeline = Pipeline(
        steps=[('step', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]
    )
    target_pipeline = Pipeline(
        steps=[('step', ce.TargetEncoder())]
    )
    ordinal_pipeline = Pipeline(
        steps=[
            (
                'step',
                OrdinalEncoder(
                    categories=[['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']]
                )
            )
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('ohe', ohe_pipeline, ohe_features),
            ('tar', target_pipeline, target_features),
            ('ord', ordinal_pipeline, ordinal_feature),
        ],
        remainder='drop'
    )

    # sgd pipeline
    sgd_pipeline = Pipeline(
        steps=[
            ('select', SelectFromModel(estimator=RandomForestClassifier(random_state=29), threshold='0.5*median')),
            ('rbf', RBFSampler(random_state=29, gamma=0.001)),
            (
                'classifier',
                SGDClassifier(random_state=29, loss='log_loss', alpha=1e-06)
            )
        ]
    )

    # lgbm pipeline
    lgbm_pipeline = LGBMClassifier(reg_lambda=1.0, reg_alpha=0.1, num_leaves=20, n_estimators=500, learning_rate=0.05, random_state=29)

    # catboost pipeline
    cbt_pipeline = CatBoostClassifier(random_state=23,
                                      logging_level='Silent',
                                      bootstrap_type='MVS',
                                      border_count=128,
                                      depth=9,
                                      iterations=200,
                                      l2_leaf_reg=10.0,
                                      learning_rate=0.1,
                                      random_strength=0.01)

    # main pipeline
    estimator_list = [
        ('lgbm', lgbm_pipeline),
        ('cbt', cbt_pipeline)
    ]
    meta_model = LogisticRegression(C=1.331359140628303, random_state=29)
    stacking_model = StackingClassifier(
        estimators=estimator_list,
        final_estimator=meta_model,
        cv=5,
        passthrough=True,
        n_jobs=4
    )
    pipeline = Pipeline(
        steps=[
            ('preprocess', preprocessing),
            ('classifier', stacking_model)
        ]
    )

    start_time = utils.get_time()
    print(f'Fitting Stacking Model at {start_time}')
    pipeline.fit(X_train, y_train)
    stacking_test_result = pipeline.predict_proba(X_test)
    df_submission = pd.DataFrame(
        {
            'id': X_test['id'],
            'diagnosed_diabetes': stacking_test_result[:, 1]
        }
    )
    timestamp = utils.get_time()
    df_submission.to_csv(f'input/submission_stacking_{timestamp}.csv', index=False)

X_train_whole = df_train.drop(columns=['diagnosed_diabetes'])
y_train_whole = df_train['diagnosed_diabetes']
stacking_proba(X_train_whole, y_train_whole, df_test)


utils.skim_data(df_train)


def transform_features(df):
    # creating families of features
    lifestyle_A = ['alcohol_consumption_per_week', 'physical_activity_minutes_per_week']
    lifestyle_B = ['diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day']
    heart_related = ['systolic_bp', 'diastolic_bp', 'heart_rate']
    cholesterol_level = ['hdl_cholesterol', 'ldl_cholesterol']

    df_copy = df.copy()

    # combinatorial feature generation (cfg)
    cfg_features = [lifestyle_A, lifestyle_B, heart_related, cholesterol_level]

    for feature in cfg_features:
        for i, a in enumerate(feature):
            for j, b in enumerate(feature):
                if i > j:
                    df_copy[f'{a}_{b}_imb'] = df.eval(f'({a}-{b})/({a}+{b})')

    # asymmetry features
    af_features = [lifestyle_B, heart_related]

    for feature in af_features:
        for i, a in enumerate(feature):
            for j, b in enumerate(feature):
                for k, c in enumerate(feature):
                    if i > j and j > k:
                        max_ = df_copy[[a, b, c]].max(axis=1)
                        min_ = df_copy[[a, b, c]].min(axis=1)
                        mid_ = df_copy[[a, b, c]].sum(axis=1)-min_-max_
                        df_copy[f'{a}_{b}_{c}_imb2'] = (max_-mid_)/(mid_-min_)

    return df_copy

transform_features(df_train.sample(3))


def catboost_proba2(X_train, y_train, X_test):
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    numeric_features = [col for col in numeric_features if col != 'id']
    ohe_features = ['gender', 'smoking_status']
    target_features = ['ethnicity', 'education_level', 'employment_status']
    ordinal_feature = ['income_level']
    categorical_features = ohe_features + target_features + ordinal_feature

    X_train_engineered = transform_features(X_train)
    X_test_engineered = transform_features(X_test)

    search_spaces = {
        'iterations': Integer(50, 200),
        'learning_rate': Real(1e-3, 0.1, 'log-uniform'),
        'depth': Integer(4, 10),
        'l2_leaf_reg': Real(1e-2, 1e1, 'log-uniform'),
        'border_count': Integer(32, 128),
        'bootstrap_type': Categorical(['Bayesian', 'MVS']),
        'random_strength': Real(1e-2, 1e0, 'log-uniform'),
    }

    cb_model = CatBoostClassifier(random_state=23,
                                  logging_level='Silent',
                                  cat_features=categorical_features)
    opt = BayesSearchCV(
        estimator=cb_model,
        search_spaces=search_spaces,
        n_iter=50,
        cv=3,
        scoring='roc_auc',
        verbose=0,
        n_jobs=4,
        random_state=29
    )

    start_time = utils.get_time()
    print(f'Fitting BayesSearch at {start_time}')
    opt.fit(X_train_engineered, y_train)
    print(f'Best parameters: {opt.best_params_}')
    best_model = opt.best_estimator_
    end_time = utils.get_time()
    utils.save_model(best_model, f'cbt_{end_time}.joblib')
    print('Predict on test dataset')
    df_test_result = opt.predict_proba(X_test_engineered)
    df_submission = pd.DataFrame(
        {
            'id': X_test['id'],
            'diagnosed_diabetes': df_test_result[:, 1]
        }
    )
    df_submission.to_csv(f'input/cbt_test_{end_time}.csv', index=False)

catboost_proba(X_train, y_train, df_test)

