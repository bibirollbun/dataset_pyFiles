import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs

from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, TargetEncoder


TRAIN = "https://raw.githubusercontent.com/evgpat/edu_stepik_practical_ml/main/datasets/chocolate_train.csv"
TEST = "https://raw.githubusercontent.com/evgpat/edu_stepik_practical_ml/main/datasets/chocolate_test_new.csv"


train_df = pd.read_csv(TRAIN)


train_df.head()


train_df.describe()


train_df.describe(include='object')


train_df[train_df['Broad Bean Origin'].isna()]


train_df[train_df['Specific Bean Origin'] == 'Madagascar']


train_df.fillna({'Broad Bean Origin':'Madagascar'}, inplace=True)


train_df['Bean Type'].value_counts().iloc[:3]


train_df.fillna({'Bean Type': 'Unknown'}, inplace=True)
train_df['Bean Type'] = train_df['Bean Type'].map(lambda x: 'Unknown' if(x.isspace()) else x)


train_df.describe(include='all').iloc[:4]


train_df['Bean Type'].value_counts()


train_df['Trinitario'] = train_df['Bean Type'].str.contains('Trinitario').astype(int)
train_df['Criollo'] = train_df['Bean Type'].str.contains('Criollo').astype(int)
train_df['Forastero'] = train_df['Bean Type'].str.contains('Forastero').astype(int)
train_df['Nacional'] = (train_df['Bean Type'].str.contains('Nacional|Arriba')).astype(int)
# train_df['Other'] = (~train_df['Bean Type'].str.contains('Trinitario|Criollo|Forastero|Nacional|Arriba')).astype(int)
train_df.drop('Bean Type', axis=1, inplace=True)



train_df.head()


train_df['Cocoa Percent'].value_counts()


train_df['Cocoa Percent'] = train_df['Cocoa Percent'].str.rstrip("%").astype(float)


%pip install association-metrics -q


train_df['Rating_cat'] = train_df['Rating'].astype(str)


train_df.describe(include='object')


import association_metrics as am

XC = train_df.select_dtypes('object').apply(lambda x: x.astype('category'))

cramersv = am.CramersV(XC)
sbs.heatmap(cramersv.fit(), annot=True)


train_df.describe(include='object')


train_df.drop(['Company Location', 'Broad Bean Origin', 'Rating_cat'], axis=1, inplace=True)


train_df.head()


num_features = train_df.describe().columns
corr = train_df[num_features].corr()
sbs.heatmap(corr, annot=True)



train_df.drop('Review', axis=1, inplace=True)


train_df.head()


def transform_bean_type(X):
    X = X.copy()
    X['Trinitario'] = X['Bean Type'].str.contains('Trinitario').astype(int)
    X['Criollo'] = X['Bean Type'].str.contains('Criollo').astype(int)
    X['Forastero'] = X['Bean Type'].str.contains('Forastero').astype(int)
    X['Nacional'] = X['Bean Type'].str.contains('Nacional|Arriba').astype(int)

    return X.drop(columns=['Bean Type'])


def transform_cacao_percent(X):
    X = X.copy()
    X['Cocoa Percent'] = X['Cocoa Percent'].str.rstrip("%").astype(float)
    return X


def drop_useless_features(X):
    return X.drop(columns=['Company Location', 'Broad Bean Origin', 'Review'])


from sklearn.pipeline import FunctionTransformer

def process_features(X):
    X = transform_bean_type(X)
    X = transform_cacao_percent(X)
    X = drop_useless_features(X)
    return X

feature_processor = FunctionTransformer(process_features)


train_df = pd.read_csv(TRAIN)

train_df.fillna({'Broad Bean Origin':'Madagascar'}, inplace=True)
train_df.fillna({'Bean Type': 'Unknown'}, inplace=True)
train_df['Bean Type'] = train_df['Bean Type'].map(lambda x: 'Unknown' if(x.isspace()) else x)


X_train = train_df.drop('Rating', axis=1)
y_train = train_df['Rating'].copy()


processed_df = process_features(X_train)
cat_features = processed_df.select_dtypes(include=['object']).columns.tolist()
num_features = processed_df.select_dtypes(include=['float', 'int']).columns.tolist()
all_features = cat_features + num_features


def objective_cat_boost(trial):

    pipeline_cb = Pipeline([
        ('feature_processor', feature_processor),
        ('model', CatBoostRegressor(cat_features=cat_features))
    ])

    params_cb = {
        "model__iterations": trial.suggest_int("iterations", 500, 1800),
        "model__depth": trial.suggest_int("depth", 4, 10),
        "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "model__l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20),
        "model__bagging_temperature": trial.suggest_float("bagging_temperature", 0, 2)
    }

    pipeline_cb.set_params(**params_cb)

    score = cross_val_score(
        estimator=pipeline_cb,
        X=X_train,
        y=y_train,
        cv=4,
        scoring='r2',
        n_jobs=-1
    ).mean()

    return score


pruner_cb = optuna.pruners.MedianPruner(n_warmup_steps=20)
study_cb = optuna.create_study(direction="maximize", pruner=pruner_cb)
study_cb.optimize(objective_cat_boost, n_trials=100)


study_cb.best_params, study_cb.best_value


def objective_linear_regression(trial):

    encoder = TargetEncoder(
        target_type='continuous',
        smooth=trial.suggest_float("smooth", 0.3, 15)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', MinMaxScaler(), num_features),
            ('catigorial', encoder, cat_features)
        ]
    )

    pipeline_lr_1 = Pipeline([
        ('feature_processor', feature_processor),
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    score = cross_val_score(
        estimator=pipeline_lr_1,
        X=X_train,
        y=y_train,
        cv=4,
        scoring='r2',
        n_jobs=-1
    ).mean()

    return score


study_lr = optuna.create_study(direction="maximize")
study_lr.optimize(objective_linear_regression, n_trials=200)


study_lr.best_params, study_lr.best_value


from sklearn.ensemble import RandomForestRegressor


def objective_random_forest(trial):

    pipeline_rf = Pipeline([
        ('feature_processor', feature_processor),
        ('encoder', TargetEncoder(target_type='continuous')),
        ('model', RandomForestRegressor())
    ])

    params_rf = {
        "encoder__smooth": trial.suggest_float("smooth", 0.3, 15),
        "model__n_estimators": trial.suggest_int("n_estimators", 50, 1500),
        "model__max_depth": trial.suggest_int("max_depth", 4, 20),
        "model__max_features": trial.suggest_int("max_features", 2, len(all_features)),
        "model__min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "model__ccp_alpha": trial.suggest_float("ccp_alpha", 0.0001, 0.02, log=True)
    }

    pipeline_rf.set_params(**params_rf)

    score = cross_val_score(
        estimator=pipeline_rf,
        X=X_train,
        y=y_train,
        cv=4,
        scoring='r2',
        n_jobs=-1
    ).mean()

    return score


study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_random_forest, n_trials=500)


study_rf.best_params, study_rf.best_value


test_df = pd.read_csv(TEST)
X_test = test_df.copy()


best_cb_study_params = study_cb.best_params
best_cb_study_params, study_cb.best_value


best_cat_boost_model = CatBoostRegressor(
    cat_features=cat_features,
    **best_cb_study_params
)

pipeline_cat_boost = Pipeline([
    ('feature_processor', feature_processor),
    ('model', best_cat_boost_model)
])

pipeline_cat_boost.fit(X_train, y_train)

pred_cb = pipeline_cat_boost.predict(X_test)


y_pred_cb = pd.DataFrame({'id': np.arange(len(X_test)), 'Rating': pred_cb})
y_pred_cb.to_csv("cat_boost.csv", index=False)


best_lr_study_params = study_lr.best_params
best_lr_study_params, study_lr.best_value


smooth = best_lr_study_params['smooth']

encoder = TargetEncoder(
    target_type='continuous',
    smooth=smooth
)

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', MinMaxScaler(), num_features),
        ('catigorial', encoder, cat_features)
    ]
)

pipeline_linear_regression = Pipeline([
    ('feature_processor', feature_processor),
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

pipeline_linear_regression.fit(X_train, y_train)

pred_lr = pipeline_linear_regression.predict(X_test)


y_pred_lr = pd.DataFrame({'id': np.arange(len(X_test)), 'Rating': pred_lr})
y_pred_lr.to_csv("linear_regression.csv", index=False)


best_rf_study_params = study_rf.best_params
best_rf_study_params, study_rf.best_value


smooth = best_rf_study_params['smooth']
rf_params = dict(list(best_rf_study_params.items())[1:])

encoder = TargetEncoder(
    target_type='continuous',
    smooth=smooth
)

pipeline_random_forest = Pipeline([
    ('feature_processor', feature_processor),
    ('encoder', encoder),
    ('model', RandomForestRegressor(**rf_params))
])

pipeline_random_forest.set_params()

pipeline_random_forest.fit(X_train, y_train)

pred_rf = pipeline_random_forest.predict(X_test)


y_pred_rf = pd.DataFrame({'id': np.arange(len(X_test)), 'Rating': pred_rf})
y_pred_rf.to_csv("random_forest.csv", index=False)


pred_rf_cb = 0.48 * pred_rf + 0.52 * pred_cb


y_pred_rf_cb = pd.DataFrame({'id': np.arange(len(X_test)), 'Rating': pred_rf_cb})
y_pred_rf_cb.to_csv("random_forest_and_cat_boost.csv", index=False)

