import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

y = train["Personality"]
X = train.drop(["Personality", "id"], axis=1)


le = LabelEncoder()
y = le.fit_transform(y)

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=42)

space = {
    # Numeric imputer
    "preprocessor__num__imputer__max_iter": hp.quniform("preprocessor__num__imputer__max_iter", 5, 40, 1),
    "preprocessor__num__imputer__tol": hp.loguniform("preprocessor__num__imputer__tol", np.log(1e-5), np.log(1e-2)),
    "preprocessor__num__imputer__initial_strategy": hp.choice("preprocessor__num__imputer__initial_strategy", ["mean", "median"]),

    # Categorical imputer
    "preprocessor__cat__imputer__max_iter": hp.quniform("preprocessor__cat__imputer__max_iter", 5, 40, 1),
    "preprocessor__cat__imputer__tol": hp.loguniform("preprocessor__cat__imputer__tol", np.log(1e-5), np.log(1e-2)),

    # Random Forest parameters
    "model__n_estimators": hp.quniform("model__n_estimators", 200, 2000, 50),
    "model__max_depth": hp.quniform("model__max_depth", 10, 50, 1),
    "model__min_samples_split": hp.quniform("model__min_samples_split", 2, 20, 1),
    "model__min_samples_leaf": hp.quniform("model__min_samples_leaf", 1, 10, 1),
    "model__max_features": hp.choice("model__max_features", ["auto", "sqrt", 0.3, 0.5, 0.8]),
    "model__bootstrap": hp.choice("model__bootstrap", [True, False]),
}


def objective(params):

    params['preprocessor__num__imputer__max_iter'] = int(params['preprocessor__num__imputer__max_iter'])
    params['preprocessor__cat__imputer__max_iter'] = int(params['preprocessor__cat__imputer__max_iter'])
    params['model__n_estimators'] = int(params['model__n_estimators'])
    params['model__max_depth'] = int(params['model__max_depth'])
    params['model__min_samples_split'] = int(params['model__min_samples_split'])
    params['model__min_samples_leaf'] = int(params['model__min_samples_leaf'])


    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(exclude=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(
            random_state=42,
            max_iter=params['preprocessor__num__imputer__max_iter'],
            tol=params['preprocessor__num__imputer__tol'],
            initial_strategy=params['preprocessor__num__imputer__initial_strategy']
        ))
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ('imputer', IterativeImputer(
            random_state=42,
            max_iter=params['preprocessor__cat__imputer__max_iter'],
            tol=params['preprocessor__cat__imputer__tol'],
            initial_strategy='most_frequent'
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=params['model__n_estimators'],
        max_depth=params['model__max_depth'],
        min_samples_split=params['model__min_samples_split'],
        min_samples_leaf=params['model__min_samples_leaf'],
        max_features=params['model__max_features'],
        bootstrap=params['model__bootstrap'],
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipe.fit(trainX, trainy)
    pred = pipe.predict(testX)
    acc = accuracy_score(testy, pred)


    return {"loss": -acc, "status": STATUS_OK}


trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials,
    rstate=np.random.default_rng(42)
)

print("Best Hyperparameters Found:")
print(best)

best_params = space_eval(space, best)

best_params['preprocessor__num__imputer__max_iter'] = int(best_params['preprocessor__num__imputer__max_iter'])
best_params['preprocessor__cat__imputer__max_iter'] = int(best_params['preprocessor__cat__imputer__max_iter'])
best_params['model__n_estimators'] = int(best_params['model__n_estimators'])
best_params['model__max_depth'] = int(best_params['model__max_depth'])
best_params['model__min_samples_split'] = int(best_params['model__min_samples_split'])
best_params['model__min_samples_leaf'] = int(best_params['model__min_samples_leaf'])


categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(
        random_state=42,
        max_iter=best_params['preprocessor__num__imputer__max_iter'],
        tol=best_params['preprocessor__num__imputer__tol'],
        initial_strategy=best_params['preprocessor__num__imputer__initial_strategy']
    ))
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ('imputer', IterativeImputer(
        random_state=42,
        max_iter=best_params['preprocessor__cat__imputer__max_iter'],
        tol=best_params['preprocessor__cat__imputer__tol'],
        initial_strategy='most_frequent'
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

model = RandomForestClassifier(
    n_estimators=best_params['model__n_estimators'],
    max_depth=best_params['model__max_depth'],
    min_samples_split=best_params['model__min_samples_split'],
    min_samples_leaf=best_params['model__min_samples_leaf'],
    max_features=best_params['model__max_features'],
    bootstrap=best_params['model__bootstrap'],
    random_state=42,
    n_jobs=-1
)

final_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

final_pipe.fit(X, y)

X_test = test.drop(["Personality", "id"], axis=1, errors='ignore') 
test_ids = test['id']

test_pred_encoded = final_pipe.predict(X_test)

test_pred_decoded = le.inverse_transform(test_pred_encoded)

submission = pd.DataFrame({'id': test_ids, 'Personality': test_pred_decoded})

submission_filename = 'submission_default_rf.csv'
submission.to_csv(submission_filename, index=False)

