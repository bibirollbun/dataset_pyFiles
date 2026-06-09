from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, make_scorer


from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, SparkTrials
#import cudf
#import cupy as cp


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings


warnings.filterwarnings("ignore")


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


class Config:
    train_file = "/kaggle/input/playground-series-s5e6/train.csv"
    test_file = "/kaggle/input/playground-series-s5e6/test.csv"
    sample_sub_file = "/kaggle/input/playground-series-s5e6/sample_submission.csv"
    target = "Fertilizer Name"
    seed = 1234
    training_depth = 30


full_train = pd.read_csv(Config.train_file, index_col="id")
full_test = pd.read_csv(Config.test_file, index_col="id")

full_train.head()


label_encoder = LabelEncoder()
full_train[Config.target] = label_encoder.fit_transform(
    full_train[Config.target]
)

# Create setup
X = full_train.drop(Config.target, axis=1)
y = full_train[Config.target]
X_test = full_test

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=Config.seed,
    stratify=y
)


full_train.head()


# Separate the categorical and numerical columns
categorical_cols = [
    cname for cname in X_train.columns
    if X_train[cname].nunique() < 20
    and X_train[cname].dtype == 'object'
]
numerical_cols = [
    cname for cname in X_train.columns
    if X_train[cname].dtype in ['int64', 'float64']
]
print(f"Categorical Columns: {categorical_cols}")
print(f"Numerical Columns: {numerical_cols}")


sample_df = full_train.sample(n=5000, random_state=Config.seed)
sns.pairplot(sample_df[numerical_cols + [Config.target]], hue=Config.target)


def mapk(y_true, y_pred_proba, k=3):
    """
    Compute mean average precision at k (MAP@k)
    y_true: array-like of shape (n_samples,)
    y_pred_proba: array-like of shape (n_samples, n_classes) — predicted probabilities
    """
    y_true = np.array(y_true)
    y_pred_topk = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :k]

    score = 0.0
    for i in range(len(y_true)):
        if y_true[i] in y_pred_topk[i]:
            rank = np.where(y_pred_topk[i] == y_true[i])[0][0] + 1
            score += 1.0 / rank
    return score / len(y_true)

# Wrap as sklearn scorer
map3_scorer = make_scorer(mapk, needs_proba=True, greater_is_better=True)


numerical_transformer = Pipeline(steps=[
    ('standard_scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='constant'))
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)


# hyperparameter search space
space = {
    'max_depth': hp.quniform('max_depth', 5, 16, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'gamma': hp.uniform('gamma', 0, 5),
#    'n_estimators': 5000,  # Fixed, use early stopping to avoid overfitting
#    'early_stopping_rounds': 100,
    'n_estimators': 300,
    'verbose': True,
    'tree_method': 'hist',
    'n_jobs': -1,
}


def objective(params):
    params['max_depth'] = int(params['max_depth'])
    params['min_child_weight'] = int(params['min_child_weight'])
    params['n_estimators'] = int(params['n_estimators'])
    if 'num_parallel_tree' in params:
        params['num_parallel_tree'] = int(params['num_parallel_tree'])

    model = XGBClassifier(
        objective='multi:softprob',
        device='cuda',
        num_class=len(np.unique(y_train)),
        random_state=Config.seed,
        **params
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring=map3_scorer,
        cv=5,
        n_jobs=-1
    )

    loss = -np.mean(scores)  # Hyperopt minimizes loss
    return {'loss': loss, 'status': STATUS_OK}


trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=Config.training_depth,
    trials=trials
)

print("Best hyperparameters:", best)


results = []
for trial in trials.trials:
    row = trial['misc']['vals'].copy()
    for k in row:
        row[k] = row[k][0] if row[k] else None  # extract scalar from list
    row['loss'] = trial['result']['loss']
    results.append(row)

trials_df = pd.DataFrame(results)


for col in trials_df.columns:
    if col != 'loss':
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=trials_df[col], y=trials_df['loss'])
        plt.title(f'Loss vs {col}')
        plt.xlabel(col)
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


best['max_depth'] = int(best['max_depth'])
best['min_child_weight'] = int(best['min_child_weight'])
best['n_estimators'] = 5000

model = XGBClassifier(
        objective='multi:softprob',
        device='cuda',
        num_class=len(np.unique(y_train)),
        random_state=Config.seed,
        **best
)
X_valid_preprocessed = preprocessor.fit_transform(X_valid, y_valid)
clf = Pipeline(steps=[
    ('preprocesser', preprocessor),
    ('model', model)
])
clf.fit(X_train, y_train,
        model__early_stopping_rounds=10,
        model__eval_set=[(X_valid_preprocessed, y_valid)])


best_n_estimators = model.best_iteration + 1


space = {
    'max_depth': hp.quniform('max_depth', best['max_depth']-2, best['max_depth']+4, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
    'subsample': best['subsample'],
    'colsample_bytree': best['colsample_bytree'],
    'min_child_weight': best['min_child_weight'],
    'gamma': best['gamma'],
    'max_delta_step': hp.quniform('max_delta_step', 0, 10, 1),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(0.01), np.log(10)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(0.1), np.log(10)),
    'num_parallel_tree': hp.quniform('num_parallel_tree', 1, 10, 1),
#    'n_estimators': 5000,  # Fixed, use early stopping to avoid overfitting
#    'early_stopping_rounds': 100,
    'n_estimators': best_n_estimators,
    'verbose': True,
    'tree_method': 'hist',
    'n_jobs': -1,
}
pbest = best.copy()
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=Config.training_depth,
    trials=trials
)

print("Best hyperparameters:", best)


results = []
for trial in trials.trials:
    row = trial['misc']['vals'].copy()
    for k in row:
        row[k] = row[k][0] if row[k] else None  # extract scalar from list
    row['loss'] = trial['result']['loss']
    results.append(row)

trials_df = pd.DataFrame(results)

for col in trials_df.columns:
    if col != 'loss':
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=trials_df[col], y=trials_df['loss'])
        plt.title(f'Loss vs {col}')
        plt.xlabel(col)
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


best['max_depth'] = int(best['max_depth'])
best['n_estimators'] = 5000
best['num_parallel_tree'] = int(best['num_parallel_tree'])
best['verbose'] = True
best['tree_method'] = 'hist'
best['n_jobs'] = -1

for k, v in pbest.items():
    if k not in best:
        best[k] = v

model = XGBClassifier(
        objective='multi:softprob',
        device='cuda',
        num_class=len(np.unique(y_train)),
        random_state=Config.seed,
        **best
)
clf = Pipeline(steps=[
    ('preprocesser', preprocessor),
    ('model', model)
])
clf.fit(X_train, y_train,
        model__early_stopping_rounds=10,
        model__eval_set=[(X_valid_preprocessed, y_valid)])


print(best)


preds = clf.predict(X_valid)


cmatrix = confusion_matrix(y_valid, preds)


g = ConfusionMatrixDisplay(cmatrix)
g.plot()


y_pred_probs = clf.predict_proba(X_test)


y_pred_probs[0]


score = mapk(y_train, clf.predict_proba(X_train), k=3)
print(score)


predictions = []
for i in np.argsort(y_pred_probs)[:, -3:][:, ::-1]:
    prediction = label_encoder.inverse_transform(i)
    predictions.append(' '.join(prediction))  # space delimited

output = pd.DataFrame({'id': X_test.index, 'Fertilizer Name': predictions})
output.to_csv('wider_xgb_submission.csv', index=False)


model.save_model('xgb_wide.json')


#from IPython.display import FileLink
#
#FileLink('xgb_wide.json')

