# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from sklearn.base          import BaseEstimator, clone, TransformerMixin
from sklearn.compose       import ColumnTransformer
from sklearn.impute        import SimpleImputer
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn import metrics


from sklearn.linear_model  import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble      import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble      import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble      import ExtraTreesClassifier, ExtraTreesRegressor
from lightgbm              import LGBMClassifier, LGBMRegressor
from xgboost               import XGBClassifier, XGBRegressor
from catboost              import CatBoostClassifier, CatBoostRegressor


train_df = pd.read_csv('/kaggle/input/predict-y-values-at-new-x-values/simple_regression_train.csv')
test_df = pd.read_csv('/kaggle/input/predict-y-values-at-new-x-values/simple_regression_test.csv')


class FourierFeaturizer(BaseEstimator, TransformerMixin):
    """
    Add sin/cos harmonics of a 1-D numeric column.

    Parameters
    ----------
    period : float
        Fundamental period *in the same units as x*.
        We have ~5 full cycles over x∈[0,20] ⇒ period ≈ 4.
    n_harmonics : int, default=3
        How many harmonics (k = 1..n) to append.
    """
    def __init__(self, period=4, n_harmonics=3):
        self.period      = period
        self.n_harmonics = n_harmonics

    # nothing to fit
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X).reshape(-1, 1)                       # (n,1)
        feats = [X]                                            # raw x  → trend / intercept
        for k in range(1, self.n_harmonics + 1):
            angle = 2 * np.pi * k * X / self.period
            s, c = np.sin(angle), np.cos(angle)
            feats.extend([s, c, X * s, X * c])                # add damping interaction
        return np.hstack(feats)                                # (n, 1 + 4h)


# Using the best_params

fourier_ridge = Pipeline([
    ('fourier', FourierFeaturizer(period=4.1, n_harmonics=2)),
    ('scaler' , StandardScaler()),
    ('ridge'  , Ridge(alpha=0.1, random_state=42))
])


# # -------------------------------------------------------------------
# # Run Grid Search on the desired parameters to improve CV score
# # -------------------------------------------------------------------
# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'ridge__alpha'     : np.arange(0.1, 10, 0.1).tolist(),   # controls peak height
#     'fourier__n_harmonics': [2, 3, 4, 5],              # adds shape flexibility
#     'fourier__period':      np.arange(3.0, 4.5, 0.1).tolist()
# }

# tscv = TimeSeriesSplit(n_splits=10)

# gs = GridSearchCV(
#         estimator = fourier_ridge,
#         param_grid= param_grid,
#         scoring   = 'neg_root_mean_squared_error',
#         cv        = tscv,
#         n_jobs    = -1,              # use all CPU cores
#         verbose   = 2                # print progress
# )

# # feature_col = [c for c in train_df.columns if c != 'y'][0]

# gs.fit(train_df['t'], train_df['y'])

# print("Best params :", gs.best_params_)
# print("Best CV RMSE:", -gs.best_score_)

# # keep the tuned pipeline for all downstream work
# best_model = gs.best_estimator_



# # keep the tuned pipeline for all downstream work
# best_model = gs.best_estimator_
# best_model


# print("Best params :", gs.best_params_)
# print("Best CV RMSE:", -gs.best_score_)


def _detect_feature_types(df, target):
    """
    Categorise columns into numeric, categorical or binary.

    The heuristic is intentionally simple yet surprisingly useful:

    * numeric   → `pandas.api.types.is_numeric_dtype`
    * binary    → exactly two distinct non-NA values
    * categorical → anything else (incl. strings)

    Parameters
    ----------
    df : pandas.DataFrame
        The training dataframe.
    target : str
        Name of the target column to be ignored.

    Returns
    -------
    list
        Numeric column names.
    list
        Categorical column names.
    list
        Binary column names.

    Notes
    -----
    If this heuristic mis-classifies features, use `feature_sets` inside
    `run_experiment` to supply your own lists.
    """
    num, cat, bin_ = [], [], []
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num.append(col)
        elif df[col].dropna().nunique() == 2:
            bin_.append(col)
        else:
            cat.append(col)
    return num, cat, bin_


def _make_preprocessor(num_cols, 
                       cat_cols, 
                       bin_cols,
                       impute=True, 
                       scale_numeric=True):
    """
    Build a `ColumnTransformer` that applies the right preprocessing
    to each feature subset.

    Parameters
    ----------
    num_cols, cat_cols, bin_cols : list
        Lists of column names.
    impute : bool, default=True
        Add `SimpleImputer` (mean for numeric, most-frequent otherwise).
    scale_numeric : bool, default=True
        Standardise numeric columns with `StandardScaler`.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Ready to drop into a `Pipeline`.
    """
    num_pipe = []
    if impute:
        num_pipe.append(('imputer', SimpleImputer(strategy='mean')))
    if scale_numeric:
        num_pipe.append(('scaler', StandardScaler()))
    num_pipe = Pipeline(num_pipe) if num_pipe else 'passthrough'

    cat_pipe = [('encoder', OneHotEncoder(handle_unknown='ignore'))]
    if impute:
        cat_pipe.insert(0, ('imputer', SimpleImputer(strategy='most_frequent')))
    cat_pipe = Pipeline(cat_pipe)

    bin_pipe = [('encoder', OneHotEncoder(drop='if_binary'))]
    if impute:
        bin_pipe.insert(0, ('imputer', SimpleImputer(strategy='most_frequent')))
    bin_pipe = Pipeline(bin_pipe)

    return ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
        ('bin', bin_pipe, bin_cols)
    ])


def _default_models(problem, rs=42):
    """
    Provide a small, sensible set of baseline models depending on the task.

    Parameters
    ----------
    problem : {'binary', 'multiclass', 'regression'}
        Task type.
    rs : int, default=42
        Random seed forwarded to all models that accept it.

    Returns
    -------
    dict
        Mapping ``name → estimator instance``.

    Raises
    ------
    ValueError
        If `problem` is not one of the supported strings.
    """
    if problem == 'binary':
        return {
            # 'logreg': LogisticRegression(max_iter=1000, n_jobs=-1, random_state=rs),
            'lgbm'  : LGBMClassifier(random_state=rs),
            'cat'   : CatBoostClassifier(random_state=rs, verbose=False),
            # 'rf'    : RandomForestClassifier(random_state=rs),
            # 'et'    : ExtraTreesClassifier(random_state=rs),
        }
    if problem == 'multiclass':
        return {
            'lgbm'  : LGBMClassifier(objective='multiclass', random_state=rs),
            # 'cat'   : CatBoostClassifier(random_state=rs, verbose=False),
            # 'rf'    : RandomForestClassifier(random_state=rs),
            'et'    : ExtraTreesClassifier(random_state=rs),
        }
    if problem == 'regression':
        return {
            'ridge' : Ridge(random_state=rs),
            'lgbm'  : LGBMRegressor(random_state=rs),
            # 'cat'   : CatBoostRegressor(random_state=rs, verbose=False),
            # 'rf'    : RandomForestRegressor(random_state=rs),
            # 'et'    : ExtraTreesRegressor(random_state=rs),
        }
    raise ValueError("problem must be 'binary', 'multiclass', or 'regression'")


def _get_cv(problem, n_splits=5, rs=42, strategy='auto'):
    """
    Choose and configure a cross-validation splitter.

    Parameters
    ----------
    problem : str
        Task type.
    n_splits : int, default=5
        Number of folds.
    rs : int, default=42
        Random seed.
    strategy : {'auto', 'kfold', 'stratified', 'time'}, default='auto'
        * auto        → StratifiedKFold for classification, KFold for regression  
        * kfold       → always KFold  
        * stratified  → always StratifiedKFold  
        * time        → always TimeSeriesSplit

    Returns
    -------
    sklearn.model_selection.BaseCrossValidator

    Raises
    ------
    ValueError
        If `strategy` is invalid.
    """
    if strategy == 'auto':
        strategy = 'stratified' if problem in ('binary', 'multiclass') else 'kfold'
    if strategy == 'stratified':
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    if strategy == 'kfold':
        return KFold(n_splits=n_splits, shuffle=True, random_state=rs)
    if strategy == 'time':
        return TimeSeriesSplit(n_splits=n_splits)
    raise ValueError("strategy must be 'auto', 'kfold', 'stratified' ot 'time'")


def _get_metrics(problem, names=None):
    """
    Build a dict of metric callables.

    Supported inputs are any function names in `sklearn.metrics` **plus** the
    convenience alias ``'neg_root_mean_squared_error'`` (converted internally).

    Parameters
    ----------
    problem : str
        Task type.
    names : list of str, optional
        Metric names. If `None`, falls back to sensible defaults:
        * binary    → ['roc_auc', 'accuracy']
        * multiclass→ ['accuracy']
        * regression→ ['neg_root_mean_squared_error']

    Returns
    -------
    dict
        Mapping ``metric_name → callable``.

    Raises
    ------
    ValueError
        If a requested metric is unknown.
    """
    default = {
        'binary'    : ['roc_auc', 'accuracy'],
        'multiclass': ['accuracy'],
        'regression': ['neg_root_mean_squared_error']
    }
    names = names or default[problem]
    scorers = {}
    for name in names:
        if hasattr(metrics, name):
            scorers[name] = getattr(metrics, name)
        elif hasattr(metrics, f'{name}_score'):
            scorers[name] = getattr(metrics, f'{name}_score')
        elif name == 'neg_root_mean_squared_error':
            scorers[name] = metrics.mean_squared_error
        else:
            raise ValueError(f"Metric {name} not found.")
    return scorers


def run_experiment(train, 
                   test, 
                   target, 
                   problem,
                   feature_sets=None,
                   model_dict=None,
                   metric_names=None,
                   cv_strategy='auto',
                   n_splits=5,
                   random_state=42,
                   impute=True,
                   scale_numeric=True,
                   verbose=True,
                   nb_name=None):
    """
    Run cross-validated model comparisons and produce tidy outputs.

    Parameters
    ----------
    train : pandas.DataFrame
        Training data **including** the target column.
    test : pandas.DataFrame
        Test/hold-out data **without** the target column.
    target : str
        Name of the target column in `train`.
    problem : {'binary', 'multiclass', 'regression'}
        Task type.
    feature_sets : tuple(list, list, list), optional
        Pre-detected feature lists ``(numeric, categorical, binary)``.
        If omitted, `_detect_feature_types` is used.
    model_dict : dict, optional
        Custom mapping ``name → estimator``. Falls back to `_default_models`.
    metric_names : list of str, optional
        Evaluation metrics. See `_get_metrics`.
    cv_strategy : {'auto', 'kfold', 'stratified', 'time'}, default='auto'
        Cross-validation splitter selection.
    n_splits : int, default=5
        Number of CV folds.
    random_state : int, default=42
        Forwarded to all RNG-aware components.
    impute : bool, default=True
        Add `SimpleImputer` steps.
    scale_numeric : bool, default=True
        Standardise numeric columns.
    verbose : bool, default=True
        Print progress to stdout.
    nb_name : str or None
        Notebook/script name to embed in submission filenames.  If *None*,
        the name is guessed via ``_detect_nb_name``.

    Returns
    -------
    dict
        * ``'cv_scores'`` → DataFrame (multi-index columns metric × fold)  
        * ``'summary'``   → DataFrame (mean & std per metric, sorted)  
        * ``'test_pred'`` → dict {model_name: ndarray predictions}  

    Examples
    --------
    ```python
    from simple_flex_cv import run_experiment
    res = run_experiment(train_df, test_df,
                         target="rainfall",
                         problem="binary",
                         metric_names=["roc_auc", "f1_score"])
    print(res["summary"])
    preds = res["test_pred"]["lgbm"]
    ```

    Notes
    -----
    Each model writes a CSV to the ``submission/`` folder with the pattern  
    ``<model>-<nb_name>_<cv_strategy>_<score>_<n_splits>cv.csv``.
    """
    X = train.drop(columns=[target])
    y = train[target].values

    # Features
    if feature_sets is None:
        nums, cats, bins = _detect_feature_types(train, target)
    else:
        nums, cats, bins = feature_sets

    pre = _make_preprocessor(nums, cats, bins, impute, scale_numeric)

    # Models, metrics, CV
    models   = model_dict or _default_models(problem, random_state)
    scorers  = _get_metrics(problem, metric_names)
    metrics_ = list(scorers.keys())
    cv       = _get_cv(problem, n_splits, random_state, cv_strategy)

    cv_scores = pd.DataFrame(
        index=models.keys(),
        columns=pd.MultiIndex.from_product(
            [metrics_, [f'fold_{i+1}' for i in range(n_splits)]]
        ),
        dtype=float
    )
    test_pred = {}

    # ---------------------- main loop ------------------------------------------
    for name, est in models.items():
        if verbose:
            print(f"\nTraining {name}")
        pipe = Pipeline([('pre', pre), ('est', est)])

        for f, (tr, vl) in enumerate(cv.split(X, y)):
            if verbose:
                print(f"    └─ Fold {f+1}/{cv.n_splits}")
            pipe_fold = clone(pipe).fit(X.iloc[tr], y[tr])

            if problem in ('binary', 'multiclass'):
                prob_pred  = pipe_fold.predict_proba(X.iloc[vl])        # shape (n, C)
                if problem == 'binary':
                    prob_pred_1d = prob_pred[:, 1]                     # keep positive-class probs
                label_pred = pipe_fold.predict(X.iloc[vl])             # hard labels
            else:                                                      # regression
                prob_pred_1d = label_pred = pipe_fold.predict(X.iloc[vl])

            # 2) Pick the correct prediction form for each metric
            for m in metrics_:
                fn = scorers[m]

                # metrics that *require* probabilities
                if m.startswith('roc_auc') or m == 'log_loss':
                    y_hat = prob_pred_1d if problem == 'binary' else prob_pred
                # special case: RMSE alias
                elif m == 'neg_root_mean_squared_error':
                    y_hat = label_pred
                    score = -np.sqrt(fn(y[vl], y_hat))
                    cv_scores.loc[name, (m, f'fold_{f+1}')] = score
                    continue
                # everything else (accuracy, f1, precision, recall…)
                else:
                    y_hat = label_pred

                score = fn(y[vl], y_hat)
                cv_scores.loc[name, (m, f'fold_{f+1}')] = score

        # fit on full data ➜ predict test
        pipe.fit(X, y)

        if problem == 'binary':
            # keep positive-class probabilities
            t_pred = pipe.predict_proba(test)[:, 1]

        elif problem == 'multiclass':
            # convert arg-max index → ORIGINAL class label
            prob_mat = pipe.predict_proba(test)
            t_pred   = pipe.classes_[prob_mat.argmax(axis=1)]

        else:  # regression
            t_pred = pipe.predict(test)

        test_pred[name] = t_pred

    # Summary (mean/std by metric)
    means = cv_scores.groupby(level=0, axis=1).mean().add_suffix('_mean')
    stds  = cv_scores.groupby(level=0, axis=1).std().add_suffix('_std')
    summary = pd.concat([means, stds], axis=1)
    summary.sort_values(summary.columns[0], ascending=False, inplace=True)

    # Auto-save a submission CSV for each model
    os.makedirs("submission", exist_ok=True)                   # create folder if absent
    first_metric = metrics_[0]                                 # e.g. "accuracy" or "roc_auc"

    for model_name, preds in test_pred.items():
        # Select what to write, depending on the task
        if problem == "binary":
            out_vals = preds                                   # 1-D positive-class prob.
        elif problem == "multiclass":
            out_vals = preds                                   # hard labels
        else:                                                  # regression
            out_vals = preds

        # Pull the CV mean score for the first metric
        cv_score = summary.loc[model_name, f"{first_metric}_mean"]

        # Build filename: model_CVtype_metricScore_CVsplits.csv
        fname = f"{nb_name}_{model_name}_{cv_strategy}_{cv_score:.4f}_{n_splits}cv.csv"
        fpath = os.path.join("submission", fname)

        # Assemble & write
        sub_df = pd.DataFrame({
            "t": test["t"] if "t" in test.columns else np.arange(len(out_vals)),
            target: out_vals
        })
        sub_df.to_csv(fpath, index=False)

        if verbose:
            print(f"Saved submission to {fpath}")

    return {'cv_scores': cv_scores, 'summary': summary, 'test_pred': test_pred}


feature_col = ['t']


results = run_experiment(
    train        = train_df,
    test         = test_df,
    target       = 'y',
    problem      = 'regression',                 
    model_dict   = {'fourier_ridge': fourier_ridge},
    metric_names = ['neg_root_mean_squared_error'],
    cv_strategy  = 'time',                       
    n_splits     = 10,
    impute       = False,                        
    scale_numeric= False,                        
    nb_name      = '02_baseline',
    verbose      = True
)


print(results['summary'])


# ----- fit on full training data & predict everywhere ------------------------
feature_col = [c for c in train_df.columns if c != 'y'][0]

final_model = fourier_ridge.fit(train_df[[feature_col]], train_df['y'])

combined_X     = pd.concat([train_df[[feature_col]], test_df[[feature_col]]])
combined_pred  = final_model.predict(combined_X)

# ----- plot ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(train_df[feature_col], train_df['y'],
            s=12, alpha=0.6, label='Actual (train)')
plt.plot(combined_X[feature_col], combined_pred,
         linewidth=2, label='Predicted (train+test)')
plt.axvline(train_df[feature_col].max(),
            color='grey', linestyle='--', label='train / test split')

plt.xlabel(feature_col)
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


