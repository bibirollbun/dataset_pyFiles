!pip install feature-engine -q
!pip install ydf -q

from functools import reduce
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, mean_squared_log_error
import ydf


DEBUG = True


train = pd.read_csv(r'/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e5/sample_submission.csv')
target = "Calories"


SAMPLE_SIZE_SHOWCASE = len(train.index)

train = train.sample(SAMPLE_SIZE_SHOWCASE, random_state=6000).reset_index(drop=True)


train = train.drop("id", axis=1)
test = test.drop("id", axis=1)


train


train.info()


train.nunique()


categorical_columns = ["Sex"]
numerical_columns = [col for col in train.columns.to_list() if col not in ["Sex", target]]

categorical_columns, numerical_columns


from feature_engine.discretisation import DecisionTreeDiscretiser

dtd = DecisionTreeDiscretiser(
    regression=True, 
    scoring="neg_root_mean_squared_log_error", 
    random_state=42,
    cv=5
)
dtd.fit(train.loc[:, numerical_columns], train[target])
train_discret = dtd.transform(train.loc[:, numerical_columns])
test_discret = dtd.transform(test.loc[:, numerical_columns])

# rename columns to prevent ambigous column names later
new_col_names = [col + "_dtd" for col in test_discret.columns.to_list()]
train_discret.columns = new_col_names
test_discret.columns = new_col_names

# store this for future usage
train_discret.to_parquet("/kaggle/working/dtd_train.parquet")
test_discret.to_parquet("/kaggle/working/dtd_test.parquet")


train_discret


from feature_engine.creation import RelativeFeatures

ref_dfs_train = []
ref_dfs_test = []

for ref_col in numerical_columns:
    print(f"Execute operation on column {ref_col}")
    variables = [col for col in numerical_columns if col != ref_col]
    rf = RelativeFeatures(
        variables=variables,
        reference=[ref_col],
        func=["div",  "add", "mod"]
    )
    rf.fit(train.loc[:, numerical_columns])
    train_ref = rf.transform(train.loc[:, numerical_columns]).drop(numerical_columns, axis=1)
    test_ref = rf.transform(test.loc[:, numerical_columns]).drop(numerical_columns, axis=1)

    # store this for future usage
    train_ref.to_parquet(f"/kaggle/working/div_train_{ref_col}.parquet")
    test_ref.to_parquet(f"/kaggle/working/div_test_{ref_col}.parquet")

    # append for later usage
    ref_dfs_train.append(train_ref)
    ref_dfs_test.append(test_ref)

    print(train_ref.columns)


train_full = reduce(
    lambda left, right: pd.concat([left, right], axis=1),
    [train_discret] + ref_dfs_train,
    train                    
)

test_full = reduce(
    lambda left, right: pd.concat([left, right], axis=1),
    [test_discret] + ref_dfs_test,
    test                    
)

train_full.to_parquet(f"/kaggle/working/full_concat_train_{ref_col}.parquet")
test_full.to_parquet(f"/kaggle/working/full_concat_test_{ref_col}.parquet")

train_full.shape, test_full.shape


# make YDF sklearn compatible: this is nice ChatGPT o1 code to be fair ;-)

class YDFGBTR(BaseEstimator, RegressorMixin):
    """
    scikit-learn-compatible wrapper around ydf.GradientBoostedTreesLearner
    """
    def __init__(self, label, task=ydf.Task.REGRESSION, **learner_kwargs):
        self.label = label
        self.task = task
        self.learner_kwargs = learner_kwargs

    def get_params(self, deep=True):
        return {"label": self.label,
                "task": self.task,
                **self.learner_kwargs}

    def set_params(self, **params):
        # update attributes in-place so clone() works
        self.label = params.pop("label", self.label)
        self.task = params.pop("task", self.task)
        self.learner_kwargs.update(params)
        return self

    def _new_learner(self):
        return ydf.GradientBoostedTreesLearner(
            label=self.label, task=self.task, **self.learner_kwargs)

    def fit(self, X, y):
        data = X.copy()
        data[self.label] = y
        self.model_ = self._new_learner().train(data)
    
        # ---- variable importances ---------------------------------------
        # ydf returns: Dict[str, List[Tuple[float, str]]]
        vis = self.model_.variable_importances()
        if vis:
            key  = next(iter(vis))                    # e.g. "NUM_AS_ROOT"
            pairs = vis[key]                          # List[(score, name)]
            scores = {name: score for score, name in pairs}
            self.feature_importances_ = np.array(
                [scores.get(col, 0.0) for col in X.columns]
            )
        else:
            # fall back: equal weights so ProbeFeatureSelection keeps all
            self.feature_importances_ = np.ones(X.shape[1])
    
        return self

    def predict(self, X):
        """
        Return a 1-D numpy array with the predictions, regardless of the
        exact type that ydf.GradientBoostedTreesLearner.predict() yields.
        """
        y_hat = self.model_.predict(X)
    
        # ydf <= 0.6.3: Prediction object with `.predictions`
        if hasattr(y_hat, "predictions"):
            return np.asarray(y_hat.predictions).ravel()
    
        # ydf >= 0.6.4: already a numpy array (or pandas Series)
        return np.asarray(y_hat).ravel()


rmsle = make_scorer(
    lambda y, y_pred: -np.sqrt(mean_squared_log_error(y, y_pred)),
    greater_is_better=False,
)


from feature_engine.selection import ProbeFeatureSelection

ydf_est = YDFGBTR(label=target, num_trees=300, random_seed=150)

sel = ProbeFeatureSelection(
    estimator=ydf_est,
    scoring=rmsle,          # custom scorer
    n_probes=3,
    distribution="normal",
    cv=5,
    random_state=150,
)

train_selected = sel.fit_transform(
    train_full.drop(columns=target),
    train_full[target],
)

test_selected = sel.transform(test_full)

print(train_selected.shape, test_selected.shape)


from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# Configuration
N_FOLDS = 5
SEED = 42
target = "Calories"  # assuming this is still the same

# Prepare data
X = train_selected.copy()

train_full[target] = np.log1p(train_full[target])
y = train_full[target].values

# Store predictions for test set
test_preds = np.zeros((test_selected.shape[0], N_FOLDS))

# Store individual models (optional)
models = []

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Training fold {fold + 1}/{N_FOLDS}...")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = YDFGBTR(label=target, num_trees=300, random_seed=SEED + fold)
    model.fit(X_train.assign(**{target: y_train}), y_train)

    # Store model
    models.append(model)

    # Predict on test set
    test_preds[:, fold] = model.predict(test_selected)

# Average predictions
final_preds = test_preds.mean(axis=1)

# Create submission
submission[target] = final_preds
submission[target] = submission[target].clip(0)
submission[target] = np.expm1(submission[target])
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("✅ Submission file created!")

