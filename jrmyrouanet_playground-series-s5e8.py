import pandas as pd

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import optuna
import itertools
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier, plot_importance
from tqdm import tqdm
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

SEED=42
STEP_000 = "STEP_000"
STEP_001 = "STEP_001"
STEP_002 = "STEP_002"
STEP_003 = "STEP_003"
STEP_004 = "STEP_004"

print("Python environment initialized.")


DATA_PATH = "data"
TEST_PATH = "/kaggle/input/playground-series-s5e8/test.csv" 
TRAIN_PATH = "/kaggle/input/playground-series-s5e8/train.csv" 
TRAIN_BANK_FULL_PATH = "/kaggle/input/bank-marketing-dataset-full/bank-full.csv"

COLUMN_NAME_ID = "id"
COLUMN_NAME_AGE = "age"
COLUMN_NAME_BALANCE = "balance"
COLUMN_NAME_DAY = "day"
COLUMN_NAME_DURATION = "duration"
COLUMN_NAME_CAMPAIGN = "campaign"
COLUMN_NAME_PDAYS = "pdays"
COLUMN_NAME_PREVIOUS = "previous"
COLUMN_NAME_JOB = "job"
COLUMN_NAME_MARITAL = "marital"
COLUMN_NAME_EDUCATION = "education"
COLUMN_NAME_DEFAULT = "default"
COLUMN_NAME_HOUSING = "housing"
COLUMN_NAME_LOAN = "loan"
COLUMN_NAME_CONTACT = "contact"
COLUMN_NAME_MONTH = "month"
COLUMN_NAME_POUTCOME = "poutcome"
COLUMN_NAME_Y = "y"

X_train = pd.read_csv(TRAIN_PATH)
X_bank_full = pd.read_csv(TRAIN_BANK_FULL_PATH, delimiter=";")
X_competition = pd.read_csv(TEST_PATH)


X_bank_full.head()


X_bank_full.info()


X_train.head()


X_train.info()


X_bank_full_preprocessed = X_bank_full.copy()
X_bank_full_preprocessed[COLUMN_NAME_Y] = X_bank_full_preprocessed[COLUMN_NAME_Y].map({"no": 0, "yes": 1})
X_train_plus_bank_full = pd.concat(
    [
        X_train,
        X_bank_full_preprocessed
    ],
    axis="rows"
)



X_train_plus_bank_full.info()


X_train_plus_bank_full.nunique()


print(f"Duplicates in X_train_plus_bank_full: {X_train_plus_bank_full.duplicated().sum()}")


print(f"NA in X_train_plus_bank_full:\n{X_train_plus_bank_full.isna().sum()}")


y_train_plus_bank_full = X_train_plus_bank_full[COLUMN_NAME_Y]
y_train_plus_bank_full.value_counts(normalize=True)


scale_pos_weight = (y_train_plus_bank_full.count() - y_train_plus_bank_full.sum()) / y_train_plus_bank_full.sum()


for idx, column_name in enumerate([COLUMN_NAME_BALANCE, COLUMN_NAME_DURATION, COLUMN_NAME_AGE]):
    sns.histplot(X_train_plus_bank_full[column_name], kde=True, bins=50)
    plt.title(f"Histogram({column_name})")
    plt.xlabel(column_name)
    plt.ylabel("Count")
    plt.show()


X_train_plus_bank_full.describe(include="object")


categorical_column_names = [
    col
    for col in X_train_plus_bank_full.columns
    if X_train_plus_bank_full[col].dtype == "object"
]

plt.figure(figsize=(12, 12))
for idx, column_name in enumerate(categorical_column_names, 1):
    plt.subplot(3, 3, idx)
    sns.countplot(x=column_name, data=X_train_plus_bank_full, hue=column_name, palette="Set2")
    plt.xticks(rotation=45)
    plt.title(f'Distribution({column_name})')
    plt.legend().remove()
plt.tight_layout()
plt.show()


for idx, column_name in enumerate([COLUMN_NAME_BALANCE, COLUMN_NAME_DURATION, COLUMN_NAME_AGE]):
    sns.boxplot(x=y_train_plus_bank_full, y=X_train_plus_bank_full[column_name], hue=y_train_plus_bank_full)
    plt.title(f"`{column_name}` vs target `y`")
    plt.show()


plt.figure(figsize=(12, 12))
for idx, column_name in enumerate(categorical_column_names, 1):
    plt.subplot(3, 3, idx)
    sns.countplot(x=column_name, data=X_train_plus_bank_full, hue=y_train_plus_bank_full, palette="Set2")
    plt.xticks(rotation=90)
    plt.title(f'`{column_name}` vs target `y`')
plt.tight_layout()
plt.show()


def new_column_transformer(transformers=None):
    if transformers is None:
        transformers = []
    return ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")



PREPROCESSING_STEP_DROP_COLUMN_ID = "PREPROCESSING_STEP_DROP_COLUMN_ID"


def step_drop_columns(column_name_list):
    return (PREPROCESSING_STEP_DROP_COLUMN_ID, "drop", column_name_list)


PREPROCESSING_STEP_OBJECT_TO_BOOLEAN = "PREPROCESSING_STEP_OBJECT_TO_BOOLEAN"


def step_object_to_boolean(column_name_list):
    def object_to_boolean(columns):
        for column_name in columns.columns:
            columns[column_name] = columns[column_name].map({"no": 0, "yes": 1})
        return columns

    return (
        PREPROCESSING_STEP_OBJECT_TO_BOOLEAN,
        FunctionTransformer(func=object_to_boolean),
        column_name_list
    )


PREPROCESSING_STEP_TARGET_ENCODING = "PREPROCESSING_STEP_TARGET_ENCODING"


def step_target_encoding():
    return (
        PREPROCESSING_STEP_TARGET_ENCODING,
        TargetEncoder(random_state=SEED),
        make_column_selector(dtype_include="object")
    )


class CombineColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Creates column combinations for each length specified in [combination_length_list].
    For example combination_length_list=[2,3] creates combinations of length 2 and 3.
    """
    _columns = []
    _combinations = []
    _new_column_names = []
    combination_length_list = []

    def __init__(self, combination_length_list):
        self.combination_length_list = combination_length_list

    def fit(self, X, y=None):
        self._columns = X.columns
        for combination_length in self.combination_length_list:
            self._combinations.extend(list(itertools.combinations(self._columns, combination_length)))
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for combinations in tqdm(self._combinations, desc="Computing column combinations"):
            new_column_name = "_".join(combinations)
            self._new_column_names.append(new_column_name)
            X_transformed[new_column_name] = X_transformed[list(combinations)].astype(str).agg('_'.join, axis=1)
        return X_transformed

    def get_feature_names_out(self, *args, **params):
        return self._columns.extend(self._new_column_names)


PREPROCESSING_STEP_COMBINE_COLUMNS = "PREPROCESSING_STEP_COMBINE_COLUMNS"


def step_combine_columns(column_name_list, combination_length_list):
    return (
        PREPROCESSING_STEP_COMBINE_COLUMNS,
        CombineColumnTransformer(combination_length_list=combination_length_list),
        column_name_list
    )



COLUMN_NAME_FIRST_CONTACT = "first_contact"
COLUMN_NAME_DAYS_SINCE_LAST_CONTACT = "days_since_last_contact"


class SplitPdaysColumnTransformer(BaseEstimator, TransformerMixin):
    _columns = []

    def fit(self, X, y=None):
        self._columns = X.columns
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        X_transformed[COLUMN_NAME_FIRST_CONTACT] = X_transformed[COLUMN_NAME_PDAYS].map(
            lambda pdays: pdays == -1
        ).astype(int)
        X_transformed[COLUMN_NAME_DAYS_SINCE_LAST_CONTACT] = X_transformed[COLUMN_NAME_PDAYS].map(
            lambda pdays: pdays if pdays != -1 else None
        )
        return X_transformed

    def get_feature_names_out(self, *args, **params):
        return self._columns.extend([COLUMN_NAME_FIRST_CONTACT, COLUMN_NAME_DAYS_SINCE_LAST_CONTACT])


PREPROCESSING_STEP_SPLIT_PDAYS = "PREPROCESSING_STEP_SPLIT_PDAYS"


def step_split_pdays():
    return (PREPROCESSING_STEP_SPLIT_PDAYS, SplitPdaysColumnTransformer(), [COLUMN_NAME_PDAYS])




PREPROCESSING_STEP_BALANCE_TO_BALANCE_LOG = "PREPROCESSING_STEP_BALANCE_TO_BALANCE_LOG"


def step_log1p(column_name_list):
    def column_to_column_log(column):
        return np.log1p(column.clip(lower=0))

    return (
        PREPROCESSING_STEP_BALANCE_TO_BALANCE_LOG,
        FunctionTransformer(func=column_to_column_log),
        column_name_list
    )


COLUMN_NAME_DURATION_SIN = "duration_sin"
COLUMN_NAME_DURATION_COS = "duration_cos"


class DurationSinCosColumnTransformer(BaseEstimator, TransformerMixin):
    _columns = []

    def fit(self, X, y=None):
        self._columns = X.columns
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        X_transformed[COLUMN_NAME_DURATION_SIN] = np.sin(2 * np.pi * X_transformed[COLUMN_NAME_DURATION] / 400)
        X_transformed[COLUMN_NAME_DURATION_COS] = np.cos(2 * np.pi * X_transformed[COLUMN_NAME_DURATION] / 400)
        return X_transformed

    def get_feature_names_out(self, *args, **params):
        return self._columns.extend([COLUMN_NAME_DURATION_SIN, COLUMN_NAME_DURATION_COS])


PREPROCESSING_STEP_DURATION_SINCOS_ENCODING = "PREPROCESSING_STEP_DURATION_SINCOS_ENCODING"


def step_duration_sincos_encoding():
    return (PREPROCESSING_STEP_DURATION_SINCOS_ENCODING, DurationSinCosColumnTransformer(), [COLUMN_NAME_DURATION])


class CountEncodingColumnTransformer(BaseEstimator, TransformerMixin):
    _columns = []
    _new_column_names = []

    def fit(self, X, y=None):
        self._columns = X.columns
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for column_name in X_transformed.columns:
            encoding_dict = X_transformed[column_name].value_counts(normalize=True).to_dict()
            new_column_name = f"{column_name}_count_encoded"
            self._new_column_names.append(new_column_name)
            X_transformed[new_column_name] = X_transformed[column_name].map(encoding_dict)
        return X_transformed

    def get_feature_names_out(self, *args, **params):
        return self._columns.extend(self._new_column_names)


PREPROCESSING_STEP_COUNT_ENCODING = "PREPROCESSING_STEP_COUNT_ENCODING"


def step_count_encoding(column_name_list):
    return (
        PREPROCESSING_STEP_COUNT_ENCODING,
        CountEncodingColumnTransformer(),
        column_name_list
    )



pipeline_preprocessor_X_bank_full = Pipeline(
    steps=[
        (
            STEP_000,
            new_column_transformer(
                [
                    step_object_to_boolean([COLUMN_NAME_Y]),
                ]
            )
        ),
    ]
)
X_bank_full_preprocessed = pipeline_preprocessor_X_bank_full.fit_transform(X_bank_full)
X_bank_full_preprocessed.head()


X = pd.concat(
    [
        X_train,
        X_bank_full_preprocessed
    ],
    axis="rows"
)
y = X.pop(COLUMN_NAME_Y)


preprocessor_pipeline = Pipeline(
    steps=[
        (
            STEP_000,
            new_column_transformer(
                [
                    step_object_to_boolean([COLUMN_NAME_HOUSING, COLUMN_NAME_LOAN, COLUMN_NAME_DEFAULT]),
                    step_split_pdays(),
                    step_log1p([COLUMN_NAME_BALANCE]),
                    step_duration_sincos_encoding(),
                ]
            )
        ),
        (
            STEP_001,
            new_column_transformer(
                [
                    step_count_encoding(
                        [
                            COLUMN_NAME_BALANCE,
                            COLUMN_NAME_DURATION,
                            COLUMN_NAME_HOUSING,
                            COLUMN_NAME_CONTACT,
                            COLUMN_NAME_MONTH,
                            COLUMN_NAME_POUTCOME,
                            COLUMN_NAME_DAY,
                            COLUMN_NAME_FIRST_CONTACT,
                            COLUMN_NAME_DAYS_SINCE_LAST_CONTACT,
                            COLUMN_NAME_AGE,
                            COLUMN_NAME_CAMPAIGN,
                            COLUMN_NAME_PDAYS,
                            COLUMN_NAME_PREVIOUS,
                            COLUMN_NAME_JOB,
                            COLUMN_NAME_MARITAL,
                            COLUMN_NAME_EDUCATION,
                            COLUMN_NAME_DEFAULT,
                            COLUMN_NAME_LOAN,
                        ]
                    ),
                ]
            )
        ),
        (
            STEP_002,
            new_column_transformer(
                [
                    step_combine_columns(
                        column_name_list=[
                            COLUMN_NAME_BALANCE,
                            COLUMN_NAME_DURATION,
                            COLUMN_NAME_HOUSING,
                            COLUMN_NAME_CONTACT,
                            COLUMN_NAME_MONTH,
                            COLUMN_NAME_POUTCOME,
                            COLUMN_NAME_DAY,
                            COLUMN_NAME_FIRST_CONTACT,
                            COLUMN_NAME_DAYS_SINCE_LAST_CONTACT,
                            COLUMN_NAME_AGE,
                            COLUMN_NAME_CAMPAIGN,
                            COLUMN_NAME_PDAYS,
                            COLUMN_NAME_PREVIOUS,
                            COLUMN_NAME_JOB,
                            COLUMN_NAME_MARITAL,
                            COLUMN_NAME_EDUCATION,
                            COLUMN_NAME_DEFAULT,
                            COLUMN_NAME_LOAN,
                        ],
                        combination_length_list=[2]
                    )
                ]
            )
        ),
        (
            STEP_003,
            new_column_transformer(
                [
                    step_drop_columns([COLUMN_NAME_ID]),
                ]
            )
        ),
        (
            STEP_004,
            new_column_transformer(
                [
                    step_target_encoding(),
                ]
            )
        )
    ]
)
X_preprocessed = preprocessor_pipeline.fit_transform(X, y)
X_competition_preprocessed = preprocessor_pipeline.transform(X_competition)
del X_bank_full, X_bank_full_preprocessed, X_train, X
X_preprocessed.head()


xgbc_params = {
    "n_estimators": 1000,
    "objective": "binary:logistic",
    "scale_pos_weight": scale_pos_weight,
    "device": "cuda",
    "seed": SEED,
}

xgbc_model = XGBClassifier(
    **xgbc_params,
)

xgbc_model.fit(X_preprocessed,y)



feature_importance = (
    pd.DataFrame(
        xgbc_model.get_booster().get_score(importance_type="weight"), #Equivalent to xgb.plot_importance().
        index=["weight"]
    )
    .T
    .sort_values(
        by="weight",
        ascending=False
    )
)
feature_importance[:20]


feature_importance_gain = (
    pd.DataFrame(
        xgbc_model.get_booster().get_score(importance_type="gain"),
        index=["gain"]
    )
    .T
    .sort_values(
        by="gain",
        ascending=False
    )
)
feature_importance_gain[:20]


for sorted_idx in xgbc_model.feature_importances_.argsort()[::-1][:20]:
    print(f"{X_preprocessed.columns[sorted_idx]:40} \t {xgbc_model.feature_importances_[sorted_idx]}")


from shap import TreeExplainer, summary_plot

explainer = TreeExplainer(xgbc_model)
shap_values = explainer.shap_values(X_preprocessed)
fig = summary_plot(shap_values, X_preprocessed)


def compute_kfold_score(params, X, y, cv=5):
    # Initialize prediction containers
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(X_competition_preprocessed))

    # Set up cross-validation
    N_FOLDS = cv
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        # Prepare fold data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train the model
        xgbc_model = XGBClassifier(
            **params
        )
        xgbc_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=0,
        )

        # Generate predictions
        fold_oof_pred = xgbc_model.predict_proba(X_val, iteration_range=(0, xgbc_model.best_iteration + 1))[:, 1]
        fold_test_pred = xgbc_model.predict_proba(X_competition_preprocessed, iteration_range=(0, xgbc_model.best_iteration + 1))[:, 1]

        # Store predictions
        oof_predictions[val_idx] = fold_oof_pred
        test_predictions += fold_test_pred / N_FOLDS

        # Calculate fold score
        fold_score = roc_auc_score(y_val, fold_oof_pred)
        fold_scores.append(fold_score)
        print(f"\t AUC={fold_score}")

    # Calculate overall score
    score = roc_auc_score(y, oof_predictions)
    print(f"Overall CV AUC: {score:.6f}")
    print(f"Standard deviation: {np.std(fold_scores):.6f}")
    print(f"Fold scores: {[f'{score:.6f}' for score in fold_scores]}")
    return score, test_predictions


xgbc_params = {
    'n_estimators': 10000,
    "early_stopping_rounds": 200,
    "objective": "binary:logistic",  # Binary classification
    "eval_metric": "auc",  # ROC AUC evaluation
    "learning_rate": 0.1,  # Conservative learning rate
    "max_depth": 0,  # Use max_leaves instead
    "subsample": 0.8,  # Row sampling for regularization
    "colsample_bytree": 0.7,  # Column sampling for regularization
    "seed": SEED,  # Reproducibility
    "device": "cuda",  # GPU acceleration
    "grow_policy": "lossguide",  # Leaf-wise tree growth
    "max_leaves": 32,  # Control tree complexity
    "alpha": 2.0,  # L1 regularization
}

score, test_predictions = compute_kfold_score(xgbc_params, X_preprocessed, y, cv=5)


xgb_submission = pd.DataFrame({"id": X_competition.id, "y": test_predictions})
xgb_submission.to_csv('xgb_submission_baseline.csv', index=False)
xgb_submission.head()


#End notebook here.
assert False


trial_predictions = dict()

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 10000),
        "max_depth": 0,
        "max_leaves": trial.suggest_int("max_leaves", 32, 128),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "alpha": trial.suggest_int("reg_alpha", 1, 10),
        "early_stopping_rounds": 200,
        "objective": "binary:logistic",
        "scale_pos_weight": scale_pos_weight,
        "device": "cuda",
        "eval_metric": "auc",
        "seed": SEED,
    }

    score, test_predictions = compute_kfold_score(params, X_preprocessed, y, cv=5)
    trial_predictions[trial.number] = test_predictions
    return score


xgbc_study_name = "XGBClassifier"
xgbc_study_url = f"sqlite:///{xgbc_study_name}.db"
storage = optuna.storages.RDBStorage(xgbc_study_url, engine_kwargs={"connect_args": {"timeout": 30.0}})

study_xgbc = optuna.create_study(
    study_name=xgbc_study_name,
    direction="maximize",
    storage=storage,
    load_if_exists=True
)

study_xgbc.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=4, gc_after_trial=True)
print(f"\nBest parameters for {xgbc_study_name}: {study_xgbc.best_params}")


xgb_submission = pd.DataFrame({"id": X_competition.id, "y": trial_predictions[study_xgbc.best_trial.number]})
xgb_submission.to_csv('xgb_submission_tuned.csv', index=False)
xgb_submission.head()


