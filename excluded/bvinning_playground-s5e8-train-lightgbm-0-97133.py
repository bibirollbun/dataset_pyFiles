!pip install pandera==0.25 -q


from pathlib import Path
from typing import Final


DATA_DIR: Final = Path("/kaggle/input/playground-series-s5e8")
TRAIN_DATA_PATH: Final = DATA_DIR / "train.csv"
TEST_DATA_PATH: Final = DATA_DIR / "test.csv"
OUTPUT_DIR: Final = Path("/kaggle/working/")


for path in [DATA_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH]:
    assert path.exists()    


from enum import StrEnum, auto


class Job(StrEnum):
    STUDENT = "student"
    HOUSEMAID = "housemaid"
    UNEMPLOYED = "unemployed"
    ENTREPRENEUR = "entrepreneur"
    SELF_EMPLOYED = "self-employed"
    RETIRED = "retired"
    SERVICES = "services"
    ADMIN = "admin."
    TECHNICIAN = "technician"
    BLUE_COLLAR = "blue-collar"
    MANAGEMENT = "management"


class Marital(StrEnum):
    MARRIED = auto()
    SINGLE = auto()
    DIVORCED = auto()


class Education(StrEnum):
    PRIMARY = auto()
    SECONDARY = auto()
    TERTIARY = auto()


class Month(StrEnum):
    JAN = auto()
    FEB = auto()
    MAR = auto()
    APR = auto()
    MAY = auto()
    JUN = auto()
    JUL = auto()
    AUG = auto()
    SEP = auto()
    OCT = auto()
    NOV = auto()
    DEC = auto()

    def encode(self) -> int:
        return _MONTH_TO_ENCODINGS[self]


_MONTH_TO_ENCODINGS: Final[dict[Month, int]] = {
    Month.JAN: 1,
    Month.FEB: 2,
    Month.MAR: 3,
    Month.APR: 4,
    Month.MAY: 5,
    Month.JUN: 6,
    Month.JUL: 7,
    Month.AUG: 8,
    Month.SEP: 9,
    Month.OCT: 10,
    Month.NOV: 11,
    Month.DEC: 12
}


class POutcome(StrEnum):
    SUCCESS = auto()
    FAILURE = auto()
    OTHER = auto()


from typing import Optional, Annotated
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series
from pandera.dtypes import Category
import kagglehub
from kagglehub import KaggleDatasetAdapter

class BankSchema(pa.DataFrameModel):
    id: Optional[Series[int]] = pa.Field()
    age: Series[int] = pa.Field(ge=0, le=100, description="Age of the client")
    job: Series[Annotated[Category, Job, False]] = pa.Field(nullable=True, description="Type of job")
    marital: Series[Annotated[Category, Marital, False]] = pa.Field(description="Marital status")
    education: Series[Annotated[Category, Education, True]] = pa.Field(nullable=True, description="Level of education")
    default: Series[bool] = pa.Field(description="Defaulted")
    balance: Series[int] = pa.Field(description="Average yearly balance in euros.")
    housing: Series[bool] = pa.Field(description="Has a housing loan?")
    loan: Series[bool] = pa.Field(description="Has a personal loan? ")
    contact: Series[Category] = pa.Field(nullable=True, description="Type of communication contact")
    day: Series[int] = pa.Field(ge=0, le=31, description="Last contact day of the month")
    month: Series[Annotated[Category, Month, True]] = pa.Field(description="Last contact month of the year")
    duration: Series[int] = pa.Field(ge=0, description="Last contact duration in seconds")
    campaign: Series[int] = pa.Field(ge=0, description="Number of contacts performed during this campaign")
    pdays: Series[int] = pa.Field(description="Number of days since the client was last contacted from a previous campaign")
    previous: Series[int] = pa.Field(description="Number of contacts performed before this campaign")
    poutcome: Series[Annotated[Category, POutcome, False]] = pa.Field(nullable=True, description="Outcome of the previous marketing campaign")

    y: Optional[Series[bool]]

    class Config:
        coerce = True
        strict = True


def load_data(*, train: bool, load_extra_train: bool) -> DataFrame[BankSchema]:
    path = TRAIN_DATA_PATH if train else TEST_DATA_PATH
    df = pd.read_csv(
        path,
        index_col=BankSchema.id,
        true_values=["yes"],
        false_values=["no"],
        na_values=["unknown"],
    )
    if train and load_extra_train:
        df_extra = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "sushant097/bank-marketing-dataset-full",
            "bank-full.csv",
            pandas_kwargs={
                'true_values': ["yes"],
                'false_values': ["no"],
                'na_values': ["unknown"],
                'sep': ";"
            }
        )
        df_extra.index.name = BankSchema.id
        df = pd.concat([df, df_extra]).reset_index(drop=True)
        df.index.name = BankSchema.id
    
    BankSchema.validate(df)
    return DataFrame[BankSchema](df)


train_df = load_data(train=True, load_extra_train=True)
train_df


test_df = load_data(train=False, load_extra_train=True)
test_df


import datetime
import numpy.typing as npt
import numpy as np
import holidays

_DAYS_IN_WEEK: Final = 7
_DAYS_IN_YEAR: Final = 365
_CURRENT_YEAR: Final = datetime.date.today().year



def _get_last_contact_date(df: DataFrame[BankSchema]) -> pd.Series:

    def _to_datetime(x: pd.Series) -> datetime.date:
        month = x[BankSchema.month]
        day = x[BankSchema.day]
    
        try:
            return datetime.date(year=_CURRENT_YEAR, month=month, day=day)
        except ValueError as e:
            return datetime.date(year=_CURRENT_YEAR, month=month + 1 if month < 12 else 1, day=1)

    return pd.to_datetime(
        df[BankSchema.month]
        .map(Month.encode)
        .to_frame()
        .join(df[BankSchema.day])
        .apply(_to_datetime, axis=1)
    )


def _get_calendar_features(s: pd.Series) -> pd.DataFrame:
    day_of_week = s.dt.day_of_week
    day_of_year = s.dt.day_of_year
    day_of_month = s.dt.day
    days_in_month = s.dt.days_in_month

    pt_holidays = holidays.country_holidays("PT", years=_CURRENT_YEAR)

    return pd.DataFrame(
        {
            "day_of_week_sin": np.sin(2 * np.pi * day_of_week / _DAYS_IN_WEEK),
            "day_of_week_cos": np.cos(2 * np.pi * day_of_week / _DAYS_IN_WEEK),
            "day_in_year_sin": np.sin(2 * np.pi * day_of_year / _DAYS_IN_YEAR),
            "day_in_year_cos": np.cos(2 * np.pi * day_of_year / _DAYS_IN_YEAR),
            "day_in_month_sin": np.sin(2 * np.pi * day_of_month / days_in_month),
            "day_in_month_cos": np.cos(2 * np.pi * day_of_month / days_in_month),
            "quarter": s.dt.quarter,
            "is_weekend": day_of_week.isin([5, 6]),
            "is_holiday": s.isin(list(pt_holidays))
        }
    )


def _get_last_contact_calendar_features(df: DataFrame[BankSchema]) -> pd.Series:
    last_contact_date_s = _get_last_contact_date(df)
    return _get_calendar_features(last_contact_date_s)


def get_X_y(df: DataFrame[BankSchema]) -> tuple[pd.DataFrame, pd.Series]:
    calendar_features_df = _get_last_contact_calendar_features(df)
    sub_df = df[
        [
            BankSchema.age,
            BankSchema.job,
            BankSchema.marital,
            BankSchema.education,
            BankSchema.default,
            BankSchema.balance,
            BankSchema.housing,
            BankSchema.loan,
            BankSchema.contact,
            BankSchema.duration,
            BankSchema.campaign,
            BankSchema.pdays,
            BankSchema.previous,
            BankSchema.poutcome,
        ]
    ]

    feature_df = sub_df.join(calendar_features_df)
    feature_df['campaign_intensity'] = feature_df[BankSchema.campaign] / (feature_df[BankSchema.pdays].replace(-1, np.nan).fillna(999) + 1)
    feature_df['avg_balance_per_contact'] = feature_df[BankSchema.balance] / (feature_df[BankSchema.previous] + 1)
    feature_df['duration_per_campaign'] = feature_df[BankSchema.duration] / (feature_df[BankSchema.campaign] + 1)

    target_df = df[BankSchema.y] if BankSchema.y in df.columns else None

    return feature_df, target_df


X, y = get_X_y(train_df)
X


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)


import lightgbm as lgb

MAX_BIN = 512

dtrain = lgb.Dataset(X_train, label=y_train, params={"max_bin": MAX_BIN})
dvalid = lgb.Dataset(X_test, label=y_test, params={"max_bin": MAX_BIN})


import optuna
from sklearn.model_selection import StratifiedKFold

TUNE = True
NUM_TUNING_TRIALS = 200
NUM_BOOSTING_ROUNDS = 2_000
FIXED_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'is_unbalance': True,
    "boosting": "gbdt"
}

CACHED_BEST_PARAMS = {
    'learning_rate': 0.024566839110977196,
    'num_leaves': 482,
    'lambda_l1': 0.10247609638866081,
    'lambda_l2': 346.0173406103292,
    'feature_fraction': 0.7157149486478368,
    'bagging_fraction': 0.9265889760905258,
    'bagging_freq': 0
}

def objective(trial):
    param = {
        #'boosting': trial.suggest_categorical('boosting', ["gbdt", "dart"]),

        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.5, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 512),

        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1e8, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1e8, log=True),

        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 7),

        #'max_bin': trial.suggest_int('max_bin', 128, 1024),
        
        **FIXED_PARAMS
    }

    bst = lgb.train(
        param,
        dtrain,
        NUM_BOOSTING_ROUNDS, 
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(stopping_rounds=5)]
    )
    return bst.best_score["valid_0"]["auc"]


def get_best_params(*, tune: bool) -> dict:
    if tune:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=NUM_TUNING_TRIALS)
        print(study.best_trial.params)
        return study.best_trial.params | FIXED_PARAMS
    return CACHED_BEST_PARAMS | FIXED_PARAMS


def train_and_predict(X, y, X_test) -> pd.Series:
    param = get_best_params(tune=TUNE)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    y_probas = []
    y_proba_weights = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
        dtrain = lgb.Dataset(X_train, label=y_train, params={"max_bin": MAX_BIN})
        dvalid = lgb.Dataset(X_val, label=y_val, params={"max_bin": MAX_BIN})
    
        bst = lgb.train(
            param,
            dtrain,
            NUM_BOOSTING_ROUNDS, 
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(stopping_rounds=5)]
        )
        y_proba = bst.predict(X_test)
        y_probas.append(y_proba)
        y_proba_weights.append(bst.best_score["valid_0"]["auc"])

    return pd.Series(
        np.average(y_probas, axis=0, weights=y_proba_weights),
        index=X_test.index,
        name=BankSchema.y
    )


X_test, _ = get_X_y(test_df)
y_proba = train_and_predict(X, y, X_test)
y_proba


SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
y_proba.to_csv(SUBMISSION_PATH)
!head $SUBMISSION_PATH

