%%time

!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


%%time

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


def add_features(df):
    sex_match = df.sex_match.astype(str)
    sex_match = sex_match.str.split("-").str[0] == sex_match.str.split("-").str[1]
    df['sex_match_bool'] = sex_match.astype("object")
    return df

from lifelines import KaplanMeierFitter


def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    df_ = df.loc[(df[time_col] < 24) | (df[event_col] == 0)]
    kmf.fit(df_[time_col], df_[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y

from metric import score as score_f

def run(p):
    FOLDS, oof_xgb, pred_xgb, train = make_predictions(p)

    pred_xgb /= FOLDS

    y_true = train[["ID", "efs", "efs_time", "race_group"]].copy()
    y_pred = train[["ID"]].copy()
    y_pred["prediction"] = oof_xgb
    m = score_f(train.copy(), y_pred.copy(), "ID")
    print(f"\nOverall CV for XGBoost KaplanMeier =", m)
    return pred_xgb


def make_predictions(p):
    hparams = dict(
        eps=2e-2,
        eps_mul=1.01,
        pos_shift=0.2
    )
    FEATURES, test, train, y = prepare_data(hparams.pop('eps'), hparams.pop('eps_mul'))
    FOLDS = 5
    kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    oof_xgb = np.zeros(len(train))
    pred_xgb = np.zeros(len(test))
    pos_shift = hparams.pop('pos_shift')
    for i, (train_index, test_index) in enumerate(kf.split(train, train.race_group)):
        print("#" * 25)
        print(f"### Fold {i + 1}")
        print("#" * 25)
        x_test, x_train, x_valid, y_train, y_valid = _prepare_training_data(
            FEATURES, test.copy(), test_index, train.copy(), train_index, y.copy(), pos_shift=pos_shift
        )
        model_xgb = XGBRegressor(
            **p,
            max_depth=5,
            colsample_bytree=0.5,
            subsample=0.8,
            n_estimators=3000,
            learning_rate=0.013440231020414639,
            min_child_weight=40,
            gamma=1,
            eta=0.0,
            reg_lambda=0.004544167887144968,
            reg_alpha=0.1,
            objective='reg:pseudohubererror',
            max_cat_to_onehot=10,
            device="cuda",
            enable_categorical=True,
            random_state=42,
            monotone_constraints={
                'hla_high_res_6': -1,
                'hla_high_res_8': -1,
                'hla_low_res_6': -1,
                'hla_match_a_high': -1,
                'hla_match_drb1_low': -1,
                'hla_match_c_low': -1,
                'hla_match_dqb1_low': -1,
                'hla_nmdp_6': -1,
            }
        )
        tt = train.loc[train_index]

        array = np.array([.95 if x else 1 for x in ((tt.efs_time < 24) & (tt.efs == 0)).values])
        array /= np.array([1.3 if x else 1 for x in tt.efs_time > 36])
        model_xgb.fit(
            x_train, y_train,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            verbose=False,
            sample_weight=array
        )

        oof_xgb[test_index] = model_xgb.predict(x_valid)
        pred_xgb += model_xgb.predict(x_test)
    return FOLDS, oof_xgb, pred_xgb, train


def _prepare_training_data(FEATURES, test, test_index, train, train_index, y, pos_shift=0.1):
    y[train.efs == 0] = y[train.efs == 1].min() - pos_shift
    std = np.std(y[train_index])
    x_train = train.loc[train_index, FEATURES].copy()
    y_train = y[train_index] / std
    x_valid = train.loc[test_index, FEATURES].copy()
    y_valid = y[test_index] / std
    x_test = test[FEATURES].copy()

    le = LabelEncoder()
    val = train.loc[test_index]
    return x_test, x_train, x_valid, y_train, y_valid

def prepare_data(eps=2e-2, eps_mul=1.1):
    test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
    test = add_features(test)
    train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
    train = add_features(train)
    train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')
    RMV = ["ID", "efs", "efs_time", "y"]
    FEATURES = [c for c in train.columns if not c in RMV]

    CATS = []
    for c in FEATURES:
        if train[c].dtype == "object":
            CATS.append(c)
            train[c] = train[c].fillna("NAN")
            test[c] = test[c].fillna("NAN")
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    for c in FEATURES:

        if c in CATS:
            combined[c], _ = combined[c].factorize()
            combined[c] -= combined[c].min()
            combined[c] = combined[c].astype("int32")
            combined[c] = combined[c].astype("category")

        else:
            if combined[c].dtype == "float64":
                combined[c] = combined[c].astype("float32")
            if combined[c].dtype == "int64":
                combined[c] = combined[c].astype("int32")
    train = combined.iloc[:len(train)].copy()
    test = combined.iloc[len(train):].reset_index(drop=True).copy()
    y = train.y.copy()
    y = (y - y.min() + eps) / (y.max() - y.min() + eps_mul * eps)
    y = np.log(y / (1 - y))
    return FEATURES, test, train, y


%%time

p = {'n_jobs': -1}
pred_xgb = run(p) # CV : 0.6821388742966566


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = pred_xgb
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()

