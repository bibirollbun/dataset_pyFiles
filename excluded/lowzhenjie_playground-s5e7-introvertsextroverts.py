# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


print(pd.__version__)


dtype_dict = {'id': pd.UInt32Dtype(),
              'Time_spent_Alone': pd.UInt8Dtype(),
              'Stage_fear': 'object',
              'Social_event_attendance': pd.UInt8Dtype(),
              'Going_outside': pd.UInt8Dtype(),
              'Drained_after_socializing': 'object',
              'Friends_circle_size': pd.UInt8Dtype(),
              'Post_frequency': pd.UInt8Dtype(),
              'Personality': 'object'}


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', 
                 dtype=dtype_dict)

for col in ['Stage_fear', 'Drained_after_socializing']:
    df[col] = df[col].map({'No': False, 'Yes': True}).astype(pd.BooleanDtype())

df['Personality'] = df['Personality'].map({'Extrovert': False, 'Introvert': True}).astype(pd.BooleanDtype())

df.info()


# About 45% of data points are lost if I try to drop all rows with any null values, so I shouldn't do that
len(df.dropna(how='any',axis=0))


def plot_hist_with_na(dataframe, x_col, color_col, extra_col):
    vc = dataframe[[extra_col, x_col, color_col]].groupby([x_col, color_col], dropna=False)
    vc = vc.agg('count').reset_index()
    vc[x_col] = vc[x_col].astype('string')
    vc[x_col] = vc[x_col].map(lambda x: 'NA' if pd.isnull(x) else x)
    fig = px.bar(vc, x=x_col, y=extra_col, color=color_col, width=600, height=400)
    return fig


fig = plot_hist_with_na(df, 'Time_spent_Alone', 'Personality', 'id')
fig.show(renderer='iframe_connected')


fig = plot_hist_with_na(df, 'Stage_fear', 'Personality', 'id')
fig.show(renderer='iframe_connected')


fig = plot_hist_with_na(df, 'Social_event_attendance', 'Personality', 'id')
fig.show(renderer='iframe_connected')


fig = plot_hist_with_na(df, 'Going_outside', 'Personality', 'id')
fig.show(renderer='iframe_connected')


fig = plot_hist_with_na(df, 'Drained_after_socializing', 'Personality', 'id')
fig.show(renderer='iframe_connected')


fig = plot_hist_with_na(df, 'Friends_circle_size', 'Personality', 'id')
fig.show(renderer='iframe_connected')


fig = plot_hist_with_na(df, 'Post_frequency', 'Personality', 'id')
fig.show(renderer='iframe_connected')


df['Personality'].value_counts(dropna=False)


import optuna
print('optuna', optuna.__version__)
import lightgbm as lgb
print('lightgbm', lgb.__version__)

from sklearn.utils.random import sample_without_replacement
from sklearn.utils import shuffle

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score


rs = 42
n_trials = 100
current_best_threshold = 0.5
final_threshold = 0.5


# # https://www.kaggle.com/code/ttahara/example-of-lgbm-custom-metric
# def binary_accuracy_for_lgbm(
#     preds: np.ndarray, data: lgb.Dataset, threshold: float=0.5,
# ) -> tuple[str, float, bool]:
#     """Calculate Binary Accuracy"""
#     label = data.get_label()
#     weight = data.get_weight()
#     pred_label = (preds > threshold).astype(int)
#     acc = np.average(label == pred_label, weights=weight)

#     # # eval_name, eval_result, is_higher_better
#     return 'my_bin_acc', acc, True


def choose_best_threshold(y_true, y_pred_possibility):
    def score(threshold_value, y_true=y_true, y_pred_possibility=y_pred_possibility):
        binaries = np.vectorize(lambda x: False if x <= threshold_value else True)(y_pred_possibility)
        score = accuracy_score(y_true, binaries)
        return score
    test_threshold = pd.DataFrame({'threshold_values': np.linspace(0,1,1001)})
    test_threshold['score'] = test_threshold['threshold_values'].map(score)
    best_score = test_threshold['score'].max()
    filtered = test_threshold[test_threshold['score']==best_score]
    best_threshold = filtered['threshold_values'].unique()[0]
    return best_threshold, best_score

static_params = {
    'objective': 'binary', 
    'metric': 'binary_error',
    "verbosity": -1,
    "boosting_type": "gbdt",
    "random_state": rs,
    "saved_feature_importance_type": 1
    }


# Turn off optuna log notes.
optuna.logging.set_verbosity(optuna.logging.WARN)

def objective(trial):
    param = {
        "learning_rate": trial.suggest_float("learning_rate", 0.08, 0.16, step=0.02),
        "lambda_l1": 10**trial.suggest_float("lambda_l1", -9, -2, step=0.5),
        "lambda_l2": 10**trial.suggest_float("lambda_l2", -9, -2, step=0.5),
        "num_leaves": trial.suggest_int("num_leaves", 80, 960, step=80),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0, step=0.1),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0, step=0.1),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 100, step=10),
        "max_depth": trial.suggest_int("max_depth", 8, 12, step=2)
    }

    param = {**static_params, **param}

    X = df.drop(columns=['id', 'Personality'])
    y = df['Personality']
    
    accuracy_scores = []
    best_threshold_values = []

    for train_index, test_index in StratifiedKFold(n_splits=3, shuffle=True, random_state=rs).split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_test, label=y_test)
    
        gbm = lgb.train(param, dtrain, valid_sets=dvalid)
        preds = gbm.predict(X_test)
        threshold, score = choose_best_threshold(y_test, preds)

        best_threshold_values.append(threshold)
        accuracy_scores.append(score)

    current_best_threshold = np.mean(best_threshold_values)
    return np.mean(accuracy_scores)

def logging_callback(study, frozen_trial):
    # This suppresses logging of trials that do not result in better values
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)

        frozen_trial.params["lambda_l1"] = 10**frozen_trial.params["lambda_l1"]
        frozen_trial.params["lambda_l2"] = 10**frozen_trial.params["lambda_l2"]
        final_threshold = current_best_threshold
        print(
            "Trial {} finished with best value: {} and parameters: {}. ".format(
            frozen_trial.number,
            frozen_trial.value,
            frozen_trial.params,
            )
        )
    else:
        print(f"Trial {frozen_trial.number} did not result in a better value.          ", end='\r')


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=rs))
    study.optimize(objective, n_trials=n_trials, callbacks=[logging_callback])
    
    best_params = {key: value for key, value in study.best_trial.params.items()}
    best_params["lambda_l1"] = 10**best_params["lambda_l1"]
    best_params["lambda_l2"] = 10**best_params["lambda_l2"]
    
    print('                                                               ')
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Threshold: {final_threshold}")
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")


testdf = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', 
                     dtype={k:v for k,v in dtype_dict.items() if k != 'Personality'})

for col in ['Stage_fear', 'Drained_after_socializing']:
    testdf[col] = testdf[col].map({'No': False, 'Yes': True}).astype(pd.BooleanDtype())


model_params = static_params | best_params
print(model_params)

X = df.drop(columns=['id', 'Personality'])
y = df['Personality']

gbm = lgb.train(model_params, lgb.Dataset(X, label=y))

feature_impt = pd.DataFrame({"Feature Name": gbm.feature_name(),
                             "Feature Importance (Gain)": gbm.feature_importance(importance_type='gain', iteration=None)})
feature_impt.sort_values("Feature Importance (Gain)", ascending=False).reset_index(drop=True)


preds = gbm.predict(testdf.drop(columns='id'))
preds_binary = np.vectorize(lambda x: 'Extrovert' if x <= final_threshold else 'Introvert')(preds)

predictions_df = pd.DataFrame(testdf['id'])
predictions_df['Personality'] = preds_binary

predictions_df.to_csv('/kaggle/working/submission.csv', index=False)

