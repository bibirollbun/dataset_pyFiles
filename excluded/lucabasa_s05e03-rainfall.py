!pip install tubesml


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
%matplotlib inline

import tubesml as tml

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)  # I will let you know, pandas, when I have performance issues


df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

df.head()


_ = tml.list_missing(df)
print("_"*40)
_ = tml.list_missing(df_test)


df.describe()


df_test.describe()


df.hist(bins=20, figsize=(15,10), grid=False)
plt.show()


num_cor = tml.plot_correlations(data=df.select_dtypes('number'), target="rainfall", annot=True)
num_cor


for col in num_cor.index[1:]:
    tml.segm_target(data=df, cat="rainfall", target=col)


def plot_frame(ax):
    ax.set_facecolor('#292525')
    ax.spines['bottom'].set_color('w')
    ax.tick_params(axis='x', colors='w')
    ax.xaxis.label.set_color('w')
    ax.spines['left'].set_color('w')
    ax.tick_params(axis='y', colors='w')
    ax.yaxis.label.set_color('w')
    return ax

def plot_time_series(data, column):
    fig = plt.figure(figsize=(15, 10), facecolor='#292525')
    fig.subplots_adjust(top=0.90)
    fig.suptitle(f"{col} over time", fontsize=18, color='w')

    gs = GridSpec(2, 4, figure=fig)
    ax0 = fig.add_subplot(gs[0, :3])
    ax1 = fig.add_subplot(gs[0, 3])
    ax2 = fig.add_subplot(gs[1, :3])
    ax3 = fig.add_subplot(gs[1, 3])

    data[data["rainfall"] == 0].set_index("id")[col].plot(ax=ax0, color='#15E498').grid(axis="y")
    sns.boxplot(data=data[data["rainfall"] == 0], y=col, ax=ax1, color='#15E498')
    data[data["rainfall"] == 1].set_index("id")[col].plot(ax=ax2, color='#C3C92E').grid(axis="y")
    sns.boxplot(data=data[data["rainfall"] == 1], y=col, ax=ax3, color='#C3C92E')
    
    ax0.set_title('No Rain', fontsize=14, color='w')
    ax0.set_ylabel(f'{col}', fontsize=12)
    ax2.set_title('Rain', fontsize=14, color='w')
    ax2.set_ylabel(f'{col}', fontsize=12)

    for ax in [ax0, ax1, ax2, ax3]:
        ax = plot_frame(ax)
        ax.set_xlabel('')


for col in ["pressure", "temparature", "humidity", "cloud", "sunshine"]:
    plot_time_series(df, col)


NFOLDS = 5
TEST_SIZE = 200

tfolds = TimeSeriesSplit(n_splits=NFOLDS, test_size=TEST_SIZE)


for i, (train_index, test_index) in enumerate(tfolds.split(df)):
    print(f"Fold {i}:")
    trn_data = df.iloc[train_index, :]
    val_data = df.iloc[test_index, :]
    print(f"  Train: id={trn_data['id'].min(), trn_data['id'].max()}")
    print(f"  Test:  id={val_data['id'].min(), val_data['id'].max()}")

    mean_pred = trn_data["rainfall"].mean()
    pred = [mean_pred] * len(val_data)
    print(roc_auc_score(y_true=val_data["rainfall"], y_score=pred))


training_cols = [c for c in df if c not in ["id", "rainfall"]]
target = df["rainfall"]

training_cols


processing_pipe = Pipeline([("imputer", tml.DfImputer(strategy="mean")),
                            ("scaler", tml.DfScaler(method="standard"))])


model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', LogisticRegression(random_state=43))])


oof, res = tml.cv_score(data=df[training_cols], target=target, estimator=model_pipe, cv=tfolds, imp_coef=True, predict_proba=True)

print(round(roc_auc_score(y_true=target[-TEST_SIZE*NFOLDS:], y_score=oof[-TEST_SIZE*NFOLDS:]), 3))

tml.plot_feat_imp(res['feat_imp'])

tml.plot_classification_probs(data=df.tail(TEST_SIZE*NFOLDS), true_label=target[-TEST_SIZE*NFOLDS:], pred_label=oof[-TEST_SIZE*NFOLDS:], thrs=0.5, feat="id")


model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', DecisionTreeClassifier(max_depth=3, random_state=44))])


oof, res = tml.cv_score(data=df[training_cols], target=target, estimator=model_pipe, cv=tfolds, imp_coef=True, predict_proba=True)

print(round(roc_auc_score(y_true=target[-TEST_SIZE*NFOLDS:], y_score=oof[-TEST_SIZE*NFOLDS:]), 3))

tml.plot_feat_imp(res['feat_imp'])

tml.plot_classification_probs(data=df.tail(TEST_SIZE*NFOLDS), true_label=target[-TEST_SIZE*NFOLDS:], pred_label=oof[-TEST_SIZE*NFOLDS:], thrs=0.5, feat="id")


model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', RandomForestClassifier(n_estimators=1000,
                                                        max_depth=5,
                                                        max_features="sqrt",
                                                        random_state=44))])


oof, res = tml.cv_score(data=df[training_cols], target=target,
                        estimator=model_pipe, cv=tfolds, imp_coef=True, predict_proba=True)

print(round(roc_auc_score(y_true=target[-TEST_SIZE*NFOLDS:], y_score=oof[-TEST_SIZE*NFOLDS:]), 3))

tml.plot_feat_imp(res['feat_imp'])

tml.plot_classification_probs(data=df.tail(TEST_SIZE*NFOLDS), true_label=target[-TEST_SIZE*NFOLDS:], pred_label=oof[-TEST_SIZE*NFOLDS:], thrs=0.5, feat="id")


model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', ExtraTreesClassifier(max_depth=5,
                                                      n_estimators=1000,
                                                      max_features="sqrt",
                                                      random_state=44))])


oof, res = tml.cv_score(data=df[training_cols], target=target, estimator=model_pipe, cv=tfolds, imp_coef=True, predict_proba=True)

print(round(roc_auc_score(y_true=target[-TEST_SIZE*NFOLDS:], y_score=oof[-TEST_SIZE*NFOLDS:]), 3))

tml.plot_feat_imp(res['feat_imp'])

tml.plot_classification_probs(data=df.tail(TEST_SIZE*NFOLDS), true_label=target[-TEST_SIZE*NFOLDS:], pred_label=oof[-TEST_SIZE*NFOLDS:], thrs=0.5, feat="id")


model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', xgb.XGBClassifier(n_estimators=10000,
                                                   max_depth=3,
                                                   subsample=0.7,
                                                   random_state=132,
                                                   early_stopping_rounds=100))])

fit_params = {'verbose': False}


oof, res = tml.cv_score(data=df[training_cols], target=target, estimator=model_pipe, cv=tfolds,
                        imp_coef=True, predict_proba=True, early_stopping=True, fit_params=fit_params)

print(res["iterations"])
print(round(roc_auc_score(y_true=target[-TEST_SIZE*NFOLDS:], y_score=oof[-TEST_SIZE*NFOLDS:]), 3))

tml.plot_feat_imp(res['feat_imp'])

tml.plot_classification_probs(data=df.tail(TEST_SIZE*NFOLDS), true_label=target[-TEST_SIZE*NFOLDS:], pred_label=oof[-TEST_SIZE*NFOLDS:], thrs=0.5, feat="id")


model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', lgb.LGBMClassifier(n_estimators=10000,
                                                    num_leaves=3,
                                                    subsample=0.8,
                                                    colsample__bytree=0.7,
                                                    verbose=-1,
                                                    random_state=132))])

callbacks = [lgb.early_stopping(100, verbose=0)]
fit_params = {"callbacks":callbacks}

oof, res = tml.cv_score(data=df[training_cols], target=target, estimator=model_pipe, cv=tfolds,
                        imp_coef=True, predict_proba=True, early_stopping=True, fit_params=fit_params)

print(res["iterations"])
print(round(roc_auc_score(y_true=target[-TEST_SIZE*NFOLDS:], y_score=oof[-TEST_SIZE*NFOLDS:]), 3))

tml.plot_feat_imp(res['feat_imp'])

tml.plot_classification_probs(data=df.tail(TEST_SIZE*NFOLDS), true_label=target[-TEST_SIZE*NFOLDS:], pred_label=oof[-TEST_SIZE*NFOLDS:], thrs=0.5, feat="id")


tmp = df.copy()
new_cols = []
for lag in [1, 2, 3]:
    for col in ["rainfall"] + training_cols:
        tmp[f"{col}_lag{lag}"] = tmp[f"{col}"].shift(lag).fillna(tmp[f"{col}"])
        if lag == 1:
            new_cols.append(f"{col}_lag{lag}")
        if (col != "rainfall") and (lag == 1):
            tmp[f"{col}_diff"] = tmp[col].diff()
            tmp[f"{col}_trend3"] = tmp[col].diff(3)
            new_cols += [f"{col}_diff", f"{col}_trend3"]

for col in ["rainfall"] + training_cols:
    tmp[f"{col}_mean_lag"] = tmp[[f"{col}_lag{n}" for n in [1,2,3]]].mean(axis=1)
    tmp[f"{col}_max_lag"] = tmp[[f"{col}_lag{n}" for n in [1,2,3]]].max(axis=1)
    tmp[f"{col}_min_lag"] = tmp[[f"{col}_lag{n}" for n in [1,2,3]]].min(axis=1)
    tmp[f"{col}_std_lag"] = tmp[[f"{col}_lag{n}" for n in [1,2,3]]].std(axis=1)
    new_cols += [f"{col}_mean_lag", f"{col}_max_lag", f"{col}_min_lag", f"{col}_std_lag"]

tmp = tmp.drop([c for c in tmp if "lag2" in c or "lag3" in c], axis=1)

pd.crosstab(tmp["rainfall"], tmp["rainfall_lag1"])


tmp["day_sin"] = np.sin(2 * np.pi * tmp["day"] / 365)
tmp["day_cos"] = np.cos(2 * np.pi * tmp["day"] / 365)

tmp["sunshine_cloud"] = np.where(tmp["cloud"] == 0, tmp["sunshine"] / tmp[tmp["cloud"] > 0]["cloud"].min(), tmp['sunshine'] / tmp['cloud'])
tmp["temp_range"] = tmp["maxtemp"] - tmp["mintemp"]

tmp["dewpoint_depression"] = tmp["temparature"] - tmp["dewpoint"]
tmp['temp_humidity_index'] = (0.8 * tmp['temparature']) + \
                                        ((tmp['humidity'] / 100) * \
                                        (tmp['temparature'] - 14.3)) + 46.4

new_cols += ["day_sin", "day_cos", "sunshine_cloud", "temp_range", "dewpoint_depression", 'temp_humidity_index']


num_cor = tml.plot_correlations(data=tmp[new_cols + ["rainfall"]], target="rainfall")
num_cor


for col in num_cor.index[1:10]:
    tml.segm_target(data=tmp, cat="rainfall", target=col)


processing_pipe = Pipeline([("imputer", tml.DfImputer(strategy="mean")),
                            ("scaler", tml.DfScaler(method="standard"))])

model_pipe = Pipeline([('processing', processing_pipe),
                       ('model', LogisticRegression(random_state=43))])

model_pipe.fit(df[training_cols], df["rainfall"])
predictions = model_pipe.predict_proba(df_test[training_cols])[:,1]


sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub["rainfall"] = predictions

sub.to_csv("submission.csv", index=False)
sub.head()




