# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import datetime as dt
import holidays
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_log_error
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    RepeatedKFold,
    cross_val_score,
    train_test_split,
)
import shap

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

SEED = 0
plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("fast")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", parse_dates=['date']).drop(columns = "id")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", parse_dates=['date']).drop(columns = "id")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


print("train Data", train_df.info(),"\n\n**********************\n")
print("test Data", test_df.info(),"\n\n**********************")


train_df.isna().sum()


plt.figure(figsize=(12, 4))
ax = sns.lineplot(
    data=train_df,
    x="date",
    y="num_sold",
    errorbar=None,
    linewidth=0.4
)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Number Sold", fontsize=10)
ax.tick_params(axis="both", labelsize=8)
plt.title("Number Sold | Years", size=10)
plt.show()


plt.figure(figsize=(12, 4))
ax = sns.lineplot(
    data=train_df,
    x="date",
    y="num_sold",
    hue="country",
    errorbar=None,
    linewidth=0.4,
    palette = "Dark2"
)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Number Sold", fontsize=10)
ax.tick_params(axis="both", labelsize=8)
ax.legend(bbox_to_anchor=(1, 1), ncols=1, fontsize=8)
plt.title("Number Sold | Years | Countries", size=10)
plt.show()


plt.figure(figsize=(12, 4))
ax = sns.lineplot(
    data=train_df,
    x="date",
    y="num_sold",
    hue="store",
    errorbar=None,
    linewidth=0.4,
    palette = "Dark2"
)
ax.set_xlabel("Stor", fontsize=10)
ax.set_ylabel("Number Sold", fontsize=10)
ax.tick_params(axis="both", labelsize=8)
ax.legend(bbox_to_anchor=(1, 1), ncols=1, fontsize=8)
plt.title("Number Sold | Stor | Countries", size=10)
plt.show()


def trans_df(df):
    ## Add Holidays
    extract_country = dict(
        zip(np.sort(df.country.unique()), ["CA", "FI", "IT", "KE", "NO", "SG"]))
    holidays_dict = {
        c: holidays.country_holidays(a, years=range(2010, 2020))
        for c, a in extract_country.items()
    }
    df["is_holiday"] = 0
    for c in holidays_dict:
        df.loc[df.country == c, "is_holiday"] = df.date.isin(holidays_dict[c]).astype(int)

    
    df["weekday_sv"] = df["date"].dt.strftime("%a").astype("category")
    df["weekday_num"] = df["date"].dt.strftime("%w").astype("int")
    df["day_of_month"] = df["date"].dt.strftime("%d").astype("int")
    df["month_name_sv"] = df["date"].dt.strftime("%b").astype("category")
    df["month_num"] = df["date"].dt.strftime("%m").astype("int")
    df["year_fv"] = df["date"].dt.strftime("%Y").astype("int")
    df["day_number_year"] = df["date"].dt.strftime("%j").astype("int")
    df["week_number_year"] = df["date"].dt.strftime("%W").astype("int")
    df["country"] = df["country"].astype("category")
    df["store"] = df["store"].astype("category")
    df["product"] = df["product"].astype("category")
    df["is_holiday"] = df["is_holiday"].astype("int")
    df["year_sin"] = np.sin(2 * np.pi * df["year_fv"]/3)
    df["year_cos"] = np.cos(2 * np.pi * df["year_fv"]/3)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12.0)
    df['day_sin'] = np.sin(2 * np.pi + df['day_of_month']  / 365.0)
    df['day_cos'] = np.cos(2 * np.pi + df['day_of_month'] / 365.0)
    df['Group'] = (df['year_fv'] - 2010) * 48 + df['month_num'] * 4 + df['day_of_month'] // 3

    return df


train_df = trans_df(train_df)
train_df = train_df.dropna()
train_df = train_df.drop_duplicates()
test_df = trans_df(test_df)


print("train Data", train_df.info(),"\n\n**********************\n")
print("test Data", test_df.info(),"\n\n**********************")


train_df.describe().round(2).style.format(precision=2).background_gradient(cmap="Blues")


test_df.describe().round(2).style.format(precision=2).background_gradient(cmap="Blues")


average_sale_per_year = train_df[["year_fv","num_sold"]].groupby(["year_fv"])["num_sold"].mean().reset_index()
average_sale_per_year["year_fv"] = average_sale_per_year["year_fv"].astype("category")
average_sale_per_year


plt.figure(figsize=(9,4))
ax = sns.barplot(data = average_sale_per_year, x = "num_sold", y = "year_fv" )
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Average Number Sold", fontsize=10)
ax.tick_params(axis="both", labelsize=8)
ax.legend(bbox_to_anchor=(1, 1), ncols=1, fontsize=8)
plt.title("Average Number Sold | Years | Countries", size=10)
plt.show()


train_df.nunique().sort_values(ascending=False).reset_index().style.format(precision=2).background_gradient(cmap="Greens")


test_df.nunique().sort_values(ascending=False).reset_index().style.format(precision=2).background_gradient(cmap="Greens")


train_df.isna().sum().reset_index().style.format(precision=2).background_gradient(cmap="Reds")


test_df.isna().sum().reset_index().style.format(precision=2).background_gradient(cmap="Reds")


train_df["store"].value_counts().reset_index().style.format(precision=2).background_gradient(cmap="Reds")


test_df["store"].value_counts().reset_index().style.format(precision=2).background_gradient(cmap="Reds")


train_df["country"].value_counts().reset_index().style.format(precision=2).background_gradient(cmap="YlOrRd")


test_df["country"].value_counts().reset_index().style.format(precision=2).background_gradient(cmap="YlOrRd")


train_df["product"].value_counts().reset_index().style.format(precision=2).background_gradient(cmap="YlOrRd")


test_df["product"].value_counts().reset_index().style.format(precision=2).background_gradient(cmap="YlOrRd")


mean_num_sold = train_df[["num_sold"]].mean().squeeze()

fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))

sns.histplot(
    data=train_df,
    x="num_sold",
    color="#038BBD",
    bins=50,
    alpha=0.7,
    lw=0.1,
    ax=ax1,
)

sns.boxplot(
    data=train_df,
    x="num_sold",
    color="#038BBD",
    linewidth=0.3,
    flierprops=dict(
        marker="o", markersize=4, markerfacecolor="darkred", markeredgecolor="darkred"
    ),
    boxprops=dict(alpha=0.7),
    ax=ax2,
)
ax2.set_title("")
ax2.set_xlabel("num_sold", fontsize=12)

ax1.set_title("Target variable Distribution", fontsize=12)
ax1.set_xlabel("")

ax1.axvline(x=mean_num_sold, color="darkred", ls="--", lw=1.5)
ax1.text(
    mean_num_sold + 50,
    39000,
    "Mean num_sold | " + str(mean_num_sold.round(0)),
    fontsize=9,
    color="#000000",
)

plt.show()



fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(data=train_df,
            x='weekday_sv',
            y='num_sold',
            hue='store',
            ax=ax,
            linewidth=0.6)
ax.set_title('Number Sold by Day of Week', fontsize=14)
ax.set_xlabel('Day of Week', fontsize=10)
ax.set_ylabel('Number Sold', fontsize=10)
ax.tick_params(axis="both", labelsize=8)
ax.legend(bbox_to_anchor=(1, 1), ncols=1, fontsize=8)
plt.show()


plt.figure(figsize=(7, 4))
ax = sns.lineplot(
    data=train_df,
    x="weekday_sv",
    y="num_sold",
    hue="country",
    errorbar=None,
    linewidth=1.5,
    palette = "tab10"
)
ax.set_xlabel("WeekDays", fontsize=10)
ax.set_ylabel("Number Sold", fontsize=10)
ax.tick_params(axis="both", labelsize=8)
ax.legend(bbox_to_anchor=(1, 1), ncols=1, fontsize=8)
plt.title("Number Sold duirng WeekDays | Countries", size=10)
plt.show()


plt.figure(figsize=(7, 4))
ax = sns.lineplot(
    data=train_df,
    x="month_name_sv",
    y="num_sold",
    hue="country",
    errorbar=None,
    linewidth=1.5,
    palette = "tab10"
)
ax.set_xlabel("Month", fontsize=10)
ax.set_ylabel("Number Sold", fontsize=10)
ax.tick_params(axis="both", labelsize=8)
ax.legend(bbox_to_anchor=(1, 1), ncols=1, fontsize=8)
plt.title("Number Sold duirng Year | Month | Countries", size=10)
plt.show()


g = sns.FacetGrid(
    train_df,
    row="country",
    hue="year_fv",
    col="product",
    palette = "Dark2",
    height=7,
    aspect=0.9,
    sharey=False,
    sharex=False,
)
g.map(sns.lineplot, "month_name_sv", "num_sold", errorbar=None)
g.set_titles(
        col_template="\n---------------------\n{col_var} = {col_name}\n---------------------\n",
        size=14,
    )

g.add_legend(loc='upper right', title= "Year", fontsize= 16, title_fontsize= 16)
g.tick_params(labelsize=8)
g.set_axis_labels(x_var="Month", y_var="Number Sold", fontsize=14)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()


g = sns.FacetGrid(
    train_df,
    row="year_fv",
    hue="product",
    col="store",
    palette = "Set2",
    height=5,
    aspect=1.2,
    sharey=False,
    sharex=False,
)
g.map(sns.lineplot, "month_name_sv", "num_sold", errorbar=None)
g.set_titles(
        row_template=" ---------------------\nYear of {row_name}\n ---------------------\n",
        col_template="--------------------- |\n{col_var} = {col_name}\n ---------------------\n",
        size=12)

g.add_legend(loc='upper right', title= "Product", fontsize= 12, title_fontsize= 12)
g.tick_params(labelsize=10)
g.set_axis_labels(x_var="Month", y_var="Number Sold", fontsize=12)
plt.subplots_adjust(hspace=0.6, wspace=0.6)
plt.show()



num_vars = train_df.select_dtypes("number").columns.to_list()
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(
    train_df[num_vars].corr(),
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    cmap="PuBuGn",
    annot_kws={"fontsize": 8},
    cbar_kws={"shrink": 1},
    ax = ax
)
cbar_ax = fig.axes[-1]
cbar_ax.tick_params(labelsize=8)
ax.tick_params(labelsize=8)
ax.tick_params(labelsize=8)
plt.title("Correlation Heatmap  | Train DF", fontdict={"fontsize": 14}, pad=20)
plt.show()


num_vars_test = test_df.select_dtypes("number").columns.to_list()
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(
    test_df[num_vars_test].corr(),
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    cmap="PuBuGn",
    annot_kws={"fontsize": 8},
    cbar_kws={"shrink": 1},
    ax = ax
)
cbar_ax = fig.axes[-1]
cbar_ax.tick_params(labelsize=8)
ax.tick_params(labelsize=8)
ax.tick_params(labelsize=8)
plt.title("Correlation Heatmap  | Test DF", fontdict={"fontsize": 14}, pad=20)
plt.show()


fig, ax = plt.subplots(figsize=(2, 4))
sns.heatmap(
    train_df[num_vars]
    .corr()
    .corr()[["num_sold"]]
    .sort_values(by="num_sold", ascending=False),
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    cmap="PuBuGn",
    annot_kws={"fontsize": 8},
    cbar_kws={"shrink": 1},
    ax = ax
)
cbar_ax = fig.axes[-1]
cbar_ax.tick_params(labelsize=6)
ax.tick_params(labelsize=8)
ax.tick_params(labelsize=8)
ax.set_title(
    "Features Correlating with Number Sold", fontdict={"fontsize": 10}, pad=16
)
plt.show()


X = train_df.drop(columns=["date", "num_sold"], axis = "columns")
y = np.log(train_df["num_sold"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=SEED
)

test_df = test_df.drop(columns=["date"])


def lgbm_objective(trial):

    lgbm_params = {
        "n_estimators": 5000,
        "subsample": trial.suggest_float("subsample", 0.3, 0.9),
        "min_child_samples": trial.suggest_int("min_child_samples", 60, 100),
        "max_depth": trial.suggest_int("max_depth", 7, 25),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.001, 0.1),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.001, 0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0)

    }

    lgbm_model = LGBMRegressor(**lgbm_params, random_state=SEED, verbose=-1)

    lgbm_model.fit(X_train, y_train)
    y_pred = np.exp(lgbm_model.predict(X_test))
    return mean_absolute_percentage_error(np.exp(y_test), y_pred)


study_LGBM = optuna.create_study(study_name="LGBM_Kaggle", direction="minimize")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_LGBM.optimize(lgbm_objective, n_trials=200, show_progress_bar=True)


print("Best trial:", study_LGBM.best_trial)


print("Best parameters:", study_LGBM.best_params)


train_df_shap = train_df.drop(columns=["date"], axis=1)
X_shap = train_df_shap.drop(["num_sold"], axis=1)
y_shap = np.log(train_df_shap["num_sold"])
X_train_shap, X_test_shap, y_train_shap, y_test_shap = train_test_split(
    X_shap, y_shap, train_size=0.3, random_state=SEED
)
X_display, y_display = X_train_shap, y_train_shap

lgbm_interp = LGBMRegressor(
    **study_LGBM.best_params,
    objective= "regression",
    metric= "rmse",
    n_estimators= 500,
    random_state=SEED,
    verbose=-1
)
lgbm_interp.fit(X_train_shap, y_train_shap)
shap.initjs()
explainer = shap.TreeExplainer(lgbm_interp)
shap_values = explainer(X_train_shap)



shap.plots.beeswarm(
    shap_values,
    show=False,
)
fig, ax = plt.gcf(), plt.gca()
fig.set_figheight(7)
fig.set_figwidth(5)
ax.tick_params(labelsize=8)
ax.set_title("SHAP Summary", fontsize=12)
plt.show()


shap.plots.bar(shap_values, max_display=15, show=False)
fig, ax = plt.gcf(), plt.gca()
fig.set_figheight(6)
fig.set_figwidth(7)
ax.tick_params(labelsize=8)
plt.show()


index = 100
shap.plots.waterfall(shap_values[index], max_display=18,  show=False)
fig, ax = plt.gcf(), plt.gca()
fig.set_figheight(5)
fig.set_figwidth(7)
plt.show()


lgbm_final = LGBMRegressor(
    **study_LGBM.best_params,
    n_estimators= 5000,
    random_state=SEED,
    verbose=-1
)
lgbm_final.fit(X_train, y_train)
y_pred = np.exp(lgbm_final.predict(X_test))
print("MAPE:",mean_absolute_percentage_error(np.exp(y_test), y_pred))


plt.rcParams["font.size"] = 5
lgb.plot_importance(
    lgbm_final,
    importance_type="gain",
    figsize=(12, 6),
    precision=0,
    grid=False,
    color="green",
)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.title("LightGBM Feature Importance (Gain)", fontsize=15)
plt.show()
plt.rcParams.update(plt.rcParamsDefault)


y_pred_test = lgbm_final.predict(test_df)
y_pred_test


sample_submission["num_sold"] = np.exp(y_pred_test)

sample_submission.to_csv("submission.csv", index=False)
sample_submission

