# We load the competition data

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    RandomizedSearchCV
)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# We load the data

bpm_train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv", index_col="id")


bpm_train.shape


bpm_train.head()


bpm_train.describe()


bpm_train.info()


# Establishing the seaborn aesthetic

sns.set_style("darkgrid")

# We establish the color palette

palette = sns.set_palette("Greens_r")


# Function to analyze distribution

def plot_numeric(data, column, figsize, suptitle):

    print(
    "Variable: ", column,
    "\nFormat: ", data[column].dtype,
    "\nNumber of null values: ", data[column].isnull().sum(),
    "\nUnique values: ", data[column].nunique(),
    "\nVariable range:", data[column].min(), "to", data[column].max(), "\n\n"
    )

    # We graph the distribution
    
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    
    sns.histplot(
        data=data, 
        x=column, 
        palette=palette,
        edgecolor="k",
        kde=True,
        ax=axes[0]
    )
    sns.boxplot(
        data=data, 
        x=column,
        palette=palette,
        ax=axes[1]
    )
    plt.suptitle(t=suptitle)
    plt.tight_layout()
    plt.show()


# We print and graph the distribution

plot_numeric(
    bpm_train, 
    "BeatsPerMinute", 
    (16, 4), 
    "Distribution of values of the 'BeatsPerMinute' variable"
)


# We print and graph the distribution

plot_numeric(
    bpm_train, 
    "RhythmScore", 
    (16, 4), 
    "Distribution of values of the 'RhythmScore' variable"
)


# We print and graph the distribution

plot_numeric(
    bpm_train, 
    "AudioLoudness", 
    (16, 4), 
    "Distribution of values of the 'AudioLoudness' variable"
)


# We print and graph the distribution

plot_numeric(
    bpm_train, 
    "VocalContent", 
    (16, 4), 
    "Distribution of values of the 'VocalContent' variable"
)


# We print and graph the distribution

plot_numeric(
    bpm_train, 
    "AcousticQuality", 
    (16, 4), 
    "Distribution of values of the 'AcousticQuality' variable"
)


# We print and graph the distribution

plot_numeric(
    bpm_train, 
    "InstrumentalScore", 
    (16, 4), 
    "Distribution of values of the 'InstrumentalScore' variable"
)


# We print and graph the distribution

plot_numeric(
    bpm_train, 
    "LivePerformanceLikelihood", 
    (16, 4), 
    "Distribution of values of the 'LivePerformanceLikelihood' variable"
)


# We print and graph the distribution

plot_numeric(
    bpm_train, 
    "MoodScore", 
    (16, 4), 
    "Distribution of values of the 'MoodScore' variable"
)


# We print and graph the distribution

plot_numeric(
    bpm_train, 
    "TrackDurationMs", 
    (16, 4), 
    "Distribution of values of the 'TrackDurationMs' variable"
)


# We print and graph the distribution

plot_numeric(
    bpm_train, 
    "Energy", 
    (16, 4), 
    "Distribution of values of the 'Energy' variable"
)


# We make a copy of the data

bpm_train_new = bpm_train.copy()


# We check for duplicate and nulls

print(f"Length: {len(bpm_train_new.duplicated())}")
print(f"Duplicates: {bpm_train_new.duplicated().sum()}\n")
print(f"Nulls:\n{bpm_train_new.isnull().sum()}")


# Function to trim the outliers of a column based on quantiles.

def clip_outliers(df, column_name, lower_quantile=0.01, upper_quantile=0.99):
    '''
    We make a copy so as not to modify the original DataFrame in place
    We calculate the limits
    We apply the clipping
    '''
    df_clipped = df.copy()
    
    lower_bound = df_clipped[column_name].quantile(lower_quantile)
    upper_bound = df_clipped[column_name].quantile(upper_quantile)
    
    df_clipped[column_name] = df_clipped[column_name].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clipped


# We apply the function for handling outliers

outliers = [
    "RhythmScore",
    "AudioLoudness",
    "VocalContent", 
    "AcousticQuality",
    "InstrumentalScore",
    "LivePerformanceLikelihood",
    "TrackDurationMs",
]

for cols in outliers:
    bpm_train_new = clip_outliers(bpm_train_new, cols)


# Function to analyze outliers

def outlier_analyzer(df):

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    n_cols = len(numerical_cols)
    n_rows = int(np.ceil(n_cols / 3))
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(18, 9))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        sns.boxplot(x=df[col], ax=axes[i], color="g")
        axes[i].set_title(f"Boxplot of {col}", fontsize=12)
        axes[i].set_xlabel("Feature values", fontsize=10)
        axes[i].set_ylabel("")
    for i in range(n_cols, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle("Distribution Analysis with Boxplots", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()


outlier_analyzer(bpm_train_new)


# We binarize the target variable

bpm_train_new["bpm_bins"] = pd.qcut(
    bpm_train_new["BeatsPerMinute"], 
    q=4, 
    labels=["Low", "Medium", "High", "Very High"]
)


# We analyze BPM by rhythm score

fig, axes = plt.subplots(figsize=(12, 4))

sns.regplot(
    data=bpm_train_new,
    x="BeatsPerMinute",
    y="RhythmScore",
    scatter_kws={"alpha": 0.3, "s": 15},
    line_kws={"color": "red"}    
)

plt.title("Impact of Rhythm Score on BPM", fontsize=16)
plt.xlabel("BPM", fontsize=12)
plt.ylabel("Rhythm Score", fontsize=12)
plt.show()


# We analyze the distribution of BPM by rhythm score

fig, axes = plt.subplots(figsize=(12, 4))

sns.histplot(
    data=bpm_train_new, 
    x="RhythmScore",
    hue="bpm_bins", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
)
sns.move_legend(
    axes, "lower center",
    bbox_to_anchor=(.5, 1.1), 
    ncol=7, 
    title=None, 
    frameon=False,
)

plt.title("BPM distribution by rhythm score")
plt.tight_layout()
plt.show()


# We analyze BPM by instrumental score

fig, axes = plt.subplots(figsize=(12, 4))

sns.regplot(
    data=bpm_train_new,
    x="BeatsPerMinute",
    y="InstrumentalScore",
    scatter_kws={"alpha": 0.3, "s": 15},
    line_kws={"color": "red"}    
)

plt.title("Impact of Instrumental Score on BPM", fontsize=16)
plt.xlabel("BPM", fontsize=12)
plt.ylabel("Instrumental Score", fontsize=12)
plt.show()


# We analyze the distribution of BPM by instrumental score

fig, axes = plt.subplots(figsize=(12, 4))

sns.histplot(
    data=bpm_train_new, 
    x="InstrumentalScore",
    hue="bpm_bins", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
)
sns.move_legend(
    axes, "lower center",
    bbox_to_anchor=(.5, 1.1), 
    ncol=7, 
    title=None, 
    frameon=False,
)

plt.title("BPM distribution by Instrumental Score")
plt.tight_layout()
plt.show()


# We analyze BPM by audio loudness

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.regplot(
    data=bpm_train_new,
    x="BeatsPerMinute",
    y="AudioLoudness",
    scatter_kws={"alpha": 0.3, "s": 15},
    line_kws={"color": "red"},
    ax=axes[0]
)

sns.barplot(
    data=bpm_train_new,
    x="bpm_bins",
    y="AudioLoudness",
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)
plt.suptitle(t="Impact of Audio Loudness on BPM", fontsize=16)
plt.tight_layout()
plt.show()


# We analyze BPM by acoustic quality

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.regplot(
    data=bpm_train_new,
    x="BeatsPerMinute",
    y="AcousticQuality",
    scatter_kws={"alpha": 0.3, "s": 15},
    line_kws={"color": "red"},
    ax=axes[0]
)

sns.barplot(
    data=bpm_train_new,
    x="bpm_bins",
    y="AcousticQuality",
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)
plt.suptitle(t="Impact of Acoustic Quality on BPM", fontsize=16)
plt.tight_layout()
plt.show()


# We analyze BPM by vocal content

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.regplot(
    data=bpm_train_new,
    x="BeatsPerMinute",
    y="VocalContent",
    scatter_kws={"alpha": 0.3, "s": 15},
    line_kws={"color": "red"},
    ax=axes[0]
)

sns.barplot(
    data=bpm_train_new,
    x="bpm_bins",
    y="VocalContent",
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)
plt.suptitle(t="Impact of Vocal Content on BPM", fontsize=16)
plt.tight_layout()
plt.show()


# We analyze BPM by live performance likelihood

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.regplot(
    data=bpm_train_new,
    x="BeatsPerMinute",
    y="LivePerformanceLikelihood",
    scatter_kws={"alpha": 0.3, "s": 15},
    line_kws={"color": "red"},
    ax=axes[0]
)

sns.barplot(
    data=bpm_train_new,
    x="bpm_bins",
    y="LivePerformanceLikelihood",
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)
plt.suptitle(t="Impact of Live Performance Likelihood on BPM", fontsize=16)
plt.tight_layout()
plt.show()


# We analyze BPM by mood score

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.regplot(
    data=bpm_train_new,
    x="BeatsPerMinute",
    y="MoodScore",
    scatter_kws={"alpha": 0.3, "s": 15},
    line_kws={"color": "red"},
    ax=axes[0]
)

sns.barplot(
    data=bpm_train_new,
    x="bpm_bins",
    y="MoodScore",
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)
plt.suptitle(t="Impact of Mood Score on BPM", fontsize=16)
plt.tight_layout()
plt.show()


# We analyze BPM by track duration in milliseconds

fig, axes = plt.subplots(figsize=(12, 4))

sns.regplot(
    data=bpm_train_new,
    x="BeatsPerMinute",
    y="TrackDurationMs",
    scatter_kws={"alpha": 0.3, "s": 15},
    line_kws={"color": "red"}    
)

plt.title("Impact of Track Duration (Ms) on BPM", fontsize=16)
plt.xlabel("BPM", fontsize=12)
plt.ylabel("Track Duration (Ms)", fontsize=12)
plt.show()


# We analyze the distribution of BPM by track duration(Ms)

fig, axes = plt.subplots(figsize=(12, 4))

sns.histplot(
    data=bpm_train_new, 
    x="TrackDurationMs",
    hue="bpm_bins", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
)
sns.move_legend(
    axes, "lower center",
    bbox_to_anchor=(.5, 1.1), 
    ncol=7, 
    title=None, 
    frameon=False,
)

plt.title("BPM distribution by Track Duration (Ms)")
plt.tight_layout()
plt.show()


# We analyze BPM by energy

fig, axes = plt.subplots(figsize=(12, 4))

sns.regplot(
    data=bpm_train_new,
    x="BeatsPerMinute",
    y="Energy",
    scatter_kws={"alpha": 0.3, "s": 15},
    line_kws={"color": "red"}    
)

plt.title("Impact of Energy on BPM", fontsize=16)
plt.xlabel("BPM", fontsize=12)
plt.ylabel("Energy", fontsize=12)
plt.show()


# We analyze the distribution of BPM by energy

fig, axes = plt.subplots(figsize=(12, 4))

sns.histplot(
    data=bpm_train_new, 
    x="Energy",
    hue="bpm_bins", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
)
sns.move_legend(
    axes, "lower center",
    bbox_to_anchor=(.5, 1.1), 
    ncol=7, 
    title=None, 
    frameon=False,
)

plt.title("BPM distribution by Energy")
plt.tight_layout()
plt.show()


# We check for duplicate and nulls

print("-----Competition Data-----\n")
print(f"Length: {len(bpm_train.duplicated())}")
print(f"Duplicates: {bpm_train.duplicated().sum()}\n")
print(f"Nulls:\n\n{bpm_train.isnull().sum()}\n")


x_bpm = bpm_train.drop(columns="BeatsPerMinute")
y_bpm = bpm_train["BeatsPerMinute"]


# We separate the data into training and validation sets

x_train, x_val, y_train, y_val = (
    train_test_split(x_bpm, y_bpm, test_size=0.2, random_state=42)
)


# We apply the function for handling outliers

outliers = [
    "RhythmScore",
    "AudioLoudness",
    "VocalContent",
    "InstrumentalScore",
    "LivePerformanceLikelihood",
    "TrackDurationMs",
]

for cols in outliers:
    x_train = clip_outliers(x_train, cols)
    x_val = clip_outliers(x_val, cols)

x_train = clip_outliers(x_train, "AcousticQuality", lower_quantile=0.01, upper_quantile=0.98)
x_val = clip_outliers(x_val, "AcousticQuality", lower_quantile=0.01, upper_quantile=0.98)


outlier_analyzer(x_train)


outlier_analyzer(x_val)


# Function to analyze distribution

def distribution_analyzer(df):

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    n_cols = len(numerical_cols)
    n_rows = int(np.ceil(n_cols / 3))
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(18, 9))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        sns.histplot(x=df[col], ax=axes[i], color="g")
        axes[i].set_title(f"Histplot of {col}", fontsize=12)
        axes[i].set_xlabel("Feature values", fontsize=10)
        axes[i].set_ylabel("")
    for i in range(n_cols, len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle("Distribution Analysis with histplots", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()


distribution_analyzer(x_train)


distribution_analyzer(x_val)


# We apply a logarithmic transformation

trans_log = [
    "AudioLoudness", 
    "VocalContent", 
    "AcousticQuality", 
    "InstrumentalScore", 
    "LivePerformanceLikelihood"
]

for col in trans_log:
    x_train[col] = np.log1p(x_train[col] - x_train[col].min())
    x_val[col] = np.log1p(x_val[col] - x_val[col].min())


# We apply a scaler to the data

scaler = StandardScaler().set_output(transform="pandas")
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_val_scaled = scaler.transform(x_val)


x_train_scaled.describe().T


x_val_scaled.describe().T


mi_scores = mutual_info_regression(x_train_scaled, y_train)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=x_train_scaled.columns)
mi_scores = mi_scores.sort_values(ascending=False)
mi_scores


# Function to evaluate the models

def evaluator(model, xval, yval, model_name):

    y_pred = model.predict(xval)
    r2 = r2_score(yval, y_pred)
    rmse = np.sqrt(mean_squared_error(yval, y_pred))
    
    print(f"{model_name}\n\nR-squared: {r2}\nRMSE: {rmse}")


# We create the model instance

xgbr = XGBRegressor()

# Train the model with the data

xgbr.fit(x_train_scaled, y_train)


# We evaluate the model

evaluator(xgbr, x_val_scaled, y_val, "XGBRegressor")


# We analyze the permutation importance

result_one = permutation_importance(
    xgbr, x_val_scaled, y_val, 
    n_repeats=30, 
    random_state=42, 
    scoring="neg_root_mean_squared_error"
)
perm_importance_one = pd.DataFrame({
    "Feature": x_train_scaled.columns,
    "Importance Mean": result_one.importances_mean,
    "Importance Std": result_one.importances_std
})
print("\nPermutation Importance XGBRegressor:\n")
print(perm_importance_one.sort_values(by="Importance Mean", ascending=False))


# We create the model instance

lgbmr = LGBMRegressor(verbose=-1)

# Train the model with the data

lgbmr.fit(x_train_scaled, y_train)


# We evaluate the model

evaluator(lgbmr, x_val_scaled, y_val, "LGBMRegressor")


# We analyze the permutation importance

result_two = permutation_importance(
    lgbmr, x_val_scaled, y_val, 
    n_repeats=30, 
    random_state=42, 
    scoring="neg_root_mean_squared_error"
)
perm_importance_two = pd.DataFrame({
    "Feature": x_train_scaled.columns,
    "Importance Mean": result_two.importances_mean,
    "Importance Std": result_two.importances_std
})
print("\nPermutation Importance LGBMRegressor:\n")
print(perm_importance_two.sort_values(by="Importance Mean", ascending=False))


# We create the model instance

cbr = CatBoostRegressor(silent=True)

# Train the model with the data

cbr.fit(x_train_scaled, y_train)


# We evaluate the model

evaluator(cbr, x_val_scaled, y_val, "CatBoostRegressor")


# We analyze the permutation importance

result_three = permutation_importance(
    cbr, x_val_scaled, y_val, 
    n_repeats=30, 
    random_state=42, 
    scoring="neg_root_mean_squared_error"
)
perm_importance_three = pd.DataFrame({
    "Feature": x_train_scaled.columns,
    "Importance Mean": result_three.importances_mean,
    "Importance Std": result_three.importances_std
})
print("\nPermutation Importance CatBoostRegressor:\n")
print(perm_importance_three.sort_values(by="Importance Mean", ascending=False))


# Create the KFold object

kfold = KFold(n_splits=5, shuffle=True, random_state=42)


# We establish the parameters to test

params_grid_xgbr = {
    "learning_rate": uniform(0.01, 0.3),
    "n_estimators": randint(200, 1000),
    "max_depth": randint(4, 8),
    "subsample" : uniform(loc=0.6, scale=0.4),
    "colsample_bytree": uniform(loc=0.6, scale=0.4),
    "gamma" : uniform(0.0, 5.0)
}


# We use RandomizedSearchCV and cv method to evaluate

xgbr_grid = RandomizedSearchCV(
    XGBRegressor(),
    params_grid_xgbr,
    cv=kfold,
    scoring="neg_root_mean_squared_error",
    return_train_score=True,
    n_iter=10
)
xgbr_search = xgbr_grid.fit(x_train_scaled, y_train)
print(f"Parameters: {xgbr_search.best_params_}\nScore: {xgbr_search.best_score_}")


# We save the results within a dataframe

xgbr_cv_results = pd.DataFrame(xgbr_search.cv_results_)

# We select the most important columns

important_cols = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score"
]

display_cols_xgbr = [col for col in important_cols if col in xgbr_cv_results.columns]
display(xgbr_cv_results[display_cols_xgbr].sort_values(by="rank_test_score").head(5))


# We fit the best estimator

xgbr_result = xgbr_search.best_estimator_  
xgbr_result.fit(x_train_scaled, y_train)


# We evaluate the model

evaluator(xgbr_result, x_val_scaled, y_val, "XGBRegressor")


# We establish the parameters to test

params_grid_lgbmr = {
    "n_estimators": randint(200, 1000),
    "learning_rate": uniform(0.01, 0.3),
    "max_depth": randint(4, 8),
    "subsample": uniform(0.6, 0.4),
    "min_child_samples": randint(20, 100)
}


# We use RandomizedSearchCV and cv method to evaluate

lgbmr_grid = RandomizedSearchCV(
    LGBMRegressor(verbose=-1),
    params_grid_lgbmr,
    cv=kfold,
    scoring="neg_root_mean_squared_error",
    return_train_score=True,
    n_iter=10
)
lgbmr_search = lgbmr_grid.fit(x_train_scaled, y_train)
print(f"Parameters: {lgbmr_search.best_params_}\nScore: {lgbmr_search.best_score_}")


# We save the results within a dataframe

lgbmr_cv_results = pd.DataFrame(lgbmr_search.cv_results_)

# We select the most important columns

display_cols_lgbmr = [col for col in important_cols if col in lgbmr_cv_results.columns]
display(lgbmr_cv_results[display_cols_lgbmr].sort_values(by="rank_test_score").head(5))


# We fit the best estimator

lgbmr_result = lgbmr_search.best_estimator_  
lgbmr_result.fit(x_train_scaled, y_train)


# We evaluate the model

evaluator(lgbmr_result, x_val_scaled, y_val, "LGBMRegressor")


# We establish the parameters to test

params_grid_cbr = {
    "learning_rate" : uniform(0.01, 0.3),
    "iterations": randint(100, 1000),
    "depth" : randint(4, 8),
    "l2_leaf_reg" : randint(1, 10),
    "random_strength" : uniform(0.0, 1.0)
}


# We use RandomizedSearchCV and cv method to evaluate

cbr_grid = RandomizedSearchCV(
    CatBoostRegressor(silent=True),
    params_grid_cbr,
    cv=kfold,
    scoring="neg_root_mean_squared_error",
    return_train_score=True,
    n_iter=10
)
cbr_search = cbr_grid.fit(x_train_scaled, y_train)
print(f"Parameters: {cbr_search.best_params_}\nScore: {cbr_search.best_score_}")


# We save the results within a dataframe

cbr_cv_results = pd.DataFrame(cbr_search.cv_results_)

# We select the most important columns

display_cols_cbr = [col for col in important_cols if col in cbr_cv_results.columns]
display(cbr_cv_results[display_cols_cbr].sort_values(by="rank_test_score").head(5))


# We fit the best estimator

cbr_result = cbr_search.best_estimator_  
cbr_result.fit(x_train_scaled, y_train)


# We evaluate the model

evaluator(cbr_result, x_val_scaled, y_val, "CatBoostRegressor")


# We load the test data

df_test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


# We check the shape and that no duplicate data is found

print(f"Length: {len(df_test.duplicated())}")
print(f"Duplicates: {df_test.duplicated().sum()}")
print(f"Shape: {df_test.shape}")


df_test.info()


# We start by removing the variables that we will not use

df_test_new = df_test.drop(columns=["id"])


# We confirm that there is no null values in test data

null_test = pd.DataFrame(
        {"Null Data" : df_test_new.isnull().sum(), 
         "Percentage" : (df_test_new.isnull().sum()) / (len(df_test_new)) * (100)})

null_test


# We apply the function for handling outliers to test

for cols in outliers:
    df_test_new = clip_outliers(df_test_new, cols)

df_test_new = clip_outliers(df_test_new, "AcousticQuality", lower_quantile=0.01, upper_quantile=0.98)


# We apply a logarithmic transformation to test

for col in trans_log:
    df_test_new[col] = np.log1p(df_test_new[col] - df_test_new[col].min())


# We apply a scaler to the test data

x_test_scaled = scaler.transform(df_test_new)

x_test_scaled.describe().T


# We apply the trained model

bpm_predictions = lgbmr_result.predict(x_test_scaled)

# We review the result

print("Total predictions: ", len(bpm_predictions), "\n")

# We create the dataframe

bpm_submission = pd.DataFrame({
    "id" : df_test["id"], 
    "BeatsPerMinute" : bpm_predictions
})

bpm_submission.head()


# We load the submission sample data

bpm_sample = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


# We compare the results with the sample

print(
    f"Shape Sample Submission: {bpm_sample.shape}",
    f"\nShape Bank Submission: {bpm_submission.shape}"
)
print("\n", bpm_sample.head())


# We convert the dataframe to a csv file

bpm_submission.to_csv("submission.csv", index=False)

