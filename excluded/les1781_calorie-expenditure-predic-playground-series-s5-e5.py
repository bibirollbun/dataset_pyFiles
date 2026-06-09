# We load the competition data

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")


!pip install sweetviz


import numpy as np
import pandas as pd
import ydata_profiling as pp
import sweetviz as sv
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    RandomizedSearchCV
)
from sklearn.metrics import (
    r2_score, 
    mean_squared_log_error
)
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# We load the data

calories_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col="id")


calories_train.shape


calories_train.head()


calories_train.describe().style.background_gradient(cmap='Greens')


calories_train.info()


# We pass the data to the profiler

profile_calories = pp.ProfileReport(calories_train, title="Calories Profiling Report")

# Get the full profile

profile_calories


# We pass the data to the profiler

calories_report = sv.analyze([calories_train, "Train"], target_feat="Calories")

# We use the comparison function of the tool

#calories_compare = sv.compare(source=calories_train, compare=calories_test, target_feat="MedHouseVal")

# Get the full profile

calories_report.show_notebook(w="100%", h="full")


eval_out = sns.PairGrid(calories_train)
eval_out.map(sns.scatterplot)
#eval_out.map_upper(sns.scatterplot)
#eval_out.map_lower(sns.kdeplot)
#eval_out.map_diag(sns.kdeplot, lw=3, legend=False)


# We analyze the Height & Weight

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.boxenplot(
    data=calories_train,
    x="Height", 
    linewidth=.5,
    line_kws=dict(linewidth=1.5, color="#cde"),
    flier_kws=dict(facecolor=".7", linewidth=.5),
    ax=axes[0]
)

sns.boxenplot(
    data=calories_train,
    x="Weight",
    linewidth=.5,
    line_kws=dict(linewidth=1.5, color="#cde"),
    flier_kws=dict(facecolor=".7", linewidth=.5),
    ax=axes[1]
)

plt.suptitle(t="Analysis of Outliers in Height and Weight")
plt.tight_layout()
plt.show()


# We analyze the Height & Weight

fig, axes = plt.subplots(ncols=3, figsize=(14, 4))

sns.boxenplot(x=calories_train["Duration"], color="g", ax=axes[0])
sns.boxenplot(x=calories_train["Body_Temp"], color="b",ax=axes[1])
sns.boxenplot(x=calories_train["Heart_Rate"], color="r",ax=axes[2])

plt.suptitle(t="Analysis of outliers by Exercise, Body temperature, and Heart rate")
plt.tight_layout()
plt.show()


# We make a copy of the original dataset

calories_new = calories_train.copy()


# We confirm that there is no null values

null_values = pd.DataFrame(
        {f"Null Data" : calories_new.isnull().sum(), 
         "Percentage" : (calories_new.isnull().sum()) / (len(calories_new)) * (100)})

null_values


# We replace with an upper threshold(95%) and lower threshold(5%) approximate value

calories_new["Height"] = calories_new["Height"].clip(lower=155.0, upper=200.0).round(decimals=1)
calories_new["Weight"] = calories_new["Weight"].clip(lower=50.0, upper=110.0).round(decimals=1)
calories_new["Heart_Rate"] = calories_new["Heart_Rate"].clip(lower=75.0, upper=115.0).round(decimals=1)
calories_new["Body_Temp"] = calories_new["Body_Temp"].clip(lower=38.00, upper=41.00).round(decimals=1)


fig, axes = plt.subplots(ncols=4, figsize=(16, 4))

sns.boxplot(x=calories_new["Height"], ax=axes[0])
sns.boxplot(x=calories_new["Weight"], ax=axes[1])
sns.boxplot(x=calories_new["Heart_Rate"], ax=axes[2])
sns.boxplot(x=calories_new["Body_Temp"], ax=axes[3])

plt.suptitle(t="Outliers Analysis")
plt.tight_layout()
plt.show()


# We check the duplicate data found

print(f"Length: {len(calories_new.duplicated())}")
print(f"Duplicates: {calories_new.duplicated().sum()}")


# Removing duplicates and keeping the first occurrence

calories_new.drop_duplicates(inplace=True)

# We check that no duplicate data is found

print(f"Length: {len(calories_new.duplicated())}")
print(f"Duplicates: {calories_new.duplicated().sum()}")


calories_new.info()


# Establishing the seaborn aesthetic

sns.set_style("dark")


gender_ce = calories_new.pivot(columns="Sex", values="Calories")

gender_ce.describe().T


# We analyze the Gender

sns.violinplot(
    data=calories_new, 
    x="Calories",
    y="Sex",
    estimator="sum",
    palette="Set3",
    edgecolor="k"
)

plt.title(label="Total calorie expenditure by sex")
plt.tight_layout()
plt.show()


sex_ce = calories_new.pivot(columns="Age", values="Calories")

sex_ce.describe()


# We analyze the Age

fig, axes = plt.subplots(figsize=(14, 6))

sns.barplot(
    data=calories_new, 
    x="Age",
    y="Calories",
    estimator="sum",
    hue="Sex",
    palette="Set3",
    edgecolor="k"
)

plt.title(label="Calorie Expenditure by Age")
plt.tight_layout()
plt.show()


height_ce = calories_new.pivot(columns="Height", values="Calories")

height_ce.describe()


# We analyze the Age

fig, axes = plt.subplots(figsize=(14, 6))

sns.barplot(
    data=calories_new, 
    x="Height",
    y="Calories",
    estimator="sum",
    hue="Sex",
    palette="Set3",
    edgecolor="k"
).tick_params(axis='x', labelrotation=45)

plt.title(label="Calorie Expenditure by Height")
plt.tight_layout()
plt.show()


weight_ce = calories_new.pivot(columns="Weight", values="Calories")

weight_ce.describe()


# We analyze the Age

fig, axes = plt.subplots(figsize=(14, 6))

sns.barplot(
    data=calories_new, 
    x="Weight",
    y="Calories",
    estimator="sum",
    hue="Sex",
    palette="Set3",
    edgecolor="k"
).tick_params(axis='x', labelrotation=45)

plt.title(label="Calorie Expenditure by Weight")
plt.tight_layout()
plt.show()


duration_ce = calories_new.pivot(columns="Duration", values="Calories")

duration_ce.describe()


# We analyze the duration & calorie expenditure

fig, axes = plt.subplots(figsize=(12, 4))

sns.lineplot(
    data=calories_new, 
    x="Duration",
    y="Calories",
    hue="Sex"
)

plt.title(label="Calorie Expenditure by Duration")
plt.tight_layout()
plt.show()


# We analyze the Age

fig, axes = plt.subplots(figsize=(14, 6))

sns.barplot(
    data=calories_new, 
    x="Age",
    y="Duration",
    estimator="sum",
    hue="Sex",
    palette="Set3",
    edgecolor="k"
)

plt.title(label="Exercise Duration by Age & Sex")
plt.tight_layout()
plt.show()


duration_hr = calories_new.pivot(columns="Heart_Rate", values="Duration")

duration_hr.describe()


duration_bt = calories_new.pivot(columns="Body_Temp", values="Duration")

duration_bt.describe()


# Set the style to "darkgrid"

sns.set_style("darkgrid")


# We plot the duration by Heart Rate & Body Temp

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.lineplot(
    data=calories_new, 
    x="Heart_Rate",
    y="Duration",
    hue="Sex",
    ax=axes[0]
)

sns.lineplot(
    data=calories_new, 
    x="Body_Temp",
    y="Duration",
    hue="Sex",
    ax=axes[1]
)

plt.suptitle(t="Exercise duration Impact on Heart Rate & Body Temperature")
plt.tight_layout()
plt.show()


duration_hr_bt = calories_new[["Heart_Rate", "Body_Temp"]].set_index(calories_new["Duration"])

duration_hr_bt.describe()


calories_hr = calories_new.pivot(columns="Heart_Rate", values="Calories")

calories_hr.describe()


calories_bt = calories_new.pivot(columns="Body_Temp", values="Calories")

calories_bt.describe()


# We plot the calories by Heart Rate & Body Temp

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.lineplot(
    data=calories_new, 
    x="Heart_Rate",
    y="Calories",
    color="b",
    ax=axes[0]
)

sns.lineplot(
    data=calories_new, 
    x="Body_Temp",
    y="Calories",
    color="r",
    ax=axes[1]
)

plt.suptitle(t="Heart Rate & Body Temperature per Calorie Expenditure")
plt.tight_layout()
plt.show()


calories_end = calories_new.copy()


calories_end.info()


# We separate the Age variable into bins

bins_age = [0, 40, 60, 80]

# Specify bin labels

labels_age = ["Young", "MiddleAge", "Old"]

# We created the new feature

calories_end["Age_Bins"] = pd.cut(calories_end["Age"], bins_age, labels=labels_age)


# We separate the Duration variable into bins

bins_duration = [0.0, 10.0, 20.0, 31.0]

# Specify bin labels

labels_duration = ["Low", "Moderate", "Intense"]

# We created the new feature

calories_end["Duration_Bins"] = pd.cut(calories_end["Duration"], bins_duration, labels=labels_duration)


calories_end.describe(exclude = np.number).T


fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.barplot(data=calories_end, x="Age_Bins", y="Calories", hue="Sex", ax=axes[0])
sns.boxenplot(data=calories_end, x="Age_Bins", y="Calories", ax=axes[1])

plt.suptitle(t="Analysis of Created Features")
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.barplot(data=calories_end, x="Duration_Bins", y="Calories", hue="Sex", ax=axes[0])
sns.boxenplot(data=calories_end, x="Duration_Bins", y="Calories", ax=axes[1])

plt.suptitle(t="Analysis of Created Features")
plt.tight_layout()
plt.show()


# We create a useful function

def mapper(data, column, order):
    
    data[column] = data[column].map(order)
    data[column] = data[column].astype("float64")

    print(data[column].value_counts())


# We map the variables and change the format

sex_order = {"male" : 0, "female" : 1}
age_order = {"Young" : 0, "MiddleAge" : 1, "Old" : 2}
duration_order = {"Low" : 0, "Moderate" : 1, "Intense" : 2}

mapper(calories_end, "Sex", sex_order)
mapper(calories_end, "Age_Bins", age_order)
mapper(calories_end, "Duration_Bins", duration_order)


calories_end.info()


# We graph the correlation between the variables

matrix_calories = calories_end.corr(numeric_only=True).round(2)

plt.figure(figsize=(10, 4))

sns.heatmap(
    matrix_calories, 
    annot=True,
    cmap=sns.cubehelix_palette(
        start=2, rot=0, 
        dark=0, light=.95, 
        reverse=True, as_cmap=True
    ))


calories_end.describe().T


# We separate the target variable from the features

x_calories = calories_end.drop(columns="Calories")
y_calories = calories_end["Calories"]


# Numerical variables to scale

calories_numeric = x_calories[[
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp"
]]

scaler = StandardScaler().set_output(transform="pandas")
scale_num = scaler.fit_transform(calories_numeric)

# We create a df with the remaining variables

calories_rest = x_calories[[
    "Sex",
    "Age_Bins",
    "Duration_Bins"
]]

# We concatenate the dataframes

x_end = pd.concat([calories_rest, scale_num], axis=1)


x_end.describe().T


x_end.corr()


calories_scores = mutual_info_regression(x_end, y_calories)
calories_scores = pd.Series(calories_scores, name="Calories MI Scores", index=x_end.columns)
calories_scores = calories_scores.sort_values(ascending=False)
calories_scores


scores = calories_scores.sort_values(ascending=True)
width = np.arange(len(calories_scores))
ticks = list(calories_scores.index)
plt.barh(width, calories_scores)
plt.yticks(width, ticks)
plt.title("Mutual Information Scores")
plt.figure(dpi=100, figsize=(8, 5))
plt.show()


# We separate the data into training and validation sets

x_train, x_val, y_train, y_val = (
    train_test_split(x_end, y_calories, test_size=0.2, random_state=42)
)


# Model eval function

def model_eval(model, model_name):

    y_pred = model.predict(x_val)
    y_pred_clip = np.clip(y_pred, a_min=0, a_max=np.inf)

    r2 = r2_score(y_val, y_pred_clip)

    rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred_clip))

    print(f"{model_name}\n\nR-squared score: {r2}\nRMSLE: {rmsle}")


# We create the model instance

lr = LinearRegression()

# Train the model with the data

lr.fit(x_train, y_train)


# We evaluate the initial performance of the model

model_eval(lr, "LinearRegression")


lr.get_params()


# Logarithmic transformation

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)


# Evaluate the model using cross-validation

cv_scores_lr = cross_val_score(
    lr, x_train, y_train_log, 
    scoring="neg_mean_squared_log_error", 
    cv=5
)

print(f"Cross-validation scores: {cv_scores_lr}")
print(f"Mean CV scores: {np.mean(cv_scores_lr):.2f}")


# We create the model instance

lgbr = LGBMRegressor(verbose=0)

# Train the model with the data

lgbr.fit(x_train, y_train)


# We evaluate the initial performance of the model

model_eval(lgbr, "LGBMRegressor")


lgbr.get_params()


# Evaluate the model using cross-validation

cv_scores_lgbr = cross_val_score(
    lgbr, x_train, y_train_log, 
    scoring="neg_mean_squared_log_error", 
    cv=5
)

print(f"Cross-validation scores lgbr: {cv_scores_lgbr}")
print(f"Mean CV scores lgbr: {np.mean(cv_scores_lgbr):.2f}")


# We create the model instance

cbr = CatBoostRegressor(silent=True)

# Train the model with the data

cbr.fit(x_train, y_train)


# We evaluate the initial performance of the model

model_eval(cbr, "CatBoostRegressor")


cbr.get_all_params()


# Evaluate the model using cross-validation

cv_scores_cbr = cross_val_score(
    cbr, x_train, y_train_log, 
    scoring="neg_mean_squared_log_error", 
    cv=5
)

print(f"Cross-validation scores lgbr: {cv_scores_cbr}")
print(f"Mean CV scores lgbr: {np.mean(cv_scores_cbr):.2f}")


# We establish the model to be optimized

final_cbr = CatBoostRegressor(silent=True)


# Create the KFold object

kfold = KFold(n_splits=10, shuffle=True, random_state=42)


# We establish the parameters to test

cbr_param_grid = {
    "learning_rate" : [0.001, 0.01, 0.004],
    "max_depth" : [3, 4, 5],
    "l2_leaf_reg" : [1.0, 5.0, 0.5],
    "min_child_samples" : [1, 3, 6]
}

# We use random search to evaluate the grid

cbr_grid = RandomizedSearchCV(
    final_cbr,
    cbr_param_grid,
    cv=kfold,
    scoring="neg_mean_squared_log_error",
    return_train_score=True
)

cbr_search = cbr_grid.fit(x_train, y_train_log)

print(f"Parameters: {cbr_search.best_params_}\nScore: {cbr_search.best_score_}")


# We save the results within a dataframe

cbr_cv_results = pd.DataFrame(cbr_search.cv_results_)

cbr_cv_results.head().sort_values(by="rank_test_score", ascending=True)


# We evaluate the performance of the model

y_pred_cbr_search_log = cbr_search.best_estimator_.predict(x_val)

y_pred_cbr_search = np.expm1(y_pred_cbr_search_log)

r2_cbr_search = r2_score(y_val, y_pred_cbr_search)

rmsle_cbr_search = np.sqrt(mean_squared_log_error(y_val, y_pred_cbr_search))

print(f"CatBoostRegressor optimization\n\nR-squared score: {r2_cbr_search}\nRMSE: {rmsle_cbr_search}")


# We define the last model

#final_model = cbr_search.best_estimator_

final_model = cbr

final_model.get_params()


# We fit the best model

final_model.fit(x_train, y_train)


# We evaluate the final model performance

model_eval(final_model, "Final Model")


# We create an explainer for the best estimator

explainer = shap.Explainer(final_model)
shap_values = explainer.shap_values(x_val)

# we visualize the importance

fig = shap.summary_plot(
    shap_values,
    x_val,
    show=False
)
plt.title("Feature Importance", fontsize=20, color='g', loc='left')
plt.xlabel("Mean SHAP Values", fontsize=20)
plt.ylabel("Features", fontsize=20)
plt.show()


# We load the test data and submission sample data

df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

calories_sample = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


# We check the test data shape

print(f"Shape: {df_test.shape}")


df_test.head()


df_test.describe().T


df_test.describe(exclude = np.number)


# We check that no duplicate data is found

print(f"Length: {len(df_test.duplicated())}")

print(f"Duplicates: {df_test.duplicated().sum()}")


# We confirm that there is no null values

null_values_test = pd.DataFrame(
        {f"Null Data" : df_test.isnull().sum(), 
         "Percentage" : (df_test.isnull().sum()) / (len(df_test)) * (100)})

null_values_test


# We start by removing the variables that we will not use

test_new = df_test.drop(columns=["id"])


# We created the new features for test

test_new["Age_Bins"] = pd.cut(test_new["Age"], bins_age, labels=labels_age)
test_new["Duration_Bins"] = pd.cut(test_new["Duration"], bins_duration, labels=labels_duration)


# We map the variables and change the format

mapper(test_new, "Sex", sex_order)
mapper(test_new, "Age_Bins", age_order)
mapper(test_new, "Duration_Bins", duration_order)


# Numerical variables to scale

test_numeric = test_new[[
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp"
]]

test_scale_num = scaler.transform(test_numeric)

# We create a df with the remaining variables

test_rest = test_new[[
    "Sex",
    "Age_Bins",
    "Duration_Bins"
]]

# We concatenate the dataframes

test_end = pd.concat([test_rest, test_scale_num], axis=1)


test_end.describe()


test_end.info()


# We apply the trained model

calories_pred = final_model.predict(test_end)
calories_predictions = np.clip(calories_pred, a_min=0, a_max=np.inf)


# We check that there are no negative values

print("Negative values in predictions:", (calories_predictions < 0).sum())

# We review the result

print("Total predictions: ", len(calories_predictions), "\n")


# We create the dataframe

calories_submission = pd.DataFrame({
    "id" : df_test["id"], 
    "Calories" : calories_predictions
})

calories_submission["Calories"] = calories_submission["Calories"].round(decimals=3)

calories_submission.head()


# We compare the results with the sample

print(
    f"Shape Sample Submission: {calories_sample.shape}",
    f"\nShape Calories Submission: {calories_submission.shape}"
)
print("\n", calories_sample.head())


# We convert the dataframe to a csv file

calories_submission.to_csv("submission.csv", index=False)

