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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


df_train.head(60)


import seaborn as sns
import matplotlib.pyplot as plt


print(df_train.info()) 


print(df_test.info()) 


def leap_year(df, column_name):
    """
    Determines if a dataset includes a leap year based on the presence of day 366.

    Parameters:
    df (DataFrame): The input DataFrame containing the day numbers.
    column_name (str): The name of the column containing day numbers.

    Returns:
    bool: True if a leap year is present, False otherwise.
    """
    if column_name in df.columns:
        return df[column_name].max() == 366
    else:
        raise ValueError(f"Column '{column_name}' not found in the dataset.")


is_leap = leap_year(df_train, "day")
print("Leap year present in train dataset:", is_leap)


is_leap = leap_year(df_test, "day")
print("Leap year present in test dataset:", is_leap)





# Add a new column for the 7-day rolling average of pressure, temperature, maxtemp, mintemp, dewpoint, windspeed, umidity and cloud
df_train["pressure_7day_avg"] = df_train["pressure"].rolling(window=7, min_periods=1).mean()
df_train["temperature_7day_avg"] = df_train["temparature"].rolling(window=7, min_periods=1).mean()
df_train["maxtemp_7day_avg"] = df_train["maxtemp"].rolling(window=7, min_periods=1).mean()
df_train["mintemp_7day_avg"] = df_train["mintemp"].rolling(window=7, min_periods=1).mean()
df_train["dewpoint_7day_avg"] = df_train["dewpoint"].rolling(window=7, min_periods=1).mean()
df_train["windspd_7day_avg"] = df_train["windspeed"].rolling(window=7, min_periods=1).mean() 
df_train["humidity_7day_median"] = df_train["humidity"].rolling(window=7, min_periods=1).median()
df_train["cloud_7day_avg"] = df_train["cloud"].rolling(window=7, min_periods=1).mean()

df_train.head(20)


df_test["pressure_7day_avg"] = df_test["pressure"].rolling(window=7, min_periods=1).mean()
df_test["temperature_7day_avg"] = df_test["temparature"].rolling(window=7, min_periods=1).mean()
df_test["maxtemp_7day_avg"] = df_test["maxtemp"].rolling(window=7, min_periods=1).mean()
df_test["mintemp_7day_avg"] = df_test["mintemp"].rolling(window=7, min_periods=1).mean()
df_test["dewpoint_7day_avg"] = df_test["dewpoint"].rolling(window=7, min_periods=1).mean()
df_test["windspd_7day_avg"] = df_test["windspeed"].rolling(window=7, min_periods=1).mean() 
df_test["humidity_7day_median"] = df_test["humidity"].rolling(window=7, min_periods=1).median()
df_test["cloud_7day_avg"] = df_test["cloud"].rolling(window=7, min_periods=1).mean()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Add a new column for the 7-day rolling average of pressure, temperature, maxtemp, mintemp, dewpoint, windspeed, humidity and cloud
df_train["pressure_7day_avg"] = scaler.fit_transform(df_train[["pressure_7day_avg"]])
df_train["temperature_7day_avg"] = scaler.fit_transform(df_train[["temperature_7day_avg"]])
df_train["maxtemp_7day_avg"] = scaler.fit_transform(df_train[["maxtemp_7day_avg"]])
df_train["mintemp_7day_avg"] = scaler.fit_transform(df_train[["mintemp_7day_avg"]])
df_train["dewpoint_7day_avg"] = scaler.fit_transform(df_train[["dewpoint_7day_avg"]])
df_train["windspd_7day_avg"] = scaler.fit_transform(df_train[["windspd_7day_avg"]]) 
df_train["humidity_7day_median"] = scaler.fit_transform(df_train[["humidity_7day_median"]])
df_train["cloud_7day_avg"] = scaler.fit_transform(df_train[["cloud_7day_avg"]])

df_train.head(20)



df_test["pressure_7day_avg"] = scaler.fit_transform(df_test[["pressure_7day_avg"]])
df_test["temperature_7day_avg"] = scaler.fit_transform(df_test[["temperature_7day_avg"]])
df_test["maxtemp_7day_avg"] = scaler.fit_transform(df_test[["maxtemp_7day_avg"]])
df_test["mintemp_7day_avg"] = scaler.fit_transform(df_test[["mintemp_7day_avg"]])
df_test["dewpoint_7day_avg"] = scaler.fit_transform(df_test[["dewpoint_7day_avg"]])
df_test["windspd_7day_avg"] = scaler.fit_transform(df_test[["windspd_7day_avg"]]) 
df_test["humidity_7day_median"] = scaler.fit_transform(df_test[["humidity_7day_median"]])
df_test["cloud_7day_avg"] = scaler.fit_transform(df_test[["cloud_7day_avg"]])


# Exclude "id" and "day" columns
numeric_columns = [col for col in df_train.columns if col in ["pressure_7day_avg", "temperature_7day_avg","maxtemp_7day_avg","mintemp_7day_avg","dewpoint_7day_avg","windspd_7day_avg","humidity_7day_median","cloud_7day_avg"]]

# Set up figure
fig, axes = plt.subplots(nrows=len(numeric_columns), ncols=2, figsize=(12, 40))

# Loop through numerical columns (excluding "id" and "day")
for i, col in enumerate(numeric_columns):  
    # Histogram against day
    sns.lineplot(data=df_train, x="day", y=col, ax=axes[i, 0])
    axes[i, 0].set_title(f'Histogram of {col} by Day')

    # Boxplot (without day)
    sns.boxplot(y=df_train[col], ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


# Add a new column for the 30-day rolling average of pressure, temperature, maxtemp, mintemp, dewpoint, windspeed,humidity, cloud
df_train["pressure_30day_avg"] = df_train["pressure"].rolling(window=30, min_periods=1).mean()
df_train["temperature_30day_avg"] = df_train["temparature"].rolling(window=30, min_periods=1).mean()
df_train["maxtemp_30day_avg"] = df_train["maxtemp"].rolling(window=30, min_periods=1).mean()
df_train["mintemp_30day_avg"] = df_train["mintemp"].rolling(window=30, min_periods=1).mean()
df_train["dewpoint_30day_avg"] = df_train["dewpoint"].rolling(window=30, min_periods=1).mean()
df_train["windspd_30day_avg"] = df_train["windspeed"].rolling(window=30, min_periods=1).mean() 
df_train["humidity_30day_median"] = df_train["humidity"].rolling(window=30, min_periods=1).median()
df_train["cloud_30day_avg"] = df_train["cloud"].rolling(window=30, min_periods=1).mean()

df_train.head(20)


df_test["pressure_30day_avg"] = df_test["pressure"].rolling(window=30, min_periods=1).mean()
df_test["temperature_30day_avg"] = df_test["temparature"].rolling(window=30, min_periods=1).mean()
df_test["maxtemp_30day_avg"] = df_test["maxtemp"].rolling(window=30, min_periods=1).mean()
df_test["mintemp_30day_avg"] = df_test["mintemp"].rolling(window=30, min_periods=1).mean()
df_test["dewpoint_30day_avg"] = df_test["dewpoint"].rolling(window=30, min_periods=1).mean()
df_test["windspd_30day_avg"] = df_test["windspeed"].rolling(window=30, min_periods=1).mean() 
df_test["humidity_30day_median"] = df_test["humidity"].rolling(window=30, min_periods=1).median()
df_test["cloud_30day_avg"] = df_test["cloud"].rolling(window=30, min_periods=1).mean()


# Add a new column for the 30-day rolling average of pressure, temperature, maxtemp, mintemp, dewpoint, windspeed,humidity, cloud
df_train["pressure_30day_avg"] = scaler.fit_transform(df_train[["pressure_30day_avg"]])
df_train["temperature_30day_avg"] = scaler.fit_transform(df_train[["temperature_30day_avg"]])
df_train["maxtemp_30day_avg"] = scaler.fit_transform(df_train[["maxtemp_30day_avg"]])
df_train["mintemp_30day_avg"] = scaler.fit_transform(df_train[["mintemp_30day_avg"]])
df_train["dewpoint_30day_avg"] = scaler.fit_transform(df_train[["dewpoint_30day_avg"]])
df_train["windspd_30day_avg"] = scaler.fit_transform(df_train[["windspd_30day_avg"]]) 
df_train["humidity_30day_median"] = scaler.fit_transform(df_train[["humidity_30day_median"]])
df_train["cloud_30day_avg"] = scaler.fit_transform(df_train[["cloud_30day_avg"]])

df_train.head(20)


df_test["pressure_30day_avg"] = scaler.fit_transform(df_test[["pressure_30day_avg"]])
df_test["temperature_30day_avg"] = scaler.fit_transform(df_test[["temperature_30day_avg"]])
df_test["maxtemp_30day_avg"] = scaler.fit_transform(df_test[["maxtemp_30day_avg"]])
df_test["mintemp_30day_avg"] = scaler.fit_transform(df_test[["mintemp_30day_avg"]])
df_test["dewpoint_30day_avg"] = scaler.fit_transform(df_test[["dewpoint_30day_avg"]])
df_test["windspd_30day_avg"] = scaler.fit_transform(df_test[["windspd_30day_avg"]]) 
df_test["humidity_30day_median"] = scaler.fit_transform(df_test[["humidity_30day_median"]])
df_test["cloud_30day_avg"] = scaler.fit_transform(df_test[["cloud_30day_avg"]])


# Exclude "id" and "day" columns
numeric_columns = [col for col in df_train.columns if col in ["pressure_30ay_avg", "temperature_30day_avg","maxtemp_30day_avg","mintemp_30day_avg","dewpoint_30day_avg","windspd_30day_avg","humidity_30day_median","cloud_30day_avg"]]

# Set up figure
fig, axes = plt.subplots(nrows=len(numeric_columns), ncols=2, figsize=(12, 40))

# Loop through numerical columns (excluding "id" and "day")
for i, col in enumerate(numeric_columns):  
    # Histogram against day
    sns.lineplot(data=df_train, x="day", y=col, ax=axes[i, 0])
    axes[i, 0].set_title(f'Histogram of {col} by Day')

    # Boxplot (without day)
    sns.boxplot(y=df_train[col], ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


# Add a new column for the 90-day rolling average of pressure, temperature, maxtemp, mintemp, dewpoint, windspeed,humidity, cloud
df_train["pressure_90day_avg"] = df_train["pressure"].rolling(window=90, min_periods=1).mean()
df_train["temperature_90day_avg"] = df_train["temparature"].rolling(window=90, min_periods=1).mean()
df_train["maxtemp_90day_avg"] = df_train["maxtemp"].rolling(window=90, min_periods=1).mean()
df_train["mintemp_90day_avg"] = df_train["mintemp"].rolling(window=90, min_periods=1).mean()
df_train["dewpoint_90day_avg"] = df_train["dewpoint"].rolling(window=90, min_periods=1).mean()
df_train["windspd_90day_avg"] = df_train["windspeed"].rolling(window=90, min_periods=1).mean() 
df_train["humidity_90day_median"] = df_train["humidity"].rolling(window=90, min_periods=1).median()
df_train["cloud_90day_avg"] = df_train["cloud"].rolling(window=90, min_periods=1).mean()

df_train.head(20)


df_test["pressure_90day_avg"] = df_test["pressure"].rolling(window=90, min_periods=1).mean()
df_test["temperature_90day_avg"] = df_test["temparature"].rolling(window=90, min_periods=1).mean()
df_test["maxtemp_90day_avg"] = df_test["maxtemp"].rolling(window=90, min_periods=1).mean()
df_test["mintemp_90day_avg"] = df_test["mintemp"].rolling(window=90, min_periods=1).mean()
df_test["dewpoint_90day_avg"] = df_test["dewpoint"].rolling(window=90, min_periods=1).mean()
df_test["windspd_90day_avg"] = df_test["windspeed"].rolling(window=90, min_periods=1).mean() 
df_test["humidity_90day_median"] = df_test["humidity"].rolling(window=90, min_periods=1).median()
df_test["cloud_90day_avg"] = df_test["cloud"].rolling(window=90, min_periods=1).mean()


# Add a new column for the 90-day rolling average of pressure, temperature, maxtemp, mintemp, dewpoint, windspeed, humidity, cloud
df_train["pressure_90day_avg"] = scaler.fit_transform(df_train[["pressure_90day_avg"]])
df_train["temperature_90day_avg"] = scaler.fit_transform(df_train[["temperature_90day_avg"]])
df_train["maxtemp_90day_avg"] = scaler.fit_transform(df_train[["maxtemp_90day_avg"]])
df_train["mintemp_90day_avg"] = scaler.fit_transform(df_train[["mintemp_90day_avg"]])
df_train["dewpoint_90day_avg"] = scaler.fit_transform(df_train[["dewpoint_90day_avg"]])
df_train["windspd_90day_avg"] = scaler.fit_transform(df_train[["windspd_90day_avg"]]) 
df_train["humidity_90day_median"] = scaler.fit_transform(df_train[["humidity_90day_median"]])
df_train["cloud_90day_avg"] = scaler.fit_transform(df_train[["cloud_90day_avg"]])

df_train.head(20)


df_test["pressure_90day_avg"] = scaler.fit_transform(df_test[["pressure_90day_avg"]])
df_test["temperature_90day_avg"] = scaler.fit_transform(df_test[["temperature_90day_avg"]])
df_test["maxtemp_90day_avg"] = scaler.fit_transform(df_test[["maxtemp_90day_avg"]])
df_test["mintemp_90day_avg"] = scaler.fit_transform(df_test[["mintemp_90day_avg"]])
df_test["dewpoint_90day_avg"] = scaler.fit_transform(df_test[["dewpoint_90day_avg"]])
df_test["windspd_90day_avg"] = scaler.fit_transform(df_test[["windspd_90day_avg"]]) 
df_test["humidity_90day_median"] = scaler.fit_transform(df_test[["humidity_90day_median"]])
df_test["cloud_90day_avg"] = scaler.fit_transform(df_test[["cloud_90day_avg"]])


# Exclude "id" and "day" columns
numeric_columns = [col for col in df_train.columns if col in ["pressure_90day_avg", "temperature_90day_avg","maxtemp_90day_avg","mintemp_90day_avg","dewpoint_90day_avg","windspd_90day_avg","humidity_90day_median","cloud_90day_avg"]]

# Set up figure
fig, axes = plt.subplots(nrows=len(numeric_columns), ncols=2, figsize=(12, 40))

# Loop through numerical columns (excluding "id" and "day")
for i, col in enumerate(numeric_columns):  
    # Histogram against day
    sns.lineplot(data=df_train, x="day", y=col, ax=axes[i, 0])
    axes[i, 0].set_title(f'Histogram of {col} by Day')

    # Boxplot (without day)
    sns.boxplot(y=df_train[col], ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


# Compute correlation matrix and drop unnecessary columns
corr_matriz = df_train.corr()

# Get absolute correlations with "rainfall" and sort in descending order
features = abs(corr_matriz.loc["rainfall"]).sort_values(ascending=True)

# Display values
print(features)

# Plot top 10 correlated features
features.plot(kind="barh", figsize=(10, 6), color="skyblue")

# Set plot labels and title
plt.title("Feature Correlations with Rainfall ")
plt.ylabel("Features")
plt.xlabel("Correlation Coefficient")

# Show the plot
plt.show()



# Get absolute correlations with "rainfall" and sort in descending order
top_features = abs(corr_matriz.loc["rainfall"]).sort_values(ascending=False).head(11)

# Display values
print(top_features)

# Plot top 10 correlated features (sorted in descending order)
top_features.sort_values(ascending=True).plot(kind="barh", figsize=(10, 6), color="skyblue")

# Set plot labels and title
plt.title("Top 10 Feature Correlations with Rainfall (Excluding day_of_year & id)")
plt.ylabel("Features")
plt.xlabel("Correlation Coefficient")

# Show the plot
plt.show()


top_features_selection = ['cloud','sunshine','humidity','cloud_7day_avg', 'humidity_7day_median','cloud_30day_avg',
                         'humidity_30day_median','windspeed','cloud_90day_avg','humidity_90day_median']


train = df_train[top_features_selection]
test = df_test[top_features_selection]


target = df_train['rainfall']


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score


models = {
    'KNN': KNeighborsClassifier(),
    'DT': DecisionTreeClassifier(),
    'ADA': AdaBoostClassifier(),
    'RF': RandomForestClassifier(),
    'XGB': XGBClassifier(),
    'CB': CatBoostClassifier(verbose=0),
    'LGBM': LGBMClassifier(verbose=0)
}


from sklearn.model_selection import train_test_split

train_predictors, eval_predictors, train_target, eval_target = train_test_split(
    train, target, test_size=0.2, random_state=42)


for model in models:
    print('Training', model)
    models[model].fit(train_predictors, train_target)


model_accuracies = {}

for model_name, model in models.items():

    predictions = model.predict(eval_predictors)
    accuracy = accuracy_score(eval_target, predictions)
    model_accuracies[model_name] = accuracy

sorted_model_accuracies = sorted(
    model_accuracies.items(), key=lambda x: x[1], reverse=True)

for model_name, accuracy in sorted_model_accuracies:
    print(f"Model: {model_name}, Accuracy: {accuracy:.4f}")


plt.figure(figsize=(20, 10))

for i, (model_name, model) in enumerate(models.items(), 1):

    y_pred_proba = model.predict_proba(eval_predictors)[:, 1]
    fpr, tpr, _ = roc_curve(eval_target, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.subplot(len(models) // 4 + 1, 4, i)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# =============================
# 1ï¸�âƒ£ BASELINE MODEL (Default Parameters)
# =============================
baseline_model = CatBoostClassifier(verbose=0)
baseline_model.fit(train_predictors, train_target)
y_pred_baseline = baseline_model.predict(eval_predictors)
accuracy_baseline = accuracy_score(eval_target, y_pred_baseline)
y_pred_baseline_prob = baseline_model.predict_proba(eval_predictors)[:, 1]  # Probabilities for ROC-AUC
roc_auc_baseline = roc_auc_score(eval_target, y_pred_baseline_prob)


from skopt import BayesSearchCV

param_space = {
    "iterations": (500, 1500),
    "depth": (3, 4),
    "learning_rate": (0.01, 0.2, "log-uniform"),
    "l2_leaf_reg": (1, 5)
}

cat_model = CatBoostClassifier(verbose=0)

bayes_search = BayesSearchCV(
    cat_model, param_space,
    n_iter=20, cv=3, scoring="accuracy", n_jobs=-1
)

bayes_search.fit(train_predictors, train_target)
# Get best parameters
best_params = bayes_search.best_params_
print("Best Parameters:", bayes_search.best_params_)


# Train best model
bayes_search_model = CatBoostClassifier(**best_params, verbose=0)
bayes_search_model.fit(train_predictors, train_target)
y_pred_bayes_search = bayes_search_model.predict(eval_predictors)
y_pred_bayessearch_prob = bayes_search_model.predict_proba(eval_predictors)[:, 1]
accuracy_bayes_search = accuracy_score(eval_target, y_pred_bayes_search)
roc_auc_bayessearch = roc_auc_score(eval_target, y_pred_bayessearch_prob)


# =============================
# 3ï¸�âƒ£ BEST PARAMS + EARLY STOPPING
# =============================
early_stopping_model = CatBoostClassifier(**best_params, verbose=100, early_stopping_rounds=100)
early_stopping_model.fit(train_predictors, train_target, eval_set=(eval_predictors, eval_target))


y_pred_earlystop = early_stopping_model.predict(eval_predictors)
y_pred_earlystop_prob = early_stopping_model.predict_proba(eval_predictors)[:, 1]

accuracy_earlystop = accuracy_score(eval_target, y_pred_earlystop)
roc_auc_earlystop = roc_auc_score(eval_target, y_pred_earlystop_prob)


# =============================
# ğŸ”¹ COMPARISON OF ACCURACIES
# =============================
results = pd.DataFrame({
    "Model": ["Baseline Model", "BayesSearch Optimized", "BayesSearch + Early Stopping"],
    "Accuracy": [accuracy_baseline, accuracy_bayes_search, accuracy_earlystop],
    "ROC-AUC": [roc_auc_baseline, roc_auc_bayessearch, roc_auc_earlystop]
})

print("\nğŸ”¹ Model Comparison:\n", results)

# =============================
# ğŸ”¹ PLOTTING ROC CURVES
# =============================
plt.figure(figsize=(8, 6))

fpr_base, tpr_base, _ = roc_curve(eval_target, y_pred_baseline_prob)
fpr_grid, tpr_grid, _ = roc_curve(eval_target, y_pred_bayessearch_prob)
fpr_stop, tpr_stop, _ = roc_curve(eval_target, y_pred_earlystop_prob)

plt.plot(fpr_base, tpr_base, label="Baseline Model (AUC = {:.3f})".format(roc_auc_baseline), linestyle='--')
plt.plot(fpr_grid, tpr_grid, label="BayesSearch Optimized (AUC = {:.3f})".format(roc_auc_bayessearch), linestyle='-')
plt.plot(fpr_stop, tpr_stop, label="BayesSearch + Early Stopping (AUC = {:.3f})".format(roc_auc_earlystop), linestyle='-.')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.show()

# =============================
# ğŸ”¹ BAR CHART VISUALIZATION (ACCURACY & ROC-AUC)
# =============================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy Bar Chart
axes[0].bar(results["Model"], results["Accuracy"], color=['blue', 'green', 'red'])
axes[0].set_xlabel("Model Type")
axes[0].set_ylabel("Accuracy Score")
axes[0].set_title("Model Accuracy Comparison")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=45)  # Rotate x-axis labels

# ROC-AUC Bar Chart
axes[1].bar(results["Model"], results["ROC-AUC"], color=['blue', 'green', 'red'])
axes[1].set_xlabel("Model Type")
axes[1].set_ylabel("ROC-AUC Score")
axes[1].set_title("Model ROC-AUC Comparison")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='x', rotation=45)  # Rotate x-axis labels

plt.show()


# Ensure 'id' is included in the test dataset
df_test['rainfall'] = early_stopping_model.predict_proba(test)[:, 1]

# Select only 'id' and 'rainfall' before saving
df_test[['id', 'rainfall']].to_csv('20-03-2025-submission.csv', index=False)

