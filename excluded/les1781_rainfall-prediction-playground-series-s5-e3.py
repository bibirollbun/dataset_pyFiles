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
warnings.filterwarnings("ignore")

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    f1_score, 
    accuracy_score, 
    roc_curve, 
    roc_auc_score
)


# We load the data

rainfall_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col="id")


rainfall_train.shape


rainfall_train.head()


rainfall_train.describe().style.background_gradient(cmap='Greens')


rainfall_train.info()


# Function to view the data of each variable in detail

def detail_columns(data, colum):

    print(
        "Variable: ", colum,
        "\nFormat: ", data[colum].dtype,
        "\nNumber of null values: ", data[colum].isnull().sum(),
        "\nUnique values: ", data[colum].nunique(),
        "\nDistribution of values: \n", data[colum].value_counts()
    )


detail_columns(rainfall_train, "rainfall")


fig,ax = plt.subplots(figsize =(6, 8))
fig.set_facecolor("#b2b2b2")
rainfall_values = ["True", "False"]

ax.pie(
    rainfall_train["rainfall"].value_counts(),
    labels=rainfall_values,
    radius=0.7, 
    startangle=90, 
    autopct= '%1.1f%%',
    colors=sns.color_palette('hls',2),
    wedgeprops={'edgecolor' : 'k'}
)

plt.title("Target variable distribution of values", color='darkgreen', fontsize=16)
plt.legend(loc='lower right', labels=rainfall_values, fontsize=10)

# Mostrar los gráficos

plt.show()


detail_columns(rainfall_train, "day")
print("-" * 50)
detail_columns(rainfall_train, "pressure")


# We analyze the distribution of the data

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.histplot(
    data=rainfall_train, 
    x="day", 
    color="green",
    edgecolor="k",
    kde=True,
    ax=axes[0]
)

sns.histplot(
    data=rainfall_train, 
    x="pressure",
    color="green",
    edgecolor="k",
    kde=True,
    stat="density",
    ax=axes[1]
)

plt.suptitle(t="Distribution of values by Day & Pressure")
plt.tight_layout()
plt.show()


print("Total days: ", rainfall_train["day"].value_counts().sum(),
      "\nTotal years: 6 (", rainfall_train["day"].value_counts().sum() ,"/ 6 = 365",
      "\nTotal correct values(==6): ", (rainfall_train["day"].value_counts() == 6).sum(),
      "\nTotal erroneous values(> 6): ", (rainfall_train["day"].value_counts() > 6).sum(),
      "\nTotal erroneous values(< 6): ", (rainfall_train["day"].value_counts() < 6).sum()
     )


detail_columns(rainfall_train, "temparature")
print("-" * 50)
detail_columns(rainfall_train, "mintemp")
print("-" * 50)
detail_columns(rainfall_train, "maxtemp")


# We analyze the distribution of the data

fig, axes = plt.subplots(ncols=3, figsize=(16, 4))

sns.histplot(
    data=rainfall_train, 
    x="temparature", 
    color="green",
    edgecolor="k",
    kde=True,
    ax=axes[0]
)

sns.histplot(
    data=rainfall_train, 
    x="mintemp",
    color="green",
    edgecolor="k",
    kde=True,
    ax=axes[1]
)

sns.histplot(
    data=rainfall_train, 
    x="maxtemp",
    color="green",
    edgecolor="k",
    kde=True,
    ax=axes[2]
)

plt.suptitle(t="Distribution of values by Temperatures")
plt.tight_layout()
plt.show()


detail_columns(rainfall_train, "dewpoint")
print("-" * 50)
detail_columns(rainfall_train, "humidity")


# We analyze the distribution of the data

fig, axes = plt.subplots(ncols=2, figsize=(16, 4))

sns.histplot(
    data=rainfall_train, 
    x="dewpoint", 
    color="green",
    edgecolor="k",
    kde=True,
    ax=axes[0]
)

sns.histplot(
    data=rainfall_train, 
    x="humidity",
    color="green",
    edgecolor="k",
    kde=True,
    ax=axes[1]
)

plt.suptitle(t="Distribution of values by Dewpoint & Humidity")
plt.tight_layout()
plt.show()


detail_columns(rainfall_train, "cloud")
print("-" * 50)
detail_columns(rainfall_train, "sunshine")


# We analyze the distribution of the data

fig, axes = plt.subplots(ncols=2, figsize=(16, 4))

sns.histplot(
    data=rainfall_train, 
    x="cloud", 
    color="green",
    edgecolor="k",
    kde=True,
    ax=axes[0]
)

sns.histplot(
    data=rainfall_train, 
    x="sunshine",
    color="green",
    edgecolor="k",
    kde=True,
    ax=axes[1]
)

plt.suptitle(t="Distribution of values by Cloud & Sunshine")
plt.tight_layout()
plt.show()


detail_columns(rainfall_train, "winddirection")
print("-" * 50)
detail_columns(rainfall_train, "windspeed")


# We analyze the distribution of the data

fig, axes = plt.subplots(ncols=2, figsize=(16, 4))

sns.histplot(
    data=rainfall_train, 
    x="winddirection", 
    color="green",
    edgecolor="k",
    kde=True,
    ax=axes[0]
)

sns.histplot(
    data=rainfall_train, 
    x="windspeed",
    color="green",
    edgecolor="k",
    kde=True,
    ax=axes[1]
)

plt.suptitle(t="Distribution of values by Winddirection & Windspeed")
plt.tight_layout()
plt.show()


# We make a copy of the original dataset

rainfall_new = rainfall_train.copy()


# We check that no duplicate data is found

print(f"Length: {len(rainfall_new.duplicated())}")
print(f"Duplicates: {rainfall_new.duplicated().sum()}")


# We check for null values

null_values = (
    pd.DataFrame(
        {f"Amount of Null Data": rainfall_new.isnull().sum(), 
         "Percentage of Null Data" : (
             rainfall_new.isnull().sum()) / (len(rainfall_new)) * (100)
        }))

null_values.style.background_gradient(cmap="Greens")


# Column "day" with adjusted values in 365-day cycles

rainfall_new["day"] = (rainfall_new.index % 365) + 1


# We check the variable

print("Total days: ", rainfall_new["day"].value_counts().sum(),
      "\nTotal years: 6 (", rainfall_new["day"].value_counts().sum() ,"/ 6 = 365",
      "\nTotal correct values(==6): ", (rainfall_new["day"].value_counts() == 6).sum(),
      "\nTotal erroneous values(> 6): ", (rainfall_new["day"].value_counts() > 6).sum(),
      "\nTotal erroneous values(< 6): ", (rainfall_new["day"].value_counts() < 6).sum()
     )


# We create a "year" column based on 365-day blocks

start_year = 1981

rainfall_new["year"] = start_year + (rainfall_new.index // 365)

# Convert "year" and "day" to "date_parsed"

rainfall_new["date_parsed"] = pd.to_datetime(
    rainfall_new["year"].astype(str) + "-" + rainfall_new["day"].astype(str), 
    format="%Y-%j"
)


rainfall_new["date_parsed"].head()


# We check all the values in the column

day_data = rainfall_new['date_parsed'].dt.day
month_data = rainfall_new['date_parsed'].dt.month
year_data = rainfall_new['date_parsed'].dt.year

print(
    "Valores únicos de día: ", day_data.unique(),
    "\nValores únicos de mes: ", month_data.unique(),
    "\nValores únicos de year: ", year_data.unique()
)


rainfall_new.info()


year_rain = rainfall_new.pivot(index="date_parsed", columns="year", values="rainfall")
year_rain.describe()


# We graph the rainfall per year

sns.barplot(data=year_rain).tick_params(axis='x', labelrotation=45)

plt.title("Rainfall per year")
plt.tight_layout()
plt.show()


sns.violinplot(
    data=rainfall_new, 
    x="year", 
    y="day", 
    hue="rainfall", 
    palette={0:"green", 1:"red"}
)


year_pressure = rainfall_new.pivot(index="date_parsed", columns="year", values="pressure")
year_pressure.describe()


# We graph the pressure per year

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

sns.lineplot(data=year_pressure, ax=axes[0]).tick_params(axis='x', labelrotation=45)
sns.boxplot(data=year_pressure, ax=axes[1]).tick_params(axis='x', labelrotation=45)

plt.suptitle(t="Pressure per year")
plt.tight_layout()
plt.show()


year_humidity = rainfall_new.pivot(index="date_parsed", columns="year", values="humidity")
year_humidity.describe()


# We graph the humidity per year

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

sns.lineplot(data=year_humidity, ax=axes[0]).tick_params(axis='x', labelrotation=45)
sns.boxplot(data=year_humidity, ax=axes[1]).tick_params(axis='x', labelrotation=45)

plt.suptitle(t="Humidity per year")
plt.tight_layout()
plt.show()


year_temparature = rainfall_new.pivot(index="date_parsed", columns="year", values="temparature")
year_temparature.describe()


# We graph the pressure per year

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

sns.lineplot(data=year_temparature, ax=axes[0]).tick_params(axis='x', labelrotation=45)
sns.boxplot(data=year_temparature, ax=axes[1]).tick_params(axis='x', labelrotation=45)

plt.suptitle(t="Temparature per year")
plt.tight_layout()
plt.show()


# Moving average with appropriate parameters for trend estimation (12 month)

year_trend = year_rain.rolling(window=12, center=True, min_periods=6).mean()


# Moving average chart

fig, axes = plt.subplots(figsize=(12, 4))

sns.lineplot(data=year_trend, dashes=False)

plt.tight_layout()
plt.show()


# Moving average with appropriate parameters for trend estimation (365 days)

average_rain = rainfall_new.groupby("date_parsed").mean()["rainfall"]

rain_trend = average_rain.rolling(
    window=365,
    center=True,
    min_periods=183,
).mean()


# Moving average chart

fig, axes = plt.subplots(figsize=(12, 4))

sns.lineplot(data=rain_trend, dashes=False)

plt.tight_layout()
plt.show()


rainfall_end = rainfall_new.drop(["day", "year", "date_parsed"], axis=1)


rainfall_end.describe().T


rainfall_end.corr().style.background_gradient(cmap="Greens")


# We graph the correlation between the variables

matrix_rainfall = rainfall_end.corr(numeric_only=True).round(1)

plt.figure(figsize=(12, 6))

sns.heatmap(
    matrix_rainfall, 
    annot=True,
    cmap=sns.cubehelix_palette(
        start=2, rot=0, 
        dark=0, light=.95, 
        reverse=True, as_cmap=True
    )
)


# We separate the target variable from the features

x_rain = rainfall_end.drop(columns="rainfall")
y_rainfall = rainfall_end["rainfall"]


# We transform the data

rs = RobustScaler()

num_va = rs.fit_transform(x_rain)

x_rainfall = pd.DataFrame(
    num_va, columns=rs.get_feature_names_out(x_rain.columns)
)


x_rainfall.describe()


mi_scores = mutual_info_classif(x_rainfall, y_rainfall)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=x_rainfall.columns)
mi_scores = mi_scores.sort_values(ascending=False)
mi_scores


scores = mi_scores.sort_values(ascending=True)
width = np.arange(len(mi_scores))
ticks = list(mi_scores.index)
plt.barh(width, mi_scores)
plt.yticks(width, ticks)
plt.title("Mutual Information Scores")
plt.figure(dpi=100, figsize=(8, 5))
plt.show()


x_rainfall = x_rainfall.drop(columns=["maxtemp", "mintemp", "winddirection"])


x_rainfall.info()


# We separate the data into training and test sets

x_train, x_val, y_train, y_val = (
    train_test_split(
        x_rainfall, y_rainfall, test_size=0.2, random_state=42
    )
)


# We review the balance of the target variable

values_counts = np.asarray(np.unique(y_train, return_counts=True))

print(values_counts)


def evaluator(model, xtrain, ytrain, xval, yval):

    '''
    We calculate the precision score
    We obtain prediction from the model
    We calculate the f1 score
    Train result and accuracy score test
    Print train and test macro f1 score
    '''

    train_ascore = model.score(xtrain, ytrain)
    val_ascore = model.score(xval, yval)

    y_train_pred = model.predict(xtrain)
    y_val_pred = model.predict(xval)

    train_fscore = f1_score(ytrain, y_train_pred, average="macro")
    val_fscore = f1_score(yval, y_val_pred, average="macro")

    print(f"Train - Accuracy score: {train_ascore}")
    print(f"Test - Accuracy score: {val_ascore}\n")
    print(f"Train - F1-Score: {train_fscore}")
    print(f"Test - F1-Score: {val_fscore}")


def matrix_evaluator(model, xval, yval, list_classes, color_map):
    '''
    We create the confusion matrix
    We plot the confusion matrix
    Report results
    '''
    y_pred = model.predict(xval)
    cm_values = confusion_matrix(yval, y_pred)
    df_cm = pd.DataFrame(
        cm_values,
        columns=list_classes,
        index=list_classes
        )
    df_cm.index.name = "Actual"
    df_cm.columns.name = "Predicted"

    plt.figure(figsize=(6, 4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap=color_map)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    print("\n")
    print(classification_report(
        yval, y_pred, target_names=list_classes
    ))


# We create the model instance

lrc = LogisticRegression()

# Train the model with the data

lrc.fit(x_train, y_train)


# We make predictions

prediction_lrc = lrc.predict(x_val)
print(
    "Confusion matrix:\n\n", 
    confusion_matrix(y_val, prediction_lrc),
    "\n\nClassification Report:\n\n", 
    classification_report(y_val, prediction_lrc)
     )


# We review the parameters

lrc.get_params()


# We establish the parameters to test

lrc_hyperparameters = {
    "C": [0.5, 1.0, 1.5],
    "max_iter": [50, 100, 150],
    "penalty": ["l1","l2","elasticnet","none"]
}

lrc_grid = GridSearchCV(
    lrc,
    lrc_hyperparameters,
    cv=5,
    scoring="accuracy",
    return_train_score=True
)

lrc_search = lrc_grid.fit(x_train, y_train)

print(
    f"Parametros: {lrc_search.best_params_}\nPuntaje: {lrc_search.best_score_}"
)


# We evaluate the accuracy and the f1-score

evaluator(lrc_search, x_train, y_train, x_val, y_val)


# We graph the confusion matrix

matrix_evaluator(
    lrc_search, 
    x_val, 
    y_val, 
    ["No Rain", "Rain"], 
    "Greens"
)


# We create the model instance

label_prop = LabelPropagation()

# Train the model with the data

label_prop.fit(x_train, y_train)


# We make predictions

lp_prediction = label_prop.predict(x_val)
print(
    "Confusion matrix:\n\n", 
    confusion_matrix(y_val, lp_prediction),
    "\n\nClassification Report:\n\n", 
    classification_report(y_val, lp_prediction)
     )


label_prop.get_params()


# We establish the parameters to test

lp_hyperparameters = {
    "gamma": [10, 20, 30],
    "kernel" : ["knn", "rbf"],
    "max_iter": [500, 1000, 1500],
    "n_neighbors" : [4, 7, 14]
}

lp_grid = GridSearchCV(
    label_prop,
    lp_hyperparameters,
    cv=5,
    scoring="accuracy",
    return_train_score=True
)

lp_search = lp_grid.fit(x_train, y_train)

print(
    f"Parametros: {lp_search.best_params_}\nPuntaje: {lp_search.best_score_}"
)


# We evaluate the accuracy and the f1-score

evaluator(lp_search, x_train, y_train, x_val, y_val)


# We graph the confusion matrix

matrix_evaluator(
    lp_search, 
    x_val, 
    y_val, 
    ["No Rain", "Rain"], 
    "Greens"
)


# We create the model instance

xgbc = xgb.XGBClassifier()

# Train the model with the data

xgbc.fit(x_train, y_train)


# We make predictions

xgbc_prediction = xgbc.predict(x_val)
print(
    "Confusion matrix:\n\n", 
    confusion_matrix(y_val, xgbc_prediction),
    "\n\nClassification Report:\n\n", 
    classification_report(y_val, xgbc_prediction)
     )


xgbc.get_params()


# We establish the parameters to test

xgbc_hyperparameters = {
    "n_estimators": [100, 150],
    "max_depth" : [6, 8],
    "gamma" : [0, 1],
    "alpha" : [0, 1],
    "subsample" : [0.5, 1],
    "scale_pos_weight" : [1, 2]
}

xgbc_grid = GridSearchCV(
    xgbc,
    xgbc_hyperparameters,
    cv=5,
    scoring="accuracy",
    return_train_score=True
)

xgbc_search = xgbc_grid.fit(x_train, y_train)

print(
    f"Parametros: {xgbc_search.best_params_}\nPuntaje: {xgbc_search.best_score_}"
)


# We evaluate the accuracy and the f1-score

evaluator(xgbc_search, x_train, y_train, x_val, y_val)


# We graph the confusion matrix

matrix_evaluator(
    xgbc_search, 
    x_val, 
    y_val, 
    ["No Rain", "Rain"], 
    "Greens"
)


# We obtain the accuracy score of the models

y_val_pred_lrc = lrc_search.predict(x_val)
y_val_pred_lp = lp_search.predict(x_val)
y_val_pred_xgbc = xgbc_search.predict(x_val)

lr_model = accuracy_score(y_val, y_val_pred_lrc)
lp_model = accuracy_score(y_val, y_val_pred_lp)
xgbc_model = accuracy_score(y_val, y_val_pred_xgbc)

# We compare the accuracy of the models

print(
    "LogisticRegression accuracy score: {0:0.3f}\n".format(lr_model),
    "\nLabelPropagation accuracy score: {0:0.3f}\n".format(lp_model),
    "\nXGBClassifier accuracy score: {0:0.3f}\n".format(xgbc_model)
)


# We are left with the probabilities of the positive class (the probability of 1)

lrc_probs = lrc_search.predict_proba(x_train)
lrc_probs = lrc_probs[:, 1]
lpc_probs = lp_search.predict_proba(x_train)
lpc_probs = lpc_probs[:, 1]
xgbc_probs = xgbc_search.predict_proba(x_train)
xgbc_probs = xgbc_probs[:, 1]

# We generate an untrained classifier, which will assign 0 to everything

ns_probs = [0 for _ in range(len(y_train))]

# We calculate the AUC

ns_auc = roc_auc_score(y_train, ns_probs)
lrc_auc = roc_auc_score(y_train, lrc_probs)
lpc_auc = roc_auc_score(y_train, lpc_probs)
xgbc_auc = roc_auc_score(y_train, xgbc_probs)

# We print on screen

print('\nUntrained model: ROC AUC = %.3f' % (ns_auc))
print('\nLogisticRegression: ROC AUC = %.3f' % (lrc_auc))
print('\nLabelPropagation: ROC AUC = %.3f' % (lpc_auc))
print('\nXGBClassifier: ROC AUC = %.3f' % (xgbc_auc))


# We calculate the ROC curves

ns_fpr, ns_tpr, _ = roc_curve(y_train, ns_probs)
lrc_fpr, lrc_tpr, _ = roc_curve(y_train, lrc_probs)
lpc_fpr, lpc_tpr, _ = roc_curve(y_train, lpc_probs)
xgbc_fpr, xgbc_tpr, _ = roc_curve(y_train, xgbc_probs)

# We plot the ROC curve

plt.plot(
    ns_fpr, ns_tpr, linestyle="--", color="r", label="Untrained model"
)
plt.plot(
    lrc_fpr, lrc_tpr, marker=".", color="b", label="LogisticRegression"
)
plt.plot(
    lpc_fpr, lpc_tpr, marker=".", color="y", label="LabelPropagation"
)
plt.plot(
    xgbc_fpr, xgbc_tpr, marker=".", color="g", label="XGBClassifier"
)

# We customize the graph

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()


# We tested a new grid with cv=10

xgbc_grid_two = GridSearchCV(
    xgbc,
    xgbc_hyperparameters,
    cv=10,
    scoring="accuracy",
    return_train_score=True
)

xgbc_search_two = xgbc_grid_two.fit(x_train, y_train)

print(
    f"Parametros: {xgbc_search_two.best_params_}\nPuntaje: {xgbc_search_two.best_score_}"
)


# We make predictions

xgbc_prediction_two = xgbc_search_two.predict(x_val)
print(
    "Confusion matrix:\n\n", 
    confusion_matrix(y_val, xgbc_prediction_two),
    "\n\nClassification Report:\n\n", 
    classification_report(y_val, xgbc_prediction_two)
     )


# Create the StratifiedKFold object with 10 divisions (k=10)

s_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# We tested a new grid with StratifiedKFold

# We establish the parameters to test

xgbc_hyperparameters = {
    "n_estimators": [250],
    "gamma" : [4],
    "max_depth" : [3],
    "min_child_weight" : [4],
    "subsample" : [0.6],
    "scale_pos_weight" : [2]
}

xgbc_grid_skf = GridSearchCV(
    xgbc,
    xgbc_hyperparameters,
    cv=s_kfold,
    scoring="accuracy",
    return_train_score=True
)

xgbc_search_skf = xgbc_grid_skf.fit(x_train, y_train)

print(
    f"Parametros: {xgbc_search_skf.best_params_}\nPuntaje: {xgbc_search_skf.best_score_}"
)


# We make predictions

xgbc_prediction_skf = xgbc_search_skf.predict(x_val)
print(
    "Confusion matrix:\n\n", 
    confusion_matrix(y_val, xgbc_prediction_skf),
    "\n\nClassification Report:\n\n", 
    classification_report(y_val, xgbc_prediction_skf)
     )


# We fit the best estimator

xgbc_result = xgbc_search_skf.best_estimator_  
xgbc_result.fit(x_train, y_train)


# We evaluate the accuracy and the f1-score

evaluator(xgbc_result, x_train, y_train, x_val, y_val)


# We graph the confusion matrix

matrix_evaluator(
    xgbc_result, 
    x_val, 
    y_val, 
    ["No Rain", "Rain"], 
    "Greens"
)


# We are left with the probabilities of the positive class (the probability of 1)

xgbc_probs = xgbc_result.predict_proba(x_train)
xgbc_probs = xgbc_probs[:, 1]

# We generate an untrained classifier, which will assign 0 to everything

ns_probs = [0 for _ in range(len(y_train))]


# We calculate the AUC

ns_auc = roc_auc_score(y_train, ns_probs)
xgbc_auc = roc_auc_score(y_train, xgbc_probs)

# We print on screen

print('\nUntrained model: ROC AUC = %.3f' % (ns_auc))
print('\nXGBClassifier: ROC AUC = %.3f' % (xgbc_auc))


# We calculate the ROC curves

ns_fpr, ns_tpr, _ = roc_curve(y_train, ns_probs)
xgbc_fpr, xgbc_tpr, _ = roc_curve(y_train, xgbc_probs)

# We plot the ROC curve

plt.plot(
    ns_fpr, ns_tpr, linestyle="--", color="r", label="Untrained model"
)
plt.plot(
    xgbc_fpr, xgbc_tpr, marker=".", color="g", label="XGBClassifier"
)

# We customize the graph

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()


# We load the test data

df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


# We check the shape and that no duplicate data is found

print(f"Length: {len(df_test.duplicated())}")

print(f"Duplicates: {df_test.duplicated().sum()}")

print(f"Shape: {df_test.shape}")


df_test.info()


# We start by removing the variables that we will not use

df_test_new = df_test.drop(columns=["id", "day", "maxtemp", "mintemp", "winddirection"])


# We check the null values

null_values_test = (
    pd.DataFrame(
        {f'Amount of Null Data' : df_test_new.isnull().sum(), 
         'Percentage of Null Data' : (
             df_test_new.isnull().sum()) / (len(df_test_new)) * (100)
        }
    ))

null_values_test.style.background_gradient(cmap='Greens')


# We transform the data

num_va_test = rs.fit_transform(df_test_new)

rainfall_test = pd.DataFrame(
    num_va_test, columns=rs.get_feature_names_out(df_test_new.columns)
)


rainfall_test.describe()


# We apply the trained model

rainfall_predictions = xgbc_result.predict_proba(rainfall_test)


# We review the result

print('Total predictions: ', len(rainfall_predictions), '\n')


# We obtain the probability of each class

print(rainfall_predictions)


# We create the dataframe

rainfall_submission = pd.DataFrame({
    'id' : df_test['id'], 
    'rainfall' : rainfall_predictions[:, 1]
})

rainfall_submission.head(10)


# We load the submission sample data

rainfall_sample = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


# We compare the results with the sample

print(
    f"Shape Sample Submission: {rainfall_sample.shape}",
    f"\nShape Rainfall Submission: {rainfall_submission.shape}"
)
print("\n", rainfall_sample.head())


# We convert the dataframe to a csv file

rainfall_submission.to_csv("submission.csv", index=False)

