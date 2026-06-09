# We load the competition data

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")


!pip install autoviz


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from autoviz.AutoViz_Class import AutoViz_Class
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    LabelEncoder,    
    OneHotEncoder,
    RobustScaler
)
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    RandomizedSearchCV
)
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    f1_score, 
    accuracy_score,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier


# We load the data

personality_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col="id")


personality_train.shape


personality_train.head()


personality_train.describe().style.background_gradient(cmap='Greens')


personality_train.describe(exclude=np.number).T


personality_train.info()


# Creating the Autoviz instance

av = AutoViz_Class()

# To displays plots automatically

%matplotlib inline

# Generating data visualization automatically

av.AutoViz(filename="", dfte=personality_train, depVar="Personality", chart_format="png")


# We make a copy of the original dataset

personality_new = personality_train.copy()


# We confirm that there is no null values

null_values = pd.DataFrame(
        {"Null Data" : personality_new.isnull().sum(), 
         "Percentage" : (personality_new.isnull().sum()) / (len(personality_new)) * (100)})

null_values


# We check for duplicate data

print(f"Length: {len(personality_new.duplicated())}")
print(f"Duplicates: {personality_new.duplicated().sum()}")


def nan_filler(data, column, group, stats):
    
    if stats == "mode":
        data[column] = (
            data[column].fillna(
                data.groupby(group)[column].transform(lambda v: v.mode()[0])))

    else:
        data[column] = (
            data[column].fillna(
                data.groupby(group)[column].transform("mean").round(1)))

    print(
            "Number of null values: ", data[column].isnull().sum(), "\n\n",
            "Distribution of values: \n", data[column].value_counts()
    )


nan_filler(personality_new, "Time_spent_Alone", "Personality", "mean")


nan_filler(personality_new, "Stage_fear", "Personality", "mode")


nan_filler(personality_new, "Social_event_attendance", "Personality", "mean")


nan_filler(personality_new, "Going_outside", "Personality", "mean")


nan_filler(personality_new, "Drained_after_socializing", "Personality", "mode")


nan_filler(personality_new, "Friends_circle_size", "Personality", "mean")


nan_filler(personality_new, "Post_frequency", "Personality", "mean")


# We changed the format for more efficient memory usage

personality_new[personality_new.select_dtypes(["object"]).columns] = (
    personality_new.select_dtypes(["object"]).apply(
        lambda x: x.astype("category"))
)


personality_new.info()


# We check again that there are no duplicates

print(f"Length: {len(personality_new.duplicated())}")
print(f"Duplicates: {personality_new.duplicated().sum()}")


'''
# We create a useful function

def mapper(data, column, order):
    
    data[column] = data[column].map(order)
    data[column] = data[column].astype("float64")

    print(data[column].value_counts())

order = {"No" : 0, "Yes" : 1}
mapper(personality_new, "Stage_fear", order)
mapper(personality_new, "Drained_after_socializing", order)

# Alternative with KNNimputer

imputer = KNNImputer(n_neighbors=2).set_output(transform="pandas")
knn_train = personality_new.drop(columns="Personality")
After_knn_train = imputer.fit_transform(knn_train)

personality_end = pd.concat([After_knn_train, personality_new["Personality"]], axis=1)

print(f"Length: {len(personality_end.duplicated())}")
print(f"Duplicates: {personality_end.duplicated().sum()}")
'''


# We remove the duplicates and reset the index

personality_end = personality_new.drop_duplicates()
personality_end.reset_index(inplace=True, drop=True)


personality_end.info()


# Establishing the seaborn aesthetic

sns.set_style("darkgrid")


personality_tsa = personality_end.pivot(columns="Personality", values="Time_spent_Alone")

personality_tsa.describe().T


# We analyze personality by time spent alone

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.barplot(
    data=personality_end, 
    x="Personality", 
    y="Time_spent_Alone", 
    estimator="sum", 
    edgecolor="k", 
    palette="Paired",
    ax=axes[0]
)
sns.violinplot(
    data=personality_end, 
    x="Personality", 
    y="Time_spent_Alone",
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)
plt.suptitle(t="Personality by Time Spent Alone")
plt.tight_layout()
plt.show()


social_tsa = personality_end.pivot(columns="Friends_circle_size", values="Time_spent_Alone")

social_tsa.describe()


# We analyze the social circle

fig, axes = plt.subplots(nrows=2, figsize=(12, 8))

sns.barplot(
    data=personality_end, 
    x="Friends_circle_size", 
    y="Time_spent_Alone", 
    hue="Personality",
    estimator="sum", 
    edgecolor="k", 
    palette="Paired",
    ax=axes[0]
)
sns.boxenplot(
    data=personality_end, 
    x="Friends_circle_size", 
    y="Time_spent_Alone",
    hue="Personality",
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)

axes[0].legend(bbox_to_anchor=(1, 1),loc="upper left",edgecolor="black")
axes[1].legend(bbox_to_anchor=(1, 1),loc="upper left",edgecolor="black")
plt.suptitle(t="Impact of Time Alone on the Social Circle")
plt.tight_layout()
plt.show()


fear_go = personality_end.pivot(columns="Stage_fear", values="Going_outside")

fear_go.describe().T


# We analyze personality by fear and going outside

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

sns.barplot(
    data=personality_end, 
    x="Stage_fear", 
    y="Going_outside", 
    hue="Personality", 
    edgecolor="k", 
    palette="Paired",
    ax=axes[0]
)
sns.violinplot(
    data=personality_end, 
    x="Stage_fear", 
    y="Going_outside",
    hue="Personality",
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)
sns.move_legend(
    axes[0], "upper right",
    bbox_to_anchor=(.5, 1.2), 
    ncol=2, 
    title=None, 
    frameon=False,
)

axes[1].get_legend().remove()
plt.suptitle(t="Fear of Going Out & Socializing")
plt.tight_layout()
plt.show()


social_drain = personality_end.pivot(columns="Drained_after_socializing", values="Social_event_attendance")

social_drain.describe().T


# We analyze personality by exhaustion and social attendance

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

sns.barplot(
    data=personality_end, 
    x="Drained_after_socializing", 
    y="Social_event_attendance", 
    hue="Personality", 
    edgecolor="k", 
    palette="Paired",
    ax=axes[0]
)
sns.violinplot(
    data=personality_end, 
    x="Drained_after_socializing", 
    y="Social_event_attendance",
    hue="Personality",
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)
sns.move_legend(
    axes[0], "upper right",
    bbox_to_anchor=(.5, 1.2), 
    ncol=2, 
    title=None, 
    frameon=False,
)

axes[1].get_legend().remove()
plt.suptitle(t="Post-socialization Exhaustion")
plt.tight_layout()
plt.show()


fcs_post = personality_end.pivot(columns="Friends_circle_size", values="Post_frequency")

fcs_post.describe()


# We analyze the social circle and post frequency

fig, axes = plt.subplots(nrows=2, figsize=(12, 8))

sns.barplot(
    data=personality_end, 
    x="Friends_circle_size", 
    y="Post_frequency", 
    hue="Personality",
    estimator="sum", 
    edgecolor="k", 
    palette="Paired",
    ax=axes[0]
)
sns.boxenplot(
    data=personality_end, 
    x="Friends_circle_size", 
    y="Post_frequency",
    hue="Personality",
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)

axes[0].legend(bbox_to_anchor=(1, 1),loc="upper left",edgecolor="black")
axes[1].legend(bbox_to_anchor=(1, 1),loc="upper left",edgecolor="black")
plt.suptitle(t="Social Circle by Post Frequency")
plt.tight_layout()
plt.show()


tsa_post = personality_end.pivot(columns="Time_spent_Alone", values="Post_frequency")

tsa_post.describe()


go_post = personality_end.pivot(columns="Going_outside", values="Post_frequency")

go_post.describe()


# We analyze the Post frequency by outdoor activity and time alone

fig, axes = plt.subplots(nrows=2, figsize=(12, 8))

sns.barplot(
    data=personality_end, 
    x="Time_spent_Alone", 
    y="Post_frequency", 
    hue="Personality",
    estimator="sum", 
    edgecolor="k", 
    palette="Paired",
    ax=axes[0]
)
sns.boxenplot(
    data=personality_end, 
    x="Going_outside", 
    y="Post_frequency",
    hue="Personality",
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)

axes[0].legend(bbox_to_anchor=(1, 1),loc="upper left",edgecolor="black")
axes[1].legend(bbox_to_anchor=(1, 1),loc="upper left",edgecolor="black")
plt.suptitle(t="Post Frequency by Outdoor Activity & Time Alone")
plt.tight_layout()
plt.show()


# We separate the target variable from the features

x_personality = personality_end.drop(columns="Personality")
y_personality = personality_end["Personality"]


x_personality.info()


x_personality.describe().T


# We separate the variables into bins

bins_time_alone = [-1.0, 1.0, 3.0, 12.0]
bins_events = [-1.0, 3.0, 6.0, 11.0]
bins_go_out = [-1.0, 3.0, 5.0, 8.0]
bins_friends = [-1.0, 5.0, 9.0, 16.0]
bins_posts = [-1.0, 3.0, 6.0, 11.0]

# Specify bin labels

labels_personality = ["Little", "Normal", "A Lot"]

# We create the new features

x_personality["Time_Alone_Bins"] = pd.cut(
    x_personality["Time_spent_Alone"], bins_time_alone, labels=labels_personality
)
x_personality["Social_Events_Bins"] = pd.cut(
    x_personality["Social_event_attendance"], bins_events, labels=labels_personality
)
x_personality["Go_outside_Bins"] = pd.cut(
    x_personality["Going_outside"], bins_go_out, labels=labels_personality
)
x_personality["Friend_Circle_Bins"] = pd.cut(
    x_personality["Friends_circle_size"], bins_friends, labels=labels_personality
)
x_personality["Post_Freq_Bins"] = pd.cut(
    x_personality["Post_frequency"], bins_posts, labels=labels_personality
)


# We review the new variables

x_personality.describe(exclude = np.number).T


# We evaluate the distribution of the new variables

fig, axes = plt.subplots(ncols=5, figsize=(20, 4))

sns.boxplot(x=x_personality["Time_Alone_Bins"], color="c", ax=axes[0])
sns.boxplot(x=x_personality["Social_Events_Bins"], color="c", ax=axes[1])
sns.boxplot(x=x_personality["Go_outside_Bins"], color="c", ax=axes[2])
sns.boxplot(x=x_personality["Friend_Circle_Bins"], color="c", ax=axes[3])
sns.boxplot(x=x_personality["Post_Freq_Bins"], color="c", ax=axes[4])

plt.suptitle(t="Distribution analysis")
plt.tight_layout()
plt.show()


# We apply LabelEncoder to the target variable

le = LabelEncoder()

y_personality = le.fit_transform(y_personality)

le_values = le.classes_

print(le_values)


x_personality.info()


# We apply OneHotEncoder to the original categorical variables

cat_cols = x_personality[["Stage_fear", "Drained_after_socializing"]]
rest_cols = x_personality.drop(columns=["Stage_fear", "Drained_after_socializing"])
encoder = OneHotEncoder(sparse=False, drop="if_binary").set_output(transform="pandas")
cat_enc = encoder.fit_transform(cat_cols)

# We changed the name of the encoded columns

cat_enc.rename(
    columns={"Stage_fear_Yes": "Stage_fear", 
             "Drained_after_socializing_Yes": "Drained_after_socializing"}, 
    inplace=True)


# We create a useful function

def mapper(data, column, order):
    
    data[column] = data[column].map(order)
    data[column] = data[column].astype("float64")

    print(data[column].value_counts())


# We map the variables labels

bins_order = {"Little" : 0, "Normal" : 1, "A Lot" : 2}

# We use a for loop to apply the function to the desired columns

for column in rest_cols.columns:
    if isinstance(rest_cols[column].dtype, pd.CategoricalDtype):
        mapper(rest_cols, column, bins_order)


# We join the two resulting df

df_encoded = pd.concat([cat_enc, rest_cols], axis=1)

df_encoded.info()


df_encoded.describe().T


# Numerical variables to scale

personality_numeric = df_encoded[[
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency"
]]

scaler = RobustScaler().set_output(transform="pandas")
x_scale = scaler.fit_transform(personality_numeric)

# We create a df with the remaining variables

x_rest = df_encoded.drop(columns=[
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency"
])

# We concatenate the dataframes

x_end = pd.concat([x_scale, x_rest], axis=1)


x_end.describe().T


# We graph the correlation between the variables

matrix_calories = x_end.corr(numeric_only=True).round(2)

plt.figure(figsize=(10, 4))

sns.heatmap(
    matrix_calories, 
    annot=True,
    cmap=sns.cubehelix_palette(rot=-.2)
    )


personality_scores = mutual_info_classif(x_end, y_personality)
personality_scores = pd.Series(personality_scores, name="Personality MI Scores", index=x_end.columns)
personality_scores = personality_scores.sort_values(ascending=False)
personality_scores


scores = personality_scores.sort_values()
width = np.arange(len(personality_scores))
ticks = list(personality_scores.index)
plt.barh(width, personality_scores, color="c", edgecolor="k")
plt.yticks(width, ticks)
plt.title("Mutual Information Scores")
plt.figure(dpi=100, figsize=(8, 5))
plt.show()


# We remove the variables that we will not use

#x_final = x_end.drop(columns="")


x_end.info()


# We separate the data into training and validation sets

x_train, x_val, y_train, y_val = (
    train_test_split(x_end, y_personality, test_size=0.2, random_state=42)
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

    print(f"Train -------> Accuracy score: {train_ascore}")
    print(f"Validation --> Accuracy score: {val_ascore}\n")
    print(f"Train -------> F1-Score: {train_fscore}")
    print(f"Validation --> F1-Score: {val_fscore}")


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


# We evaluate the accuracy and the f1-score

evaluator(lrc, x_train, y_train, x_val, y_val)


# We graph the confusion matrix

matrix_evaluator(
    lrc, 
    x_val, 
    y_val, 
    ["Extrovert", "Introvert"], 
    "Blues"
)


# We create the model instance

lgbmc = LGBMClassifier()

# Train the model with the data

lgbmc.fit(x_train, y_train)


# We evaluate the accuracy and the f1-score

evaluator(lgbmc, x_train, y_train, x_val, y_val)


# We graph the confusion matrix

matrix_evaluator(
    lgbmc, 
    x_val, 
    y_val, 
    ["Extrovert", "Introvert"], 
    "Blues"
)


# We create the model instance

bagc = BaggingClassifier(estimator=LGBMClassifier())

# Train the model with the data

bagc.fit(x_train, y_train)


# We evaluate the accuracy and the f1-score

#evaluator(bagc, x_train, y_train, x_val, y_val)


# We graph the confusion matrix

matrix_evaluator(
    bagc, 
    x_val, 
    y_val, 
    ["Extrovert", "Introvert"], 
    "Blues"
)


# Create the StratifiedKFold object with 5 divisions (k=5)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


lgbmc.get_params()


# We establish the parameters to test

param_grid = {
    "learning_rate" : [0.1, 0.2, 0.3],
    "n_estimators" : [100, 150, 200],
    "num_leaves" : [31, 63, 127],
    "max_depth" : [-1, 3, 4],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "objective": ["binary"]
}

# We use random search to evaluate the grid

lgbmc_grid = RandomizedSearchCV(
    LGBMClassifier(force_col_wise=True, verbose=-1),
    param_grid,
    cv=skf,
    scoring="f1_weighted",
    return_train_score=True
)

lgbmc_search = lgbmc_grid.fit(x_train, y_train)

print(f"Parameters: {lgbmc_search.best_params_}\nScore: {lgbmc_search.best_score_}")


# We save the results within a dataframe

lgbmc_cv_results = pd.DataFrame(lgbmc_search.cv_results_)

lgbmc_cv_results.head().sort_values(by="rank_test_score", ascending=True).T


# We fit the best estimator

lgbmc_result = lgbmc_search.best_estimator_  
lgbmc_result.fit(x_train, y_train)


# We evaluate the accuracy and the f1-score

evaluator(lgbmc_result, x_train, y_train, x_val, y_val)


# We graph the confusion matrix

matrix_evaluator(
    lgbmc_result, 
    x_val, 
    y_val, 
    ["Extrovert", "Introvert"], 
    "Blues"
)


# We define the final model

final_model = lgbmc_result

# We obtain the final parameters of the model

final_model.get_params()


# We create an explainer for the best estimator

explainer = shap.Explainer(final_model)
shap_values = explainer.shap_values(x_val)

# we visualize the importance

fig = shap.summary_plot(
    shap_values,
    x_val,
    show=False
)

plt.legend(bbox_to_anchor=(1.1, 1.15), ncol=2)
plt.title("Feature Importance", fontsize=20, color="b", loc="left")
plt.xlabel("Mean SHAP Values", fontsize=20)
plt.ylabel("Features", fontsize=20)
plt.show()


# We load the test data

df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


# We check the shape and that no duplicate data is found

print(f"Length: {len(df_test.duplicated())}")
print(f"Duplicates: {df_test.duplicated().sum()}")
print(f"Shape: {df_test.shape}")


df_test.describe().T


df_test.describe(exclude = np.number).T


df_test.info()


# We start by removing the variables that we will not use

df_test_new = df_test.drop(columns="id")


# We confirm that there is no null values in test data

null_test = pd.DataFrame(
        {"Null Data" : df_test_new.isnull().sum(), 
         "Percentage" : (df_test_new.isnull().sum()) / (len(df_test_new)) * (100)})

null_test


# We use a for loop to complete the data

for column in df_test_new.columns:
    
    if df_test_new[column].isnull().sum() > 0:
        
        if pd.api.types.is_numeric_dtype(df_test_new[column]):
            imputed_value = df_test_new[column].mean().round(1)
        else:
            imputed_value = df_test_new[column].mode()[0]
        
        df_test_new[column].fillna(imputed_value, inplace=True)
        print(f"{column} : {df_test_new[column].isnull().sum()}")


df_test_new.describe().T


df_test_new.describe(exclude=np.number).T


df_test_new["Time_Alone_Bins"] = pd.cut(df_test_new["Time_spent_Alone"], bins_time_alone, labels=labels_personality)
df_test_new["Social_Events_Bins"] = pd.cut(df_test_new["Social_event_attendance"], bins_events, labels=labels_personality)
df_test_new["Go_outside_Bins"] = pd.cut(df_test_new["Going_outside"], bins_go_out, labels=labels_personality)
df_test_new["Friend_Circle_Bins"] = pd.cut(df_test_new["Friends_circle_size"], bins_friends, labels=labels_personality)
df_test_new["Post_Freq_Bins"] = pd.cut(df_test_new["Post_frequency"], bins_posts, labels=labels_personality)


df_test_new.info()


# We apply OneHotEncoder

test_cat_cols = df_test_new[["Stage_fear", "Drained_after_socializing"]]
test_rest_cols = df_test_new.drop(columns=["Stage_fear", "Drained_after_socializing"])
test_cat_enc = encoder.transform(test_cat_cols)

# We changed the name of the encoded columns

test_cat_enc.rename(
    columns={"Stage_fear_Yes": "Stage_fear", 
             "Drained_after_socializing_Yes": "Drained_after_socializing"}, 
    inplace=True)

# We use a for loop to apply the function to the test bin columns

for column in test_rest_cols.columns:
    if isinstance(test_rest_cols[column].dtype, pd.CategoricalDtype):
        mapper(test_rest_cols, column, bins_order)

test_encoded = pd.concat([test_cat_enc, test_rest_cols], axis=1)


test_encoded.info()


# Variables to scale

test_numeric = test_encoded[[
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency"
]]

test_scale = scaler.transform(test_numeric)
test_rest = test_encoded.drop(columns=[
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency"
])

test_end = pd.concat([test_scale, test_rest], axis=1)


test_end.describe().T


test_end.info()


# We apply the trained model

personality_predictions = final_model.predict(test_end)

# Decode the predictions back to their original labels

decoded_predictions = le.inverse_transform(personality_predictions)

# We review the result

print("Total predictions: ", len(decoded_predictions), "\n")


# We create the dataframe

personality_submission = pd.DataFrame({
    "id" : df_test["id"], 
    "Personality" : decoded_predictions
})

personality_submission.head()


# We load the submission sample data

personality_sample = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


# We compare the results with the sample

print(
    f"Shape Sample Submission: {personality_sample.shape}",
    f"\nShape Personality Submission: {personality_submission.shape}"
)
print("\n", personality_sample.head())


# We convert the dataframe to a csv file

personality_submission.to_csv("submission.csv", index=False)

