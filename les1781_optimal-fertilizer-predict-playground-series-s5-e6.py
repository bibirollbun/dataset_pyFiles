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
import scipy.stats as stats

from sklearn.preprocessing import (
    LabelEncoder,    
    OneHotEncoder,
    StandardScaler
)
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold, 
    RandomizedSearchCV
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.inspection import permutation_importance


# We load the data

fertilizers_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col="id")


fertilizers_train.shape


fertilizers_train.head()


fertilizers_train.describe().style.background_gradient(cmap='Greens')


fertilizers_train.describe(exclude=np.number).T


fertilizers_train.info()


# Establishing the seaborn aesthetic

sns.set_style("darkgrid")


# Function to view the data of each variable in detail

def detail_columns(data, column, chart_title):

    print(
        "Variable: ", column,
        "\nFormat: ", data[column].dtype,
        "\nNumber of null values: ", data[column].isnull().sum(),
        "\nUnique values: ", data[column].nunique(),
        "\nDistribution of values: \n", data[column].value_counts()
    )

    # We analyze the distribution of the data

    fig, axes = plt.subplots(figsize=(12, 4))
    
    sns.histplot(
        data=data, 
        x=column, 
        color="green",
        edgecolor="k",
        kde=True
    )
    
    plt.title(label=chart_title)
    plt.tight_layout()
    plt.show()


detail_columns(
    fertilizers_train,
    "Fertilizer Name",
    "Fertilizer Name Distribution"
)


fig,ax = plt.subplots(figsize =(8, 10))
fig.set_facecolor("#b2b2b2")
fertilizer_values = ["28-28", "17-17-17", "10-26-26", "DAP", "20-20", "14-35-14", "Urea"]

ax.pie(
    fertilizers_train["Fertilizer Name"].value_counts(),
    labels=fertilizer_values,
    radius=0.7, 
    startangle=90, 
    autopct= "%1.1f%%",
    colors=sns.color_palette("hls",7),
    wedgeprops={'edgecolor' : "k"}
)

plt.title("Target variable distribution of values", color="darkgreen", fontsize=16)
plt.legend(loc="lower right", labels=fertilizer_values, fontsize=10)
plt.show()


detail_columns(
    fertilizers_train,
    "Temparature",
    "Temparature Distribution"
)


detail_columns(
    fertilizers_train,
    "Humidity",
    "Humidity Distribution"
)


detail_columns(
    fertilizers_train,
    "Moisture",
    "Moisture Distribution"
)


detail_columns(
    fertilizers_train,
    "Soil Type",
    "Soil Type Distribution"
)


detail_columns(
    fertilizers_train,
    "Crop Type",
    "Crop Type Distribution"
)


detail_columns(
    fertilizers_train,
    "Nitrogen",
    "Nitrogen Distribution"
)


detail_columns(
    fertilizers_train,
    "Potassium",
    "Potassium Distribution"
)


detail_columns(
    fertilizers_train,
    "Phosphorous",
    "Phosphorous Distribution"
)


# We make a copy of the original dataset

fertilizers_new = fertilizers_train.copy()


# We confirm that there is no null values

null_values = pd.DataFrame(
        {f"Null Data" : fertilizers_new.isnull().sum(), 
         "Percentage" : (fertilizers_new.isnull().sum()) / (len(fertilizers_new)) * (100)})

null_values


# We check for duplicate data

print(f"Length: {len(fertilizers_new.duplicated())}")
print(f"Duplicates: {fertilizers_new.duplicated().sum()}")


eval_out = sns.PairGrid(fertilizers_new, palette=sns.light_palette("seagreen"))
eval_out.map(sns.boxplot)
eval_out.tick_params(axis="both", labelbottom=False)


# We changed the format for more efficient memory usage

fertilizers_new[fertilizers_new.select_dtypes(["object"]).columns] = (
    fertilizers_new.select_dtypes(["object"]).apply(
        lambda x: x.astype("category"))
)


# Establishing the seaborn aesthetic

sns.set_style("dark")


fer_n = fertilizers_new.pivot(columns="Fertilizer Name", values="Nitrogen")
fer_k = fertilizers_new.pivot(columns="Fertilizer Name", values="Potassium")
fer_p = fertilizers_new.pivot(columns="Fertilizer Name", values="Phosphorous")


# We analyze the nitrogen values

fer_n.describe()


# We analyze the potassium values

fer_k.describe()


# We analyze the phosphorus values

fer_p.describe()


# We analyze fertilizers by element

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

sns.barplot(
    data=fertilizers_new, 
    x="Fertilizer Name", 
    y="Nitrogen", 
    estimator="sum", 
    edgecolor="k", 
    palette="Paired",
    ax=axes[0]
).tick_params(axis='x', labelrotation=45)
sns.barplot(
    data=fertilizers_new, 
    x="Fertilizer Name", 
    y="Potassium", 
    estimator="sum", 
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
).tick_params(axis='x', labelrotation=45)
sns.barplot(
    data=fertilizers_new, 
    x="Fertilizer Name", 
    y="Phosphorous", 
    estimator="sum", 
    edgecolor="k", 
    palette="Paired",
    ax=axes[2]
).tick_params(axis='x', labelrotation=45)
plt.suptitle(t="Fertilizers by Different Elements")
plt.tight_layout()
plt.show()


# We analyze the fertilizers by soil type

fig, axes = plt.subplots(figsize=(10, 4))

sns.histplot(
    data=fertilizers_new, 
    x="Soil Type",
    hue="Fertilizer Name",
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
)

sns.move_legend(
    axes, "upper left", 
    bbox_to_anchor=(1, 1), 
    edgecolor="black"
)

plt.title("Fertilizers by Soil Type")
plt.tight_layout()
plt.show()


# We analyze the fertilizers by crop type

fig, axes = plt.subplots(figsize=(10, 4))

sns.histplot(
    data=fertilizers_new, 
    x="Crop Type",
    hue="Fertilizer Name", 
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

plt.title("Fertilizers by Crop Type")
plt.tight_layout()
plt.show()


# We analyze the fertilizers by Crop & Soil type

graph = sns.FacetGrid(
    fertilizers_new, 
    col="Soil Type", 
    hue="Fertilizer Name", 
    height=5, aspect=1
)
graph.map(
    sns.histplot, 
    "Crop Type", 
    palette="Paired",  
    shrink=.8,
    edgecolor="k"
)
graph.set_xticklabels(rotation=45)

plt.legend(loc="best", bbox_to_anchor=(1, 1), edgecolor="black")
plt.tight_layout()
plt.show()


soil_tem = fertilizers_new.pivot(columns="Soil Type", values="Temparature")
soil_hum = fertilizers_new.pivot(columns="Soil Type", values="Humidity")
soil_moi = fertilizers_new.pivot(columns="Soil Type", values="Moisture")


soil_tem.describe().T


soil_hum.describe().T


soil_moi.describe().T


# We analyze Soil Type by Climate

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

sns.barplot(
    data=fertilizers_new, 
    x="Temparature",
    y="Soil Type", 
    edgecolor="k", 
    palette="Paired",
    ax=axes[0]
)
sns.barplot(
    data=fertilizers_new, 
    x="Humidity", 
    y="Soil Type", 
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)
sns.barplot(
    data=fertilizers_new, 
    x="Moisture",
    y="Soil Type",  
    edgecolor="k", 
    palette="Paired",
    ax=axes[2]
)

plt.suptitle(t="Soil Type by Climate")
plt.tight_layout()
plt.show()


crop_tem = fertilizers_new.pivot(columns="Crop Type", values="Temparature")
crop_hum = fertilizers_new.pivot(columns="Crop Type", values="Humidity")
crop_moi = fertilizers_new.pivot(columns="Crop Type", values="Moisture")


crop_tem.describe()


crop_hum.describe()


crop_moi.describe()


# We analyze Crop Type by Climate

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

sns.barplot(
    data=fertilizers_new, 
    x="Temparature",
    y="Crop Type", 
    edgecolor="k", 
    palette="Paired",
    ax=axes[0]
)
sns.barplot(
    data=fertilizers_new, 
    x="Humidity", 
    y="Crop Type", 
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)
sns.barplot(
    data=fertilizers_new, 
    x="Moisture",
    y="Crop Type",  
    edgecolor="k", 
    palette="Paired",
    ax=axes[2]
)

plt.suptitle(t="Crop Type by Climate")
plt.tight_layout()
plt.show()


fer_tem = fertilizers_new.pivot(columns="Fertilizer Name", values="Temparature")
fer_hum = fertilizers_new.pivot(columns="Fertilizer Name", values="Humidity")
fer_moi = fertilizers_new.pivot(columns="Fertilizer Name", values="Moisture")


fer_tem.describe()


fer_hum.describe()


fer_moi.describe()


# We analyze fertilizers by Climate

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

sns.barplot(
    data=fertilizers_new, 
    x="Temparature",
    y="Fertilizer Name", 
    edgecolor="k", 
    palette="Paired",
    ax=axes[0]
)
sns.barplot(
    data=fertilizers_new, 
    x="Humidity", 
    y="Fertilizer Name", 
    edgecolor="k",
    palette="Paired",
    ax=axes[1]
)
sns.barplot(
    data=fertilizers_new, 
    x="Moisture",
    y="Fertilizer Name",  
    edgecolor="k", 
    palette="Paired",
    ax=axes[2]
)

plt.suptitle(t="Fertilizers by Climate")
plt.tight_layout()
plt.show()


fertilizers_end = fertilizers_new.copy()


fertilizers_end.info()


# We separate the climate variables into bins

bins_tem = [0.0, 29.0, 33.0, 39.0]
bins_hum = [0.0, 57.0, 65.0, 73.0]
bins_moi = [0.0, 36.0, 51.0, 66.0]

# Specify bin labels

labels_climate = ["Low", "Medium", "High"]

# We create the new features

fertilizers_end["Temparature_Bins"] = pd.cut(fertilizers_end["Temparature"], bins_tem, labels=labels_climate)
fertilizers_end["Humidity_Bins"] = pd.cut(fertilizers_end["Humidity"], bins_hum, labels=labels_climate)
fertilizers_end["Moisture_Bins"] = pd.cut(fertilizers_end["Moisture"], bins_moi, labels=labels_climate)

# We separate the elements variables into bins

bins_n = [0.0, 15.0, 30.0, 43.0]
bins_k = [-0.1, 6.0, 12.0, 20.0]
bins_p = [-0.1, 15.0, 30.0, 43.0]

# Specify bin labels

labels_elements = ["Small", "Medium", "Large"]

# We create the new features

fertilizers_end["Nitrogen_Bins"] = pd.cut(fertilizers_end["Nitrogen"], bins_n, labels=labels_elements)
fertilizers_end["Potassium_Bins"] = pd.cut(fertilizers_end["Potassium"], bins_k, labels=labels_elements)
fertilizers_end["Phosphorous_Bins"] = pd.cut(fertilizers_end["Phosphorous"], bins_p, labels=labels_elements)


# We review the new variables

fertilizers_end.describe(exclude = np.number).T


# We check that we have not generated null or duplicate data

print(f"Duplicate data: {fertilizers_end.duplicated().sum()}")
print(fertilizers_end.isnull().sum())


# We apply LabelEncoder to the target variable

le = LabelEncoder()

fertilizers_end["Fertilizer Name"] = le.fit_transform(fertilizers_end["Fertilizer Name"])


# We create a df with the categorical variables to encode

cat_cols = fertilizers_end[["Soil Type", "Crop Type"]]
rest_cols = fertilizers_end.drop(columns=["Soil Type", "Crop Type"])

# We apply OneHotEncoder

encoder = OneHotEncoder(sparse=False, drop="if_binary").set_output(transform="pandas")
cat_enc = encoder.fit_transform(cat_cols)

# We join the resulting dataframes

df_encoded = pd.concat([rest_cols, cat_enc], axis=1)


# We create a useful function

def mapper(data, column, order):
    
    data[column] = data[column].map(order)
    data[column] = data[column].astype("float64")

    print(data[column].value_counts())


# We map the variables and change the format

climate_order = {"Low" : 0, "Medium" : 1, "High" : 2}
elements_order = {"Small" : 0, "Medium" : 1, "Large" : 2}

mapper(df_encoded, "Temparature_Bins", climate_order)
mapper(df_encoded, "Humidity_Bins", climate_order)
mapper(df_encoded, "Moisture_Bins", climate_order)
mapper(df_encoded, "Nitrogen_Bins", elements_order)
mapper(df_encoded, "Potassium_Bins", elements_order)
mapper(df_encoded, "Phosphorous_Bins", elements_order)


df_encoded.info()


# We graph the correlation between the variables

matrix_fertilizers = df_encoded.corr(numeric_only=True).round(2)

plt.figure(figsize=(16, 8))

sns.heatmap(
    matrix_fertilizers, 
    annot=True,
    cmap=sns.cubehelix_palette(
        start=2, rot=0, 
        dark=0, light=.95, 
        reverse=True, as_cmap=True
    ))


df_encoded.describe().T


# We separate the target variable from the features

x_fertilizers = df_encoded.drop(columns="Fertilizer Name")
y_fertilizers = df_encoded["Fertilizer Name"]


# Numerical variables to scale

fertilizer_numeric = x_fertilizers[[
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous"
]]

scaler = StandardScaler().set_output(transform="pandas")
scale_num = scaler.fit_transform(fertilizer_numeric)

# We create a df with the remaining variables

fertilizer_rest = x_fertilizers.drop(columns=[
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous"
])

# We concatenate the dataframes

x_end = pd.concat([scale_num, fertilizer_rest], axis=1)


x_end.info()


x_end.describe().T


fertilizers_scores = mutual_info_classif(x_end, y_fertilizers)
fertilizers_scores = pd.Series(fertilizers_scores, name="Fertilizers MI Scores", index=x_end.columns)
fertilizers_scores = fertilizers_scores.sort_values(ascending=False)
fertilizers_scores


scores = fertilizers_scores.sort_values(ascending=True)
width = np.arange(len(fertilizers_scores))
ticks = list(fertilizers_scores.index)
plt.barh(width, fertilizers_scores, color="g")
plt.yticks(width, ticks)
plt.title("Mutual Information Scores")
plt.figure(dpi=100, figsize=(8, 5))
plt.show()


# We separate the data into training and validation sets

x_train, x_val, y_train, y_val = (
    train_test_split(x_end, y_fertilizers, test_size=0.2, random_state=42)
)


# We review the balance of the target variable

values_counts = np.asarray(np.unique(y_train, return_counts=True))

print(values_counts)


# Function to calculate AP@K and then MAP@K

def apk(actual, predicted, k):

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score

def mapk(actual, predicted, k):

    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


# Function for an initial evaluation of the model

def evaluator(model, val_x, val_y, enc, k, model_name):
    
    y_pred_prob = model.predict_proba(val_x)
    top_3_indices_val = np.argsort(-y_pred_prob, axis=1)[:, :3]
    
    predicted_val = []
    
    for row_indices in top_3_indices_val:
        names = enc.inverse_transform(row_indices)
        predicted_val.append(list(names))
    
    actual_val = [[enc.inverse_transform([label])[0]] for label in val_y]
    map_3_score_val = mapk(actual_val, predicted_val, k=k)
    
    print(f"\n{model_name} initial MAP@3 Score: {map_3_score_val:.4f}")


# we create the StratifiedKFold object

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


# Function to evaluate models with a cross-validation method

def cv_evaluator(model_class, model_params, x, y, enc, k_map, model_name):

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):

        print(f"\nFold {fold + 1}/{n_splits}")
        x_train_fold, x_val_fold = x.iloc[train_idx], x.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = model_class(**model_params)
        model.fit(x_train_fold, y_train_fold)

        y_pred_prob = model.predict_proba(x_val_fold)
        top_k_indices_val = np.argsort(-y_pred_prob, axis=1)[:, :k_map]

        predicted_fold = []

        for row_indices in top_k_indices_val:

            valid_indices = [idx for idx in row_indices if idx < len(enc.classes_)]
            names = enc.inverse_transform(valid_indices)
            predicted_fold.append(list(names))

        actual_fold = [[enc.inverse_transform([label])[0]] for label in y_val_fold]

        fold_map_score = mapk(actual_fold, predicted_fold, k=k_map)
        fold_scores.append(fold_map_score)
        print(f"MAP@{k_map} Fold {fold + 1}: {fold_map_score:.4f}")

    print(f"\n--- CV Summary for {model_name} ---\n")
    print(f"MAP@{k_map} Scores for fold: {np.round(fold_scores, 4)}")
    print(f"MAP@{k_map} average: {np.mean(fold_scores):.4f}")
    print(f"Standard deviation of MAP@{k_map}: {np.std(fold_scores):.4f}")

    return fold_scores


# We create the model instance

lrc = LogisticRegression(multi_class="multinomial")

# Train the model with the data

lrc.fit(x_train, y_train)


evaluator(lrc, x_val, y_val, le, 3, "LogisticRegression")


# We apply the function for CV

lrc_params = {"multi_class": "multinomial", "random_state": 42}

lrc_cv_scores = cv_evaluator(
    model_class=LogisticRegression,
    model_params=lrc_params,
    x=x_end,
    y=y_fertilizers, 
    enc=le,
    k_map=3,
    model_name="Logistic Regression"
)


# We create the model instance

xgbc = XGBClassifier()

# Train the model with the data

xgbc.fit(x_train, y_train)


evaluator(xgbc, x_val, y_val, le, 3, "XGBClassifier")


# We apply the function for CV

xgbc_params = xgbc.get_params()

lrc_cv_scores = cv_evaluator(
    model_class=XGBClassifier,
    model_params=xgbc_params,
    x=x_end,
    y=y_fertilizers, 
    enc=le,
    k_map=3,
    model_name="XGBClassifier"
)


# We create the model instance

lgbmc = LGBMClassifier()

# Train the model with the data

lgbmc.fit(x_train, y_train)


evaluator(lgbmc, x_val, y_val, le, 3, "LGBMClassifier")


# We apply the function for CV

lgbmc_params = lgbmc.get_params()

lrc_cv_scores = cv_evaluator(
    model_class=LGBMClassifier,
    model_params=lgbmc_params,
    x=x_end,
    y=y_fertilizers, 
    enc=le,
    k_map=3,
    model_name="LGBMClassifier"
)


# We create the model instance

#bagc = BaggingClassifier(estimator=xgbc)

# Train the model with the data

#bagc.fit(x_train, y_train)


#evaluator(bagc, x_val, y_val, le, 3, "BaggingClassifier")


'''
# We apply the function for CV

bagc_params = {"estimator" : xgbc}

lrc_cv_scores = cv_evaluator(
    model_class=BaggingClassifier,
    model_params=bagc_params,
    x=x_end,
    y=y_fertilizers, 
    enc=le,
    k_map=3,
    model_name="BaggingClassifier"
)
'''


# We establish the parameters to test

params_grid = {
    "n_estimators": [100, 150],
    "max_depth" : [6, 8],
    "gamma" : [0, 1],
    "alpha" : [0, 1],
    "subsample" : [0.5, 1],
    "scale_pos_weight" : [1, 2]
}

# We use random search to evaluate the grid

xgbc_grid = RandomizedSearchCV(
    XGBClassifier(),
    params_grid,
    cv=skf,
    scoring="accuracy",
    return_train_score=True
)

xgbc_search = xgbc_grid.fit(x_train, y_train)

print(
    f"Parameters: {xgbc_search.best_params_}\nScore: {xgbc_search.best_score_}"
)


# We save the results within a dataframe

xgbc_cv_results = pd.DataFrame(xgbc_search.cv_results_)

xgbc_cv_results.head().sort_values(by="rank_test_score", ascending=True)


# We fit the best estimator

xgbc_result = xgbc_search.best_estimator_  
xgbc_result.fit(x_train, y_train)


evaluator(xgbc_result, x_val, y_val, le, 3, "XGBClassifier initial optimization")


# We create the model instance

bagc = BaggingClassifier(estimator=xgbc_result)

# Train the model with the data

bagc.fit(x_train, y_train)


evaluator(bagc, x_val, y_val, le, 3, "BaggingClassifier")


# We define the final model

final_model = bagc

# We obtain the final parameters of the model

final_model.get_params()


'''
# Permutation Importance

perm_importance = permutation_importance(final_model, x_val, y_val, n_repeats=30, random_state=42, n_jobs=-1)
perm_importance_df = pd.DataFrame({
    'Feature': x_end.columns,
    'Importance Mean': perm_importance.importances_mean,
    'Importance Std': perm_importance.importances_std
})
print("\nPermutation Importance:\n")
print(perm_importance_df.sort_values(by='Importance Mean', ascending=False))
'''


# We load the test data and submission sample data

df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

fertilizer_sample = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


# We check the shape

print(f"Shape: {df_test.shape}")


df_test.head()


df_test.info()


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


# We separate the climate variables into bins

test_new["Temparature_Bins"] = pd.cut(test_new["Temparature"], bins_tem, labels=labels_climate)
test_new["Humidity_Bins"] = pd.cut(test_new["Humidity"], bins_hum, labels=labels_climate)
test_new["Moisture_Bins"] = pd.cut(test_new["Moisture"], bins_moi, labels=labels_climate)

# We separate the elements variables into bins

test_new["Nitrogen_Bins"] = pd.cut(test_new["Nitrogen"], bins_n, labels=labels_elements)
test_new["Potassium_Bins"] = pd.cut(test_new["Potassium"], bins_k, labels=labels_elements)
test_new["Phosphorous_Bins"] = pd.cut(test_new["Phosphorous"], bins_p, labels=labels_elements)


# We encode categorical variables

test_cat_cols = test_new[["Soil Type", "Crop Type"]]
test_rest_cols = test_new.drop(columns=["Soil Type", "Crop Type"])
test_cat_enc = encoder.fit_transform(test_cat_cols)
test_encoded = pd.concat([test_rest_cols, test_cat_enc], axis=1)


mapper(test_encoded, "Temparature_Bins", climate_order)
mapper(test_encoded, "Humidity_Bins", climate_order)
mapper(test_encoded, "Moisture_Bins", climate_order)
mapper(test_encoded, "Nitrogen_Bins", elements_order)
mapper(test_encoded, "Potassium_Bins", elements_order)
mapper(test_encoded, "Phosphorous_Bins", elements_order)


test_encoded.info()


# Numerical variables to scale

test_numeric = test_encoded[[
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous"
]]
test_scale_num = scaler.transform(test_numeric)
test_rest = test_encoded.drop(columns=[
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous"
])
test_end = pd.concat([test_scale_num, test_rest], axis=1)


test_end.describe().T


# We remove the variables that we will not use

#test_end_new = test_end.drop(columns=[])


test_end.info()


# We apply the trained model

test_pred_prob = final_model.predict_proba(test_end)
test_top_3 = np.argsort(-test_pred_prob, axis=1)[:, :3]

top_3_fertilizer_names = []
for row_indices in test_top_3:
    valid_indices = [idx for idx in row_indices if idx < len(le.classes_)]
    names = le.inverse_transform(valid_indices)
    top_3_fertilizer_names.append(names)

formatted_predictions = [" ".join(names) for names in top_3_fertilizer_names]


# We review the result

print("Total predictions: ", len(formatted_predictions), "\n")


# We create the dataframe

fertilizer_submission = pd.DataFrame({
    "id" : df_test["id"], 
    "Fertilizer Name" : formatted_predictions
})

fertilizer_submission.head()


# We compare the results with the sample

print(
    f"Shape Sample Submission: {fertilizer_sample.shape}",
    f"\nShape Fertilizer Submission: {fertilizer_submission.shape}"
)
print("\n", fertilizer_sample.head())


# We convert the dataframe to a csv file

fertilizer_submission.to_csv("submission.csv", index=False)

