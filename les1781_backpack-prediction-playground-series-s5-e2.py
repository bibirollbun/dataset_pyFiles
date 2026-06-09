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
import lightgbm as lgb

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder, 
    RobustScaler
)
from sklearn.feature_selection import (
    mutual_info_regression
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)
from sklearn.metrics import (
    r2_score, 
    mean_squared_error
)
from sklearn.ensemble import (
    ExtraTreesRegressor, 
    BaggingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor


# We load the data

play_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")


play_train.shape


play_train.head()


play_train.describe()


play_train.describe(exclude = np.number)


play_train.info()


# Function to view the data of each variable in detail

def detail_columns(data, colum):

    print(
        "Variable: ", colum,
        "\nFormat: ", data[colum].dtype,
        "\nNumber of null values: ", data[colum].isnull().sum(),
        "\nUnique values: ", data[colum].nunique(),
        "\nDistribution of values: \n", data[colum].value_counts()
    )


detail_columns(play_train, "id")
print("-"* 39)
detail_columns(play_train, "Brand")


# We analyze the distribution of the data

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.histplot(
    data=play_train, 
    x="Brand", 
    color="green",
    edgecolor="k",
    ax=axes[0]
)

sns.histplot(
    data=play_train, 
    x="Price",
    hue="Brand",
    edgecolor="k",
    ax=axes[1]
)

plt.suptitle(t="Distribution of values by brand")
plt.tight_layout()
plt.show()


detail_columns(play_train, "Material")
print("-"* 39)
detail_columns(play_train, "Waterproof")


# We analyze the distribution of the data

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.histplot(
    data=play_train, 
    x="Waterproof",
    color="green",
    edgecolor="k",
    ax=axes[0]
)

sns.histplot(
    data=play_train, 
    x="Material", 
    hue="Waterproof",
    multiple="stack",
    edgecolor="k",
    ax=axes[1]
)

plt.suptitle(t="Distribution of values by Materials and Endurance")
plt.tight_layout()
plt.show()


wp_nulls = play_train.loc[play_train["Waterproof"].isnull()]

wp_nulls.head()


fig_nulls = sns.FacetGrid(play_train, col="Material", row="Waterproof")
fig_nulls.map(sns.histplot, "Price", color="green", edgecolor="k")
plt.show()


detail_columns(play_train, "Compartments")
print("-"* 39)
detail_columns(play_train, "Laptop Compartment")


# We analyze the distribution of the data

fig, axes = plt.subplots(ncols=3, figsize=(14, 4))

sns.histplot(
    data=play_train, 
    x="Laptop Compartment", 
    color="green",
    edgecolor="k",
    ax=axes[0]
)

sns.histplot(
    data=play_train, 
    x="Compartments",
    hue="Laptop Compartment",
    multiple="dodge", 
    edgecolor="k",
    ax=axes[1]
)

sns.histplot(
    data=play_train, 
    x="Size",
    hue="Laptop Compartment", 
    multiple="dodge",
    color="green",
    edgecolor="k",
    ax=axes[2]
)

plt.suptitle(
    t="Distribution of values by quantity and specific compartments"
)
plt.tight_layout()
plt.show()


sns.histplot(
    data=play_train, 
    x="Weight Capacity (kg)",
    hue="Laptop Compartment",
    multiple="dodge", 
    edgecolor="k"
)


detail_columns(play_train, "Size")
print("-"* 39)
detail_columns(play_train, "Weight Capacity (kg)")


# We analyze the distribution of the data

fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.histplot(
    data=play_train, 
    x="Size", 
    stat="percent", 
    discrete=True,
    color="green",
    edgecolor="k",
    ax=axes[0]
)

sns.histplot(
    data=play_train, 
    x="Weight Capacity (kg)",
    hue="Size",
    element="step",
    stat="density", 
    common_norm=False,
    edgecolor="k",
    ax=axes[1]
)

plt.suptitle(
    t="Distribution of values by Size & Weight Capacity (kg)"
)
plt.tight_layout()
plt.show()


wc_nulls = play_train.loc[play_train["Weight Capacity (kg)"].isnull()]

wc_nulls.head(10)


fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.histplot(
    data=play_train, 
    x="Size", 
    hue="Compartments",
    multiple="dodge",
    edgecolor="k",
    ax=axes[0]
)

sns.histplot(
    data=play_train, 
    x="Weight Capacity (kg)", 
    hue="Compartments",
    palette=sns.color_palette("husl", 10),
    edgecolor="k",
    ax=axes[1]
)

plt.tight_layout()
plt.show()


detail_columns(play_train, "Style")
print("-"* 39)
detail_columns(play_train, "Color")


# We analyze the distribution of the data

fig, axes = plt.subplots(ncols=3, figsize=(14, 4))

sns.histplot(
    data=play_train, 
    x="Style", 
    color="green",
    edgecolor="k",
    ax=axes[0]
)

sns.histplot(
    data=play_train, 
    x="Color",
    hue="Style",
    multiple="dodge",
    edgecolor="k",
    ax=axes[1]
)

sns.histplot(
    data=play_train, 
    x="Brand",
    hue="Style",
    multiple="dodge",
    edgecolor="k",
    ax=axes[2]
)

plt.suptitle(t="Distribution of values by Style & color")
plt.tight_layout()
plt.show()


detail_columns(play_train, "Price")


# We analyze the distribution of the data

sns.histplot(
    data=play_train, 
    x="Price", 
    color="green",
    edgecolor="k",
    kde=True
)

plt.suptitle(t="Distribution of values")
plt.tight_layout()
plt.show()


# We make a copy of the original dataset

play_new = play_train.copy()


# We check that no duplicate data is found

print(f'Length: {len(play_new.duplicated())}')

print(f'Duplicates: {play_new.duplicated().sum()}')


# We check the null values

null_values = (
    pd.DataFrame(
        {f'Amount of Null Data' : play_new.isnull().sum(), 
         'Percentage of Null Data' : (
             play_new.isnull().sum()) / (len(play_new)) * (100)
        }))

null_values.style.background_gradient(cmap='Greens')


play_new["Brand"].fillna(play_new["Brand"].mode()[0], inplace=True)

play_new["Brand"].value_counts()


play_new["Material"].fillna(play_new["Material"].mode()[0], inplace=True)

play_new["Material"].value_counts()


# We fill null values with the mode groupby Material

play_new["Waterproof"] = (
    play_new["Waterproof"].fillna(
        play_new.groupby("Material")["Waterproof"].transform(lambda v: v.mode()[0])
    )
)

print(
        "Number of null values: ", play_new["Waterproof"].isnull().sum(), "\n\n",
        "Distribution of values: \n", play_new["Waterproof"].value_counts()
)


# We fill null values with the mode groupby product compartments

play_new["Weight Capacity (kg)"] = (
    play_new["Weight Capacity (kg)"].fillna(
        play_new.groupby("Compartments")["Weight Capacity (kg)"].transform("mean")
    )
)

print(
        "Number of null values: ", play_new["Weight Capacity (kg)"].isnull().sum(), "\n\n",
        "Unique values: ", play_new["Weight Capacity (kg)"].nunique()
)


# We fill null values with the mean

play_new["Size"].fillna(play_new["Size"].mode()[0], inplace=True)

print(
        "Number of null values: ", play_new["Size"].isnull().sum(), "\n\n",
        "Distribution of values: \n", play_new["Size"].value_counts()
)


# We fill null values with the mode groupby the Weight Capacity Group

play_new["Laptop Compartment"].fillna(play_new["Laptop Compartment"].mode()[0], inplace=True)

print(
        "Number of null values: ", play_new["Laptop Compartment"].isnull().sum(), "\n\n",
        "Distribution of values: \n", play_new["Laptop Compartment"].value_counts()
)


# We fill null values with the mode groupby Brand

play_new["Style"] = (
    play_new["Style"].fillna(
        play_new.groupby("Brand")["Style"].transform(lambda v: v.mode()[0])
    )
)

print(
        "Number of null values: ", play_new["Style"].isnull().sum(), "\n\n",
        "Distribution of values: \n", play_new["Style"].value_counts()
)


# We fill null values with the mode groupby Style

play_new["Color"] = (
    play_new["Color"].fillna(
        play_new.groupby("Style")["Color"].transform(lambda v: v.mode()[0])
    )
)

print(
        "Number of null values: ", play_new["Color"].isnull().sum(), "\n\n",
        "Distribution of values: \n", play_new["Color"].value_counts()
)


play_new.info()


fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.scatterplot(
    data=play_new,
    x="Brand",
    y="Price",  
    hue="Brand", 
    legend=False,
    ax=axes[0]
)

sns.boxenplot(
    data=play_new,
    x="Brand",
    y="Price",
    hue="Brand",
    #legend=False,
    ax=axes[1]
)

plt.suptitle(t="Brand by price range")
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.scatterplot(
    data=play_new,
    x="Material",
    y="Price",  
    hue="Material", 
    legend=False,
    ax=axes[0]
)

sns.boxenplot(
    data=play_new,
    x="Material",
    y="Price",
    hue="Material",
    #legend=False,
    ax=axes[1]
)

axes[1].get_legend().set_visible(False)
plt.suptitle(t="Materials by price range")
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(ncols=3, figsize=(16, 4))

sns.boxenplot(
    data=play_new,
    x="Material",
    y="Price",  
    hue="Style", 
    ax=axes[0]
)

sns.boxenplot(
    data=play_new,
    x="Material",
    y="Price",
    hue="Color",
    ax=axes[1]
)

sns.violinplot(
    data=play_new,
    x="Material",
    y="Weight Capacity (kg)",
    hue="Material",
    inner="quart",
    ax=axes[2]
)


axes[2].get_legend().set_visible(False)
plt.suptitle(t="Materials by Styles and weight")
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.boxenplot(
    data=play_new,
    x="Size",
    y="Price",  
    ax=axes[0]
)

sns.scatterplot(
    data=play_new,
    x="Weight Capacity (kg)",
    y="Price",
    ax=axes[1]
)

plt.suptitle(t="Price by Size and Capacity")
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(ncols=2, figsize=(12, 4))

sns.boxenplot(
    data=play_new,
    x="Compartments",
    y="Price",  
    ax=axes[0]
)

sns.boxenplot(
    data=play_new,
    x="Laptop Compartment",
    y="Price",
    ax=axes[1]
)

plt.suptitle(t="Price by Compartments")
plt.tight_layout()
plt.show()


play_end = play_new.copy()


# We map the variable and change the format

m_size = {"Small" : 0, "Medium" : 1, "Large" : 2}
play_end["Size"] = play_end["Size"].map(m_size)
play_end["Size"] = play_end["Size"].astype("float64")


# We apply LabelEncoder to columns with multiple classes

le = LabelEncoder()

play_end["Brand"] = le.fit_transform(play_end["Brand"])
play_end["Material"] = le.fit_transform(play_end["Material"])
play_end["Style"] = le.fit_transform(play_end["Style"])
play_end["Color"] = le.fit_transform(play_end["Color"])


# We create a df with the categorical variables

df_category = play_end[[
    "Laptop Compartment",
    "Waterproof"
]]

# We create the encoder

encoder = OneHotEncoder(sparse=False, drop="if_binary")

# We adjust and transform the data

df_cat_encoded = encoder.fit_transform(df_category)

# We convert the result to a Pandas DataFrame with the column names

df_cat_encoded = pd.DataFrame(
    df_cat_encoded, columns=encoder.get_feature_names_out(df_category.columns)
)

# We correct the names of the columns

df_cat_encoded.rename(
    columns={
        "Laptop Compartment_Yes" : "Laptop Compartment",
        "Waterproof_Yes" : "Waterproof",
    }, inplace=True)

df_cat_encoded.info()


# Numerical variables to scale

df_numeric = play_end[["Weight Capacity (kg)"]]

# We transform the data

rs = RobustScaler()

num_rs = rs.fit_transform(df_numeric)

df_scale = pd.DataFrame(
    num_rs, columns=rs.get_feature_names_out(df_numeric.columns)
)


# We create a df with the remaining variables

df_rest = play_end[[
    "Brand",
    "Material",
    "Style",
    "Color",
    "Size", 
    "Compartments",
    "Price"
]]

# We concatenate the dataframes

df_backpack = pd.concat([df_rest, df_cat_encoded, df_scale], axis=1)


df_backpack.info()


df_backpack.corr().style.background_gradient(cmap='Greens')


# We separate the target variable from the features

x_backpack = df_backpack.drop(columns="Price")
y_backpack = df_backpack["Price"]


mi_scores = mutual_info_regression(x_backpack, y_backpack)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=x_backpack.columns)
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


# We separate the data into training and test sets

x_train, x_test, y_train, y_test = (
    train_test_split(
        x_backpack, y_backpack, test_size=0.2, random_state=42
    )
)


# We create the model instance

#etr = ExtraTreesRegressor()

# Train the model with the data

#etr.fit(x_train, y_train)


#y_pred_etr = etr.predict(x_test)

#r2_etr = r2_score(y_test, y_pred_etr)

#rmse_etr = np.sqrt(mean_squared_error(y_test, y_pred_etr))

#print(f"ExtraTreesRegressor\n\nR-squared score: {r2_etr}\nRMSE: {rmse_etr}")


# Evaluate the model using cross-validation

#cv_scores_etr = cross_val_score(etr, x_train, y_train, cv=5)

#print(f"Cross-validation scores: {cv_scores_etr}")
#print(f"Mean CV accuracy: {np.mean(cv_scores_etr):.2f}")


# Create the KFold object

#kfold = KFold(n_splits=5, shuffle=True, random_state=42)


# We evaluate the model with the KFold method

#kfold_scores_etr = cross_val_score(etr, x_train, y_train, cv=kfold)

#print(f"Cross-validation Kfold scores: {kfold_scores_etr}")
#print(f"Mean CV-kfold accuracy: {np.mean(kfold_scores_etr):.2f}")


# We create the model instance

#bagr = BaggingRegressor(base_estimator=etr)

# Train the model with the data

#bagr.fit(x_train, y_train)


#y_pred_bagr = bagr.predict(x_test)

#r2_bagr = r2_score(y_test, y_pred_bagr)

#rmse_bagr = np.sqrt(mean_squared_error(y_test, y_pred_bagr))

#print(f"BaggingRegressor\n\nR-squared score: {r2_bagr}\nRMSE: {rmse_bagr}")


'''
error_rate = []

for i in range(1,10):
    knr = KNeighborsRegressor(n_neighbors=i)
    knr.fit(x_train, y_train)
    pred = knr.predict(x_test)
    error_rate.append(np.mean(pred != y_test))

plt.figure(figsize=(12,4))
plt.plot(range(1,10),error_rate, marker='o', markersize=9)
'''


# We create the model instance

#knr = KNeighborsRegressor(n_neighbors=2)

# Train the model with the data

#knr.fit(x_train, y_train)


#y_pred_knr = knr.predict(x_test)

#r2_knr = r2_score(y_test, y_pred_knr)

#rmse_knr = np.sqrt(mean_squared_error(y_test, y_pred_knr))

#print(f"KNeighborsRegressor\n\nR-squared score: {r2_knr}\nRMSE: {rmse_knr}")


# We create the model instance

#lr = LinearRegression()

# Train the model with the data

#lr.fit(x_train, y_train)


#y_pred_lr = lr.predict(x_test)

#r2_lr = r2_score(y_test, y_pred_lr)

#rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

#print(f"KNeighborsRegressor\n\nR-squared score: {r2_lr}\nRMSE: {rmse_lr}")


# We create the model instance

lgbr = LGBMRegressor(max_depth=2, num_leaves=24, verbose=0)

# Train the model with the data

lgbr.fit(x_train, y_train)


y_pred_lgbr = lgbr.predict(x_test)

r2_lgbr = r2_score(y_test, y_pred_lgbr)

rmse_lgbr = np.sqrt(mean_squared_error(y_test, y_pred_lgbr))

print(f"LGBMRegressor\n\nR-squared score: {r2_lgbr}\nRMSE: {rmse_lgbr}")


# We load the test data

df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


# We check the shape and that no duplicate data is found

print(f"Length: {len(df_test.duplicated())}")

print(f"Duplicates: {df_test.duplicated().sum()}")

print(f"Shape: {df_test.shape}")


# We start by removing the variables that we will not use

df_test_clean = df_test.drop(columns=["id"])


# We check the null values

null_values_test = (
    pd.DataFrame(
        {f'Amount of Null Data' : df_test_clean.isnull().sum(), 
         'Percentage of Null Data' : (
             df_test_clean.isnull().sum()) / (len(df_test_clean)) * (100)
        }
    ))

null_values_test.style.background_gradient(cmap='Greens')


df_test_clean["Brand"].fillna(df_test_clean["Brand"].mode()[0], inplace=True)

df_test_clean["Material"].fillna(df_test_clean["Material"].mode()[0], inplace=True)

df_test_clean["Waterproof"] = df_test_clean["Waterproof"].fillna(
        df_test_clean.groupby("Material")["Waterproof"].transform(lambda v: v.mode()[0])
)
df_test_clean["Weight Capacity (kg)"] = df_test_clean["Weight Capacity (kg)"].fillna(
        df_test_clean.groupby("Compartments")["Weight Capacity (kg)"].transform("mean")
)
df_test_clean["Size"].fillna(df_test_clean["Size"].mode()[0], inplace=True)

df_test_clean["Laptop Compartment"].fillna(df_test_clean["Laptop Compartment"].mode()[0], inplace=True)

df_test_clean["Style"] = df_test_clean["Style"].fillna(
        df_test_clean.groupby("Brand")["Style"].transform(lambda v: v.mode()[0])
)
df_test_clean["Color"] = df_test_clean["Color"].fillna(
        df_test_clean.groupby("Style")["Color"].transform(lambda v: v.mode()[0])
)

df_test_clean.info()


test_m_size = {"Small" : 0, "Medium" : 1, "Large" : 2}
df_test_clean["Size"] = df_test_clean["Size"].map(test_m_size)
df_test_clean["Size"] = df_test_clean["Size"].astype("float64")

df_test_clean["Brand"] = le.fit_transform(df_test_clean["Brand"])
df_test_clean["Material"] = le.fit_transform(df_test_clean["Material"])
df_test_clean["Style"] = le.fit_transform(df_test_clean["Style"])
df_test_clean["Color"] = le.fit_transform(df_test_clean["Color"])

test_category = df_test_clean[[
    "Laptop Compartment",
    "Waterproof"
]]

test_cat_encoded = encoder.fit_transform(test_category)

test_cat_encoded = pd.DataFrame(
    test_cat_encoded, columns=encoder.get_feature_names_out(test_category.columns)
)

test_cat_encoded.rename(
    columns={
        "Laptop Compartment_Yes" : "Laptop Compartment",
        "Waterproof_Yes" : "Waterproof",
    }, inplace=True)

test_numeric = df_test_clean[["Weight Capacity (kg)"]]

test_num_rs = rs.fit_transform(test_numeric)

test_scale = pd.DataFrame(
    test_num_rs, columns=rs.get_feature_names_out(test_numeric.columns)
)

test_rest = df_test_clean[[
    "Brand",
    "Material",
    "Style",
    "Color",
    "Size", 
    "Compartments"
]]

# We concatenate the dataframes

test_end = pd.concat([test_rest, test_cat_encoded, test_scale], axis=1)

test_end.info()


# We apply the trained model

test_predictions = lgbr.predict(test_end)


# We review the result

print('Total predictions: ', len(test_predictions), '\n')


# We create the dataframe

df_submission = pd.DataFrame({
    'id' : df_test['id'], 
    'Price' : test_predictions
})

df_submission.head(10)


# We load the test data

sample_s = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


sample_s.head()


# We review the results

print(
    f"Shape sample_submission: {sample_s.shape}",
    f"\nShape test_predictions: {df_submission.shape}"
)


# We convert the dataframe to a csv file

df_submission.to_csv("submission.csv", index=False)

