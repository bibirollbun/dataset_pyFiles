import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from sklearn.model_selection import RandomizedSearchCV


df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df_sample = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


df_train.head()


df_train.info()


df_train.drop(columns = "id", inplace = True)


df_train.head()


df_train["road_type"].value_counts()


ordinal_ecoder = OrdinalEncoder(categories=[['rural', 'urban', 'highway']])
df_train["road_type"] = ordinal_ecoder.fit_transform(df_train[["road_type"]])


df_train


df_train["lighting"].value_counts()


ordinal_encoder = OrdinalEncoder(categories = [["dim", "daylight", "night"]])
df_train["lighting"] = ordinal_encoder.fit_transform(df_train[["lighting"]])


df_train


df_train["weather"].value_counts()


ordinal_encoder = OrdinalEncoder(categories = [["foggy", "rainy", "clear"]])
df_train["weather"] = ordinal_encoder.fit_transform(df_train[["weather"]])


df_train


df_train["road_signs_present"].value_counts()


ordinal_encoder = OrdinalEncoder(categories = [[False, True]])
df_train["road_signs_present"] = ordinal_encoder.fit_transform(df_train[["road_signs_present"]])
df_train["public_road"] = ordinal_encoder.fit_transform(df_train[["public_road"]])
df_train["holiday"] = ordinal_encoder.fit_transform(df_train[["holiday"]])
df_train["school_season"] = ordinal_encoder.fit_transform(df_train[["school_season"]])


df_train


df_train["time_of_day"].value_counts()


ordinal_encoder = OrdinalEncoder(categories = [["morning","afternoon", "evening" ]])
df_train["time_of_day"] = ordinal_encoder.fit_transform(df_train[["time_of_day"]])


df_train


df_train.isnull().sum()


print(df_train["num_lanes"].value_counts())

print(df_train["num_lanes"].value_counts(normalize = True) * 100)


print("=====================Value Counts==================\n")
print(df_train["curvature"].value_counts())

print("\n=====================Percentage Count================\n")
print(df_train["curvature"].value_counts(normalize = True) * 100)

print("\n======================Box Plot========================\n")

plt.figure(figsize = (10, 6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["curvature"],
    vert = True
)
plt.title("Box Plot of Curvature", fontsize = 20, fontweight = "bold", color = "black")
plt.ylabel("curvature", fontsize = 16, fontweight = "bold", color = "black")

plt.show()


print("=====================Value Counts==================\n")
print(df_train["speed_limit"].value_counts())

print("\n=====================Percentage Count================\n")
print(df_train["speed_limit"].value_counts(normalize = True) * 100)


print("=====================Value Counts==================\n")
print(df_train["num_reported_accidents"].value_counts())

print("\n=====================Percentage Count================\n")
print(df_train["num_reported_accidents"].value_counts(normalize = True) * 100)

print("\n======================Box Plot========================\n")

plt.figure(figsize = (10, 6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["num_reported_accidents"],
    vert = True
)
plt.title("Box Plot of num_reported_accidents", fontsize = 20, fontweight = "bold", color = "black")
plt.ylabel("num_reported_accidents", fontsize = 16, fontweight = "bold", color = "black")

plt.show()


df_train["num_reported_accidents"] = df_train["num_reported_accidents"].replace({
    4:3,
    5:2,
    6:1,
    7:3
})


print("=====================Value Counts==================\n")
print(df_train["num_reported_accidents"].value_counts())

print("\n=====================Percentage Count================\n")
print(df_train["num_reported_accidents"].value_counts(normalize = True) * 100)

print("\n======================Box Plot========================\n")

plt.figure(figsize = (10, 6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["num_reported_accidents"],
    vert = True
)
plt.title("Box Plot of num_reported_accidents", fontsize = 20, fontweight = "bold", color = "black")
plt.ylabel("num_reported_accidents", fontsize = 16, fontweight = "bold", color = "black")

plt.show()


print("==========================Value Counts======================\n")
print(df_train["accident_risk"].value_counts())

print("\n===================Percentage Counts=======================\n")
print(df_train["accident_risk"].value_counts(normalize = True) * 100)

print("\n==============================Box Plot========================\n")
plt.figure(figsize = (15, 6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["accident_risk"],
    vert = True,
    patch_artist = True,
    boxprops=dict(facecolor="lightblue", color="blue"),
    medianprops=dict(color="red"),
    whiskerprops=dict(color="blue"),
    capprops=dict(color="blue")
)
plt.title("Box Plot of accident_risk", fontsize = 20, fontweight = "bold", color = "black")
plt.ylabel("accident_risk", fontsize  = 16)
plt.grid(axis = "y", linestyle = "--", alpha = 0.7)
plt.show()


Q1 = np.percentile(df_train["accident_risk"], 25)
Q3 = np.percentile(df_train["accident_risk"], 75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_train.loc[(df_train["accident_risk"] < lower_bound) | (df_train["accident_risk"] > upper_bound), "accident_risk"] = df_train["accident_risk"].median()


plt.figure(figsize = (15, 6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["accident_risk"],
    vert = True,
    patch_artist = True,
    boxprops=dict(facecolor="lightblue", color="blue"),
    medianprops=dict(color="red"),
    whiskerprops=dict(color="blue"),
    capprops=dict(color="blue")
)
plt.title("Box Plot of accident_risk", fontsize = 20, fontweight = "bold", color = "black")
plt.ylabel("accident_risk", fontsize  = 16)
plt.grid(axis = "y", linestyle = "--", alpha = 0.7)
plt.show()


df_train


categorical_cols = ["road_type", "num_lanes", "lighting", "weather", "road_signs_present", "public_road", "time_of_day", "holiday", "school_season", "num_reported_accidents"]

for col in categorical_cols:
    plt.figure(figsize = (10, 6), dpi  = 100, facecolor = "white", edgecolor = "black")
    plt.bar(
        df_train[col].value_counts().index,
        df_train[col].value_counts().values,
        color = "gray",
        linewidth = 0.5,
        edgecolor = "black",
    )
    plt.title(f'Bar Plot of Column {col}', fontsize = 20, fontweight = "bold", color = "black", loc = "center")
    plt.xlabel(f"{col}", fontsize = 16, fontweight = "bold", color = "black")
    plt.ylabel(f"Count of {col}", fontsize = 16, fontweight = "bold", color = "black")
    plt.show()


continuous_cols = ["curvature", "speed_limit", "accident_risk"]

for col in continuous_cols:
    plt.figure(figsize = (25, 6), dpi  = 100, facecolor = "white", edgecolor = "black")
    plt.hist(
        df_train[col],
        bins = int(np.sqrt(df_train[col].nunique())),
        color = "gray",
    )
    plt.title(f"Hist Plot of {col}", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
    plt.xlabel(f"{col}",  fontsize = 16, fontweight = "bold", color = "black")
    plt.ylabel(f"Count of {col}",  fontsize = 16, fontweight = "bold", color = "black")
    plt.show()


df_train


sns.heatmap(df_train.corr(), cmap = "icefire")


df_train.describe()


X = df_train.drop(columns = "accident_risk")
Y = df_train["accident_risk"] # This is the target variable 


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# Define the parameter grid for XGBoost
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Initialize XGBoost model
xgb_model = XGBRegressor(random_state=42, n_jobs=-1)

# Create RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit the model
random_search.fit(X_train, Y_train)
# Get the best model
best_xgb_model = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)


df_test


# df_test.drop(columns = "id", inplace = True)


ordinal_ecoder = OrdinalEncoder(categories=[['rural', 'urban', 'highway']])
df_test["road_type"] = ordinal_ecoder.fit_transform(df_test[["road_type"]])


ordinal_encoder = OrdinalEncoder(categories = [["dim", "daylight", "night"]])
df_test["lighting"] = ordinal_encoder.fit_transform(df_test[["lighting"]])


ordinal_encoder = OrdinalEncoder(categories = [["foggy", "rainy", "clear"]])
df_test["weather"] = ordinal_encoder.fit_transform(df_test[["weather"]])


ordinal_encoder = OrdinalEncoder(categories = [[False, True]])
df_test["road_signs_present"] = ordinal_encoder.fit_transform(df_test[["road_signs_present"]])
df_test["public_road"] = ordinal_encoder.fit_transform(df_test[["public_road"]])
df_test["holiday"] = ordinal_encoder.fit_transform(df_test[["holiday"]])
df_test["school_season"] = ordinal_encoder.fit_transform(df_test[["school_season"]])


ordinal_encoder = OrdinalEncoder(categories = [["morning","afternoon", "evening" ]])
df_test["time_of_day"] = ordinal_encoder.fit_transform(df_test[["time_of_day"]])


df_test


df_test_modified = df_test.drop(columns = "id")


df_test["accident_risk"] = best_xgb_model.predict(df_test_modified)


df_test


df_submission = pd.DataFrame({
    'id': df_test['id'], 
    'accident_risk' : df_test['accident_risk']
})


df_submission.to_csv('submission.csv', index = False)


df_submission.to_csv("/kaggle/working/submission.csv", index = False)

