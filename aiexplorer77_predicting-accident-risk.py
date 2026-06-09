import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.preprocessing import LabelEncoder , OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor




df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df_sample = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


df_train.head()


df_train.info()


df_train.drop(columns = "id", inplace = True)


df_train.head()


df_train["road_type"].value_counts()


ordinal_encoder = OrdinalEncoder(categories = [["rural" ,"urban","highway"]])
df_train["road_type"] = ordinal_encoder.fit_transform(df_train[["road_type"]])


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


ordinal_encoder = OrdinalEncoder(categories = [["morning", "afternoon", "evening"]])
df_train["time_of_day"] = ordinal_encoder.fit_transform(df_train[["time_of_day"]])


df_train


df_train.isnull().sum()


print(df_train["num_lanes"].value_counts())

print(df_train["num_lanes"].value_counts(normalize = True) * 100)


print("\n*************** Value Counts ****************\n")

print(df_train["curvature"].value_counts())

print("\n******************************** Percentage Count *******************************\n")

print(df_train["curvature"].value_counts(normalize = True) * 100)

print("\n******************************** Box Plot ********************************\n")

plt.figure(figsize = (10, 6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["curvature"],
   
)

plt.title("Box Plot of Curvature", fontsize = 20, fontweight = "bold")
plt.ylabel("curvature",  fontsize = 12, fontweight = "bold")
plt.show()


print("\n*************** Value Counts ****************\n")

print(df_train["speed_limit"].value_counts())

print("\n******************************** Percentage Count *******************************\n")

print(df_train["speed_limit"].value_counts(normalize = True) * 100)



print("\n*************** Value Counts ****************\n")

print(df_train["num_reported_accidents"].value_counts())

print("\n******************************** Percentage Count *******************************\n")

print(df_train["num_reported_accidents"].value_counts(normalize = True) * 100)

print("\n******************************** Box Plot ********************************\n")

plt.figure(figsize = (10, 6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["num_reported_accidents"],
   
)

plt.title("Box Plot of num_reported_accidents", fontsize = 20, fontweight = "bold")
plt.ylabel("num_reported_accidents",  fontsize = 12, fontweight = "bold")
plt.show()


df_train["num_reported_accidents"]  = df_train["num_reported_accidents"].replace({
    4:3,
    5:2,
    6:1,
    7:3
    
    
})


print("\n*************** Value Counts ****************\n")

print(df_train["num_reported_accidents"].value_counts())

print("\n******************************** Percentage Count *******************************\n")

print(df_train["num_reported_accidents"].value_counts(normalize = True) * 100)

print("\n******************************** Box Plot ********************************\n")

plt.figure(figsize = (10, 6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["num_reported_accidents"],
   
)

plt.title("Box Plot of num_reported_accidents", fontsize = 20, fontweight = "bold")
plt.ylabel("num_reported_accidents",  fontsize = 12, fontweight = "bold")
plt.show()


print("\n*************** Value Counts ****************\n")

print(df_train["accident_risk"].value_counts())

print("\n******************************** Percentage Count *******************************\n")

print(df_train["accident_risk"].value_counts(normalize = True) * 100)

print("\n********************************** Box Plot *********************************\n")

plt.figure(figsize = (15,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["accident_risk"],
    vert = True,
    patch_artist = True,
    boxprops = dict(facecolor = "Lightblue", color = "blue"),
    medianprops = dict(color = "red"),
    whiskerprops = dict(color = "blue"),
    capprops = dict(color = "blue")

)

plt.title("Box Plot of accident_risk", fontsize = 28, fontweight = "bold", color = "black")
plt.ylabel("accident_risk",fontsize = 16)
plt.show()


Q1 = np.percentile(df_train["accident_risk"], 25)
Q3 = np.percentile(df_train["accident_risk"], 75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR 
upper_bound = Q3 + 1.5 * IQR 

df_train.loc[(df_train["accident_risk"] < lower_bound) | (df_train["accident_risk"] > upper_bound), "accident_risk"] = df_train["accident_risk"].median()



print("\n********************************** Box Plot *********************************\n")

plt.figure(figsize = (15,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["accident_risk"],
    vert = True,
    patch_artist = True,
    boxprops = dict(facecolor = "Lightblue", color = "blue"),
    medianprops = dict(color = "red"),
    whiskerprops = dict(color = "blue"),
    capprops = dict(color = "blue")

)

plt.title("Box Plot of accident_risk", fontsize = 28, fontweight = "bold", color = "black")
plt.ylabel("accident_risk",fontsize = 16)
plt.show()


df_train 


categorical_cols = ["road_type","num_lanes","lighting","weather","road_signs_present","public_road","time_of_day","holiday","num_reported_accidents"]

for col in categorical_cols:
    plt.figure(figsize = (10,6), dpi = 100, facecolor = "white", edgecolor = "black")
    plt.bar(
        df_train[col].value_counts().index,
        df_train[col].value_counts().values,
        color = "grey",
        linewidth = 0.0,
        edgecolor = "black"
    )
    
    plt.title(f'Bar Plot of {col}', fontsize = 20, fontweight = "bold", color = "black", loc = "center")
    plt.xlabel(f"{col}", fontsize = 16, fontweight = "bold", color = "black")
    plt.ylabel(f"Count of {col}" , fontsize = 16, fontweight = "bold", color = "black" )
    plt.show()



continuous_cols = ["curvature", "speed_limit", "accident_risk"]

for col in continuous_cols:
    plt.figure(figsize = (20,6), dpi = 100, facecolor = "white", edgecolor = "black")
    plt.hist(
        df_train[col],
        bins = int(np.sqrt(df_train[col].nunique())),
        color = "gray"
    
    )
    
    plt.title(f"Histogram of {col}", fontsize = 20, color = "black", fontweight = "bold", loc = "center" )
    plt.xlabel(f"{col}",fontsize = 16, color = "black", fontweight = "bold")
    plt.ylabel(f" Count of {col}",fontsize = 16, color = "black", fontweight = "bold")
    plt.show()




df_train


df_train.corr()


sns.heatmap(df_train.corr(), cmap = "icefire")



fig, axes = plt.subplots(6,2, figsize = (25,40))
axes  = axes.flatten()

for ax, i in zip(axes, df_train.columns.tolist()):
    sns.scatterplot(data = df_train, x = i, y = "accident_risk", ax = ax)
    ax.set_title(f" Scatter Plot describe correlation between {i} and accident_risk")
    ax.set_xlabel(i)
    ax.set_ylabel("accident_risk")
    ax.set_xlim(0,df_train[i].max() * 1.1)
    for i in ax.get_xticklabels():
        i.set_rotation(45)
        
plt.tight_layout()
plt.show()


df_train.describe()


X = df_train.drop(columns = "accident_risk")
Y = df_train["accident_risk"]                      # Target variable


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


Model_linear_regression = LinearRegression()


Model_linear_regression.fit(X_train, Y_train)


Y_pred = Model_linear_regression.predict(X_test)


r2_score(Y_test,Y_pred)


plt.figure(figsize = (26,6), dpi = 100)
plt.scatter(
    Y_test[:1000],
    Y_pred[:1000]
)
plt.show()


model_lightGBM = LGBMRegressor(max_depth = 6, n_estimators = 4000, learning_rate = 0.04, random_state = 42)


model_lightGBM.fit(X_train, Y_train)


Y_pred = model_lightGBM.predict(X_test)


r2_score(Y_test, Y_pred)


plt.figure(figsize = (26,6), dpi = 100)
plt.scatter(
    Y_test[:1000],
    Y_pred[:1000]
)
plt.show()


model_XGBRegressor = XGBRegressor(
    objective = "reg:squarederror",   # standard Regresion objective
    eval_metric = "rmse",
    n_estimators = 1000,
    learning_rate = 0.05,
    max_depth = 6,
    subssample = 0.8,
    colsample_bytes= 0.8,
    random_state = 42,
    n_jobs =- 1
    
)


model_XGBRegressor.fit(X_train, Y_train)


Y_pred = model_XGBRegressor.predict(X_test)


r2_score(Y_test, Y_pred)


df_test





# df_test.drop(columns = "id", inplace = True)


ordinal_encoder = OrdinalEncoder(categories = [["rural" ,"urban","highway"]])
df_test["road_type"] = ordinal_encoder.fit_transform(df_test[["road_type"]])


ordinal_encoder = OrdinalEncoder(categories = [["dim", "daylight", "night"]])
df_test["lighting"] = ordinal_encoder.fit_transform(df_test[["lighting"]])    


ordinal_encoder = OrdinalEncoder(categories = [["foggy", "rainy", "clear"]])
df_test["weather"] = ordinal_encoder.fit_transform(df_test[["weather"]])


ordinal_encoder = OrdinalEncoder(categories = [[False, True]])
df_test["road_signs_present"] = ordinal_encoder.fit_transform(df_test[["road_signs_present"]])
df_test["public_road"] = ordinal_encoder.fit_transform(df_test[["public_road"]])
df_test["holiday"] = ordinal_encoder.fit_transform(df_test[["holiday"]])
df_test["school_season"] = ordinal_encoder.fit_transform(df_test[["school_season"]])


ordinal_encoder = OrdinalEncoder(categories = [["morning", "afternoon", "evening"]])
df_test["time_of_day"] = ordinal_encoder.fit_transform(df_test[["time_of_day"]])


df_test


df_sample 


df_test_modified = df_test.drop(columns = "id")


df_test["accident_risk"] = model_XGBRegressor.predict(df_test_modified)


df_test


df_submission = pd.DataFrame({
    "id" : df_test["id"],
    "accident_risk" : df_test["accident_risk"]
})


df_submission.to_csv("submission.csv", index = False)


df_submission.to_csv("/kaggle/working/submission.csv", index = False)




