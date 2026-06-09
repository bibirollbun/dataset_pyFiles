import warnings
warnings.filterwarnings("ignore")

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import shuffle

# Model selection & evaluation
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Other tools
from statsmodels.nonparametric.smoothers_lowess import lowess



def evalR(y_val,y_pred):
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    
def evalRE(y_val,y_pred):
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}
    
def XGBRF(df,col,target,df_test):
    if target in col:
        col.remove(target)
    X = df[col]
    y = df[target]
    model = XGBRegressor()
    model.fit(X,y)
    y_pred = model.predict(df_test[col])
    df_final = pd.DataFrame({
        "id": df_test["id"].values,
        target : y_pred
    })
    return df_final


def plotcat(df,col,target,a=8,b=3):
    col_G = num_col.copy()
    col_G.append(col)
    df_G = df[col_G].groupby(col).mean()
    plt.figure(figsize = (a,b))
    plt.bar(df_G.index,df_G[target])
    plt.xlabel(col)
    plt.ylabel("Average Listening Time")
    plt.xticks(rotation = 90)
    plt.show()
    return df_G[target]
def plot_count(df,col,a=6,b=3):
    df_G_count = df[col].value_counts()
    plt.figure(figsize=(a, b))
    plt.bar(df_G_count.index, df_G_count.values)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
def scatter(df, x_col, y_col, a=8, b=6, alpha=0.6):
    plt.figure(figsize=(a, b))
    plt.scatter(df[x_col], df[y_col], alpha=alpha)
    plt.title(f"{x_col} vs. {y_col}")
    plt.xlabel(f"{x_col.replace('_', ' ').title()} (minutes)")
    plt.ylabel(f"{y_col.replace('_', ' ').title()} (minutes)")
    plt.tight_layout()
    plt.show()
def impute_episode_length(df, target_col):
    df = df.copy()
    df_missing = df[df[target_col].isna()]
    df_not_missing = df[df[target_col].notna()]
    feature_cols = df.columns.tolist()
    
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    model = XGBRegressor()
    model.fit(df_not_missing[feature_cols], df_not_missing[target_col])
    
    predicted_values = model.predict(df_missing[feature_cols])
    df.loc[df[target_col].isna(), target_col] = predicted_values
    return df


df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df["Episode_Length_minutes"] = df["Episode_Length_minutes"].apply(lambda x: 120 if x > 120 else x)
df = df.drop(columns = "id")
df = df.dropna(subset = "Number_of_Ads")
df.shape


df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
df_test["Episode_Length_minutes"] = df_test["Episode_Length_minutes"].apply(lambda x: 120 if x > 120 else x)
df_test.shape


df.head()


df.describe()


df.info()


df.isnull().sum()


target = "Listening_Time_minutes" 
print(df.columns.tolist())


num_col = [col for col in df.columns.tolist() if df[col].dtype != "object"]
cat_col = [col for col in df.columns.tolist() if df[col].dtype == "object"]


df_train = df.copy()
l = LabelEncoder()

for i in cat_col:
    combined_data = pd.concat([df_train[i], df_test[i]], axis=0)
    l.fit(combined_data)
    df_train[i] = l.transform(df_train[i])
    df_test[i] = l.transform(df_test[i])


pod = plotcat(df,"Podcast_Name",target,9,3)


plot_count(df,"Podcast_Name",10,4)


Epi = plotcat(df,"Episode_Title",target,18,3)


plot_count(df,"Episode_Title",18,4)


df_train.isnull().sum()


scatter(df,"Episode_Length_minutes",target,7,5)


mask = df_train["Episode_Length_minutes"].notna() & \
       (df_train["Listening_Time_minutes"] <= df_train["Episode_Length_minutes"] * 1.2)
df_train = df_train[mask | df_train["Episode_Length_minutes"].isna()]

scatter(df_train,"Episode_Length_minutes",target,7,5)


df_train = impute_episode_length(df_train,"Episode_Length_minutes")
df_test = impute_episode_length(df_test,"Episode_Length_minutes")


Gen = plotcat(df,"Genre",target)


plot_count(df,"Genre")


scatter(df,"Host_Popularity_percentage",target,7,5)


df_train["Host_Popularity_percentage"] = df_train["Host_Popularity_percentage"].apply(lambda x: 20 if x<20 else x)
df_train["Host_Popularity_percentage"] = df_train["Host_Popularity_percentage"].apply(lambda x: 100 if x>100 else x)
# FOR TEST 
df_test["Host_Popularity_percentage"] = df_test["Host_Popularity_percentage"].apply(lambda x: 20 if x<20 else x)
df_test["Host_Popularity_percentage"] = df_test["Host_Popularity_percentage"].apply(lambda x: 100 if x>100 else x)

scatter(df_train,"Host_Popularity_percentage",target,7,5)


Pub = plotcat(df,"Publication_Day",target)


plot_count(df,"Publication_Day")


Tim = plotcat(df,"Publication_Time",target)


plot_count(df,"Publication_Time")


scatter(df,"Guest_Popularity_percentage",target)


df_train["Guest_Popularity_percentage"] = df_train["Guest_Popularity_percentage"].apply(lambda x: 100 if x>100 else x)
# for Test
df_test["Guest_Popularity_percentage"] = df_test["Guest_Popularity_percentage"].apply(lambda x: 100 if x>100 else x)

scatter(df_train,"Guest_Popularity_percentage",target)


df_train = impute_episode_length(df_train,"Guest_Popularity_percentage")
df_test = impute_episode_length(df_test,"Guest_Popularity_percentage")


scatter(df,"Number_of_Ads",target)


df_train["Number_of_Ads"] = df_train["Number_of_Ads"].apply(lambda x: 3 if x>3 else x)
# for test
df_test["Number_of_Ads"] = df_test["Number_of_Ads"].apply(lambda x: 3 if x>3 else x)

scatter(df_train,"Number_of_Ads",target)


Sen = plotcat(df,"Episode_Sentiment",target)


plot_count(df,"Episode_Sentiment")


sub_Model = RandomForestRegressor(n_jobs=-1,n_estimators =500)
sub_Model.fit(df_train.drop(columns = target),df_train[target])
y_sub = sub_Model.predict(df_test.drop(columns="id"))
df_test["Listening_Time_minutes"] = y_sub


final = df_test[["id","Listening_Time_minutes"]]
final.to_csv("submission.csv", index=False)

