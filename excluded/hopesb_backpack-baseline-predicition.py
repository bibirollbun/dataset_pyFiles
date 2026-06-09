# import all necessary libraries.
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px

import sklearn
from sklearn.metrics import mean_squared_error
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from cuml.preprocessing import TargetEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingRegressor, StackingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor, plot_importance
from tqdm.notebook import tqdm
import optuna
import warnings
warnings.simplefilter("ignore")


# Reading the files
test_filepath = "/kaggle/input/playground-series-s5e2/test.csv"
train_filepath = "/kaggle/input/playground-series-s5e2/train.csv"
df = pd.read_csv(train_filepath, index_col="id")
df_test = pd.read_csv(test_filepath, index_col="id")


extra_filepath = "/kaggle/input/playground-series-s5e2/training_extra.csv"
df_extra = pd.read_csv(extra_filepath, index_col="id")
df_extra.shape


print(df.shape)
df.head()


df.info()


df.isnull().sum().sort_values()


df.nunique().sort_values()


df.describe()


def wrangle(filepaths, train=True):
    if train: 
        df_1 = pd.read_csv(filepaths[0], index_col= "id")
        df_2 = pd.read_csv(filepaths[-1], index_col="id")
        # Merge the dataset.
        df = pd.concat([df_1, df_2])
    else:
        df = pd.read_csv(filepaths, index_col="id")

    
    # Fill the weight with mean.
    df["Weight Capacity (kg)"] = df["Weight Capacity (kg)"].fillna(df["Weight Capacity (kg)"].mean())
    # fill the color.
    df["Color"] = df["Color"].fillna("Multi-color")
    # fill the Brand with unknown brand.
    df["Brand"] = df["Brand"].fillna("Unknown")
    # fill the size.
    df["Size"] = df["Size"].fillna("Others")
    # fill the waterproof.
    df["Waterproof"] = df["Waterproof"].fillna("Undetermined")
    # fill the Laptop compartment.
    df["Laptop Compartment"] = df["Laptop Compartment"].fillna("Unknown")
    # Fill the material ans style.
    df["Material"] = df["Material"].fillna("Unknown")
    df["Style"] = df["Style"].fillna("Unknown")

    df["Compartments"] = df["Compartments"].astype(int)
    df["Weight Capacity (kg)"] = round(df["Weight Capacity (kg)"], 3)
    df["Weight Class"] = df["Weight Capacity (kg)"].apply(lambda x: "Minimum" if x < 10 else(
        "Above_Minimum" if x < 15 else(
            "Medium" if x < 25 else "High"
        )
    ))

    return df


filepaths = [train_filepath, extra_filepath]
df = wrangle(filepaths)
df.isnull().sum()


df.head()


target = "Price"
X = df.drop(columns= target)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


test_df = wrangle(test_filepath, train=False)


# Create a function that takes in a list of algortithm and then make the prediction then save the prediction.
def score_predict(algorithms, X_train, X_test, y_train, y_test, test_df=None, save=True): 

    data_dict = {}
    # loop throught the algorithms.
    for algorithm in tqdm(algorithms, desc="Starting"):
        # make the pipeline
        model = make_pipeline(
            OrdinalEncoder(),
            StandardScaler(),
            algorithm
        )
        # Fitting the model.
        alg_name = list(model.named_steps.keys())[-1]
        print(f"Fitting the {alg_name}")
        model.fit(X_train, y_train)
        # Get the score.
        print("Making Prediction")
        y_test_pred = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, y_test_pred))
        data_dict[alg_name] = score

        if save == True:
            y_pred = model.predict(test_df)
            pd.DataFrame({"Price": y_pred}, index=test_df.index).to_csv(f"{alg_name}.csv")

    score_df = pd.DataFrame(data_dict, index=["Accuracy"]).T
    return score_df


xgb_params = {'max_depth': 8, 
              'n_estimators': 1701, 
              'learning_rate': 0.006899755927601937, 
              'min_child_weight': 0.2751236938771351, 
              'reg_alpha': 0.527480714707444, 
              'reg_lambda': 13.198806851176458, 
              'colsample_bylevel': 0.9772481366672879, 
              'colsample_bytree': 0.7774789504388184, 
              'colsample_bynode': 0.8541180728630634}
lgb_params = {'max_depth': 17, 
              'n_estimators': 2995, 
              'learning_rate': 0.00943875220877273, 
              'min_child_weight': 0.9260630565539196, 
              'reg_alpha': 0.9200603427801367, 
              'reg_lambda': 1.4831257707229202, 
              'colsample_bylevel': 0.5223344478716733, 
              'colsample_bytree': 0.9937966786732069, 
              'colsample_bynode': 0.8479188035499157}
cat_params = {'max_depth': 8, 
              'n_estimators': 2830, 
              'learning_rate': 0.009960055354186655}
algorithms = [LGBMRegressor(**lgb_params, verbose=-1, random_state=42),
             XGBRegressor(**xgb_params, random_state=42),
             CatBoostRegressor(verbose=0, random_state=42)]
score = score_predict(algorithms, X_train, X_test, y_train, y_test, test_df)
score


lgb = make_pipeline(
    OrdinalEncoder(),
    StandardScaler(),
    LGBMRegressor(**lgb_params, verbose=-1, random_state=42)
)
xgb = make_pipeline(
    OrdinalEncoder(),
    StandardScaler(),
    XGBRegressor(**xgb_params, random_state=42)
)

estimators = [("lgb", lgb), ("xgb", xgb)]
vote = VotingRegressor(estimators=estimators, weights=[2, 1])
vote.fit(X_train, y_train)
y_pred_test = vote.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred_test)))
y_pred = vote.predict(test_df)
pd.DataFrame({"Price": y_pred}, index=test_df.index).to_csv(f"submission.csv")


stack = StackingRegressor(estimators, CatBoostRegressor(verbose=0, random_state=42), cv=5)
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))
y_pred = vote.predict(test_df)
pd.DataFrame({"Price": y_pred}, index=test_df.index).to_csv(f"submission_stack.csv")

