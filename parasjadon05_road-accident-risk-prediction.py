import pandas as pd
import numpy as np


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df.head()


df = df.drop("id", axis=1)
df.head()


df.describe()


df.info()


df["road_type"].value_counts()


df["num_lanes"].value_counts()


df["speed_limit"].value_counts()


df["lighting"].value_counts()


df["weather"].value_counts()


df["num_reported_accidents"].value_counts()


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(df.drop(["accident_risk"], axis=1), df["accident_risk"], test_size=0.2)


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


preprocessor = ColumnTransformer([
    # OneHot for categorical nominal columns
    ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first"),
     ["road_type", "weather", "road_signs_present", "public_road", "time_of_day", "holiday"]),

    # Ordinal for ordered categorical columns
    ("oe_num_lanes", OrdinalEncoder(categories=[[1, 2, 3, 4]], handle_unknown="use_encoded_value", unknown_value=-1),
     ["num_lanes"]),

    ("oe_speed_limit", OrdinalEncoder(categories=[[25, 35, 45, 60, 70]], handle_unknown="use_encoded_value", unknown_value=-1),
     ["speed_limit"]),

    ("oe_lighting", OrdinalEncoder(categories=[["night", "dim", "daylight"]], handle_unknown="use_encoded_value", unknown_value=-1),
     ["lighting"]),

    # Numerical scaling
    ("minmax", MinMaxScaler(), ["curvature"])
], remainder="passthrough")


model = LinearRegression()


pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


pipe.fit(x_train, y_train)


pred = pipe.predict(x_test)


from sklearn.metrics import r2_score


r2_score(pred, y_test)


import pickle as pkl


pkl.dump(pipe, open("Road_Accident_Risk_Prediction_Model.pkl", "wb"))

