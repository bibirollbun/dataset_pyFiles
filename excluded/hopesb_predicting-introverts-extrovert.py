import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from category_encoders import OneHotEncoder, TargetEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import optuna
import warnings
from tqdm.notebook import tqdm
warnings.simplefilter("ignore")


train_filepath = "/kaggle/input/playground-series-s5e7/train.csv"
test_filepath = "/kaggle/input/playground-series-s5e7/test.csv"


def wrangle(filepath, process = False):
    """
    Cleaning and Processing the data ready for Model.
    """
    # Read the CSV file.
    df = pd.read_csv(filepath, index_col="id")
    col_norm = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
       'Friends_circle_size', 'Post_frequency']

    df["New"] = df["Drained_after_socializing"] + " " + df["Stage_fear"]
    df["New2"] = df["Social_event_attendance"] + df["Time_spent_Alone"]
    if process:
        df["Stage_fear"].fillna(df["Stage_fear"].mode()[0], inplace=True)
        df["Drained_after_socializing"].fillna(df["Drained_after_socializing"].mode()[0], inplace=True)
    
        # Fill the null values.
        agg_1_tsa = df.groupby(["Stage_fear", "Drained_after_socializing", "Going_outside", "Post_frequency"])["Time_spent_Alone"].transform("mean")
        agg_2_tsa = df.groupby(["Stage_fear", "Drained_after_socializing", "Going_outside"])["Time_spent_Alone"].transform("mean")
        agg_3_tsa = df.groupby(["Stage_fear", "Drained_after_socializing"])["Time_spent_Alone"].transform("mean")
        df["Time_spent_Alone"].fillna(agg_1_tsa, inplace=True)
        df["Time_spent_Alone"].fillna(agg_2_tsa, inplace=True)
        df["Time_spent_Alone"].fillna(agg_3_tsa, inplace=True)
        df["Time_spent_Alone"].fillna(df["Time_spent_Alone"].mean(), inplace=True)
    
        cols = ['Friends_circle_size', 'Social_event_attendance', 'Post_frequency', 'Going_outside']
        for col in cols:
            agg_1_tsa = df.groupby(["Stage_fear", "Drained_after_socializing"])[col].transform("mean")
            df[col].fillna(agg_1_tsa, inplace=True)
            df[col].fillna(df[col].mean(), inplace=True)
    
        ohe = OneHotEncoder(use_cat_names=True)
        ohe_df = ohe.fit_transform(df[["Stage_fear", "Drained_after_socializing"]])

        df = pd.concat([df, ohe_df], axis=1)

        for col in col_norm:
            df[col] = np.log1p(df[col])

        df.drop(columns=["Stage_fear", "Drained_after_socializing"], inplace=True)
        return df
    else:
        for col in col_norm:
            df[col] = np.log1p(df[col])
        return df


df_train = wrangle(train_filepath, process=False)
df_test = wrangle(test_filepath, process=False)
df_train.head()


df_train.isnull().sum().sort_values()


df_train.shape


df_train["Personality"].value_counts(normalize=True)


df_train.describe()


target = "Personality"
X = df_train.drop(columns=target)
y = df_train[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print(X_train.shape, y_train.shape)


def score_sub(models, X_train, X_test, y_train, y_test, test_df):
    score_dict = {}
    y_train = y_train.replace({"Extrovert": 1, "Introvert": 0})
    y_test = y_test.replace({"Extrovert": 1, "Introvert": 0})
    for model in tqdm(models, desc="Processing"):
        model = make_pipeline(
                    OrdinalEncoder(),
                    SimpleImputer(),
                    StandardScaler(),
                    model
                )
        model.fit(X_train, y_train)
        alg_name = list(model.named_steps.keys())[-1]
        pred = model.predict(X_test)
        score = accuracy_score(y_test, pred)
        # Submission.
        test_pred = model.predict(test_df)
        test_pred = test_pred
        sub_df = pd.DataFrame({"Personality": test_pred}, index=test_df.index)
        sub_df["Personality"] = sub_df["Personality"].replace({1: "Extrovert", 0: "Introvert"})
        sub_df.to_csv(f"{alg_name}.csv")
        print(f"\nSubmission File for {alg_name} Created.")
        score_dict[alg_name]= score
    df = pd.DataFrame(score_dict, index=["Score"])
    return df.T.sort_values("Score", ascending=False)


algorithms = [GradientBoostingClassifier(random_state=42),
             CatBoostClassifier(random_state=42, verbose=0),
             XGBClassifier(random_state=42),
             LGBMClassifier(random_state=42, verbose=-1)]
score_sub(algorithms, X_train, X_test, y_train, y_test, df_test)




