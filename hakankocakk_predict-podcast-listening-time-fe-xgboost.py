import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from xgboost import XGBRegressor



def data_collection():
    df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv", index_col=False)
    df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv", index_col=False)
    df_train, df_validation = train_test_split(df_train, test_size=0.2, random_state = 33)
    df_train.drop(columns = ['id'], inplace=True)
    df_validation.drop(columns = ["id"], inplace=True)
    df_test.drop(columns = ["id"], inplace=True)
    return df_train, df_validation, df_test

df_train, df_validation, df_test = data_collection()


def feature_engineering(df):
    df["Episode"] = df["Episode_Title"].str.replace("Episode ", "").astype(int)
    df.drop(columns = ["Episode_Title"], inplace=True)
    
    df.loc[df["Host_Popularity_percentage"] > 100, "Host_Popularity_percentage"] = 100
    df.loc[df["Guest_Popularity_percentage"] > 100, "Host_Popularity_percentage"] = 100
    
    df["Combined_Popularity"] = (
        0.76 * df["Host_Popularity_percentage"] - 0.24 * df["Guest_Popularity_percentage"]
    )
    df.loc[df["Guest_Popularity_percentage"].isnull(), "Combined_Popularity"] = 0.76 * df["Host_Popularity_percentage"]

    df["Episode_minutes_per_ad"] = df["Number_of_Ads"] / df["Episode_Length_minutes"]

    df["is_weekend"] = df["Publication_Day"].isin(["Saturday", "Sunday"]).astype(str)

    df["is_morning"] = (df["Publication_Time"] == "Morning").astype(str)
    df["is_night"] = (df["Publication_Time"] == "Night").astype(str)

    df["length_cat"] = pd.cut(df["Episode_Length_minutes"], bins=[0, 30, 60, 90, 200],
                              labels=['short', 'medium', 'long', 'very_long'])

    sentiment_map = {"Negative": -1, "Neutral": 0, "Positive": 1}
    df["sentiment_score"] = df["Episode_Sentiment"].map(sentiment_map)

    df["genre_sentiment"] = df["Genre"].astype(str) + "_" + df["sentiment_score"].astype(str)

    return df

df_train = feature_engineering(df_train)
df_validation = feature_engineering(df_validation)
df_test = feature_engineering(df_test)


def int_to_object(df):
    df["Number_of_Ads"] = df["Number_of_Ads"].astype(str)
    return df

df_train = int_to_object(df_train)
df_validation = int_to_object(df_validation)
df_test = int_to_object(df_test)


def normalization(df, train=True):
    num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage',
                'Guest_Popularity_percentage', 'Episode', 'Combined_Popularity',
                'Number_of_Ads', 'sentiment_score']
    if train:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        joblib.dump(scaler, 'standardscaler.pkl')
    else:
        loaded_scaler = joblib.load('standardscaler.pkl')
        df[num_cols] = loaded_scaler.transform(df[num_cols])
        
    return df


df_train = normalization(df_train, train=True)
df_validation = normalization(df_validation, train=False)
df_test = normalization(df_test, train=False)


def normalization_target(df, train=True):
    target = ['Listening_Time_minutes']
    if train:
        scaler = StandardScaler()
        df[target] = scaler.fit_transform(df[target])
        joblib.dump(scaler, 'standardscaler_target.pkl')
    else:
        loaded_scaler = joblib.load('standardscaler_target.pkl')
        df[target] = loaded_scaler.transform(df[target])
        
    return df


df_train = normalization_target(df_train, train=True)
df_validation = normalization_target(df_validation, train=False)


categorical_features = ['Podcast_Name', 'Genre', 'Publication_Day',
                        'Publication_Time', 'Number_of_Ads', 'Episode_Sentiment',
                        'is_weekend', 'is_morning', 'is_night', 'length_cat',
                        'genre_sentiment']

for col in categorical_features:
    df_train[col] = df_train[col].astype('category')
    df_validation[col] = df_validation[col].astype('category')
    df_test[col] = df_test[col].astype('category')
    


def plot_importance(model, df):

    importance = model.feature_importances_
    features = df.columns

    indices = importance.argsort()[-20:]
    plt.rcParams.update({'font.size': 5})

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices], align="center")
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("Özellik Önemi")
    plt.title("")
    plt.show()


def split(df):
    y = df["Listening_Time_minutes"]
    X = df.drop(["Listening_Time_minutes"], axis=1)

    return X, y

X_train, y_train = split(df_train)
X_val, y_val = split(df_validation)


def xgboost_final_model(x_train, y_train, x_val, y_val):

    params = {

        'n_estimators': 1000,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.6,
        'colsample_bytree': 0.9,
        'gamma': 1.4,
        'reg_alpha': 2,
        'reg_lambda': 3,
        'random_state': 42,
        'tree_method' : "hist",
        'enable_categorical' : True,
        'device' : "cuda",
        'early_stopping_rounds' : 25,
        'verbosity' : 0,
        'eval_metric': 'rmse'
    }
    


    xgboost_final = XGBRegressor(**params).fit(
        x_train, y_train,
        eval_set = [(x_train, y_train), (x_val, y_val)]
    )


    return xgboost_final

xgboost_final = xgboost_final_model(X_train, y_train, X_val, y_val)


def submission(model, model_name):
    predict = model.predict(df_test)
    predict = predict.reshape(-1, 1)
    loaded_scaler = joblib.load('standardscaler_target.pkl')
    predict_transform = loaded_scaler.inverse_transform(predict)
    submission_df = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv",encoding='utf-8',low_memory=False)
    submission_df["Listening_Time_minutes"] = predict_transform
    submission_df.to_csv(f'/kaggle/working/{model_name}_submission.csv', index=False)

submission(xgboost_final, "xgboost_cat_v5")

