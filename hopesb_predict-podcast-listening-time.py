# import libraries.
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from category_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm
import optuna

import warnings
warnings.simplefilter("ignore")


%load_ext cudf.pandas
train_filepath = "/kaggle/input/playground-series-s5e4/train.csv"
test_filepath = "/kaggle/input/playground-series-s5e4/test.csv"


# Create a function.
def wrangle(filepath, train=True):
    """
    This will transform the dataset, making it fit for visualization and modelling.

    --------
    Parameter:
        filepath: str to the csv file.
        eda: boolean 

    --------
    Return:
        DataFrame.
    """
    df = pd.read_csv(filepath, index_col="id")

    null_col = ["Number_of_Ads", "Episode_Length_minutes", "Guest_Popularity_percentage"]

    for col in null_col:
        agg = df.groupby(['Podcast_Name', 'Episode_Title', 'Episode_Sentiment', 'Genre'])[col].transform("mean")
        df[col].fillna(agg, inplace=True)
        agg2 = df.groupby(['Podcast_Name', 'Episode_Title'])[col].transform("mean")
        df[col].fillna(agg2, inplace=True)

    cols = ["Host_Popularity_percentage", "Guest_Popularity_percentage", "Episode_Length_minutes", "Number_of_Ads"]
    for col in cols:
        if col == "Episode_Length_minutes":
            df[col] = df[col].where(df[col] < 150, 150)
        elif col == "Number_of_Ads":
            df[col] = df[col].where(df[col] < 15, 15)
        else:
            df[col] = df[col].where(df[col] < 100, 100)

    df["pod_guest_host"] = df.groupby(
        ["Podcast_Name", "Guest_Popularity_percentage", "Host_Popularity_percentage"]
    )["Episode_Length_minutes"].transform("mean")


    df["new2"] = df["Genre"] + df["Publication_Day"] + df["Publication_Time"] + df["Episode_Sentiment"]

    df["Episode_Num"] = df["Episode_Title"].str.split(" ", expand=True)[1]

    encoded_columns = []

    selected_comb = [
        # 2-interaction
        ['Episode_Length_minutes', 'Host_Popularity_percentage'],
        ['Episode_Length_minutes', 'Guest_Popularity_percentage'],
        ['Episode_Length_minutes', 'Number_of_Ads'],
        ['Episode_Num', 'Host_Popularity_percentage'],
        ['Episode_Num', 'Guest_Popularity_percentage'],
        ['Episode_Num', 'Number_of_Ads'],    
        ['Host_Popularity_percentage', 'Guest_Popularity_percentage'],
        ['Host_Popularity_percentage', 'Number_of_Ads'],
        ['Host_Popularity_percentage', 'Episode_Sentiment'],
        ['Episode_Length_minutes', 'Podcast_Name'],
        ['Episode_Num', 'Podcast_Name'],  
        ['Guest_Popularity_percentage', 'Podcast_Name'],
        
    
        
        # 3-interaction
        ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage'],
        ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage'],
        ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads'],
        ['Episode_Length_minutes', 'Episode_Num', 'Episode_Sentiment'],
        ['Episode_Length_minutes', 'Episode_Num', 'Publication_Day'],
        ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage'],
        ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads'],
        ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Episode_Sentiment'],
        ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day'],
        ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Time'],
        ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads'],
        ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Publication_Day'],
        ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Publication_Time'],
        ['Episode_Length_minutes', 'Number_of_Ads', 'Episode_Sentiment'],
        ['Episode_Length_minutes', 'Number_of_Ads', 'Publication_Day'],
        ['Episode_Length_minutes', 'Episode_Sentiment', 'Publication_Time'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Day'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Time'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Genre'],
        ['Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads'],
        ['Episode_Num', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
        ['Episode_Num', 'Guest_Popularity_percentage', 'Publication_Day'],
        ['Episode_Num', 'Guest_Popularity_percentage', 'Publication_Time'],
        ['Episode_Num', 'Guest_Popularity_percentage', 'Genre'],
        ['Episode_Num', 'Number_of_Ads', 'Episode_Sentiment'],
        ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads'],
        ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
        ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Day'],
        ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Time'],
        ['Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
    
        ['Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
        ['Guest_Popularity_percentage', 'Number_of_Ads', 'Genre'],   
        
        # 4-interaction
        ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage'],
        ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads'],
        ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment'],
        ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Publication_Day'],
        ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Publication_Time'],
        ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Genre'],
        ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads'],
        ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
        ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Publication_Day'],
        ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Publication_Time'],
        ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads', 'Episode_Sentiment'],
        ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads', 'Publication_Day'],
        ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads', 'Publication_Time'],
        ['Episode_Length_minutes', 'Episode_Num', 'Publication_Day', 'Publication_Time'],
        ['Episode_Length_minutes', 'Episode_Num', 'Publication_Day', 'Genre'],    
        ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads'],
        ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
        ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Day'],
        ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Time'],
        ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
        ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
        ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time'],
        ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day', 'Genre'],
        ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
        ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
        ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Publication_Time'],
        ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Genre'],
        ['Episode_Length_minutes', 'Episode_Num', 'Publication_Time', 'Podcast_Name'],
        
        ['Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Time'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Publication_Day'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Publication_Time'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Genre'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Time', 'Genre'],
        ['Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
        ['Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Genre'],
        ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Podcast_Name'],
        ['Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Podcast_Name'],
        ['Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day', 'Podcast_Name'],
        ['Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Time', 'Podcast_Name'],
        
    ]
    
    for comb in selected_comb:
        name = '_'.join(comb)
            
        if len(comb) == 2:
            df[name] = df[comb[0]].astype(str) + '_' + df[comb[1]].astype(str)
            
            
        elif len(comb) == 3:
            df[name] = (df[comb[0]].astype(str) + '_' +
                           df[comb[1]].astype(str) + '_' +
                           df[comb[2]].astype(str))
            
        elif len(comb) == 4:
            df[name] = (df[comb[0]].astype(str) + '_' +
                           df[comb[1]].astype(str) + '_' +
                           df[comb[2]].astype(str) + '_' +
                           df[comb[3]].astype(str))
            
        
        encoded_columns.append(name)
    
    df[encoded_columns] = df[encoded_columns].astype('category')

    ohe = OneHotEncoder(use_cat_names=True)
    ohe_df = ohe.fit_transform(df[["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]])

    df = pd.concat([df, ohe_df], axis=1)


    df.drop(columns=["Episode_Length_minutes", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"], inplace=True)
    
    if train:
        df
    else:
        df = df[['Podcast_Name',
         'Episode_Title',
         'Host_Popularity_percentage',
         'Guest_Popularity_percentage',
         'Number_of_Ads',
         'pod_guest_host',
         'new2',
         'Episode_Num',
         'Episode_Length_minutes_Host_Popularity_percentage',
         'Episode_Length_minutes_Guest_Popularity_percentage',
         'Episode_Length_minutes_Number_of_Ads',
         'Episode_Num_Host_Popularity_percentage',
         'Episode_Num_Guest_Popularity_percentage',
         'Episode_Num_Number_of_Ads',
         'Host_Popularity_percentage_Guest_Popularity_percentage',
         'Host_Popularity_percentage_Number_of_Ads',
         'Host_Popularity_percentage_Episode_Sentiment',
         'Episode_Length_minutes_Podcast_Name',
         'Episode_Num_Podcast_Name',
         'Guest_Popularity_percentage_Podcast_Name',
         'Episode_Length_minutes_Episode_Num_Host_Popularity_percentage',
         'Episode_Length_minutes_Episode_Num_Guest_Popularity_percentage',
         'Episode_Length_minutes_Episode_Num_Number_of_Ads',
         'Episode_Length_minutes_Episode_Num_Episode_Sentiment',
         'Episode_Length_minutes_Episode_Num_Publication_Day',
         'Episode_Length_minutes_Host_Popularity_percentage_Guest_Popularity_percentage',
         'Episode_Length_minutes_Host_Popularity_percentage_Number_of_Ads',
         'Episode_Length_minutes_Host_Popularity_percentage_Episode_Sentiment',
         'Episode_Length_minutes_Host_Popularity_percentage_Publication_Day',
         'Episode_Length_minutes_Host_Popularity_percentage_Publication_Time',
         'Episode_Length_minutes_Guest_Popularity_percentage_Number_of_Ads',
         'Episode_Length_minutes_Guest_Popularity_percentage_Publication_Day',
         'Episode_Length_minutes_Guest_Popularity_percentage_Publication_Time',
         'Episode_Length_minutes_Number_of_Ads_Episode_Sentiment',
         'Episode_Length_minutes_Number_of_Ads_Publication_Day',
         'Episode_Length_minutes_Episode_Sentiment_Publication_Time',
         'Episode_Num_Host_Popularity_percentage_Guest_Popularity_percentage',
         'Episode_Num_Host_Popularity_percentage_Number_of_Ads',
         'Episode_Num_Host_Popularity_percentage_Episode_Sentiment',
         'Episode_Num_Host_Popularity_percentage_Publication_Day',
         'Episode_Num_Host_Popularity_percentage_Publication_Time',
         'Episode_Num_Host_Popularity_percentage_Genre',
         'Episode_Num_Guest_Popularity_percentage_Number_of_Ads',
         'Episode_Num_Guest_Popularity_percentage_Episode_Sentiment',
         'Episode_Num_Guest_Popularity_percentage_Publication_Day',
         'Episode_Num_Guest_Popularity_percentage_Publication_Time',
         'Episode_Num_Guest_Popularity_percentage_Genre',
         'Episode_Num_Number_of_Ads_Episode_Sentiment',
         'Host_Popularity_percentage_Guest_Popularity_percentage_Number_of_Ads',
         'Host_Popularity_percentage_Guest_Popularity_percentage_Episode_Sentiment',
         'Host_Popularity_percentage_Guest_Popularity_percentage_Publication_Day',
         'Host_Popularity_percentage_Guest_Popularity_percentage_Publication_Time',
         'Host_Popularity_percentage_Number_of_Ads_Publication_Day',
         'Guest_Popularity_percentage_Number_of_Ads_Episode_Sentiment',
         'Guest_Popularity_percentage_Number_of_Ads_Genre',
         'Episode_Length_minutes_Episode_Num_Host_Popularity_percentage_Guest_Popularity_percentage',
         'Episode_Length_minutes_Episode_Num_Host_Popularity_percentage_Number_of_Ads',
         'Episode_Length_minutes_Episode_Num_Host_Popularity_percentage_Episode_Sentiment',
         'Episode_Length_minutes_Episode_Num_Host_Popularity_percentage_Publication_Day',
         'Episode_Length_minutes_Episode_Num_Host_Popularity_percentage_Publication_Time',
         'Episode_Length_minutes_Episode_Num_Host_Popularity_percentage_Genre',
         'Episode_Length_minutes_Episode_Num_Guest_Popularity_percentage_Number_of_Ads',
         'Episode_Length_minutes_Episode_Num_Guest_Popularity_percentage_Episode_Sentiment',
         'Episode_Length_minutes_Episode_Num_Guest_Popularity_percentage_Publication_Day',
         'Episode_Length_minutes_Episode_Num_Guest_Popularity_percentage_Publication_Time',
         'Episode_Length_minutes_Episode_Num_Number_of_Ads_Episode_Sentiment',
         'Episode_Length_minutes_Episode_Num_Number_of_Ads_Publication_Day',
         'Episode_Length_minutes_Episode_Num_Number_of_Ads_Publication_Time',
         'Episode_Length_minutes_Episode_Num_Publication_Day_Publication_Time',
         'Episode_Length_minutes_Episode_Num_Publication_Day_Genre',
         'Episode_Length_minutes_Host_Popularity_percentage_Guest_Popularity_percentage_Number_of_Ads',
         'Episode_Length_minutes_Host_Popularity_percentage_Guest_Popularity_percentage_Episode_Sentiment',
         'Episode_Length_minutes_Host_Popularity_percentage_Guest_Popularity_percentage_Publication_Day',
         'Episode_Length_minutes_Host_Popularity_percentage_Guest_Popularity_percentage_Publication_Time',
         'Episode_Length_minutes_Host_Popularity_percentage_Number_of_Ads_Episode_Sentiment',
         'Episode_Length_minutes_Host_Popularity_percentage_Number_of_Ads_Publication_Day',
         'Episode_Length_minutes_Host_Popularity_percentage_Publication_Day_Publication_Time',
         'Episode_Length_minutes_Host_Popularity_percentage_Publication_Day_Genre',
         'Episode_Length_minutes_Guest_Popularity_percentage_Number_of_Ads_Episode_Sentiment',
         'Episode_Length_minutes_Guest_Popularity_percentage_Number_of_Ads_Publication_Day',
         'Episode_Length_minutes_Guest_Popularity_percentage_Number_of_Ads_Publication_Time',
         'Episode_Length_minutes_Guest_Popularity_percentage_Number_of_Ads_Genre',
         'Episode_Length_minutes_Episode_Num_Publication_Time_Podcast_Name',
         'Episode_Num_Host_Popularity_percentage_Guest_Popularity_percentage_Number_of_Ads',
         'Episode_Num_Host_Popularity_percentage_Guest_Popularity_percentage_Episode_Sentiment',
         'Episode_Num_Host_Popularity_percentage_Number_of_Ads_Publication_Day',
         'Episode_Num_Host_Popularity_percentage_Number_of_Ads_Publication_Time',
         'Episode_Num_Host_Popularity_percentage_Episode_Sentiment_Publication_Day',
         'Episode_Num_Host_Popularity_percentage_Episode_Sentiment_Publication_Time',
         'Episode_Num_Host_Popularity_percentage_Episode_Sentiment_Genre',
         'Episode_Num_Host_Popularity_percentage_Publication_Day_Publication_Time',
         'Episode_Num_Host_Popularity_percentage_Publication_Time_Genre',
         'Episode_Num_Guest_Popularity_percentage_Number_of_Ads_Episode_Sentiment',
         'Episode_Num_Guest_Popularity_percentage_Number_of_Ads_Genre',
         'Episode_Num_Host_Popularity_percentage_Episode_Sentiment_Podcast_Name',
         'Host_Popularity_percentage_Number_of_Ads_Episode_Sentiment_Podcast_Name',
         'Host_Popularity_percentage_Number_of_Ads_Publication_Day_Podcast_Name',
         'Host_Popularity_percentage_Number_of_Ads_Publication_Time_Podcast_Name',
         'Genre_True Crime',
         'Genre_Comedy',
         'Genre_Education',
         'Genre_Technology',
         'Genre_Health',
         'Genre_News',
         'Genre_Music',
         'Genre_Sports',
         'Genre_Business',
         'Genre_Lifestyle',
         'Publication_Day_Thursday',
         'Publication_Day_Saturday',
         'Publication_Day_Tuesday',
         'Publication_Day_Monday',
         'Publication_Day_Sunday',
         'Publication_Day_Wednesday',
         'Publication_Day_Friday',
         'Publication_Time_Night',
         'Publication_Time_Afternoon',
         'Publication_Time_Evening',
         'Publication_Time_Morning',
         'Episode_Sentiment_Positive',
         'Episode_Sentiment_Negative',
         'Episode_Sentiment_Neutral']]

    
    return df


def score_predict(algorithms, X, y, test_df=None, save=True): 

    test_data_dict = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
        
        kfold= KFold(n_splits=5)
        score = cross_val_score(model, X, y, cv=kfold, scoring="neg_root_mean_squared_error").mean()
        # Get the score.
        print("Making Prediction on the test dataset")
        y_test_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_data_dict[alg_name] = [test_rmse, score]
        
        if save == True:
            y_pred = model.predict(test_df)
            pd.DataFrame({"Listening_Time_minutes": y_pred}, index=test_df.index).to_csv(f"{alg_name}.csv")

    test_score_df = pd.DataFrame(test_data_dict, index=["Test_rmse", "CV_Score"]).T.sort_values(by="Test_rmse", ascending=False)
    
    return test_score_df


def info(data, filepath=True):
    """
    Get Insight about the dataset.

    --------
    Parameter: 
        filepath: boolean
        data: can be dataframe or filepath.
    """
    if filepath:
        df = wrangle(filpath)
    else:
        df = data
    
    shape = df.shape

    print("=" * 50)
    print("Shape of the Data\n")
    print(f"The dataset contains {shape[0]} rows and {shape[1]} columns")
    print("\n")
    print("=" * 50)
    print("Checking For Percentage of Null Value\n")
    null = df.isnull().sum()
    null_series = null[null > 0].sort_values() / shape[0]
    if len(list(null_series.values)) == 0:
        print("No Null Values in the dataset.")
    else:
        print(null_series)
    print("\n")
    print("=" * 50)
    print("Statistical Distribution of the Dataset\n")
    return df.describe()


df = wrangle(train_filepath)
df.head()


test_df = wrangle(test_filepath, train=False)


"Listening_Time_minutes" in df.columns


info(df, filepath=False)


sns.heatmap(df.select_dtypes("number").corr());


abs(df.select_dtypes("number").corr()["Listening_Time_minutes"]).sort_values(ascending=False)


target = "Listening_Time_minutes"
X = df.drop(columns=target)
y = df[target]


info(test_df, filepath=False)


xgb_params= {'learning_rate': 0.0059569306860287755,
 'max_depth': 16,
 'n_estimators': 2798,
 'colsample_bylevel': 0.9020921245601042,
 'subsample': 0.8232351140650805,
 'min_child_weight': 4}

lgb_params = {'learning_rate': 0.009784041553146454,
 'max_depth': 11,
 'n_estimators': 2999,
 'colsample_bylevel': 0.9059246397755069,
 'subsample': 0.8794471636263226,
 'min_child_weight': 16}
algorithms = [LGBMRegressor(verbose=-1, random_state=42),
             XGBRegressor(random_state=42),
             #CatBoostRegressor(verbose=0, random_state=42),
             ]
score = score_predict(algorithms, X, y, test_df)
score


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
estimators = [("lgb", algorithms[0]), ("xgb", algorithms[1])]
vote = make_pipeline(
    OrdinalEncoder(),
    StandardScaler(),
    VotingRegressor(estimators=estimators, weights=[1, 2])
)
vote.fit(X_train, y_train)
y_test_pred = vote.predict(X_test)
test_roc_auc_score = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(test_roc_auc_score)
y_pred = vote.predict(test_df)
pd.DataFrame({"Listening_Time_minutes": y_pred}, index=test_df.index).to_csv(f"submission.csv")

