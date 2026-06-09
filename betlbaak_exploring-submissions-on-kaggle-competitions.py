# Manipulation
import numpy as np
import pandas as pd
import kagglehub

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Engineering
import string
from tqdm import tqdm


#nltk.download("stopwords")
#nltk.download("punkt")
#nltk.download("wordnet")
#nltk.download("vader_lexicon")
#nltk.download('punkt_tab')


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.width", 500)
pd.set_option('display.max_colwidth', None)


def reduce_mem_usage(data, verbose=True):
    """
    Optimizes numeric columns to reduce memory usage.
    
    Parameters:
    ----------------
        data: dataframe
                dataframe that wants to apply  
                
    Returns:
    ---------------
        data: dataframe
    """
    start_mem = data.memory_usage().sum() / 1024**2
    
    for col in data.columns:
        col_type = data[col].dtype
        
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = data[col].min()
            c_max = data[col].max()

            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                else:
                    data[col] = data[col].astype(np.int64)
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)
                    

    end_mem = data.memory_usage().sum() / 1024**2

    if verbose:
        print(f'🔍 Memory usage before: {start_mem:.2f} MB')
        print(f'✅ Memory usage after:  {end_mem:.2f} MB')
        print(f'💡 Decreased by:       {(100 * (start_mem - end_mem) / start_mem):.1f}%')

    return data


def data_summary(dataframe):
    """
    It returns the summary of data

    Parameters:
    ----------------
        dataframe: dataframe
                dataframe that wants to apply  
                
    Returns:
    ---------------
        shape:  number of observations and columns
        dtypes: types of the columns
        isnull: number of the null values of the each columns
        describe: descriptive statistical analysis of data
    """
    print("############## SHAPE ##############")
    print(dataframe.shape)
    print("############## TYPES ##############")
    print(dataframe.dtypes)
    print("############## NULL ##############")
    print(dataframe.isnull().sum())
    print("############ DESCRIBE ############")
    print(dataframe.describe([0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T)


def Remove_punctuation(text): 
    """
    Removes punctuation from text data

    Parameters:
    ----------------
        text: text
                text data that wants to apply  
                
    Returns:
    ---------------
        text: text 
                text data with punctuation removed
    """
    return text.translate(str.maketrans("", 
                                        "", 
                                        string.punctuation))


def model_clustering(text):
    """
    to determine which model type the competition is

     Parameters:
    ----------------
        text: text data that wants to apply 
                   
    Returns:
    ---------------
        value: clustering label
                
    """
    text = str(text).lower()
    
    if any(word in text for word in ["regression", "binary regression", "ridge","lasso","svr","xgboost","adaboost","lightgbm","prediction","predict","forecasting","forecast","estimate"]):
        return "regression"
    elif any(word in text for word in ["classification","categorize","classify","classifier","binary classification","home credit","credit risk","churn","decision tree","random forest","svm","knn"]):
        return "classification"
    elif any(word in text for word in ["clustering","cluster","kmeans","k-means","dbscan","anomaly","outlier","fraud","detect","detection","pca","noise"]):
        return "clustering"
    elif any(word in text for word in ["recommendation","recommender","recomendation","association rule learning","association","apriori"]):
        return "recommendation"
    elif any(word in text for word in ["nlp","tweet", "natural language" ,"language model","bert","text","sentiment","comment","sentiment analysis","llm","image","video", "vision","segmentation","mnist","3D","recognition","identify","ai","ann","neural networks","deep learning"]):
        return "deep_learning"
    else:
        return "other"


def categoric_data(dataframe, cat_th = 10, car_th=20):
    """
    It serves to determine whether the variables in the dataset are categorical, numerical or cardinal variables.
    
    Parameters:
    ----------------
        dataframe: dataframe
                dataframe that wants to apply
        cat_th: int, optional
                Class threshold for numeric but categorical variables
        car_th: int, optional
                Class threshold for categorical but cardinal variables
                
    Returns:
    ---------------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numerical variable list
        cat_but_car: list
                Categorical but cardinal variable list
        num_but_cat: list
                Numerical but categorical variable list
            
    Notes:
    ---------------
    cat_cols + num_cols + cat_but_car = total variables
    num_but_cat variables are in cat_cols.
    """
    
    cat_cols = [col for col in dataframe.columns if str(dataframe.dtypes[col]) in ["category", "bool", "object"]]
    num_but_cat = [col for col in dataframe.columns if str(dataframe.dtypes[col]) in ["int64", "float64", "int32", "float32", "int16", "float16", "int8", "float8"] and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if str(dataframe.dtypes[col]) in ["category", "object"] and dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["int64", "float64", "int32", "float32", "int16", "float16", "int8", "float8"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variebles: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    
    return cat_cols, num_cols, cat_but_car, num_but_cat


path = kagglehub.dataset_download("kaggle/meta-kaggle")

print("Path to dataset files:", path)


comp_cols = ["Id", # int, contains the unique ID number for each competition
            "Title", # object, contains the title for each competition
            "Subtitle", # object, contains the subtitle for each competition
            "HostSegmentTitle", # object, provides information on which segment the competition is in
            "EnabledDate", # datetime, competition start date
            "DeadlineDate", # datetime, competition end date 
            "EvaluationAlgorithmAbbreviation", # object, contains the evaluation algorithm abbreviation
            "EvaluationAlgorithmName", # object, contains the evaluation algorithm full name
            "EvaluationAlgorithmIsMax", # bool, contains whether a contest's evaluation metric is considered better with a "maximum" value or a "minimum" value.
            "MaxDailySubmissions", # int, the maximum daily submission amount allowed by the competition
            "NumScoredSubmissions", # int, contains the number of scored submissions
            "MaxTeamSize", # int, contains the maximum number of users that the teams participating in the competition should consist of
            "BanTeamMergers", # bool, contains whether a competition allows team merges
            "EnableTeamModels", #bool, specifies whether teams can submit models together during the competition
            "RewardType", # object, contains what type of reward the competition offers
            "RewardQuantity", # float, contains the reward quantitiy
            "NumPrizes", # int, gives the number of prizes to be awarded
            "UserRankMultiplier", # float, contains the coefficient that affects the scoring system for users in competitions
            "CanQualifyTiers", # bool, contains whether the contest has an impact on the user's tier level (Contributor, Expert, Master, Grandmaster)
            "TotalTeams", # int, gives the total number of teams participating in each competition
            "TotalCompetitors", # int, gives the total number of users in each competition
            "TotalSubmissions", # int, gives the total number of submissions in each competition 
            ]

competitions = pd.read_csv("/kaggle/input/meta-kaggle/Competitions.csv", usecols=comp_cols)


competitions = reduce_mem_usage(competitions)


sub_cols = ["Id", # int, contains the unique ID number for each submission
            "TeamId", # int, contains the ID number of the team making the submission
           "SubmissionDate", # datetime, the date of submission
           "IsAfterDeadline", # bool, gives whether the submission is after deadline or not
           "IsSelected", # object, indicates whether the submission is the winner of the competition
           "PublicScoreLeaderboardDisplay", # float, gives the publicly visible score shown on the leaderboard during the competition
           "PublicScoreFullPrecision", # float, gives the full precision version of the public score
           "PrivateScoreLeaderboardDisplay", #float, gives the score computed on the hidden portion of the test set
           "PrivateScoreFullPrecision"] # float, gives the exact, full-precision version of the private score

cutoff_date = pd.Timestamp("2024-12-31") 
start_date = pd.Timestamp("2022-01-01")

chunks = pd.read_csv("/kaggle/input/meta-kaggle/Submissions.csv",
                     parse_dates=["SubmissionDate"],
                     chunksize=1000000,
                     low_memory=False,
                    usecols = sub_cols)

filtered_chunks = []

for chunk in chunks:
    chunk_filtered = chunk[
        (chunk["SubmissionDate"] >= start_date) &
        (chunk["SubmissionDate"] <= cutoff_date)
    ]
    filtered_chunks.append(chunk_filtered)

submissions = pd.concat(filtered_chunks, ignore_index=True)


submissions.shape


submissions = reduce_mem_usage(submissions)


teams_cols = ["Id", # int, contains the unique ID number for each team
              "CompetitionId", # int, contains the unique ID number for each competition
              "TeamLeaderId" # int, contains the unique ID number for each team leader
             ]
teams = pd.read_csv("/kaggle/input/meta-kaggle/Teams.csv", usecols = teams_cols)


teams = reduce_mem_usage(teams)


submissions.rename(columns = {"Id": "SubmissionId"}, 
                    inplace = True)

submissions.head()


data_summary(submissions)


submissions[submissions["PublicScoreFullPrecision"]>1].sort_values(by = "PublicScoreFullPrecision", ascending = False).head()


index_list = submissions[
    (submissions["PublicScoreLeaderboardDisplay"] > 1) |
    (submissions["PublicScoreLeaderboardDisplay"] < -1) |
    (submissions["PrivateScoreLeaderboardDisplay"] > 1) |
    (submissions["PrivateScoreLeaderboardDisplay"] < -1) |
    (submissions["PublicScoreFullPrecision"] > 1) |
    (submissions["PublicScoreFullPrecision"] < -1) |
    (submissions["PrivateScoreFullPrecision"] > 1) |
    (submissions["PrivateScoreFullPrecision"] < -1)
].index.tolist()

len(index_list)


submissions = submissions.drop(index_list, axis=0).reset_index(drop=True)
submissions.shape


teams.rename(columns = {"Id": "TeamId"}, 
             inplace = True)

teams.head()


data_summary(teams)


competitions.rename(columns = {"Id": "CompetitionId"}, 
                    inplace = True)

competitions.head()


data_summary(competitions)


data = pd.merge(submissions,
               teams,
               on = "TeamId",
               how = "left")

data.shape


data = pd.merge(data,
               competitions,
               on = "CompetitionId",
               how = "left")
data.shape


data.head()


data_summary(data)


null_index = data[data["PublicScoreLeaderboardDisplay"].isnull()].index.tolist() # these rows have no score values for submissions

data = data.drop(null_index,
                 axis = 0)

data.shape


data[data["HostSegmentTitle"] == "Community"].head()


data.fillna({"RewardType" : "NA",
             "RewardQuantity" : 0,
             "Subtitle" : "no title",
             "TeamLeaderId" : "NA",
               "IsSelected": "NA"},
            inplace = True)


data.isnull().sum()


cat_cols, num_cols, cat_but_car, num_but_cat = categoric_data(data)


for col in cat_cols:
    print(data[col].value_counts())
    print("####################################")


data[data["HostSegmentTitle"] == "Analytics"]


num_cols.remove('SubmissionId')
num_cols.remove('TeamId')
num_cols.remove('CompetitionId')


num_cols


for col in num_cols:
    print(data[col].name)
    print("#####################################################################################################################")
    print(data[col].describe())
    print("#####################################################################################################################")


for col in num_cols:
    sns.boxplot(x = data[col])
    plt.show()


data[data["MaxDailySubmissions"]>800]["Title"].value_counts() 
# As can be seen in the boxplot chart, these 39 values appear to be very outliers compared to the others.


data.drop(data[data["MaxDailySubmissions"]>800].index.tolist(), 
          axis=0, 
          inplace = True)


data[data["NumScoredSubmissions"]>8000]["Title"].value_counts() 
# As seen in the boxplot, it appears to be a significant outlier compared to the others. 


data.drop(data[data["NumScoredSubmissions"]>8000].index.tolist(),
          axis = 0,
          inplace = True)


data[data["MaxTeamSize"]>17]["Title"].nunique() 
# There are too many values here. This means that maxteamsize may have a non-normal distribution, and that doesn't necessarily mean it's an outlier.
# This may vary depending on the rules set by the competition host.


data[data["RewardQuantity"]>0.5]["Title"].nunique() 
# This and the NumPrize are entirely dependent on the host and the size of the competition, so it's normal to see fluctuations in the prize amount.


data[data["TotalSubmissions"]>100000]["Title"].value_counts() 
# This, along with variables like TotalCompetitor and TotalTeam, are also values that can change depending on the popularity and returns of the competition.
# It would be wrong to say that these are definitely outlier data.


data["Title"] = data["Title"] + " " + data["Subtitle"]


data.drop("Subtitle",
         axis = 1,
         inplace = True)


data["Title"] = data["Title"].str.lower()


data["Title"] = data["Title"].apply(Remove_punctuation)


data["Title"].head()


# In order to easily process big datasets, dividing the data into batches allows us to get better and faster results.

batch_size = 100000
model_types = []

for start in tqdm(range(0, len(data), batch_size)):
    end = min(start + batch_size, len(data))
    batch_titles = data["Title"].iloc[start:end]

    batch_results = [model_clustering(title) for title in batch_titles]
    model_types.extend(batch_results)

data["ModelType"] = model_types


data["ModelType"].value_counts()


df = data[["Title", 
           "HostSegmentTitle", 
           "EvaluationAlgorithmName",
           "EvaluationAlgorithmIsMax", 
           "ModelType", 
           "RewardType", 
           "RewardQuantity",
          "TotalSubmissions"]]
df = df.dropna().drop_duplicates() 
df.head()


for label in df["ModelType"].unique():
    text = " ".join(df[df["ModelType"] == label]["Title"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Model Tipi: {label}")
    plt.show()


# Countplot showing how many competitions there are of each model type
plt.figure(figsize=(8, 5))
sns.countplot(x="ModelType", data=df, order=df["ModelType"].value_counts().index)
plt.title("How Many of Each Model Type Are There?")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Countplot showing how many competitions there are of each host segment
plt.figure(figsize=(8, 5))
sns.countplot(x="HostSegmentTitle", 
              data=df, 
              order=df["HostSegmentTitle"].value_counts().index)
plt.title("How Many of Each Host Are There?")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Filter out NA values
reward_type_df = df[df["RewardType"] != "NA"] 

# Number of competitions in the reward type breakdown for each host segment
plt.figure(figsize=(12, 6))
ax = sns.countplot(
    x="HostSegmentTitle",
    data=reward_type_df,
    order=reward_type_df["HostSegmentTitle"].value_counts().index,
    hue="RewardType",
    palette="Set2"
)
plt.title("Distribution of Host Types by Reward Type", fontsize=14)
plt.xlabel("Host Segment Title")
plt.ylabel("Number of Competitions")
plt.xticks(rotation=45)
plt.legend(title="Reward Type")
plt.tight_layout()

for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=9)
    
plt.show()


# Let's group the total number of submissions according to Reward Type.
reward_submission = df.groupby("RewardType")["TotalSubmissions"].mean().sort_values(ascending=False)

# Bar plot showing submission average by reward type
plt.figure(figsize=(10, 6))
sns.barplot(x=reward_submission.index, y=reward_submission.values, palette="viridis")

plt.title("Total Submissions by Reward Type", fontsize=14)
plt.xlabel("Reward Type")
plt.ylabel("Total Submissions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


late_sub_df = data[data["IsAfterDeadline"]==1] # filter late submissions
late_sub_df.groupby(["Title", "HostSegmentTitle", "RewardType"]).agg({"CompetitionId" : "count"}).sort_values(by = "CompetitionId", ascending = False).head()


#Count chart showing which host segment has the most late submissions
plt.figure(figsize=(8, 5))
sns.countplot(x="HostSegmentTitle", 
              data=late_sub_df, 
              order=late_sub_df["HostSegmentTitle"].value_counts().index)
plt.title("The Host Segment Title That Received The Most Submissions After Deadline")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


most_common_algo_per_model = (
    df.groupby(["ModelType", "EvaluationAlgorithmName"])
      .size() #counting how many times each combination is repeated.
      .reset_index(name='count') #converts the result to DataFrame and sets the count column to "count".
      .sort_values(['ModelType', 'count'], ascending=[True, False]) #the highest count for each model type comes on top.
      .drop_duplicates(subset=['ModelType']) #only retrieves the first (most used) of each ModelType.
      .reset_index(drop=True)
)

most_common_algo_per_model


# Bar plot showing the most commonly used evaluation algorithm for each model type
plt.figure(figsize=(8, 6))
sns.barplot(
    data=most_common_algo_per_model,
    x="ModelType",
    y="count",
    hue="EvaluationAlgorithmName",
    dodge=False,
    palette="Set2"
)

plt.title("Most Used EvaluationAlgorithmName for Each ModelType")
plt.ylabel("Count")
plt.xlabel("Model Type")
plt.xticks(rotation=45)
plt.legend(title="Evaluation Algorithm", loc='upper right')
plt.tight_layout()
plt.show()


top_10_algorithms = df['EvaluationAlgorithmName'].value_counts().nlargest(10).index
top_10_algorithms # top 10 most used algorithms


df_top_algorithms = data[
    data["EvaluationAlgorithmName"].isin(top_10_algorithms)
].dropna(subset=["PrivateScoreFullPrecision", "ModelType"]) 


# Average score values of the top 10 most used evaluation algorithms according to model types
pivot_top10 = df_top_algorithms.pivot_table(
    values="PrivateScoreFullPrecision",
    index="ModelType",
    columns="EvaluationAlgorithmName",
    aggfunc="mean"
)

pivot_top10 


df_top_algorithms.groupby(["EvaluationAlgorithmIsMax", "EvaluationAlgorithmName"])["PrivateScoreFullPrecision"].mean()


#Heat map showing the average score values of the top 10 most used evaluation algorithms by model type
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_top10, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5, linecolor='gray')
plt.title("Average Score by Model Type for Top 10 Evaluation Algorithms", fontsize=14)
plt.xlabel("Evaluation Algorithm")
plt.ylabel("Model Type")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


num_cols = num_cols + num_but_cat
num_cols


# Correlation matrix showing the relation between all numerical values
correlation_matrix = data[num_cols].corr()
sns.clustermap(correlation_matrix, annot = True, fmt = ".2f")
plt.title("Corralation Between Numeric Features")
plt.show()

