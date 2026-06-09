import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col=["id"]) # as I have id column, I will set it as index_col
df.head()


df.info() 


personality_counts = df["Personality"].value_counts()

plt.pie(personality_counts.values, 
        labels = personality_counts.index, 
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white'})


df.describe()


df.hist(figsize=(15, 10), bins=35)


fear_counts = df["Stage_fear"].value_counts()

plt.pie(fear_counts,
       labels=fear_counts.index,
       autopct="%1.1f%%",
       startangle=90,
       wedgeprops={'edgecolor': 'white'})

plt.title("Distribution of Stage_fear")


drained_counts = df["Drained_after_socializing"].value_counts()

plt.pie(drained_counts,
       labels=drained_counts.index,
       autopct="%1.1f%%",
       startangle=90,
       wedgeprops={"edgecolor": "white"})

plt.title("Distribution of drained_after_socializing")


fig = sns.boxplot(x="Personality", y = "Time_spent_Alone", data=df)
fig.set_title("Time_spent_Alone for each personality type")


fig = sns.boxplot(x="Personality", y="Social_event_attendance", data=df)
fig.set_title("Social_event attendance for each personality type")


fig = sns.boxplot(x="Personality", y="Post_frequency", data=df)
fig.set_title("Post_frequency for each personality type")


fig = sns.boxplot(x="Personality", y="Going_outside", data=df)
fig.set_title = "Going outside for each personality type"


fig = sns.boxplot(x="Personality", y="Friends_circle_size", data=df)
fig.set_title = "Friends circle size for each personality type"


fear_counts = df.groupby(["Personality", "Stage_fear"]).size() # aggregation function is the count of values
fear_counts = fear_counts.unstack(fill_value=0) # converts it to a table where rows are personality types, columns stage fear, and values are counts for each combination, if there is no count, then it will be 0

fear_counts.plot(kind="bar")


drained_counts = df.groupby(["Personality", "Drained_after_socializing"]).size()
drained_counts = drained_counts.unstack(fill_value=0)

drained_counts.plot(kind="bar")


corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Heatmap between numerical variables")
plt.show()


binary_map = {"No": 0, "Yes": 1}
df["Stage_fear"] = df["Stage_fear"].map(binary_map)
df["Drained_after_socializing"] = df["Drained_after_socializing"].map(binary_map)

target_map= {"Extrovert": 0, "Introvert": 1}
df["Personality"] = df["Personality"].map(target_map)


missing = df.isnull().sum() # number of missing values per column
total_rows = len(df) # total number of rows

percentage_missing = round(missing / total_rows * 100, 1)
percentage_missing.sort_values(ascending=True, inplace=True)

ax = percentage_missing.plot(kind="barh")

plt.xlabel ("Missing (%)")
plt.title("Percentage of Missing values by column")


df[df.isna().all(axis=1)] # There are no rows where all values are missing


df.fillna({"Post_frequency" : 0}, inplace = True)


df.dropna(axis=0, subset=["Drained_after_socializing"], inplace=True)


mode_circle = df["Friends_circle_size"].mode()
df["Friends_circle_size"] = df["Friends_circle_size"].fillna(value=mode_circle)


def random_sample_imputation(df):
    """Code source: https://www.datacamp.com/tutorial/techniques-to-handle-missing-data-values"""
    cols_with_missing_values = df.columns[df.isna().any()].tolist()
    
    for var in cols_with_missing_values:
    
        # extract a random sample
        random_sample_df = df[var].dropna().sample(df[var].isnull().sum(),
                                                      random_state=0)
        # re-index the randomly extracted sample
        random_sample_df.index = df[
                df[var].isnull()].index
    
        # replace the NA
        df.loc[df[var].isnull(), var] = random_sample_df
     
    return df


df = random_sample_imputation(df)


df.hist(figsize=(15, 10), bins=35)

