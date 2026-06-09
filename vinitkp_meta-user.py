import numpy as np 
import pandas as pd 
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))




import kagglehub

# Download latest version
path = kagglehub.dataset_download("kaggle/meta-kaggle")

print("Path to dataset files:", path)



import kagglehub

# Download latest version
path = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print("Path to dataset files:", path)



# !pip install sweetviz autoviz dtale pycaret lux-api


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# class AutoDataAnalyzer:
#     def __init__(self, csv_path):
#         self.data = pd.read_csv(csv_path)

#     def describe_data(self):
#         print("\n===== BASIC INFO =====")
#         print(self.data.info())
#         print("\n===== DESCRIPTIVE STATS =====")
#         print(self.data.describe(include='all'))

#     def check_missing(self):
#         print("\n===== MISSING VALUES =====")
#         print(self.data.isnull().sum())

#     def correlation_matrix(self):
#         print("\n===== CORRELATION MATRIX =====")
#         corr = self.data.corr(numeric_only=True)
#         print(corr)
#         sns.heatmap(corr, annot=True, cmap='coolwarm')
#         plt.title("Correlation Heatmap")
#         plt.show()

#     def value_counts(self):
#         print("\n===== VALUE COUNTS FOR CATEGORICAL COLUMNS =====")
#         cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
#         for col in cat_cols:
#             print(f"\nColumn: {col}")
#             print(self.data[col].value_counts())

#     def distribution_plots(self):
#         print("\n===== DISTRIBUTION PLOTS =====")
#         num_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
#         self.data[num_cols].hist(figsize=(15, 10), bins=30)
#         plt.tight_layout()
#         plt.show()

#     def pairplot(self):
#         print("\n===== PAIR PLOT FOR NUMERICAL COLUMNS =====")
#         num_cols = self.data.select_dtypes(include=['float64', 'int64']).columns[:5]  
#         sns.pairplot(self.data[num_cols])
#         plt.show()



# analyzer = AutoDataAnalyzer("/kaggle/input/meta-kaggle/Users.csv")




#analyzer.describe_data()
# analyzer.check_missing()
# analyzer.correlation_matrix()
# analyzer.value_counts()
# analyzer.distribution_plots()
# analyzer.pairplot()


# from pandas_profiling import ProfileReport
# users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv')
# profile = ProfileReport(users)
# profile


import pandas as pd

competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv', usecols=['Id', 'Title', 'Id'])
users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv', usecols=['Id', 'UserName', 'PerformanceTier'])

# Shape & memory usage BEFORE merging
print("Competitions shape:", competitions.shape)
print("Users shape:", users.shape)
print("Competitions memory:")
print(competitions.memory_usage(deep=True).sum() / (1024**2), "MB")
print("Users memory:")
print(users.memory_usage(deep=True).sum() / (1024**2), "MB")

# Perform inner join using HostUserId and Users.Id
uc_data = pd.merge(competitions, users, left_on='Id', right_on='Id', how='inner')
print("Merged data shape:", uc_data.shape)
print(uc_data.head())



import pandas as pd

users_path = "/kaggle/input/meta-kaggle/Users.csv"
achievements_path = "/kaggle/input/meta-kaggle/UserAchievements.csv"

# Foptimize memory usage
def optimize_dataframe(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    return df


chunksize = 10000

# Users.csv in chunks
users_chunks = []
for chunk in pd.read_csv(users_path, usecols=["Id", "UserName", "DisplayName", "RegisterDate", "PerformanceTier", "Country", "LocationSharingOptOut"], chunksize=chunksize):
    users_chunks.append(optimize_dataframe(chunk))
users_df = pd.concat(users_chunks, ignore_index=True)

# UserAchievements.csv in chunks
achievements_chunks = []
for chunk in pd.read_csv(achievements_path, usecols=["Id", "UserId", "AchievementType", "Tier", "TierAchievementDate", "Points", "CurrentRanking", "HighestRanking", "TotalGold", "TotalSilver"], chunksize=chunksize):
    achievements_chunks.append(optimize_dataframe(chunk))
achievements_df = pd.concat(achievements_chunks, ignore_index=True)

# Merge on UserId -> Id
merged_df = pd.merge(achievements_df, users_df, left_on="UserId", right_on="Id", how="inner")

# Drop duplicate Id column
merged_df.drop(columns=["Id_y"], inplace=True)
merged_df.rename(columns={"Id_x": "AchievementId", "Id": "UserId"}, inplace=True)

print(merged_df.info())
print(merged_df.head())



import pandas as pd

users_path = "/kaggle/input/meta-kaggle/Users.csv"
achievements_path = "/kaggle/input/meta-kaggle/UserAchievements.csv"

# Function to optimize memory usage
def optimize_dataframe(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    return df

users_df = pd.read_csv(users_path, usecols=["Id", "UserName", "DisplayName", "RegisterDate", "PerformanceTier", "Country", "LocationSharingOptOut"], nrows=30000)
achievements_df = pd.read_csv(achievements_path, usecols=["Id", "UserId", "AchievementType", "Tier", "TierAchievementDate", "Points", "CurrentRanking", "HighestRanking", "TotalGold", "TotalSilver"], nrows=30000)

# Optimize memory
users_df = optimize_dataframe(users_df)
achievements_df = optimize_dataframe(achievements_df)

# Merge on UserId -> Id
merged_df = pd.merge(achievements_df, users_df, left_on="UserId", right_on="Id", how="inner")

# duplicate Id column from users_df if needed
merged_df.drop(columns=["Id_y"], inplace=True)
merged_df.rename(columns={"Id_x": "AchievementId", "Id": "UserId"}, inplace=True)

print(merged_df.info())
print(merged_df.head())



import pandas as pd

users_path = "/kaggle/input/meta-kaggle/Users.csv"
achievements_path = "/kaggle/input/meta-kaggle/UserAchievements.csv"

def optimize_dataframe(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    return df

users_df = pd.read_csv(users_path, usecols=["Id", "UserName", "DisplayName", "RegisterDate", "PerformanceTier", "Country", "LocationSharingOptOut"], nrows=30000)
achievements_df = pd.read_csv(achievements_path, usecols=["Id", "UserId", "AchievementType", "Tier", "TierAchievementDate", "Points", "CurrentRanking", "HighestRanking", "TotalGold", "TotalSilver"], nrows=30000)

# Optimize memory
users_df = optimize_dataframe(users_df)
achievements_df = optimize_dataframe(achievements_df)

# Merge on UserId -> Id
merged_df = pd.merge(achievements_df, users_df, left_on="UserId", right_on="Id", how="inner")

# Drop duplicate Id column from users_df if needed
merged_df.drop(columns=["Id_y"], inplace=True)
merged_df.rename(columns={"Id_x": "AchievementId", "Id": "UserId"}, inplace=True)

print(merged_df.info())
print(merged_df.head())



# Compare CurrentRanking and HighestRanking i

# Ensure numeric types
merged_df["CurrentRanking"] = pd.to_numeric(merged_df["CurrentRanking"], errors='coerce')
merged_df["HighestRanking"] = pd.to_numeric(merged_df["HighestRanking"], errors='coerce')

# Drop rows with missing ranking values
ranking_df = merged_df.dropna(subset=["CurrentRanking", "HighestRanking"]).copy()

# Compare rankings safely
ranking_df.loc[:, "Improved"] = ranking_df["CurrentRanking"] < ranking_df["HighestRanking"]
ranking_df.loc[:, "Same"] = ranking_df["CurrentRanking"] == ranking_df["HighestRanking"]
ranking_df.loc[:, "Worsened"] = ranking_df["CurrentRanking"] > ranking_df["HighestRanking"]

# Count summary
summary = {
    "Improved": ranking_df["Improved"].sum(),
    "Same": ranking_df["Same"].sum(),
    "Worsened": ranking_df["Worsened"].sum()
}

# Show summary
print("Ranking comparison summary:")
print(summary)

# Optional: show few improved users
print("\nSample of improved users:")
print(ranking_df[ranking_df["Improved"]][["UserName", "CurrentRanking", "HighestRanking"]].head())



# Investigation why all users show 'Worsened' rankings only

# Display some of the values for inspection
print("\nSample of CurrentRanking and HighestRanking values:")
print(ranking_df[["UserName", "CurrentRanking", "HighestRanking"]].head(20))

# Check if CurrentRanking is consistently higher than HighestRanking
ranking_diff = ranking_df["CurrentRanking"] - ranking_df["HighestRanking"]
print("\nRanking difference stats:")
print(ranking_diff.describe())

# Check if HighestRanking is always 1 or a small value
print("\nTop unique HighestRanking values:")
print(ranking_df["HighestRanking"].value_counts().head())

# Check min/max of both columns
print("\nCurrentRanking range:", ranking_df["CurrentRanking"].min(), "-", ranking_df["CurrentRanking"].max())
print("HighestRanking range:", ranking_df["HighestRanking"].min(), "-", ranking_df["HighestRanking"].max())



# Checking class distribution before model training

# Recreate target column
ranking_df["ImprovedBefore"] = ranking_df["CurrentRanking"] < ranking_df["HighestRanking"]

# Count value distribution
print("\nClass distribution in ImprovedBefore:")
print(ranking_df["ImprovedBefore"].value_counts())

# Show a few samples for verification
print("\nSample rows:")
print(ranking_df[["UserName", "CurrentRanking", "HighestRanking", "ImprovedBefore"]].head(10))




# Check number of users with improved rank
merged_df["CurrentRanking"] = pd.to_numeric(merged_df["CurrentRanking"], errors='coerce')
merged_df["HighestRanking"] = pd.to_numeric(merged_df["HighestRanking"], errors='coerce')

ranking_df = merged_df.dropna(subset=["CurrentRanking", "HighestRanking"]).copy()
ranking_df["ImprovedBefore"] = ranking_df["CurrentRanking"] < ranking_df["HighestRanking"]

print("\nNew class distribution:")
print(ranking_df["ImprovedBefore"].value_counts())



# Predict probability of achieving high rank in future

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# high rank as CurrentRanking <= 500
ranking_df["HighRank"] = ranking_df["CurrentRanking"] <= 500

# Features for prediction
features = ["Points", "TotalGold", "TotalSilver"]

# Drop NA values and filtering data
model_df = ranking_df.dropna(subset=features + ["HighRank"]).copy()
X = model_df[features]
y = model_df["HighRank"]

# Check if 2 classes
print("\nClass distribution for HighRank:")
print(y.value_counts())

# Proceed if both classes are present
if y.nunique() > 1:
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))

    # Predict probability of high rank
    model_df["FutureHighRankProb"] = clf.predict_proba(X_scaled)[:, 1]
    print("\nTop 10 users with highest probability of high rank:")
    print(model_df[["UserName", "CurrentRanking", "FutureHighRankProb"]].sort_values(by="FutureHighRankProb", ascending=False).head(10))
else:
    print("\nOnly one class present. Cannot train classifier.")



# Predict probability of a user having a low future rank

# low rank as CurrentRanking >= 10000
ranking_df["LowRank"] = ranking_df["CurrentRanking"] >= 10000

# Features to use
features = ["Points", "TotalGold", "TotalSilver"]

# Drop NA & filter usable rows
model_df = ranking_df.dropna(subset=features + ["LowRank"]).copy()
X = model_df[features]
y = model_df["LowRank"]

# Checking class balance
print("\nClass distribution for LowRank:")
print(y.value_counts())

# Train only if both classes are present
if y.nunique() > 1:
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))

    # Predict probabilities
    model_df["FutureLowRankProb"] = clf.predict_proba(X_scaled)[:, 1]
    print("\nTop 10 users most likely to have low rank:")
    print(model_df[["UserName", "CurrentRanking", "FutureLowRankProb"]].sort_values(by="FutureLowRankProb", ascending=False).head(10))
else:
    print("\nOnly one class present. Cannot train classifier.")





