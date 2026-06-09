import pandas as pd

PATH = "/kaggle/input/meta-kaggle"

users_df = pd.read_csv(f"{PATH}/Users.csv", usecols=["Id", "UserName"]).rename(columns={"Id": "UserId"})
teams_df = pd.read_csv(f"{PATH}/Teams.csv", usecols=["Id", "CompetitionId", "TeamName"]).rename(columns={"Id": "TeamId"})
team_members_df = pd.read_csv(f"{PATH}/TeamMemberships.csv", usecols=["UserId", "TeamId"])
subs_df = pd.read_csv(f"{PATH}/Submissions.csv", usecols=["Id", "TeamId", "SubmissionDate"]).rename(columns={"Id": "SubmissionId"})
comps_df = pd.read_csv(f"{PATH}/Competitions.csv", usecols=["Id", "Title"]).rename(columns={"Id": "CompetitionId"})
kernels_df = pd.read_csv(f"{PATH}/Kernels.csv", usecols=["AuthorUserId"])



subs_df["SubmissionDate"] = pd.to_datetime(subs_df["SubmissionDate"], errors="coerce")



# Merge team_members with users
merged = team_members_df.merge(users_df, on="UserId", how="left")

# Merge with teams
merged = merged.merge(teams_df, on="TeamId", how="left")

# Merge with submissions
merged = merged.merge(subs_df, on="TeamId", how="left")

# Merge with competitions
merged = merged.merge(comps_df, on="CompetitionId", how="left")



summary = (
    merged.groupby(["UserName", "Title", "TeamName"])
    .agg(
        NumSubmissions=("SubmissionId", "count"),
        FirstSubmission=("SubmissionDate", "min"),
        LastSubmission=("SubmissionDate", "max")
    )
    .reset_index()
)

summary.head(5)



active_users = (
    merged.groupby("UserName")["SubmissionId"]
    .count()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name="NumSubmissions")
)

active_users.plot(kind="barh", x="UserName", y="NumSubmissions", figsize=(10,6), title="Top 10 Most Active Users")



team_with_comp = team_members_df.merge(teams_df, on="TeamId", how="left")
user_comp_counts = team_with_comp.groupby("UserId")["CompetitionId"].nunique().reset_index(name="NumCompetitions")
team_with_sub = team_members_df.merge(subs_df, on="TeamId", how="inner")
user_sub_counts = team_with_sub.groupby("UserId")["TeamId"].count().reset_index(name="NumSubmissions")
user_kernel_counts = kernels_df.groupby("AuthorUserId").size().reset_index(name="NumKernels").rename(columns={"AuthorUserId": "UserId"})
user_team_counts = team_members_df.groupby("UserId")["TeamId"].nunique().reset_index(name="NumTeams")



# Start with users
user_summary = users_df.copy()

# Merge in all metrics
user_summary = user_summary.merge(user_comp_counts, on="UserId", how="left")
user_summary = user_summary.merge(user_kernel_counts, on="UserId", how="left")
user_summary = user_summary.merge(user_sub_counts, on="UserId", how="left")
user_summary = user_summary.merge(user_team_counts, on="UserId", how="left")

# Fill NaNs with 0
user_summary = user_summary.fillna(0)

# Convert counts to integer
cols = ["NumCompetitions", "NumKernels", "NumSubmissions", "NumTeams"]
user_summary[cols] = user_summary[cols].astype(int)

# Show top 10 users by competitions
user_summary.sort_values("NumCompetitions", ascending=False).head(10)



f=user_summary[user_summary["UserName"]=="youssefelzahar"]
f


from matplotlib import pyplot as plt
user_comp_count = merged.groupby("UserName")["Title"].nunique()

user_comp_count.hist(bins=30, figsize=(8,6))
plt.title("Distribution of Number of Competitions per User")
plt.xlabel("Number of Competitions")
plt.ylabel("Number of Users")



# Count how many members in each team
team_sizes = team_members_df.groupby("TeamId")["UserId"].count().reset_index(name="TeamSize")

# Merge back to users
merged_with_team_size = merged.merge(team_sizes, on="TeamId", how="left")

# Mark as solo or team
merged_with_team_size["IsSolo"] = merged_with_team_size["TeamSize"] == 1

solo_team_counts = merged_with_team_size.groupby("IsSolo")["UserName"].nunique()

solo_team_counts.plot(kind="bar", title="Solo vs Team Participants", figsize=(6,4))
plt.xticks(ticks=[0,1], labels=["Team", "Solo"], rotation=0)


