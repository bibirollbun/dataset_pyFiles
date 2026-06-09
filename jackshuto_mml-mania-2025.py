import os
import pandas as pd

# ãƒ‡ãƒ¼ã‚¿ãƒ•ã‚©ãƒ«ãƒ€ã�®ãƒ‘ã‚¹
data_path = "/kaggle/input/march-machine-learning-mania-2025"

# ãƒ•ã‚¡ã‚¤ãƒ«ä¸€è¦§ã‚’ç¢ºèª�
files = os.listdir(data_path)
print("Available files:", files)

# ä¸»è¦�ã�ªãƒ•ã‚¡ã‚¤ãƒ«ã‚’ãƒ­ãƒ¼ãƒ‰
m_teams = pd.read_csv(f"{data_path}/MTeams.csv")
w_teams = pd.read_csv(f"{data_path}/WTeams.csv")
m_results = pd.read_csv(f"{data_path}/MRegularSeasonCompactResults.csv")
w_results = pd.read_csv(f"{data_path}/WRegularSeasonCompactResults.csv")

# å�„ãƒ‡ãƒ¼ã‚¿ã�®åŸºæœ¬æƒ…å ±
print("\nMTeams.csv:")
print(m_teams.info())
print(m_teams.head())

print("\nWTeams.csv:")
print(w_teams.info())
print(w_teams.head())

print("\nMRegularSeasonCompactResults.csv:")
print(m_results.info())
print(m_results.head())

print("\nWRegularSeasonCompactResults.csv:")
print(w_results.info())
print(w_results.head())



import matplotlib.pyplot as plt

# ã‚·ãƒ¼ã‚ºãƒ³ã�”ã�¨ã�®è©¦å�ˆæ•°ã‚’ã‚«ã‚¦ãƒ³ãƒˆ
m_season_games = m_results.groupby("Season").size()
w_season_games = w_results.groupby("Season").size()

# å�¯è¦–åŒ–
plt.figure(figsize=(12, 5))
plt.plot(m_season_games.index, m_season_games.values, label="Men's Games")
plt.plot(w_season_games.index, w_season_games.values, label="Women's Games", linestyle="dashed")
plt.xlabel("Season")
plt.ylabel("Number of Games")
plt.title("Number of Games per Season")
plt.legend()
plt.show()



import matplotlib.pyplot as plt

# å‹�ç�‡ã�®è¨ˆç®—
m_wins = m_results['WTeamID'].value_counts()
m_losses = m_results['LTeamID'].value_counts()
m_total_games = m_wins + m_losses
m_win_rate = (m_wins / m_total_games).fillna(0)

w_wins = w_results['WTeamID'].value_counts()
w_losses = w_results['LTeamID'].value_counts()
w_total_games = w_wins + w_losses
w_win_rate = (w_wins / w_total_games).fillna(0)

# å‹�ç�‡ã�®åˆ†å¸ƒã‚’å�¯è¦–åŒ–
plt.figure(figsize=(12, 5))
plt.hist(m_win_rate, bins=20, alpha=0.5, label="Men's Teams")
plt.hist(w_win_rate, bins=20, alpha=0.5, label="Women's Teams")
plt.xlabel("Win Rate")
plt.ylabel("Number of Teams")
plt.title("Win Rate Distribution")
plt.legend()
plt.show()

# å¾—ç‚¹ã�®ãƒ’ã‚¹ãƒˆã‚°ãƒ©ãƒ 
plt.figure(figsize=(12, 5))
plt.hist(m_results["WScore"], bins=50, alpha=0.5, label="Men's Winning Score")
plt.hist(w_results["WScore"], bins=50, alpha=0.5, label="Women's Winning Score")
plt.xlabel("Score")
plt.ylabel("Number of Games")
plt.title("Distribution of Winning Scores")
plt.legend()
plt.show()



# å‹�ç�‡ã�®è¨ˆç®—ï¼ˆã‚·ãƒ¼ã‚ºãƒ³ã‚’ç„¡è¦–ã�—ã�Ÿç´¯ç©�å‹�ç�‡ï¼‰
m_wins = m_results['WTeamID'].value_counts()
m_losses = m_results['LTeamID'].value_counts()
m_total_games = m_wins + m_losses
m_win_rate = (m_wins / m_total_games).fillna(0)

w_wins = w_results['WTeamID'].value_counts()
w_losses = w_results['LTeamID'].value_counts()
w_total_games = w_wins + w_losses
w_win_rate = (w_wins / w_total_games).fillna(0)

# å�¯è¦–åŒ–
plt.figure(figsize=(12, 5))
plt.hist(m_win_rate, bins=20, alpha=0.5, label="Men's Teams")
plt.hist(w_win_rate, bins=20, alpha=0.5, label="Women's Teams")
plt.xlabel("Win Rate")
plt.ylabel("Number of Teams")
plt.title("Win Rate Distribution")
plt.legend()
plt.show()



import pandas as pd

# ãƒ�ãƒ¼ãƒ ã�”ã�¨ã�®ç´¯ç©�å‹�ç�‡ã€�å¹³å�‡å¾—ç‚¹ã€�å¹³å�‡å¤±ç‚¹ã‚’è¨ˆç®—
def compute_team_stats(results):
    wins = results.groupby("WTeamID")["WScore"].agg(["count", "mean"]).rename(columns={"count": "Wins", "mean": "AvgWinScore"})
    losses = results.groupby("LTeamID")["LScore"].agg(["count", "mean"]).rename(columns={"count": "Losses", "mean": "AvgLossScore"})
    
    # å…¨è©¦å�ˆæ•°ã�¨å‹�ç�‡
    stats = wins.join(losses, how="outer").fillna(0)
    stats["TotalGames"] = stats["Wins"] + stats["Losses"]
    stats["WinRate"] = stats["Wins"] / stats["TotalGames"]
    
    # å¹³å�‡å¾—ç‚¹ï¼ˆå‹�åˆ©æ™‚ãƒ»æ•—åŒ—æ™‚ã�®ä¸¡æ–¹ã‚’è€ƒæ…®ï¼‰
    stats["AvgScore"] = (stats["Wins"] * stats["AvgWinScore"] + stats["Losses"] * stats["AvgLossScore"]) / stats["TotalGames"]
    
    return stats

# ãƒ‡ãƒ¼ã‚¿ã�®èª­ã�¿è¾¼ã�¿
data_path = "/kaggle/input/march-machine-learning-mania-2025"
m_results = pd.read_csv(f"{data_path}/MRegularSeasonCompactResults.csv")
w_results = pd.read_csv(f"{data_path}/WRegularSeasonCompactResults.csv")

# è¨ˆç®—
m_team_stats = compute_team_stats(m_results)
w_team_stats = compute_team_stats(w_results)

# ãƒ‡ãƒ¼ã‚¿ã�®è¡¨ç¤º
display(m_team_stats.head())
display(w_team_stats.head())



import pandas as pd

# ã‚·ãƒ¼ã‚ºãƒ³ã�”ã�¨ã�®å‹�ç�‡ã€�å¹³å�‡å¾—ç‚¹ã€�å¹³å�‡å¾—ç‚¹å·®ã‚’è¨ˆç®—
def compute_seasonal_stats(results):
    wins = results.groupby(["Season", "WTeamID"])["WScore"].agg(["count", "mean"]).rename(columns={"count": "Wins", "mean": "AvgWinScore"})
    losses = results.groupby(["Season", "LTeamID"])["LScore"].agg(["count", "mean"]).rename(columns={"count": "Losses", "mean": "AvgLossScore"})
    
    # ã‚·ãƒ¼ã‚ºãƒ³ã�”ã�¨ã�®çµ±è¨ˆã‚’ä½œæˆ�
    stats = wins.join(losses, how="outer").fillna(0)
    stats["TotalGames"] = stats["Wins"] + stats["Losses"]
    stats["WinRate"] = stats["Wins"] / stats["TotalGames"]
    
    # å¹³å�‡å¾—ç‚¹ï¼ˆå‹�åˆ©ãƒ»æ•—åŒ—ã‚’è€ƒæ…®ï¼‰
    stats["AvgScore"] = (stats["Wins"] * stats["AvgWinScore"] + stats["Losses"] * stats["AvgLossScore"]) / stats["TotalGames"]
    
    # å¹³å�‡å¾—ç‚¹å·®
    stats["AvgScoreDiff"] = stats["AvgWinScore"] - stats["AvgLossScore"]
    
    return stats

# ãƒ‡ãƒ¼ã‚¿ã�®èª­ã�¿è¾¼ã�¿
data_path = "/kaggle/input/march-machine-learning-mania-2025"
m_results = pd.read_csv(f"{data_path}/MRegularSeasonCompactResults.csv")
w_results = pd.read_csv(f"{data_path}/WRegularSeasonCompactResults.csv")

# ã‚·ãƒ¼ã‚ºãƒ³ã�”ã�¨ã�®çµ±è¨ˆã‚’å�–å¾—
m_season_stats = compute_seasonal_stats(m_results)
w_season_stats = compute_seasonal_stats(w_results)

# ãƒ‡ãƒ¼ã‚¿è¡¨ç¤º
display(m_season_stats.head())
display(w_season_stats.head())



# ç›´è¿‘5å¹´é–“ã�®ãƒ‡ãƒ¼ã‚¿ã‚’ä½¿ã�£ã�¦ç‰¹å¾´é‡�ã‚’è¨ˆç®—
def compute_recent_stats(results, recent_years=5):
    latest_season = results["Season"].max()
    filtered_results = results[results["Season"] >= latest_season - recent_years]

    return compute_seasonal_stats(filtered_results).groupby("WTeamID").mean()

# ç›´è¿‘5å¹´é–“ã�®çµ±è¨ˆã‚’å�–å¾—
m_recent_stats = compute_recent_stats(m_results)
w_recent_stats = compute_recent_stats(w_results)

# ãƒ‡ãƒ¼ã‚¿è¡¨ç¤º
display(m_recent_stats.head())
display(w_recent_stats.head())



import numpy as np

# åˆ�æœŸEloãƒ¬ãƒ¼ãƒ†ã‚£ãƒ³ã‚°
INITIAL_ELO = 1500

# Eloãƒ¬ãƒ¼ãƒ†ã‚£ãƒ³ã‚°ã‚’æ›´æ–°ã�™ã‚‹é–¢æ•°
def update_elo(winner_elo, loser_elo, k=20):
    """
    Eloãƒ¬ãƒ¼ãƒ†ã‚£ãƒ³ã‚°ã‚’æ›´æ–°ã�™ã‚‹
    :param winner_elo: å‹�è€…ã�®ç�¾åœ¨ã�®Eloãƒ¬ãƒ¼ãƒ†ã‚£ãƒ³ã‚°
    :param loser_elo: æ•—è€…ã�®ç�¾åœ¨ã�®Eloãƒ¬ãƒ¼ãƒ†ã‚£ãƒ³ã‚°
    :param k: æ›´æ–°ä¿‚æ•°ï¼ˆãƒ‡ãƒ•ã‚©ãƒ«ãƒˆ20ï¼‰
    :return: æ›´æ–°å¾Œã�®å‹�è€…ã�¨æ•—è€…ã�®Eloãƒ¬ãƒ¼ãƒ†ã‚£ãƒ³ã‚°
    """
    expected_win_prob = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    winner_elo += k * (1 - expected_win_prob)
    loser_elo += k * (0 - (1 - expected_win_prob))
    return winner_elo, loser_elo

# ãƒ�ãƒ¼ãƒ ã�”ã�¨ã�®Eloã‚¹ã‚³ã‚¢ã‚’è¨ˆç®—
def compute_elo_ratings(results):
    elo_ratings = {}  # ãƒ�ãƒ¼ãƒ ã�®Eloã‚¹ã‚³ã‚¢ã‚’æ ¼ç´�
    results = results.sort_values("DayNum")  # æ—¥ä»˜é †ã�«ã‚½ãƒ¼ãƒˆ
    
    for _, row in results.iterrows():
        w_team = row["WTeamID"]
        l_team = row["LTeamID"]

        # åˆ�æœŸEloã‚¹ã‚³ã‚¢ã‚’è¨­å®š
        if w_team not in elo_ratings:
            elo_ratings[w_team] = INITIAL_ELO
        if l_team not in elo_ratings:
            elo_ratings[l_team] = INITIAL_ELO

        # Eloã‚¹ã‚³ã‚¢ã‚’æ›´æ–°
        new_w_elo, new_l_elo = update_elo(elo_ratings[w_team], elo_ratings[l_team])
        elo_ratings[w_team] = new_w_elo
        elo_ratings[l_team] = new_l_elo

    return elo_ratings

# ç”·å­�ãƒ»å¥³å­�ã�®Eloã‚¹ã‚³ã‚¢ã‚’è¨ˆç®—
m_elo_ratings = compute_elo_ratings(m_results)
w_elo_ratings = compute_elo_ratings(w_results)

# DataFrameã�«å¤‰æ�›
m_elo_df = pd.DataFrame(m_elo_ratings.items(), columns=["TeamID", "EloRating"]).sort_values("EloRating", ascending=False)
w_elo_df = pd.DataFrame(w_elo_ratings.items(), columns=["TeamID", "EloRating"]).sort_values("EloRating", ascending=False)

# ãƒ‡ãƒ¼ã‚¿è¡¨ç¤º
display(m_elo_df.head(10))  # ç”·å­�ãƒˆãƒƒãƒ—10ãƒ�ãƒ¼ãƒ 
display(w_elo_df.head(10))  # å¥³å­�ãƒˆãƒƒãƒ—10ãƒ�ãƒ¼ãƒ 



# æ��å‡ºç”¨ãƒ‡ãƒ¼ã‚¿ã�®èª­ã�¿è¾¼ã�¿
data_path = "/kaggle/input/march-machine-learning-mania-2025"
submission = pd.read_csv(f"{data_path}/SampleSubmissionStage1.csv")

# Eloãƒ¬ãƒ¼ãƒ†ã‚£ãƒ³ã‚°ã‚’ç‰¹å¾´é‡�ã�«çµ±å�ˆ
def add_elo_features(df, elo_df):
    df["Season"] = df["ID"].apply(lambda x: int(x.split("_")[0]))
    df["Team1"] = df["ID"].apply(lambda x: int(x.split("_")[1]))
    df["Team2"] = df["ID"].apply(lambda x: int(x.split("_")[2]))

    # Eloã‚¹ã‚³ã‚¢ã‚’ãƒ�ãƒ¼ã‚¸
    df = df.merge(elo_df, left_on="Team1", right_on="TeamID", how="left").rename(columns={"EloRating": "EloTeam1"}).drop(columns=["TeamID"])
    df = df.merge(elo_df, left_on="Team2", right_on="TeamID", how="left").rename(columns={"EloRating": "EloTeam2"}).drop(columns=["TeamID"])

    # Eloã‚¹ã‚³ã‚¢å·®ã‚’è¨ˆç®—
    df["EloDiff"] = df["EloTeam1"] - df["EloTeam2"]

    return df

# ç”·å­�ã�¨å¥³å­�ã‚’çµ±å�ˆï¼ˆã‚µãƒ³ãƒ—ãƒ«ãƒ‡ãƒ¼ã‚¿ã�Œä¸¡æ–¹å…¥ã�£ã�¦ã�„ã‚‹å‰�æ��ï¼‰
submission = add_elo_features(submission, pd.concat([m_elo_df, w_elo_df]))

# ãƒ‡ãƒ¼ã‚¿è¡¨ç¤º
display(submission.head())



import pandas as pd
import os

# ãƒ‡ãƒ¼ã‚¿ã�®ãƒ‘ã‚¹
data_path = "/kaggle/input/march-machine-learning-mania-2025"

# 2025å¹´ã�«é–¢é€£ã�™ã‚‹ãƒ•ã‚¡ã‚¤ãƒ«
files_to_check = [
    "MGameCities.csv", "MMasseyOrdinals.csv", "MConferenceTourneyGames.csv",
    "MRegularSeasonCompactResults.csv", "MRegularSeasonDetailedResults.csv", "MSeasons.csv",
    "MTeamCoaches.csv", "MTeamConferences.csv", "SampleSubmissionStage2.csv",
    "WConferenceTourneyGames.csv", "WGameCities.csv", "WRegularSeasonCompactResults.csv",
    "WRegularSeasonDetailedResults.csv", "WSeasons.csv", "WTeamConferences.csv"
]

# å�„ãƒ•ã‚¡ã‚¤ãƒ«ã�®ãƒ˜ãƒƒãƒ€ãƒ¼ã‚’ç¢ºèª�
for file in files_to_check:
    file_path = os.path.join(data_path, file)
    try:
        df = pd.read_csv(file_path, nrows=5)  # æœ€åˆ�ã�®5è¡Œã� ã�‘èª­ã‚€
        print(f"\nğŸ“‚ {file}:")
        print(df.head(), "\n")
    except Exception as e:
        print(f"\nâš  {file} ã�®èª­ã�¿è¾¼ã�¿ã�§ã‚¨ãƒ©ãƒ¼: {e}\n")



import pandas as pd

# ãƒ‡ãƒ¼ã‚¿ã�®ãƒ‘ã‚¹
data_path = "/kaggle/input/march-machine-learning-mania-2025"

# 2025å¹´ã�®è©¦å�ˆãƒªã‚¹ãƒˆã‚’ãƒ­ãƒ¼ãƒ‰
submission_2025 = pd.read_csv(f"{data_path}/SampleSubmissionStage2.csv")

# IDã�‹ã‚‰ Season, Team1, Team2 ã‚’æŠ½å‡º
submission_2025[["Season", "Team1", "Team2"]] = submission_2025["ID"].str.split("_", expand=True)
submission_2025["Season"] = submission_2025["Season"].astype(int)
submission_2025["Team1"] = submission_2025["Team1"].astype(int)
submission_2025["Team2"] = submission_2025["Team2"].astype(int)

# Eloãƒ¬ãƒ¼ãƒ†ã‚£ãƒ³ã‚°ã‚’çµ±å�ˆ
def add_elo_features(df, elo_df):
    df = df.merge(elo_df, left_on="Team1", right_on="TeamID", how="left").rename(columns={"EloRating": "EloTeam1"}).drop(columns=["TeamID"])
    df = df.merge(elo_df, left_on="Team2", right_on="TeamID", how="left").rename(columns={"EloRating": "EloTeam2"}).drop(columns=["TeamID"])
    
    # Eloã‚¹ã‚³ã‚¢å·®ã‚’è¨ˆç®—
    df["EloDiff"] = df["EloTeam1"] - df["EloTeam2"]
    
    return df

# ç”·å­�ãƒ»å¥³å­�ã�®Eloãƒ‡ãƒ¼ã‚¿ã‚’çµ±å�ˆ
elo_combined_df = pd.concat([m_elo_df, w_elo_df])

# Eloç‰¹å¾´é‡�ã‚’è¿½åŠ 
submission_2025 = add_elo_features(submission_2025, elo_combined_df)

# ãƒ‡ãƒ¼ã‚¿è¡¨ç¤º
display(submission_2025.head())



# ã‚«ãƒ³ãƒ•ã‚¡ãƒ¬ãƒ³ã‚¹æƒ…å ±ã�®ãƒ­ãƒ¼ãƒ‰
m_team_conf = pd.read_csv(f"{data_path}/MTeamConferences.csv")
w_team_conf = pd.read_csv(f"{data_path}/WTeamConferences.csv")

# 2025å¹´ã�®ã‚«ãƒ³ãƒ•ã‚¡ãƒ¬ãƒ³ã‚¹æƒ…å ±ã�®ã�¿å�–å¾—
m_team_conf_2025 = m_team_conf[m_team_conf["Season"] == 2025].drop(columns=["Season"])
w_team_conf_2025 = w_team_conf[w_team_conf["Season"] == 2025].drop(columns=["Season"])

# ã‚«ãƒ³ãƒ•ã‚¡ãƒ¬ãƒ³ã‚¹æƒ…å ±ã‚’ãƒ�ãƒ¼ã‚¸
def add_conference_features(df, conf_df):
    df = df.merge(conf_df, left_on="Team1", right_on="TeamID", how="left").rename(columns={"ConfAbbrev": "ConfTeam1"}).drop(columns=["TeamID"])
    df = df.merge(conf_df, left_on="Team2", right_on="TeamID", how="left").rename(columns={"ConfAbbrev": "ConfTeam2"}).drop(columns=["TeamID"])
    
    # å�Œã�˜ã‚«ãƒ³ãƒ•ã‚¡ãƒ¬ãƒ³ã‚¹ã�‹ã�©ã�†ã�‹
    df["SameConf"] = (df["ConfTeam1"] == df["ConfTeam2"]).astype(int)
    
    return df

# ç”·å­�ãƒ»å¥³å­�ã�®ã‚«ãƒ³ãƒ•ã‚¡ãƒ¬ãƒ³ã‚¹æƒ…å ±ã‚’çµ±å�ˆ
team_conf_2025 = pd.concat([m_team_conf_2025, w_team_conf_2025])

# ã‚«ãƒ³ãƒ•ã‚¡ãƒ¬ãƒ³ã‚¹æƒ…å ±ã‚’è¿½åŠ 
submission_2025 = add_conference_features(submission_2025, team_conf_2025)

# ãƒ‡ãƒ¼ã‚¿è¡¨ç¤º
display(submission_2025.head())



import pandas as pd

# ãƒ‡ãƒ¼ã‚¿ã�®ãƒ‘ã‚¹
data_path = "/kaggle/input/march-machine-learning-mania-2025"

# ã‚³ãƒ¼ãƒ�æƒ…å ±ã�®ãƒ­ãƒ¼ãƒ‰ï¼ˆç”·å­�ã�®ã�¿ï¼‰
m_coaches = pd.read_csv(f"{data_path}/MTeamCoaches.csv")

# 2024å¹´ã�¨2025å¹´ã�®ã‚³ãƒ¼ãƒ�æƒ…å ±ã‚’å�–å¾—
m_coaches_2024 = m_coaches[m_coaches["Season"] == 2024][["TeamID", "CoachName"]].rename(columns={"CoachName": "CoachName_2024"})
m_coaches_2025 = m_coaches[m_coaches["Season"] == 2025][["TeamID", "CoachName"]].rename(columns={"CoachName": "CoachName_2025"})

# 2024å¹´ã�¨2025å¹´ã�§ã‚³ãƒ¼ãƒ�ã�Œç•°ã�ªã‚‹ãƒ�ãƒ¼ãƒ ã‚’ç¢ºèª�
coaches_diff = m_coaches_2024.merge(m_coaches_2025, on="TeamID", how="inner")
coaches_diff = coaches_diff[coaches_diff["CoachName_2024"] != coaches_diff["CoachName_2025"]]

# ã‚³ãƒ¼ãƒ�ã�Œå¤‰ã‚�ã�£ã�Ÿãƒ�ãƒ¼ãƒ æ•°ã‚’è¡¨ç¤º
print(f"âš  2025å¹´ã�«ã‚³ãƒ¼ãƒ�ã�Œå¤‰ã‚�ã�£ã�Ÿãƒ�ãƒ¼ãƒ æ•°: {coaches_diff.shape[0]}")

# ã‚³ãƒ¼ãƒ�ã�Œå¤‰ã‚�ã�£ã�Ÿãƒ�ãƒ¼ãƒ ã‚’è¡¨ç¤º
if coaches_diff.shape[0] > 0:
    from IPython.display import display
    display(coaches_diff)
else:
    print("2025å¹´ã�«ã‚³ãƒ¼ãƒ�å¤‰æ›´ã�Œã�‚ã�£ã�Ÿãƒ�ãƒ¼ãƒ ã�¯ã�‚ã‚Šã�¾ã�›ã‚“ã€‚")



# `MTeamCoaches.csv` ã‚’ãƒ­ãƒ¼ãƒ‰ï¼ˆæœ€æ–°ã�®2024å¹´ãƒ‡ãƒ¼ã‚¿ã‚’å�–å¾—ï¼‰
coaches_2024 = pd.read_csv(f"{data_path}/MTeamCoaches.csv")

# 2024å¹´ã�®ãƒ‡ãƒ¼ã‚¿ã�®ã�¿ã‚’å�–å¾—
coaches_2024 = coaches_2024[coaches_2024["Season"] == 2024]

# **2025å¹´ã�®ãƒ‡ãƒ¼ã‚¿ã�¨ã�—ã�¦ã‚³ãƒ”ãƒ¼**
coaches_2025 = coaches_2024.copy()
coaches_2025["Season"] = 2025  # ã‚·ãƒ¼ã‚ºãƒ³ã‚’æ›´æ–°

# **2025å¹´ã�®ãƒ‡ãƒ¼ã‚¿ã‚’ CSV ã�«ä¿�å­˜**
coaches_2025_path = "/kaggle/working/MTeamCoaches_2025.csv"
coaches_2025.to_csv(coaches_2025_path, index=False)
print(f"2025å¹´ã�®ã‚³ãƒ¼ãƒ�ãƒ‡ãƒ¼ã‚¿ã‚’ä½œæˆ�ã�—ã�¾ã�—ã�Ÿ: {coaches_2025_path}")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Kaggle ãƒ‡ãƒ¼ã‚¿ã�®ãƒ‘ã‚¹
data_path = "/kaggle/input/march-machine-learning-mania-2025"

# **Elo ãƒ¬ãƒ¼ãƒ†ã‚£ãƒ³ã‚°ã‚’è¨ˆç®—**
def calculate_elo(df, k=20):
    elo_dict = {}
    base_elo = 1500

    for season in df["Season"].unique():
        season_games = df[df["Season"] == season]
        
        for _, row in season_games.iterrows():
            team1, team2 = row["WTeamID"], row["LTeamID"]
            elo_dict.setdefault(team1, base_elo)
            elo_dict.setdefault(team2, base_elo)

            # ç�¾åœ¨ã�® Elo ãƒ¬ãƒ¼ãƒˆ
            r1, r2 = elo_dict[team1], elo_dict[team2]
            e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
            e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))

            # å‹�è€…ã�¨æ•—è€…ã�® Elo ã‚’æ›´æ–°
            elo_dict[team1] = r1 + k * (1 - e1)
            elo_dict[team2] = r2 + k * (0 - e2)

    return pd.DataFrame(elo_dict.items(), columns=["TeamID", "EloRating"])

# **è©¦å�ˆçµ�æ�œãƒ‡ãƒ¼ã‚¿ã‚’ãƒ­ãƒ¼ãƒ‰**
m_results = pd.read_csv(f"{data_path}/MRegularSeasonCompactResults.csv")

# **Elo ãƒ¬ãƒ¼ãƒ†ã‚£ãƒ³ã‚°ã‚’è¨ˆç®—**
teams_elo = calculate_elo(m_results)

# **Elo ãƒ¬ãƒ¼ãƒ†ã‚£ãƒ³ã‚°ã‚’ä¿�å­˜**
elo_path = "/kaggle/working/teams_elo_2025.csv"
teams_elo.to_csv(elo_path, index=False)
print(f"Elo ratings saved to: {elo_path}")

# **äºˆæ¸¬å¯¾è±¡ãƒ‡ãƒ¼ã‚¿ã�®ãƒ­ãƒ¼ãƒ‰**
submission_2025 = pd.read_csv(f"{data_path}/SampleSubmissionStage2.csv")

# ID ã‚’åˆ†å‰²ã�—ã�¦ Team1, Team2 ã‚’æŠ½å‡º
submission_2025["Season"] = submission_2025["ID"].apply(lambda x: int(x.split("_")[0]))
submission_2025["Team1"] = submission_2025["ID"].apply(lambda x: int(x.split("_")[1]))
submission_2025["Team2"] = submission_2025["ID"].apply(lambda x: int(x.split("_")[2]))

# **Elo ãƒ¬ãƒ¼ãƒ†ã‚£ãƒ³ã‚°ã‚’ãƒ�ãƒ¼ã‚¸**
submission_2025 = submission_2025.merge(teams_elo, left_on="Team1", right_on="TeamID", how="left").rename(columns={"EloRating": "EloTeam1"})
submission_2025.drop(columns=["TeamID"], inplace=True)

submission_2025 = submission_2025.merge(teams_elo, left_on="Team2", right_on="TeamID", how="left").rename(columns={"EloRating": "EloTeam2"})
submission_2025.drop(columns=["TeamID"], inplace=True)

# **NaN ã�® Elo ãƒ¬ãƒ¼ãƒ†ã‚£ãƒ³ã‚°ã‚’ 1500 ã�«è£œå®Œ**
submission_2025["EloTeam1"].fillna(1500, inplace=True)
submission_2025["EloTeam2"].fillna(1500, inplace=True)

# **EloDiff ã‚’è¨ˆç®—**
submission_2025["EloDiff"] = submission_2025["EloTeam1"] - submission_2025["EloTeam2"]

# **ã‚«ãƒ³ãƒ•ã‚¡ãƒ¬ãƒ³ã‚¹ãƒ‡ãƒ¼ã‚¿ã‚’é�©ç”¨**
team_conferences = pd.read_csv(f"{data_path}/MTeamConferences.csv")

# Team1 ã�®ã‚«ãƒ³ãƒ•ã‚¡ãƒ¬ãƒ³ã‚¹æƒ…å ±ã‚’ãƒ�ãƒ¼ã‚¸
submission_2025 = submission_2025.merge(team_conferences, left_on=["Season", "Team1"], right_on=["Season", "TeamID"], how="left").rename(columns={"ConfAbbrev": "ConfTeam1"})
submission_2025.drop(columns=["TeamID"], inplace=True)

# Team2 ã�®ã‚«ãƒ³ãƒ•ã‚¡ãƒ¬ãƒ³ã‚¹æƒ…å ±ã‚’ãƒ�ãƒ¼ã‚¸
submission_2025 = submission_2025.merge(team_conferences, left_on=["Season", "Team2"], right_on=["Season", "TeamID"], how="left").rename(columns={"ConfAbbrev": "ConfTeam2"})
submission_2025.drop(columns=["TeamID"], inplace=True)

# `SameConf` ã‚’ä½œæˆ�ï¼ˆå�Œã�˜ã‚«ãƒ³ãƒ•ã‚¡ãƒ¬ãƒ³ã‚¹ã�ªã‚‰ 1, ã��ã�†ã�§ã�ªã�‘ã‚Œã�° 0ï¼‰
submission_2025["SameConf"] = (submission_2025["ConfTeam1"] == submission_2025["ConfTeam2"]).astype(int)

# **NaN ã�® `SameConf` ã‚’ 0 ã�«è£œå®Œ**
submission_2025["SameConf"].fillna(0, inplace=True)

# **ã‚³ãƒ¼ãƒ�ãƒ‡ãƒ¼ã‚¿ã�®é�©ç”¨**
coaches_2024 = pd.read_csv(f"{data_path}/MTeamCoaches.csv")

# **2025å¹´ã�®ã‚³ãƒ¼ãƒ�ãƒ‡ãƒ¼ã‚¿ã‚’ 2024å¹´ãƒ‡ãƒ¼ã‚¿ã�‹ã‚‰ã‚³ãƒ”ãƒ¼**
coaches_2025 = coaches_2024[coaches_2024["Season"] == 2024].copy()
coaches_2025["Season"] = 2025

# **ã‚³ãƒ¼ãƒ�æƒ…å ±ã‚’ãƒ�ãƒ¼ã‚¸**
for team_col, coach_col in [("Team1", "CoachTeam1"), ("Team2", "CoachTeam2")]:
    submission_2025 = submission_2025.merge(coaches_2024, left_on=["Season", team_col], right_on=["Season", "TeamID"], how="left").rename(columns={"CoachName": f"{coach_col}_2024"})
    submission_2025.drop(columns=["TeamID", "FirstDayNum", "LastDayNum"], inplace=True)

    submission_2025 = submission_2025.merge(coaches_2025, left_on=["Season", team_col], right_on=["Season", "TeamID"], how="left").rename(columns={"CoachName": f"{coach_col}_2025"})
    submission_2025.drop(columns=["TeamID", "FirstDayNum", "LastDayNum"], inplace=True)

# **ã‚³ãƒ¼ãƒ�å¤‰æ›´ã�®ãƒ•ãƒ©ã‚°**
submission_2025["CoachChanged1"] = (submission_2025["CoachTeam1_2024"] != submission_2025["CoachTeam1_2025"]).astype(int)
submission_2025["CoachChanged2"] = (submission_2025["CoachTeam2_2024"] != submission_2025["CoachTeam2_2025"]).astype(int)

# **NaN ã�® `CoachChanged1, CoachChanged2` ã‚’ 0 ã�«è£œå®Œ**
submission_2025["CoachChanged1"].fillna(0, inplace=True)
submission_2025["CoachChanged2"].fillna(0, inplace=True)

# **ãƒ¢ãƒ‡ãƒ«é�©ç”¨**
features = ["EloDiff", "SameConf", "CoachChanged1", "CoachChanged2"]
scaler = StandardScaler()
submission_2025[features] = scaler.fit_transform(submission_2025[features])

model = LogisticRegression(C=0.05)
model.fit(submission_2025[features], (submission_2025["EloDiff"] > 0).astype(int))

submission_2025["Pred"] = model.predict_proba(submission_2025[features])[:, 1]

duplicate_count = submission_2025["ID"].duplicated().sum()
print(f"é‡�è¤‡ã�—ã�¦ã�„ã‚‹è¡Œæ•°: {duplicate_count}")

if duplicate_count > 0:
    print("é‡�è¤‡ã‚’å‰Šé™¤ã�—ã�¾ã�™...")
    submission_2025.drop_duplicates(subset=["ID"], keep="first", inplace=True)

# **é‡�è¤‡ã�Œæ­£ã�—ã��å‰Šé™¤ã�•ã‚Œã�Ÿã�‹å†�ãƒ�ã‚§ãƒƒã‚¯**
duplicate_count_after = submission_2025["ID"].duplicated().sum()
print(f"é‡�è¤‡å‰Šé™¤å¾Œã�®è¡Œæ•°: {duplicate_count_after}")


# **ãƒ‡ãƒ¼ã‚¿ã‚’ CSV ã�«ä¿�å­˜**
submission_2025[["ID", "Pred"]].to_csv("/kaggle/working/submission_final_2025.csv", index=False)
print("Submission file saved!")


