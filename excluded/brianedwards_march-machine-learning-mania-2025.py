try:
    get_ipython().run_line_magic("reset", "-f")
except NameError:
    is_notebook = False
    p = print
else:
    is_notebook = True
    p = display
    # p = print

import warnings

warnings.simplefilter("ignore")

from math import floor
import numpy as np
import pandas as pd

if is_notebook:
    import matplotlib.pyplot as plt
    import seaborn as sns
else:
    import asciichartpy

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss
from scipy.stats import norm
from scipy.optimize import minimize

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

input_dir = "march-machine-learning-mania-2025"


df_names = []
df_names_compact = []
df_names_detailed = []

for gender in ["M", "W"]:
    for part in ["RegularSeason", "NCAATourney", "SecondaryTourney"]:
        for result_type in ["Compact", "Detailed"]:
            if part == "SecondaryTourney" and result_type == "Detailed":
                continue
            df_name = f"{gender}{part}{result_type}Results"
            df_names.append(df_name)
            if result_type == "Compact":
                df_names_compact.append(df_name)
            else:
                df_names_detailed.append(df_name)
            print(df_name)
            path = f"../input/{input_dir}/{df_name}.csv"
            df = pd.read_csv(path)
            df["Part"] = part
            globals()[df_name] = df

season_min = WRegularSeasonDetailedResults["Season"].min()

for df_name in df_names:
    df = globals()[df_name]
    globals()[df_name] = df[df["Season"] >= season_min]


game_stats_1_2 = pd.DataFrame()

for df_name in df_names_detailed:
    game_stats_1_2 = pd.concat([game_stats_1_2, globals()[df_name]])

game_stats_1_2 = game_stats_1_2.reset_index(drop=True)
mask = game_stats_1_2["WTeamID"] < game_stats_1_2["LTeamID"]


def to_1_2(col_W_L):
    game_stats_1_2.loc[mask, f"{col_W_L}_1"] = game_stats_1_2[f"W{col_W_L}"]
    game_stats_1_2.loc[mask, f"{col_W_L}_2"] = game_stats_1_2[f"L{col_W_L}"]
    game_stats_1_2.loc[~mask, f"{col_W_L}_1"] = game_stats_1_2[f"L{col_W_L}"]
    game_stats_1_2.loc[~mask, f"{col_W_L}_2"] = game_stats_1_2[f"W{col_W_L}"]
    game_stats_1_2[f"{col_W_L}_1"] = game_stats_1_2[f"{col_W_L}_1"].astype(
        game_stats_1_2[f"W{col_W_L}"].dtype
    )
    game_stats_1_2[f"{col_W_L}_2"] = game_stats_1_2[f"{col_W_L}_2"].astype(
        game_stats_1_2[f"W{col_W_L}"].dtype
    )


game_stats_1_2["WLoc"] = game_stats_1_2["WLoc"].astype("category")
game_stats_1_2["LLoc"] = game_stats_1_2["WLoc"].cat.rename_categories(
    {"A": "H", "H": "A"}
)

for c in game_stats_1_2.columns:
    if c.startswith("W"):
        to_1_2(c[1:])

game_stats_1_2["Spread_1"] = game_stats_1_2["Score_1"] - game_stats_1_2["Score_2"]
game_stats_1_2["Spread_2"] = game_stats_1_2["Score_2"] - game_stats_1_2["Score_1"]
game_stats_1_2 = game_stats_1_2[[c for c in game_stats_1_2 if c[0] not in ("W", "L")]]
game_stats_1_2 = game_stats_1_2.drop(columns="NumOT")
game_stats_1_2.insert(2, "TeamID_1", game_stats_1_2.pop("TeamID_1"))
game_stats_1_2.insert(3, "TeamID_2", game_stats_1_2.pop("TeamID_2"))
game_stats_1_2 = game_stats_1_2.sort_values(
    ["Season", "DayNum", "TeamID_1", "TeamID_2"]
)
game_stats_1_2 = game_stats_1_2.reset_index(drop=True)
print(f"game_stats_1_2     {game_stats_1_2.shape[0]:,}")
p(game_stats_1_2)
print()

print(f"MRegularSeasonDetai{MRegularSeasonDetailedResults.shape[0]:>7,}")
print(f"WRegularSeasonDetai{WRegularSeasonDetailedResults.shape[0]:>7,}")
print(
    f"RegularSeasonDetail{MRegularSeasonDetailedResults.shape[0]+WRegularSeasonDetailedResults.shape[0]:>7,}"
)

reg_season = game_stats_1_2[game_stats_1_2["Part"] == "RegularSeason"]
print(f"reg_season         {reg_season.shape[0]:,}")

game_stats_1_o = reg_season.rename(
    columns={c: f"{c[:-2]}_o" for c in reg_season if c[-2:] == "_1"}
)
game_stats_1_o = game_stats_1_o.rename(
    columns={c: f"{c[:-2]}_d" for c in game_stats_1_o if c[-2:] == "_2"}
)
print(f"game_stats_1_o     {game_stats_1_o.shape[0]:,}")

game_stats_2_o = reg_season.rename(
    columns={c: f"{c[:-2]}_o" for c in reg_season if c[-2:] == "_2"}
)
game_stats_2_o = game_stats_2_o.rename(
    columns={c: f"{c[:-2]}_d" for c in game_stats_2_o if c[-2:] == "_1"}
)
print(f"game_stats_2_o     {game_stats_2_o.shape[0]:,}")
print(f"game_stats_1_o+2_o {game_stats_1_o.shape[0]+game_stats_2_o.shape[0]:,}\n")

game_stats_o_d = pd.concat([game_stats_1_o, game_stats_2_o])
game_stats_o_d = game_stats_o_d.drop(columns="Part")
game_stats_o_d.insert(2, "TeamID_o", game_stats_o_d.pop("TeamID_o"))
game_stats_o_d.insert(3, "TeamID_d", game_stats_o_d.pop("TeamID_d"))
game_stats_o_d = game_stats_o_d.sort_values(
    ["Season", "DayNum", "TeamID_o", "TeamID_d"]
)
game_stats_o_d = game_stats_o_d.reset_index(drop=True)
print(f"game_stats_o_d     {game_stats_o_d.shape[0]:,}")
p(game_stats_o_d)
print()


# Look at the distribution of games per team
games_per_team = game_stats_o_d.rename(columns={"TeamID_o": "TeamID"}).drop(
    columns="TeamID_d"
)
games_per_team_count = games_per_team.groupby(["Season", "TeamID"]).size()

# Count how many team-seasons have each number of games
games_count_distribution = games_per_team_count.value_counts().sort_index()
print("Distribution of games per team-season:")
print(games_count_distribution)

# Look at specific examples of teams with very few games
teams_with_few_games = games_per_team_count[games_per_team_count <= 10].reset_index()
print("\nSample of teams with 10 or fewer games:")
print(teams_with_few_games.head(10))

# Check if there's a pattern by season
few_games_by_season = teams_with_few_games.groupby("Season").size()
print("\nNumber of teams with few games by season:")
print(few_games_by_season)


# p(game_stats_o_d)
season_stats = game_stats_o_d.rename(columns={"TeamID_o": "TeamID"})
season_stats = season_stats.drop(columns="TeamID_d")
season_stats_g = season_stats.groupby(["Season", "TeamID"])

season_stats = season_stats_g[
    [c for c in season_stats if c.endswith("_o")]
    + [c for c in season_stats if c.endswith("_d")]
].mean()

season_stats = season_stats.reset_index()
season_stats = season_stats.sort_values(["Season", "TeamID"])
season_stats = season_stats.reset_index(drop=True)

print(f"season_stats   {season_stats.shape[0]:,}")
p(season_stats)
print()


p(season_stats[(season_stats["Season"] == 2010) & (season_stats["TeamID"] == 1393)])
# p(game_stats_o_d[(game_stats_o_d["Season"]==2010) & (game_stats_o_d["TeamID_o"]==1393)])
# p(MRegularSeasonDetailedResults[(MRegularSeasonDetailedResults["Season"]==2010) & (MRegularSeasonDetailedResults["WTeamID"]==1393)])
# p(MRegularSeasonDetailedResults[(MRegularSeasonDetailedResults["Season"]==2010) & (MRegularSeasonDetailedResults["LTeamID"]==1393)])


matchups = game_stats_o_d[["Season", "DayNum", "TeamID_o", "TeamID_d"]]
p(matchups)

# matchups with the season stats of d
opp_season_stats = pd.merge(
    matchups,
    season_stats,
    left_on=["Season", "TeamID_d"],
    right_on=["Season", "TeamID"],
)
opp_season_stats = opp_season_stats.drop(columns="TeamID")
opp_season_stats = opp_season_stats.sort_values(
    ["Season", "DayNum", "TeamID_o", "TeamID_d"]
)
opp_season_stats = opp_season_stats.reset_index(drop=True)
p(opp_season_stats)

# strength of schedule
sos_stats = opp_season_stats.rename(columns={"TeamID_o": "TeamID"})
sos_stats = sos_stats.drop(columns="TeamID_d")
sos_stats_g = sos_stats.groupby(["Season", "TeamID"])

sos_stats = sos_stats_g[
    [c for c in sos_stats if c.endswith("_o")]
    + [c for c in sos_stats if c.endswith("_d")]
].mean()

sos_stats = sos_stats.reset_index()
sos_stats = sos_stats.sort_values(["Season", "TeamID"])
sos_stats = sos_stats.reset_index(drop=True)

print(f"sos_stats   {sos_stats.shape[0]:,}")
p(sos_stats)
print()


season_self_sos = pd.merge(
    season_stats, sos_stats, on=["Season", "TeamID"], suffixes=["_self", "_sos"]
)
print(f"season_self_sos {season_self_sos.shape[0]:,}")
p(season_self_sos)
print()


train = (
    game_stats_1_2.groupby(["Season", "TeamID_1", "TeamID_2"])["Spread_1"]
    .mean()
    .reset_index()
)

train = pd.merge(
    train,
    season_self_sos,
    left_on=["Season", "TeamID_1"],
    right_on=["Season", "TeamID"],
)
train = train.drop(columns="TeamID")

train = train.rename(columns={c: f"{c}_1" for c in train if c[-4:] in ("self", "_sos")})

train = pd.merge(
    train,
    season_self_sos,
    left_on=["Season", "TeamID_2"],
    right_on=["Season", "TeamID"],
)
train = train.drop(columns="TeamID")

train = train.rename(columns={c: f"{c}_2" for c in train if c[-4:] in ("self", "_sos")})

train = train.drop(columns=[c for c in train if c.startswith("Spread_d_")])

train = train.sort_values(["Season", "TeamID_1", "TeamID_2"])
train = train.reset_index(drop=True)

train.to_csv("train_orig.csv", index=False)

print(f"train        {train.shape[0]:,}")
p(train)
print()

X = train.drop(columns=["Season", "TeamID_1", "TeamID_2", "Spread_1"])

print(f"X        {X.shape[0]:,}")
p(X)
print()

y = train["Spread_1"].rename("y")

print(f"y        {y.shape[0]:,}")
p(y)
print()


p(train[train["Season"] == 2021])


kfold = KFold(shuffle=True, random_state=42)
fold_models = []
y_pred_oof = np.zeros(len(train))

for fold_n, (i_fold, i_oof) in enumerate(kfold.split(train.index)):
    print(f"fold {fold_n}", flush=True)
    m = xgb.XGBRegressor()
    m.fit(X.iloc[i_fold], y.iloc[i_fold])
    fold_models.append(m)
    y_pred_oof[i_oof] = m.predict(X.iloc[i_oof])


suptitle = "NCAA D1 basketball 2010-2025 men's and women's"
title = "Predicted point spread distribution"

if is_notebook:
    sns.histplot(y_pred_oof, bins=50)
    plt.suptitle(suptitle)
    plt.title(title)

else:
    print()
    print(suptitle)
    print(title)
    hist, edges = np.histogram(y_pred_oof, bins=50)
    width = 80
    print(asciichartpy.plot(hist, {"height": 10, "width": width}))
    w50 = (width - 11) // 2
    w66 = (width - 11) * 2 // 3 - w50
    print(
        f"{floor(edges[len(edges)//2-1]):>{w50}}"
        f"{floor(edges[len(edges)*2//3]):>{w66}}"
    )
    print()


y_pred_prob = 1 / (1 + np.exp(-y_pred_oof * 0.25))
y_true = (train["Spread_1"] > 0).astype(int)
brier_score = np.mean((y_pred_prob - y_true) ** 2)
print(f"Brier score: {brier_score:.4f}")


suptitle = "NCAA D1 basketball 2010-2025 men's and women's"
title = "Predicted win probabilities distribution"

if is_notebook:
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred_prob, bins=50, kde=True)
    plt.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="50% threshold")
    plt.xlabel("Predicted Win Probability")
    plt.ylabel("Count")
    plt.suptitle(suptitle)
    plt.title(title)
    plt.legend()
    plt.show()
else:
    print()
    print(suptitle)
    print(title)
    hist, _ = np.histogram(y_pred_prob, bins=50)
    print(asciichartpy.plot(hist, {"height": 10, "width": 80}))
    print()


def to_csv(df, fn):
    df.to_csv(fn, index=False)
    with open(fn, "r+") as f:
        content = f.read()
        f.seek(0)
        f.truncate()
        f.write(content.rstrip("\n"))


print("Preparing submission")
sample_sub = pd.read_csv(f"../input/{input_dir}/SampleSubmissionStage2.csv")

sample_sub[["Season", "Team1", "Team2"]] = (
    sample_sub["ID"].str.split("_", expand=True).astype(int)
)

team_data = season_self_sos[season_self_sos["Season"] == 2025].set_index("TeamID")
X_submit = pd.DataFrame(index=sample_sub.index)

for col in X.columns:
    parts = col.split("_")
    team_num = parts[-1]
    base_col = "_".join(parts[:-1])
    team_id_col = f"Team{team_num}"
    X_submit[col] = sample_sub[team_id_col].map(team_data[base_col])

ensemble_preds = np.mean([model.predict(X_submit) for model in fold_models], axis=0)
win_probs = 1 / (1 + np.exp(-ensemble_preds * 0.25))
sample_sub["Pred"] = win_probs
to_csv(sample_sub[["ID", "Pred"]], "submission.csv")
print("Wrote submission.csv")


print("Generating multiple independent simulation-based submissions...")
base_ids = sample_sub["ID"].copy()
probabilities = sample_sub["Pred"].values
num_matchups = len(probabilities)

for num_sims in range(1, 11):
    random_values = np.random.random((num_sims, num_matchups))
    sim_results = (random_values < probabilities).astype(int)

    submission = pd.DataFrame(
        {"ID": base_ids, "Pred": sim_results.sum(axis=0) / num_sims}
    )

    filename = f"submission{num_sims}.csv"
    to_csv(submission, filename)

    print(
        f"Wrote {filename:>16}, simulation rounds: {num_sims:>2}, win probabilities: {[f'{p:.2f}' for p in sorted(np.unique(submission['Pred']))]}"
    )

print("All independent simulation-based submissions complete!")


m_teams = pd.read_csv(f"../input/{input_dir}/MTeams.csv")
w_teams = pd.read_csv(f"../input/{input_dir}/WTeams.csv")
stats_2025 = season_self_sos[season_self_sos["Season"] == 2025]
w_stats = stats_2025[stats_2025["TeamID"] >= 3000].copy()
m_stats = stats_2025[stats_2025["TeamID"] < 3000].copy()

for df in [w_stats, m_stats]:
    df["poss"] = (
        df["FGA_o_self"] - df["OR_o_self"] + df["TO_o_self"] + 0.44 * df["FTA_o_self"]
    )
    df["off_eff"] = 100 * df["Score_o_self"] / df["poss"]
    df["ast_to"] = df["Ast_o_self"] / df["TO_o_self"]
    df["three_rate"] = 100 * df["FGA3_o_self"] / df["FGA_o_self"]
    df["wtd_diff"] = df["Spread_o_self"] * (
        df["Score_o_sos"] / df["Score_o_sos"].mean()
    )
    df["win_prob"] = 1 / (1 + np.exp(-df["wtd_diff"] * 0.2))
    df["tourney_win_prob"] = df["win_prob"] ** 6

w_stats["tourney_win_prob"] = (
    w_stats["tourney_win_prob"] / w_stats["tourney_win_prob"].sum()
)
m_stats["tourney_win_prob"] = (
    m_stats["tourney_win_prob"] / m_stats["tourney_win_prob"].sum()
)

w_top25 = (
    w_stats.sort_values("tourney_win_prob", ascending=False)
    .head(25)
    .merge(w_teams[["TeamID", "TeamName"]], on="TeamID")
)
m_top25 = (
    m_stats.sort_values("tourney_win_prob", ascending=False)
    .head(25)
    .merge(m_teams[["TeamID", "TeamName"]], on="TeamID")
)


def print_rankings(df, title):
    print(f"\n{title}")
    print(
        f"{'Rank':<4} {'Team':<18} {'Tourney Win':<10} {'PPG':<6} {'Wtd Diff':<8} {'SOS':<6} {'Off Eff':<8} {'Ast/TO':<6}"
    )
    print("-" * 70)

    for i, row in df.reset_index(drop=True).iterrows():
        print(
            f"{i + 1:<4} {row['TeamName']:<18} {row['tourney_win_prob'] * 100:7.2f}%  {row['Score_o_self']:5.1f} {row['wtd_diff']:+7.1f} {row['Score_o_sos']:5.1f} {row['off_eff']:7.1f} {row['ast_to']:5.2f}"
        )


print_rankings(w_top25, "Top 25 Women's Teams (2025)")
print_rankings(m_top25, "Top 25 Men's Teams (2025)")




