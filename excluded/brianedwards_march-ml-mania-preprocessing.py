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

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if is_notebook:
    import matplotlib.pyplot as plt
    import seaborn as sns
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    plt.rcParams["figure.figsize"] = (10, 2)
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})


data_dir = f"../datasets/march-machine-learning-mania-2025"
if not os.path.isdir(data_dir):
    data_dir = f"../input/march-machine-learning-mania-2025"


def read_detailed_results(fn):
    df = pd.read_csv(f"{data_dir}/{fn}.csv")
    for c in df.select_dtypes("int64"):
        df[c] = df[c].astype("int32")
    df = df.rename(columns={"Season": "WSeason"})
    df["LSeason"] = df["WSeason"]
    df = df.drop(columns=["NumOT", "WLoc"])
    assert all(df[c].dtype == "int32" for c in df)
    print(f"{fn:<29} {df.shape[0]:>7,} {df.shape[1]:>2} {df['WSeason'].min()} {df['WSeason'].max()}")
    return df


MNCAATourneyDetailedResults = read_detailed_results("MNCAATourneyDetailedResults")
MRegularSeasonDetailedResults = read_detailed_results("MRegularSeasonDetailedResults")
WNCAATourneyDetailedResults = read_detailed_results("WNCAATourneyDetailedResults")
WRegularSeasonDetailedResults = read_detailed_results("WRegularSeasonDetailedResults")

detailed_results = pd.concat(
    [
        MNCAATourneyDetailedResults,
        MRegularSeasonDetailedResults,
        WNCAATourneyDetailedResults,
        WRegularSeasonDetailedResults,
    ]
).reset_index(drop=True)

print(f"{'-'*50}\n{'detailed_results':<29} {detailed_results.shape[0]:>7,} {detailed_results.shape[1]:>2} {detailed_results['WSeason'].min()} {detailed_results['WSeason'].max()}")
assert 1 == len([c for c in detailed_results if c[0] not in ("W", "L")])
print(f"\n{[c[1:] for c in detailed_results if c[0] == 'W']}\n")

if is_notebook:
    sns.histplot(detailed_results["WSeason"])
    plt.title(
        "Detailed results (before 2010: men only, 2021: COVID, 2025: regular season only)"
    )
    plt.xlabel("Season")
    plt.ylabel("Games");


margin = detailed_results[
    ["WSeason", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"]
]
margin = margin.rename(columns={"WSeason": "Season"})
mask = margin["WTeamID"] < margin["LTeamID"]

margin.loc[mask, ["TeamID_1", "TeamID_2", "Margin"]] = np.column_stack(
    [
        margin.loc[mask, "WTeamID"].values,
        margin.loc[mask, "LTeamID"].values,
        (margin.loc[mask, "WScore"] - margin.loc[mask, "LScore"]).values,
    ]
)

margin.loc[~mask, ["TeamID_1", "TeamID_2", "Margin"]] = np.column_stack(
    [
        margin.loc[~mask, "LTeamID"].values,
        margin.loc[~mask, "WTeamID"].values,
        (margin.loc[~mask, "LScore"] - margin.loc[~mask, "WScore"]).values,
    ]
)

margin["TeamID_1"] = margin["TeamID_1"].astype("int32")
margin["TeamID_2"] = margin["TeamID_2"].astype("int32")
margin["Margin"] = margin["Margin"].astype("float32")
margin = margin.drop(columns=["WTeamID", "LTeamID", "WScore", "LScore"])

p(margin)
margin.info()
print()

if is_notebook:
    sns.lineplot(
        margin.groupby("Season")["Margin"].mean().reset_index(),
        x="Season",
        y="Margin",
    )
    plt.title(
        "Sanity check, average point margin should be near zero for arbitrary team perspective (lower TeamID)"
    );


def from_WL_to_od(from_prefix, to_suffix):
    df = detailed_results[[c for c in detailed_results if c[0] == from_prefix]]
    return df.rename(columns={c: f"{c[1:]}_{to_suffix}" for c in df})


def calculate_possessions(df, prefix="", suffix=""):
    df["Poss_o"] = (
        df[f"{prefix}FGA_o{suffix}"]
        - df[f"{prefix}OR_o{suffix}"]
        + df[f"{prefix}TO_o{suffix}"]
        + 0.44 * df[f"{prefix}FTA_o{suffix}"]
    ).astype("float32")
    df["Poss_d"] = (
        df[f"{prefix}FGA_d{suffix}"]
        - df[f"{prefix}OR_d{suffix}"]
        + df[f"{prefix}TO_d{suffix}"]
        + 0.44 * df[f"{prefix}FTA_d{suffix}"]
    ).astype("float32")
    return df


game_team = pd.concat(
    [
        pd.concat(
            [
                from_WL_to_od("W", "o"),
                from_WL_to_od("L", "d"),
                detailed_results["LTeamID"].rename("Opponent"),
            ],
            axis=1,
        ),
        pd.concat(
            [
                from_WL_to_od("L", "o"),
                from_WL_to_od("W", "d"),
                detailed_results["WTeamID"].rename("Opponent"),
            ],
            axis=1,
        ),
    ]
).reset_index(drop=True)

game_team = game_team.rename(columns={"Season_o": "Season", "TeamID_o": "TeamID"})
game_team = game_team.drop(columns=["Season_d", "TeamID_d"])
game_team = calculate_possessions(game_team)

game_team = game_team[
    ["Season", "TeamID", "Opponent"]
    + [c for c in game_team if c[-2:] == "_o"]
    + [c for c in game_team if c[-2:] == "_d"]
]

p(game_team)

if is_notebook:
    sns.barplot(
        game_team.groupby("Season")["TeamID"].nunique().reset_index(name="Teams"),
        x="Season",
        y="Teams",
        color=sns.color_palette()[0],
    )

    plt.title("Sanity check, number of teams per year (before 2010: men only)")
    plt.xticks(rotation=90)
    plt.figure()

    sns.barplot(
        game_team.groupby(["Season", "TeamID"])["Score_o"]
        .count()
        .reset_index()
        .groupby("Season")["Score_o"]
        .median()
        .reset_index(name="Median games per team"),
        x="Season",
        y="Median games per team",
        color=sns.color_palette()[0],
    )

    plt.title("Sanity check, games per team (upper 20s to 30, 2021: COVID, 2025: regular season only)")
    plt.xticks(rotation=90);


season_sum = (
    game_team.drop(columns="Opponent")
    .groupby(["Season", "TeamID"])
    .agg(
        dict(
            Score_o=["count", "sum"],
            **{c: "sum" for c in game_team if c[-2] == "_" and c != "Score_o"}
        )
    )
)

season_sum.columns = season_sum.columns.map("_".join)
season_sum = season_sum.reset_index()
season_sum = season_sum.rename(columns={"Score_o_count": "Games"})
p(season_sum)


def from_sum_to_pct(sum_df, prefix=""):
    pct_df = sum_df.copy()

    for c in pct_df:
        if c.endswith("_sum"):
            pct_df[f"{c[:-6]}_pg_{c[-5:-4]}"] = (pct_df[c] / pct_df["Games"]).astype(
                "float32"
            )
            if "_o_" in c:
                pct_df[f"{c[:-6]}_poss_{c[-5:-4]}"] = (
                    pct_df[c] / pct_df[f"{prefix}Poss_o_sum"]
                ).astype("float32")
            if "_d_" in c:
                pct_df[f"{c[:-6]}_poss_{c[-5:-4]}"] = (
                    pct_df[c] / pct_df[f"{prefix}Poss_d_sum"]
                ).astype("float32")

    for side in ["o", "d"]:
        pct_df[f"{prefix}FGPct_{side}"] = (
            pct_df[f"{prefix}FGM_{side}_sum"] / pct_df[f"{prefix}FGA_{side}_sum"]
        ).astype("float32")
        pct_df[f"{prefix}FGPct3_{side}"] = (
            pct_df[f"{prefix}FGM3_{side}_sum"] / pct_df[f"{prefix}FGA3_{side}_sum"]
        ).astype("float32")
        pct_df[f"{prefix}FTPct_{side}"] = (
            pct_df[f"{prefix}FTM_{side}_sum"] / pct_df[f"{prefix}FTA_{side}_sum"]
        ).astype("float32")
        pct_df[f"{prefix}AstPct_{side}"] = (
            pct_df[f"{prefix}Ast_{side}_sum"] / pct_df[f"{prefix}FGM_{side}_sum"]
        ).astype("float32")
        pct_df[f"{prefix}AstTO_{side}"] = (
            pct_df[f"{prefix}Ast_{side}_sum"] / pct_df[f"{prefix}TO_{side}_sum"]
        ).astype("float32")

    pct_df = pct_df.drop(columns=[c for c in pct_df if c.endswith("_sum")])
    pct_df = pct_df.drop(
        columns=["Games", f"{prefix}Poss_poss_o", f"{prefix}Poss_poss_d"]
    )

    for c in pct_df:
        if c.endswith("_o"):
            pct_df[f"{c[:-2]}_diff"] = pct_df[c] - pct_df[f"{c[:-2]}_d"]

    return pct_df


season_team = from_sum_to_pct(season_sum)
p(season_team)
season_team.info()
print(flush=True)

if is_notebook:
    sns.lineplot(
        season_team[season_team["TeamID"] == 1196],
        x="Season",
        y="Score_pg_o",
        label="Points per game",
    )
    sns.lineplot(
        season_team[season_team["TeamID"] == 1196],
        x="Season",
        y="Score_pg_d",
        label="Opponents' ppg",
    )
    plt.figure()
    sns.lineplot(
        season_team[season_team["TeamID"] == 1196],
        x="Season",
        y="Score_poss_o",
        label="Points per poss",
    )
    sns.lineplot(
        season_team[season_team["TeamID"] == 1196],
        x="Season",
        y="Score_poss_d",
        label="Opponents' ppp",
    );


game_sos = pd.merge(
    game_team[["Season", "TeamID", "Opponent"]],
    season_sum,
    left_on=["Season", "Opponent"],
    right_on=["Season", "TeamID"],
    suffixes=["", "_"],
)

game_sos = game_sos.drop(columns=["TeamID_"])
game_sos = game_sos.rename(
    columns={c: f"sos_{c}" for c in game_sos if c[-4:] == "_sum"}
)

print("The season stats (simple sums and possessions) for each opponent of TeamID (multiple rows for multiple matchups throughout the season)")
p(game_sos)

if is_notebook:
    df = (
        game_sos.groupby(["Season", "TeamID", "Opponent"])["Games"]
        .count()
        .reset_index()
    )
    sns.barplot(
        df.groupby("Season")["Games"].max().reset_index(),
        x="Season",
        y="Games",
        color=sns.color_palette()[0],
    )
    plt.title("Sanity check, maximum occurences of same matchup within a season")
    plt.xticks(rotation=90)
    plt.show()
    df = df[df["Games"] == 6]
    df = df[df["TeamID"] < df["Opponent"]].reset_index(drop=True)
    mteams = pd.read_csv(f"{data_dir}/MTeams.csv")
    df["Team_1"] = pd.merge(df, mteams, left_on="TeamID", right_on="TeamID")["TeamName"]
    df["Team_2"] = pd.merge(df, mteams, left_on="Opponent", right_on="TeamID")[
        "TeamName"
    ]
    p(df[["Season", "Team_1", "Team_2", "Games"]])


sum_sos = (
    game_sos.drop(columns="Opponent")
    .groupby(["Season", "TeamID"], as_index=False)
    .sum()
)
p(sum_sos[(sum_sos["Season"] > 2021) & (sum_sos["TeamID"] == 1196)])


season_sos = from_sum_to_pct(sum_sos, prefix="sos_")
p(season_sos[(sum_sos["Season"] > 2021) & (season_sos["TeamID"] == 1196)])


season = pd.merge(season_team, season_sos, on=["Season", "TeamID"])
p(season)


train = pd.merge(
    margin, season, left_on=["Season", "TeamID_1"], right_on=["Season", "TeamID"]
)
train = train.drop(columns="TeamID")
train = train.rename(
    columns={c: f"{c}_1" for c in train if c.split("_")[-1] in ("o", "d", "diff")}
)

train = pd.merge(
    train, season, left_on=["Season", "TeamID_2"], right_on=["Season", "TeamID"]
)
train = train.drop(columns="TeamID")
train = train.rename(
    columns={c: f"{c}_2" for c in train if c.split("_")[-1] in ("o", "d", "diff")}
)

train = train.sort_values(["Season", "DayNum", "TeamID_1", "TeamID_2"]).reset_index(
    drop=True
)

train.to_csv("train.csv", index=False)

# p(train)
train.info()
print()
print(train.select_dtypes("float64").columns, flush=True)


X_df = train.drop(["Season", "DayNum", "TeamID_1", "TeamID_2", "Margin"], axis=1)

print(X_df.columns)
print()
print(sorted(set(train.columns) - set(X_df.columns)))
print()

X = X_df.values
X = StandardScaler().fit_transform(X)

# keep y_scaler around to do inverse_transform
# before converting to win probability and Brier scoring
y_scaler = StandardScaler()
y = train["Margin"].values.reshape(-1, 1)
y = y_scaler.fit_transform(y)

print(f"{'detailed_results':<16} {detailed_results.shape[0]:>7,} {detailed_results.shape[1]:>3} {detailed_results['WSeason'].min()} {detailed_results['WSeason'].max()}")

print(f"{'train':<16} {train.shape[0]:>7,} {train.shape[1]:>3} {train['Season'].min()} {train['Season'].max()}")

print(f"{'X':<16} {X.shape[0]:>7,} {X.shape[1]:>3}")
print(f"{'y':<16} {y.shape[0]:>7,} {y.shape[1]:>3}", flush=True)

