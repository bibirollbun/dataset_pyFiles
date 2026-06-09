import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import xgboost as xgb
import statsmodels.api as sm
import tqdm
import warnings
from sklearn.metrics import mean_absolute_error, brier_score_loss
from scipy.interpolate import UnivariateSpline


warnings.filterwarnings("ignore")
data_dir = "../input/march-machine-learning-mania-2025"

# W -> women's basketball, M -> men's basketball
regular_results = pd.concat(pd.read_csv(f"{data_dir}/{mw}RegularSeasonDetailedResults.csv") for mw in "MW")
tourney_results = pd.concat(pd.read_csv(f"{data_dir}/{mw}NCAATourneyDetailedResults.csv") for mw in "MW")
seeds = pd.concat(pd.read_csv(f"{data_dir}/{mw}NCAATourneySeeds.csv") for mw in "MW")
massey = pd.read_csv(f"{data_dir}/MMasseyOrdinals.csv")

season = 2003  # season cut-off
regular_results = regular_results.loc[regular_results["Season"] >= season]
tourney_results = tourney_results.loc[tourney_results["Season"] >= season]
seeds = seeds.loc[seeds["Season"] >= season]
massey = massey.loc[massey["Season"] >= season]


# double the dataset with swapped team positions in box scores
def swap_location(x):
    if x == "H":
        return "A"
    if x == "A":
        return "H"
    return x

def prepare_data(df):
    df.rename({"WLoc": "location"}, axis=1, inplace=True)    
    df = df[["Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", "NumOT", "location",
            "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
            "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]]
    
    # adjustment factor for overtimes, as more stats are accumulated during overtimes
    adjot = (40 + 5 * df["NumOT"]) / 40
    adjcols = ["LScore", "WScore", 
               "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
               "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]
    for col in adjcols:
        df[col] = df[col] / adjot    
    
    dfswap = df.copy()
    dfswap["location"] = dfswap["location"].apply(swap_location)    
    df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in list(df.columns)]
    dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in list(dfswap.columns)]
    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output["PointDiff"] = output["T1_Score"] - output["T2_Score"]
    output["win"] = (output["PointDiff"] > 0) * 1
    output["men_women"] = (output["T1_TeamID"].apply(lambda t: str(t).startswith("1"))) * 1  # 0: women, 1: men
    return output

regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)


# extract seed number from `Seed` field
seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))
seeds_T1 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T2 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]

regular_data = pd.merge(regular_data, seeds_T1, on=["Season", "T1_TeamID"], how="left")
regular_data = pd.merge(regular_data, seeds_T2, on=["Season", "T2_TeamID"], how="left")


# box score columns, for which we want features to our model
boxcols = [
    "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3", "T1_FTM", "T1_FTA",
    "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
    "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3", "T2_FTM", "T2_FTA",
    "T2_OR", "T2_DR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk", "T2_PF",
    "PointDiff",
]

# calculate season averages
ss = regular_data.groupby(["Season", "T1_TeamID"])[boxcols].agg("mean").reset_index()

ss_T1 = ss.copy()
ss_T1.columns = ["T1_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(ss_T1.columns)]
ss_T1 = ss_T1.rename({"T1_avg_Season": "Season", "T1_avg_TeamID": "T1_TeamID"}, axis=1)
ss_T2 = ss.copy()
ss_T2.columns = ["T2_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(ss_T2.columns)]
ss_T2 = ss_T2.rename({"T2_avg_Season": "Season", "T2_avg_TeamID": "T2_TeamID"}, axis=1)



# we use POM, WLK, MOR from Massey as they are most predictive and available since 2003
def get_massey(system_name, daynum):
    x = massey.loc[massey["SystemName"] == system_name].drop("SystemName", axis=1)
    x = x.loc[x["RankingDayNum"] <= daynum]
    mx = x.groupby("Season")["RankingDayNum"].max().reset_index(name="MaxDayNum")
    x = pd.merge(x, mx, on = "Season")
    x = x.loc[x["RankingDayNum"] == x["MaxDayNum"]]
    x.drop(["RankingDayNum", "MaxDayNum"], axis=1, inplace=True)
    x.rename({"OrdinalRank": f"{system_name}{daynum}"}, axis=1, inplace=True)
    return x

pom132 = get_massey("POM", 132) # for some reason DayNum=133 is worse
wlk132 = get_massey("WLK", 132)
mor132 = get_massey("MOR", 132)

qrnk = pd.concat([pom132[["Season", "TeamID"]], 
                  wlk132[["Season", "TeamID"]], 
                  mor132[["Season", "TeamID"]]]).drop_duplicates()
qrnk = pd.merge(qrnk, pom132, on=["Season", "TeamID"], how="left")
qrnk = pd.merge(qrnk, wlk132, on=["Season", "TeamID"], how="left")
qrnk = pd.merge(qrnk, mor132, on=["Season", "TeamID"], how="left")
qrnk["massey"] = qrnk[["POM132", "WLK132", "MOR132"]].mean(axis=1)
qrnk.drop(["POM132", "WLK132", "MOR132"], axis=1, inplace=True)
qrnk_T1 = qrnk.copy().rename({"TeamID": "T1_TeamID", "massey": "T1_massey"}, axis=1)
qrnk_T2 = qrnk.copy().rename({"TeamID": "T2_TeamID", "massey": "T2_massey"}, axis=1)


# change in POM rating month prior

pom100 = get_massey("POM", 100)
pom_ch = pd.merge(pom132, pom100, on = ["Season","TeamID"], how="left")
pom_ch["pom_diff1mo"] = pom_ch["POM132"] - pom_ch["POM100"]
pom_ch.drop(["POM132", "POM100"], axis=1, inplace=True)
pom_ch_T1 = pom_ch.copy().rename({"TeamID": "T1_TeamID", "pom_diff1mo": "T1_pom_diff1mo"}, axis=1)
pom_ch_T2 = pom_ch.copy().rename({"TeamID": "T2_TeamID", "pom_diff1mo": "T2_pom_diff1mo"}, axis=1)


# win rate against better teams, according to massey rank

regular_data = pd.merge(regular_data, qrnk_T1, on=["Season", "T1_TeamID"], how="left")
regular_data = pd.merge(regular_data, qrnk_T2, on=["Season", "T2_TeamID"], how="left")
T1_b_massey = regular_data.loc[(regular_data["T1_massey"] >= regular_data["T2_massey"])]
T1_b_massey = T1_b_massey.groupby(["Season", "T1_TeamID"])["win"].mean().reset_index(name="T1_winpct_massey")
T2_b_massey = T1_b_massey.copy().rename({"T1_TeamID": "T2_TeamID", "T1_winpct_massey": "T2_winpct_massey"}, axis=1)


base_elo = 1000
elo_width = 600
k_factor = 70
p_factor = 12

def update_elo(winner_elo, loser_elo, pointdiff):
    expected = expected_result(winner_elo, loser_elo)
    # we add custom logic for determining result; originally it should be 0.5 for draw and 1 for win
    result = 1 / (1 + np.exp(-pointdiff / p_factor)) 
    elo_change = k_factor * (result - expected)
    return winner_elo + elo_change, loser_elo - elo_change

def expected_result(elo_a, elo_b):
    return 1.0 / (1 + 10 ** ((elo_b - elo_a) / elo_width))

elos = []
for season in sorted(set(seeds["Season"])):
    rs = regular_data.loc[regular_data["Season"] == season]
    for mw in (0, 1):
        ss = rs.loc[rs["men_women"] == mw]
        ss = ss.loc[ss["win"] == 1].reset_index(drop=True)
        if ss.shape[0] > 0:
            teams = set(ss["T1_TeamID"]) | set(ss["T2_TeamID"])
            elo = dict(zip(teams, [base_elo] * len(teams)))
            sd = seeds.loc[seeds["Season"] == season]
            sd = dict(zip(sd["TeamID"], 45 * (17 - sd["seed"])))              
            for k, v in sd.items():
                if k in elo:
                    elo[k] += v
            for _ in range(5): # make 5 runs, base elo from previous run
                for i in range(ss.shape[0]):
                    w_team, l_team = ss.loc[i, "T1_TeamID"], ss.loc[i, "T2_TeamID"]
                    w_elo, l_elo = elo[w_team], elo[l_team]
                    pointdiff = ss.loc[i, "PointDiff"] * (ss.loc[i, "NumOT"] == 0)
                    w_elo_new, l_elo_new = update_elo(w_elo, l_elo, pointdiff)
                    elo[w_team] = w_elo_new
                    elo[l_team] = l_elo_new
            elo = pd.DataFrame.from_dict(elo, orient="index").reset_index()
            elo = elo.rename({"index": "TeamID", 0: "elo"}, axis=1)
            elo["Season"] = season
            elos.append(elo)
elos = pd.concat(elos)

elos_T1 = elos.copy().rename({"TeamID": "T1_TeamID", "elo": "T1_elo"}, axis=1)
elos_T2 = elos.copy().rename({"TeamID": "T2_TeamID", "elo": "T2_elo"}, axis=1)


# win rate against better teams, according to ELO

regular_data = pd.merge(regular_data, elos_T1, on=["Season", "T1_TeamID"], how="left")
regular_data = pd.merge(regular_data, elos_T2, on=["Season", "T2_TeamID"], how="left")
T1_b_elo = regular_data.loc[(regular_data["T1_elo"] <= regular_data["T2_elo"])]
T1_b_elo = T1_b_elo.groupby(["Season", "T1_TeamID"])["win"].mean().reset_index(name="T1_winpct_elo")
T2_b_elo = T1_b_elo.copy().rename({"T1_TeamID": "T2_TeamID", "T1_winpct_elo": "T2_winpct_elo"}, axis=1)


regular_data["ST1"] = regular_data.apply(lambda t: str(int(t["Season"])) + "/" + str(int(t["T1_TeamID"])), axis=1)
regular_data["ST2"] = regular_data.apply(lambda t: str(int(t["Season"])) + "/" + str(int(t["T2_TeamID"])), axis=1)
seeds_T1["ST1"] = seeds_T1.apply(lambda t: str(int(t["Season"])) + "/" + str(int(t["T1_TeamID"])), axis=1)
seeds_T2["ST2"] = seeds_T2.apply(lambda t: str(int(t["Season"])) + "/" + str(int(t["T2_TeamID"])), axis=1)

# collect tourney teams
st = set(seeds_T1["ST1"]) | set(seeds_T2["ST2"])
# append non-tourney teams which were able to beat tourney team at least once
st = st | set(regular_data.loc[(regular_data["T1_Score"] > regular_data["T2_Score"]) & 
                               (regular_data["ST2"].isin(st)), "ST1"])
# append non-tourney teams which had X%+ winrate during regular season
win_count = regular_data.groupby(["ST1"])["win"].mean().reset_index(name="winrate")
st = st | set(win_count.loc[win_count["winrate"] > 0.20, "ST1"])

def team_quality(season, men_women):
    # mixed effects: fixed intercept=0, random slope
    formula = "PointDiff~-1+T1_TeamID+T2_TeamID+location" # +location to adjust for home advantage
    data = dt.loc[(dt["Season"] == season) & (dt["men_women"] == men_women), :]
    w = np.array(1 / (data["T1_seed"].fillna(25) + data["T2_seed"].fillna(25))) # more weights for higher seeded matchups
    glm = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Gaussian(), freq_weights=w).fit()
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ["TeamID", "quality"]
    quality["Season"] = season
    quality = quality.loc[quality["TeamID"].str.contains("T1_")].reset_index(drop=True)
    quality["TeamID"] = quality["TeamID"].apply(lambda x: x[10:14]).astype(int)
    return quality

glm_quality = []

dt = regular_data.copy().loc[regular_data["ST1"].isin(st) | regular_data["ST2"].isin(st)]
dt["T1_TeamID"] = dt["T1_TeamID"].astype(str)
dt["T2_TeamID"] = dt["T2_TeamID"].astype(str)
dt.loc[~dt["ST1"].isin(st), "T1_TeamID"] = "0000"
dt.loc[~dt["ST2"].isin(st), "T2_TeamID"] = "0000"
seasons = sorted(set(seeds["Season"]))
for s in tqdm.tqdm(seasons, unit="season"):
    if s >= 2010:  # min season for women
        glm_quality.append(team_quality(s, 0))
    if s >= 2003:  # min season for men
        glm_quality.append(team_quality(s, 1))

glm_quality = pd.concat(glm_quality).reset_index(drop=True)
glm_quality_T1 = glm_quality.copy().rename({"TeamID": "T1_TeamID", "quality": "T1_quality"}, axis=1)
glm_quality_T2 = glm_quality.copy().rename({"TeamID": "T2_TeamID", "quality": "T2_quality"}, axis=1)


# win rate against better teams, according to quality

regular_data = pd.merge(regular_data, glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
regular_data = pd.merge(regular_data, glm_quality_T2, on=["Season", "T2_TeamID"], how="left")
T1_b_quality = regular_data.loc[(regular_data["T1_quality"] <= regular_data["T2_quality"])]
T1_b_quality = T1_b_quality.groupby(["Season", "T1_TeamID"])["win"].mean().reset_index(name="T1_winpct_quality")
T2_b_quality = T1_b_quality.copy().rename({"T1_TeamID": "T2_TeamID", "T1_winpct_quality": "T2_winpct_quality"}, axis=1)


# average opponent quality

opp_quality1 = regular_data.groupby(["Season", "T1_TeamID"])["T2_quality"].agg("mean").reset_index()
opp_quality2 = regular_data.groupby(["Season", "T2_TeamID"])["T1_quality"].agg("mean").reset_index()
opp_quality1.rename({"T2_quality": "T1_opp_avg_quality"}, axis=1, inplace=True)
opp_quality2.rename({"T1_quality": "T2_opp_avg_quality"}, axis=1, inplace=True)

# max opponent quality

opp_quality1x = regular_data.groupby(["Season", "T1_TeamID"])["T2_quality"].agg("max").reset_index()
opp_quality2x = regular_data.groupby(["Season", "T2_TeamID"])["T1_quality"].agg("max").reset_index()
opp_quality1x.rename({"T2_quality": "T1_opp_max_quality"}, axis=1, inplace=True)
opp_quality2x.rename({"T1_quality": "T2_opp_max_quality"}, axis=1, inplace=True)


# collect features

tourney_data = tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]]

tourney_data = pd.merge(tourney_data, seeds_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, seeds_T2, on=["Season", "T2_TeamID"], how="left")

tourney_data = pd.merge(tourney_data, ss_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, ss_T2, on=["Season", "T2_TeamID"], how="left")

tourney_data = pd.merge(tourney_data, qrnk_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, qrnk_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, T1_b_massey, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, T2_b_massey, on=["Season", "T2_TeamID"], how="left")

tourney_data = pd.merge(tourney_data, pom_ch_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, pom_ch_T2, on=["Season", "T2_TeamID"], how="left")

tourney_data = pd.merge(tourney_data, elos_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, elos_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, T1_b_elo, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, T2_b_elo, on=["Season", "T2_TeamID"], how="left")

tourney_data = pd.merge(tourney_data, glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, glm_quality_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, T1_b_quality, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, T2_b_quality, on=["Season", "T2_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, opp_quality1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, opp_quality2, on=["Season", "T2_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, opp_quality1x, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, opp_quality2x, on=["Season", "T2_TeamID"], how="left")

tourney_data["seed_diff"] = tourney_data["T1_seed"] - tourney_data["T2_seed"]
tourney_data["massey_diff"] = tourney_data["T1_massey"] - tourney_data["T2_massey"]
tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]
tourney_data["quality_diff"] = tourney_data["T1_quality"] - tourney_data["T2_quality"]

features = [
    "men_women",  
    "T1_seed",
    "T2_seed",
    "seed_diff",
    "T1_avg_Score",
    "T1_avg_Blk",
    "T1_avg_PF",
    "T1_avg_opponent_FGA",
    "T1_avg_opponent_Blk",
    "T1_avg_opponent_PF",
    "T2_avg_Score",
    "T2_avg_Blk",
    "T2_avg_PF",
    "T2_avg_opponent_FGA",
    "T2_avg_opponent_Blk",
    "T2_avg_opponent_PF",
    "T1_elo",
    "T2_elo",    
    "elo_diff",
    "T1_winpct_elo",
    "T2_winpct_elo",
    "T1_quality",
    "T2_quality",
    "quality_diff",   
    "T1_winpct_quality",
    "T2_winpct_quality",
    "T1_massey",
    "T2_massey",  
    "massey_diff",
    "T1_pom_diff1mo",
    "T2_pom_diff1mo",
    "T1_winpct_massey",
    "T2_winpct_massey",   
    "T1_opp_avg_quality",
    "T2_opp_avg_quality",   
    "T1_opp_max_quality",
    "T2_opp_max_quality",            
]

print(f"Number of features {len(features)}")


param = {}
param["objective"] = "reg:squarederror"
param["booster"] = "gbtree"
param["eta"] = 0.01
param["subsample"] = 0.6
param["colsample_bynode"] = 0.8
param["num_parallel_tree"] = 2
param["min_child_weight"] = 64
param["max_depth"] = 4
param["tree_method"] = "hist"
param["max_bin"] = 36

num_rounds = 600

models = {}
oof_mae = []
oof_preds = []
oof_targets = []
oof_ss = []

# leave-one-season out models
tdata = tourney_data.copy()
tdata = tdata.loc[tdata['Season']>=2003]

for oof_season in sorted(set(tdata.Season)):
    x_train = tdata.loc[tdata["Season"] != oof_season, features].values
    y_train = tdata.loc[tdata["Season"] != oof_season, "PointDiff"].values
    x_val = tdata.loc[tdata["Season"] == oof_season, features].values
    y_val = tdata.loc[tdata["Season"] == oof_season, "PointDiff"].values
    s_val = tdata.loc[tdata["Season"] == oof_season, "Season"].values
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    models[oof_season] = xgb.train(
        params=param,
        dtrain=dtrain,
        num_boost_round = num_rounds,        
    )
    preds = models[oof_season].predict(dval)
    print(f"oof season {oof_season} mae: {mean_absolute_error(y_val, preds)}")
    oof_mae.append(mean_absolute_error(y_val, preds))
    oof_preds += list(preds)
    oof_targets += list(y_val)
    oof_ss += list(s_val)
    
print(f"average mae: {np.mean(oof_mae)}")


df = pd.DataFrame(
    {"Season": oof_ss, "pred": oof_preds, "label": [(t > 0) * 1 for t in oof_targets], "men_women": tdata["men_women"]}
)
df["pred_pointdiff"] = df["pred"].astype(int)

xdf_all = df.clip(-40, 40).groupby("pred_pointdiff")["label"].mean().reset_index(name="average_win_pct")
xdf_men = df.clip(-40, 40).loc[df["men_women"] == 0].groupby("pred_pointdiff")["label"].mean().reset_index(name="average_win_pct")
xdf_women = df.clip(-40, 40).loc[df["men_women"] == 1].groupby("pred_pointdiff")["label"].mean().reset_index(name="average_win_pct")

seaborn.lineplot(x=xdf_all["pred_pointdiff"], y=xdf_all["average_win_pct"])
seaborn.lineplot(x=xdf_men["pred_pointdiff"], y=xdf_men["average_win_pct"])
seaborn.lineplot(x=xdf_women["pred_pointdiff"], y=xdf_women["average_win_pct"])


t = 25
dat = list(zip(oof_preds, np.array(oof_targets)>0))
dat = sorted(dat, key = lambda x: x[0])
pred, label = list(zip(*dat))
spline_model = UnivariateSpline(np.clip(pred, -t, t), label, k=5)
spline_fit = np.clip(spline_model(np.clip(oof_preds, -t, t)), 0.01, 0.99)
df["spline"] = spline_fit
xdf = df.clip(-40,40).groupby('pred_pointdiff')[['spline','label']].mean().reset_index()

plt.figure()
plt.plot(xdf['pred_pointdiff'],xdf['label'])
plt.plot(xdf['pred_pointdiff'],xdf['spline'])


print(f"brier: {brier_score_loss(np.array(oof_targets)>0, spline_fit)}")

for oof_season in sorted(set(tdata.Season)):
    x = df.loc[df["Season"] == oof_season, "spline"].values
    y = df.loc[df["Season"] == oof_season, "label"].values
    print(oof_season, np.round(brier_score_loss(y, x),5))


X = pd.read_csv(f"{data_dir}/SampleSubmissionStage2.csv")

X["Season"] = X["ID"].apply(lambda t: int(t.split("_")[0]))
X["T1_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[1]))
X["T2_TeamID"] = X["ID"].apply(lambda t: int(t.split("_")[2]))
X["men_women"] = X["T1_TeamID"].apply(lambda t: 0 if str(t)[0] == "1" else 1)

X = pd.merge(X, seeds_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, seeds_T2, on=["Season", "T2_TeamID"], how="left")

X = pd.merge(X, ss_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, ss_T2, on=["Season", "T2_TeamID"], how="left")

X = pd.merge(X, qrnk_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, qrnk_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, T1_b_massey, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, T2_b_massey, on=["Season", "T2_TeamID"], how="left")

X = pd.merge(X, pom_ch_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, pom_ch_T2, on=["Season", "T2_TeamID"], how="left")

X = pd.merge(X, elos_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, elos_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, T1_b_elo, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, T2_b_elo, on=["Season", "T2_TeamID"], how="left")

X = pd.merge(X, glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, glm_quality_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, T1_b_quality, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, T2_b_quality, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, opp_quality1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, opp_quality2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, opp_quality1x, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, opp_quality2x, on=["Season", "T2_TeamID"], how="left")

X["seed_diff"] = X["T1_seed"] - X["T2_seed"]
X["massey_diff"] = X["T1_massey"] - X["T2_massey"]
X["elo_diff"] = X["T1_elo"] - X["T2_elo"]
X["quality_diff"] = X["T1_quality"] - X["T2_quality"]


# run models on given dataset
preds = []
for oof_season in set(tourney_data.Season):
    dtest = xgb.DMatrix(X[features].values)
    margin_preds = models[oof_season].predict(dtest) * 1.0 # aggressive submissions >1, conservative submissions <1
    probs = np.clip(spline_model(np.clip(margin_preds, -t, t)), 0.01, 0.99)
    preds.append(probs)
X['Pred'] = np.array(preds).mean(axis=0) 


# sanity check to check for seed matchup win probability distribution
Z = X.copy()
Z["T1_seed"] = 17 - Z["T1_seed"]
Z["T2_seed"] = 17 - Z["T2_seed"]
Z["Pred"] = 1 - Z["Pred"]
pd.pivot_table(data=pd.concat([X, Z]), index="T1_seed", columns="T2_seed", values="Pred", aggfunc="mean").style.bar(
    color="#5fba7d", vmin=0, vmax=1
)


X.loc[X["ID"]=="2025_1161_1272", "Pred"] = 0
X[["ID", "Pred"]].to_csv("predictions-1.csv", index=None)

X.loc[X["ID"]=="2025_1161_1272", "Pred"] = 1
X[["ID", "Pred"]].to_csv("predictions-2.csv", index=None)

