import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import gc 
import warnings  
from collections import defaultdict
import seaborn as sns
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score ,brier_score_loss,mean_absolute_error
from sklearn.svm import SVC,SVR,NuSVR,NuSVC    
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsRegressor     
import xgboost as xgb 
import catboost as cb 


warnings.filterwarnings('ignore')    
pd.set_option('display.max_column',999)   


data_dir = "../input/march-machine-learning-mania-2025"

# W -> women's basketball, M -> men's basketball
M_regular_results = pd.read_csv(f"{data_dir}/MRegularSeasonDetailedResults.csv")
M_tourney_results = pd.read_csv(f"{data_dir}/MNCAATourneyDetailedResults.csv")
M_seeds = pd.read_csv(f"{data_dir}/MNCAATourneySeeds.csv")

W_regular_results = pd.read_csv(f"{data_dir}/WRegularSeasonDetailedResults.csv")
W_tourney_results = pd.read_csv(f"{data_dir}/WNCAATourneyDetailedResults.csv")
W_seeds = pd.read_csv(f"{data_dir}/WNCAATourneySeeds.csv")


regular_results = pd.concat([M_regular_results, W_regular_results])
tourney_results = pd.concat([M_tourney_results, W_tourney_results])
seeds = pd.concat([M_seeds, W_seeds])


season = 2003  
regular_results = regular_results.loc[regular_results["Season"] >= season]
tourney_results = tourney_results.loc[tourney_results["Season"] >= season]
seeds = seeds.loc[seeds["Season"] >= season]


regular_results


wloc = {'H':1, 'A':-1, 'N': np.nan} 
regular_results['WHome'] = regular_results['WLoc'].map(lambda x: wloc[x])
regular_results['WHome'].value_counts()


tourney_results['WHome'] = tourney_results['WLoc'].map(lambda x: wloc[x])
tourney_results['WHome'].value_counts()


tourney_results


season = 2024
teamid = 3163

r = regular_results.loc[
    (regular_results["Season"] == season)
    & ((regular_results["WTeamID"] == teamid) | (regular_results["LTeamID"] == teamid))
]
t = tourney_results.loc[
    (tourney_results["Season"] == season)
    & ((tourney_results["WTeamID"] == teamid) | (tourney_results["LTeamID"] == teamid))
]
r["win"] = np.where(r["WTeamID"] == teamid, "win", "lose")
t["win"] = np.where(t["WTeamID"] == teamid, "win", "lose")
r["type"] = "regular season"
t["type"] = "tournament"

rt = pd.concat([r, t])
rt[["DayNum", "WScore", "LScore", "type", "win"]]


seeds


s = W_seeds.loc[W_seeds["Season"] == 2015]
[s.loc[s["Seed"].str.startswith(d)] for d in ("X", "Y", "Z", "W")]


seeds.loc[(seeds["Season"] == season) & (seeds["TeamID"] == teamid)]


def prepare_data(df):
    df = df[["Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", "NumOT","WHome",
            "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
            "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]]
    
    adjot = (40 + 5 * df["NumOT"]) / 40
    adjcols = ["LScore", "WScore", 
               "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA", "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
               "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA", "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF"]
    for col in adjcols:
        df[col] = df[col] / adjot    
    
    dfswap = df.copy()
    df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in list(df.columns)]
    dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in list(dfswap.columns)]
    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output["PointDiff"] = output["T1_Score"] - output["T2_Score"]
    output["win"] = (output["PointDiff"] > 0) * 1
    output["men_women"] = (output["T1_TeamID"].apply(lambda t: str(t).startswith("1"))) * 1 
    return output

regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)


regular_data


tourney_data



season = regular_data["Season"] == 2025
t1, t2 = 1182, 1433
match1 = (regular_data["T1_TeamID"] == t1) & (regular_data["T2_TeamID"] == t2)
match2 = (regular_data["T1_TeamID"] == t2) & (regular_data["T2_TeamID"] == t1)
regular_data.loc[season & (match1 | match2)]


seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))
seeds


seeds_T1 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T2 = seeds[["Season", "TeamID", "seed"]].copy()
seeds_T1.columns = ["Season", "T1_TeamID", "T1_seed"]
seeds_T2.columns = ["Season", "T2_TeamID", "T2_seed"]

tourney_data = tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]]
tourney_data = pd.merge(tourney_data, seeds_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, seeds_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]

tourney_data


tmpmean = tourney_data.pivot_table(columns="men_women", index="T1_seed", values="PointDiff", aggfunc="mean").ffill()
tmpstd = tourney_data.pivot_table(columns="men_women", index="T1_seed", values="PointDiff", aggfunc="std").ffill()
fig, axis = plt.subplots(ncols=2, figsize=(12, 4))
(line_1,) = axis[0].plot(tmpmean.index, tmpmean[0], "b-")
fill_1 = axis[0].fill_between(tmpmean.index, tmpmean[0] - tmpstd[0], tmpmean[0] + tmpstd[0], color="b", alpha=0.1)
(line_2,) = axis[1].plot(tmpmean.index, tmpmean[1], "r--")
fill_2 = axis[1].fill_between(tmpmean.index, tmpmean[1] - tmpstd[1], tmpmean[1] + tmpstd[1], color="r", alpha=0.1)
plt.margins(x=0)
plt.legend([(line_1, fill_1), (line_2, fill_2)], ["Women", "Men"])


tmpmean = tourney_data.pivot_table(columns="men_women", index="Seed_diff", values="PointDiff", aggfunc="mean").ffill()
tmpstd = tourney_data.pivot_table(columns="men_women", index="Seed_diff", values="PointDiff", aggfunc="std").ffill()
fig, axis = plt.subplots(ncols=2, figsize=(12, 4))
(line_1,) = axis[0].plot(tmpmean.index, tmpmean[0], "b-")
fill_1 = axis[0].fill_between(tmpmean.index, tmpmean[0] - tmpstd[0], tmpmean[0] + tmpstd[0], color="b", alpha=0.1)
(line_2,) = axis[1].plot(tmpmean.index, tmpmean[1], "r--")
fill_2 = axis[1].fill_between(tmpmean.index, tmpmean[1] - tmpstd[1], tmpmean[1] + tmpstd[1], color="r", alpha=0.1)
plt.margins(x=0)
plt.legend([(line_1, fill_1), (line_2, fill_2)], ["Women", "Men"])


boxcols = [
        "T1_Home","T2_Home",
    "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3", "T1_FTM", "T1_FTA",
    "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
    "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3", "T2_FTM", "T2_FTA",
    "T2_OR", "T2_DR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk", "T2_PF",
    "PointDiff"
]
def summarize_cols(x):
    return pd.Series({
                      'WinRatio14d': x['win14days'].sum() /  (1 + x['last14days'].sum()),
                     })

regular_data['win14days'] = (regular_data['DayNum'] > 118) & (regular_data['T1_Score'] > regular_data['T2_Score'])
regular_data['win14days'] = regular_data['win14days'].map(int)
regular_data['last14days'] = (regular_data['DayNum'] > 118).map(int)

ss_new = regular_data.groupby(['Season','T1_TeamID']).apply(summarize_cols, include_groups=False).reset_index()
ss_new['season_team'] = ss_new['Season'].map(str) + '_' + ss_new['T1_TeamID'].map(str)
ratio_dict = dict(zip(ss_new['season_team'].tolist(),ss_new['WinRatio14d'].tolist()))

s = regular_data['Season'].tolist()
t1 = regular_data['T1_TeamID'].tolist()
t2 = regular_data['T2_TeamID'].tolist()
s1 = regular_data['T1_Score'].tolist()
s2 = regular_data['T2_Score'].tolist()
d = defaultdict(lambda: [1,1])
for i in range(len(s)):
    tup = str(s[i])+'_'+str(t1[i])+'_'+str(t2[i])
    if s1[i] > s2[i]:
        d[tup][0] += 1
    else:
        d[tup][1] += 1

match_dict = defaultdict(lambda: 0.5)
for k,v in d.items():
    match_dict[k] = v[0]/v[1]

l1 = regular_data['T1_TeamID'].tolist()
l2 = regular_data['T2_TeamID'].tolist()
s0 = regular_data['Season'].tolist()
s1 = regular_data['T1_Score'].tolist()
s2 = regular_data['T2_Score'].tolist()
h1 = regular_data['T1_Home'].tolist()
h2 = regular_data['T2_Home'].tolist()

home_dict = defaultdict(lambda: [0,0]) 
for i in range(len(l1)):
    k = str(s0[i])+'_'+str(l1[i])
    if s1[i]>s2[i] and h1[i] == -1:
        home_dict[k][0]+= 1 
    elif s1[i]<s2[i] and h1[i] == -1:
        home_dict[k][1] += 1 

for i in range(len(l2)):
    k = str(s0[i])+'_'+str(l2[i])
    if s1[i]<s2[i] and h2[i] == -1:
        home_dict[k][0]+= 1 
    elif s1[i]>s2[i] and h2[i] == -1 :
        home_dict[k][1] += 1 

away_dict = {}
for k,v in home_dict.items():
    away_dict[k] = v[0]/(v[0]+v[1])


l1 = regular_data['T1_TeamID'].tolist()
l2 = regular_data['T2_TeamID'].tolist()
s0 = regular_data['Season'].tolist()
s1 = regular_data['T1_Score'].tolist()
s2 = regular_data['T2_Score'].tolist()

l_dict = defaultdict(list)
for i in range(len(l1)):
    k = str(s0[i])+'_'+str(l1[i])
    if s1[i]<s2[i]:
        l_dict[k].append(0)
    else:
        l_dict[k].append(1)
        
for i in range(len(l2)):
    k = str(s0[i])+'_'+str(l2[i])
    if s1[i]>s2[i]:
        l_dict[k].append(0)
    else:
        l_dict[k].append(1)

awins_dict = defaultdict(lambda: np.nan)
for k,v in l_dict.items():
    awins_dict[k] = np.mean(l_dict[k])    


l_dict = defaultdict(list)
for i in range(len(l1)):
    k = str(s0[i])+'_'+str(l1[i])
    if s1[i]<s2[i]:
        l_dict[k].append(0)
    else:
        l_dict[k].append(awins_dict[str(s0[i])+'_'+str(l2[i])])
        
for i in range(len(l2)):
    k = str(s0[i])+'_'+str(l2[i])
    if s1[i]>s2[i]:
        l_dict[k].append(0)
    else:
        l_dict[k].append(awins_dict[str(s0[i])+'_'+str(l1[i])])

awins_dict = defaultdict(lambda: np.nan)
for k,v in l_dict.items():
    awins_dict[k] = np.mean(l_dict[k])



ss = regular_data.groupby(["Season", "T1_TeamID"])[boxcols].agg("mean").reset_index()

ss_T1 = ss.copy()
ss_T1.columns = ["T1_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(ss_T1.columns)]
ss_T1 = ss_T1.rename({"T1_avg_Season": "Season", "T1_avg_TeamID": "T1_TeamID"}, axis=1)

ss_T1['T1_WinRatio14d'] = ss_T1['Season'].map(str) + '_' + ss_T1['T1_TeamID'].map(str)
ss_T1['T1_away_wins'] = ss_T1['T1_WinRatio14d'].map(away_dict)
ss_T1['T1_awins'] = ss_T1['T1_WinRatio14d'].map(awins_dict)
#ss_T1['T1_loss_delta'] = ss_T1['T1_WinRatio14d'].map(loss_delta_dict)
ss_T1['T1_WinRatio14d'] = ss_T1['T1_WinRatio14d'].map(ratio_dict)



ss_T2 = ss.copy()
ss_T2.columns = ["T2_avg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(ss_T2.columns)]
ss_T2 = ss_T2.rename({"T2_avg_Season": "Season", "T2_avg_TeamID": "T2_TeamID"}, axis=1)

ss_T2['T2_WinRatio14d'] = ss_T2['Season'].map(str) + '_' + ss_T2['T2_TeamID'].map(str)
ss_T2['T2_away_wins'] = ss_T2['T2_WinRatio14d'].map(away_dict)
ss_T2['T2_awins'] = ss_T2['T2_WinRatio14d'].map(awins_dict)
#ss_T2['T2_loss_delta'] = ss_T2['T2_WinRatio14d'].map(loss_delta_dict)
ss_T2['T2_WinRatio14d'] = ss_T2['T2_WinRatio14d'].map(ratio_dict)

tourney_data = pd.merge(tourney_data, ss_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, ss_T2, on=["Season", "T2_TeamID"], how="left")

tourney_data['laplace_matchup'] = tourney_data['Season'].map(str)+'_'+tourney_data['T1_TeamID'].map(str)+'_'+tourney_data['T2_TeamID'].map(str)
tourney_data['laplace_matchup'] = tourney_data['laplace_matchup'].map(match_dict)


tourney_data


def update_elo(winner_elo, loser_elo):
    expected_win = expected_result(winner_elo, loser_elo)
    change_in_elo = k_factor * (1 - expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo


def expected_result(elo_a, elo_b):
    return 1.0 / (1 + 10 ** ((elo_b - elo_a) / elo_width))

base_elo = 1000
elo_width = 400
k_factor = 100

elos = []
for season in sorted(set(seeds["Season"])):
    ss = regular_data.loc[regular_data["Season"] == season]
    ss = ss.loc[ss["win"] == 1].reset_index(drop=True)
    teams = set(ss["T1_TeamID"]) | set(ss["T2_TeamID"])
    elo = dict(zip(teams, [base_elo] * len(teams)))
    for i in range(ss.shape[0]):
        w_team, l_team = ss.loc[i, "T1_TeamID"], ss.loc[i, "T2_TeamID"]
        w_elo, l_elo = elo[w_team], elo[l_team]
        w_elo_new, l_elo_new = update_elo(w_elo, l_elo)
        elo[w_team] = w_elo_new
        elo[l_team] = l_elo_new
    elo = pd.DataFrame.from_dict(elo, orient="index").reset_index()
    elo = elo.rename({"index": "TeamID", 0: "elo"}, axis=1)
    elo["Season"] = season
    elos.append(elo)
elos = pd.concat(elos)

elos_T1 = elos.copy().rename({"TeamID": "T1_TeamID", "elo": "T1_elo"}, axis=1)
elos_T2 = elos.copy().rename({"TeamID": "T2_TeamID", "elo": "T2_elo"}, axis=1)
tourney_data = pd.merge(tourney_data, elos_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, elos_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]



tmp = pd.merge(elos, tourney_data[['Season', 'T1_TeamID']].drop_duplicates(), 
               left_on = ['Season', 'TeamID'], 
               right_on = ['Season', 'T1_TeamID'],
               how = 'left')

plt.figure(figsize=(6,4))
sns.distplot(tmp.loc[pd.isnull(tmp['T1_TeamID']),'elo'], kde=False)
sns.distplot(tmp.loc[~pd.isnull(tmp['T1_TeamID']),'elo'], kde=False)


plt.figure(figsize=(6,4))
sns.stripplot(data = tourney_data, y = 'T1_elo', x = 'T1_seed', size=3)


plt.figure(figsize=(12,4))
sns.stripplot(data = tourney_data, y = 'elo_diff', x = 'Seed_diff', hue='win', dodge=True, size=3)
sns.lineplot([0]*29,color='gray',lw = 1)


regular_data["ST1"] = regular_data.apply(lambda t: str(int(t["Season"])) + "/" + str(int(t["T1_TeamID"])), axis=1)
regular_data["ST2"] = regular_data.apply(lambda t: str(int(t["Season"])) + "/" + str(int(t["T2_TeamID"])), axis=1)
seeds_T1["ST1"] = seeds_T1.apply(lambda t: str(int(t["Season"])) + "/" + str(int(t["T1_TeamID"])), axis=1)
seeds_T2["ST2"] = seeds_T2.apply(lambda t: str(int(t["Season"])) + "/" + str(int(t["T2_TeamID"])), axis=1)

# collect tourney teams
st = set(seeds_T1["ST1"]) | set(seeds_T2["ST2"])
# append non-tourney teams which were able to beat tourney team at least once
st = st | set(regular_data.loc[(regular_data["T1_Score"] > regular_data["T2_Score"]) & 
                               (regular_data["ST2"].isin(st)), "ST1"])

def team_quality(season, men_women):
    # mixed effects: fixed intercept=0, random slope
    formula = "PointDiff~-1+T1_TeamID+T2_TeamID"
    glm = sm.GLM.from_formula(
        formula=formula,
        data=dt.loc[(dt["Season"] == season) & (dt["men_women"] == men_women), :],
        family=sm.families.Gaussian(),
    ).fit()
    
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ["TeamID", "quality"]
    quality["quality"] = quality["quality"]
    quality["Season"] = season
    quality = quality.loc[quality.TeamID.str.contains("T1_")].reset_index(drop=True)
    quality["TeamID"] = quality["TeamID"].apply(lambda x: x[10:14]).astype(int)
    return quality


glm_quality = []

dt = regular_data.loc[regular_data["ST1"].isin(st) | regular_data["ST2"].isin(st)]
dt["T1_TeamID"] = dt["T1_TeamID"].astype(str)
dt["T2_TeamID"] = dt["T2_TeamID"].astype(str)
dt.loc[~dt["ST1"].isin(st), "T1_TeamID"] = "0000"
dt.loc[~dt["ST2"].isin(st), "T2_TeamID"] = "0000"
seasons = sorted(set(seeds["Season"]))
for s in tqdm(seasons, unit="season"):
    if s >= 2010:  # min season for women
        glm_quality.append(team_quality(s, 0))
    if s >= 2003:  # min season for men
        glm_quality.append(team_quality(s, 1))

glm_quality = pd.concat(glm_quality).reset_index(drop=True)

glm_quality_T1 = glm_quality.copy()
glm_quality_T2 = glm_quality.copy()
glm_quality_T1.columns = ["T1_TeamID", "T1_quality", "Season"]
glm_quality_T2.columns = ["T2_TeamID", "T2_quality", "Season"]
tourney_data = pd.merge(tourney_data, glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
tourney_data = pd.merge(tourney_data, glm_quality_T2, on=["Season", "T2_TeamID"], how="left")
tourney_data["diff_quality"] = tourney_data["T1_quality"] - tourney_data["T2_quality"]



tmp = (
    tourney_data[["Season", "men_women", "T1_seed", "T1_quality"]]
    .drop_duplicates()
    .sort_values("T1_quality")
    .reset_index(drop=True)
)

fig, axs = plt.subplots(ncols=2, figsize=(12, 4))

sns.lineplot(tmp.loc[tmp["men_women"] == 0, "T1_quality"], color="lightgray", ax=axs[0])
sns.scatterplot(tmp.loc[(tmp["men_women"] == 0) & (tmp.T1_seed == 1), "T1_quality"], color="red", ax=axs[0])
sns.scatterplot(tmp.loc[(tmp["men_women"] == 0) & (tmp.T1_seed == 7), "T1_quality"], color="blue", ax=axs[0])
sns.scatterplot(tmp.loc[(tmp["men_women"] == 0) & (tmp.T1_seed == 16), "T1_quality"], color="green", ax=axs[0])

sns.lineplot(tmp.loc[tmp["men_women"] == 1, "T1_quality"], color="lightgray", ax=axs[1])
sns.scatterplot(tmp.loc[(tmp["men_women"] == 1) & (tmp.T1_seed == 1), "T1_quality"], color="red", ax=axs[1])
sns.scatterplot(tmp.loc[(tmp["men_women"] == 1) & (tmp.T1_seed == 7), "T1_quality"], color="blue", ax=axs[1])
sns.scatterplot(tmp.loc[(tmp["men_women"] == 1) & (tmp.T1_seed == 16), "T1_quality"], color="green", ax=axs[1])


tmp["QualitySeed"] = (
    (tmp.groupby(["Season", "men_women"])["T1_quality"].rank(ascending=False) // 4 + 1).clip(1, 16).astype(int)
)
pd.pivot_table(data=tmp, index="T1_seed", columns="QualitySeed", values="men_women", aggfunc="count").fillna(0).astype(int).style.bar(color="#5fba7d", vmin=0, vmax=50)


print("Seed AUC    :", np.round(roc_auc_score(1 - tourney_data["win"], tourney_data["T1_seed"] - tourney_data["T2_seed"]), 3))
print("Quality AUC :", np.round(roc_auc_score(tourney_data["win"], tourney_data["T1_quality"] - tourney_data["T2_quality"]), 3))


# who is better, experts or statistics, by season
for s in sorted(set(tourney_data['Season'])):
    st = tourney_data['Season'] == s
    print(s, 
          a:=np.round(roc_auc_score(1-tourney_data.loc[st, "win"],tourney_data.loc[st, 'T1_seed'] - tourney_data.loc[st, 'T2_seed']),3),
          b:=np.round(roc_auc_score(tourney_data.loc[st, "win"],tourney_data.loc[st, 'T1_quality'] - tourney_data.loc[st, 'T2_quality']),3),
          np.where(a>b, '', 'Q')
         )



tourney_data


cat_features = [
    "T1_avg_FGM",
    "T1_avg_FGM3",
    "T1_avg_FGA3",
    "T1_avg_FTM",
    "T1_avg_FTA",
    "T1_avg_OR",
    "T1_avg_DR",
    "T1_avg_Ast",
    "T1_avg_TO",
    "T1_avg_Stl",
    "T1_avg_opponent_Score",
    "T1_avg_opponent_FGM",
    "T1_avg_opponent_FGM3",
    "T1_avg_opponent_FGA3",
    "T1_avg_opponent_FTM",
    "T1_avg_opponent_FTA",
    "T1_avg_opponent_OR",
    "T1_avg_opponent_DR",
    "T1_avg_opponent_Ast",
    "T1_avg_opponent_TO",
    "T1_avg_opponent_Stl",
    "T2_avg_FGM",
    "T2_avg_FGM3",
    "T2_avg_FGA3",
    "T2_avg_FTM",
    "T2_avg_FTA",
    "T2_avg_OR",
    "T2_avg_DR",
    "T2_avg_Ast",
    "T2_avg_TO",
    "T2_avg_Stl",
    "T2_avg_opponent_Score",
    "T2_avg_opponent_FGM",
    "T2_avg_opponent_FGM3",
    "T2_avg_opponent_FGA3",
    "T2_avg_opponent_FTM",
    "T2_avg_opponent_FTA",
    "T2_avg_opponent_OR",
    "T2_avg_opponent_DR",
    "T2_avg_opponent_Ast",
    "T2_avg_opponent_TO",
    "T2_avg_opponent_Stl",    
]   

np.random.seed(0)    
cb_model1a=make_pipeline(StandardScaler(),NuSVC(probability=True,nu=0.6,kernel='poly',gamma='scale',degree=2))
cb_model1b=make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=100,weights='uniform',p=2,metric='minkowski'))    
x_train = tourney_data.loc[tourney_data[ "men_women"] == 1, cat_features].values
y_train = tourney_data.loc[tourney_data[ "men_women"] == 1, "PointDiff"].values


y_train1 = (y_train > 0) *1.0
cb_model1a.fit(x_train, y_train1)
cb_model1b.fit(x_train, y_train)

x_test = tourney_data.loc[tourney_data[ "men_women"] != 1, cat_features].values

preds1a = cb_model1a.predict_proba(x_test)[:,1]
preds1b = cb_model1b.predict(x_test)

cb_model2a = make_pipeline(StandardScaler(), NuSVC(probability=True, nu = 0.6, kernel = 'poly', gamma='scale', degree=2))
cb_model2b = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=100, weights='uniform', p=2, metric='minkowski'))
x_train = tourney_data.loc[tourney_data[ "men_women"] != 1, cat_features].values
y_train = tourney_data.loc[tourney_data[ "men_women"] != 1, "PointDiff"].values
y_train2 = (y_train > 0) *1.0
cb_model2a.fit(x_train, y_train2)
cb_model2b.fit(x_train, y_train)

x_test = tourney_data.loc[tourney_data[ "men_women"] == 1, cat_features].values

preds2a = cb_model2a.predict_proba(x_test)[:,1]
preds2b = cb_model2b.predict(x_test)


tourney_data['cb_preds1'] = np.nan 
tourney_data.loc[tourney_data[ "men_women"] == 1, 'cb_preds1'] = preds2a
tourney_data.loc[tourney_data[ "men_women"] != 1, 'cb_preds1'] = preds1a   
tourney_data['cb_preds1'].describe()


tourney_data['cb_preds2'] = np.nan
tourney_data.loc[tourney_data[ "men_women"] == 1, 'cb_preds2'] = preds2b
tourney_data.loc[tourney_data[ "men_women"] != 1, 'cb_preds2'] = preds1b
tourney_data['cb_preds2'].describe()


features = [
    ### EASY FEATURES ###
    "men_women",    
    "T1_seed",
    "T2_seed",
    "Seed_diff",
    ### MEDIUM FEATURES ###
    "T1_avg_Score",
    "T1_avg_FGA",
    "T1_awins", "T2_awins",
    "T1_away_wins","T2_away_wins",
    "laplace_matchup",
    "T1_WinRatio14d",
    "T2_WinRatio14d",
    "cb_preds1","cb_preds2",
    "T1_avg_Blk",
    "T1_avg_PF",
    "T1_avg_opponent_FGA",
    "T1_avg_opponent_Blk",
    "T1_avg_opponent_PF",
    "T1_avg_PointDiff",
    "T2_avg_Score",
    "T2_avg_FGA",
    "T2_avg_Blk",
    "T2_avg_PF",
    "T2_avg_opponent_FGA",
    "T2_avg_opponent_Blk",
    "T2_avg_opponent_PF",
    "T2_avg_PointDiff",
    ### HARD FEATURES ###
    "T1_elo",
    "T2_elo",    
    "elo_diff",
    ### HARDEST FEATURES ###
    "T1_quality",
    "T2_quality",
]

print(f"Number of features {len(features)}")


param = {}
param["objective"] = "reg:squarederror"
param["booster"] = "gbtree"
param["eta"] = 0.01
param["subsample"] = 0.6
param["colsample_bynode"] = 0.8
param["num_parallel_tree"] = 2
param["min_child_weight"] = 4
param["max_depth"] = 4
param["tree_method"] = "hist"
param['grow_policy'] = 'lossguide'
param["max_bin"] = 32


num_rounds = 700  

def cauchyobj(preds,dtrain): 
    labels=dtrain.get_label() 
    c=5000 
    x=preds-labels  
    x2=x**2 
    c2=c**2 
    grad=x/(x2/c2+1)   
    hess=-c2*(x2-c2)/(x2+c2)**2 
    return grad,hess 


xgb_parameters = {
    'eval_metric': 'mae',
    'eta': 0.02,
    'subsample': 0.35,
    'colsample_bytree': 0.7,
    'num_parallel_tree': 10,
    'min_child_weight': 40,
    'max_depth': 4, #worse at both 3,5,
    'gamma': 10,
}


models = {}
oof_mae = []
oof_preds = []
oof_targets = []
oof_ss = []    

for oof_season in set(tourney_data.Season): 
    x_train = tourney_data.loc[tourney_data['Season']!=oof_season,features].values    
    y_train = tourney_data.loc[tourney_data['Season']!=oof_season,'PointDiff'].values  
    x_val = tourney_data.loc[tourney_data["Season"] == oof_season, features].values
    y_val = tourney_data.loc[tourney_data["Season"] == oof_season, "PointDiff"].values
    s_val = tourney_data.loc[tourney_data["Season"] == oof_season, "Season"].values 


    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)   

    models[oof_season]=xgb.train(params=xgb_parameters,dtrain=dtrain,num_boost_round=400,obj=cauchyobj)   

    preds=models[oof_season].predict(dval)    

    print(f'oof season {oof_season} mae:{mean_absolute_error(y_val,preds)}')    

    oof_mae.append(mean_absolute_error(y_val,preds))               
    oof_preds+=list(preds)   
    oof_targets+=list(y_val)  
    oof_ss+=list(s_val)
    
print(f'average mae:{np.mean(oof_mae)}')


def make_strings(x):
    output_list = []
    for i in range(x.shape[0]):
        tmp_str = ''
        for j in range(x.shape[1]):
            tmp_str += str(j)+'_'+ str(int(x[i][j])) + ' '
        output_list.append(tmp_str)
    return output_list


brier_list = []
logistic_dict = dict()
# leave-one-season out models
for oof_season in set(tourney_data.Season):
    gc.collect()
    x_train = tourney_data.loc[tourney_data["Season"] != oof_season, features].values
    y_train = tourney_data.loc[tourney_data["Season"] != oof_season, "PointDiff"].values
    x_val = tourney_data.loc[tourney_data["Season"] == oof_season, features].values
    y_val = tourney_data.loc[tourney_data["Season"] == oof_season, "PointDiff"].values
    s_val = tourney_data.loc[tourney_data["Season"] == oof_season, "Season"].values
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    
    tmp = models[oof_season].predict(dtrain, pred_leaf=True)
    tmp = make_strings(tmp)
    calibrater = make_pipeline(TfidfVectorizer(), LogisticRegression(solver='liblinear',random_state =0,max_iter=300))
                                                                    # penalty='elasticnet', l1_ratio=0.5)) #penalty=None bad
    y_val_bin = (y_train>0)*1
    calibrater.fit(tmp, y_val_bin)
    logistic_dict[oof_season] = calibrater
    
    tmp = models[oof_season].predict(dval, pred_leaf=True)
    tmp = make_strings(tmp)
    preds = calibrater.predict_proba(tmp)[:,1]
    y_val_bin = (y_val>0)*1
    bscore = brier_score_loss(y_val_bin, preds)
    brier_list.append(bscore)
    print(f"oof season {oof_season} brier: {bscore}")


np.mean(brier_list)



X = pd.read_csv(f"{data_dir}/SampleSubmissionStage2.csv")
X


X['Season'] = X['ID'].apply(lambda t: int(t.split('_')[0]))
X['T1_TeamID'] = X['ID'].apply(lambda t: int(t.split('_')[1]))
X['T2_TeamID'] = X['ID'].apply(lambda t: int(t.split('_')[2]))
X['men_women'] = X['T1_TeamID'].apply(lambda t: 0 if str(t)[0]=='1' else 1)
X = pd.merge(X, ss_T1, on = ['Season', 'T1_TeamID'], how = 'left')
X = pd.merge(X, ss_T2, on = ['Season', 'T2_TeamID'], how = 'left')
X = pd.merge(X, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
X = pd.merge(X, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')
X = pd.merge(X, glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, glm_quality_T2, on=["Season", "T2_TeamID"], how="left")
X = pd.merge(X, elos_T1, on=["Season", "T1_TeamID"], how="left")
X = pd.merge(X, elos_T2, on=["Season", "T2_TeamID"], how="left")
X["Seed_diff"] = X["T2_seed"] - X["T1_seed"]
X["elo_diff"] = X["T1_elo"] - X["T2_elo"]
X["diff_quality"] = X["T1_quality"] - X["T2_quality"]

X['laplace_matchup'] =  X['Season'].map(str)+'_'+X['T1_TeamID'].map(str)+'_'+X['T2_TeamID'].map(str)
X['laplace_matchup'] = X['laplace_matchup'].map(match_dict)


x_test = X.loc[X[ "men_women"] != 1, cat_features].values
#preds1 = cb_model1.predict_proba(x_test)[:,1]
preds1 = cb_model1a.predict_proba(x_test)[:,1]

x_test = X.loc[X[ "men_women"] == 1, cat_features].values
#preds2 = cb_model2.predict_proba(x_test)[:,1]
preds2 = cb_model2a.predict_proba(x_test)[:,1]

X['cb_preds1'] = np.nan
X.loc[X[ "men_women"] == 1, 'cb_preds1'] = preds2
X.loc[X[ "men_women"] != 1, 'cb_preds1'] = preds1
X['cb_preds1'].describe()


x_test = X.loc[X[ "men_women"] != 1, cat_features].values
preds1 = cb_model1b.predict(x_test)
x_test = X.loc[X[ "men_women"] == 1, cat_features].values
preds2 = cb_model2b.predict(x_test)

X['cb_preds2'] = np.nan
X.loc[X[ "men_women"] == 1, 'cb_preds2'] = preds2
X.loc[X[ "men_women"] != 1, 'cb_preds2'] = preds1
X['cb_preds2'].describe()


X.head(40)



X['laplace_matchup'].value_counts()



seeds_set = set( seeds[seeds['Season']==2025]['TeamID'].tolist())



b1 = X['T1_TeamID'].isin(seeds_set)
b2 = X['T2_TeamID'].isin(seeds_set)
b3 = b1 & b2
np.sum(b3)


X_select = X[b3].reset_index(drop=True).copy()
print(X_select.shape)
X_select.head(3)



preds = []
for oof_season in set(tourney_data.Season):
    print(oof_season)
    gc.collect()
    dtest = xgb.DMatrix(X_select[features].values)
    margin_preds = models[oof_season].predict(dtest, pred_leaf=True)
    margin_preds = make_strings(margin_preds)
    probs = logistic_dict[oof_season].predict_proba(margin_preds)[:,1]
    preds.append(probs)
X_select['Pred'] = np.array(preds).mean(axis=0) 


X_not = X[~b3].reset_index(drop=True).copy()
X_not['Pred'] = 0.5
print(X_not.shape)


X_both = pd.concat([X_select[['ID','Pred']], X_not[['ID','Pred']]], axis=0).reset_index(drop=True)
print(X_both.shape)
print(X.shape)



X_both[['ID','Pred']].to_csv('submission.csv',index=None)

