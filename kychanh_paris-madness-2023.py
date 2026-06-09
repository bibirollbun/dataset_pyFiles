import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from scipy.interpolate import UnivariateSpline
from sklearn import preprocessing
from sklearn.metrics import brier_score_loss, log_loss, silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from tqdm import tqdm
import collections

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option("display.max_column", 200)
pd.set_option("display.max_rows", 20)
# print(os.listdir("../input"))
xgb.__version__ # I used '1.2.0-SNAPSHOT'


SUB=True
NORMALIZE_QUALITY=False
REPEAT_CV=5
USE_GPU=True
CLUSTERING_METHOD="KMeans"
N_CLUSTER=11


DATA_PATH = '/kaggle/input/march-machine-learning-mania-2025/'


tourney_results = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneyDetailedResults.csv"),
    pd.read_csv(DATA_PATH + "WNCAATourneyDetailedResults.csv"),
], ignore_index=True)

seeds = pd.concat([
    pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv"),
    pd.read_csv(DATA_PATH + "WNCAATourneySeeds.csv"),
], ignore_index=True)

regular_results = pd.concat([
    pd.read_csv(DATA_PATH + "MRegularSeasonDetailedResults.csv"),
    pd.read_csv(DATA_PATH + "WRegularSeasonDetailedResults.csv"),
], ignore_index=True)


mteams=pd.read_csv(DATA_PATH+'MTeams.csv')
wteams=pd.read_csv(DATA_PATH+'WTeams.csv')


def prepare_data(df):
    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 
    'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 
    'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]

    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
    df.columns.values[6] = 'location'
    dfswap.columns.values[6] = 'location'    
      
    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]

    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output.loc[output.location=='N','location'] = '0'
    output.loc[output.location=='H','location'] = '1'
    output.loc[output.location=='A','location'] = '-1'
    output.location = output.location.astype(int)
    
    output['PointDiff'] = output['T1_Score'] - output['T2_Score']
    
    return output


regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)


index_cols=['T1_PI','T1_IRI','T1_TPRI',
            'T1_FTR','T1_ORAI',
            'T1_DPI',
            'T2_PI','T2_IRI','T2_TPRI',
            'T2_FTR','T2_ORAI',
            'T2_DPI']


regular_data['T1_PI']=(regular_data['T1_FGA']+.475*regular_data['T1_FTA']+regular_data['T1_TO']-regular_data['T1_OR'])/(40+5*regular_data['NumOT'])
regular_data['T1_IRI']=1-regular_data['T1_Ast']/regular_data['T1_FGM']
regular_data['T1_TPRI']=regular_data['T1_FGA3']/regular_data['T1_FGA']
regular_data['T1_FTR']=regular_data['T1_FTA']/regular_data['T1_FGA']
regular_data['T1_ORAI']=regular_data['T1_OR']/(regular_data['T1_OR']+regular_data['T2_DR'])
regular_data['T1_DPI']=(regular_data['T1_Stl']+regular_data['T2_TO'])/(regular_data['T2_FGA']+regular_data['T2_FTA'])

regular_data['T2_PI']=(regular_data['T2_FGA']+.475*regular_data['T2_FTA']+regular_data['T2_TO']-regular_data['T2_OR'])/(40+5*regular_data['NumOT'])
regular_data['T2_IRI']=1-regular_data['T2_Ast']/regular_data['T2_FGM']
regular_data['T2_TPRI']=regular_data['T2_FGA3']/regular_data['T2_FGA']
regular_data['T2_FTR']=regular_data['T2_FTA']/regular_data['T2_FGA']
regular_data['T2_ORAI']=regular_data['T2_OR']/(regular_data['T2_OR']+regular_data['T1_DR'])
regular_data['T2_DPI']=(regular_data['T2_Stl']+regular_data['T1_TO'])/(regular_data['T1_FGA']+regular_data['T1_FTA'])


tourney_data.columns


boxscore_cols = ['T1_Score', 'T2_Score', 
        'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_FTM', 'T1_FTA', 'T1_OR', 'T1_DR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_Blk', 'T1_PF', 
        'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_FTM', 'T2_FTA', 'T2_OR', 'T2_DR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk', 'T2_PF', 
        'PointDiff']

boxscore_cols = [
        'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF', 
        'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk',  
        'PointDiff']

funcs = [np.mean]


season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols+index_cols].agg(funcs).reset_index()
season_statistics.head()


season_statistics.columns = [''.join(col).strip() for col in season_statistics.columns.values]
season_statistics.head()


season_statistics_T1 = season_statistics.copy()
season_statistics_T2 = season_statistics.copy()

season_statistics_T1.columns = ["T1_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T1.columns)]
season_statistics_T2.columns = ["T2_" + x.replace("T1_","").replace("T2_","opponent_") for x in list(season_statistics_T2.columns)]
season_statistics_T1.columns.values[0] = "Season"
season_statistics_T2.columns.values[0] = "Season"


season_statistics_T1.head()


season_statistics_T2.head()


tourney_data.head()


tourney_data = tourney_data[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID' ,'T2_Score']]
tourney_data.head()


tourney_data = pd.merge(tourney_data, season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, season_statistics_T2, on = ['Season', 'T2_TeamID'], how = 'left')


tourney_data.head()


last14days_stats_T1 = regular_data.loc[regular_data.DayNum>118].reset_index(drop=True)
last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff']>0,1,0)
last14days_stats_T1 = last14days_stats_T1.groupby(['Season','T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')

last14days_stats_T2 = regular_data.loc[regular_data.DayNum>118].reset_index(drop=True)
last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff']<0,1,0)
last14days_stats_T2 = last14days_stats_T2.groupby(['Season','T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')


tourney_data = pd.merge(tourney_data, last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')


regular_season_effects = regular_data[['Season','T1_TeamID','T2_TeamID','PointDiff']].copy()
regular_season_effects['T1_TeamID'] = regular_season_effects['T1_TeamID'].astype(str)
regular_season_effects['T2_TeamID'] = regular_season_effects['T2_TeamID'].astype(str)
regular_season_effects['win'] = np.where(regular_season_effects['PointDiff']>0,1,0)
march_madness = pd.merge(seeds[['Season','TeamID']],seeds[['Season','TeamID']],on='Season')
march_madness.columns = ['Season', 'T1_TeamID', 'T2_TeamID']
march_madness.T1_TeamID = march_madness.T1_TeamID.astype(str)
march_madness.T2_TeamID = march_madness.T2_TeamID.astype(str)
regular_season_effects = pd.merge(regular_season_effects, march_madness, on = ['Season','T1_TeamID','T2_TeamID'])
regular_season_effects.shape


regular_season_effects.head()


def team_quality(season):
    formula = 'win~-1+T1_TeamID+T2_TeamID'
    glm = sm.GLM.from_formula(formula=formula, 
                              data=regular_season_effects.loc[regular_season_effects.Season==season,:], 
                              family=sm.families.Binomial()).fit()
    
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ['TeamID','quality']
    quality['Season'] = season
    #quality['quality'] = np.exp(quality['quality'])
    quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)
    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)
    return quality


if NORMALIZE_QUALITY:     
    def normalize_column(values):
        themean = np.mean(values)
        thestd = np.std(values)
        norm = (values - themean)/(thestd) 
        return(pd.DataFrame(norm))
    
    def team_quality(season):
        formula = 'win~-1+T1_TeamID+T2_TeamID'
        glm = sm.GLM.from_formula(formula=formula, 
                                  data=regular_season_effects.loc[regular_season_effects.Season==season,:], 
                                  family=sm.families.Binomial()).fit()
        quality = pd.DataFrame(glm.params).reset_index()
        quality.columns = ['TeamID','quality']
        quality['Season'] = season
        quality['quality'] = normalize_column(quality['quality'])
        quality['quality'] = np.exp(quality['quality'])
        quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)
        quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)
        print(quality['quality'].mean(), quality['quality'].std())
        return quality


glm_quality = pd.concat([team_quality(2010),
                         team_quality(2011),
                         team_quality(2012),
                         team_quality(2013),
                         team_quality(2014),
                         team_quality(2015),
                         team_quality(2016),
                         team_quality(2017),
                         team_quality(2018),
                         team_quality(2019),
                         ##team_quality(2020),
                         team_quality(2021),
                         team_quality(2022),
                         team_quality(2023),
                         team_quality(2024),
                         team_quality(2025)
                         ]).reset_index(drop=True)


glm_quality.head(20)


glm_quality_T1 = glm_quality.copy()
glm_quality_T2 = glm_quality.copy()
glm_quality_T1.columns = ['T1_TeamID','T1_quality','Season']
glm_quality_T2.columns = ['T2_TeamID','T2_quality','Season']


glm_quality_T2.head()


tourney_data = pd.merge(tourney_data, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')


tourney_data.head()


seeds.head()


seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
seeds.tail()


seeds_T1 = seeds[['Season','TeamID','seed']].copy()
seeds_T2 = seeds[['Season','TeamID','seed']].copy()
seeds_T1.columns = ['Season','T1_TeamID','T1_seed']
seeds_T2.columns = ['Season','T2_TeamID','T2_seed']


tourney_data = pd.merge(tourney_data, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')


tourney_data["Seed_diff"] = tourney_data["T1_seed"] - tourney_data["T2_seed"]


# 데이터 파일 로드
df = season_statistics.copy()

if not SUB:
    df=df[df.Season<2023]

# 팀별 고유 ID 생성 (Season + TeamID)
df["Season_TeamID_T1"] = df["Season"].astype(str) + "_" + df["T1_TeamID"].astype(str)
#df["Season_TeamID_T2"] = df["Season"].astype(str) + "_" + df["T2_TeamID"].astype(str)

# 팀별 경기 성적 기반 Feature 선택
team_features = [
    "T1_Scoremean", "T1_opponent_Scoremean",  # 득점력 & 상대 실점
    "T1_FGMmean", "T1_FGAmean", "T1_FGA3mean","T1_FTAmean",  # 야투 효율성
    "T1_ORmean", "T1_DRmean",  # 리바운드 능력
    "T1_TOmean", "T1_Stlmean", "T1_Blkmean"  # 턴오버, 스틸, 블록
]

team_features=[_+'mean' for _ in index_cols[:6]]

# 팀별 평균 경기 성적 계산
team_stats = df.groupby("Season_TeamID_T1")[team_features].mean()
scaler=StandardScaler()
team_stats_scaled=scaler.fit_transform(team_stats)


# 여러 개의 클러스터 개수에 대해 WCSS 계산
wcss = [];aics=[];bics=[]
if CLUSTERING_METHOD=="KMeans":
    for k in range(2, 26):  # k=2부터 10까지 테스트
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(team_stats)
        wcss.append(kmeans.inertia_)  # WCSS 저장

    # WCSS 그래프 출력
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 26), wcss, marker="o", linestyle="-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.title("Elbow Method for Optimal k")
    plt.show()

elif CLUSTERING_METHOD=="GMM":
    for k in range(2,32):
        gmm=GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm.fit(team_stats_scaled)
        aics.append((gmm.aic(team_stats_scaled)))
        bics.append(gmm.bic(team_stats_scaled))

    plt.figure(figsize=(8, 5))
    #plt.plot(range(2,32), bics, marker="o", linestyle="-", label="BIC Score")
    #plt.title('BIC')
    #plt.show()
    #plt.figure(figsize=(8,5))
    plt.plot(range(2,32), aics, marker="s", linestyle="--", label="AIC Score")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("AIC & BIC")
    #plt.title('AIC')
    #plt.show()
    plt.title("BIC & AIC for Optimal k in GMM")
    #plt.legend()


# K-Means 클러스터링 수행 (클러스터 개수 4개, 조정 가능)
if CLUSTERING_METHOD=="KMeans":
    kmeans = KMeans(n_clusters=N_CLUSTER, random_state=42, n_init=10)
    team_stats["Cluster"] = kmeans.fit_predict(team_stats)
elif CLUSTERING_METHOD=="GMM":
    gmm = GaussianMixture(n_components=N_CLUSTER, covariance_type="full", random_state=42)
    team_stats['Cluster']=gmm.fit_predict(team_stats_scaled)

# 원래 데이터에 클러스터 정보 추가 (T1, T2 팀 각각)
df = df.merge(team_stats[["Cluster"]], left_on="Season_TeamID_T1", right_index=True)
df = df.rename(columns={"Cluster": "Cluster_T1"})  # T1 클러스터

#df = df.merge(team_stats[["Cluster"]], left_on="Season_TeamID_T2", right_index=True)
#df = df.rename(columns={"Cluster": "Cluster_T2"})  # T2 클러스터

# 클러스터 차이를 Feature로 추가 (팀 간 경기력 패턴 차이 반영)
#df["Cluster_Diff"] = df["Cluster_T1"] - df["Cluster_T2"]

# 결과 확인
df[['T1_TeamID','Cluster_T1']].head()
#print(df[["T1_TeamID", "T2_TeamID", "Cluster_T1", "Cluster_T2", "Cluster_Diff"]].head())


tourney_data=pd.merge(tourney_data,df[["Season",'T1_TeamID','Cluster_T1']],on=['Season','T1_TeamID'])
df_T2=df.copy()
df_T2.columns=[x.replace('T1','T2') for x in df_T2.columns]
tourney_data=pd.merge(tourney_data,df_T2[['Season','T2_TeamID','Cluster_T2']],on=['Season','T2_TeamID'])
tourney_data


regular_data=pd.merge(regular_data,df[["Season",'T1_TeamID','Cluster_T1']],on=['Season','T1_TeamID'])
df_T2=df.copy()
df_T2.columns=[x.replace('T1','T2') for x in df_T2.columns]
regular_data=pd.merge(regular_data,df_T2[['Season','T2_TeamID','Cluster_T2']],on=['Season','T2_TeamID'])
regular_data


# 승리한 팀의 클러스터 정보 추가
df=pd.concat([tourney_data[['Season','T1_TeamID','T1_Score','T2_TeamID','T2_Score','Cluster_T1','Cluster_T2']],#])
             regular_data[['Season','T1_TeamID','T1_Score','T2_TeamID','T2_Score','Cluster_T1','Cluster_T2']]])

df["Winner_Cluster"] = np.where(df["T1_Score"] > df["T2_Score"], df["Cluster_T1"], df["Cluster_T2"])
df["Loser_Cluster"] = np.where(df["T1_Score"] > df["T2_Score"], df["Cluster_T2"], df["Cluster_T1"])

# 클러스터 간 매치업 통계 생성
cluster_matchup = df.groupby(["Winner_Cluster", "Loser_Cluster"]).size().reset_index(name="Wins")

# 총 경기 수 계산 (클러스터1 vs 클러스터2의 모든 경기)
total_games = df.groupby(["Cluster_T1", "Cluster_T2"]).size().reset_index(name="Total_Games")

# 양방향 매치업을 위해 (Cluster_T1 vs Cluster_T2)와 (Cluster_T2 vs Cluster_T1) 합치기
total_games_reversed = df.groupby(["Cluster_T2", "Cluster_T1"]).size().reset_index(name="Total_Games")
total_games = pd.concat([total_games, total_games_reversed])

# 클러스터1 vs 클러스터2와 반대 방향을 맞춰주기 위해 정렬 후 중복 제거
total_games = total_games.groupby(["Cluster_T1", "Cluster_T2"])["Total_Games"].sum().reset_index()

# 승률 계산을 위해 매치업 데이터를 병합
cluster_winrate = cluster_matchup.merge(total_games, 
                                        left_on=["Winner_Cluster", "Loser_Cluster"], 
                                        right_on=["Cluster_T1", "Cluster_T2"])

# 승률 계산
cluster_winrate["Win_Rate"] = cluster_winrate["Wins"] / cluster_winrate["Total_Games"]

# 불필요한 컬럼 정리
cluster_winrate = cluster_winrate[["Winner_Cluster", "Loser_Cluster", "Win_Rate"]]

# 승률 데이터 출력
cluster_winrate.head()

# CSV로 저장 (원하면 사용)
#cluster_winrate.to_csv("/mnt/data/cluster_winrate.csv", index=False)


# 피벗 테이블 생성 (Heatmap을 위한 형태로 변환)
winrate_matrix = cluster_winrate.pivot(index="Winner_Cluster", columns="Loser_Cluster", values="Win_Rate")

# 히트맵 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(winrate_matrix, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)

# 그래프 설정
plt.title("Cluster Matchup Win Rates")
plt.xlabel("Loser Cluster")
plt.ylabel("Winner Cluster")

# 그래프 출력
plt.show()


# 실루엣 스코어 계산
silhouette_avg = silhouette_score(team_stats, team_stats["Cluster"])
print(f"Silhouette Score: {silhouette_avg:.4f}")


# PCA를 사용하여 2차원으로 변환
pca = PCA(n_components=2)
team_stats_pca = pca.fit_transform(team_stats.drop(columns=["Cluster"]))  # 클러스터 컬럼 제외

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(team_stats_pca[:, 0], team_stats_pca[:, 1], c=team_stats["Cluster"], cmap="rainbow", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Cluster Visualization using PCA")
plt.colorbar(label="Cluster")
plt.show()


# 클러스터 승률을 원래 데이터에 병합
tourney_data = tourney_data.merge(cluster_winrate, 
              left_on=["Cluster_T1", "Cluster_T2"], 
              right_on=["Winner_Cluster", "Loser_Cluster"], 
              how="left")

# 결측값 처리 (매치업 데이터가 없을 경우 50% 승률로 설정)
tourney_data["Win_Rate"].fillna(0.5, inplace=True)
#tourney_data['Win_Rate']*=10

# ✅ Win_Rate를 -1 ~ 1 범위로 조정 (기존 0~1에서 -1~1로 변환)
tourney_data["Cluster_Win_Ratio"] = (tourney_data["Win_Rate"] - 0.5) * 2

# ✅ OneHotEncoder 초기화 (drop="first" 제거)
encoder = OneHotEncoder(sparse=False)  # 기존 drop="first" 제거

# ✅ 클러스터 원-핫 인코딩 수행
cluster_features = encoder.fit_transform(tourney_data[["Cluster_T1", "Cluster_T2"]])
#cluster_feature_names = encoder.get_feature_names_out(["Cluster_T1", "Cluster_T2"])
# ✅ feature 이름을 수동으로 생성 (get_feature_names_out() 대체)
categories = encoder.categories_
cluster_feature_names = [f"Cluster_{col}_{int(val)}" for col, cat in zip(["T1", "T2"], categories) for val in cat]

# ✅ 데이터프레임 변환 후 병합
cluster_df = pd.DataFrame(cluster_features, columns=cluster_feature_names)

# ✅ 첫 번째 컬럼 제거 (다중공선성 방지)
cluster_df = cluster_df.drop(columns=['Cluster_T1_0','Cluster_T2_0'])

tourney_data = pd.concat([tourney_data, cluster_df], axis=1)

# ✅ 결과 확인
tourney_data.head()


pd.set_option('mode.chained_assignment',  None)
cluster_sub=season_statistics_T1[season_statistics_T1.Season>=2021]
cluster_sub['Cluster_T1']=gmm.predict(scaler.transform(cluster_sub[team_features])) if CLUSTERING_METHOD=="GMM" else kmeans.predict(cluster_sub[team_features])
for _ in range(1,N_CLUSTER):
    cluster_sub[f'Cluster_T1_{_}']=(cluster_sub['Cluster_T1']==_).astype(float)

cluster_sub_T2=cluster_sub.copy()


y = tourney_data['T1_Score'] - tourney_data['T2_Score']
y.describe()


features = (
    list(season_statistics_T1.columns[2:999]) +
    list(season_statistics_T2.columns[2:999]) + \
    list(seeds_T1.columns[2:999]) + \
    list(seeds_T2.columns[2:999]) + ['Seed_diff']+['T1_quality','T2_quality']
    +list(last14days_stats_T1.columns[2:999])
    +list(last14days_stats_T2.columns[2:999])
    #+['Cluster_Win_Ratio']
    +['Win_Rate']
    +[_ for _ in cluster_feature_names if _[-1]!='0']
)
len(features)


X = tourney_data[features].values
dtrain = xgb.DMatrix(X, label = y)


def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000 
    x =  preds-labels    
    grad = x / (x**2/c**2+1)
    hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
    return grad, hess


param = {} 
#param['objective'] = 'reg:linear'
param['eval_metric'] =  'mae'
param['booster'] = 'gbtree'
param['eta'] = 0.05 #change to ~0.02 for final run
param['subsample'] = 0.35
param['colsample_bytree'] = 0.7
param['num_parallel_tree'] = 20 if USE_GPU else 3 #recommend 10
param['min_child_weight'] = 40
param['gamma'] = 10
param['max_depth'] =  3
#param['silent'] = 1
if USE_GPU: 
    param['tree_method'] ='hist'
    param['device']='cuda'

print(param)


xgb_cv = []
repeat_cv = REPEAT_CV # recommend 10

for i in range(repeat_cv): 
    print(f"Fold repeater {i}")
    xgb_cv.append(
        xgb.cv(
          params = param,
          dtrain = dtrain,
          obj = cauchyobj,
          num_boost_round = 3000,
          folds = KFold(n_splits = 5, shuffle = True, random_state = i),
          early_stopping_rounds = 25,
          verbose_eval = 50
        )
    )


iteration_counts = [np.argmin(x['test-mae-mean'].values) for x in xgb_cv]
val_mae = [np.min(x['test-mae-mean'].values) for x in xgb_cv]
iteration_counts, val_mae


oof_preds = []
for i in range(repeat_cv):
    print(f"Fold repeater {i}")
    preds = y.copy()
    kfold = KFold(n_splits = 5, shuffle = True, random_state = i)    
    for train_index, val_index in kfold.split(X,y):
        dtrain_i = xgb.DMatrix(X[train_index], label = y[train_index])
        dval_i = xgb.DMatrix(X[val_index], label = y[val_index])  
        model = xgb.train(
              params = param,
              dtrain = dtrain_i,
              num_boost_round = iteration_counts[i],
              verbose_eval = 50
        )
        preds[val_index] = model.predict(dval_i)
    oof_preds.append(np.clip(preds,-30,30))


plot_df = pd.DataFrame({"pred":oof_preds[0], "label":np.where(y>0,1,0)})
plot_df["pred_int"] = plot_df["pred"].astype(int)
plot_df = plot_df.groupby('pred_int')['label'].mean().reset_index(name='average_win_pct')

plt.figure()
plt.plot(plot_df.pred_int,plot_df.average_win_pct)


spline_model = []

for i in range(repeat_cv):
    dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
    dat = sorted(dat, key = lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]]= dat[k][1]
        
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])
    
    print(f"logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}") 


plot_df = pd.DataFrame({"pred":oof_preds[0], "label":np.where(y>0,1,0), "spline":spline_model[0](oof_preds[0])})
plot_df["pred_int"] = (plot_df["pred"]).astype(int)
plot_df = plot_df.groupby('pred_int')[['spline','label']].mean().reset_index()

plt.figure()
plt.plot(plot_df.pred_int,plot_df.spline)
plt.plot(plot_df.pred_int,plot_df.label)


spline_model = []

for i in range(repeat_cv):
    dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
    dat = sorted(dat, key = lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]]= dat[k][1]
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])
    spline_fit = np.clip(spline_fit,0.015,0.985)
    
    print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}") 


spline_model = []

for i in range(repeat_cv):
    dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
    dat = sorted(dat, key = lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]]= dat[k][1]
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])
    spline_fit = np.clip(spline_fit,0.015,0.985)
    spline_fit[(tourney_data.T1_seed==1) & (tourney_data.T2_seed==16) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
    spline_fit[(tourney_data.T1_seed==2) & (tourney_data.T2_seed==15) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
    spline_fit[(tourney_data.T1_seed==3) & (tourney_data.T2_seed==14) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
    spline_fit[(tourney_data.T1_seed==4) & (tourney_data.T2_seed==13) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
    spline_fit[(tourney_data.T1_seed==16) & (tourney_data.T2_seed==1) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
    spline_fit[(tourney_data.T1_seed==15) & (tourney_data.T2_seed==2) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
    spline_fit[(tourney_data.T1_seed==14) & (tourney_data.T2_seed==3) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
    spline_fit[(tourney_data.T1_seed==13) & (tourney_data.T2_seed==4) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
    
    print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}") 


val_cv = []
spline_model = []

for i in range(repeat_cv):
    dat = list(zip(oof_preds[i],np.where(y>0,1,0)))
    dat = sorted(dat, key = lambda x: x[0])
    datdict = {}
    for k in range(len(dat)):
        datdict[dat[k][0]]= dat[k][1]
    spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
    spline_fit = spline_model[i](oof_preds[i])
    spline_fit = np.clip(spline_fit,0.015,0.985)
    spline_fit[(tourney_data.T1_seed==1) & (tourney_data.T2_seed==16) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
    spline_fit[(tourney_data.T1_seed==2) & (tourney_data.T2_seed==15) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
    spline_fit[(tourney_data.T1_seed==3) & (tourney_data.T2_seed==14) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
    spline_fit[(tourney_data.T1_seed==4) & (tourney_data.T2_seed==13) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
    spline_fit[(tourney_data.T1_seed==16) & (tourney_data.T2_seed==1) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
    spline_fit[(tourney_data.T1_seed==15) & (tourney_data.T2_seed==2) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
    spline_fit[(tourney_data.T1_seed==14) & (tourney_data.T2_seed==3) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
    spline_fit[(tourney_data.T1_seed==13) & (tourney_data.T2_seed==4) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
    
    val_cv.append(pd.DataFrame({"y":np.where(y>0,1,0), "pred":spline_fit, "season":tourney_data.Season}))
    print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y>0,1,0),spline_fit)}") 
    
val_cv = pd.concat(val_cv)
val_cv.groupby('season').apply(lambda x: log_loss(x.y, x.pred))


sub = pd.read_csv(DATA_PATH + f"SampleSubmissionStage{int(SUB)+1}.csv")
sub['Season'] = sub['ID'].apply(lambda x: int(x.split('_')[0]))
sub["T1_TeamID"] = sub['ID'].apply(lambda x: int(x.split('_')[1]))
sub["T2_TeamID"] = sub['ID'].apply(lambda x: int(x.split('_')[2]))

sub.head()


seeds_T1.tail()


sub = pd.merge(sub, season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, season_statistics_T2, on = ['Season', 'T2_TeamID'], how = 'left')

sub = pd.merge(sub, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')

sub = pd.merge(sub, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')
sub = pd.merge(sub, last14days_stats_T1, on = ['Season', 'T1_TeamID'], how = 'left')
sub = pd.merge(sub, last14days_stats_T2, on = ['Season', 'T2_TeamID'], how = 'left')

sub["Seed_diff"] = sub["T1_seed"] - sub["T2_seed"]

sub.head()


sub=pd.merge(sub,
             cluster_sub[['Season','T1_TeamID','Cluster_T1']+list(cluster_feature_names[1:len(cluster_feature_names)//2])],
             on=['Season','T1_TeamID'],how='left')
cluster_sub.columns=[x.replace('T1','T2') for x in cluster_sub.columns]
sub=pd.merge(sub,
             cluster_sub[['Season','T2_TeamID','Cluster_T2']+list(cluster_feature_names[len(cluster_feature_names)//2+1:])],
             on=['Season','T2_TeamID'],how='left')
print(sub.shape)
sub=sub.merge(cluster_winrate,
              left_on=['Cluster_T1','Cluster_T2'],right_on=['Winner_Cluster','Loser_Cluster'],how='left')
#sub['Win_Rate']*=10
sub['Cluster_Win_Ratio']=(sub['Win_Rate']-0.5)*2
#sub=sub.drop_duplicates(ignore_index=True)
print(sub.shape)


Xsub = sub[features].values
dtest = xgb.DMatrix(Xsub)


sub_models = []
for i in range(repeat_cv):
    print(f"Fold repeater {i}")
    sub_models.append(
        xgb.train(
          params = param,
          dtrain = dtrain,
          num_boost_round = int(iteration_counts[i] * 1.05),
          verbose_eval = 50
        )
    )


sub_preds = []
for i in range(repeat_cv):
    sub_preds.append(np.clip(spline_model[i](np.clip(sub_models[i].predict(dtest),-30,30)),0.025,0.975))
    
sub["Pred"] = pd.DataFrame(sub_preds).mean(axis=0)
"""
sub.loc[(sub.T1_seed==1) & (sub.T2_seed==16), 'Pred'] = 1.0
sub.loc[(sub.T1_seed==2) & (sub.T2_seed==15), 'Pred'] = 1.0
sub.loc[(sub.T1_seed==3) & (sub.T2_seed==14), 'Pred'] = 1.0
sub.loc[(sub.T1_seed==4) & (sub.T2_seed==13), 'Pred'] = 1.0
sub.loc[(sub.T1_seed==16) & (sub.T2_seed==1), 'Pred'] = 0.0
sub.loc[(sub.T1_seed==15) & (sub.T2_seed==2), 'Pred'] = 0.0
sub.loc[(sub.T1_seed==14) & (sub.T2_seed==3), 'Pred'] = 0.0
sub.loc[(sub.T1_seed==13) & (sub.T2_seed==4), 'Pred'] = 0.0
"""
#sub[['ID','Pred']].to_csv("submission.csv", index = None)
#sub_men=sub[['ID','Pred']]


sub


sub[~sub.Seed_diff.isna()]['Pred'].hist(bins=50)


MenSouthSuper=['Auburn','Michigan St']
MenEastSuper=['Duke']#+['Alabama','Arizona']
MenMidwestSuper=['Tennessee','Houston']#+['Clemson']
#MenWestSuper=['Florida',"St John's",'Texas Tech']


def search_conference(x):
    return seeds[(seeds.Season==2025)&(seeds.TeamID==x)]['Seed'].apply(lambda x:x[0]).values.tolist()

def MSuperTeam(submission,super_teams):
    team_id=[int(mteams[mteams.TeamName==_].TeamID) for _ in super_teams]
    conf=search_conference(team_id[0])

    for _ in team_id:
        for i in range(len(submission)):
            if submission.loc[i,'T1_TeamID']==_:
                t2=submission.loc[i,'T2_TeamID']
                if search_conference(t2)==conf and t2 not in team_id:
                    submission.loc[i,'Pred']=1.0
                    print(submission.loc[i,'ID'])

            if submission.loc[i,'T2_TeamID']==_:
                t1=submission.loc[i,'T1_TeamID']
                if search_conference(t1)==conf and t1 not in team_id:
                    submission.loc[i,'Pred']=0.0
                    print(submission.loc[i,'ID'])
    print()
    return submission    


sub=MSuperTeam(sub,MenSouthSuper)
sub=MSuperTeam(sub,MenEastSuper)
sub=MSuperTeam(sub,MenMidwestSuper)
#sub=MSuperTeam(sub,MenWestSuper)


WSpokane1Super=['UCLA','LSU','NC State']
WSpokane4Super=['Connecticut','USC']
WBirmingham1Super=['South Carolina']
WBirmingham3Super=['Texas','TCU','Notre Dame']


def WSuperTeam(submission,super_teams):
    team_id=[int(wteams[wteams.TeamName==_].TeamID) for _ in super_teams]

    for _ in team_id:
        conf=search_conference(_)
        for i in range(len(submission)):
            if submission.loc[i,'T1_TeamID']==_:
                t2=submission.loc[i,'T2_TeamID']
                if search_conference(t2)==conf and t2 not in team_id:
                    submission.loc[i,'Pred']=1.0
                    print(submission.loc[i,'ID'])

            if submission.loc[i,'T2_TeamID']==_:
                t1=submission.loc[i,'T1_TeamID']
                if search_conference(t1)==conf and t1 not in team_id:
                    submission.loc[i,'Pred']=0.0
                    print(submission.loc[i,'ID'])
    print()
    return submission    


#sub=WSuperTeam(sub,WSpokane1Super)
sub=WSuperTeam(sub,WSpokane4Super)
sub=WSuperTeam(sub,WBirmingham1Super)
#sub=WSuperTeam(sub,WBirmingham3Super)


sub[~sub.Seed_diff.isna()]['Pred'].hist(bins=50)


sub[["ID","Pred"]].to_csv('submission.csv',index=None)


if not SUB:
    tourney_results = pd.concat([
        pd.read_csv(DATA_PATH + "MNCAATourneyDetailedResults.csv"),
        pd.read_csv(DATA_PATH + "WNCAATourneyDetailedResults.csv"),
    ], ignore_index=True)
    tourney_data=prepare_data(tourney_results)
    
    true=pd.DataFrame()
    true["ID"] = tourney_data["Season"].astype(str) + "_" + tourney_data["T1_TeamID"].astype(str) + "_" + tourney_data["T2_TeamID"].astype(str)
    true["Win"] = (tourney_data["T1_Score"] > tourney_data["T2_Score"]).astype(int)


if not SUB:
    test=sub[(sub.Season==2023)][['ID','Pred']].reset_index(drop=True)
    scoring=test.merge(true,on="ID")
    print("{:.6f}".format(brier_score_loss(scoring['Win'].values,scoring['Pred'].values)))
        
    test=sub[(sub.Season==2024)][['ID','Pred']].reset_index(drop=True)
    scoring=test.merge(true,on="ID")
    print("{:.6f}".format(brier_score_loss(scoring['Win'].values,scoring['Pred'].values)))
    
    test=sub[(sub.Season<=2024)&(sub.Season>=2021)][['ID','Pred']].reset_index(drop=True)
    scoring=test.merge(true,on="ID")
    print("{:.6f}".format(brier_score_loss(scoring['Win'].values,scoring['Pred'].values)))




