import pandas as pd
import numpy as np
from pathlib import Path
import os
import requests
import json
from tqdm.auto import tqdm
import datetime
import time
import glob
import collections
import polars as pl 
import time


# this data is taken from leaderboard
teams_data = [
    {'team_name': 'linrock', 'submission_id': 42838867},
    {'team_name': 'Approvers', 'submission_id': 42838539},
    {'team_name': 'Fix the bugs?', 'submission_id': 42819500},
    {'team_name': 'km', 'submission_id': 42806580},
    {'team_name': 'SSE', 'submission_id': 42664972},
    {'team_name': 'A', 'submission_id': 42834571},
    {'team_name': 'Noggenfogger', 'submission_id': 42837841},
    {'team_name': 'nagiss', 'submission_id': 42836820},
    {'team_name': 'nodchip', 'submission_id': 42813724},
    {'team_name': 'James Day', 'submission_id': 42820996},
    {'team_name': 'Oleg Trott', 'submission_id': 42610265},
    {'team_name': 'ymg_aq', 'submission_id': 42819813},
    {'team_name': 'Ascalon', 'submission_id': 42834303},
    {'team_name': 'Niboshi', 'submission_id': 42838968},
    {'team_name': 'Diogo&Vitaly', 'submission_id': 42823526},
    {'team_name': 'Blue', 'submission_id': 42839014},
    {'team_name': 'Avirence', 'submission_id': 42838122},
    {'team_name': 'baellouf', 'submission_id': 42830269},
    {'team_name': 'FooBar', 'submission_id': 42175209},
    {'team_name': 'rn5f107s2', 'submission_id': 42785368}
]


META_DIR = Path("../input/meta-kaggle/")
BASE_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/"
GET_URL = BASE_URL + "GetEpisodeReplay"


%%time

def get_competition_id_by_submission(sample_submission_id: int) -> int:
    agents_df = pl.scan_csv(META_DIR / "EpisodeAgents.csv", schema_overrides={'Reward':pl.Float32})
    sample_episode_id = agents_df.filter(pl.col("SubmissionId")==sample_submission_id).collect()["EpisodeId"].unique()[0]
    episodes_df = pl.scan_csv(META_DIR / "Episodes.csv")
    competition_id = episodes_df.filter(pl.col("Id")==sample_episode_id).collect()["CompetitionId"].unique()[0]
    print(f"{competition_id=}")
    return competition_id


sample_submission_id = 42823526
COMPETITION_ID = get_competition_id_by_submission(sample_submission_id)
COMPETITION_ID


# COMPETITION_ID = 86524  # FIDE & Google Efficient Chess AI Challenge


%%time

episodes_df = pl.scan_csv(META_DIR / "Episodes.csv")
episodes_df = (
    episodes_df
    .filter(pl.col('CompetitionId')==COMPETITION_ID)
    .with_columns(
        pl.col("CreateTime").str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False),
        pl.col("EndTime").str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False),
    )
    .sort("Id")
    .collect()
)
print(f'Episodes.csv: {len(episodes_df)} rows.')
episodes_df


%%time

agents_df = pl.scan_csv(
    META_DIR / "EpisodeAgents.csv", 
    schema_overrides={'Reward':pl.Float32, 'UpdatedConfidence': pl.Float32, 'UpdatedScore': pl.Float32}
)

agents_df = (
    agents_df
    .filter(pl.col("EpisodeId").is_in(episodes_df['Id'].to_list()))
    .with_columns([
        pl.when(pl.col("InitialConfidence") == "")
        .then(None)
        .otherwise(pl.col("InitialConfidence"))
        .cast(pl.Float64)
        .alias("InitialConfidence"),
        
        pl.when(pl.col("InitialScore") == "")
        .then(None)
        .otherwise(pl.col("InitialScore"))
        .cast(pl.Float64)
        .alias("InitialScore")])
    .collect()
)
print(f'EpisodeAgents.csv: {len(agents_df)} rows.')
agents_df


def load_episode(episode_id: int) -> dict:
    time.sleep(0.3)
    # request
    replay = requests.post(GET_URL, json = {"episodeId": int(episode_id)})
        
    replay = replay.json()
    return replay


last_n_episodes = 500

results = []
for team_data in tqdm(teams_data):
    team_name = team_data['team_name']
    submission_id = team_data['submission_id']

    
    team_episode_stats = []
    for episode_id in agents_df.filter(pl.col('SubmissionId') == submission_id)[-last_n_episodes:]['EpisodeId'].to_list():
        replay = load_episode(episode_id)
        is_white = replay['info']['TeamNames'].index(team_name) == 0
        team_episode_stats.append({'is_white': is_white})
    
    df_team_episode_stats = pd.DataFrame(team_episode_stats)
    
    
    result = {
        'team_name': team_name,
        'submission_id': submission_id,
        'is_white_share': df_team_episode_stats['is_white'].mean(),
        'last_n_episodes': len(df_team_episode_stats)
    }
    results.append(result)

df_results = pd.DataFrame(results)
df_results


# for discussion table
print(df_results.to_markdown())




