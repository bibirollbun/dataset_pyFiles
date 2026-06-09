import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_squared_error
from glob import glob


cf = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
cf.head()


combineFiles = glob('/kaggle/input/march-machine-learning-mania-2025/*Compact*')

combDfList = []

for cf in combineFiles:
    combDfList.append(pd.read_csv(cf))


comDf = pd.concat(combDfList)
comDf.head()


len(comDf)


unique_WTeam = comDf['WTeamID'].unique()
unique_LTeam = comDf['LTeamID'].unique()

# print(unique_WTeam,  unique_LTeam)
uniqueTeams = unique_WTeam + unique_LTeam
uniqueTeams = list(set(uniqueTeams))

# Declare unique teams
dfUniqueTeams = pd.DataFrame()
dfUniqueTeams['ID'] = uniqueTeams


idNo = 3104
win = comDf[comDf['WTeamID']==idNo]
win.head()

