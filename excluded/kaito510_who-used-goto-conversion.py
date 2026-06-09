import numpy as np
import pandas as pd
import os
dummy_df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')


#Set parameters
fileNames_list = [12,16,24,32,42,48,56,64,72,78,85,88,93,96,98,100,105,108,112,114,118,120,122,124,125,126]
#The first two snapshots, `after4.csv` and `after8.csv` were unused because they may not be reliable. More details about this:
#https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/569248
fileNames_prefix = '/kaggle/input/march-madness-2025-live-lb-raw-snapshots/after'
fileNames_suffix = '.csv'


#Merge all snapshots
for fileNumber in fileNames_list:
    file_path = fileNames_prefix + str(fileNumber) + fileNames_suffix
    curr_df = pd.read_csv(file_path)
    curr_df['Score'] = curr_df['Score']*fileNumber
    try:
        merged_df = pd.merge(merged_df, curr_df[['TeamId', 'Score']], on='TeamId', how='inner', suffixes=('', fileNumber))
    except:
        merged_df = curr_df[['TeamId', 'TeamName', 'Score']]
    merged_df = merged_df.rename(columns={'Score': 'Score'+str(fileNumber)})

#Compute change in total brier-score after each update
diffs_merged_df = merged_df.iloc[:,2:].diff(axis=1)
diffs_merged_df.columns = [col+'Change' for col in diffs_merged_df.columns]
diffs_merged_df = pd.concat([merged_df[['TeamId', 'TeamName']], diffs_merged_df], axis=1)


#Get my change in total brier-score after each update
target_row = diffs_merged_df[diffs_merged_df['TeamName'] == 'kaito510'] #Get my score changes
target_row = pd.Series(target_row.iloc[0,3:], index=diffs_merged_df.columns[3:]) #Convert my score changes to a series 
#with TeamId, TeamName and first numerical column removed


#Get all participants change in total brier-score after each update
diffs_merged_df = diffs_merged_df.iloc[:,3:] #Remove TeamId and TeamName


#For each participant, compute number of updates with exact same score change as my solution
merged_df['IdenticalWithMe'] = (diffs_merged_df.subtract(target_row).abs() < 1e-3).sum(axis=1)
identical_df = merged_df.sort_values(by='IdenticalWithMe', ascending=False)[['TeamId','TeamName','IdenticalWithMe']].copy()
threshold = identical_df.loc[76,'IdenticalWithMe'] #76 is index of participant "baellouf", who used my solution as their basis (https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/572528)
display_df = identical_df[identical_df['IdenticalWithMe'] >= threshold]
print('Number of participants that submitted my solution (at least partially):', len(display_df))
identical_df.to_csv('identicalWithMe.csv') #Export


#For each participant, compute correlation with my submission
merged_df['CorrelationWithMe'] = diffs_merged_df.corrwith(target_row, axis=1) #Compute correlation of my score changes to everyone else
corr_df = merged_df.sort_values(by='CorrelationWithMe', ascending=False)[['TeamId','TeamName','CorrelationWithMe']].copy()
threshold = corr_df.loc[621,'CorrelationWithMe'] #621 is index of participant "Robert Hatch", who blended my solution (https://www.kaggle.com/competitions/march-machine-learning-mania-2025/discussion/570587#3162716)
display_df = corr_df[corr_df['CorrelationWithMe'] >= threshold]
print('Number of participants that blended my solution:', len(display_df))
corr_df.to_csv('corrWithMe.csv') #Export


#Find participants that either submitted (at least partially) or blended my solution
either_df = pd.merge(identical_df[['TeamId', 'TeamName', 'IdenticalWithMe']], corr_df[['TeamId', 'CorrelationWithMe']], on='TeamId', how='inner', suffixes=('', ''))
corrThreshold = corr_df.loc[621,'CorrelationWithMe']
identicalThreshold = identical_df.loc[76,'IdenticalWithMe']
either_df = either_df[(either_df['CorrelationWithMe'] >= corrThreshold) | (either_df['IdenticalWithMe'] >= identicalThreshold)]
print('Number of participants that used my solution:', len(either_df))
either_df.to_csv('eitherWithMe.csv') #Export


#Filter to Medalists
afterFinal_df = pd.read_csv('/kaggle/input/march-madness-2025-live-lb-raw-snapshots/afterFinal.csv')
medalists_df = afterFinal_df[afterFinal_df['Rank'] <= 172]
print('Medalists that used my solution:')
pd.set_option('display.max_rows', 150)
either_df = either_df[either_df['TeamId'].isin(medalists_df['TeamId'].values)]
display(either_df)
print('Number of medalists that used my solution:', len(either_df))
either_df.to_csv('medalistsWithMe.csv') #Export


#Filter to Gold Medalists
gold_df = medalists_df[medalists_df['Rank'] <= 13]
print('Gold Medalists that used my solution:')
either_df = either_df[either_df['TeamId'].isin(gold_df['TeamId'].values)]
display(either_df)
print('Number of gold medalists that used my solution:', len(either_df))
either_df.to_csv('goldsWithMe.csv') #Export

