# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install nfisis
from scipy.stats import poisson
from nfisis.fuzzy import NewMamdaniRegressor, NTSK
from nfisis.genetic import GEN_NMR, GEN_NTSK
from nfisis.ensemble import R_NMR, R_NTSK
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#---------------------------------
# Men
#---------------------------------

# Import dataset of match results
compact_results_M = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv")

# print(compact_results_M.head())

# Import dataset of statistics
detailed_results_M = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv")

# print(detailed_results_M.head())

window = 5
# Print specific match
season = 2003
wteamid = 1104
lteamid = 1328

filtered_df1 = compact_results_M[
    (compact_results_M['Season'] == season) &
    (compact_results_M['WTeamID'] == wteamid) &
    (compact_results_M['LTeamID'] == lteamid)
]

# print(filtered_df1)

filtered_df2 = detailed_results_M[
    (detailed_results_M['Season'] == season) &
    (detailed_results_M['WTeamID'] == wteamid) &
    (detailed_results_M['LTeamID'] == lteamid)
]

print(filtered_df2)

# Check dataset

# Step 1: Sort by 'Season' and 'DayNum'
compact_results_M_sorted = compact_results_M.sort_values(by=['Season', 'DayNum'])

# Step 2: Check for NaN values
nan_check = compact_results_M_sorted.isna().sum()

# Display the sorted DataFrame
# print(compact_results_M_sorted)

# Display the count of NaN values in each column
# print(nan_check)


# Step 1: Sort by 'Season' and 'DayNum'
detailed_results_M_sorted = detailed_results_M.sort_values(by=['Season', 'DayNum'])

# Step 2: Check for NaN values
nan_check = detailed_results_M_sorted.isna().sum()

# Display the sorted DataFrame
# print(detailed_results_M_sorted)

# Display the count of NaN values in each column
# print(nan_check)

# Merge the two DataFrames on the specified columns using an inner join
merged_df = pd.merge(compact_results_M.drop(columns=['WScore', 'LScore', 'WLoc', 'NumOT']), detailed_results_M, 
                     on=['Season', 'DayNum', 'WTeamID', 'LTeamID'], 
                     how='inner')

# Define the mapping dictionary
mapping_dict = {'A': -1, 'N': 0, 'H': 1}

# Apply the mapping to the column
merged_df['WLoc'] = merged_df['WLoc'].map(mapping_dict)

# Diff columns 1
merged_df["Score_diff1"] = merged_df["WScore"] - merged_df["LScore"]
merged_df["FGM_diff1"] = merged_df["WFGM"] - merged_df["LFGM"]
merged_df["FGA_diff1"] = merged_df["WFGA"] - merged_df["LFGA"]
merged_df["FGM3_diff1"] = merged_df["WFGM3"] - merged_df["LFGM3"]
merged_df["FGA3_diff1"] = merged_df["WFGA3"] - merged_df["LFGA3"]
merged_df["FTA_diff1"] = merged_df["WFTA"] - merged_df["LFTA"]
merged_df["OR_diff1"] = merged_df["WOR"] - merged_df["LOR"]
merged_df["DR_diff1"] = merged_df["WDR"] - merged_df["LDR"]
merged_df["Ast_diff1"] = merged_df["WAst"] - merged_df["LAst"]
merged_df["TO_diff1"] = merged_df["WTO"] - merged_df["LTO"]
merged_df["Stl_diff1"] = merged_df["WStl"] - merged_df["LStl"]
merged_df["Blk_diff1"] = merged_df["WBlk"] - merged_df["LBlk"]
merged_df["PF_diff1"] = merged_df["WPF"] - merged_df["LPF"]

# Diff columns2
merged_df["Score_diff2"] = merged_df["LScore"] - merged_df["WScore"]
merged_df["FGM_diff2"] = merged_df["LFGM"] - merged_df["WFGM"]
merged_df["FGA_diff2"] = merged_df["LFGA"] - merged_df["WFGA"]
merged_df["FGM3_diff2"] = merged_df["LFGM3"] - merged_df["WFGM3"]
merged_df["FGA3_diff2"] = merged_df["LFGA3"] - merged_df["WFGA3"]
merged_df["FTA_diff2"] = merged_df["LFTA"] - merged_df["WFTA"]
merged_df["OR_diff2"] = merged_df["LOR"] - merged_df["WOR"]
merged_df["DR_diff2"] = merged_df["LDR"] - merged_df["WDR"]
merged_df["Ast_diff2"] = merged_df["LAst"] - merged_df["WAst"]
merged_df["TO_diff2"] = merged_df["LTO"] - merged_df["WTO"]
merged_df["Stl_diff2"] = merged_df["LStl"] - merged_df["WStl"]
merged_df["Blk_diff2"] = merged_df["LBlk"] - merged_df["WBlk"]
merged_df["PF_diff2"] = merged_df["LPF"] - merged_df["WPF"]

merged_df[['MEAN_WFGM_1', 'MEAN_WFGA_1', 'MEAN_WFGM3_1', 'MEAN_WFGA3_1', 'MEAN_WFTM_1', 'MEAN_WFTA_1', 'MEAN_WOR_1', 'MEAN_WDR_1',
       'MEAN_WAst_1', 'MEAN_WTO_1', 'MEAN_WStl_1', 'MEAN_WBlk_1', 'MEAN_WPF_1', 'MEAN_LFGM_1', 'MEAN_LFGA_1', 'MEAN_LFGM3_1', 'MEAN_LFGA3_1',
       'MEAN_LFTM_1', 'MEAN_LFTA_1', 'MEAN_LOR_1', 'MEAN_LDR_1', 'MEAN_LAst_1', 'MEAN_LTO_1', 'MEAN_LStl_1', 'MEAN_LBlk_1', 'MEAN_LPF_1',
       'MEAN_Score_diff1', 'MEAN_FGM_diff1', 'MEAN_FGA_diff1', 'MEAN_FGM3_diff1', 'MEAN_FGA3_diff1',
       'MEAN_FTA_diff1', 'MEAN_OR_diff1', 'MEAN_DR_diff1', 'MEAN_Ast_diff1', 'MEAN_TO_diff1',
       'MEAN_Stl_diff1', 'MEAN_Blk_diff1', 'MEAN_PF_diff1']] = merged_df.groupby('WTeamID')[['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
       'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',
       'Score_diff1', 'FGM_diff1', 'FGA_diff1', 'FGM3_diff1', 'FGA3_diff1',
       'FTA_diff1', 'OR_diff1', 'DR_diff1', 'Ast_diff1', 'TO_diff1',
       'Stl_diff1', 'Blk_diff1', 'PF_diff1']].rolling(window=window, min_periods=window, closed="left").mean().reset_index(level=0, drop=True)

merged_df[['MEAN_WFGM_2', 'MEAN_WFGA_2', 'MEAN_WFGM3_2', 'MEAN_WFGA3_2', 'MEAN_WFTM_2', 'MEAN_WFTA_2', 'MEAN_WOR_2', 'MEAN_WDR_2',
       'MEAN_WAst_2', 'MEAN_WTO_2', 'MEAN_WStl_2', 'MEAN_WBlk_2', 'MEAN_WPF_2', 'MEAN_LFGM_2', 'MEAN_LFGA_2', 'MEAN_LFGM3_2', 'MEAN_LFGA3_2',
       'MEAN_LFTM_2', 'MEAN_LFTA_2', 'MEAN_LOR_2', 'MEAN_LDR_2', 'MEAN_LAst_2', 'MEAN_LTO_2', 'MEAN_LStl_2', 'MEAN_LBlk_2', 'MEAN_LPF_2',
       'MEAN_Score_diff2', 'MEAN_FGM_diff2', 'MEAN_FGA_diff2', 'MEAN_FGM3_diff2', 'MEAN_FGA3_diff2',
       'MEAN_FTA_diff2', 'MEAN_OR_diff2', 'MEAN_DR_diff2', 'MEAN_Ast_diff2', 'MEAN_TO_diff2',
       'MEAN_Stl_diff2', 'MEAN_Blk_diff2', 'MEAN_PF_diff2']] = merged_df.groupby('LTeamID')[['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
       'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',
       'Score_diff2', 'FGM_diff2',
       'FGA_diff2', 'FGM3_diff2', 'FGA3_diff2', 'FTA_diff2', 'OR_diff2',
       'DR_diff2', 'Ast_diff2', 'TO_diff2', 'Stl_diff2', 'Blk_diff2',
       'PF_diff2']].rolling(window=window, min_periods=window, closed="left").mean().reset_index(level=0, drop=True)

# Display the resulting merged DataFrame
# print(len(merged_df.columns))

# Prepared dataframe
prepared_df = merged_df.copy()

columns_to_clean = ['MEAN_WFGM_1', 'MEAN_WFGA_1', 'MEAN_WFGM3_1', 'MEAN_WFGA3_1', 'MEAN_WFTM_1', 'MEAN_WFTA_1', 'MEAN_WOR_1', 'MEAN_WDR_1',
       'MEAN_WAst_1', 'MEAN_WTO_1', 'MEAN_WStl_1', 'MEAN_WBlk_1', 'MEAN_WPF_1', 'MEAN_LFGM_1', 'MEAN_LFGA_1', 'MEAN_LFGM3_1', 'MEAN_LFGA3_1',
       'MEAN_LFTM_1', 'MEAN_LFTA_1', 'MEAN_LOR_1', 'MEAN_LDR_1', 'MEAN_LAst_1', 'MEAN_LTO_1', 'MEAN_LStl_1', 'MEAN_LBlk_1', 'MEAN_LPF_1',
       'MEAN_Score_diff1', 'MEAN_FGM_diff1', 'MEAN_FGA_diff1', 'MEAN_FGM3_diff1', 'MEAN_FGA3_diff1',
       'MEAN_FTA_diff1', 'MEAN_OR_diff1', 'MEAN_DR_diff1', 'MEAN_Ast_diff1', 'MEAN_TO_diff1',
       'MEAN_Stl_diff1', 'MEAN_Blk_diff1', 'MEAN_PF_diff1', 'MEAN_WFGM_2', 'MEAN_WFGA_2', 'MEAN_WFGM3_2', 'MEAN_WFGA3_2', 'MEAN_WFTM_2', 'MEAN_WFTA_2', 'MEAN_WOR_2', 'MEAN_WDR_2',
       'MEAN_WAst_2', 'MEAN_WTO_2', 'MEAN_WStl_2', 'MEAN_WBlk_2', 'MEAN_WPF_2', 'MEAN_LFGM_2', 'MEAN_LFGA_2', 'MEAN_LFGM3_2', 'MEAN_LFGA3_2',
       'MEAN_LFTM_2', 'MEAN_LFTA_2', 'MEAN_LOR_2', 'MEAN_LDR_2', 'MEAN_LAst_2', 'MEAN_LTO_2', 'MEAN_LStl_2', 'MEAN_LBlk_2', 'MEAN_LPF_2',
       'MEAN_Score_diff2', 'MEAN_FGM_diff2', 'MEAN_FGA_diff2', 'MEAN_FGM3_diff2', 'MEAN_FGA3_diff2',
       'MEAN_FTA_diff2', 'MEAN_OR_diff2', 'MEAN_DR_diff2', 'MEAN_Ast_diff2', 'MEAN_TO_diff2',
       'MEAN_Stl_diff2', 'MEAN_Blk_diff2', 'MEAN_PF_diff2']  # Replace with your column names

# Clean df
prepared_df = prepared_df.dropna(subset=columns_to_clean)

# print(prepared_df.columns)

# Separate into train and test
df_train_M, df_test_M = train_test_split(prepared_df, test_size=0.2, shuffle=False)

# Columns to put together
top_l = ['MEAN_WFGM_1', 'MEAN_WFGA_1', 'MEAN_WFGM3_1', 'MEAN_WFGA3_1', 'MEAN_WFTM_1', 'MEAN_WFTA_1', 'MEAN_WOR_1', 'MEAN_WDR_1',
       'MEAN_WAst_1', 'MEAN_WTO_1', 'MEAN_WStl_1', 'MEAN_WBlk_1', 'MEAN_WPF_1', 'MEAN_LFGM_1', 'MEAN_LFGA_1', 'MEAN_LFGM3_1', 'MEAN_LFGA3_1',
       'MEAN_LFTM_1', 'MEAN_LFTA_1', 'MEAN_LOR_1', 'MEAN_LDR_1', 'MEAN_LAst_1', 'MEAN_LTO_1', 'MEAN_LStl_1', 'MEAN_LBlk_1', 'MEAN_LPF_1',
       'MEAN_Score_diff1', 'MEAN_FGM_diff1', 'MEAN_FGA_diff1', 'MEAN_FGM3_diff1', 'MEAN_FGA3_diff1',
       'MEAN_FTA_diff1', 'MEAN_OR_diff1', 'MEAN_DR_diff1', 'MEAN_Ast_diff1', 'MEAN_TO_diff1',
       'MEAN_Stl_diff1', 'MEAN_Blk_diff1', 'MEAN_PF_diff1']

bottom_l = ['MEAN_WFGM_2', 'MEAN_WFGA_2', 'MEAN_WFGM3_2', 'MEAN_WFGA3_2', 'MEAN_WFTM_2', 'MEAN_WFTA_2', 'MEAN_WOR_2', 'MEAN_WDR_2',
       'MEAN_WAst_2', 'MEAN_WTO_2', 'MEAN_WStl_2', 'MEAN_WBlk_2', 'MEAN_WPF_2', 'MEAN_LFGM_2', 'MEAN_LFGA_2', 'MEAN_LFGM3_2', 'MEAN_LFGA3_2',
       'MEAN_LFTM_2', 'MEAN_LFTA_2', 'MEAN_LOR_2', 'MEAN_LDR_2', 'MEAN_LAst_2', 'MEAN_LTO_2', 'MEAN_LStl_2', 'MEAN_LBlk_2', 'MEAN_LPF_2',
       'MEAN_Score_diff2', 'MEAN_FGM_diff2', 'MEAN_FGA_diff2', 'MEAN_FGM3_diff2', 'MEAN_FGA3_diff2',
       'MEAN_FTA_diff2', 'MEAN_OR_diff2', 'MEAN_DR_diff2', 'MEAN_Ast_diff2', 'MEAN_TO_diff2',
       'MEAN_Stl_diff2', 'MEAN_Blk_diff2', 'MEAN_PF_diff2']

# Prepare X_train
X_train1_M = df_train_M[top_l].values
X_train2_M = df_train_M[bottom_l].values
#X_train = np.vstack((top, bottom))
#print(X_train.shape)

# Prepare y_train
y_train1 = df_train_M['WScore'].values
y_train2 = df_train_M['LScore'].values
#print(y_train.shape)

# Run the model
#model1_M = NewMamdaniRegressor()
#model1_M = NTSK()
#model1_M = GEN_NMR()
#model1_M = GEN_NTSK()
#model1_M = R_NMR(n_estimators=1, error_metric="MAE", parallel_processing=-1)
model1_M = R_NTSK(n_estimators=1, error_metric="MAE", parallel_processing=-1)
model1_M.fit(X_train1_M, y_train1)

#model2_M = NewMamdaniRegressor()
#model2_M = NTSK()
#model2_M = GEN_NMR()
#model2_M = GEN_NTSK()
#model2_M = R_NMR(n_estimators=1, error_metric="MAE", parallel_processing=-1)
model2_M = R_NTSK(n_estimators=1, error_metric="MAE", parallel_processing=-1)
model2_M.fit(X_train2_M, y_train2)

# Reset index
df_test_M = df_test_M.reset_index(drop=True)

# Prepare X_train
X_test_1_M = df_test_M[top_l].values
y_pred_1_M = model1_M.predict(X_test_1_M)
df_test_M["y_pred_1_M"] = y_pred_1_M

X_test_2_M = df_test_M[bottom_l].values
y_pred_2_M = model2_M.predict(X_test_2_M)
df_test_M["y_pred_2_M"] = y_pred_2_M

# print(df_test_M)

# Probability
def prob_p1_less_p2(row):
    lambda1, lambda2 = row['y_pred_1_M'], row['y_pred_2_M']
    k_max = int(max(lambda1, lambda2) * 2)  # Define um limite baseado nos lambdas
    return sum(poisson.pmf(k, lambda1) * (1 - poisson.cdf(k, lambda2)) for k in range(k_max))

# Aplica a função a cada linha do DataFrame
df_test_M['P_P1_less_P2'] = df_test_M.apply(prob_p1_less_p2, axis=1)

# print(df_test_M[['y_pred_1_M', 'y_pred_2_M', 'P_P1_less_P2']].head())

# Verificar se há valores NaN antes da conversão
df_test_M[['Season', 'WTeamID', 'LTeamID']] = df_test_M[['Season', 'WTeamID', 'LTeamID']].fillna(0)

# Converter colunas para inteiro antes de formatar
df_test_M['Season'] = df_test_M['Season'].astype(int)
df_test_M['WTeamID'] = df_test_M['WTeamID'].astype(int)
df_test_M['LTeamID'] = df_test_M['LTeamID'].astype(int)

# print(df_test_M["Season"])

# print(df_test_M[['Season', 'WTeamID', 'LTeamID']].head(10))
# print(df_test_M[['Season', 'WTeamID', 'LTeamID']].isna().sum())

# for i in range(df_test_M.shape[0]):
#     print(f'{df_test_M.loc[i,"Season"]}, {type(df_test_M.loc[i,"Season"])}')

# Criar a coluna ID no formato correto
df_test_M['ID'] = df_test_M.apply(lambda row: f"{row['Season']:.0f}_{min(row['WTeamID'], row['LTeamID']):.0f}_{max(row['WTeamID'], row['LTeamID']):.0f}", axis=1)

# Ajustar a coluna Pred de acordo com a ordem dos times
df_test_M['Pred'] = df_test_M.apply(lambda row: row['P_P1_less_P2'] if row['WTeamID'] < row['LTeamID'] else 1 - row['P_P1_less_P2'], axis=1)



#---------------------------------
# Women
#---------------------------------

# Import dataset of match results
compact_results_W = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv")

# print(compact_results_W.head())

# Import dataset of statistics
detailed_results_W = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonDetailedResults.csv")

# print(detailed_results_W.head())

window = 5
# Print specific match
season = 2003
wteamid = 1104
lteamid = 1328

filtered_df1 = compact_results_W[
    (compact_results_W['Season'] == season) &
    (compact_results_W['WTeamID'] == wteamid) &
    (compact_results_W['LTeamID'] == lteamid)
]

# print(filtered_df1)

filtered_df2 = detailed_results_W[
    (detailed_results_W['Season'] == season) &
    (detailed_results_W['WTeamID'] == wteamid) &
    (detailed_results_W['LTeamID'] == lteamid)
]

# print(filtered_df2)

# Check dataset

# Step 1: Sort by 'Season' and 'DayNum'
compact_results_W_sorted = compact_results_W.sort_values(by=['Season', 'DayNum'])

# Step 2: Check for NaN values
nan_check = compact_results_W_sorted.isna().sum()

# Display the sorted DataFrame
# print(compact_results_W_sorted)

# Display the count of NaN values in each column
# print(nan_check)


# Step 1: Sort by 'Season' and 'DayNum'
detailed_results_W_sorted = detailed_results_W.sort_values(by=['Season', 'DayNum'])

# Step 2: Check for NaN values
nan_check = detailed_results_W_sorted.isna().sum()

# Display the sorted DataFrame
# print(detailed_results_W_sorted)

# Display the count of NaN values in each column
# print(nan_check)

# Merge the two DataFrames on the specified columns using an inner join
merged_df = pd.merge(compact_results_W.drop(columns=['WScore', 'LScore', 'WLoc', 'NumOT']), detailed_results_W, 
                     on=['Season', 'DayNum', 'WTeamID', 'LTeamID'], 
                     how='inner')

# Define the mapping dictionary
mapping_dict = {'A': -1, 'N': 0, 'H': 1}

# Apply the mapping to the column
merged_df['WLoc'] = merged_df['WLoc'].map(mapping_dict)

# Diff columns 1
merged_df["Score_diff1"] = merged_df["WScore"] - merged_df["LScore"]
merged_df["FGM_diff1"] = merged_df["WFGM"] - merged_df["LFGM"]
merged_df["FGA_diff1"] = merged_df["WFGA"] - merged_df["LFGA"]
merged_df["FGM3_diff1"] = merged_df["WFGM3"] - merged_df["LFGM3"]
merged_df["FGA3_diff1"] = merged_df["WFGA3"] - merged_df["LFGA3"]
merged_df["FTA_diff1"] = merged_df["WFTA"] - merged_df["LFTA"]
merged_df["OR_diff1"] = merged_df["WOR"] - merged_df["LOR"]
merged_df["DR_diff1"] = merged_df["WDR"] - merged_df["LDR"]
merged_df["Ast_diff1"] = merged_df["WAst"] - merged_df["LAst"]
merged_df["TO_diff1"] = merged_df["WTO"] - merged_df["LTO"]
merged_df["Stl_diff1"] = merged_df["WStl"] - merged_df["LStl"]
merged_df["Blk_diff1"] = merged_df["WBlk"] - merged_df["LBlk"]
merged_df["PF_diff1"] = merged_df["WPF"] - merged_df["LPF"]

# Diff columns2
merged_df["Score_diff2"] = merged_df["LScore"] - merged_df["WScore"]
merged_df["FGM_diff2"] = merged_df["LFGM"] - merged_df["WFGM"]
merged_df["FGA_diff2"] = merged_df["LFGA"] - merged_df["WFGA"]
merged_df["FGM3_diff2"] = merged_df["LFGM3"] - merged_df["WFGM3"]
merged_df["FGA3_diff2"] = merged_df["LFGA3"] - merged_df["WFGA3"]
merged_df["FTA_diff2"] = merged_df["LFTA"] - merged_df["WFTA"]
merged_df["OR_diff2"] = merged_df["LOR"] - merged_df["WOR"]
merged_df["DR_diff2"] = merged_df["LDR"] - merged_df["WDR"]
merged_df["Ast_diff2"] = merged_df["LAst"] - merged_df["WAst"]
merged_df["TO_diff2"] = merged_df["LTO"] - merged_df["WTO"]
merged_df["Stl_diff2"] = merged_df["LStl"] - merged_df["WStl"]
merged_df["Blk_diff2"] = merged_df["LBlk"] - merged_df["WBlk"]
merged_df["PF_diff2"] = merged_df["LPF"] - merged_df["WPF"]

merged_df[['MEAN_WFGM_1', 'MEAN_WFGA_1', 'MEAN_WFGM3_1', 'MEAN_WFGA3_1', 'MEAN_WFTM_1', 'MEAN_WFTA_1', 'MEAN_WOR_1', 'MEAN_WDR_1',
       'MEAN_WAst_1', 'MEAN_WTO_1', 'MEAN_WStl_1', 'MEAN_WBlk_1', 'MEAN_WPF_1', 'MEAN_LFGM_1', 'MEAN_LFGA_1', 'MEAN_LFGM3_1', 'MEAN_LFGA3_1',
       'MEAN_LFTM_1', 'MEAN_LFTA_1', 'MEAN_LOR_1', 'MEAN_LDR_1', 'MEAN_LAst_1', 'MEAN_LTO_1', 'MEAN_LStl_1', 'MEAN_LBlk_1', 'MEAN_LPF_1',
       'MEAN_Score_diff1', 'MEAN_FGM_diff1', 'MEAN_FGA_diff1', 'MEAN_FGM3_diff1', 'MEAN_FGA3_diff1',
       'MEAN_FTA_diff1', 'MEAN_OR_diff1', 'MEAN_DR_diff1', 'MEAN_Ast_diff1', 'MEAN_TO_diff1',
       'MEAN_Stl_diff1', 'MEAN_Blk_diff1', 'MEAN_PF_diff1']] = merged_df.groupby('WTeamID')[['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
       'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',
       'Score_diff1', 'FGM_diff1', 'FGA_diff1', 'FGM3_diff1', 'FGA3_diff1',
       'FTA_diff1', 'OR_diff1', 'DR_diff1', 'Ast_diff1', 'TO_diff1',
       'Stl_diff1', 'Blk_diff1', 'PF_diff1']].rolling(window=window, min_periods=window, closed="left").mean().reset_index(level=0, drop=True)

merged_df[['MEAN_WFGM_2', 'MEAN_WFGA_2', 'MEAN_WFGM3_2', 'MEAN_WFGA3_2', 'MEAN_WFTM_2', 'MEAN_WFTA_2', 'MEAN_WOR_2', 'MEAN_WDR_2',
       'MEAN_WAst_2', 'MEAN_WTO_2', 'MEAN_WStl_2', 'MEAN_WBlk_2', 'MEAN_WPF_2', 'MEAN_LFGM_2', 'MEAN_LFGA_2', 'MEAN_LFGM3_2', 'MEAN_LFGA3_2',
       'MEAN_LFTM_2', 'MEAN_LFTA_2', 'MEAN_LOR_2', 'MEAN_LDR_2', 'MEAN_LAst_2', 'MEAN_LTO_2', 'MEAN_LStl_2', 'MEAN_LBlk_2', 'MEAN_LPF_2',
       'MEAN_Score_diff2', 'MEAN_FGM_diff2', 'MEAN_FGA_diff2', 'MEAN_FGM3_diff2', 'MEAN_FGA3_diff2',
       'MEAN_FTA_diff2', 'MEAN_OR_diff2', 'MEAN_DR_diff2', 'MEAN_Ast_diff2', 'MEAN_TO_diff2',
       'MEAN_Stl_diff2', 'MEAN_Blk_diff2', 'MEAN_PF_diff2']] = merged_df.groupby('LTeamID')[['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
       'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',
       'Score_diff2', 'FGM_diff2',
       'FGA_diff2', 'FGM3_diff2', 'FGA3_diff2', 'FTA_diff2', 'OR_diff2',
       'DR_diff2', 'Ast_diff2', 'TO_diff2', 'Stl_diff2', 'Blk_diff2',
       'PF_diff2']].rolling(window=window, min_periods=window, closed="left").mean().reset_index(level=0, drop=True)

# Display the resulting merged DataFrame
# print(len(merged_df.columns))

# Prepared dataframe
prepared_df = merged_df.copy()

columns_to_clean = ['MEAN_WFGM_1', 'MEAN_WFGA_1', 'MEAN_WFGM3_1', 'MEAN_WFGA3_1', 'MEAN_WFTM_1', 'MEAN_WFTA_1', 'MEAN_WOR_1', 'MEAN_WDR_1',
       'MEAN_WAst_1', 'MEAN_WTO_1', 'MEAN_WStl_1', 'MEAN_WBlk_1', 'MEAN_WPF_1', 'MEAN_LFGM_1', 'MEAN_LFGA_1', 'MEAN_LFGM3_1', 'MEAN_LFGA3_1',
       'MEAN_LFTM_1', 'MEAN_LFTA_1', 'MEAN_LOR_1', 'MEAN_LDR_1', 'MEAN_LAst_1', 'MEAN_LTO_1', 'MEAN_LStl_1', 'MEAN_LBlk_1', 'MEAN_LPF_1',
       'MEAN_Score_diff1', 'MEAN_FGM_diff1', 'MEAN_FGA_diff1', 'MEAN_FGM3_diff1', 'MEAN_FGA3_diff1',
       'MEAN_FTA_diff1', 'MEAN_OR_diff1', 'MEAN_DR_diff1', 'MEAN_Ast_diff1', 'MEAN_TO_diff1',
       'MEAN_Stl_diff1', 'MEAN_Blk_diff1', 'MEAN_PF_diff1', 'MEAN_WFGM_2', 'MEAN_WFGA_2', 'MEAN_WFGM3_2', 'MEAN_WFGA3_2', 'MEAN_WFTM_2', 'MEAN_WFTA_2', 'MEAN_WOR_2', 'MEAN_WDR_2',
       'MEAN_WAst_2', 'MEAN_WTO_2', 'MEAN_WStl_2', 'MEAN_WBlk_2', 'MEAN_WPF_2', 'MEAN_LFGM_2', 'MEAN_LFGA_2', 'MEAN_LFGM3_2', 'MEAN_LFGA3_2',
       'MEAN_LFTM_2', 'MEAN_LFTA_2', 'MEAN_LOR_2', 'MEAN_LDR_2', 'MEAN_LAst_2', 'MEAN_LTO_2', 'MEAN_LStl_2', 'MEAN_LBlk_2', 'MEAN_LPF_2',
       'MEAN_Score_diff2', 'MEAN_FGM_diff2', 'MEAN_FGA_diff2', 'MEAN_FGM3_diff2', 'MEAN_FGA3_diff2',
       'MEAN_FTA_diff2', 'MEAN_OR_diff2', 'MEAN_DR_diff2', 'MEAN_Ast_diff2', 'MEAN_TO_diff2',
       'MEAN_Stl_diff2', 'MEAN_Blk_diff2', 'MEAN_PF_diff2']  # Replace with your column names

# Clean df
prepared_df = prepared_df.dropna(subset=columns_to_clean)

# print(prepared_df.columns)

# Separate into train and test
df_train_W, df_test_W = train_test_split(prepared_df, test_size=0.2, shuffle=False)

# Columns to put together
top_l = ['MEAN_WFGM_1', 'MEAN_WFGA_1', 'MEAN_WFGM3_1', 'MEAN_WFGA3_1', 'MEAN_WFTM_1', 'MEAN_WFTA_1', 'MEAN_WOR_1', 'MEAN_WDR_1',
       'MEAN_WAst_1', 'MEAN_WTO_1', 'MEAN_WStl_1', 'MEAN_WBlk_1', 'MEAN_WPF_1', 'MEAN_LFGM_1', 'MEAN_LFGA_1', 'MEAN_LFGM3_1', 'MEAN_LFGA3_1',
       'MEAN_LFTM_1', 'MEAN_LFTA_1', 'MEAN_LOR_1', 'MEAN_LDR_1', 'MEAN_LAst_1', 'MEAN_LTO_1', 'MEAN_LStl_1', 'MEAN_LBlk_1', 'MEAN_LPF_1',
       'MEAN_Score_diff1', 'MEAN_FGM_diff1', 'MEAN_FGA_diff1', 'MEAN_FGM3_diff1', 'MEAN_FGA3_diff1',
       'MEAN_FTA_diff1', 'MEAN_OR_diff1', 'MEAN_DR_diff1', 'MEAN_Ast_diff1', 'MEAN_TO_diff1',
       'MEAN_Stl_diff1', 'MEAN_Blk_diff1', 'MEAN_PF_diff1']

bottom_l = ['MEAN_WFGM_2', 'MEAN_WFGA_2', 'MEAN_WFGM3_2', 'MEAN_WFGA3_2', 'MEAN_WFTM_2', 'MEAN_WFTA_2', 'MEAN_WOR_2', 'MEAN_WDR_2',
       'MEAN_WAst_2', 'MEAN_WTO_2', 'MEAN_WStl_2', 'MEAN_WBlk_2', 'MEAN_WPF_2', 'MEAN_LFGM_2', 'MEAN_LFGA_2', 'MEAN_LFGM3_2', 'MEAN_LFGA3_2',
       'MEAN_LFTM_2', 'MEAN_LFTA_2', 'MEAN_LOR_2', 'MEAN_LDR_2', 'MEAN_LAst_2', 'MEAN_LTO_2', 'MEAN_LStl_2', 'MEAN_LBlk_2', 'MEAN_LPF_2',
       'MEAN_Score_diff2', 'MEAN_FGM_diff2', 'MEAN_FGA_diff2', 'MEAN_FGM3_diff2', 'MEAN_FGA3_diff2',
       'MEAN_FTA_diff2', 'MEAN_OR_diff2', 'MEAN_DR_diff2', 'MEAN_Ast_diff2', 'MEAN_TO_diff2',
       'MEAN_Stl_diff2', 'MEAN_Blk_diff2', 'MEAN_PF_diff2']

# Prepare X_train
X_train1_W = df_train_W[top_l].values
X_train2_W = df_train_W[bottom_l].values
#X_train = np.vstack((top, bottom))
# print(X_train.shape)

# Prepare y_train
y_train1 = df_train_W['WScore'].values
y_train2 = df_train_W['LScore'].values
#print(y_train.shape)

# Run the model
#model1_W = NewMamdaniRegressor()
#model1_W = NTSK()
#model1_W = GEN_NMR()
#model1_W = GEN_NTSK()
#model1_W = R_NMR(n_estimators=1, error_metric="MAE", parallel_processing=-1)
model1_W = R_NTSK(n_estimators=1, error_metric="MAE", parallel_processing=-1)
model1_W.fit(X_train1_W, y_train1)

#model2_W = NewMamdaniRegressor()
#model2_W = NTSK()
#model2_W = GEN_NMR()
#model2_W = GEN_NTSK()
#model2_W = R_NMR(n_estimators=1, error_metric="MAE", parallel_processing=-1)
model2_W = R_NTSK(n_estimators=1, error_metric="MAE", parallel_processing=-1)
model2_W.fit(X_train2_W, y_train2)

# Reset index
df_test_W = df_test_W.reset_index(drop=True)

# Prepare X_train
X_test_1_W = df_test_W[top_l].values
y_pred_1_W = model1_W.predict(X_test_1_W)
df_test_W["y_pred_1_W"] = y_pred_1_W

X_test_2_W = df_test_W[bottom_l].values
y_pred_2_W = model2_W.predict(X_test_2_W)
df_test_W["y_pred_2_W"] = y_pred_2_W

# print(df_test_W)

# Probability
def prob_p1_less_p2(row):
    lambda1, lambda2 = row['y_pred_1_W'], row['y_pred_2_W']
    k_Wax = int(max(lambda1, lambda2) * 2)  # Define um limite baseado nos lambdas
    return sum(poisson.pmf(k, lambda1) * (1 - poisson.cdf(k, lambda2)) for k in range(k_Wax))

# Aplica a função a cada linha do DataFrame
df_test_W['P_P1_less_P2'] = df_test_W.apply(prob_p1_less_p2, axis=1)

# print(df_test_W[['y_pred_1_W', 'y_pred_2_W', 'P_P1_less_P2']].head())

# Verificar se há valores NaN antes da conversão
df_test_W[['Season', 'WTeamID', 'LTeamID']] = df_test_W[['Season', 'WTeamID', 'LTeamID']].fillna(0)

# Converter colunas para inteiro antes de formatar
df_test_W['Season'] = df_test_W['Season'].astype(int)
df_test_W['WTeamID'] = df_test_W['WTeamID'].astype(int)
df_test_W['LTeamID'] = df_test_W['LTeamID'].astype(int)

# print(df_test_W["Season"])

# print(df_test_W[['Season', 'WTeamID', 'LTeamID']].head(10))
# print(df_test_W[['Season', 'WTeamID', 'LTeamID']].isna().sum())

# for i in range(df_test_W.shape[0]):
#     print(f'{df_test_W.loc[i,"Season"]}, {type(df_test_W.loc[i,"Season"])}')

# Criar a coluna ID no formato correto
df_test_W['ID'] = df_test_W.apply(lambda row: f"{row['Season']:.0f}_{min(row['WTeamID'], row['LTeamID']):.0f}_{max(row['WTeamID'], row['LTeamID']):.0f}", axis=1)

# Ajustar a coluna Pred de acordo com a ordem dos times
df_test_W['Pred'] = df_test_W.apply(lambda row: row['P_P1_less_P2'] if row['WTeamID'] < row['LTeamID'] else 1 - row['P_P1_less_P2'], axis=1)



#---------------------------------
# Save
#---------------------------------

# Selecionar apenas as colunas necessárias
df_submission = pd.concat([df_test_M[['ID', 'Pred']], df_test_W[['ID', 'Pred']]], ignore_index=True)

# Salvar como arquivo de texto
df_submission.to_csv('SampleSubmissionStage1.csv', index=False)

print("Arquivo 'SampleSubmissionStage1.txt' salvo com sucesso!")

