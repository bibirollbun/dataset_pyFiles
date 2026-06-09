# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





DATA_PATH = '/kaggle/input/march-machine-learning-mania-2025/'




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from scipy.interpolate import UnivariateSpline
from sklearn import preprocessing
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import KFold
from tqdm import tqdm

pd.set_option("display.max_column", 200)
# print(os.listdir("../input"))
xgb.__version__ # I used '1.2.0-SNAPSHOT'


import os
MTeamname = pd.read_csv(DATA_PATH + "MTeamSpellings.csv")
WTeamname = pd.read_csv(DATA_PATH + "WTeamSpellings.csv")
Melo = pd.read_csv("/kaggle/input/warrennolan-elo/Melo.csv")
Welo = pd.read_csv("/kaggle/input/welo-2025/WELO.csv")



print(Welo.head())



import re

def normalize_team_name(name):
    return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()



MTeamname["Normalized"] = MTeamname["TeamNameSpelling"].apply(normalize_team_name)
WTeamname["Normalized"] = WTeamname["TeamNameSpelling"].apply(normalize_team_name)

Melo["Normalized"] = Melo["Team"].apply(normalize_team_name)
Welo["Normalized"] = Welo["Team"].apply(normalize_team_name)


# 'ELO Delta' ì»¬ëŸ¼ ì‚­ì œ
if 'ELO Delta' in Melo.columns:
    Melo.drop(columns=['ELO Delta'], inplace=True)
if 'ELO Delta' in Welo.columns:
    Welo.drop(columns=['ELO Delta'], inplace=True)



MTeamname.rename(columns=lambda x: x.strip(), inplace=True)
Melo.rename(columns=lambda x: x.strip(), inplace=True)
Welo.rename(columns=lambda x: x.strip(), inplace=True)

# ë�°ì�´í„° í™•ì�¸ (ì˜¤ë¥˜ê°€ ë°œìƒ�í•˜ëŠ”ì§€ í…ŒìŠ¤íŠ¸)
print(MTeamname.tail(5))
print(Welo.tail(5))


Melo = pd.merge(
    Melo, 
    MTeamname[['TeamID', 'Normalized']], 
    on="Normalized", 
    how="left"
)

Welo = pd.merge(
    Welo, 
    WTeamname[['TeamID', 'Normalized']], 
    on="Normalized", 
    how="left"
)
Melo["TeamID"] = Melo["TeamID"].fillna(0).astype(int)  # ë˜�ëŠ” astype('Int64') ì‚¬ìš© ê°€ëŠ¥
Welo["TeamID"] = Welo["TeamID"].fillna(0).astype(int)


Melo[Melo["TeamID"]==0]



import pandas as pd
from fuzzywuzzy import process

# fau & utaì�˜ Normalized ê°’ì�„ ë³€ê²½
Melo.loc[Melo["Normalized"] == "fau", "Normalized"] = "florida atlantic"
Melo.loc[Melo["Normalized"] == "uta", "Normalized"] = "texas arlington"
Melo.loc[Melo["Normalized"] == "southeastmissouri", "Normalized"] = "se missouri st"
#  ë°œê²¬ë�˜ì§€ ì•Šì�€ íŒ€(TeamIDê°€ 0ì�¸ íŒ€) ë¦¬ìŠ¤íŠ¸
unmatched_teams = Melo[Melo["TeamID"] == 0]["Normalized"].tolist()

# ê¸°ì¤€ì�´ ë�˜ëŠ” ê¸°ì¡´ íŒ€ ë¦¬ìŠ¤íŠ¸ (MTeamnameì�˜ Normalized ì»¬ëŸ¼)
existing_teams = MTeamname["Normalized"].tolist()

# ìœ ì‚¬ë�„ ê¸°ë°˜ ì��ë�™ ë§¤ì¹­
matched_results = []
for team in unmatched_teams:
    best_match, score = process.extractOne(team, existing_teams)
    matched_results.append((team, best_match, score))

# ë§¤ì¹­ ê²°ê³¼ë¥¼ DataFrameìœ¼ë¡œ ë³€í™˜
matched_df = pd.DataFrame(matched_results, columns=["Unmatched", "Matched", "Score"])

#  ê¸°ì¡´ íŒ€ ë�°ì�´í„°ì™€ ID ê°€ì ¸ì˜¤ê¸°
matched_df = matched_df.merge(MTeamname[["Normalized", "TeamID"]], left_on="Matched", right_on="Normalized", how="left")
print(matched_df)
#Meloì—� ì—…ë�°ì�´íŠ¸ ì �ìš©
Melo = Melo.merge(matched_df[["Unmatched", "TeamID"]], left_on="Normalized", right_on="Unmatched", how="left", suffixes=("", "_new"))
Melo["TeamID"] = Melo["TeamID_new"].fillna(Melo["TeamID"]).astype(int)
Melo.drop(columns=["Unmatched", "TeamID_new"], inplace=True)  # ë¶ˆí•„ìš”í•œ ì»¬ëŸ¼ ì œê±°




Welo[Welo["TeamID"] == 0]


import pandas as pd
from fuzzywuzzy import process

# fau & utaì�˜ Normalized ê°’ì�„ ë³€ê²½
Welo.loc[Welo["Normalized"] == "fau", "Normalized"] = "florida atlantic"
Welo.loc[Welo["Normalized"] == "uta", "Normalized"] = "texas arlington"
Welo.loc[Welo["Normalized"] == "southeastmissouri", "Normalized"] = "se missouri st"
#  ë°œê²¬ë�˜ì§€ ì•Šì�€ íŒ€(TeamIDê°€ 0ì�¸ íŒ€) ë¦¬ìŠ¤íŠ¸
unmatched_teams = Welo[Welo["TeamID"] == 0]["Normalized"].tolist()

# ê¸°ì¤€ì�´ ë�˜ëŠ” ê¸°ì¡´ íŒ€ ë¦¬ìŠ¤íŠ¸ (MTeamnameì�˜ Normalized ì»¬ëŸ¼)
existing_teams = WTeamname["Normalized"].tolist()

# ìœ ì‚¬ë�„ ê¸°ë°˜ ì��ë�™ ë§¤ì¹­
matched_results = []
for team in unmatched_teams:
    best_match, score = process.extractOne(team, existing_teams)
    matched_results.append((team, best_match, score))

# ë§¤ì¹­ ê²°ê³¼ë¥¼ DataFrameìœ¼ë¡œ ë³€í™˜
matched_df = pd.DataFrame(matched_results, columns=["Unmatched", "Matched", "Score"])

#  ê¸°ì¡´ íŒ€ ë�°ì�´í„°ì™€ ID ê°€ì ¸ì˜¤ê¸°
matched_df = matched_df.merge(WTeamname[["Normalized", "TeamID"]], left_on="Matched", right_on="Normalized", how="left")
print(matched_df)
#Meloì—� ì—…ë�°ì�´íŠ¸ ì �ìš©
Welo = Welo.merge(matched_df[["Unmatched", "TeamID"]], left_on="Normalized", right_on="Unmatched", how="left", suffixes=("", "_new"))
Welo["TeamID"] = Welo["TeamID_new"].fillna(Welo["TeamID"]).astype(int)
Welo.drop(columns=["Unmatched", "TeamID_new"], inplace=True)  # ë¶ˆí•„ìš”í•œ ì»¬ëŸ¼ ì œê±°







elo_df = pd.concat([Melo, Welo], ignore_index=True)






# ë�°ì�´í„° ë¡œë“œ
sub = pd.read_csv('../input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv')

# ë¨¼ì € T1_TeamID, T2_TeamID, Seasonì�„ ìƒ�ì„±í•´ì•¼ í•¨
sub["Season"] = sub["ID"].apply(lambda x: x[:4]).astype(int)
sub["T1_TeamID"] = sub["ID"].apply(lambda x: x[5:9]).astype(int)
sub["T2_TeamID"] = sub["ID"].apply(lambda x: x[10:14]).astype(int)



elo_df.tail(5)


sub = sub.merge(
    elo_df[['TeamID', 'ELO']].rename(columns={"ELO": "T1_ELO"}), 
    left_on="T1_TeamID", right_on="TeamID", how="left"
).drop(columns=["TeamID"])

sub = sub.merge(
    elo_df[['TeamID', 'ELO']].rename(columns={"ELO": "T2_ELO"}), 
    left_on="T2_TeamID", right_on="TeamID", how="left"
).drop(columns=["TeamID"])

print(sub.head())  # ìµœì¢… í™•ì�¸



print("T1_ELO NaN í–‰:")
print(sub[sub["T1_ELO"].isna()])

print("T2_ELO NaN í–‰:")
print(sub[sub["T2_ELO"].isna()])



missing_t1 = sub[~sub["T1_TeamID"].isin(elo_df["TeamID"])]
print(f"ğŸš¨ T1_TeamIDê°€ elo_dfì—� ì—†ëŠ” ê°œìˆ˜: {len(missing_t1)}")



# elo_dfì—� ì—†ëŠ” T1_TeamID ëª©ë¡�
missing_t1_ids = list(set(sub["T1_TeamID"]) - set(elo_df["TeamID"]))

# elo_dfì—� ì—†ëŠ” T2_TeamID ëª©ë¡�
missing_t2_ids = list(set(sub["T2_TeamID"]) - set(elo_df["TeamID"]))

print("ğŸš¨ elo_dfì—� ì—†ëŠ” T1_TeamID ê³ ìœ ê°’ ë¦¬ìŠ¤íŠ¸:", missing_t1_ids)
print("ğŸš¨ elo_dfì—� ì—†ëŠ” T2_TeamID ê³ ìœ ê°’ ë¦¬ìŠ¤íŠ¸:", missing_t2_ids)




# ELO ì°¨ì�´ë¥¼ ì�´ìš©í•´ Pred ê°’ì�„ ì—…ë�°ì�´íŠ¸í•˜ëŠ” í•¨ìˆ˜ ì •ì�˜
def update_pred_by_elo(df):
    """
    ELO ì°¨ì�´ë¥¼ ê¸°ë°˜ìœ¼ë¡œ Pred ê°’ì�„ ì—…ë�°ì�´íŠ¸í•˜ëŠ” í•¨ìˆ˜
    - ELO ì°¨ì�´ê°€ í�´ìˆ˜ë¡� ìŠ¹ë¥ (Pred)ì�´ ë†’ì•„ì§�
    - ë¡œì§€ìŠ¤í‹± í•¨ìˆ˜ ê¸°ë°˜ìœ¼ë¡œ ìŠ¹ë¥ ì�„ ê³„ì‚°í•˜ì—¬ ë°˜ì˜�
    """
    import numpy as np
    
    # ELO ì°¨ì�´ ê³„ì‚° (T1ì�´ T2ë³´ë‹¤ ë†’ì�€ ê²½ìš° ì–‘ìˆ˜, ë‚®ì�€ ê²½ìš° ì�Œìˆ˜)
    df["ELO_Diff"] = df["T1_ELO"] - df["T2_ELO"]
    
    # ë¡œì§€ìŠ¤í‹± ë³€í™˜ì�„ ì�´ìš©í•˜ì—¬ ìŠ¹ë¥ (Pred) ê³„ì‚°
    df["Pred"] = 1 / (1 + np.exp(-df["ELO_Diff"] / 400))  # 400ì�€ ì�¼ë°˜ì �ì�¸ ELO ìŠ¤ì¼€ì�¼ë§�
    
    return df

# ë�°ì�´í„° ì—…ë�°ì�´íŠ¸ ì �ìš©
updated_df = update_pred_by_elo(sub)

updated_df



updated_df



# IDì™€ Pred ì»¬ëŸ¼ë§Œ ì¶”ì¶œí•œ ë�°ì�´í„° ìƒ�ì„±
id_pred_df = updated_df[["ID", "Pred"]]

# CSV íŒŒì�¼ë¡œ ì €ì�¥
csv_file_path = "id_pred.csv"  # ì›�í•˜ëŠ” ê²½ë¡œë¡œ ë³€ê²½ ê°€ëŠ¥
id_pred_df.to_csv(csv_file_path, index=False)

print(f" CSV íŒŒì�¼ì�´ ì €ì�¥ë�˜ì—ˆìŠµë‹ˆë‹¤: {csv_file_path}")


















































































































































































