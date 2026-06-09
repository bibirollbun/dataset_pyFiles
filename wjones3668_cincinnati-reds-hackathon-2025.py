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


df_players = pd.read_csv("/kaggle/input/reds-hackathon-2025/lahman_people.csv")
df_savant = pd.read_excel("/kaggle/input/reds-hackathon-2025/codebook.xlsx", sheet_name='Baseball Savant')
df_lahman = pd.read_excel("/kaggle/input/reds-hackathon-2025/codebook.xlsx", sheet_name='Lahman')
df_pitch = pd.read_csv("/kaggle/input/reds-hackathon-2025/savant_data_2021_2023.csv", nrows=20)   # 2GB
df_sub = pd.read_csv("/kaggle/input/reds-hackathon-2025/sample_submission.csv")

df_players.tail(10)


df_savant.tail(10)


for i in df_lahman.index:
    print(df_lahman["Column Name"].loc[i]," ==> ",df_lahman["Defintion"].loc[i])


df_pitch.tail(20)


df_players[df_players["player_mlb_id"]=="c7c83eaa9fe8da2f81c5fce172059af61448b3e7"]







