# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

'''
import os

print(os.listdir('/kaggle/input'))

filepaths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        filepaths.append(os.path.join(dirname, filename))

#print(filepaths)
for filepath in filepaths:
    print(filepath)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Cek salah satu file (misal: tim pria)
teams_men = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')
print(teams_men.head())
print()
teams_men.info()
print()
teams_men.describe()

#Tabel teams_men 


#eksplorasi data pertandingan historis
games = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
print(games)
print()
games.info()
print()
games.describe()




#cek jumlah pertandingan permusim
#games['Season'].value_counts().sort_index()
#print()
print(games['Season'].value_counts().sort_index())
print()
games.info()


#Cek distribusi skor
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(games['WScore'], bins=30, kde=True)
plt.title('Distribusi Skor Tim Pemenang')
plt.xlabel('Winning Score')
plt.ylabel('Jumlah')
plt.show()


#games[~np.isfinite(games['WScore'])]
#Cek apakah ada nilai inf atau NaN
print(games['WScore'].isin([float('inf'), -float('inf')]).sum())
print()
print()
print(np.isinf(games[['WScore', 'LScore']]).sum())  # Cek nilai inf
print()
print(games.isna().sum()) # Cek nilai NaN


print(games.head())
print(teams_men.head())


print(games.head())


#menggabungkan dengan data tim untuk nama tim: Biar lebih mudah dibaca,
#gabungkan WTeamID dan LTeamID dengan data tim (teams_men).
games = games.merge(teams_men[['TeamID', 'TeamName']], left_on='WTeamID', right_on='TeamID', how='left').rename(columns={'TeamName': 'WinningTeam'})
games = games.merge(teams_men[['TeamID', 'TeamName']], left_on='LTeamID', right_on='TeamID', how='left').rename(columns={'TeamName': 'LosingTeam'})

games[['Season', 'DayNum', 'WinningTeam', 'WScore', 'LosingTeam', 'LScore']].head()




print(games)


#Hitung jumlah kemenangan dan kekalahan
#win_counts = games['WinningTeam'].value_counts().reset_index().rename(columns={'index': 'Team', 'WinningTeam': 'Wins'})
#loss_counts = games['LosingTeam'].value_counts().reset_index().rename(columns={'index': 'Team', 'LosingTeam': 'Losses'})
win_counts = games['WinningTeam'].value_counts().reset_index().rename(columns={'index': 'Team', 'WinningTeam': 'Team', 'count': 'Wins'})
loss_counts = games['LosingTeam'].value_counts().reset_index().rename(columns={'index': 'Team', 'LosingTeam': 'Team', 'count': 'Losses'})


print(win_counts.columns)
print(loss_counts.columns)

print(win_counts.head())
print(loss_counts.head())


#Gabungkan ke dua tabel
team_stats = pd.merge(win_counts, loss_counts, on='Team', how='outer').fillna(0)

#Hitung total pertandingan dan selisih skor rata-rata
games['ScoreDiff'] = games['WScore'] - games['LScore']
avg_score_diff = games.groupby('WinningTeam')['ScoreDiff'].mean().reset_index().rename(columns={'WinningTeam': 'Team', 'ScoreDiff': 'AvgWinMargin'})

#Gabungkan ke team_stats
team_stats = pd.merge(team_stats, avg_score_diff, on='Team', how='left').fillna(0)
team_stats.head()



import pandas as pd

#load data
seasons_men = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv')

print(seasons_men.head())
print()

seasons_men.info()
print()




#konversi kolom DayZero menjadi DateTime
seasons_men['DayZero'] =  pd.to_datetime(seasons_men['DayZero'], format='%m/%d/%Y')

print(seasons_men.dtypes)
print(seasons_men['DayZero'].head())
#print(np.diff(seasons_men['DayZero']))


#tambahkan kolom hari dalam seminggu
seasons_men['DayZero_Weekday'] = seasons_men['DayZero'].dt.day_name()
print(seasons_men.head())
print()


#Cek distribusi hari
print(seasons_men['DayZero_Weekday'].value_counts())


#Tambahkan kolom akhir musim (DayNum 154)
seasons_men['EndOfSeason'] = seasons_men['DayZero']+pd.to_timedelta(154, unit='D')

print(seasons_men[['Season','DayZero','EndOfSeason']].head())


print(seasons_men['Season'].value_counts().sort_index())


#cek missing values
print(seasons_men.isnull().sum())


#visualisasi distribusi seasons

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
sns.countplot(x='Season', data=seasons_men)

plt.xticks(rotation=45)
plt.xlabel('Musim')
plt.ylabel('Frekuensi')
plt.title('Distribusi Jumlah Season')

plt.show()


#Cek kolom Region
regions = ['RegionW', 'RegionX', 'RegionY', 'RegionZ']

#cek jumlah unik di tiap region
for region in regions:
    print(seasons_men[region].value_counts())


#menghitung jumlah kemenangan per tim
wins_per_team_season = games.groupby(['Season', 'WinningTeam']).size().reset_index(name='Wins')

#melihat distribusi kemenangan
plt.figure(figsize=(10,6))
sns.histplot(wins_per_team_season['Wins'], bins=30, kde=True)
plt.title('Distribusi Jumlah Kemenangan per Tim per Musim')
plt.xlabel('Jumlah Kemenangan')
plt.ylabel('Frekuensi')
plt.show()


#Rata-rata skor pemenang dan skor kalah per musim
avg_scores = games.groupby('Season')[['WScore','LScore']].mean().reset_index()

plt.figure(figsize=(10,6))
plt.plot(avg_scores['Season'], avg_scores['WScore'], marker='o', label='Rata-rata skor pemenang')
plt.plot(avg_scores['Season'], avg_scores['LScore'], marker='x', label='Rata-rata skor kalah')
plt.title('Rata-rata Skor Pemenang dan Kalah per Musim')
plt.xlabel('Musim')
plt.ylabel('Rata-rata Skor')
plt.legend()
plt.show()


#Menghitung jumlah pertandingan overtime per musim
ot_games=games[games['NumOT']>0].groupby('Season')['NumOT'].count().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Season', y='NumOT', data=ot_games)
plt.title('Frekuensi Pertandingan Overtime per Musim')
plt.xlabel('Musim')
plt.ylabel('Jumlah Pertandingan OT')
plt.xticks(rotation=45)
plt.show()


#del plt.xlabel
#del plt.ylabel

#reload modul matplotlib
import importlib
import matplotlib.pyplot as plt

importlib.reload(plt)




# Menghitung frekuensi overtime per musim
overtime_trend = games[games['NumOT'] > 0].groupby('Season')['NumOT'].count().reset_index()

# Visualisasi tren overtime
plt.figure(figsize=(10, 6))
plt.plot(overtime_trend['Season'], overtime_trend['NumOT'], marker='o', color='blue')
plt.title('Frekuensi Overtime per Musim')
plt.xlabel('Musim')
plt.ylabel('Jumlah Pertandingan dengan Overtime')
plt.grid(True)
plt.show()



#Distribusi Lokasi Kemenangan (Home, Away, Neutral)

plt.figure(figsize=(6,6))
sns.countplot(x='WLoc', data=games, order=['H', 'A', 'N'])
plt.title('Distribusi Lokasi Kemenangan')
plt.xlabel('Lokasi Kemenangan')
plt.ylabel('Jumlah Pertandingan')
plt.show()


#Tren Total Skor Per Musim
import matplotlib.pyplot as plt
#plt.xlabel = plt.axes().set_xlabel
#plt.ylabel = plt.axes().set_ylabel

games['TotalScore'] = games['WScore'] + games['LScore']
avg_total_score = games.groupby('Season')['TotalScore'].mean().reset_index()

plt.figure(figsize=(10,6))
plt.plot(avg_total_score['Season'], avg_total_score['TotalScore'], marker='o', color='purple')
plt.title('Tren Rata-rata Total Skor per Musim')
plt.xlabel('Musim')
plt.ylabel('Rata-rata Total Skor')
plt.show()


'xlabel' in locals(), 'xlabel' in globals()


# Melihat jumlah tim unik per musim (tim pemenang dan tim kalah)
teams_per_season = pd.concat([games[['Season', 'WTeamID']], games[['Season', 'LTeamID']]])
teams_per_season = teams_per_season.drop_duplicates().groupby('Season').nunique()

# Menampilkan jumlah tim unik per musim
teams_per_season.rename(columns={'WTeamID': 'UniqueTeams'}, inplace=True)
print(teams_per_season)

# Visualisasi jumlah tim unik per musim
plt.figure(figsize=(10, 6))
plt.plot(teams_per_season.index, teams_per_season['UniqueTeams'], marker='o', linestyle='-', color='blue')
plt.title('Jumlah Tim Unik yang Bertanding per Musim')
plt.xlabel('Musim')
plt.ylabel('Jumlah Tim Unik')
plt.grid(True)
plt.show()



# Gabungkan semua tim dari kolom WTeamID dan LTeamID
all_teams = pd.concat([
    games[['Season', 'WTeamID']].rename(columns={'WTeamID': 'TeamID'}),
    games[['Season', 'LTeamID']].rename(columns={'LTeamID': 'TeamID'})
])

# Hitung jumlah musim di mana setiap tim muncul
team_appearance_count = all_teams.groupby('TeamID')['Season'].nunique().reset_index()
team_appearance_count.columns = ['TeamID', 'TotalSeasons']

# Gabungkan dengan data nama tim
team_appearance_count = team_appearance_count.merge(teams_men[['TeamID', 'TeamName']], on='TeamID', how='left')

# Tampilkan tim dengan partisipasi terbanyak
team_appearance_count.sort_values(by='TotalSeasons', ascending=False).head(10)



# Hitung jumlah kemenangan dan kekalahan masing-masing tim
wins = games['WTeamID'].value_counts().reset_index()

print(wins)

wins.columns = ['TeamID', 'Wins']  #merubah nama kolom tbel wins

print(wins)
print()

losses = games['LTeamID'].value_counts().reset_index()
print(losses)
losses.columns = ['TeamID', 'Losses']
print(losses)

# Gabungkan data kemenangan dan kekalahan
team_performance = pd.merge(wins, losses, on='TeamID', how='outer').fillna(0)
print(team_performance)
print()

# Hitung total pertandingan dan win rate
team_performance['TotalGames'] = team_performance['Wins'] + team_performance['Losses']
team_performance['WinRate'] = team_performance['Wins'] / team_performance['TotalGames']

# Gabungkan dengan nama tim
team_performance = team_performance.merge(teams_men[['TeamID', 'TeamName']], on='TeamID', how='left')

# Urutkan berdasarkan win rate tertinggi
team_performance = team_performance.sort_values(by='WinRate', ascending=False)

# Lihat 10 tim dengan win rate tertinggi
team_performance.head(10)


import pandas as pd

#load data
seeds_men = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')

#cek struktur data
print(seeds_men.head())
print(seeds_men.info())
print(seeds_men['Seed'].unique())

#Ekstraksi region dan seed number
seeds_men['Region'] = seeds_men['Seed'].str[0] #Ambil huruf pertama dari region (W, X, Y, Z)
seeds_men['SeedNum'] = seeds_men['Seed'].str[1:3].astype(int) # ambil 2 angka berikutnya sebagai nomor seed
print(seeds_men.head())


#Cek distribusi seed
seed_distribution = seeds_men['SeedNum'].value_counts().sort_index()
print(seed_distribution)

#Cek jumlah tim per musim
teams_per_season = seeds_men.groupby('Season')['TeamID'].nunique()
print(teams_per_season)

#Cek frekuensi play-in games
play_in_games = seeds_men[seeds_men['Seed'].str.len() == 4]  #seed dengan panjang 4 berarti ada 'a' atau 'b'
print(play_in_games['Season'].value_counts())
print(play_in_games)

#Visualisasi jumlah tim per musim
import matplotlib.pyplot as plt

teams_per_season.plot(kind='bar', figsize=(12, 6), color='skyblue')
plt.title('Jumlah Tim di NCAA Tournament per Musim')
plt.xlabel('Musim')
plt.ylabel('Jumlah Tim')
plt.show()


#Pisahkan kolom Seed jadi region, seednum, dan playin
seeds_men['Region'] = seeds_men['Seed'].str[0]
seeds_men['SeedNum'] = seeds_men['Seed'].str[1:3].astype(int)
seeds_men['PlayIn'] = seeds_men['Seed'].str[3].fillna('')

#Cek hasil
print(seeds_men.head())


import seaborn as sns
plt.figure(figsize=(10,6))

#Distribusi seednum
sns.countplot(x='SeedNum', data=seeds_men, palette='viridis')
plt.title('Distribusi Seed Number')
plt.xlabel('Seed Number')
plt.ylabel('Jumlah Tim')
plt.show()


#Distribusi Region
plt.figure(figsize=(8,5))
sns.countplot(x='Region', data=seeds_men, palette='pastel')
plt.title('Distribusi Region')
plt.xlabel('Region')
plt.ylabel('Jumlah Tim')
plt.show()


#Gabungkan seed untuk tim pemenang (WTeamID)
games_with_seeds = games.merge(seeds_men, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
#print(games_with_seeds)
games_with_seeds.rename(columns={'Seed': 'WSeed'}, inplace=True)
#print(games_with_seeds)
games_with_seeds.drop('TeamID', axis=1, inplace=True)

#Cek hasil
#print(games_with_seeds)
games_with_seeds.head()


print(games_with_seeds['WSeed'].isna().sum())

#konfirmasi dengan melihat jumlah total baris
print(games_with_seeds.shape[0])


tourney_games = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
#/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv

tourney_games = tourney_games.merge(seeds_men, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_games.rename(columns={'Seed': 'WSeed'}, inplace=True)
tourney_games.drop('TeamID', axis=1, inplace=True)

tourney_games = tourney_games.merge(seeds_men, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_games.rename(columns={'Seed': 'LSeed'}, inplace=True)
tourney_games.drop('TeamID', axis=1, inplace=True)

#cek jumlah NaN
print(tourney_games.isna().sum())

#Hapus baris yang mengandung Nan (jika perlu)
tourney_games.dropna(subset=['WSeed', 'LSeed'], inplace=True)

#Cek data hasil pembersihan
print(tourney_games.shape)
tourney_games.head()


#Tambahkan kolom selisih seed
tourney_games['SeedDiff'] = tourney_games['SeedNum_y'] - tourney_games['SeedNum_x']

#cek distribusi SeedDiff
plt.figure(figsize=(8, 6))
sns.histplot(tourney_games['SeedDiff'], bins=20, kde=True)
plt.title('Distribusi Perbedaan Seed antara Pemenang dan Kalah')
plt.xlabel('SeedDiff (Seed Kalah - Seed Menang)')
plt.ylabel('Jumlah Pertandingan')
plt.show()

#cek tingkat kemenangan berdasarkan SeedDiff
win_rate_by_seed_diff = tourney_games['SeedDiff'].value_counts().sort_index()
print(win_rate_by_seed_diff)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load data
tourney_games = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
#'/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv'
print(tourney_games.head())
print(tourney_games.info())

#Cek missing values
print(tourney_games.isna().sum())

#statistik deskriptif
print(tourney_games.describe())


#Distribusi skor pemenang dan yang kalah
plt.figure(figsize=(8,5))
sns.histplot(tourney_games['WScore'], bins=30, kde=True, color='blue', label='Winning Score')
sns.histplot(tourney_games['LScore'], bins=30, kde=True, color='red', label='Losing Score')
plt.legend()
plt.title('Distribusi Skor Pemenang dan Kalah')
plt.xlabel('Score')
plt.ylabel('Frekuensi')

plt.show()


#Distribusi perbedaan skor (margin kemenangan)
tourney_games['ScoreDiff'] = tourney_games['WScore'] - tourney_games['LScore']

plt.figure(figsize=(8, 5))
sns.histplot(tourney_games['ScoreDiff'], bins=30, kde=True, color='purple')
plt.title('Distribusi Margin Kemenangan')
plt.xlabel('Perbedaan Skor (WScore - LScore)')
plt.ylabel('Frekuensi')
plt.show()




#Cek tren jumlah overtime (NumOT)
plt.figure(figsize=(8, 5))
sns.countplot(x='NumOT', data=tourney_games, palette='viridis')
plt.title('Distribusi Jumlah Overtime dalam Pertandingan')
plt.xlabel('Jumlah Overtime')
plt.ylabel('Jumlah Pertandingan')
plt.show()


#Analisis lokasi kemenangan (WLoc)
plt.figure(figsize=(5, 5,))
sns.countplot(x='WLoc', data=tourney_games, palette='coolwarm')
plt.title('Lokasi Kemenangan (WLoc)')
plt.xlabel('WLoc')
plt.ylabel('Jumlah Pertandingan')
plt.show()



#Analisis performa tim berdasarkan seed

#perlu penggabungan data seed terlebih dahulu
seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
print(seeds.head())

tourney_games = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
print(tourney_games.head())
#print(tourney_games.info())
#Cek missing values
#print(tourney_games.isna().sum())
#statistik deskriptif
#print(tourney_games.describe())
#Distribusi perbedaan skor (margin kemenangan)
tourney_games['ScoreDiff'] = tourney_games['WScore'] - tourney_games['LScore']



#gabung seed untuk tim pemenang 
tourney_games = tourney_games.merge(seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_games.rename(columns={'Seed': 'WSeed'}, inplace=True)
tourney_games.drop('TeamID', axis=1, inplace=True)
print(tourney_games.head())

#gabung seed untuk tim yang kalah
tourney_games = tourney_games.merge(seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
tourney_games.rename(columns={'Seed': 'LSeed'}, inplace=True)
tourney_games.drop('TeamID', axis=1, inplace=True)
print(tourney_games.head())

#Ekstrak angka seed saja tanpa (tanpa huruf region)
tourney_games['WSeedNum'] = tourney_games['WSeed'].str[1:3].astype(float)
tourney_games['LSeedNum'] = tourney_games['LSeed'].str[1:3].astype(float)

#cek distribusi seed pemenang
plt.figure(figsize=(8,5))
sns.histplot(tourney_games['WSeedNum'], bins=16, kde=True, color='green')
plt.title('Distribusi Seed Pemenang')
plt.xlabel('Seed Number')
plt.ylabel('Frekuensi')
plt.show()
#del tourney_games
print(tourney_games)




#Analisis perbedaan seed (SeedDif) dan hubungaanya dengan kemenangan
tourney_games['SeedDiff'] = tourney_games['WSeedNum'] - tourney_games['LSeedNum']

plt.figure(figsize=(8, 5))
sns.histplot(tourney_games['SeedDiff'], bins=30, kde=True, color='orange')
plt.title('Distribusi Perbedaan Seed antara Tim Pemenang dan Kalah')
plt.xlabel('Perbedaan Seed (WseedNum - LSeedNum)')
plt.ylabel('Frekuensi')
plt.show()


#Analisis distribusi seed

import seaborn as sns
import matplotlib.pyplot as plt

#ekstrak nomor seed saja (tanpa huruf region)
tourney_games['WSeedNum'] = tourney_games['WSeed'].str[1:3].astype(int)
tourney_games['LSeedNum'] = tourney_games['LSeed'].str[1:3].astype(int)

print(tourney_games.head())

#Distribusi seed pemenang
plt.figure(figsize=(10, 6))
sns.countplot(x='WSeedNum', data=tourney_games, order=sorted(tourney_games['WSeedNum'].unique()))
plt.title('Distribusi Seed Tim Pemenang')
plt.xlabel('Seed')
plt.ylabel('Jumlah Kemenangan')
plt.show()

#Distribusi seed tim yang kalah
plt.figure(figsize=(10, 6))
sns.countplot(x='LSeedNum', data=tourney_games, order=sorted(tourney_games['LSeedNum'].unique()))
plt.title('Distribusi Seed Tim Kalah')
plt.xlabel('Seed')
plt.ylabel('Jumlah Kekalahan')
plt.show()

#presentase kemenangan berdasarkan seed
win_rate_by_seed = tourney_games.groupby('WSeedNum').size() / (tourney_games.groupby('WSeedNum').size() + tourney_games.groupby('LSeedNum').size())
win_rate_by_seed.plot(kind='bar', figsize=(10,6), title='Win Rate Berdasarkan Seed', ylabel='Win Rate', xlabel='Seed')

plt.show()


#Analisis kejutan (upset)

#buat kolom apakah terjadi "upset" (tim dengan seed lebih tinggi menang)
tourney_games['upset'] = (tourney_games['WSeedNum'] > tourney_games['LSeedNum'])

#hitung frekuensi upset
upset_count = tourney_games['upset'].value_counts(normalize=True)
print(f'Persentase upset: {upset_count[True] * 100:.2f}%')
print(f'Persentase Non-Upset: {upset_count[False] * 100:.2f}%')

#visulisasi upset berdasarkan selisih seed
tourney_games['SeedDiff'] = tourney_games['WSeedNum']-tourney_games['LSeedNum']
plt.figure(figsize=(10,6))
sns.countplot(x='SeedDiff', hue='upset', data=tourney_games)
plt.title('Distribusi Kejutan Berdasarkan Selisih Seed')
plt.xlabel('Selisih Seed (WSeed-LSeed)')
plt.ylabel('Jumlah Pertandingan')
plt.legend(title='upset', labels=['Tidak', 'Ya'])
plt.show()


#Buat kolom selisih skor
tourney_games['ScoreDiff'] = tourney_games['WScore'] - tourney_games['LScore']

#visualisasi selisih skor berdasarkan seed pemenang
plt.figure(figsize=(10, 6))
sns.boxplot(x='WSeedNum', y='ScoreDiff', data=tourney_games)
plt.title('Distribusi Selisih Skor Berdasarkan Seed Pemenang')
plt.xlabel('Seed Pemenang')
plt.ylabel('Selisih Skor (WScore - LScore)')
plt.show()





#EDA dataset samplesubmissionstage1.csv

submission = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage1.csv')
print(submission.head())

#pecah kolom ID
submission[['Season', 'Team1', 'Team2']] = submission['ID'].str.split('_', expand=True)
print(submission.head())

submission['Season'] = submission['Season'].astype(int)
submission['Team1'] = submission['Team1'].astype(int)
submission['Team2'] = submission['Team2'].astype(int)
print(submission.head())


#GABUNGKAN DENGAN DATA LAIN
#Gabungkan seed untuk team1
#submission = submission.merge(seeds_men, left_on=['Season', 'Team1'], right_on=['Season', 'Team1'], how='left')   #error
submission = submission.merge(seeds_men, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
submission.rename(columns={'SeedNum': 'SeedNum_Team1'}, inplace=True)
submission.drop('TeamID', axis=1, inplace=True)

#Gabungkan seed untuk team2
#submission = submission.merge(seeds_men, left_on=['Season', 'Team2'], right_on=['Season', 'Team2'], how='left')
submission = submission.merge(seeds_men, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left')
submission.rename(columns={'SeedNum': 'SeedNum_Team2'}, inplace=True)
submission.drop('TeamID', axis=1, inplace=True)

submission.head()

'''
left_on dan right_on di fungsi merge() di Pandas dipakai buat menentukan kolom mana yang jadi "kunci" untuk penggabungan (join) antara dua DataFrame.

left_on → kolom di DataFrame pertama (submission dalam contoh tadi)
right_on → kolom di DataFrame kedua (seeds_men)
'''


#print(submission.columns)
#print(seeds_men.columns)


#Buat Fitur/kolom/variabel/field Tambahan
#Kolom perbedaan seed
submission['SeedDiff'] = submission['SeedNum_Team1'] - submission['SeedNum_Team2']
print(submission.head())

# (Nanti bisa tambah fitur lain seperti WinRate, ScoreDiff, dll.)


#Persiapkan DATA TRAINING
#4. Persiapkan Data Training
#Ambil data historis untuk melatih model.

#Data training
train = tourney_games.copy()

#fitur seedDiff
train['SeedDiff'] = train['WSeedNum'] - train['LSeedNum']

#target: 1 jika tim WTeamID menang, 0 jika sebaliknya
train['Result'] = 1

#Balik data untuk kekalahan (agar bisa prediksi dua arah)
train_flip = train.copy()
train_flip['SeedDiff'] = -train_flip['SeedDiff']
train_flip['Result'] = 0

#Gabung data asli dan yang dibalik
train = pd.concat([train, train_flip])

#fitur dan target
X = train[['SeedDiff']]
y = train['Result']

#print(X)
#print(y)


#5. Bangun Model ML
#Coba Logistic Regression dulu.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Model
model = LogisticRegression()
model.fit(X_train, y_train)

#validasi
y_pred = model.predict(X_val)
print(f'Accuracy: {accuracy_score(y_val, y_pred):.4f}')


#print(submission['SeedDiff'].isna().sum())

nan_rows =  submission[submission['SeedDiff'].isna()]
print(nan_rows.head())
#print(nan_rows.shape())


#Cek tim yang bermasalah
print(nan_rows[['ID']].head(100))


#Menggunakan model untuk prediksi hasil turnamen
#prediksi probabilitas
submission['Pred'] = model.predict_proba(submission[['SeedDiff']])[:, 1]

#pastikan Pred di renang 0-1
submission['Pred'] = submission['Pred'].clip(0, 1)

submission.head()


import pandas as pd

#load dataset
df = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv')
print(df.head())

#lihat ringkasan statistik
print(df.describe())

#cek kolom
print(df.info())

#Cek missing values
print(df.isnull().sum())


#Visualisasi distribusi skor tim pememang dan yang kalah

import seaborn as sns
import matplotlib.pyplot as plt

#hilangkan warning inf pandas
import numpy as np
df.replace([np.inf, -np.inf], np.nan, inplace=True)
#kalau masih muncul warning, sembunyikan saja
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

plt.figure(figsize=(10,5))
sns.histplot(df['WScore'], bins=30, kde=True, color='blue', label='Winning Team Score')
sns.histplot(df['LScore'], bins=30, kde=True, color='blue', label='Losing Team Score')
plt.legend()
plt.title('Distribusi Skor Tim Pemenang dan Kalah')
plt.show()


#Eksplorasi efisiensi tembakan

#efisiensi field goal dan three-point
df['WFG%'] = df['WFGM'] / df['WFGA']
df['LFG%'] = df['LFGM'] / df['LFGA']

df['W3P%'] = df['WFGM3'] / df['WFGA3']
df['L3P%'] = df['LFGM3'] / df['LFGA3']

print(df.head())

#visualisasi efisiensi tembakan
plt.figure(figsize=(10,5))
sns.kdeplot(df['WFG%'].dropna(), color='blue', label='Winning FG%')
sns.kdeplot(df['LFG%'].dropna(), color='red', label='Losing FG%')
plt.legend()
plt.title('Distribusi Field Goal Percentage Tim Pemenang dan Kalah')
plt.show()


#efisiensi free throw (FT%)
df['WFT%'] = df['WFTM'] / df['WFTA']
df['LFT%'] = df['LFTM'] / df['LFTA']

print(df.head())

#Visualisasi efisiensi free throw
plt.figure(figsize=(10,5))
sns.kdeplot(df['WFT%'].dropna(), color='blue', label='Winning FT%')
sns.kdeplot(df['LFT%'].dropna(), color='red', label='Losing FT%')
plt.legend()
plt.title('Distribusi Free Throw Percentage Tim Pemenang dan Kalah')
plt.show()


#Margin Kemenangan
df['ScoreDiff'] = df['WScore'] - df['LScore']
print(df.head())

#Visualisasi distribusi margin kemenangan
plt.figure(figsize=(10, 5))
sns.histplot(df['ScoreDiff'], bins=30, kde=True, color='green')
plt.title('Distribusi Margin Kemenangan')
plt.show()


#Statistik Rebound (Offensive & Defensive)
plt.figure(figsize=(10, 5))
sns.kdeplot(df['WOR'], color='blue', label='Winning Offensive Rebounds')
sns.kdeplot(df['LOR'], color='red', label='Losing Offensive Rebounds')
plt.legend()
plt.title('Distribusi Offensive Rebounds Tim Pemenang dan Kalah')

#Visualisasi defensive rebounds
plt.figure(figsize=(10, 5))
sns.kdeplot(df['WDR'], color='blue', label='Winning Defensive Rebounds')
sns.kdeplot(df['LDR'], color='red', label='Losing Defensive Rebounds')
plt.legend()
plt.title('Distribusi Defensive Rebounds Tim Pemenang dan Kalah')
plt.show()


#Jumlah Turnover (TO) dan Personal Foul (PF)
#Visualisasi turnovers
plt.figure(figsize=(10, 5))
sns.kdeplot(df['WTO'], color='blue', label='Winning Turnovers')
sns.kdeplot(df['LTO'], color='red', label='Losing Turnovers')
plt.legend()
plt.title('Distribusi Turnovers Tim Pemenang dan Kalah')
plt.show()

#Visualisasi personal fouls
plt.figure(figsize=(10, 5))
sns.kdeplot(df['WPF'], color='blue', label='Winning Personal Fouls')
sns.kdeplot(df['LPF'], color='red', label='Losing Personal Fouls')
plt.legend()
plt.title('Distribusi Personal Fouls Tim Pemenang dan Kalah')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load data
tourney_details = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv')

#cek info data
print(tourney_details.info())

#cek data teratas
print(tourney_details.head())

#Distribusi skor pemenang dan kalah
plt.figure(figsize=(10, 5))
sns.histplot(tourney_details['WScore'], bins=30, kde=True, color='blue', label='Winnning Score')
sns.histplot(tourney_details['LScore'], bins=30, kde=True, color='red', label='Lossing Score')
plt.legend()
plt.title('Distribution of Winning and Losing Scores')
plt.show()

#Distribusi field goals (2P dan 3P)
plt.figure(figsize=(10, 5))
sns.histplot(tourney_details['WFGM'], bins=30, kde=True, color='blue', label='Winning Field Goals Made')
sns.histplot(tourney_details['LFGM'], bins=30, kde=True, color='red', label='Losing Field Goals Made')
plt.legend()
plt.title('Distribution of Field Goals Made')
plt.show

#efisiensi tembakan
tourney_details['WFG%'] = tourney_details['WFGM'] / tourney_details['WFGA']
tourney_details['LFG%'] = tourney_details['LFGM'] / tourney_details['LFGA']

plt.figure(figsize=(10, 5))
sns.histplot(tourney_details['WFG%'], bins=30, kde=True, color='blue', label='Winning FG%')
sns.histplot(tourney_details['LFG%'], bins=30, kde=True, color='red', label='Losing FG%')
plt.legend()
plt.title('Field Goal Percentage (Winning vs Losing)')
plt.show()

#Distribusi Rebound
plt.figure(figsize=(10, 5))
sns.histplot(tourney_details['WOR'] + tourney_details['WDR'], bins=30, kde=True, color='blue', label='Winning Rebounds')
sns.histplot(tourney_details['LOR'] + tourney_details['LDR'], bins=30, kde=True, color='red', label='Losing Rebounds')
plt.legend()
plt.title('Total Rebounds (Winning vs Losing)')
plt.show()

#Distribusi Turnover
plt.figure(figsize=(10, 5))
sns.histplot(tourney_details['WTO'], bins=30, kde=True, color='blue', label='Winning Turnovers')
sns.histplot(tourney_details['LTO'], bins=30, kde=True, color='red', label='Losing Turnovers')
plt.legend()
plt.title('Turnovers (Winning vs Losing)')
plt.show()

#Distribusi Margin Kemenangan
tourney_details['ScoreDiff'] = tourney_details['WScore'] - tourney_details['LScore']
plt.figure(figsize=(10, 5))
sns.histplot(tourney_details['ScoreDiff'], bins=30, kde=True, color='green')
plt.title('Distribution of Score Difference')
plt.show()

#Distribusi jumlah overtime
plt.figure(figsize=(10,5))
sns.countplot(x='NumOT', data=tourney_details, color='purple')
plt.title('Number of Overtime Games')
plt.show()



#Rata-rata statistik tim pemenang
winning_stats = tourney_details.groupby('WTeamID').agg({'WFGM': 'sum', 'WFGA': 'sum', 'WAst': 'sum', 'WTO': 'sum', 'WOR': 'sum', 'WDR': 'sum', 'WFTM': 'sum', 'WFTA': 'sum',
                                                       'WScore': 'sum', 'WLoc': 'count'}).reset_index()
print(winning_stats.head())
print()

#Rata-rata statistik tim kalah
losing_stats = tourney_details.groupby('LTeamID').agg({'LFGM': 'sum', 'LFGA': 'sum', 'LAst': 'sum', 'LTO': 'sum', 'LOR': 'sum', 'LDR': 'sum', 'LFTM': 'sum', 'LFTA': 'sum',
                                                      'LScore': 'sum', 'WLoc': 'count'}).reset_index()
print(losing_stats.head())
print()

#Tambahkan FG% dan Turnover Rate
winning_stats['WFGP'] = winning_stats['WFGM'] / winning_stats['WFGA']
winning_stats['WTO_Rate'] = winning_stats['WTO'] / winning_stats['WLoc']

losing_stats['LFGP'] = losing_stats['LFGM'] / losing_stats['LFGA']
losing_stats['LTO_Rate'] = losing_stats['LTO'] / losing_stats['WLoc']

#Visualisasi FG% tim pemenang dan tim yang kalah
plt.figure(figsize=(10, 5))
sns.kdeplot(winning_stats['WFGP'], fill=True, label='Winning Teams FG%')
sns.kdeplot(losing_stats['LFGP'], fill=True, label='Losing Teams FG%')
plt.legend()
plt.title('Field Goal Percentage: Winning vs Losing Teams')
plt.xlabel('Field Goal Percentage')
plt.show()

#Visualisasi Turnover Rate tim Pemenang dan tim yang kalah
plt.figure(figsize=(10, 5))
sns.kdeplot(winning_stats['WTO_Rate'], fill=True, label='Winning Teams Turnover Rate')
sns.kdeplot(losing_stats['LTO_Rate'], fill=True, label='Losing Teams Turnover Rate')
plt.legend()
plt.title('Turnover Rate: Winning vs Losing Temas')
plt.xlabel('Turnover per Game')
plt.show()

#Perbandingan rata-rata
avg_stats = pd.DataFrame({'Stat': ['FG%', 'Turnover Rate'],
                          'Winning Teams': [winning_stats['WFGP'].mean(), winning_stats['WTO_Rate'].mean()],
                          'Losing Teams': [losing_stats['LFGP'].mean(), losing_stats['LTO_Rate'].mean()]
                         })
print(avg_stats)


import seaborn as sns
import matplotlib.pyplot as plt

#Distribusi offensive rebounds
plt.figure(figsize=(10, 5))
sns.kdeplot(winning_stats['WOR'], label='Winning Teams - Offensive Rebounds', shade=True)
sns.kdeplot(losing_stats['LOR'], label='Losing Teams - Offensive Rebounds', shade=True)
plt.legend()
plt.title('Distribution of Offensive Rebounds')
plt.xlabel('Offensive Rebounds')
plt.ylabel('Density')
plt.show()

#Distribusi defensive rebounds
plt.figure(figsize=(10, 5))
sns.kdeplot(winning_stats['WDR'], label='Winning Teams - Defensive Rebounds', shade=True)
sns.kdeplot(losing_stats['LDR'], label='Losing Teams - Defensive Rebounds', shade=True)
plt.legend()
plt.title('Distribution of Defensive Rebounds')
plt.xlabel('Defensive Rebounds')
plt.ylabel('Density')
plt.show()

#Distribusi Turnovers
plt.figure(figsize=(10,5))
sns.kdeplot(winning_stats['WTO'], label='Winning Teams - Turnovers', shade=True)
sns.kdeplot(losing_stats['LTO'], label='Losing Teams - Turnovers', shade=True)
plt.legend()
plt.title('Distribution of Turnovers')
plt.xlabel('Turnovers')
plt.ylabel('Density')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load data
cities = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/Cities.csv')

#cek struktur data
print(cities.info(), "\n")
print(cities.head(), '\n')

#cek jumlah kota dan negara bagian unik
print(f"Jumlah kota unik: {cities['City'].nunique()}")
print(f"Jumlah negara bagian unik: {cities['State'].nunique()}")




