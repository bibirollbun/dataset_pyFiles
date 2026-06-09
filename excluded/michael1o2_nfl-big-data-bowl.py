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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


player_play = pd.read_csv("../input/nfl-big-data-bowl-2025/player_play.csv")
games = pd.read_csv("../input/nfl-big-data-bowl-2025/games.csv")
plays = pd.read_csv("../input/nfl-big-data-bowl-2025/plays.csv")
players = pd.read_csv("../input/nfl-big-data-bowl-2025/players.csv")


players.head()


players.info()


players.isnull().sum()


print("Data type of birthDate column before parsing : ", players["birthDate"].dtypes)
players["birthDate"] = pd.to_datetime(players["birthDate"], format='mixed')
print("Data type of birthDate column after parsing : ", players["birthDate"].dtypes)
print(players["birthDate"].head())


players['birthYear']= pd.DatetimeIndex(players['birthDate']).year
print(players['birthYear'])


print(players['birthYear'].value_counts())


print(2025-max(players['birthYear']))
print(2025-min(players['birthYear']))


hist=players['birthYear'].plot.hist(bins=20,color='orange',edgecolor='black')


college_names=players.pivot_table(index= ['collegeName'], aggfunc='size')
college_names=college_names.reset_index()
college_names.columns=['College Names','Counts']
college_names=college_names.sort_values('Counts',ascending=False)
print(college_names)


top_colleges=college_names[0:10]
print(top_colleges)


fig=plt.figure(figsize=(8,8))
circle=plt.Circle((0,0),0.5,color='white')
plt.pie(top_colleges['Counts'],labels=top_colleges['College Names'])
p=plt.gcf()
p.gca().add_artist(circle)
plt.legend(top_colleges['Counts'])
plt.title("Top 10 Colleges having Highest Number of Players",fontsize=25)
plt.show()


pos_val=players.pivot_table(index=['position'], aggfunc='size')
pos_val = pos_val.reset_index()
pos_val.columns=['Positions','Counts']
pos_val = pos_val.sort_values('Counts',ascending=False)
print(pos_val)


height = players[players['height'] == max(players['height'])]
height


lowheight = players[players['height'] == min(players['height'])]
lowheight


oldest = players[players['birthYear'] == min(players['birthYear'])]
oldest


youngest = players[players['birthYear'] == max(players['birthYear'])]
youngest


mean=np.ceil(players['weight'].mean())


median=np.ceil(players['weight'].median())


plt.figure(figsize=(10, 5))
sns.set_style('white')
hist_plot = sns.histplot(players['weight'], )
hist_plot.axvline(mean, color='r', linestyle='--', linewidth = 4, label = f'mean-{mean}')
hist_plot.axvline(median, color='g', linestyle='-', linewidth = 4, label = f'median-{median}')
plt.suptitle("Players Weight Distribution")
plt.legend();


games.tail()


print('NFL Unique values and Their Counts')
g_season=games.pivot_table(index=['season'], aggfunc='size')
g_season = g_season.reset_index()
g_season.columns = ['Seasons','Counts']
g_season = g_season.sort_values('Counts',ascending=False)
print(g_season)


g_week = games.pivot_table(index = ['week'], aggfunc = 'size') 
g_week = g_week.reset_index()
g_week.columns= ["Weeks", "Counts"]
g_week = g_week.sort_values("Counts", ascending = False)
print(g_week)


bar_plot = g_week.plot.barh()
bar_plot.set_title('Unique NFL Weeks and their Counts')
bar_plot.set_xlabel('Counts')
bar_plot.set_ylabel('Weeks')
bar_plot.invert_yaxis()
plt.show(bar_plot)


print('Unique NFL Dates and Their Counts')
g_date = games.pivot_table(index=['gameDate'],aggfunc = 'size')
g_date = g_date.reset_index()
g_date.columns=['Date','Counts']
g_date=g_date.sort_values('Counts',ascending=False)
print(g_date)


bar_plot1=g_date.plot.barh()
bar_plot1.set_title('NFL Event Dates')
bar_plot1.set_xlabel('COunts')
bar_plot1.set_ylabel('Dates')
bar_plot1.invert_yaxis()
plt.show(bar_plot1)


games['gameDay'] = pd.DatetimeIndex(games['gameDate']).day
print(games['gameDay'])


print("Unique NFL days and their counts :")
g_days = games.pivot_table(index = ['gameDay'], aggfunc = 'size') 
g_days = g_days.reset_index()
g_days.columns= ["Day", "Counts"]
g_days = g_days.sort_values("Counts", ascending = False)
print(g_days)


bar_plot3=g_days.plot.barh()
bar_plot3.set_title('NFL Event Days')
bar_plot3.set_xlabel('Counts')
bar_plot3.set_ylabel('Days')
bar_plot3.invert_yaxis()
plt.show(bar_plot3)


print('Unique NFL Timings and Their Counts')
g_time = games.pivot_table(index= ['gameTimeEastern'],aggfunc='size')
g_time = g_time.reset_index()
g_time.columns = ['Time','Counts']
g_time = g_time.sort_values('Counts',ascending=False)
print(g_time)


games['gameTimeEastern'].value_counts().sort_values().plot.barh(color=['blue','red'], title='NFL Event Time')
plt.xlabel('Counts');


print('Unique NFl home and Their Values')
g_home = games.pivot_table(index = ['homeTeamAbbr'], aggfunc='size')
g_home = g_home.reset_index()
g_home.columns = ['Home Team','Counts']
g_home = g_home.sort_values('Counts', ascending = False)
print(g_home)


g_home['Home Team'].value_counts().head(20).plot.barh(color='purple',title='NFL Home Team')
plt.xlabel('Counts')


print("Unique NFL yards to go and their counts :")
g_yards = plays.pivot_table(index = ['yardsToGo'], aggfunc = 'size') 
g_yards = g_yards.reset_index()
g_yards.columns= ["Yards To Go", "Counts"]
g_yards = g_yards.sort_values("Counts", ascending = False)
print(g_yards)


bar_plot = g_yards.plot.barh()
bar_plot.set_title("NFL, Yards to Go")
bar_plot.set_xlabel("Counts")
bar_plot.set_ylabel("Yards to Go ")
bar_plot.invert_yaxis() #order increasing
plt.show(bar_plot)


print("Unique NFL Offense Formation and their counts :")
gp_type = plays.pivot_table(index = ['offenseFormation'], aggfunc = 'size') 
gp_type = gp_type.reset_index()
gp_type.columns= ["Offense Formation", "Counts"]
gp_type = gp_type.sort_values("Counts", ascending = False)
print(gp_type)


plays["offenseFormation"].value_counts().plot.barh(color='orange', title='NFL Offense Formation')
plt.xlabel('Counts');


print("Unique NFL Pre-snap Home Team Win Probability and their counts :")
g_home = plays.pivot_table(index = ['preSnapHomeTeamWinProbability'], aggfunc = 'size') 
g_home = g_home.reset_index()
g_home.columns= ["Pre-Snap HomeTeam Win Probability", "Counts"]
g_home = g_home.sort_values("Counts", ascending = False)
print(g_home)


hist = plays["preSnapHomeTeamWinProbability"].plot.hist(bins=25, color="orange", edgecolor="black")
plt.title('NFL Pre-Snap HomeTeam Win Probability');


print("Unique NFL pass results and their counts :")
g_res = plays.pivot_table(index = ['passResult'], aggfunc = 'size') 
g_res = g_res.reset_index()
g_res.columns= ["Pass Results", "Counts"]
g_res = g_res.sort_values("Counts", ascending = False)
print(g_res)


plays["passResult"].value_counts().sort_values().plot.barh(color='red', title='NFL Pass Results')
plt.xlabel('Counts');


print("Unique NFL absolute yardline numbers and their counts :")
g_abyl = plays.pivot_table(index = ['absoluteYardlineNumber'], aggfunc = 'size') 
g_abyl = g_abyl.reset_index()
g_abyl.columns= ["Absolute YardLine Number", "Counts"]
g_abyl = g_abyl.sort_values("Counts", ascending = False)
print(g_abyl)


plays["absoluteYardlineNumber"].value_counts().head(20).sort_values().plot.barh(color='red', title='NFL Absolute Yard Line Number')
plt.xlabel('Counts');


tracking1 = pd.read_csv("../input/nfl-big-data-bowl-2025/tracking_week_1.csv")
tracking1.head()


tracking1['date'] = pd.DatetimeIndex(tracking1['time']).date
print(tracking1["date"])


print("Unique NFL dates and their counts :")
tr_date = tracking1.pivot_table(index = ['date'], aggfunc = 'size') 
tr_date = tr_date.reset_index()
tr_date.columns= ["Date", "Counts"]
tr_date = tr_date.sort_values("Counts", ascending = False)
print(tr_date)


tracking1["date"].value_counts().sort_values().plot.bar(color='red', title='NFL dates')
plt.xticks(rotation=0)
plt.ylabel('Counts');


data = tracking1.query('playId == 56 and gameId == 2022090800')
print(data[["x", "y", "club"]])


my_palette = ["#95a5a6", "#e74c3c", "#34495e"]
sns.pairplot(
    data,
    x_vars=["x"],
    y_vars=["y"],
    height=3.5,
    hue="club", #hue define the color-code variable
    palette=my_palette,   # <-- see here, custom palette
)
plt.title('Players Positions')
plt.show()


# only looking at data from plays when the home team is on the offense
right = tracking1[tracking1['playDirection'] == 'right']
# only looking at a specific match
match = right[right['gameId'] == 2022090800]


fig, ax = plt.subplots(figsize=(15,10))
plt.hist2d(right['x'][right['event'] == 'tackle'], right['y'][right['event'] == 'tackle'],bins=70, cmap='summer')
plt.xlim(0 , 120)
plt.ylim(0,  53.3)
plt.title('Heatmap of all player locations during a tackle when the offense is moving to the right')
plt.show()


fig, ax = plt.subplots(figsize=(6, 4)) #figsize=(width, height))
tracking1["club"].value_counts().head(10).sort_values(ascending=True).plot(
    kind="barh", color='g', ax=ax, title="NFL 2025"
)
ax.set_xlabel("Number of Training Examples")
plt.show()




