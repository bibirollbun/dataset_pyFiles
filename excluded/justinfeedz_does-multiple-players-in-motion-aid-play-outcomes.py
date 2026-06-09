import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


tracking_1 = pd.read_csv("./tracking_week_1.csv")
tracking_2 = pd.read_csv("./tracking_week_2.csv")
tracking_3 = pd.read_csv("./tracking_week_3.csv")
tracking_4 = pd.read_csv("./tracking_week_4.csv")
tracking_5 = pd.read_csv("./tracking_week_5.csv")
tracking_6 = pd.read_csv("./tracking_week_6.csv")
tracking_7 = pd.read_csv("./tracking_week_7.csv")
tracking_8 = pd.read_csv("./tracking_week_8.csv")
tracking_9 = pd.read_csv("./tracking_week_9.csv")
plays = pd.read_csv("./plays.csv")
players = pd.read_csv("./players.csv")
games = pd.read_csv("./games.csv")


colors = {
    'ARI':["#97233F","#000000","#FFB612"], 
    'ATL':["#A71930","#000000","#A5ACAF"], 
    'BAL':["#241773","#000000"], 
    'BUF':["#00338D","#C60C30"], 
    'CAR':["#0085CA","#101820","#BFC0BF"], 
    'CHI':["#0B162A","#C83803"], 
    'CIN':["#FB4F14","#000000"], 
    'CLE':["#311D00","#FF3C00"], 
    'DAL':["#003594","#041E42","#869397"],
    'DEN':["#FB4F14","#002244"], 
    'DET':["#0076B6","#B0B7BC","#000000"], 
    'GB' :["#203731","#FFB612"], 
    'HOU':["#03202F","#A71930"], 
    'IND':["#002C5F","#A2AAAD"], 
    'JAX':["#101820","#D7A22A","#9F792C"], 
    'KC' :["#E31837","#FFB81C"], 
    'LA' :["#003594","#FFA300","#FF8200"], 
    'LAC':["#0080C6","#FFC20E","#FFFFFF"], 
    'LV' :["#000000","#A5ACAF"],
    'MIA':["#008E97","#FC4C02","#005778"], 
    'MIN':["#4F2683","#FFC62F"], 
    'NE' :["#002244","#C60C30","#B0B7BC"], 
    'NO' :["#101820","#D3BC8D"], 
    'NYG':["#0B2265","#A71930","#A5ACAF"], 
    'NYJ':["#125740","#000000","#FFFFFF"], 
    'PHI':["#004C54","#A5ACAF","#ACC0C6"], 
    'PIT':["#FFB612","#101820"], 
    'SEA':["#002244","#69BE28","#A5ACAF"], 
    'SF' :["#AA0000","#B3995D"],
    'TB' :["#D50A0A","#FF7900","#0A0A08"], 
    'TEN':["#0C2340","#4B92DB","#C8102E"], 
    'WAS':["#5A1414","#FFB612"], 
    'football':["#CBB67C","#663831"]
}


def hex_to_rgb_array(hex_color):
    '''take in hex val and return rgb np array'''
    return np.array(tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))) 

def ColorDistance(hex1,hex2):
    '''d = {} distance between two colors(3)'''
    if hex1 == hex2:
        return 0
    rgb1 = hex_to_rgb_array(hex1)
    rgb2 = hex_to_rgb_array(hex2)
    rm = 0.5*(rgb1[0]+rgb2[0])
    d = abs(sum((2+rm,4,3-rm)*(rgb1-rgb2)**2))**0.5
    return d

def ColorPairs(team1,team2):
    color_array_1 = colors[team1]
    color_array_2 = colors[team2]
    # If color distance is small enough then flip color order
    if ColorDistance(color_array_1[0],color_array_2[0])<500:
        return {team1:[color_array_1[0],color_array_1[1]],team2:[color_array_2[1],color_array_2[0]],'football':colors['football']}
    else:
        return {team1:[color_array_1[0],color_array_1[1]],team2:[color_array_2[0],color_array_2[1]],'football':colors['football']}


def animate_play(games,tracking_df,play_df,players,gameId,playId):
    selected_game_df = games[games.gameId==gameId].copy()
    selected_play_df = play_df[(play_df.playId==playId)&(play_df.gameId==gameId)].copy()
    
    tracking_players_df = pd.merge(tracking_df,players,how="left",on = "nflId")
    selected_tracking_df = tracking_players_df[(tracking_players_df.playId==playId)&(tracking_players_df.gameId==gameId)].copy()
    
    sorted_frame_list = selected_tracking_df.frameId.unique()
    sorted_frame_list.sort()
    
    # get good color combos
    team_combos = list(set(selected_tracking_df.club.unique())-set(["football"]))
    color_orders = ColorPairs(team_combos[0],team_combos[1])
    
    # get play General information 
    line_of_scrimmage = selected_play_df.absoluteYardlineNumber.values[0]
    ## Fixing first down marker issue from last year
    if selected_tracking_df.playDirection.values[0] == "right":
        first_down_marker = line_of_scrimmage + selected_play_df.yardsToGo.values[0]
    else:
        first_down_marker = line_of_scrimmage - selected_play_df.yardsToGo.values[0]
    down = selected_play_df.down.values[0]
    quarter = selected_play_df.quarter.values[0]
    gameClock = selected_play_df.gameClock.values[0]
    playDescription = selected_play_df.playDescription.values[0]
    # Handle case where we have a really long Play Description and want to split it into two lines
    if len(playDescription.split(" "))>15 and len(playDescription)>115:
        playDescription = " ".join(playDescription.split(" ")[0:16]) + "<br>" + " ".join(playDescription.split(" ")[16:])

    # initialize plotly start and stop buttons for animation
    updatemenus_dict = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]
    # initialize plotly slider to show frame position in animation
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Frame:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }


    frames = []
    for frameId in sorted_frame_list:
        data = []
        # Add Numbers to Field 
        data.append(
            go.Scatter(
                x=np.arange(20,110,10), 
                y=[5]*len(np.arange(20,110,10)),
                mode='text',
                text=list(map(str,list(np.arange(20, 61, 10)-10)+list(np.arange(40, 9, -10)))),
                textfont_size = 30,
                textfont_family = "Courier New, monospace",
                textfont_color = "#ffffff",
                showlegend=False,
                hoverinfo='none'
            )
        )
        data.append(
            go.Scatter(
                x=np.arange(20,110,10), 
                y=[53.5-5]*len(np.arange(20,110,10)),
                mode='text',
                text=list(map(str,list(np.arange(20, 61, 10)-10)+list(np.arange(40, 9, -10)))),
                textfont_size = 30,
                textfont_family = "Courier New, monospace",
                textfont_color = "#ffffff",
                showlegend=False,
                hoverinfo='none'
            )
        )
        # Add line of scrimage 
        data.append(
            go.Scatter(
                x=[line_of_scrimmage,line_of_scrimmage], 
                y=[0,53.5],
                line_dash='dash',
                line_color='blue',
                showlegend=False,
                hoverinfo='none'
            )
        )
        # Add First down line 
        data.append(
            go.Scatter(
                x=[first_down_marker,first_down_marker], 
                y=[0,53.5],
                line_dash='dash',
                line_color='yellow',
                showlegend=False,
                hoverinfo='none'
            )
        )
        # Add Endzone Colors 
        endzoneColors = {0:color_orders[selected_game_df.homeTeamAbbr.values[0]][0],
                         110:color_orders[selected_game_df.visitorTeamAbbr.values[0]][0]}
        for x_min in [0,110]:
            data.append(
                go.Scatter(
                    x=[x_min,x_min,x_min+10,x_min+10,x_min],
                    y=[0,53.5,53.5,0,0],
                    fill="toself",
                    fillcolor=endzoneColors[x_min],
                    mode="lines",
                    line=dict(
                        color="white",
                        width=3
                        ),
                    opacity=1,
                    showlegend= False,
                    hoverinfo ="skip"
                )
            )
        # Plot Players
        for team in selected_tracking_df.club.unique():
            plot_df = selected_tracking_df[(selected_tracking_df.club==team)&(selected_tracking_df.frameId==frameId)].copy()
            if team != "football":
                hover_text_array=[]
                for nflId in plot_df.nflId:
                    selected_player_df = plot_df[plot_df.nflId==nflId]
                    hover_text_array.append("nflId:{}<br>displayName:{}<br>Player Speed:{} yd/s".format(selected_player_df["nflId"].values[0],
                                                                                      selected_player_df["displayName_x"].values[0],
                                                                                      selected_player_df["s"].values[0]))
                data.append(go.Scatter(x=plot_df["x"], y=plot_df["y"],mode = 'markers',marker=go.scatter.Marker(
                                                                                             color=color_orders[team][0],
                                                                                             line=go.scatter.marker.Line(width=2,
                                                                                                            color=color_orders[team][1]),
                                                                                             size=10),
                                        name=team,hovertext=hover_text_array,hoverinfo="text"))
            else:
                data.append(go.Scatter(x=plot_df["x"], y=plot_df["y"],mode = 'markers',marker=go.scatter.Marker(
                                                                                             color=color_orders[team][0],
                                                                                             line=go.scatter.marker.Line(width=2,
                                                                                                            color=color_orders[team][1]),
                                                                                             size=10),
                                        name=team,hoverinfo='none'))
        # add frame to slider
        slider_step = {"args": [
            [frameId],
            {"frame": {"duration": 100, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 0}}
        ],
            "label": str(frameId),
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)
        frames.append(go.Frame(data=data, name=str(frameId)))

    scale=10
    layout = go.Layout(
        autosize=False,
        width=120*scale,
        height=60*scale,
        xaxis=dict(range=[0, 120], autorange=False, tickmode='array',tickvals=np.arange(10, 111, 5).tolist(),showticklabels=False),
        yaxis=dict(range=[0, 53.3], autorange=False,showgrid=False,showticklabels=False),

        plot_bgcolor='#00B140',
        # Create title and add play description at the bottom of the chart for better visual appeal
        title=f"GameId: {gameId}, PlayId: {playId}<br>{gameClock} {quarter}Q"+"<br>"*19+f"{playDescription}",
        updatemenus=updatemenus_dict,
        sliders = [sliders_dict]
    )

    fig = go.Figure(
        data=frames[0]["data"],
        layout= layout,
        frames=frames[1:]
    )
    # Create First Down Markers 
    for y_val in [0,53]:
        fig.add_annotation(
                x=first_down_marker,
                y=y_val,
                text=str(down),
                showarrow=False,
                font=dict(
                    family="Courier New, monospace",
                    size=16,
                    color="black"
                    ),
                align="center",
                bordercolor="black",
                borderwidth=2,
                borderpad=4,
                bgcolor="#ff7f0e",
                opacity=1
                )
    # Add Team Abbreviations in EndZone's
    for x_min in [0,110]:
        if x_min == 0:
            angle = 270
            teamName=selected_game_df.homeTeamAbbr.values[0]
        else:
            angle = 90
            teamName=selected_game_df.visitorTeamAbbr.values[0]
        fig.add_annotation(
            x=x_min+5,
            y=53.5/2,
            text=teamName,
            showarrow=False,
            font=dict(
                family="Courier New, monospace",
                size=32,
                color="White"
                ),
            textangle = angle
        )
    return fig


animate_play(games,tracking_1,plays,players,2022091200,64).show()


def tracking_play_merge(plays, tracking_week):
    motion = tracking_week[(tracking_week['club'] != "football")]
    merged = pd.merge(motion, plays, how='inner', on=['gameId', 'playId'])
    merged['offense'] = np.where(merged['club'] == merged['possessionTeam'], True, False)
    offense_merged = merged[(merged['offense'] == True) & (merged['frameType'] == 'BEFORE_SNAP')]
    return offense_merged


def find_multiple_motion_plays(offense_merged):
    week = offense_merged.gameId.unique()
    
    extra_motion = []

    for game in week:
        game_plays = offense_merged[offense_merged['gameId'] == game].playId.unique()
        for play in game_plays:
            line_set_frame = 0
            if offense_merged.frameId[(offense_merged['playId'] == play) & (offense_merged['gameId'] == game) & (offense_merged['event'] == 'man_in_motion')].unique().size > 0:
                man_motion_frame = offense_merged.frameId[(offense_merged['playId'] == play) & (offense_merged['gameId'] == game) & (offense_merged['event'] == 'man_in_motion')].unique().astype(int)[0]
                if offense_merged.frameId[(offense_merged['playId'] == play) & (offense_merged['gameId'] == game) & (offense_merged['event'] == 'line_set')].unique().size > 0:
                    line_set_frame = offense_merged.frameId[(offense_merged['playId'] == play) & (offense_merged['gameId'] == game) & (offense_merged['event'] == 'line_set')].unique().astype(int)[0]
                if line_set_frame > man_motion_frame:
                    continue
                game_play = offense_merged[(offense_merged['playId'] == play) & (offense_merged['gameId'] == game) & (offense_merged['frameId'] > man_motion_frame) & (offense_merged['s'] > 2.6)]
                if len(game_play.displayName.unique()) > 1:
                    extra_motion.append([game_play['gameId'].unique()[0], game_play['playId'].unique()[0]])
    return extra_motion


all_weeks = [tracking_1, tracking_2, tracking_3, tracking_4, tracking_5, tracking_6, tracking_7, tracking_8, tracking_9]

week_append = []

for week_track in all_weeks:
    week_merge = tracking_play_merge(plays, week_track)
    week_motion = find_multiple_motion_plays(week_merge)
    week_append = week_append + week_motion



import csv

fields = ['gameId', 'playId']

with open('multi_motion_plays.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(week_append)


motion_plays_read = pd.read_csv("./multi_motion_plays.csv")
motion_plays_read.head()


filtered_plays = plays[plays.set_index(['gameId', 'playId']).index.isin(zip(motion_plays_read['gameId'], motion_plays_read['playId']))]
filtered_plays.head()


team_count = filtered_plays['possessionTeam'].value_counts()

plt.figure(figsize=(12,6))
plt.bar(team_count.index, team_count.values)
plt.title('Play Count by Team (Weeks 1-9)')
plt.xlabel('Team')
plt.ylabel('Count')
plt.show()


formation = filtered_plays['offenseFormation'].value_counts()

plt.figure(figsize=(12,6))
plt.bar(formation.index, formation.values)
plt.title('Play Count by Formation')
plt.xlabel('Formation')
plt.ylabel('Count')
plt.show()


motion_runs = filtered_plays['rushLocationType'].count()
motion_passes = filtered_plays['passLocationType'].count()
all_runs = plays['rushLocationType'].count()
all_passes = plays['passLocationType'].count()

fig, ax = plt.subplots(1, 2, figsize=(10, 7))

ax[0].pie([motion_runs, motion_passes], labels=['Runs', 'Passes'], autopct='%.2f')
ax[0].set_title('Multi Motion \n % of Runs vs Passes')
ax[1].pie([all_runs, all_passes], labels=['Runs', 'Passes'], autopct='%.2f')
ax[1].set_title('All Plays \n % of Runs vs Passes')
plt.show


motion_formation = filtered_plays['pff_manZone'].value_counts()
all_formation = plays['pff_manZone'].value_counts()

fig, ax = plt.subplots(1, 2, figsize=(10, 7))

ax[0].pie(motion_formation.values, labels=['Zone', 'Man', 'Other'], autopct='%.2f')
ax[0].set_title('Multi Motion \n Defensive Cover Type')
ax[1].pie(all_formation.values, labels=['Zone', 'Man', 'Other'], autopct='%.2f')
ax[1].set_title('All Plays \n Defensive Cover Type')
plt.show


print(f"Avg Yards gained (multi-motion): {filtered_plays['yardsGained'].mean()}")
print(f"Avg Yards gained (all plays): {plays['yardsGained'].mean()}")


print(f"Avg Expected Points Added (multi-motion): {filtered_plays['expectedPointsAdded'].mean()}")
print(f"Avg Expected Points Added (all plays): {plays['expectedPointsAdded'].mean()}")


print(f"Percentage of Completed Passes (multi-motion): {len(filtered_plays[filtered_plays['passResult'] == 'C']) / filtered_plays['passResult'].value_counts().sum()}")
print(f"Percentage of Completed Passes (all plays): {len(plays[plays['passResult'] == 'C']) / plays['passResult'].value_counts().sum()}")


