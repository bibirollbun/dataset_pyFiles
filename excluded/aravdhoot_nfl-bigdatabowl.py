import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from IPython.display import HTML
from base64 import b64encode


tracking_week_1 = pd.read_csv(f"/kaggle/input/nfl-big-data-bowl-2025/tracking_week_1.csv")


play1 = tracking_week_1[(tracking_week_1.gameId == 2022091200) & (tracking_week_1.playId == 64)]

before_snap = play1[play1.frameType == "BEFORE_SNAP"]
snap = play1[play1.frameType == "SNAP"]
after_snap = play1[play1.frameType == "AFTER_SNAP"]


def ball_distance_dictionary(player_df : pd.DataFrame) -> (dict, int): 
    player_sep = {}
    players = player_df.displayName.unique()
    football = player_df[player_df.displayName == 'football'].reset_index(drop=True)
    for i in range(len(player_df.displayName.unique())):
        player = player_df[player_df.displayName == players[i]].reset_index(drop=True)
        if player.displayName[0] != "football":
            distances = list(zip(player.x, player.y))
            seps = []
            for idx, (x_dist, y_dist) in enumerate(distances):
                sep = ((football.x[idx] - x_dist)**2 + (football.y[idx] - y_dist)**2)
                seps.append(sep)
                frames = len(seps)
        player_sep[players[i]] = seps
    return player_sep, frames

def player_with_ball(player_df : pd.DataFrame) -> list:
    player_sep, frames = ball_distance_dictionary(player_df)
    player_with_ball = [None] * frames
    players = player_df.displayName.unique()
    
    for frame in range(frames):
        curr = None
        curr_dist = 17209
        for player in players:
            if player_sep[player][frame] < 1:
                curr = player
                curr_dist = player_sep[player][frame]
            if curr is not None and player_sep[player][frame] < curr_dist:
                curr = player
                curr_dist = player_sep[player][frame]
        if curr is not None:
            player_with_ball[frame] = (curr, curr_dist)
        else:
            player_with_ball[frame] = ('In air', None)

    return player_with_ball

def predict_next_k_frames(play: pd.DataFrame, k: int) -> pd.DataFrame: 
    players = play['displayName'].unique()
    new_after_snap = pd.DataFrame()

    for value in players:
        player_data = play.loc[play['displayName'] == value]
        x_data = [(row['frameId'], row['x']) for _, row in player_data.iterrows()]
        y_data = [(row['frameId'], row['y']) for _, row in player_data.iterrows()]

        X = np.array([point[0] for point in x_data]).reshape(-1, 1)  # Reshape to 2D array
        x_pred = np.array([point[1] for point in x_data])
        model_x = LinearRegression()
        model_x.fit(X, x_pred)
        max_frameid = y_data[-1][0]
        next_frame_ids = np.array([max_frameid + i for i in range(1, k + 1)]).reshape(-1, 1)
        predicted_x = model_x.predict(next_frame_ids)  
        
        Y = np.array([point[0] for point in y_data]).reshape(-1, 1)  # Reshape to 2D array
        y_pred = np.array([point[1] for point in y_data])
        model_y = LinearRegression()
        model_y.fit(Y, y_pred)
        max_frameid = y_data[-1][0]
        next_frame_ids = np.array([max_frameid + i for i in range(1, k + 1)]).reshape(-1, 1)
        predicted_y = model_y.predict(next_frame_ids)    
        

        new_rows = pd.DataFrame({
            'frameId': next_frame_ids.flatten(),
            'x': predicted_x.flatten(),
            'y': predicted_y.flatten()
        })

        for col in play.columns:
            if col not in new_rows.columns:  
                new_rows[col] = play[col].iloc[-1]
        
        updated_player_data = pd.concat([player_data, new_rows], ignore_index=True)
        new_after_snap = pd.concat([new_after_snap, updated_player_data], ignore_index=True)

    return new_after_snap, model_x, model_y


def generateMovementMap(gameID: int, playID: int):
    play = tracking_week_1[(tracking_week_1.gameId == gameID) & (tracking_week_1.playId == playID)]
    before_snap = play[play.frameType == "BEFORE_SNAP"]
    snap = play[play.frameType == "SNAP"]
    after_snap = play[play.frameType == "AFTER_SNAP"]
    
    n = 0
    frames = []
    for frameType in [before_snap, snap, after_snap]:
        players = frameType.displayName.unique()
        plt.xlim(0, 120)
        plt.ylim(0, 53.3)
        for i in range(len(players)):
            player = frameType[frameType.displayName == players[i]].reset_index(drop=True)
            if player.displayName[0] == "football":
                plt.scatter(player.x, player.y, c="red")
                plt.text(list(player.x)[-1], list(player.y)[-1], "FB")
            else:
                plt.scatter(player.x, player.y, c=player.frameId)
        
        title = ""
        match n:
            case 0: 
                title = "before snap"
            case 1: 
                title = "snap"
            case 2: 
                title = "after snap"
        plt.title(title)
    
        plt.show()
        n+=1


def delete_png_files(folder_path):
    """Deletes all PNG files in the specified folder."""

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")

# Specify the folder path where you want to delete PNG files
folder_path = "/kaggle/working" 

# Call the function to delete PNG files
delete_png_files(folder_path)


def generateVideo(gameID: int, playID: int):
    play = tracking_week_1[(tracking_week_1.gameId == gameID) & (tracking_week_1.playId == playID)]
    
    after_snap = play[play.frameType == "AFTER_SNAP"]
    prediction_play = None
    n_frames = 20
    train_start = after_snap.frameId.min()
    train_stop = after_snap.frameId.max() - n_frames 

    y_actual = after_snap.loc[after_snap['frameId'] >= train_stop, 'y']
    
    if train_stop - train_start >= 10:
        prediction_play, model_x, model_y = predict_next_k_frames(after_snap[after_snap.frameId <= train_stop], n_frames) 

    y_test = prediction_play.loc[prediction_play['frameId'] >= train_stop, 'y']

    from sklearn.metrics import mean_squared_error, r2_score

    r2 = r2_score(y_test, y_actual)
    mse = mean_squared_error(y_test, y_actual)

    print('=' * 40)
    print('Linear Regresison Stats for this Visualization')
    print(f'R2 Score = {r2:.4f}')
    print(f'MSE = {(mse/after_snap.displayName.unique().size):.4f}')
    print('=' * 40)
    
    frames = []
    i = 0
    ball_posessions = player_with_ball(play)
    for frame in play.frameId.unique():
        play_at_frame = play[play.frameId == frame].sort_values(by="jerseyNumber").reset_index(drop=True)
        players = play_at_frame[:-1]
        ball = play_at_frame[-1:] 
    
        plt.xlim(0, 120)
        plt.ylim(0, 53.3)
        plt.scatter(players.x, players.y, c=range(len(players)))
        plt.scatter(ball.x, ball.y, c="red", marker="*") 
        plt.title(play_at_frame.frameType[0])
    
        player = ball_posessions[i][0]
        if (player != "In air"): 
            player_x = list(play_at_frame[play_at_frame.displayName == player].x)[0]
            player_y = list(play_at_frame[play_at_frame.displayName == player].y)[0]
            plt.text(player_x, player_y, player)
    
        if (not prediction_play is None) and play_at_frame.frameType[0] == "AFTER_SNAP":
            pred_at_frame = prediction_play[prediction_play.frameId == frame].sort_values(by="jerseyNumber").reset_index(drop=True)
            pred_players = pred_at_frame[:-1]
            pred_ball = pred_at_frame[-1:] 
            plt.scatter(pred_players.x, pred_players.y, c=range(len(players)), alpha=0.3)
            plt.scatter(pred_ball.x, pred_ball.y, c="red", marker="*", alpha = 0.3) 
    
        plt.savefig(f"img{i}.png")
        plt.close()
        i += 1

    #generate video from PNGs
    play_length = (pd.to_datetime(play.time.max()) - pd.to_datetime(play.time.min())).total_seconds()
    num_frames = len(play.frameId.unique())
    fps = num_frames / play_length
    os.system(f"ffmpeg -loglevel quiet -r {fps} -i img%01d.png -vcodec h264 -y {gameID}_{playID}.mp4")
    
    delete_png_files("/kaggle/working")


plays = 0
for game in tracking_week_1.gameId.unique():
    game_df = tracking_week_1[tracking_week_1.gameId == game]
    print(f"Game: {game} - Plays: {len(game_df.playId.unique())}")
    plays += len(game_df.playId.unique())
    
    for play in game_df.playId.unique():
        play_df = game_df[game_df.playId == play]
        len(play_df[play_df.frameType == "AFTER_SNAP"])
        
print(f"Total Games: {len(tracking_week_1.gameId.unique())} - Total Plays: {plays}")    


tracking_week_1.gameId.unique()


game = tracking_week_1.gameId.unique()[0]
tracking_week_1[tracking_week_1.gameId == game].playId.unique()


#generate movement heat map
game = 2022091200
play = 64

generateMovementMap(game, play)


#generate example videos
game = 2022091109
play = 128
generateVideo(game, play)

game = 2022091200
play = 64
generateVideo(game, play)

game = 2022091101
play = 4242
generateVideo(game, play)


def play(filename):
    html = ''
    video = open(filename,'rb').read()
    src = 'data:video/mp4;base64,' + b64encode(video).decode()
    html += '<video width=800 controls autoplay loop><source src="%s" type="video/mp4"></video>' % src 
    return HTML(html)

play('/kaggle/working/2022091200_64.mp4')


def play(filename):
    html = ''
    video = open(filename,'rb').read()
    src = 'data:video/mp4;base64,' + b64encode(video).decode()
    html += '<video width=800 controls autoplay loop><source src="%s" type="video/mp4"></video>' % src 
    return HTML(html)

play('/kaggle/working/2022091109_128.mp4')


def play(filename):
    html = ''
    video = open(filename,'rb').read()
    src = 'data:video/mp4;base64,' + b64encode(video).decode()
    html += '<video width=800 controls autoplay loop><source src="%s" type="video/mp4"></video>' % src 
    return HTML(html)

play('/kaggle/working/2022091101_4242.mp4')

