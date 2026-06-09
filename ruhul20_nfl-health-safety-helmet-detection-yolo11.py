import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
import seaborn as sns
import plotly

import warnings
warnings.filterwarnings("ignore")


DATA_DIR = "/kaggle/input/nfl-health-and-safety-helmet-assignment"

os.listdir(DATA_DIR)


# Training data
# -----------------------------------------------------------------------
# Player information is included

# Bounding Box
train_df = pd.read_csv(f'{DATA_DIR}/train_labels.csv')

# Tracking Information using Sensor
train_tracking_df = pd.read_csv(f'{DATA_DIR}/train_player_tracking.csv')
test_tracking_df = pd.read_csv(f'{DATA_DIR}/test_player_tracking.csv')

# images/
# -----------------------------------------------------------------------
# Trained images using images_labels.csv and predict the train, test
# The prediction result is [train/test]_baseline_helmets.csv
# No player information is included

# information of images without player information
image_df = pd.read_csv(f'{DATA_DIR}/image_labels.csv')

# Baseline Prediction - Trained by images inside folder images/
train_predict_df = pd.read_csv(f'{DATA_DIR}/train_baseline_helmets.csv')

test_predict_df = pd.read_csv(f'{DATA_DIR}/test_baseline_helmets.csv')


# reference : https://www.kaggle.com/coldfir3/eda-helmet-keypoint-tracking-data-comparison
def get_frame_from_video(video_path, frame):
    video_path = f"{DATA_DIR}/train/{video_path}"
    frame = frame - 1
    
    !ffmpeg \
        -hide_banner \
        -loglevel fatal \
        -nostats \
        -i $video_path -vf "select=eq(n\,$frame)" -vframes 1 frame.png
    
    img = Image.open('frame.png')
    os.remove('frame.png')
    return img


get_frame_from_video('57586_001934_Endzone.mp4', 1)


def draw_rect(image, bbox_df):
    new_image = image.copy()
    draw = ImageDraw.Draw(new_image)
    for _, (left, width, top, height) in bbox_df[['left', 'width', 'top', 'height']].iterrows():
        draw.rectangle(((left, top), (left + width, top + height)), outline=(255, 0, 0), width=2)
    
    return new_image


def frame_bbox(df, video_frame):
    video_name = '_'.join(video_frame.split('_')[:3]) + '.mp4'
    frame = int(video_frame.split('_')[-1])
    
    image = get_frame_from_video(video_name, frame)
    bbox_df = df.query('video_frame == @video_frame')
    
    bbox_image = draw_rect(image, bbox_df)
    
    return bbox_image


frame_bbox(train_df, '57583_000082_Endzone_1')


from IPython.display import Video, display

def video(video_path, ratio=0.7):
    nfl_video = Video(f"{DATA_DIR}/train/{video_path}",
                      embed=True,
                      height=int(720 * ratio),
                      width=int(1280 * ratio))
    return nfl_video
    
video('57583_000082_Endzone.mp4')


def add_track_features(tracks, fps=59.94, snap_frame=10):
    """
    Add column features helpful for syncing with video data.
    Returns est_frame as an integer frame index (uses -1 for rows without a snap).
    """
    tracks = tracks.copy()
    tracks["game_play"] = (
        tracks["gameKey"].astype("str")
        + "_"
        + tracks["playID"].astype("str").str.zfill(6)
    )
    tracks["time"] = pd.to_datetime(tracks["time"])

    # The time when snap happened (first 'ball_snap' time per game_play)
    snap_dict = (
        tracks.query('event == "ball_snap"')
        .groupby("game_play")["time"]
        .first()
        .to_dict()
    )
    tracks["snap"] = tracks["game_play"].map(snap_dict)
    tracks["isSnap"] = tracks["snap"] == tracks["time"]

    # Use .dt.total_seconds() to get a numeric seconds value (float).
    tracks["snap_offset"] = (tracks["time"] - tracks["snap"]).dt.total_seconds()

    # Estimated video frame: numeric math -> round -> integer.
    # For rows where snap is missing, snap_offset will be NaN; set est_frame to -1 (or another sentinel).
    est_frames = ((tracks["snap_offset"] * fps) + snap_frame).round()

    # Fill NaN (rows without snap) with -1 and convert to int
    tracks["est_frame"] = est_frames.fillna(-1).astype(int)

    # Optional: keep snap_offset as float seconds (already numeric)
    return tracks


train_tracking_df = add_track_features(train_tracking_df)


import plotly.express as px
import plotly.graph_objects as go
import plotly


def add_plotly_field(fig):
    # Reference https://www.kaggle.com/ammarnassanalhajali/nfl-big-data-bowl-2021-animating-players
    fig.update_traces(marker_size=20)
    
    fig.update_layout(paper_bgcolor='#29a500', plot_bgcolor='#29a500', font_color='white',
        width = 800,
        height = 600,
        title = "",
        
        xaxis = dict(
            nticks = 10,
            title = "",
            visible=False
        ),
        
        yaxis = dict(
            scaleanchor = "x",
            title = "Temp",
            visible=False
        ),
        showlegend= True,
  
        annotations=[
       dict(
            x=-5,
            y=26.65,
            xref="x",
            yref="y",
            text="ENDZONE",
            font=dict(size=16,color="#e9ece7"),
            align='center',
            showarrow=False,
            yanchor='middle',
            textangle=-90
        ),
        dict(
            x=105,
            y=26.65,
            xref="x",
            yref="y",
            text="ENDZONE",
            font=dict(size=16,color="#e9ece7"),
            align='center',
            showarrow=False,
            yanchor='middle',
            textangle=90
        )]  
        ,
        legend=dict(
            traceorder="normal",
            font=dict(family="sans-serif",size=12),
            title = "",
            orientation="h",
            yanchor="bottom",
            y=1.00,
            xanchor="center",
            x=0.5
        ),
    )
    ####################################################
        
    fig.add_shape(type="rect", x0=-10, x1=0,  y0=0, y1=53.3,line=dict(color="#c8ddc0",width=3),fillcolor="#217b00" ,layer="below")
    fig.add_shape(type="rect", x0=100, x1=110, y0=0, y1=53.3,line=dict(color="#c8ddc0",width=3),fillcolor="#217b00" ,layer="below")
    for x in range(0, 100, 10):
        fig.add_shape(type="rect", x0=x,   x1=x+10, y0=0, y1=53.3,line=dict(color="#c8ddc0",width=3),fillcolor="#29a500" ,layer="below")
    for x in range(0, 100, 1):
        fig.add_shape(type="line",x0=x, y0=1, x1=x, y1=2,line=dict(color="#c8ddc0",width=2),layer="below")
    for x in range(0, 100, 1):
        fig.add_shape(type="line",x0=x, y0=51.3, x1=x, y1=52.3,line=dict(color="#c8ddc0",width=2),layer="below")
    
    for x in range(0, 100, 1):
        fig.add_shape(type="line",x0=x, y0=20.0, x1=x, y1=21,line=dict(color="#c8ddc0",width=2),layer="below")
    for x in range(0, 100, 1):
        fig.add_shape(type="line",x0=x, y0=32.3, x1=x, y1=33.3,line=dict(color="#c8ddc0",width=2),layer="below")
    
    
    fig.add_trace(go.Scatter(
    x=[2,10,20,30,40,50,60,70,80,90,98], y=[5,5,5,5,5,5,5,5,5,5,5],
    text=["G","1 0","2 0","3 0","4 0","5 0","4 0","3 0","2 0","1 0","G"],
    mode="text",
    textfont=dict(size=20,family="Arail"),
    showlegend=False,
    ))
    
    fig.add_trace(go.Scatter(
    x=[2,10,20,30,40,50,60,70,80,90,98], y=[48.3,48.3,48.3,48.3,48.3,48.3,48.3,48.3,48.3,48.3,48.3],
    text=["G","1 0","2 0","3 0","4 0","5 0","4 0","3 0","2 0","1 0","G"],
    mode="text",
    textfont=dict(size=20,family="Arail"),
    showlegend=False,
    ))
    
    return fig


train_tracking_df


def football_animation(game_play, df=train_tracking_df, fps=59.94):
    """
    Animated scatter of tracking for a single game_play id.
    - Ensures 'team' exists (maps player first char H/V to Home/Away if needed).
    - Ensures 'track_time_count' is present and integer for animation_frame.
    """
    df = df.copy()

    # Ensure time is datetime (safe)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    # Create team column if missing
    if "team" not in df.columns:
        if "player" in df.columns and df["player"].dtype == object:
            # Attempt to map first char 'H'/'V' to Home/Away; otherwise mark Unknown
            df["team"] = df["player"].str[0].map({"H": "Home", "V": "Away"})
            df["team"] = df["team"].fillna("Unknown")
        else:
            df["team"] = "Unknown"

    # Create a dense rank per game_play to use as animation_frame
    if "track_time_count" not in df.columns:
        df["track_time_count"] = (
            df.sort_values("time")
            .groupby("game_play")["time"]
            .rank(method="dense")
            .astype("Int64")  # nullable integer
        )

    # Filter to the requested game_play
    subset = df.query("game_play == @game_play").copy()
    if subset.empty:
        raise ValueError(f"No rows found for game_play {game_play!r}")

    # Plotly wants a non-null animation frame — convert to int or str
    # use a string frame to avoid Plotly complaining about Int64 with NA
    subset["anim_frame_str"] = subset["track_time_count"].astype(str)

    fig = px.scatter(
        subset,
        x="x",
        y="y",
        range_x=[-10, 110],
        range_y=[-10, 53.3],
        hover_data=["player", "s", "a", "dir"],
        color="team",                # now guaranteed to exist
        animation_frame="anim_frame_str",
        animation_group="player",    # keeps points consistent across frames
        text="player",
        title=f"Animation of NGS data for game_play {game_play}",
    )

    fig.update_traces(textfont_size=10, marker=dict(size=10))
    fig.update_layout(yaxis_autorange="reversed")  # flip if you want typical football orientation
    fig = add_plotly_field(fig)
    fig.show()



football_animation('57583_000082')


train_videos = os.listdir(f'{DATA_DIR}/train')
test_videos = os.listdir(f'{DATA_DIR}/test')

len(train_videos), len(test_videos)


set(test_videos).issubset(set(train_videos))


end_count = 0
side_count = 0
endzone_list = []
sideline_list = []
for train_video in train_videos:
    name = train_video.split('.')[0]
    video_id, play_id, view = name.split('_')
    
    if view == "Endzone":
        endzone_list.append('_'.join([video_id, play_id]))
        end_count += 1
    else:
        sideline_list.append('_'.join([video_id, play_id]))
        side_count += 1

print(end_count, side_count)


not_match_video = []

for play_id in train_df.playID.unique():
    end_frame_n = train_df.query('playID == @play_id and view == "Endzone"').frame.max()
    side_frame_n = train_df.query('playID == @play_id and view == "Sideline"').frame.max()
    
    if end_frame_n != side_frame_n:
        not_match_video.append(play_id)
        print(f'Not same at playID {play_id} endzone [{end_frame_n}] sideline [{side_frame_n}] difference [{abs(end_frame_n - side_frame_n)}]')


def get_total_frame(video_path):
    cap = cv2.VideoCapture(f"{DATA_DIR}/train/{video_path}")
    property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    length = int(cv2.VideoCapture.get(cap, property_id))
    
    return length


play2frame = train_df.groupby('video').frame.max().to_dict()


for video_name, label_frame_n in play2frame.items():
    video_frame_n = get_total_frame(video_name)
    if video_frame_n != label_frame_n:
        print('Not Match!')


frame_df = train_df.query('video == "57584_000336_Sideline.mp4"')
frame_df


frame_df.frame.max()


get_total_frame("57584_000336_Sideline.mp4")


test_df = train_df.query("video in @test_videos").reset_index().copy()
test_df


sns.displot(train_df.groupby(['gameKey', 'playID', 'view'])['frame'].nunique().values);


train_df.groupby(['gameKey', 'playID', 'view'])['frame'].nunique().sum()


play_per_game = train_df.groupby('gameKey')['playID'].nunique().reset_index().groupby('playID')['gameKey'].unique().to_dict()
play_per_game


train_df.groupby(['gameKey', 'playID', 'view'])['label'].nunique().value_counts()


# check what game does only 16 players are running for?
train_df.groupby(['gameKey', 'playID', 'view'])['label'].nunique().reset_index().query('label == 16')




