import os
import glob
import tqdm
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LANDMARK_FILES_DIR = "/kaggle/input/asl-signs/train_landmark_files"
TRAIN_FILE = "/kaggle/input/asl-signs/train.csv"


participants = os.listdir(LANDMARK_FILES_DIR)
print(f"Total number of participants = {len(participants)}")
print(f"Average number of sequences per participant = {len(glob.glob(LANDMARK_FILES_DIR + '/*/*.parquet'))/len(participants)}")


sample = pd.read_parquet("/kaggle/input/asl-signs/train_landmark_files/16069/100015657.parquet")
print(f"Sample shape = {sample.shape}")
sample.sample(10)


sample.describe()


print(f"All different types of landmark = {sample.type.unique()}")


sample_left_hand = sample[sample.type == "left_hand"]
sample_right_hand = sample[sample.type == "right_hand"]

print(f"Percentage of nulls in Left Hand data = {100*np.mean(sample_left_hand['x'].isnull()):.02f} %")
print(f"Percentage of nulls in Right Hand data = {100*np.mean(sample_right_hand['x'].isnull()):.02f} %")


edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(0,17),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
         (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20)]

def plot_frame(df, frame_id, ax):
    df = df[df.frame == frame_id].sort_values(['landmark_index'])
    x = list(df.x)
    y = list(df.y)
    
    ax.scatter(df.x, df.y, color='dodgerblue')
    for i in range(len(x)):
        ax.text(x[i], y[i], str(i))
        
    for edge in edges:
        ax.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], color='salmon')
        ax.set_xlabel(f"Frame no. {frame_id}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    
def plot_frame_seq(df, frame_range, n_frames):
    frames = np.linspace(frame_range[0],frame_range[1],n_frames, dtype = int, endpoint=True)
    fig, ax = plt.subplots(n_frames, 1, figsize=(5,25))
    for i in range(n_frames):
        plot_frame(df, frames[i], ax[i])
        
    plt.show()

    
plot_frame_seq(sample_left_hand, (178,186), 5)


train = pd.read_csv(TRAIN_FILE)
print(f"Train shape = {train.shape}")
train.sample(10)


train.groupby(['participant_id']).agg(unique_signs=('sign', 'nunique')).reset_index()


count_per_sign = train.groupby(['sign'])['sequence_id'].count().reset_index().sort_values(['sequence_id'], ascending=False)
fig, ax = plt.subplots(1, 2, figsize=(16,5))
ax[0].bar(range(10), count_per_sign['sequence_id'][:10], color='salmon')
ax[1].bar(range(10), count_per_sign['sequence_id'][-10:], color='dodgerblue')

ax[0].set_xticks(range(10))
ax[1].set_xticks(range(10))

ax[0].set_xticklabels(count_per_sign['sign'][:10])
ax[1].set_xticklabels(count_per_sign['sign'][-10:])

ax[0].set_ylim(250,420)
ax[1].set_ylim(250,420)

ax[0].set_xlabel("Most frequent signs")
ax[1].set_xlabel("Least frequent signs")
plt.show()


word_dict = [[] for i in range(26)]
for word in train['sign'].unique():
    word_dict[ord(word[0].lower()) - 97].append(word)
for i in range(26):
    print(chr(i+97), str(sorted(word_dict[i])))


def get_details_per_sign(sign):
    train_sign_sample = train[train['sign'] == sign]
    n_frames = 0
    n_left_hand = 0
    n_right_hand = 0
    n_face = 0
    n_both_hands = 0
    for _,row in train_sign_sample.iterrows():
        df = pd.read_parquet(os.path.join("/kaggle/input/asl-signs", row.path))
        n_frames += df['frame'].nunique()
        n_left_hand += np.sum(df[(df['type'] == 'left_hand') & (df['landmark_index'] == 0)]['x'].isnull() == False)
        n_right_hand += np.sum(df[(df['type'] == 'right_hand') & (df['landmark_index'] == 0)]['x'].isnull() == False)
        n_face += np.sum(df[(df['type'] == 'face') & (df['landmark_index'] == 0)]['x'].isnull() == False)
        
        df_both_hands = df[(df['type'] == 'left_hand') & (df['landmark_index'] == 0)].merge(\
                            df[(df['type'] == 'right_hand') & (df['landmark_index'] == 0)], on='frame', suffixes=('_left', '_right'))
        n_both_hands += df_both_hands[(df_both_hands['x_left'].isnull() == False) &\
                                             (df_both_hands['x_right'].isnull() == False)]['frame'].count()
            
    return n_frames/len(train_sign_sample), n_left_hand/n_frames, n_right_hand/n_frames, n_both_hands/n_frames, n_face/n_frames



for sign in ['cloud', 'thankyou', 'donkey', 'because', 'yellow', 'icecream']:
    total_frames, pct_left, pct_right, pct_both, pct_face = get_details_per_sign(sign)
    print("="*20, f"{sign}", "="*20)
    print(f"Average Number of Frames per Sequence = {total_frames}")
    print(f"Percent of Frames in which a body part exists: Left Hand: {pct_left*100:.02f} %, Right Hand: {pct_right*100:.02f} %, Both Hands: {pct_both*100:.02f} %, Face: {pct_face*100:.02f} %")
    print()


# Kumpulan import jadi satu di sini
import pandas as pd 
# import plotly.express as px
import plotly.io as pio
import plotly.express as px
# untuk animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


# Seolah mau berdiri sendiri

# Load data utamanya
df = pd.read_csv('/kaggle/input/asl-signs/train.csv')

# ingin tahu total parquet
total_parquet = df.shape[0]
print(f"Total file Parquet di df: {total_parquet}")

# ingin tahu ada berapa banyak sign
print(df['sign'].unique().tolist())

# contoh isinya
print("Isi df (data utama):")
print(df.head())


# Ingin tahu berapa jumlah data signnya 

# kata yang mau difilter
sign_name = "hello"
# untuk semua participant untuk kata itu
sign_df = df[df['sign'] == sign_name]

# Print summary untuk semua participant
print(
    f"Jumlah data sign '{sign_name}': {len(sign_df)}\n\n"
    f"Distribusi participant_id:\n{sign_df['participant_id'].value_counts()}\n\n"
    # f"Daftar path Parquet:\n{sign_df['path'].tolist()}"
)

# tertarik hanya untuk 1 participant saja
participant_id = 61333  # Ganti dengan participant_id yang Kakak mau
# filter sign_name = "hello" dan participant_id
filtered_df = df[(df['sign'] == sign_name) & (df['participant_id'] == participant_id)]

# Print summary untuk satu participant_id saja
print(
    f"Jumlah data sign '{sign_name}' untuk participant '{participant_id}': {len(filtered_df)} 'rekaman.'\n\n"
)

parquet_paths = filtered_df['path'].tolist()
result = "Daftar path Parquet:\n" + "\n".join(parquet_paths)
print(result)




# # Ini renderer paling aman untuk Kaggle:
# # pio.renderers.default = "notebook_connected"
# # Alternatif lain (kadang juga works): 
# pio.renderers.default = "notebook"
# # Prepare data for plotly
# freq = sign_df['participant_id'].value_counts().sort_index()
# freq_df = freq.reset_index()
# freq_df.columns = ['participant_id', 'frequency']

# # Interactive bar chart
# fig = px.bar(
#     freq_df,
#     x='participant_id',
#     y='frequency',
#     title=f'Frequency of participant_id for sign \"{sign_name}\"',
#     labels={'participant_id': 'Participant ID', 'frequency': 'Frequency'},
#     color='frequency',
#     color_continuous_scale='Blues'
# )
# fig.update_layout(xaxis_tickangle=-45)
# fig.show()


# Path folder landmark files dari dataset ASL Signs
LANDMARK_FILES_DIR = "/kaggle/input/asl-signs/train_landmark_files"

# Contoh: pilih ID peserta dan file Parquet yang ingin diakses
participant_id = "61333"           # Ganti sesuai kebutuhan
parquet_file = "1474345895.parquet"  # Ganti sesuai kebutuhan

# Cek isi folder peserta
participant_folder = os.path.join(LANDMARK_FILES_DIR, participant_id)
# print(f"Isi folder participant {participant_id}:")
# print(os.listdir(participant_folder))

# Path lengkap file Parquet
parquet_path = os.path.join(participant_folder, parquet_file)
# print(f"\nPath file Parquet yang dipilih: {parquet_path}")

# Baca file Parquet
df_target = pd.read_parquet(parquet_path)
print("Jumlah baris (dan kolom):", df_target.shape)

print("Frame terkecil:", df_target['frame'].min())
print("Frame terbesar:", df_target['frame'].max())

jumlah_frame = df_target['frame'].nunique()
print("Jumlah frame dalam video ini:", jumlah_frame)

baris_per_frame = df_target['frame'].value_counts().sort_index()
print("Jumlah landmark per frame:\n", baris_per_frame)

print("\nPreview isi file Parquet:")
# print(df_target.head())
print(df_target)


# --- DataFrame sudah ada: df_target ---

frames = sorted(df_target['frame'].unique())

def plot_landmarks(ax, df_frame):
    ax.clear()
    types = df_frame['type'].unique()
    colors = {
        'face': 'orange',
        'left_hand': 'blue',
        'right_hand': 'green',
        'pose': 'red'
    }
    for t in types:
        subset = df_frame[df_frame['type'] == t]
        ax.scatter(subset['x'], -subset['y'], label=t, color=colors.get(t, 'black'), s=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 0)
    ax.legend()
    ax.set_title(f"Frame: {df_frame['frame'].iloc[0]}")
    ax.axis('off')

fig, ax = plt.subplots(figsize=(5, 5))

def animate(i):
    frame_num = frames[i]
    df_frame = df_target[df_target['frame'] == frame_num]
    plot_landmarks(ax, df_frame)

ani = FuncAnimation(fig, animate, frames=len(frames), interval=200, repeat=True)
ani.save('animasi_landmark.gif', writer='pillow')
plt.close()

# Tampilkan animasinya
from IPython.display import Image
Image(filename='animasi_landmark.gif')


