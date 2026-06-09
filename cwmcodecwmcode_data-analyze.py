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
# è¯»å�– parquet æ–‡ä»¶
file_path = "/kaggle/input/MABe-mouse-behavior-detection/test_tracking/AdaptableSnail/438887472.parquet"
df = pd.read_parquet(file_path)
#çœ‹æ•°æ�®é›†ï¼Œçœ‹é€�æ¾ˆï¼Œç‰¹å¾�ï¼Œé¼ æ•°
#è¿�è¡Œé«˜åˆ†æ–¹æ¡ˆï¼Œé€�å‡½æ•°æŸ¥çœ‹ï¼Œè¾“å…¥è¾“å‡ºï¼Œå�˜é‡�


num_unique_frames = df["video_frame"].nunique()
print("ä¸�å�Œå¸§çš„æ•°é‡�ï¼š", num_unique_frames)


num_unique_frames = df["mouse_id"].nunique()
print("ä¸�å�Œé¼ çš„æ•°é‡�ï¼š", num_unique_frames)


df.head(1000)


df.tail(2000)


df.iloc[1000:5000]


df.info()         # æŸ¥çœ‹åˆ—å��ã€�æ•°æ�®ç±»å�‹ã€�é��ç©ºæ•°
df.columns.tolist()  # æŸ¥çœ‹æ‰€æœ‰åˆ—å��


file_path = "/kaggle/input/MABe-mouse-behavior-detection/train_annotation/AdaptableSnail/1212811043.parquet"
df1 = pd.read_parquet(file_path)


num_unique_mouses = df1["agent_id"].nunique()
print("ä¸�å�Œé¼ çš„æ•°é‡�ï¼š", num_unique_mouses)


file_path = "/kaggle/input/MABe-mouse-behavior-detection/train_annotation/AdaptableSnail/1260392287.parquet"
df11 = pd.read_parquet(file_path)


num_unique_mouses1 = df11["agent_id"].nunique()
print("ä¸�å�Œé¼ çš„æ•°é‡�ï¼š", num_unique_mouses1)


df1.head(1000)


df1.tail(2000)


df1.iloc[1000:5000]


df1.shape


df1.info()         # æŸ¥çœ‹åˆ—å��ã€�æ•°æ�®ç±»å�‹ã€�é��ç©ºæ•°
df1.columns.tolist()  # æŸ¥çœ‹æ‰€æœ‰åˆ—å��


file_path = "/kaggle/input/MABe-mouse-behavior-detection/train_annotation/BoisterousParrot/1059582964.parquet"
df2 = pd.read_parquet(file_path)


num_unique_mouses2 = df2["agent_id"].nunique()
print("ä¸�å�Œé¼ çš„æ•°é‡�ï¼š", num_unique_mouses2)


df2.head(1000)


df2.tail(2000)


df2.iloc[1000:5000]


df2.shape


df2.columns.tolist()  # æŸ¥çœ‹æ‰€æœ‰åˆ—å��


file_path = "/kaggle/input/MABe-mouse-behavior-detection/train_annotation/CRIM13/1009459450.parquet"
df3 = pd.read_parquet(file_path)


num_unique_mouses3= df3["agent_id"].nunique()
print("ä¸�å�Œé¼ çš„æ•°é‡�ï¼š", num_unique_mouses3)


df3.head(1000)


df3.shape


df3.columns.tolist()  


file_path = "/kaggle/input/MABe-mouse-behavior-detection/train_tracking/BoisterousParrot/1059582964.parquet"
df4 = pd.read_parquet(file_path)


num_unique_frames1 = df4["video_frame"].nunique()
print("ä¸�å�Œå¸§çš„æ•°é‡�ï¼š", num_unique_frames1)


num_unique_mouse = df4["mouse_id"].nunique()
print("ä¸�å�Œé¼ çš„æ•°é‡�ï¼š", num_unique_mouse)


df4.head(1000)


df4.shape


df4.columns.tolist()


file_path = "/kaggle/input/MABe-mouse-behavior-detection/train.csv"
df5 = pd.read_csv(file_path)


df5.head(1000)


num_unique_labs = df5["lab_id"].nunique()
print("ä¸�å�Œå®�éªŒå®¤çš„æ•°é‡�ï¼š", num_unique_labs)



df5.shape





# =====================================================
# ğŸ�­ MABe Challenge - Social Action Recognition in Mice
# Exploratory Data Analysis (EDA) Notebook
# =====================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

plt.style.use('ggplot')

# ============ 1ï¸�âƒ£ åŸºæœ¬è·¯å¾„è®¾ç½® ============

BASE_DIR = "/kaggle/input/MABe-mouse-behavior-detection"

# å�¯ä¿®æ”¹æŸ¥çœ‹è®­ç»ƒæˆ–æµ‹è¯•
tracking_dir = os.path.join(BASE_DIR, "train_tracking")
annotation_dir = os.path.join(BASE_DIR, "train_annotation")

# å…ƒæ•°æ�®æ–‡ä»¶
train_meta_path = os.path.join(BASE_DIR, "train.csv")

print("ğŸ“‚ æ•°æ�®é›†ç›®å½•ç»“æ�„ï¼š")
for folder in os.listdir(BASE_DIR):
    print("  -", folder)


# ============ 2ï¸�âƒ£ æŸ¥çœ‹å…ƒæ•°æ�®æ–‡ä»¶ç»“æ�„ ============

train_meta = pd.read_csv(train_meta_path)
print("\nğŸ“‹ train.csv åŸºæœ¬ä¿¡æ�¯ï¼š")
print(train_meta.head())

print("\nğŸ”¢ å…ƒæ•°æ�®åˆ—ï¼š", train_meta.columns.tolist())

print("\nğŸ“Š å®�éªŒå®¤åˆ†å¸ƒï¼š")
print(train_meta['lab_id'].value_counts())


# ============ 3ï¸�âƒ£ é€‰æ‹©ä¸€ä¸ªè§†é¢‘æ ·æœ¬è¿›è¡Œåˆ†æ�� ============

sample_video = train_meta.iloc[0]['video_id']
sample_lab = train_meta.iloc[0]['lab_id']
print(f"\nğŸ��ï¸� ç¤ºä¾‹è§†é¢‘: {sample_video} | å®�éªŒå®¤: {sample_lab}")

sample_tracking_path = os.path.join(tracking_dir, f"{sample_lab}/{sample_video}.parquet")
sample_annotation_path = os.path.join(annotation_dir, f"{sample_lab}/{sample_video}.parquet")

df_track = pd.read_parquet(sample_tracking_path)
df_anno = pd.read_parquet(sample_annotation_path)

print("\nâœ… Tracking æ•°æ�®ï¼š")
print(df_track.head())
print("\nğŸ“� Tracking æ•°æ�®ç»´åº¦:", df_track.shape)
print("ğŸ“Œ Tracking åˆ—å��:", df_track.columns.tolist())
print("--------------------------------------------------")
print("\nâœ… Annotation æ•°æ�®ï¼š")
print(df_anno.head())
print("\nğŸ“� Annotation æ•°æ�®ç»´åº¦:", df_anno.shape)
print("ğŸ“Œ Annotation åˆ—å��:", df_anno.columns.tolist())


# ============ 4ï¸�âƒ£ Tracking æ•°æ�®åˆ†æ�� ============

# æŸ¥çœ‹ bodypart ç§�ç±»
bodypart_counts = df_track['bodypart'].value_counts()
print("\nğŸ�� Trackingä¸­bodypartç§�ç±»åˆ†å¸ƒ:")
print(bodypart_counts)

print("# å�¯è§†åŒ–èº«ä½“éƒ¨ä½�åˆ†å¸ƒ")
plt.figure(figsize=(10,5))
sns.barplot(x=bodypart_counts.index, y=bodypart_counts.values)
plt.title("Bodyparts Distribution")
plt.xticks(rotation=45)
plt.show()

# é¼ IDåˆ†å¸ƒ
plt.figure(figsize=(8,4))
sns.countplot(x='mouse_id', data=df_track)
plt.title("Mouse ID Distribution")
plt.show()


# ============ 5ï¸�âƒ£ å��æ ‡èŒƒå›´åˆ†æ�� ============

plt.figure(figsize=(6,6))
sns.scatterplot(x='x', y='y', data=df_track.sample(5000))
plt.title("Bodypart Positions Sample (5000 points)")
plt.show()

print("\nğŸ“� å��æ ‡èŒƒå›´ï¼š")
print("x:", (df_track['x'].min(), df_track['x'].max()))
print("y:", (df_track['y'].min(), df_track['y'].max()))


# ============ 6ï¸�âƒ£ Annotation æ•°æ�®åˆ†æ�� ============

print("\nğŸ“Š è¡Œä¸ºç±»åˆ«ç»Ÿè®¡:")
action_counts = df_anno['action'].value_counts()
print(action_counts)

plt.figure(figsize=(10,5))
sns.barplot(x=action_counts.index, y=action_counts.values)
plt.title("Action Label Distribution")
plt.xticks(rotation=45)
plt.show()

# è¡Œä¸ºæŒ�ç»­æ—¶é—´ç»Ÿè®¡
df_anno['duration'] = df_anno['stop_frame'] - df_anno['start_frame']
plt.figure(figsize=(8,4))
sns.histplot(df_anno['duration'], bins=50)
plt.title("Behavior Duration Distribution (frames)")
plt.show()

print("\nâ�±ï¸� å¹³å�‡è¡Œä¸ºæŒ�ç»­å¸§æ•°:", df_anno['duration'].mean())


# ============ 7ï¸�âƒ£ Agent-Target å…³ç³» ============

print("\nğŸ¤� Agent-Target åˆ†æ��:")
print(df_anno[['agent_id', 'target_id']].value_counts().head())

plt.figure(figsize=(6,4))
sns.countplot(x=df_anno['agent_id']==df_anno['target_id'])
plt.title("Is Single-Mouse Behavior?")
plt.xlabel("agent == target")
plt.show()


# ============ 8ï¸�âƒ£ é¼ é—´è·�ç¦»å�¯è§†åŒ–ï¼ˆç¤ºä¾‹ï¼‰ ============

# é€‰å�–ä¸¤å�ªé¼ ï¼ˆå�‡è®¾å­˜åœ¨ mouse_id 1 å’Œ 2ï¼‰
if df_track['mouse_id'].nunique() >= 2:
    m1 = df_track[df_track['mouse_id'] == 1]
    m2 = df_track[df_track['mouse_id'] == 2]
    
    # nose å��æ ‡
    nose1 = m1[m1['bodypart']=='nose'].sort_values('video_frame')
    nose2 = m2[m2['bodypart']=='nose'].sort_values('video_frame')

    # å¯¹é½�å¸§æ•°
    n = min(len(nose1), len(nose2))
    dist = np.sqrt((nose1['x'].iloc[:n] - nose2['x'].iloc[:n])**2 + 
                   (nose1['y'].iloc[:n] - nose2['y'].iloc[:n])**2)

    plt.figure(figsize=(10,4))
    plt.plot(dist[:2000])
    plt.title("Nose-Nose Distance (first 2000 frames)")
    plt.xlabel("Frame")
    plt.ylabel("Distance (pixels)")
    plt.show()
else:
    print("âš ï¸� æ•°æ�®ä¸­å�ªæœ‰ä¸€å�ªé¼ ï¼Œè·³è¿‡è·�ç¦»å�¯è§†åŒ–ã€‚")


# ============ 9ï¸�âƒ£ å…¨å±€è¡Œä¸ºç»Ÿè®¡ï¼ˆå�¯æ‰©å±•ï¼‰ ============

print("\nğŸ“ˆ è¡Œä¸ºç±»å�‹æ€»è§ˆï¼š")
global_counts = []

for lab in tqdm(os.listdir(annotation_dir)):
    lab_path = os.path.join(annotation_dir, lab)
    for file in os.listdir(lab_path):
        fpath = os.path.join(lab_path, file)
        ann = pd.read_parquet(fpath)
        for act, cnt in ann['action'].value_counts().items():
            global_counts.append({'lab': lab, 'action': act, 'count': cnt})

global_df = pd.DataFrame(global_counts)
plt.figure(figsize=(10,5))
sns.barplot(data=global_df, x='action', y='count', estimator=sum)
plt.title("Overall Action Distribution Across Labs")
plt.xticks(rotation=45)
plt.show()

print("âœ… æ•°æ�®æ�¢ç´¢åˆ†æ��å®Œæˆ�ï¼�")


