# å¯¼å…¥æ“�ä½œç³»ç»Ÿç›¸å…³çš„æ¨¡å�—ï¼Œç”¨äº�å¤„ç�†æ–‡ä»¶å’Œç›®å½•
import os
# å¯¼å…¥å�ƒåœ¾å›�æ”¶æ¨¡å�—ï¼Œç”¨äº�æ‰‹åŠ¨ç®¡ç�†å†…å­˜
import gc
# å¯¼å…¥è­¦å‘Šæ¨¡å�—ï¼Œç”¨äº�æ�§åˆ¶è­¦å‘Šä¿¡æ�¯çš„æ˜¾ç¤º
import warnings
# å¯¼å…¥æ—¥å¿—æ¨¡å�—ï¼Œç”¨äº�è®°å½•ç¨‹åº�è¿�è¡Œæ—¶çš„ä¿¡æ�¯
import logging
# å¯¼å…¥æ—¶é—´æ¨¡å�—ï¼Œç”¨äº�å¤„ç�†æ—¶é—´ç›¸å…³çš„æ“�ä½œ
import time
# å¯¼å…¥æ•°å­¦æ¨¡å�—ï¼Œæ��ä¾›æ•°å­¦å‡½æ•°
import math
# å¯¼å…¥OpenCVåº“ï¼Œç”¨äº�å›¾åƒ�å¤„ç�†
import cv2
# å¯¼å…¥Pathæ¨¡å�—ï¼Œç”¨äº�å¤„ç�†æ–‡ä»¶è·¯å¾„
from pathlib import Path

# å¯¼å…¥NumPyåº“ï¼Œç”¨äº�æ•°å€¼è®¡ç®—
import numpy as np
# å¯¼å…¥Pandasåº“ï¼Œç”¨äº�æ•°æ�®å¤„ç�†å’Œåˆ†æ��
import pandas as pd
# å¯¼å…¥librosaåº“ï¼Œç”¨äº�éŸ³é¢‘å¤„ç�†
import librosa
# å¯¼å…¥PyTorchåº“ï¼Œç”¨äº�æ·±åº¦å­¦ä¹ 
import torch
# å¯¼å…¥PyTorchçš„ç¥�ç»�ç½‘ç»œæ¨¡å�—
import torch.nn as nn
# å¯¼å…¥PyTorchçš„å‡½æ•°æ¨¡å�—ï¼ŒåŒ…å�«å¸¸ç”¨çš„å‡½æ•°æ“�ä½œ
import torch.nn.functional as F
# å¯¼å…¥timmåº“ï¼Œæ��ä¾›é¢„è®­ç»ƒçš„æ·±åº¦å­¦ä¹ æ¨¡å�‹
import timm
# å¯¼å…¥tqdmåº“ï¼Œç”¨äº�æ˜¾ç¤ºè¿›åº¦æ�¡
from tqdm.auto import tqdm

# å†�æ¬¡å¯¼å…¥NumPyå’ŒPandasåº“ï¼ˆé‡�å¤�å¯¼å…¥ï¼Œæ— å®�é™…ä½œç”¨ï¼‰
import numpy as np
import pandas as pd

# å¯¼å…¥matplotlib.pyplotæ¨¡å�—ï¼Œç”¨äº�ç»˜å›¾
import matplotlib.pyplot as plt
# å¯¼å…¥seabornåº“ï¼Œç”¨äº�æ•°æ�®å�¯è§†åŒ–
import seaborn as sns
# å¯¼å…¥foliumåº“ï¼Œç”¨äº�åœ°ç�†æ•°æ�®å�¯è§†åŒ–
import folium

# å†�æ¬¡å¯¼å…¥librosaåº“ï¼ˆé‡�å¤�å¯¼å…¥ï¼Œæ— å®�é™…ä½œç”¨ï¼‰
import librosa
# å¯¼å…¥librosa.displayæ¨¡å�—ï¼Œç”¨äº�éŸ³é¢‘å�¯è§†åŒ–
import librosa.display
# å¯¼å…¥IPython.displayæ¨¡å�—ï¼Œç”¨äº�åœ¨Jupyter Notebookä¸­æ’­æ”¾éŸ³é¢‘
from IPython.display import Audio

# å¿½ç•¥æ‰€æœ‰è­¦å‘Šä¿¡æ�¯
warnings.filterwarnings("ignore")
# é…�ç½®æ—¥å¿—è®°å½•çº§åˆ«ä¸ºERRORï¼Œå�ªè®°å½•é”™è¯¯ä¿¡æ�¯
logging.basicConfig(level=logging.ERROR)



# å®šä¹‰ä¸€ä¸ªå��ä¸ºCFGçš„é…�ç½®ç±»ï¼Œç”¨äº�å­˜å‚¨å’Œç®¡ç�†æ¨¡å�‹è®­ç»ƒå’Œæ�¨ç�†çš„å�‚æ•°
class CFG:
    # æµ‹è¯•éŸ³é¢‘æ–‡ä»¶çš„è·¯å¾„
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    # æ��äº¤æ–‡ä»¶çš„è·¯å¾„ï¼ˆæ ·æœ¬æ��äº¤æ–‡ä»¶ï¼‰
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    # é¸Ÿç±»åˆ†ç±»å­¦æ–‡ä»¶çš„è·¯å¾„
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    # é¢„è®­ç»ƒæ¨¡å�‹æ–‡ä»¶çš„è·¯å¾„
    model_path = '/kaggle/input/birdclef-2025-efficientnet-b0'  
    
    # éŸ³é¢‘å¤„ç�†å�‚æ•°
    FS = 32000  # é‡‡æ ·ç�‡ï¼ˆ32kHzï¼‰
    WINDOW_SIZE = 5  # çª—å�£å¤§å°�ï¼ˆç§’ï¼‰
    
    # Melé¢‘è°±å›¾å�‚æ•°
    N_FFT = 1024  # FFTçª—å�£å¤§å°�
    HOP_LENGTH = 512  # å¸§ç§»å¤§å°�
    N_MELS = 128  # Melé¢‘å¸¦æ•°é‡�
    FMIN = 50  # æœ€å°�é¢‘ç�‡
    FMAX = 14000  # æœ€å¤§é¢‘ç�‡
    TARGET_SHAPE = (256, 256)  # ç›®æ ‡å›¾åƒ�å°ºå¯¸
    
    # æ¨¡å�‹ç›¸å…³å�‚æ•°
    model_name = 'efficientnet_b0'  # ä½¿ç”¨çš„æ¨¡å�‹å��ç§°
    in_channels = 1  # è¾“å…¥é€šé�“æ•°ï¼ˆç�°åº¦å›¾åƒ�ä¸º1ï¼‰
    device = 'cpu'  # ä½¿ç”¨çš„è®¾å¤‡ï¼ˆCPUæˆ–GPUï¼‰
    
    # æ�¨ç�†å�‚æ•°
    batch_size = 16  # æ‰¹é‡�å¤§å°�
    use_tta = False  # æ˜¯å�¦ä½¿ç”¨æµ‹è¯•æ—¶æ•°æ�®å¢�å¼ºï¼ˆTest Time Augmentationï¼‰
    tta_count = 3  # TTAçš„æ¬¡æ•°
    threshold = 0.5  # åˆ†ç±»é˜ˆå€¼
    
    # æ¨¡å�‹æŠ˜å� ç›¸å…³å�‚æ•°
    use_specific_folds = False  # æ˜¯å�¦ä½¿ç”¨ç‰¹å®šçš„æ¨¡å�‹æŠ˜å� ï¼ˆFalseè¡¨ç¤ºä½¿ç”¨æ‰€æœ‰æ‰¾åˆ°çš„æ¨¡å�‹ï¼‰
    folds = [0, 1]  # ä½¿ç”¨çš„æ¨¡å�‹æŠ˜å� åˆ—è¡¨ï¼ˆä»…åœ¨use_specific_foldsä¸ºTrueæ—¶ç”Ÿæ•ˆï¼‰
    
    # è°ƒè¯•å�‚æ•°
    debug = False  # æ˜¯å�¦å�¯ç”¨è°ƒè¯•æ¨¡å¼�
    debug_count = 3  # è°ƒè¯•æ¨¡å¼�ä¸‹å¤„ç�†çš„æ ·æœ¬æ•°é‡�

# åˆ›å»ºCFGç±»çš„å®�ä¾‹ï¼Œç”¨äº�è®¿é—®é…�ç½®å�‚æ•°
cfg = CFG()



# è¯»å�–è®­ç»ƒæ•°æ�®æ–‡ä»¶
df = pd.read_csv('../input/birdclef-2025/train.csv')

# è¯»å�–åˆ†ç±»å­¦æ–‡ä»¶
df_taxo = pd.read_csv('../input/birdclef-2025/taxonomy.csv')

# å°†è®­ç»ƒæ•°æ�®ä¸�åˆ†ç±»å­¦æ•°æ�®å�ˆå¹¶ï¼ŒåŸºäº�'primary_label'åˆ—è¿›è¡Œå·¦è¿�æ�¥
# æ·»åŠ åˆ†ç±»å­¦æ•°æ�®ä¸­çš„'inat_taxon_id'å’Œ'class_name'åˆ—åˆ°è®­ç»ƒæ•°æ�®ä¸­
df = pd.merge(left=df, right=df_taxo[['primary_label', 'inat_taxon_id', 'class_name']], how='left', on='primary_label')

# æ‰“å�°å�ˆå¹¶å��çš„æ•°æ�®çš„å‰�å‡ è¡Œï¼Œç”¨äº�æ£€æŸ¥æ•°æ�®æ˜¯å�¦æ­£ç¡®å�ˆå¹¶
print(df.head())

# æ‰“å�°æ•°æ�®çš„åŸºæœ¬ä¿¡æ�¯ï¼ŒåŒ…æ‹¬åˆ—å��ã€�æ•°æ�®ç±»å�‹å’Œé��ç©ºå€¼æ•°é‡�
print(df.info())

# ç»Ÿè®¡å¹¶æ‰“å�°'collection'åˆ—ä¸­å�„å€¼çš„é¢‘æ¬¡ï¼Œç”¨äº�äº†è§£æ•°æ�®åˆ†å¸ƒ
print(df.collection.value_counts())



# åˆ›å»ºä¸€ä¸ªç®€å�•çš„æ•£ç‚¹å›¾ï¼Œæ˜¾ç¤ºæ•°æ�®é›†ä¸­é¸Ÿç±»è®°å½•çš„åœ°ç�†ä½�ç½®åˆ†å¸ƒ
plt.figure(figsize=(12, 6))  # è®¾ç½®å›¾åƒ�çš„å¤§å°�ä¸º12x6è‹±å¯¸

# ä½¿ç”¨Seabornç»˜åˆ¶æ•£ç‚¹å›¾ï¼Œxè½´ä¸ºç»�åº¦ï¼ˆlongitudeï¼‰ï¼Œyè½´ä¸ºçº¬åº¦ï¼ˆlatitudeï¼‰
# æ•°æ�®æ�¥è‡ªdfï¼Œç‚¹çš„é¢œè‰²è®¾ç½®ä¸ºæ·±è“�è‰²ï¼ˆdarkblueï¼‰
sns.scatterplot(data=df, x='longitude', y='latitude', color='darkblue')

plt.grid()  # æ·»åŠ ç½‘æ ¼çº¿ï¼Œä¾¿äº�è§‚å¯Ÿæ•°æ�®åˆ†å¸ƒ
plt.show()  # æ˜¾ç¤ºå›¾åƒ�



# åŠ è½½éŸ³é¢‘æ–‡ä»¶
filename = 'XC112602.ogg'  # éŸ³é¢‘æ–‡ä»¶å��
y, sr = librosa.load('../input/birdclef-2025/train_audio/banana/' + filename)  # ä½¿ç”¨librosaåŠ è½½éŸ³é¢‘æ–‡ä»¶ï¼Œyä¸ºéŸ³é¢‘æ•°æ�®ï¼Œsrä¸ºé‡‡æ ·ç�‡

# æ’­æ”¾éŸ³é¢‘
Audio(y, rate=sr)  # ä½¿ç”¨IPython.display.Audioæ’­æ”¾éŸ³é¢‘

# ç»˜åˆ¶éŸ³é¢‘æ³¢å½¢å›¾
plt.figure(figsize=(14, 5))  # è®¾ç½®å›¾åƒ�å¤§å°�ä¸º14x5è‹±å¯¸
plt.plot(y, color='darkblue')  # ç»˜åˆ¶éŸ³é¢‘æ³¢å½¢ï¼Œé¢œè‰²ä¸ºæ·±è“�è‰²
plt.grid()  # æ·»åŠ ç½‘æ ¼çº¿
plt.show()  # æ˜¾ç¤ºå›¾åƒ�



# æ’­æ”¾éŸ³é¢‘
Audio(y, rate=sr)  # ä½¿ç”¨IPython.display.Audioæ’­æ”¾éŸ³é¢‘



# ç¤ºä¾‹éŸ³é¢‘è·¯å¾„
# audio_path = "../input/birdclef-2025/train_audio/XC12345.ogg"  # æ›¿æ�¢ä¸ºå®�é™…çš„éŸ³é¢‘æ–‡ä»¶è·¯å¾„

# åŠ è½½éŸ³é¢‘æ–‡ä»¶å¹¶è�·å�–å…¶æŒ�ç»­æ—¶é—´
# y, sr = librosa.load(audio_path, sr=None)  # åŠ è½½éŸ³é¢‘ï¼Œyä¸ºéŸ³é¢‘æ•°æ�®ï¼Œsrä¸ºé‡‡æ ·ç�‡
# duration = librosa.get_duration(y=y, sr=sr)  # è®¡ç®—éŸ³é¢‘çš„æŒ�ç»­æ—¶é—´
# print(f"Duration: {duration:.2f} seconds")  # æ‰“å�°éŸ³é¢‘æŒ�ç»­æ—¶é—´ï¼Œä¿�ç•™ä¸¤ä½�å°�æ•°

# ç»˜åˆ¶éŸ³é¢‘æ³¢å½¢å›¾
# plt.figure(figsize=(10, 4))  # è®¾ç½®å›¾åƒ�å¤§å°�ä¸º10x4è‹±å¯¸
# librosa.display.waveshow(y, sr=sr)  # ç»˜åˆ¶éŸ³é¢‘æ³¢å½¢
# plt.title('Waveform of Sample Audio')  # è®¾ç½®å›¾åƒ�æ ‡é¢˜
# plt.xlabel('Time (s)')  # è®¾ç½®xè½´æ ‡ç­¾
# plt.ylabel('Amplitude')  # è®¾ç½®yè½´æ ‡ç­¾
# plt.show()  # æ˜¾ç¤ºå›¾åƒ�

# å®šä¹‰ç»˜åˆ¶æ¢…å°”é¢‘è°±å›¾çš„å‡½æ•°
# def plot_spectrogram(audio_path):
#     y, sr = librosa.load(audio_path, sr=None)  # åŠ è½½éŸ³é¢‘
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)  # è®¡ç®—æ¢…å°”é¢‘è°±å›¾
#     S_dB = librosa.power_to_db(S, ref=np.max)  # å°†é¢‘è°±å›¾è½¬æ�¢ä¸ºåˆ†è´�å�•ä½�

#     plt.figure(figsize=(10, 4))  # è®¾ç½®å›¾åƒ�å¤§å°�ä¸º10x4è‹±å¯¸
#     librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')  # æ˜¾ç¤ºæ¢…å°”é¢‘è°±å›¾
#     plt.colorbar(format='%+2.0f dB')  # æ·»åŠ é¢œè‰²æ�¡
#     plt.title('Mel Spectrogram')  # è®¾ç½®å›¾åƒ�æ ‡é¢˜
#     plt.show()  # æ˜¾ç¤ºå›¾åƒ�

# ç¤ºä¾‹ï¼šç»˜åˆ¶æŒ‡å®šéŸ³é¢‘çš„æ¢…å°”é¢‘è°±å›¾
# plot_spectrogram(audio_path)  # è°ƒç”¨å‡½æ•°ç»˜åˆ¶é¢‘è°±å›¾



# å¯¼å…¥librosaåº“ï¼Œç”¨äº�éŸ³é¢‘å¤„ç�†
import librosa

# å®šä¹‰æ­£ç¡®çš„éŸ³é¢‘æ–‡ä»¶è·¯å¾„
audio_path = "../input/birdclef-2025/train_audio/greani1/XC132190.ogg"

# åŠ è½½éŸ³é¢‘æ–‡ä»¶
y, sr = librosa.load(audio_path, sr=None)
# æ‰“å�°éŸ³é¢‘åŠ è½½æˆ�åŠŸä¿¡æ�¯å�ŠéŸ³é¢‘æŒ�ç»­æ—¶é—´
print(f"Audio Loaded! âœ… Duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")

# å¯¼å…¥globåº“ï¼Œç”¨äº�æ–‡ä»¶è·¯å¾„åŒ¹é…�
import glob

# è�·å�–æ‰€æœ‰å­�ç›®å½•ä¸­çš„OGGæ–‡ä»¶
all_audio_files = glob.glob("../input/birdclef-2025/train_audio/**/*.ogg", recursive=True)

# åŠ è½½å¹¶åˆ†æ��å‰�3ä¸ªéŸ³é¢‘æ–‡ä»¶
for audio_path in all_audio_files[:3]:
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Loaded: {audio_path.split('/')[-1]} | Duration: {duration:.2f} seconds")

# å¯¼å…¥librosa.displayå’Œmatplotlib.pyplotåº“ï¼Œç”¨äº�éŸ³é¢‘å�¯è§†åŒ–
import librosa.display
import matplotlib.pyplot as plt

# ç»˜åˆ¶éŸ³é¢‘æ³¢å½¢å›¾
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform of Sample Audio')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# ç»˜åˆ¶æ¢…å°”é¢‘è°±å›¾
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()



# å®šä¹‰ä¸€ä¸ªå‡½æ•°ï¼Œç”¨äº�ä»�éŸ³é¢‘æ–‡ä»¶ä¸­æ��å�–ç‰¹å¾�
def extract_features(audio_path, max_pad_len=128):
    # åŠ è½½éŸ³é¢‘æ–‡ä»¶ï¼Œé‡‡æ ·ç�‡ä¸º32 kHz
    y, sr = librosa.load(audio_path, sr=32000)
    # è®¡ç®—æ¢…å°”é¢‘è°±å›¾
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    # å°†æ¢…å°”é¢‘è°±å›¾è½¬æ�¢ä¸ºåˆ†è´�å�•ä½�
    S_db = librosa.power_to_db(S, ref=np.max)

    # å¯¹é¢‘è°±å›¾è¿›è¡Œå¡«å……æˆ–æˆªæ–­ï¼Œä½¿å…¶é•¿åº¦å›ºå®šä¸ºmax_pad_len
    if S_db.shape[1] < max_pad_len:  # å¦‚æ�œé¢‘è°±å›¾é•¿åº¦å°�äº�max_pad_len
        pad_width = max_pad_len - S_db.shape[1]  # è®¡ç®—éœ€è¦�å¡«å……çš„å®½åº¦
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')  # åœ¨å�³ä¾§å¡«å……0
    else:  # å¦‚æ�œé¢‘è°±å›¾é•¿åº¦å¤§äº�max_pad_len
        S_db = S_db[:, :max_pad_len]  # æˆªæ–­å�³ä¾§å¤šä½™çš„éƒ¨åˆ†

    return S_db  # è¿”å›�å¤„ç�†å��çš„é¢‘è°±å›¾

# ç¤ºä¾‹ï¼šæ��å�–éŸ³é¢‘æ–‡ä»¶çš„ç‰¹å¾�
audio_path = "../input/birdclef-2025/train_audio/greani1/XC132190.ogg"  # éŸ³é¢‘æ–‡ä»¶è·¯å¾„
features = extract_features(audio_path)  # è°ƒç”¨å‡½æ•°æ��å�–ç‰¹å¾�
print(f"Extracted Features Shape: {features.shape}")  # æ‰“å�°æ��å�–çš„ç‰¹å¾�çš„å½¢çŠ¶



# è¯»å�–è®­ç»ƒæ•°æ�®æ–‡ä»¶
train_df = pd.read_csv('../input/birdclef-2025/train.csv')

# è¯»å�–åˆ†ç±»å­¦æ–‡ä»¶
taxonomy_df = pd.read_csv('../input/birdclef-2025/taxonomy.csv')

# å°†è®­ç»ƒæ•°æ�®ä¸�åˆ†ç±»å­¦æ•°æ�®å�ˆå¹¶ï¼ŒåŸºäº�'primary_label'åˆ—è¿›è¡Œå·¦è¿�æ�¥
# æ·»åŠ åˆ†ç±»å­¦æ•°æ�®ä¸­çš„'class_name'åˆ—åˆ°è®­ç»ƒæ•°æ�®ä¸­
train_df = pd.merge(train_df, taxonomy_df[['primary_label', 'class_name']], how='left', on='primary_label')

# æ‰“å�°å�ˆå¹¶å��çš„æ•°æ�®çš„å‰�å‡ è¡Œï¼Œç”¨äº�æ£€æŸ¥æ•°æ�®æ˜¯å�¦æ­£ç¡®å�ˆå¹¶
print(train_df.head())



# å¯¼å…¥å¿…è¦�çš„åº“
import os
import numpy as np
import pandas as pd
import librosa
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# å®šä¹‰å‡½æ•°ï¼šå¯¹é¢‘è°±å›¾è¿›è¡Œå¡«å……æˆ–æˆªæ–­ï¼Œä½¿å…¶å›ºå®šå¤§å°�ä¸º128x128
def pad_or_truncate(S, max_len=128):
    """Pad or truncate the spectrogram to a fixed size of 128x128."""
    if S.shape[1] < max_len:  # å¦‚æ�œé¢‘è°±å›¾é•¿åº¦å°�äº�max_len
        # åœ¨å�³ä¾§å¡«å……0
        pad_width = max_len - S.shape[1]
        S = np.pad(S, ((0, 0), (0, pad_width)), mode='constant')
    else:  # å¦‚æ�œé¢‘è°±å›¾é•¿åº¦å¤§äº�max_len
        # æˆªæ–­å�³ä¾§å¤šä½™çš„éƒ¨åˆ†
        S = S[:, :max_len]

    return S

# å®šä¹‰å‡½æ•°ï¼šä»�éŸ³é¢‘æ–‡ä»¶ä¸­æ��å�–ç‰¹å¾�ï¼Œå¹¶è¿›è¡Œå¡«å……æˆ–æˆªæ–­
def extract_features(audio_path):
    try:
        # åŠ è½½éŸ³é¢‘æ–‡ä»¶
        y, sr = librosa.load(audio_path, sr=None)
        # è®¡ç®—æ¢…å°”é¢‘è°±å›¾
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        # å°†æ¢…å°”é¢‘è°±å›¾è½¬æ�¢ä¸ºåˆ†è´�å�•ä½�
        S_dB = librosa.power_to_db(S, ref=np.max)

        # å¯¹é¢‘è°±å›¾è¿›è¡Œå¡«å……æˆ–æˆªæ–­ï¼Œä½¿å…¶å›ºå®šå¤§å°�ä¸º128x128
        S_fixed = pad_or_truncate(S_dB, max_len=128)
        
        return S_fixed
    except Exception as e:
        # å¦‚æ�œå¤„ç�†å¤±è´¥ï¼Œæ‰“å�°é”™è¯¯ä¿¡æ�¯
        print(f"â�Œ Error processing {audio_path}: {e}")
        return None

# ä»�è®­ç»ƒæ•°æ�®ä¸­éš�æœºæŠ½å�–100ä¸ªæ ·æœ¬ï¼ˆä¸ºäº†å¿«é€Ÿæµ‹è¯•ï¼‰
sample_df = train_df.sample(100, random_state=42)  # ä½¿ç”¨42ä½œä¸ºéš�æœºç§�å­�ä»¥ç¡®ä¿�å�¯é‡�å¤�æ€§
X, y = [], []  # åˆ�å§‹åŒ–ç‰¹å¾�å’Œæ ‡ç­¾åˆ—è¡¨

# é��å�†æ ·æœ¬æ•°æ�®ï¼Œæ��å�–ç‰¹å¾�
for i, row in sample_df.iterrows():
    # æ�„é€ éŸ³é¢‘æ–‡ä»¶è·¯å¾„
    audio_file = f"../input/birdclef-2025/train_audio/{row['filename']}"
    if os.path.exists(audio_file):  # æ£€æŸ¥æ–‡ä»¶æ˜¯å�¦å­˜åœ¨
        # æ��å�–ç‰¹å¾�
        feature = extract_features(audio_file)
        
        # æ£€æŸ¥ç‰¹å¾�æ˜¯å�¦æœ‰æ•ˆä¸”å½¢çŠ¶ä¸º128x128
        if feature is not None and feature.shape == (128, 128):
            X.append(feature)
            y.append(row['primary_label'])
        else:
            # å¦‚æ�œç‰¹å¾�æ— æ•ˆï¼Œè·³è¿‡å¹¶æ‰“å�°è­¦å‘Š
            print(f"âš ï¸� Skipping {row['filename']} due to invalid feature shape.")

# æ£€æŸ¥æ˜¯å�¦æœ‰æœ‰æ•ˆçš„ç‰¹å¾�å’Œæ ‡ç­¾
if len(X) == 0 or len(y) == 0:
    raise ValueError("â�Œ No valid audio files were processed. Check file paths and feature extraction!")

# å°†ç‰¹å¾�å’Œæ ‡ç­¾è½¬æ�¢ä¸ºNumPyæ•°ç»„
X = np.array(X)
# å°†ç‰¹å¾�é‡�å¡‘ä¸ºé€‚å�ˆCNNè¾“å…¥çš„å½¢çŠ¶ï¼ˆæ ·æœ¬æ•°, é«˜åº¦, å®½åº¦, é€šé�“æ•°ï¼‰
X = X.reshape(X.shape[0], 128, 128, 1)
# å°†æ–‡æœ¬æ ‡ç­¾ç¼–ç �ä¸ºæ•´æ•°ï¼Œå¹¶è½¬æ�¢ä¸ºone-hotç¼–ç �
y_encoded, y_labels = pd.factorize(y)
y = to_categorical(y_encoded)

# å°†æ•°æ�®é›†æ‹†åˆ†ä¸ºè®­ç»ƒé›†å’ŒéªŒè¯�é›†
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# æ‰“å�°è®­ç»ƒé›†å’ŒéªŒè¯�é›†çš„å½¢çŠ¶
print(f"âœ… Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")



 


# æ‰“å�°æˆ�åŠŸå¤„ç�†çš„éŸ³é¢‘æ–‡ä»¶æ•°é‡�
print(f"âœ… Files successfully processed: {len(X)}")

# æ‰“å�°æ•°æ�®é›†ä¸­å”¯ä¸€çš„ç±»åˆ«æ•°é‡�
print(f"âœ… Unique classes: {len(np.unique(y_encoded))}")



# å¯¼å…¥å¿…è¦�çš„Kerasæ¨¡å�—
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam

# å®šä¹‰ä¼˜åŒ–çš„CNNæ¨¡å�‹
def build_cnn_model(input_shape=(128, 128, 1), num_classes=100):
    model = Sequential()

    # å�·ç§¯å�—1
    model.add(Conv2D(16, (3, 3), activation="relu", input_shape=input_shape, padding="same"))
    model.add(MaxPooling2D((2, 2)))

    # å�·ç§¯å�—2
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))

    # å�·ç§¯å�—3
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))

    # å±•å¹³å±‚å’Œå…¨è¿�æ�¥å±‚
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))  # é™�ä½�Dropoutç�‡ä»¥ä¿�ç•™æ›´å¤šæœ‰ç”¨ç‰¹å¾�

    # è¾“å‡ºå±‚
    model.add(Dense(num_classes, activation="softmax"))

    # ä½¿ç”¨è¾ƒä½�çš„å­¦ä¹ ç�‡ç¼–è¯‘æ¨¡å�‹
    optimizer = Adam(learning_rate=1e-4)  # è¾ƒä½�çš„å­¦ä¹ ç�‡ä»¥ç¨³å®šè®­ç»ƒ
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# æ�„å»ºå¹¶ç¼–è¯‘æ”¹è¿›å��çš„æ¨¡å�‹
model = build_cnn_model(input_shape=(128, 128, 1), num_classes=len(y_labels))

# æ‰“å�°æ¨¡å�‹æ‘˜è¦�
model.summary()



# å¯¼å…¥å¿…è¦�çš„åº“
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam

# ğŸ“š åˆ›å»ºè™šæ‹Ÿæ•°æ�®é›†ï¼ˆæ›¿æ�¢ä¸ºçœŸå®�æ•°æ�®ï¼‰
X_train = np.random.rand(300, 64, 64, 3)  # 300ä¸ª64x64çš„3é€šé�“å›¾åƒ�
y_train = np.random.randint(0, 5, 300)  # 300ä¸ªæ ‡ç­¾ï¼ŒèŒƒå›´ä¸º0åˆ°4
X_val = np.random.rand(50, 64, 64, 3)  # 50ä¸ª64x64çš„3é€šé�“å›¾åƒ�
y_val = np.random.randint(0, 5, 50)  # 50ä¸ªæ ‡ç­¾ï¼ŒèŒƒå›´ä¸º0åˆ°4

# ğŸ§  å°†æ ‡ç­¾è½¬æ�¢ä¸ºOne-Hotç¼–ç �
y_train_onehot = tf.keras.utils.to_categorical(y_train, 5)  # 5ä¸ªç±»åˆ«
y_val_onehot = tf.keras.utils.to_categorical(y_val, 5)

# ğŸ�¨ æ�„å»ºæ¨¡å�‹
model = Sequential()

# ğŸ”¥ å�·ç§¯å±‚1
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))  # 64ä¸ª3x3çš„å�·ç§¯æ ¸
model.add(BatchNormalization())  # æ‰¹é‡�å½’ä¸€åŒ–
model.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2çš„æœ€å¤§æ± åŒ–

# ğŸ”¥ å�·ç§¯å±‚2
model.add(Conv2D(128, (3, 3), activation='relu'))  # 128ä¸ª3x3çš„å�·ç§¯æ ¸
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# ğŸ”¥ å�·ç§¯å±‚3
model.add(Conv2D(256, (3, 3), activation='relu'))  # 256ä¸ª3x3çš„å�·ç§¯æ ¸
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# ğŸ§  å±•å¹³å±‚å’Œå…¨è¿�æ�¥å±‚
model.add(Flatten())  # å±•å¹³å¤šç»´è¾“å…¥
model.add(Dense(512, activation='relu'))  # 512ä¸ªç¥�ç»�å…ƒçš„å…¨è¿�æ�¥å±‚
model.add(Dropout(0.5))  # 50%çš„Dropout
model.add(Dense(5, activation='softmax'))  # è¾“å‡ºå±‚ï¼Œ5ä¸ªç±»åˆ«

# âš¡ï¸� ç¼–è¯‘æ¨¡å�‹
model.compile(
    optimizer='adam',  # ä½¿ç”¨Adamä¼˜åŒ–å™¨
    loss='categorical_crossentropy',  # å¤šåˆ†ç±»äº¤å�‰ç†µæ�Ÿå¤±å‡½æ•°
    metrics=['accuracy']  # è¯„ä¼°æŒ‡æ ‡ä¸ºå‡†ç¡®ç�‡
)

# ğŸ�‹ï¸�â€�â™‚ï¸� è®­ç»ƒæ¨¡å�‹
history = model.fit(
    X_train / 255.0, y_train_onehot,  # è®­ç»ƒæ•°æ�®ï¼ˆå½’ä¸€åŒ–ï¼‰å’Œæ ‡ç­¾
    validation_data=(X_val / 255.0, y_val_onehot),  # éªŒè¯�æ•°æ�®ï¼ˆå½’ä¸€åŒ–ï¼‰å’Œæ ‡ç­¾
    epochs=5,  # è®­ç»ƒ5ä¸ªepoch
    batch_size=32,  # æ‰¹é‡�å¤§å°�ä¸º32
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),  # åŠ¨æ€�è°ƒæ•´å­¦ä¹ ç�‡
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)  # æ—©å�œç­–ç•¥
    ]
)

# ğŸ“ˆ è¯„ä¼°æ¨¡å�‹
val_loss, val_acc = model.evaluate(X_val / 255.0, y_val_onehot)  # åœ¨éªŒè¯�é›†ä¸Šè¯„ä¼°
print(f"âœ… Final Validation Accuracy: {val_acc:.4f}")  # æ‰“å�°æœ€ç»ˆçš„éªŒè¯�å‡†ç¡®ç�‡



# print(y_train[:5])  # If this looks like [0, 1, 2, 3, 4] â€” use sparse_categorical_crossentropy
# print(y_train_onehot[:5])  # If this looks like one-hot â€” use categorical_crossentropy



# import numpy as np
# unique, counts = np.unique(y_train, return_counts=True)
# print(dict(zip(unique, counts)))
# print(X_val.shape, y_val.shape)
# print(np.argmax(y_val[:5], axis=1))  # Check first few labels
# preds = model.predict(X_val / 255.0)
# print(np.argmax(preds[:5], axis=1))  # Check predictions



# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# # ğŸ“¸ Load MobileNetV2 with pre-trained ImageNet weights, excluding the top layer
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# # ğŸ› ï¸� Add custom classification head
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.5)(x)  # Add dropout to reduce overfitting
# predictions = Dense(5, activation='softmax')(x)

# # ğŸ§  Create final model
# model = Model(inputs=base_model.input, outputs=predictions)

# # ğŸ§Š Freeze base model layers initially
# for layer in base_model.layers:
#     layer.trainable = False

# # ğŸš€ Compile the model with a small learning rate for initial training
# model.compile(
#     optimizer=Adam(learning_rate=1e-4),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # ğŸ“‰ Callbacks for better training
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # ğŸ�‹ï¸�â€�â™‚ï¸� Train the model (initial training with frozen base layers)
# history = model.fit(
#     X_train / 255.0, y_train_onehot,
#     validation_data=(X_val / 255.0, y_val_onehot),
#     epochs=10,
#     batch_size=32,
#     callbacks=[reduce_lr, early_stop]
# )

# # ğŸ”“ Unfreeze some of the top layers in base_model for fine-tuning
# for layer in base_model.layers[-20:]:
#     layer.trainable = True

# # ğŸ†™ Recompile with a smaller learning rate for fine-tuning
# model.compile(
#     optimizer=Adam(learning_rate=1e-5),  # Smaller LR for fine-tuning
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # ğŸ�¯ Fine-tuning the model with more epochs
# fine_tune_history = model.fit(
#     X_train / 255.0, y_train_onehot,
#     validation_data=(X_val / 255.0, y_val_onehot),
#     epochs=10,
#     batch_size=32,
#     callbacks=[reduce_lr, early_stop]
# )



# è¯»å�–åˆ†ç±»å­¦CSVæ–‡ä»¶
taxonomy_df = pd.read_csv(cfg.taxonomy_csv)

# è�·å�–æ‰€æœ‰ç‰©ç§�çš„primary_labelï¼ˆå”¯ä¸€æ ‡è¯†ï¼‰å¹¶è½¬æ�¢ä¸ºåˆ—è¡¨
species_ids = taxonomy_df['primary_label'].tolist()

# è®¡ç®—ç±»åˆ«çš„æ•°é‡�ï¼ˆå�³ç‰©ç§�çš„æ€»æ•°ï¼‰
num_classes = len(species_ids)



# å¯¼å…¥PyTorchçš„ç¥�ç»�ç½‘ç»œæ¨¡å�—
import torch.nn as nn

# å®šä¹‰BirdCLEFæ¨¡å�‹ç±»
class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()  # è°ƒç”¨çˆ¶ç±»çš„åˆ�å§‹åŒ–æ–¹æ³•
        self.cfg = cfg  # ä¿�å­˜é…�ç½®å¯¹è±¡
        
        # ä½¿ç”¨timmåº“åˆ›å»ºé¢„è®­ç»ƒæ¨¡å�‹ï¼ˆä¸�åŠ è½½é¢„è®­ç»ƒæ�ƒé‡�ï¼‰
        self.backbone = timm.create_model(
            cfg.model_name,  # æ¨¡å�‹å��ç§°ï¼ˆå¦‚'efficientnet_b0'æˆ–'resnet18'ï¼‰
            pretrained=False,  # ä¸�ä½¿ç”¨é¢„è®­ç»ƒæ�ƒé‡�
            in_chans=cfg.in_channels,  # è¾“å…¥é€šé�“æ•°
            drop_rate=0.2,  # Dropoutç�‡
            drop_path_rate=0.2  # DropPathç�‡
        )
        
        # æ ¹æ�®æ¨¡å�‹å��ç§°è�·å�–backboneçš„è¾“å‡ºç‰¹å¾�ç»´åº¦
        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features  # EfficientNetçš„ç‰¹å¾�ç»´åº¦
            self.backbone.classifier = nn.Identity()  # ç§»é™¤åˆ†ç±»å™¨
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features  # ResNetçš„ç‰¹å¾�ç»´åº¦
            self.backbone.fc = nn.Identity()  # ç§»é™¤å…¨è¿�æ�¥å±‚
        else:
            backbone_out = self.backbone.get_classifier().in_features  # å…¶ä»–æ¨¡å�‹çš„ç‰¹å¾�ç»´åº¦
            self.backbone.reset_classifier(0, '')  # é‡�ç½®åˆ†ç±»å™¨
        
        # å®šä¹‰è‡ªé€‚åº”å¹³å�‡æ± åŒ–å±‚
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out  # ä¿�å­˜ç‰¹å¾�ç»´åº¦
        # å®šä¹‰åˆ†ç±»å™¨ï¼ˆå…¨è¿�æ�¥å±‚ï¼‰
        self.classifier = nn.Linear(backbone_out, num_classes)
        
    def forward(self, x):
        # æ��å�–ç‰¹å¾�
        features = self.backbone(x)
        
        # å¦‚æ�œè¾“å‡ºæ˜¯å­—å…¸ï¼Œæ��å�–'features'é”®çš„å€¼
        if isinstance(features, dict):
            features = features['features']
            
        # å¦‚æ�œç‰¹å¾�ç»´åº¦ä¸º4ï¼ˆå¦‚[batch_size, channels, height, width]ï¼‰ï¼Œè¿›è¡Œæ± åŒ–å’Œå±•å¹³
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        # é€šè¿‡åˆ†ç±»å™¨ç”Ÿæˆ�logits
        logits = self.classifier(features)
        return logits



import numpy as np
import librosa
import cv2

def audio2melspec(audio_data, cfg):
    """å°†éŸ³é¢‘æ•°æ�®è½¬æ�¢ä¸ºæ¢…å°”é¢‘è°±å›¾"""
    # æ£€æŸ¥éŸ³é¢‘æ•°æ�®ä¸­æ˜¯å�¦å­˜åœ¨NaNå€¼ï¼Œå¦‚æ�œå­˜åœ¨åˆ™ç”¨å�‡å€¼å¡«å……
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    '''
    # è®¡ç®—æ¢…å°”é¢‘è°±å›¾
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,  # éŸ³é¢‘æ•°æ�®
        sr=cfg.FS,  # é‡‡æ ·ç�‡
        n_fft=2048,       # å¢�åŠ FFTçª—å�£å¤§å°�
        hop_length=512,   # å¢�åŠ å¸§ç§»
        n_mels=128,       # å¢�åŠ æ¢…å°”é¢‘å¸¦æ•°
        fmin=50,          # è°ƒæ•´æœ€ä½�é¢‘ç�‡
        fmax=8000         # è°ƒæ•´æœ€é«˜é¢‘ç�‡
    )
    '''

    
    # è®¡ç®—æ¢…å°”é¢‘è°±å›¾
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,  # éŸ³é¢‘æ•°æ�®
        sr=cfg.FS,  # é‡‡æ ·ç�‡
        n_fft=cfg.N_FFT,  # FFTçª—å�£å¤§å°�
        hop_length=cfg.HOP_LENGTH,  # å¸§ç§»
        n_mels=cfg.N_MELS,  # æ¢…å°”é¢‘å¸¦æ•°
        fmin=cfg.FMIN,  # æœ€ä½�é¢‘ç�‡
        fmax=cfg.FMAX,  # æœ€é«˜é¢‘ç�‡
        power=2.0  # åŠŸç�‡è°±çš„æŒ‡æ•°
    )

    
    # å°†æ¢…å°”é¢‘è°±å›¾è½¬æ�¢ä¸ºåˆ†è´�å�•ä½�
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # å¯¹æ¢…å°”é¢‘è°±å›¾è¿›è¡Œå½’ä¸€åŒ–
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def process_audio_segment(audio_data, cfg):
    """å¤„ç�†éŸ³é¢‘ç‰‡æ®µä»¥è�·å�–æ¢…å°”é¢‘è°±å›¾"""
    # å¦‚æ�œéŸ³é¢‘æ•°æ�®é•¿åº¦å°�äº�ç›®æ ‡é•¿åº¦ï¼Œåˆ™ç”¨0å¡«å……
    if len(audio_data) < cfg.FS * cfg.WINDOW_SIZE:
        audio_data = np.pad(audio_data, 
                          (0, cfg.FS * cfg.WINDOW_SIZE - len(audio_data)), 
                          mode='constant')
    # æ·»åŠ éš�æœºæ—¶é—´æ‹‰ä¼¸
    if cfg.use_augmentation:
        stretch_factor = np.random.uniform(0.8, 1.2)
        audio_data = librosa.effects.time_stretch(audio_data, rate=stretch_factor)
    
    # å°†éŸ³é¢‘æ•°æ�®è½¬æ�¢ä¸ºæ¢…å°”é¢‘è°±å›¾
    mel_spec = audio2melspec(audio_data, cfg)
    
    # å¦‚æ�œéœ€è¦�ï¼Œè°ƒæ•´æ¢…å°”é¢‘è°±å›¾çš„å°ºå¯¸ä¸ºç›®æ ‡å½¢çŠ¶
    if mel_spec.shape != cfg.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        
    return mel_spec.astype(np.float32)



from pathlib import Path
import torch
import librosa
import numpy as np

def find_model_files(cfg):
    """
    åœ¨æŒ‡å®šçš„æ¨¡å�‹ç›®å½•ä¸­æŸ¥æ‰¾æ‰€æœ‰.pthæ¨¡å�‹æ–‡ä»¶
    """
    model_files = []
    
    model_dir = Path(cfg.model_path)
    
    # é€’å½’æŸ¥æ‰¾æ‰€æœ‰.pthæ–‡ä»¶
    for path in model_dir.glob('**/*.pth'):
        model_files.append(str(path))
    
    return model_files

def load_models(cfg, num_classes):
    """
    åŠ è½½æ‰€æœ‰æ‰¾åˆ°çš„æ¨¡å�‹æ–‡ä»¶å¹¶å‡†å¤‡ç”¨äº�é›†æˆ�
    """
    models = []
    
    # æŸ¥æ‰¾æ¨¡å�‹æ–‡ä»¶
    model_files = find_model_files(cfg)
    
    # å¦‚æ�œæ²¡æœ‰æ‰¾åˆ°æ¨¡å�‹æ–‡ä»¶ï¼Œæ‰“å�°è­¦å‘Šå¹¶è¿”å›�ç©ºåˆ—è¡¨
    if not model_files:
        print(f"Warning: No model files found under {cfg.model_path}!")
        return models
    
    print(f"Found a total of {len(model_files)} model files.")
    
    # å¦‚æ�œæŒ‡å®šäº†ç‰¹å®šçš„foldsï¼Œè¿‡æ»¤æ¨¡å�‹æ–‡ä»¶
    if cfg.use_specific_folds:
        filtered_files = []
        for fold in cfg.folds:
            fold_files = [f for f in model_files if f"fold{fold}" in f]
            filtered_files.extend(fold_files)
        model_files = filtered_files
        print(f"Using {len(model_files)} model files for the specified folds ({cfg.folds}).")
    
    # åŠ è½½æ¯�ä¸ªæ¨¡å�‹æ–‡ä»¶
    for model_path in model_files:
        try:
            print(f"Loading model: {model_path}")
            # åŠ è½½æ¨¡å�‹æ£€æŸ¥ç‚¹
            checkpoint = torch.load(model_path, map_location=torch.device(cfg.device))
            
            # åˆ�å§‹åŒ–æ¨¡å�‹å¹¶åŠ è½½æ�ƒé‡�
            model = BirdCLEFModel(cfg, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(cfg.device)
            model.eval()  # è®¾ç½®ä¸ºè¯„ä¼°æ¨¡å¼�
            
            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
    
    return models

def predict_on_spectrogram(audio_path, models, cfg, species_ids):
    """
    å¤„ç�†å�•ä¸ªéŸ³é¢‘æ–‡ä»¶å¹¶é¢„æµ‹æ¯�ä¸ª5ç§’ç‰‡æ®µçš„ç‰©ç§�å­˜åœ¨æƒ…å†µ
    """
    predictions = []
    row_ids = []
    soundscape_id = Path(audio_path).stem  # è�·å�–éŸ³é¢‘æ–‡ä»¶å��ï¼ˆä¸�å�«æ‰©å±•å��ï¼‰
    
    try:
        print(f"Processing {soundscape_id}")
        # åŠ è½½éŸ³é¢‘æ•°æ�®
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        
        # è®¡ç®—æ€»ç‰‡æ®µæ•°
        total_segments = int(len(audio_data) / (cfg.FS * cfg.WINDOW_SIZE))
        
        # é��å�†æ¯�ä¸ªç‰‡æ®µ
        for segment_idx in range(total_segments):
            start_sample = segment_idx * cfg.FS * cfg.WINDOW_SIZE
            end_sample = start_sample + cfg.FS * cfg.WINDOW_SIZE
            segment_audio = audio_data[start_sample:end_sample]
            
            # ç”Ÿæˆ�ç‰‡æ®µID
            end_time_sec = (segment_idx + 1) * cfg.WINDOW_SIZE
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)

            # å¦‚æ�œå�¯ç”¨TTAï¼ˆæµ‹è¯•æ—¶å¢�å¼ºï¼‰
            if cfg.use_tta:
                all_preds = []
                
                # å¯¹æ¯�ä¸ªTTAç´¢å¼•è¿›è¡Œé¢„æµ‹
                for tta_idx in range(cfg.tta_count):
                    mel_spec = process_audio_segment(segment_audio, cfg)
                    mel_spec = apply_tta(mel_spec, tta_idx)

                    # å°†æ¢…å°”é¢‘è°±å›¾è½¬æ�¢ä¸ºå¼ é‡�å¹¶ç§»åŠ¨åˆ°æŒ‡å®šè®¾å¤‡
                    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    mel_spec = mel_spec.to(cfg.device)

                    # å¦‚æ�œå�ªæœ‰ä¸€ä¸ªæ¨¡å�‹ï¼Œç›´æ�¥é¢„æµ‹
                    if len(models) == 1:
                        with torch.no_grad():
                            outputs = models[0](mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            all_preds.append(probs)
                    else:
                        # å¦‚æ�œæœ‰å¤šä¸ªæ¨¡å�‹ï¼Œå¯¹æ¯�ä¸ªæ¨¡å�‹è¿›è¡Œé¢„æµ‹å¹¶å�–å¹³å�‡
                        segment_preds = []
                        for model in models:
                            with torch.no_grad():
                                outputs = model(mel_spec)
                                probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                                segment_preds.append(probs)
                        
                        avg_preds = np.mean(segment_preds, axis=0)
                        all_preds.append(avg_preds)

                # å¯¹æ‰€æœ‰TTAç»“æ�œå�–å¹³å�‡
                final_preds = np.mean(all_preds, axis=0)
            else:
                # å¦‚æ�œä¸�å�¯ç”¨TTAï¼Œç›´æ�¥å¤„ç�†éŸ³é¢‘ç‰‡æ®µ
                mel_spec = process_audio_segment(segment_audio, cfg)
                
                # å°†æ¢…å°”é¢‘è°±å›¾è½¬æ�¢ä¸ºå¼ é‡�å¹¶ç§»åŠ¨åˆ°æŒ‡å®šè®¾å¤‡
                mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mel_spec = mel_spec.to(cfg.device)
                
                # å¦‚æ�œå�ªæœ‰ä¸€ä¸ªæ¨¡å�‹ï¼Œç›´æ�¥é¢„æµ‹
                if len(models) == 1:
                    with torch.no_grad():
                        outputs = models[0](mel_spec)
                        final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
                else:
                    # å¦‚æ�œæœ‰å¤šä¸ªæ¨¡å�‹ï¼Œå¯¹æ¯�ä¸ªæ¨¡å�‹è¿›è¡Œé¢„æµ‹å¹¶å�–å¹³å�‡
                    segment_preds = []
                    for model in models:
                        with torch.no_grad():
                            outputs = model(mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            segment_preds.append(probs)

                    final_preds = np.mean(segment_preds, axis=0)
                    
            # ä¿�å­˜é¢„æµ‹ç»“æ�œ
            predictions.append(final_preds)
            
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
    
    return row_ids, predictions



import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def apply_tta(spec, tta_idx):
    """
    åº”ç”¨æµ‹è¯•æ—¶å¢�å¼ºï¼ˆTest-Time Augmentation, TTAï¼‰åˆ°é¢‘è°±å›¾ä¸Š
    """
    if tta_idx == 0:
        # å�Ÿå§‹é¢‘è°±å›¾ï¼Œä¸�å�šä»»ä½•å¤„ç�†
        return spec
    elif tta_idx == 1:
        # æ—¶é—´å��ç§»ï¼ˆæ°´å¹³ç¿»è½¬ï¼‰
        return np.flip(spec, axis=1)
    elif tta_idx == 2:
        # é¢‘ç�‡å��ç§»ï¼ˆå�‚ç›´ç¿»è½¬ï¼‰
        return np.flip(spec, axis=0)
    else:
        # é»˜è®¤è¿”å›�å�Ÿå§‹é¢‘è°±å›¾
        return spec

def run_inference(cfg, models, species_ids):
    """
    å¯¹æ‰€æœ‰æµ‹è¯•éŸ³é¢‘æ–‡ä»¶è¿�è¡Œæ�¨ç�†
    """
    # è�·å�–æ‰€æœ‰æµ‹è¯•éŸ³é¢‘æ–‡ä»¶
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
    
    # å¦‚æ�œå�¯ç”¨è°ƒè¯•æ¨¡å¼�ï¼Œå�ªä½¿ç”¨éƒ¨åˆ†æ–‡ä»¶
    if cfg.debug:
        print(f"Debug mode enabled, using only {cfg.debug_count} files")
        test_files = test_files[:cfg.debug_count]
    
    print(f"Found {len(test_files)} test soundscapes")

    all_row_ids = []
    all_predictions = []

    # é��å�†æ‰€æœ‰æµ‹è¯•æ–‡ä»¶å¹¶è¿›è¡Œæ�¨ç�†
    for audio_path in tqdm(test_files):
        row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg, species_ids)
        all_row_ids.extend(row_ids)
        all_predictions.extend(predictions)
    
    return all_row_ids, all_predictions

def create_submission(row_ids, predictions, species_ids, cfg):
    """
    åˆ›å»ºæ��äº¤æ–‡ä»¶çš„æ•°æ�®æ¡†
    """
    print("Creating submission dataframe...")

    # åˆ�å§‹åŒ–æ��äº¤å­—å…¸ï¼ŒåŒ…å�«row_idå’Œæ¯�ä¸ªç‰©ç§�çš„é¢„æµ‹å€¼
    submission_dict = {'row_id': row_ids}
    
    # å°†æ¯�ä¸ªç‰©ç§�çš„é¢„æµ‹å€¼æ·»åŠ åˆ°å­—å…¸ä¸­
    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] for pred in predictions]

    # å°†å­—å…¸è½¬æ�¢ä¸ºæ•°æ�®æ¡†
    submission_df = pd.DataFrame(submission_dict)

    # å°†row_idè®¾ç½®ä¸ºç´¢å¼•
    submission_df.set_index('row_id', inplace=True)

    # è¯»å�–ç¤ºä¾‹æ��äº¤æ–‡ä»¶
    sample_sub = pd.read_csv(cfg.submission_csv, index_col='row_id')

    # æ£€æŸ¥æ˜¯å�¦æœ‰ç¼ºå¤±çš„ç‰©ç§�åˆ—
    missing_cols = set(sample_sub.columns) - set(submission_df.columns)
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} species columns in submission")
        # ä¸ºç¼ºå¤±çš„ç‰©ç§�åˆ—å¡«å……0.0
        for col in missing_cols:
            submission_df[col] = 0.0

    # ç¡®ä¿�æ��äº¤æ–‡ä»¶çš„åˆ—é¡ºåº�ä¸�ç¤ºä¾‹æ–‡ä»¶ä¸€è‡´
    submission_df = submission_df[sample_sub.columns]

    # é‡�ç½®ç´¢å¼•ï¼Œå°†row_idæ�¢å¤�ä¸ºåˆ—
    submission_df = submission_df.reset_index()
    
    return submission_df



import time

def main():
    """
    ä¸»å‡½æ•°ï¼šæ‰§è¡ŒBirdCLEF-2025æ�¨ç�†æµ�ç¨‹
    """
    # è®°å½•å¼€å§‹æ—¶é—´
    start_time = time.time()
    print("Starting BirdCLEF-2025 inference...")
    
    # æ‰“å�°TTAï¼ˆæµ‹è¯•æ—¶å¢�å¼ºï¼‰é…�ç½®
    print(f"TTA enabled: {cfg.use_tta} (variations: {cfg.tta_count if cfg.use_tta else 0})")

    # åŠ è½½æ¨¡å�‹
    models = load_models(cfg, num_classes)
    
    # å¦‚æ�œæ²¡æœ‰åŠ è½½åˆ°æ¨¡å�‹ï¼Œæ‰“å�°é”™è¯¯ä¿¡æ�¯å¹¶é€€å‡º
    if not models:
        print("No models found! Please check model paths.")
        return
    
    # æ‰“å�°æ¨¡å�‹ä½¿ç”¨æƒ…å†µï¼ˆå�•ä¸ªæ¨¡å�‹æˆ–é›†æˆ�æ¨¡å�‹ï¼‰
    print(f"Model usage: {'Single model' if len(models) == 1 else f'Ensemble of {len(models)} models'}")

    # è¿�è¡Œæ�¨ç�†ï¼Œè�·å�–ç‰‡æ®µIDå’Œé¢„æµ‹ç»“æ�œ
    row_ids, predictions = run_inference(cfg, models, species_ids)

    # åˆ›å»ºæ��äº¤æ–‡ä»¶çš„æ•°æ�®æ¡†
    submission_df = create_submission(row_ids, predictions, species_ids, cfg)

    # ä¿�å­˜æ��äº¤æ–‡ä»¶
    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    # è®°å½•ç»“æ�Ÿæ—¶é—´å¹¶æ‰“å�°æ€»è€—æ—¶
    end_time = time.time()
    print(f"Inference completed in {(end_time - start_time)/60:.2f} minutes")





if __name__ == "__main__":
    main()






