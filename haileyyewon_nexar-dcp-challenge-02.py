# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

max_files = 10 

count = 0

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        count += 1
        if count >= max_files:
            break

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import gc
import time
import warnings
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models

warnings.filterwarnings("ignore")

# Check GPU availability and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# Suppress unnecessary formatting warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Paths to the CSV files
train_csv_path = '/kaggle/input/nexar-collision-prediction/train.csv'
test_csv_path = '/kaggle/input/nexar-collision-prediction/test.csv'
submission_csv_path = '/kaggle/input/nexar-collision-prediction/sample_submission.csv'

# Paths to the video directories
train_video_dir = '/kaggle/input/nexar-collision-prediction/train'
test_video_dir = '/kaggle/input/nexar-collision-prediction/test'

# Load the CSV files
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
submission_df = pd.read_csv(submission_csv_path)

# (ì¶”ê°€) id ì»¬ëŸ¼ì�„ ë¬¸ì��ì—´(str)ë¡œ ë³€í™˜í•´ì„œ .0 ë¬¸ì œ ì—†ì• ê¸°
train_df['id'] = train_df['id'].astype(str)

# Display the first few rows of the DataFrames
print("Train.csv:")
print(train_df.head())

print("\nTest.csv:")
print(test_df.head())

print("\nSample Submission:")
print(submission_df.head())

# Optional: handle NaN values if needed, filling with zero or another value
train_df['time_of_event'] = train_df['time_of_event'].fillna(0)
train_df['time_of_alert'] = train_df['time_of_alert'].fillna(0)

# (ì¶”ê°€) Check the video directory paths
print("\nVideo Directory Paths:")
print(f"Train videos are located at: {train_video_dir}")
print(f"Test videos are located at: {test_video_dir}")



# ì�¼ë°˜ì �ìœ¼ë¡œ ì¶©ë�Œì�´ ë°œìƒ�í•˜ëŠ” ë§ˆì§€ë§‰ ë¶€ë¶„ì—� ì´ˆì �ì�„ ë§�ì¶° ë¹„ë””ì˜¤ì—�ì„œ ì£¼ìš” í”„ë ˆì�„ì�„ ì¶”ì¶œ
# ì§€ìˆ˜ ë¶„í�¬ë¥¼ ì‚¬ìš©í•˜ì—¬ ë§ˆì§€ë§‰ì—� ê°€ê¹Œìš´ í”„ë ˆì�„ì—� ë�” ë§�ì�€ ê°€ì¤‘ì¹˜ë¥¼ ë¶€ì—¬

def extract_keyframes(video_path, num_frames=12, target_size=(160, 160)):
    """
    Extracts key frames from the video, focusing on the final part where collisions typically occur.
    Uses exponential distribution to give more weight to frames closer to the end.
    """
    cap = cv2.VideoCapture(video_path) # ë�™ì˜�ìƒ�ì�„ ë¶ˆëŸ¬ì˜¤ê¸° ìœ„í•´ OpenCVì�˜ videoCapture ê°�ì²´ ìƒ�ì„± 

    # íŒŒì�¼ì�´ ì œëŒ€ë¡œ ì—´ë¦¬ì§€ ì•Šì•˜ì�„ ê²½ìš° ëŒ€ë¹„í•œ ì˜ˆì™¸ ì²˜ë¦¬
    if not cap.isOpened():
        print(f"Could not open the video: {video_path}")
        return np.zeros((num_frames, target_size[0], target_size[1], 3), dtype=np.uint8)

    # ì´� í”„ë ˆì�„ ìˆ˜ì™€ ì´ˆë‹¹ í”„ë ˆì�„ ìˆ˜(FPS)ë¥¼ ê°€ì ¸ì˜¤ê¸° 
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0:
        print(f"Video without frames: {video_path}")
        cap.release()
        return np.zeros((num_frames, target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # ì˜�ìƒ� ê¸¸ì�´(ì´ˆ ë‹¨ìœ„) ê³„ì‚°
    duration = total_frames / fps if fps > 0 else 0
    
    # ì§§ì�€ ì˜�ìƒ� (10ì´ˆ ë¯¸ë§Œ): ê· ë“±í•œ ê°„ê²©ìœ¼ë¡œ í”„ë ˆì�„ ì¶”ì¶œ
    if duration < 10:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    # ê¸´ ì˜�ìƒ� (10ì´ˆ ì�´ìƒ�): í›„ë°˜ë¶€ì—� ë�” ì§‘ì¤‘í•´ì„œ ì¶”ì¶œ
    else:
        # ë§ˆì§€ë§‰ 3ì´ˆ ë�™ì•ˆ í”„ë ˆì�„ì�˜ 80% ì§‘ì¤‘(ì¤‘ìš” ì˜�ì—­)
        end_frames = int(num_frames * 0.8)
        start_frames = num_frames - end_frames
        
        # ì§€ë‚œ 3ì´ˆ ë�™ì•ˆì�˜ ì‹œì�‘ ì�¸ë�±ìŠ¤ë¥¼ ê³„ì‚°
        last_seconds = 3
        last_frame_count = min(int(fps * last_seconds), total_frames - 1)
        start_idx = max(0, total_frames - last_frame_count)
        
        # ë§ˆì§€ë§‰ í”„ë ˆì�„ì—� ë�” ë§�ì�€ ê°€ì¤‘ì¹˜ë¥¼ ë¶€ì—¬í•˜ëŠ” ì§€ìˆ˜ ë¶„í�¬
        # ì�´ë ‡ê²Œ í•˜ë©´ ë§ˆì§€ë§‰ì—� ë�” ë°€ì§‘ë�œ ì�¸ë�±ìŠ¤ê°€ ìƒ�ì„±ë�œë‹¤ ("í”„ë ˆì�„ì�„ ë½‘ëŠ” ê°„ê²©"ì��ì²´ë¥¼ ì¡°ì ˆ â†’ ë��ë¶€ë¶„ì—� ë�” ë§�ì�´ ëª°ë¦¬ê²Œ ë§Œë“œëŠ” ë°©ì‹�)
        end_indices = np.array([
            start_idx + int((total_frames - start_idx - 1) * (i/end_frames)**2) 
            for i in range(1, end_frames + 1)
        ])
        
        # contextì—� ë§�ê²Œ ê· ì�¼í•˜ê²Œ ë°°í�¬ë�œ ì´ˆê¸° í”„ë ˆì�„ (ì´ˆë°˜ë¶€ì—�ì„œ ê· ë“±í•˜ê²Œ ì¶”ì¶œí•œ í”„ë ˆì�„ë“¤)
        # contextë�€? ì‚¬ê³  ì§�ì „ì—� ì–´ë–¤ ìƒ�í™©ì�´ í�¼ì³�ì¡ŒëŠ”ì§€ì—� ëŒ€í•œ í��ë¦„, ë°°ê²½, ë§¥ë�½ 
        begin_indices = np.linspace(0, start_idx - 1, start_frames, dtype=int) if start_idx > 0 else np.zeros(start_frames, dtype=int)
        
        # ì�¸ë�±ìŠ¤ ê²°í•©
        frame_indices = np.concatenate([begin_indices, end_indices])
    
    # ì„ íƒ�í•œ í”„ë ˆì�„ ì¶”ì¶œ 
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Use higher resolution and better interpolation
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            frames.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
    
    cap.release()
    return np.array(frames, dtype=np.uint8)

# ë¨¼ì €, ì „ì—­ ë²”ìœ„ì—�ì„œ ë³€í™˜ í�´ë�˜ìŠ¤ë¥¼ ì •ì�˜ 
# ì�…ë ¥ë�œ ì˜�ìƒ� í”„ë ˆì�„ì�„ ì�¼ì • í™•ë¥ ë¡œ ì¢Œìš° ë°˜ì „ì‹œì¼œì„œ, ë�°ì�´í„° ë‹¤ì–‘ì„±ì�„ ëŠ˜ë¦¬ëŠ” ì—­í• 
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, frames):
        if np.random.random() < self.p:
            return frames[:, :, ::-1, :].copy()  # horizontally flip each frame
        return frames

# ì˜�ìƒ� í”„ë ˆì�„ì�˜ ë°�ê¸°ì™€ ëŒ€ë¹„ë¥¼ ë¬´ì�‘ìœ„ë¡œ ì¡°ì •í•´, ë‹¤ì–‘í•œ ì¡°ëª… í™˜ê²½ì�„ ì‹œë®¬ë ˆì�´ì…˜í•˜ëŠ” ì¦�ê°• í�´ë�˜ìŠ¤
class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0):
        self.brightness = brightness
        self.contrast = contrast
        
    def __call__(self, frames):
        # Apply brightness jitter
        if self.brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1-self.brightness), 1+self.brightness)
            frames = frames * brightness_factor
            frames = np.clip(frames, 0, 255)
        
        # Apply contrast jitter
        if self.contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1-self.contrast), 1+self.contrast)
            frames = (frames - 128) * contrast_factor + 128
            frames = np.clip(frames, 0, 255)
            
        return frames

# í”„ë ˆì�„ì—� í��ë¦¿í•œ ì•ˆê°œ íš¨ê³¼ë¥¼ ë„£ì–´, ì‹œì•¼ê°€ ë‚˜ì�œ ë‚ ì”¨ ìƒ�í™©ì�„ ì‹œë®¬ë ˆì�´ì…˜í•˜ëŠ” í�´ë�˜ìŠ¤
class AddFog(object):
    def __call__(self, frames):
        fog = np.random.uniform(0.7, 0.9, frames.shape).astype(np.float32)
        return frames * 0.8 + fog * 50  # Adjusted for 0-255 scale

# í”„ë ˆì�„ì—� í�°ìƒ‰ ì„ í˜• ë…¸ì�´ì¦ˆ(ë¹—ë°©ìš¸)ë¥¼ ì¶”ê°€í•´ ë¹„ ì˜¤ëŠ” ë‚ ì”¨ë¥¼ ì‹œë®¬ë ˆì�´ì…˜í•˜ëŠ” í�´ë�˜ìŠ¤
class AddRain(object):
    def __call__(self, frames):
        h, w = frames.shape[1:3]
        rain = np.random.uniform(0, 1, (len(frames), h, w, 1)).astype(np.float32)
        rain = (rain > 0.97).astype(np.float32) * 200  # White rain drops
        return np.clip(frames * 0.9 + rain, 0, 255)  # Darken a bit and add drops

# ì§€ì •ë�œ í™•ë¥ ì—� ë”°ë�¼ ì–´ë–¤ ë³€í™˜ì�„ ì �ìš©í• ì§€ ë§�ì§€ë¥¼ ë¬´ì�‘ìœ„ë¡œ ê²°ì •í•˜ëŠ” ì»¨íŠ¸ë¡¤ëŸ¬ í�´ë�˜ìŠ¤(ë�œë�¤ì„± ë¶€ì—¬)
class RandomApply(object):
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p
        
    def __call__(self, frames):
        if np.random.random() < self.p:
            return self.transform(frames)
        return frames

# ì—¬ëŸ¬ ê°œì�˜ ë³€í™˜(Flip, Jitter, Fog ë“±)ì�„ ìˆœì„œëŒ€ë¡œ ì �ìš©í•˜ëŠ” ë�°ì�´í„° ì¦�ê°• íŒŒì�´í”„ë�¼ì�¸ í�´ë�˜ìŠ¤
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, frames):
        for t in self.transforms:
            frames = t(frames)
        return frames

# ì˜�ìƒ� í”„ë ˆì�„ ë°°ì—´ì�„ PyTorch í…�ì„œë¡œ ë°”ê¾¸ê³ , í”½ì…€ ê°’ì�„ 0~1 ë²”ìœ„ë¡œ ì •ê·œí™”í•˜ëŠ” í�´ë�˜ìŠ¤
class ToTensor(object):
    def __call__(self, frames):
        # Convert from (T, H, W, C) to (T, C, H, W)
        frames = frames.transpose(0, 3, 1, 2)
        # Convert to tensor and normalize to [0, 1]
        return torch.from_numpy(frames).float() / 255.0


# ë�™ì˜�ìƒ�ì—�ì„œ ë�°ì�´í„° ì¦�ê°•ì�„ ìœ„í•œ ë³€í™˜ì�„ ë°˜í™˜

def get_video_transforms():
    """
    Returns transformations for data augmentation in videos.
    """
    return {
        'train': Compose([
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.3, contrast=0.3),
            RandomApply(AddFog(), p=0.15),
            RandomApply(AddRain(), p=0.15),
            RandomApply(RandomNoise(0.05), p=0.2), 
            RandomApply(RandomOcclusion(), p=0.1),
            ToTensor()
        ]),
        'val': Compose([
            ToTensor()  # Only tensor conversion for validation
        ])
    }

# ë¹„ë””ì˜¤ í”„ë ˆì�„ì—�ì„œ ë¬´ì�‘ìœ„ ê°€ìš°ì‹œì•ˆ(ì •ê·œë¶„í�¬) ë…¸ì�´ì¦ˆë¥¼ ì¶”ê°€í•˜ì—¬, ì‹¤ì œ ì´¬ì˜� í™˜ê²½ì—�ì„œ 
# ë°œìƒ�í•  ìˆ˜ ì�ˆëŠ” ì�¡ì�Œì—� ëŒ€í•´ ëª¨ë�¸ì�´ ë�” ê°•ê±´í•´ì§€ë�„ë¡� ë§Œë“œëŠ” í�´ë�˜ìŠ¤
class RandomNoise(object):
    """
    Applies random Gaussian noise to video frames for data augmentation.
    
    This transformation helps the model become more robust to noise
    that may be present in real-world video data.
    
    Args:
        std (float): Standard deviation of the Gaussian noise as a fraction
                     of the pixel value range (default: 0.05)
    """
    def __init__(self, std=0.05):
        self.std = std
        
    def __call__(self, frames):
        """
        Apply random noise to the input frames.
        
        Args:
            frames (numpy.ndarray): Input video frames of shape (T, H, W, C)
                                   where T is number of frames
        
        Returns:
            numpy.ndarray: Noise-augmented frames, clipped to valid pixel range [0, 255]
        """
        # ì§€ì •ë�œ í‘œì¤€ í�¸ì°¨ë¥¼ ê°€ì§„ ê°€ìš°ì‹œì•ˆ ë…¸ì�´ì¦ˆ ìƒ�ì„±
        noise = np.random.normal(0, self.std * 255, frames.shape).astype(np.float32)
        
        # ìœ íš¨í•œ í”½ì…€ ë²”ìœ„ì—� ë…¸ì�´ì¦ˆ ë°� í�´ë¦½ ì¶”ê°€í•˜ê¸°
        # ì˜�ìƒ�ì�€ ì •ìˆ˜í˜• ë�°ì�´í„°ì—¬ì•¼ í•˜ë¯€ë¡œ í˜• ë³€í™˜ (astype)
        return np.clip(frames + noise, 0, 255).astype(np.uint8)

# ì˜�ìƒ� í”„ë ˆì�„ì—� ê²€ì�€ìƒ‰ ì‚¬ê°�í˜•ì�„ ë¬´ì�‘ìœ„ë¡œ ë�®ì–´ ì”Œì›Œ, ì�¼ë¶€ ì •ë³´ê°€ ê°€ë ¤ì¡Œì�„ ë•Œë�„ ëª¨ë�¸ì�´ ê²¬ë”œ ìˆ˜ ì�ˆë�„ë¡� í›ˆë ¨ì‹œí‚¤ëŠ” í�´ë�˜ìŠ¤
class RandomOcclusion(object):
    """
    Simulates occlusion in video frames by adding black rectangles.
    
    This transformation helps the model learn to handle partial occlusions
    that may occur in real-world scenarios when objects block the camera view.
    """
    def __call__(self, frames):
        """
        Apply random occlusion to the input frames.
        
        Args:
            frames (numpy.ndarray): Input video frames of shape (T, H, W, C)
                                   where T is number of frames
        
        Returns:
            numpy.ndarray: Frames with random occlusion applied
        """
        # í”„ë ˆì�„ í•˜ë‚˜ì�˜ ì„¸ë¡œ(h), ê°€ë¡œ(w) ê¸¸ì�´ ê°€ì ¸ì˜¤ê¸°
        h, w = frames.shape[1:3]
        
        # ì „ì²´ í”„ë ˆì�„ í�¬ê¸°ì�˜ 10%~25% ì‚¬ì�´ í�¬ê¸°ì�˜ ê°€ë¦¼ ì˜�ì—­ í�¬ê¸° ì„¤ì •
        occl_h = np.random.randint(int(h * 0.1), int(h * 0.25))
        occl_w = np.random.randint(int(w * 0.1), int(w * 0.25))
        
        # ì�´ ê°€ë¦¼ ì˜�ì—­ì�´ ë“¤ì–´ê°ˆ ë¬´ì�‘ìœ„ ìœ„ì¹˜ ì¢Œí‘œ ì„¤ì • 
        occl_x = np.random.randint(0, w - occl_w)
        occl_y = np.random.randint(0, h - occl_h)
        
        # ì›�ë³¸ í”„ë ˆì�„ì�„ ìˆ˜ì •í•˜ì§€ ì•Šë�„ë¡� ë³µì‚¬ë³¸ ë§Œë“¤ê¸°
        frames_copy = frames.copy()
        
        # í”½ì…€ì�„ 0(ê²€ì •ìƒ‰)ìœ¼ë¡œ ì„¤ì •í•˜ì—¬ ëª¨ë“  í”„ë ˆì�„ì—� occlusion ì �ìš©
        for i in range(len(frames)):
            frames_copy[i, occl_y:occl_y+occl_h, occl_x:occl_x+occl_w, :] = 0
            
        return frames_copy


# ë¹„ë””ì˜¤ í”„ë ˆì�„ ê°„ ì›€ì§�ì�„(ëª¨ì…˜)ì�„ ì¶”ì �í•˜ëŠ” 'optical_flow'ë¥¼ ê³„ì‚°í•´, ê°�ì²´ë‚˜ ë°°ê²½ì�˜ ì�´ë�™ ë°©í–¥ê³¼ ì†�ë�„ë¥¼ ë²¡í„° í˜•íƒœë¡œ ë°˜í™˜í•˜ëŠ” í•¨ìˆ˜
# ë‘� ì—°ì†�ë�œ ì�´ë¯¸ì§€(ë˜�ëŠ” í”„ë ˆì�„) ì‚¬ì�´ì—�ì„œ, ê°� í”½ì…€ì�´ ì–´ë–»ê²Œ ì�´ë�™í–ˆëŠ”ì§€ë¥¼ ë²¡í„°ë¡œ í‘œí˜„í•˜ëŠ” ê¸°ìˆ  -> optical_flow
# Farneback ë°©ì‹�ë§Œ ì‚¬ìš©
# "ëª¨ë“  í”½ì…€ì�˜ ë°©í–¥ + ì†�ë�„ ì •ë³´ë¥¼ ë‹¤ ë‚¨ê¹€"
def compute_optical_flow_sequence(frames, skip_frames=1):
    """
    Calculates per-frame optical flow magnitudes as a sequence.
    
    Args:
        frames (numpy.ndarray): (T, H, W, C)
        
    Returns:
        numpy.ndarray: (T, 1) array of flow magnitudes (first frame is 0)
    """
    T = len(frames)
    if T < 2:
        return np.zeros((T, 1), dtype=np.float32)
    
    magnitudes = [0.0]  # ì²« í”„ë ˆì�„ì�€ optical flowê°€ ì—†ìœ¼ë‹ˆ 0ìœ¼ë¡œ ì±„ì›€

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    
    for i in range(1, T, skip_frames):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_magnitude = np.linalg.norm(flow, axis=-1).mean()  # (H, W) â†’ scalar mean
            magnitudes.append(flow_magnitude)
        except Exception as e:
            print(f"Error calculating optical flow: {str(e)}")
            magnitudes.append(0.0)
        
        prev_gray = curr_gray

    # ê¸¸ì�´ê°€ ë¶€ì¡±í•˜ë©´ padding
    while len(magnitudes) < T:
        magnitudes.append(0.0)
    
    return np.array(magnitudes, dtype=np.float32).reshape(T, 1)  # (T, 1)



import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import EfficientNetB0


# InceptionV3 ëª¨ë�¸ë¡œ íŠ¹ì„± ì¶”ì¶œ
base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
cnn_feature_dim = base_model.output_shape[-1]

def get_hybrid_feature_sequence(video_path, num_frames=12):
    """
    Extract per-frame hybrid features (CNN + Optical flow) as a sequence.
    
    Args:
        video_path (str): Path to video file.
        num_frames (int): Number of frames to extract.
    
    Returns:
        np.ndarray: (T, 1281) array of per-frame features.
    """
    # 1. í”„ë ˆì�„ ì¶”ì¶œ
    frames = extract_keyframes(video_path, num_frames=num_frames, target_size=(160,160))
    
    if len(frames) == 0:
        print(f"Skipping {video_path}: no frames")
        return np.zeros((num_frames, 1281), dtype=np.float32)

    # 2. CNN feature per frame (Inception expects (N, H, W, C))
    spatial_features = base_model.predict(
        preprocess_input(frames.astype('float32')),
        batch_size=32,
        verbose=0
    )  # shape: (T, 1280)

    # 3. Optical flow sequence
    flow_magnitudes = compute_optical_flow_sequence(frames)  # shape: (T, 1)

    # 4. Concatenate per frame
    hybrid_features = np.concatenate([spatial_features, flow_magnitudes], axis=1)  # (T, 1281)

    return hybrid_features



def get_hybrid_feature_sequence_from_frames(frames):
    """
    Extract per-frame hybrid features (CNN + Optical flow) from pre-loaded frames.
    
    Args:
        frames (torch.Tensor): (T, 3, 160, 160) tensor (after transform).
    
    Returns:
        np.ndarray: (T, 1281) array of per-frame features.
    """
    if len(frames) == 0:
        print("Warning: empty frames input")
        return np.zeros((1, 1281), dtype=np.float32)

    # 1ï¸�âƒ£ PyTorch tensor â†’ numpy (T, 160, 160, 3), [0, 255] scale
    frames_np = frames.permute(0, 2, 3, 1).numpy() * 255.0  # [0,1] â†’ [0,255]
    frames_np = frames_np.astype(np.uint8)

    # 2ï¸�âƒ£ CNN Features per frame
    spatial_features = base_model.predict(
        preprocess_input(frames_np.astype('float32')),
        batch_size=32,
        verbose=0
    )  # shape: (T, 1280)

    # 3ï¸�âƒ£ Optical Flow per frame
    flow_magnitudes = compute_optical_flow_sequence(frames_np)  # shape: (T, 1)

    # 4ï¸�âƒ£ Concatenate â†’ (T, 1281)
    hybrid_features = np.concatenate([spatial_features, flow_magnitudes], axis=1)

    return hybrid_features



def compute_optical_flow_sequence(frames, skip_frames=1):
    """
    Computes per-frame optical flow magnitudes.
    
    Args:
        frames (np.ndarray): (T, H, W, 3) numpy array of frames.
    
    Returns:
        np.ndarray: (T, 1) array of per-frame optical flow magnitudes.
    """
    T = len(frames)
    if T < 2:
        return np.zeros((T, 1), dtype=np.float32)

    magnitudes = []

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

    for i in range(1, T, skip_frames):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        try:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # magnitude = sqrt(u^2 + v^2)
            mag = np.linalg.norm(flow, axis=-1)  # shape (H, W)
            avg_mag = np.mean(mag)  # scalar
            magnitudes.append(avg_mag)
        except Exception as e:
            print(f"Error calculating flow at frame {i}: {str(e)}")
            magnitudes.append(0.0)

        prev_gray = curr_gray

    # ë§ˆì§€ë§‰ ê¸¸ì�´ ë§�ì¶¤ (T, 1)
    if len(magnitudes) < T:
        magnitudes.append(0.0)  # ë§ˆì§€ë§‰ í”„ë ˆì�„ì�€ flowê°€ ì—†ì�Œ

    magnitudes = np.array(magnitudes, dtype=np.float32).reshape(-1, 1)  # (T, 1)

    return magnitudes



# ì•„ì§�ë�„ ìµœì¢…ì �ìœ¼ë¡œ Transformerì—� ë„£ì�€ (T, 1281) ì‹œí€€ìŠ¤ëŠ” ë§Œë“¤ì–´ì§€ì§€ ì•Šì�Œ
# 1. í”„ë ˆì�„ë³„ CNN Feature ì¶”ì¶œ (InceptionV3) ì¶”ì¶œ
# 2. optical flow sequence (compute_optical_flow_sequence) ì¶”ì¶œ
# 3. ë‘� ê²°ê³¼ë¬¼ concat
# 4. ì�´ê±¸ Transformerì�˜ input ì‹œí€€ìŠ¤ë¡œ ì‚¬ìš©

# ì „ì²´ ì²˜ë¦¬ í•¨ìˆ˜ (ì�´ì œ ë‘˜ì�„ ê²°í•©í•˜ëŠ” í•¨ìˆ˜ ìƒ�ì„±)
# CNN + optical flow ë¶™ì—¬ì„œ (T, 1281) ë§Œë“¤ì–´ì£¼ëŠ” í•¨ìˆ˜ 

def prepare_transformer_input(video_path, num_frames=12, target_size=(160, 160)):
    """
    Prepares (T, 1281) input sequence combining CNN features + optical flow
    for a given video.
    """
    # === Step 1: Extract frames ===
    frames = extract_keyframes(video_path, num_frames=num_frames, target_size=target_size)
    if frames.shape[0] == 0:
        print(f"Skipping video {video_path} (no frames)")
        return None  # or np.zeros((num_frames, 1281)) as fallback

    # === Step 2: Extract CNN (spatial) features per frame ===
    frames_float = preprocess_input(frames.astype('float32'))  # preprocess for InceptionV3
    spatial_features = base_model.predict(frames_float, batch_size=32, verbose=0)  # (T, 1280)

    # === Step 3: Compute optical flow sequence ===
    optical_flow_sequence = compute_optical_flow_sequence(frames)  # (T, 1)

    # === Step 4: Combine both ===
    combined_features = np.concatenate([spatial_features, optical_flow_sequence], axis=1)  # (T, 1281)

    return combined_features


# ì¤‘ê°„ ì €ì�¥ í�¬í•¨ ì½”ë“œ
import numpy as np
import os
from tqdm import tqdm

# 1ï¸�âƒ£ transform ì¤€ë¹„
transforms = get_video_transforms()
train_transform = transforms['train']

# 2ï¸�âƒ£ ì €ì�¥í•  í�´ë�” ì„¤ì •
output_dir = '/kaggle/working/'
os.makedirs(output_dir, exist_ok=True)

# 3ï¸�âƒ£ feature ì €ì�¥ìš© ë¦¬ìŠ¤íŠ¸
all_sequences = []

# 4ï¸�âƒ£ ì¤‘ê°„ ì €ì�¥ ì£¼ê¸°
save_every = 50

# 5ï¸�âƒ£ ë°˜ë³µ
for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    video_id = row['id']
    video_path = os.path.join(train_video_dir, f"{int(video_id):05d}.mp4")

    sequence = prepare_transformer_input(video_path, num_frames=12)

    if sequence is None:
        print(f"Skipping video {video_id} (no valid sequence)")
        continue

    all_sequences.append(sequence)

    # ğŸ”¥ Nê°œë§ˆë‹¤ ì¤‘ê°„ ì €ì�¥
    if (idx + 1) % save_every == 0:
        partial_path = os.path.join(output_dir, f'all_sequences_partial_{idx+1}.npy')
        np.save(partial_path, np.array(all_sequences))
        print(f"Saved {idx + 1} sequences â†’ {partial_path}")

# 6ï¸�âƒ£ ìµœì¢… ì €ì�¥
final_path = os.path.join(output_dir, 'all_sequences_final.npy')
np.save(final_path, np.array(all_sequences))
print(f"\nFinal saved â†’ {final_path}")



import os
import numpy as np

# ê²½ë¡œ ì„¤ì •
output_dir = '/kaggle/working/'
final_file = os.path.join(output_dir, 'all_sequences_final.npy')
combined_save_path = os.path.join(output_dir, 'all_sequences_combined.npy')

# âœ… ìµœì¢… ì €ì�¥ë�œ íŒŒì�¼ë§Œ ë¶ˆëŸ¬ì˜¤ê¸°
final_array = np.load(final_file)
print(f"Final array shape: {final_array.shape}")

# âœ… combined íŒŒì�¼ë¡œ ë”°ë¡œ ì €ì�¥ (ë§Œì•½ í•„ìš”í•  ë•Œ ëŒ€ë¹„)
np.save(combined_save_path, final_array)
print(f"Combined array saved to: {combined_save_path}")



# (1) combined feature ë¶ˆëŸ¬ì˜¤ê¸°
import numpy as np
all_sequences = np.load('/kaggle/working/all_sequences_combined.npy')
print(all_sequences.shape)  # â†’ (n_videos, 12, 2049) ê°™ì�€ ì¶œë ¥ í™•ì�¸

# (2) train.csv ë‹¤ì‹œ ë¡œë“œ
import pandas as pd
train_df = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')

# (3) label ì¶”ì¶œ
labels = train_df['target'].values  # shape (n_videos,)



import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# 1ï¸�âƒ£ ì €ì�¥ë�œ feature ë¶ˆëŸ¬ì˜¤ê¸°
all_sequences = np.load('all_sequences_combined.npy')
print(f"Loaded all_sequences shape: {all_sequences.shape}")  # (1500, 12, 2049)

# 2ï¸�âƒ£ train_dfì—�ì„œ label ë¶ˆëŸ¬ì˜¤ê¸° (ì£¼ì�˜: feature ê°œìˆ˜ì—� ë§�ê²Œ ì�˜ë�¼ì£¼ê¸°!)
labels = train_df['target'].values[:all_sequences.shape[0]]  # shape (1500,)
print(f"Labels shape: {labels.shape}")

# 3ï¸�âƒ£ Dataset í�´ë�˜ìŠ¤ ì •ì�˜
class VideoSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        Args:
            sequences (numpy.ndarray): shape (n_samples, T, feature_dim)
            labels (numpy.ndarray or list): shape (n_samples,)
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = self.sequences[idx]  # (T, feature_dim)
        y = self.labels[idx]     # scalar or class
        return x, y

# 4ï¸�âƒ£ Dataset ê°�ì²´ ìƒ�ì„±
# Dataset ê°�ì²´ ìƒ�ì„±
dataset = VideoSequenceDataset(all_sequences, labels)

# ì •í™•í•œ ê¸¸ì�´ ì²´í�¬
print(f"Dataset length: {len(dataset)}")  # ê¼­ ì°�ì–´ë´�!

# Train/Val split
train_size = int(0.8 * len(dataset))  # 80% split â†’ 1200 if 1500 total
val_size = len(dataset) - train_size

print(f"Train size: {train_size}, Val size: {val_size}")

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)






# # DataLoader ìƒ�ì„±
# from torch.utils.data import DataLoader, random_split

# dataset = VideoSequenceDataset(all_sequences, labels)

# # Train/Val ë¶„í• 
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Temporal Transformer ëª¨ë�¸ ì„¤ê³„
import torch
import torch.nn as nn

class TemporalTransformerModel(nn.Module):
    def __init__(self, input_dim=1281, embed_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super(TemporalTransformerModel, self).__init__()

        # 1. Input â†’ embedding layer
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Classification head (binary classification)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Binary output (0~1)
        )

    def forward(self, x):
        """
        x: (batch_size, T, input_dim)
        """
        # Step 1: Project input features
        x = self.input_proj(x)  # â†’ (batch_size, T, embed_dim)

        # Step 2: Apply Transformer Encoder
        x = self.transformer_encoder(x)  # â†’ (batch_size, T, embed_dim)

        # Step 3: Aggregate (mean pooling over time)
        x = x.mean(dim=1)  # â†’ (batch_size, embed_dim)

        # Step 4: Final classification
        out = self.classifier(x)  # â†’ (batch_size, 1)

        return out


# Spatial Transformer ëª¨ë�¸ ì„¤ê³„

import torch
import torch.nn as nn

class SpatialTransformer(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, num_heads=8, num_layers=2, dropout=0.1):
        super(SpatialTransformer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ë§ˆì§€ë§‰ summaryë¥¼ ìœ„í•œ pooling ë˜�ëŠ” projection
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        Args:
            x: shape (batch_size, T, input_dim) â†’ per-frame spatial features
        
        Returns:
            out: shape (batch_size, input_dim) â†’ aggregated spatial feature
        """
        # transformer expects (batch_size, T, input_dim)
        x_transformed = self.transformer(x)  # (batch_size, T, input_dim)

        # Pooling over time (mean pooling)
        x_pooled = x_transformed.mean(dim=1)  # (batch_size, input_dim)

        out = self.output_layer(x_pooled)  # (batch_size, input_dim)
        return out



class CollisionPredictionModel(nn.Module):
    def __init__(self, temporal_input_dim=2049, spatial_input_dim=1280, embed_dim=256, dropout=0.1):
        super(CollisionPredictionModel, self).__init__()

        self.temporal_transformer = TemporalTransformerModel(
            input_dim=temporal_input_dim, embed_dim=embed_dim
        )
        self.spatial_transformer = SpatialTransformer(
            input_dim=spatial_input_dim
        )

        fused_dim = embed_dim + spatial_input_dim  # Temporal + Spatial ì¶œë ¥ ì—°ê²°

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Binary classification
        )

    def forward(self, temporal_input, spatial_input):
        """
        temporal_input: (batch, T, 2049)
        spatial_input: (batch, T, 1280)
        """
        temporal_out = self.temporal_transformer(temporal_input)  # (batch, embed_dim)
        spatial_out = self.spatial_transformer(spatial_input)      # (batch, spatial_input_dim)

        # Fusion: concatenate
        fused = torch.cat([temporal_out, spatial_out], dim=1)  # (batch, fused_dim)

        out = self.classifier(fused)  # (batch, 1)

        return out



import torch
import torch.nn as nn

class TemporalTransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super(TemporalTransformerModel, self).__init__()

        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_proj(x)            # (batch, T, embed_dim)
        x = self.transformer_encoder(x)   # (batch, T, embed_dim)
        x = x.mean(dim=1)                 # (batch, embed_dim)
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=512, num_heads=8, num_layers=2, dropout=0.1):
        super(SpatialTransformer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x_transformed = self.transformer(x)  # (batch, T, input_dim)
        x_pooled = x_transformed.mean(dim=1) # (batch, input_dim)
        out = self.output_layer(x_pooled)    # (batch, input_dim)
        return out


class CombinedModel(nn.Module):
    def __init__(self, temporal_input_dim=2049, spatial_input_dim=1280,
                 temporal_embed_dim=256, combined_dim=256, dropout=0.1):
        super(CombinedModel, self).__init__()

        self.temporal_transformer = TemporalTransformerModel(
            input_dim=temporal_input_dim, embed_dim=temporal_embed_dim
        )
        self.spatial_transformer = SpatialTransformer(
            input_dim=spatial_input_dim
        )

        # temporal (256) + spatial (1280) = 1536
        self.classifier = nn.Sequential(
            nn.Linear(1536, combined_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, temporal_input, spatial_input):
        temporal_out = self.temporal_transformer(temporal_input)  # (batch, 256)
        spatial_out = self.spatial_transformer(spatial_input)     # (batch, 1280)

        combined = torch.cat([temporal_out, spatial_out], dim=1) # (batch, 1536)
        out = self.classifier(combined)                          # (batch, 1)

        return out



import torch
import torch.nn as nn
import torch.optim as optim

# ëª¨ë�¸ ì¤€ë¹„
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CombinedModel(
    temporal_input_dim=2049, spatial_input_dim=1280,
    temporal_embed_dim=256, combined_dim=256
).to(device)

# ì†�ì‹¤ í•¨ìˆ˜ ë°� ì˜µí‹°ë§ˆì�´ì €
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)  # (batch, T, 2049)
        labels = labels.to(device).unsqueeze(1)  # (batch, 1)

        # temporal_input = spatial(1280) + flow(1) + ì¶”ê°€ optical flowë“¤ â†’ (2049)
        temporal_input = inputs[:, :, :2049]

        # spatial_input = spatial part only â†’ (1280)
        spatial_input = inputs[:, :, :1280]

        # forward
        outputs = model(temporal_input, spatial_input)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # === Validation ===
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)
    
            temporal_input = inputs[:, :, :2049]
            spatial_input = inputs[:, :, :1280]
    
            outputs = model(temporal_input, spatial_input)
            predicted = (outputs > 0.5).float()
    
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")



# ëª¨ë�¸ ì €ì�¥ 
torch.save(model.state_dict(), 'best_model.pth')


# ê°™ì�€ ëª¨ë�¸ ì•„í‚¤í…�ì²˜ ì¤€ë¹„ (ëª¨ë�¸ ë¶ˆëŸ¬ì˜¤ê¸°)
model = CombinedModel(
    temporal_input_dim=2049,  # ì£¼ì�˜: í•™ìŠµí•  ë•Œì™€ ë�™ì�¼í•´ì•¼ í•œë‹¤
    spatial_input_dim=1280,
    temporal_embed_dim=256,
    combined_dim=256
)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()



test_sequences = []
for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing Test Videos"):
    video_path = f"{test_video_dir}/{int(float(row['id'])):05d}.mp4"
    sequence = prepare_transformer_input(video_path, num_frames=12)
    if sequence is not None:
        test_sequences.append(sequence)

test_sequences = np.array(test_sequences)  # shape: (n_test, 12, 2049)



# í…ŒìŠ¤íŠ¸ìš© Dataset, DataLoader

import torch
from torch.utils.data import DataLoader

# í…ŒìŠ¤íŠ¸ìš©: dummy labels (ì˜ˆì¸¡ìš©ì�´ë�¼ ì‹¤ì œ labelì�€ í•„ìš” ì—†ì�Œ)
dummy_labels = np.zeros(len(test_sequences))

# Dataset
test_dataset = VideoSequenceDataset(test_sequences, dummy_labels)

# DataLoader
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



# í…ŒìŠ¤íŠ¸ ë�°ì�´í„° ì˜ˆì¸¡ ì‹¤í–‰

model.eval()
all_predictions = []

with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        temporal_input = inputs[:, :, :2049]
        spatial_input = inputs[:, :, :1280]
        outputs = model(temporal_input, spatial_input)
        all_predictions.extend(outputs.cpu().numpy().flatten())

all_predictions = np.array(all_predictions)
print(f"Predictions shape: {all_predictions.shape}")



# Kaggle ì œì¶œìš© CSV
submission = pd.DataFrame({
    'id': test_df['id'],
    'score': all_predictions
})

submission.to_csv('submission.csv', index=False)
print("Saved submission.csv!")

# ìš”ì•½ í™•ì�¸
print(submission.describe())




