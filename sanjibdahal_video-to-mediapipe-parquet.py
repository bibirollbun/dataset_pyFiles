# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages except errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
# Disable oneDNN optimizations warnings
import sys
import warnings
import logging
from contextlib import redirect_stderr
import io

# HARDCORE SUPPRESSION - Put this at the VERY TOP of your script
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU only to reduce warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Suppress all warnings
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# Redirect stderr to suppress C++ level warnings
class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

# Import TensorFlow AFTER setting environment variables
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Suppress ABSL
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except:
    pass
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


%%capture 
!pip install mediapipe


import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
from IPython.display import Image, display
import matplotlib.pyplot as plt
def transform(path , start_frame , end_frame , fps):
    frame_number = 0
    frame = []
    type_ = []
    index = []
    x = []
    y = []
    z = []
    
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_FPS, fps)
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            frame_number += 1
            if frame_number < start_frame:
                continue
            if end_frame != -1 and frame_number > end_frame:
                break
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            #face
            if(results.face_landmarks is None):
                for i in range(478):
                    frame.append(frame_number)
                    type_.append("face")
                    index.append(ind)
                    x.append(0)
                    y.append(0)
                    z.append(0)
            else:
                for ind,val in enumerate(results.face_landmarks.landmark):
                    frame.append(frame_number)
                    type_.append("face")
                    index.append(ind)
                    x.append(val.x)
                    y.append(val.y)
                    z.append(val.z)
            #pose
            if(results.pose_landmarks is None):
                for i in range(32):
                    frame.append(frame_number)
                    type_.append("pose")
                    index.append(ind)
                    x.append(0)
                    y.append(0)
                    z.append(0)
            else:
                for ind,val in enumerate(results.pose_landmarks.landmark):
                    frame.append(frame_number)
                    type_.append("pose")
                    index.append(ind)
                    x.append(val.x)
                    y.append(val.y)
                    z.append(val.z)
            #left hand
            if(results.left_hand_landmarks is None):
                for i in range(20):
                    frame.append(frame_number)
                    type_.append("left_hand")
                    index.append(ind)
                    x.append(0)
                    y.append(0)
                    z.append(0)
            else:
                for ind,val in enumerate(results.left_hand_landmarks.landmark):
                    frame.append(frame_number)
                    type_.append("left_hand")
                    index.append(ind)
                    x.append(val.x)
                    y.append(val.y)
                    z.append(val.z)
            #right hand
            if(results.right_hand_landmarks is None):
                for i in range(20):
                    frame.append(frame_number)
                    type_.append("right_hand")
                    index.append(ind)
                    x.append(0)
                    y.append(0)
                    z.append(0)
            else:
                for ind,val in enumerate(results.right_hand_landmarks.landmark):
                    frame.append(frame_number)
                    type_.append("right_hand")
                    index.append(ind)
                    x.append(val.x)
                    y.append(val.y)
                    z.append(val.z)
            
    return pd.DataFrame({
        "frame" : frame,
        "type"  : type_,
        "landmark_index" : index,
        "x" : x,
        "y" : y,
        "z" : z
    })


import json
metadata = {}
with open('/kaggle/input/wlasl-processed/WLASL_v0.3.json' , 'r') as file:
    metadata = json.load(file)


metadata


labelMap = {} # new metadata with relevant info
for i in metadata:
    label = i['gloss']
    for instance in i['instances']:
        Id = int(instance['video_id'])
        frame_start = instance['frame_start']
        frame_end = instance['frame_end']
        fps = instance['fps']
        labelMap[Id] = [label , frame_start , frame_end , fps] # this is the info matters to us


theoretical_file_count = len(labelMap)
theoretical_file_count # words said to be present in the dataset


labelMap[38540] # sample format of metadata 


  # Disable oneDNN optimizations warnings
import sys
import warnings
import logging
from contextlib import redirect_stderr
import io

# HARDCORE SUPPRESSION - Put this at the VERY TOP of your script
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU only to reduce warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Suppress all warnings
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# Redirect stderr to suppress C++ level warnings
class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

# Import TensorFlow AFTER setting environment variables
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Suppress ABSL
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except:
    pass

path = '/kaggle/working/temp'

videoPath = []
videoId = []
labelName = []

if not os.path.exists(path):
    os.makedirs(path)

video_path = '/kaggle/input/wlasl-processed/videos'
for video in os.listdir(video_path):
    if(video.endswith('.mp4')):
        video_filename = os.path.basename(video)
        video_id = int(os.path.splitext(video_filename)[0])
        
        start_frame = labelMap[video_id][1]
        end_frame = labelMap[video_id][2]
        fps = labelMap[video_id][3]
        
        df = transform(video , start_frame , end_frame , fps)
        
        dest_path = os.path.join(path, f'{video_id}.parquet')
        df.to_parquet(dest_path)
        
        videoPath.append(dest_path)
        videoId.append(video_id)
        labelName.append(labelMap[video_id][0])

data_frame = pd.DataFrame({
    'video_id' : videoId,
    'video_path' : videoPath,
    'label' : labelName
})

data_frame.to_csv('/kaggle/working/temp/summary.csv')


import os
video_path = '/kaggle/working/temp'
c = 0
for video in os.listdir(video_path):
    if(video.endswith('.csv')):
        c += 1
c




