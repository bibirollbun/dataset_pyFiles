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


pip install pandas numpy matplotlib seaborn scikit-learn torch torchvision



import pandas as pd

train_meta = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
print(train_meta.head())



print(train_meta.shape)          # rows x columns
print(train_meta.columns)        # column names
print(train_meta['lab_id'].value_counts())  # kaunse labs kitne videos diye
print(train_meta['behaviors_labeled'].head(5))  # example behaviors per video



import ast
from collections import Counter
import pandas as pd

# Convert string to list, NaN ko empty list me convert
def safe_eval(x):
    if pd.isna(x):
        return []
    else:
        return ast.literal_eval(x)

train_meta['behaviors_list'] = train_meta['behaviors_labeled'].apply(safe_eval)

# Flatten all behaviors
all_behaviors = [b.split(',')[-1] for blist in train_meta['behaviors_list'] for b in blist]

# Count frequency
behavior_counts = Counter(all_behaviors)
print(behavior_counts.most_common(10))  # top 10 behaviors



import os

# Trackings
tracking_root = '/kaggle/input/MABe-mouse-behavior-detection/train_tracking'
tracking_files = []
for lab in os.listdir(tracking_root):
    lab_path = os.path.join(tracking_root, lab)
    if os.path.isdir(lab_path):
        for file in os.listdir(lab_path):
            if file.endswith('.parquet'):
                tracking_files.append(os.path.join(lab_path, file))

print(f'Total tracking files: {len(tracking_files)}')
print(tracking_files[:5])



annotation_root = '/kaggle/input/MABe-mouse-behavior-detection/train_annotation'
annotation_files = []
for lab in os.listdir(annotation_root):
    lab_path = os.path.join(annotation_root, lab)
    if os.path.isdir(lab_path):
        for file in os.listdir(lab_path):
            if file.endswith('.parquet'):
                annotation_files.append(os.path.join(lab_path, file))

print(f'Total annotation files: {len(annotation_files)}')
print(annotation_files[:5])



# Pick first video
tracking_file = tracking_files[0]
annotation_file = annotation_files[0]

pose_data = pd.read_parquet(tracking_file)
annotations = pd.read_parquet(annotation_file)

print(pose_data.head())
print(annotations.head())



import numpy as np

max_frame = pose_data['video_frame'].max()
frame_labels = np.array(['none'] * (max_frame+1))  # default 'none'

# Fill labels from annotations
for _, row in annotations.iterrows():
    frame_labels[row['start_frame']:row['stop_frame']+1] = row['action']

print(frame_labels[1750:1770])  # check around first annotated frame



# Example: simple min-max normalization per video
pose_data['x_norm'] = pose_data['x'] / pose_data['x'].max()
pose_data['y_norm'] = pose_data['y'] / pose_data['y'].max()



import numpy as np
import pandas as pd

sequence_length = 30
sequences = []
labels = []

# frame-level labels already created
# frame_labels = np.array([...])

# check all unique actions
unique_actions = np.unique(frame_labels)
print("Unique actions in this video:", unique_actions)

# automatically create action map
action_map = {act:i for i, act in enumerate(unique_actions)}

max_frame = pose_data['video_frame'].max()

for start in range(0, max_frame-sequence_length, sequence_length):
    seq = pose_data[(pose_data['video_frame']>=start) & (pose_data['video_frame']<start+sequence_length)]
    
    # Pivot with aggregation to handle duplicates
    seq_array = seq.pivot_table(
        index='video_frame',
        columns='bodypart',
        values=['x','y'],    # <-- changed from x_norm/y_norm
        aggfunc='mean'
    ).values
    
    if seq_array.shape[0] == 0:   # skip empty sequences
        continue
    
    sequences.append(seq_array)
    
    # Majority action in this window
    window_labels = frame_labels[start:start+sequence_length]
    
    # map actions to integers safely
    label_ints = [action_map.get(a, action_map.get('none',0)) for a in window_labels]
    labels.append(np.bincount(label_ints).argmax())

print(f"Number of sequences: {len(sequences)}")
print(f"Shape of first sequence: {sequences[0].shape}")
print(f"First label: {labels[0]}")



import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Find max sequence length
max_seq_len = max([s.shape[0] for s in sequences])

# Pad sequences to same length
# seq.shape = (frames, features) -> pad along frames axis
padded_sequences = []
for s in sequences:
    pad_len = max_seq_len - s.shape[0]
    if pad_len > 0:
        # Pad with zeros at the end
        s_padded = np.pad(s, ((0,pad_len),(0,0)), mode='constant', constant_values=0)
    else:
        s_padded = s
    padded_sequences.append(s_padded)

# Convert to numpy array
X = np.array(padded_sequences)
y = np.array(labels)

print("X shape after padding:", X.shape)
print("y shape:", y.shape)



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shapes:", X_train.shape, y_train.shape)
print("Validation shapes:", X_val.shape, y_val.shape)



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking

num_classes = len(np.unique(y))  # Number of unique behaviors

model = Sequential([
    Masking(mask_value=0., input_shape=(X_train.shape[1], X_train.shape[2])),  # ignore zero-padding
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()



history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,           # chhota epoch se start karo, GPU resources dekh ke badha sakte ho
    batch_size=32
)




import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation Accuracy:", val_acc)



y_pred_probs = model.predict(X_val)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Map back to behavior names
inv_action_map = {0:'none',1:'sniff',2:'attack',3:'sniffgenital',4:'chase',
                  5:'approach',6:'mount',7:'rear',8:'escape',9:'avoid',10:'chaseattack'}

pred_behaviors = [inv_action_map[i] for i in y_pred_classes]
print(pred_behaviors[:10])



from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_val, y_pred_classes))
cm = confusion_matrix(y_val, y_pred_classes)
print(cm)



import pandas as pd

submission = []
row_id = 0
sequence_length = 30
mouse_map = {1: 'mouse1', 2: 'mouse2'}

# Loop through all videos in train_meta
for idx, row in train_meta.iterrows():
    video_id = row['video_id']
    
    # Pose data load karo
    pose_data = pd.read_parquet(f'/kaggle/input/MABe-mouse-behavior-detection/train_tracking/{row["lab_id"]}/{video_id}.parquet')
    
    # frame_labels ya model se predictions
    # sequences aur pred_behaviors calculate karo (jaise pehle kiya)
    # yahan ek example prediction list le rahe hain
    max_frame = pose_data['video_frame'].max()
    num_sequences = max_frame // sequence_length
    pred_behaviors = ['none'] * num_sequences   # placeholder, model predictions
    
    # Submission rows add karo
    for i in range(num_sequences):
        start_frame = i * sequence_length
        stop_frame = start_frame + sequence_length - 1
        action = pred_behaviors[i]
        
        submission.append([
            row_id,
            video_id,
            mouse_map[1],
            mouse_map[2],
            action,
            start_frame,
            stop_frame
        ])
        row_id += 1

sub_df = pd.DataFrame(submission, columns=['row_id','video_id','agent_id','target_id','action','start_frame','stop_frame'])
sub_df.to_csv('submission.csv', index=False)
print(sub_df.head())


