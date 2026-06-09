


import numpy as np
import pandas as pd
import os
import cv2
from datetime import datetime
import librosa
import soundfile as sf


!rm -rf output


data0=pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
data1=data0[['common_name','filename']]
class_names=data1['common_name'].unique().tolist()[0:5]
print(class_names)
data=data1[data1['common_name'].isin(class_names)]
display(data)
N=list(range(len(class_names)))
normal_mapping=dict(zip(class_names,N)) 
reverse_mapping=dict(zip(N,class_names))       
data['label']=data['common_name'].map(normal_mapping)


def create_path_label_list(df):
    path_label_list = []
    for _, row in df.iterrows():
        path = row['filename']
        label = row['label']
        path_label_list.append((path, label))
    return path_label_list

path_label_list = create_path_label_list(data)
print(path_label_list)


paths=data['filename'].tolist()


N=list(range(len(class_names)))
normal_mapping=dict(zip(class_names,N)) 
reverse_mapping=dict(zip(N,class_names))       
data['common_name']=data['common_name'].map(normal_mapping)


dir0='/kaggle/input/birdclef-2025/train_audio'


audio_paths=[]
labels=[]
for path,label in path_label_list:
    labels+=[label]
    audio_paths+=[os.path.join(dir0,path)]
print(audio_paths)
print(labels)


# Configuration
output_root = 'output'
interval_sec = 1
sample_rate = 22050  # Common sampling rate

# Create output directory
os.makedirs(output_root, exist_ok=True)

for i,audio_path in enumerate(audio_paths):
    label=str(labels[i])
    
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sample_rate)
    total_samples = len(y)
    samples_per_interval = int(sample_rate * interval_sec)
    num_intervals = int(np.ceil(total_samples / samples_per_interval))

    print(f"\nProcessing {audio_path}")
    print(f"  Sample Rate: {sr}, Total Samples: {total_samples}, Samples per Interval: {samples_per_interval}")

    # Create subdirectory for each audio file
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = os.path.join(output_root, label)
    os.makedirs(output_dir, exist_ok=True)

    for clip_idx in range(num_intervals):
        start_sample = clip_idx * samples_per_interval
        end_sample = start_sample + samples_per_interval
        
        # Zero-padding if the last clip is shorter
        if end_sample > total_samples:
            clip = np.zeros(samples_per_interval)
            valid_length = total_samples - start_sample
            clip[:valid_length] = y[start_sample:]
        else:
            clip = y[start_sample:end_sample]

        # Extract features if needed (e.g., MFCC)
        # mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=13)
        
        # Save raw audio waveform
        npy_filename = os.path.join(output_dir, f'{audio_name}_clip_{clip_idx:05d}.npy')
        np.save(npy_filename, clip)
        print(clip.shape)
        
    print(f"  -> Saved {num_intervals} clips to {output_dir}")

print("\nAll audio files have been processed.")








