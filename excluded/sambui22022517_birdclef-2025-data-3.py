import os
import shutil
import librosa
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
import sys
import pandas as pd

source_folder = "/kaggle/input/birdclef-2025"
destination_folder = "/kaggle/working/"
idx = 3

# Đảm bảo thư mục đích tồn tại
os.makedirs(destination_folder, exist_ok=True)

model, (get_speech_timestamps, _, read_audio, _, _) = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

torchaudio.set_audio_backend("ffmpeg")

def process_ogg_file(src_path, dest_path, human_path):
    # Load file OGG bằng torchaudio
    wav_tensor, sr = torchaudio.load(src_path)  # wav_tensor: (channels, time)

    # Nếu stereo → chuyển sang mono
    if wav_tensor.shape[0] > 1:
        wav_tensor = torch.mean(wav_tensor, dim=0, keepdim=True)

    # Phát hiện vùng có tiếng người
    speech_timestamps = get_speech_timestamps(wav_tensor.squeeze(0), model)

    if len(speech_timestamps) > 0:
        nonhuman_wav = []
        human_wav = []
        start_point = 0

        for st in speech_timestamps:
            # Lưu vùng không có tiếng người
            if st['start'] > start_point:
                nonhuman_wav.append(wav_tensor[:, start_point:st['start']])
            # Lưu vùng có tiếng người
            human_wav.append(wav_tensor[:, st['start']:st['end']])
            start_point = st['end']

        # Phần còn lại sau đoạn cuối
        if start_point < wav_tensor.shape[1]:
            nonhuman_wav.append(wav_tensor[:, start_point:])

        # Nối lại
        if len(nonhuman_wav) > 0:
            nonhuman_tensor = torch.cat(nonhuman_wav, dim=1)
            torchaudio.save(dest_path, nonhuman_tensor, sr, format="mp3")
        if len(human_wav) > 0:
            human_tensor = torch.cat(human_wav, dim=1)
            torchaudio.save(human_path, human_tensor, sr, format="mp3")

list_src_files = []
list_dest_files = []

# Duyệt toàn bộ file và thư mục con
for root, _, files in os.walk(source_folder):
    relative_path = os.path.relpath(root, source_folder)
    dest_dir = os.path.join(destination_folder, relative_path)
    os.makedirs(dest_dir, exist_ok=True)
    
    for file in files:
        src_file_path = os.path.join(root, file)
        dest_file_path = os.path.join(dest_dir, file)

        list_src_files.append(src_file_path)
        list_dest_files.append(dest_file_path)

total_files = len(list_src_files) // 5

for src_file, dest_file in tqdm(zip(list_src_files[idx*total_files:(idx+1)*total_files], list_dest_files[idx*total_files:(idx+1)*total_files]), total=total_files):
    if src_file.endswith(".ogg") and 'train_soundscapes' not in src_file:
        name, ext = os.path.splitext(dest_file)
        dest_file = f"{name}.mp3"
        human_file = f"{name}-human.mp3"
        process_ogg_file(src_file, dest_file, human_file)
    else:
        shutil.copy2(src_file, dest_file)

train = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
train['filename'] = train['filename'].str.replace('.ogg', '.mp3')
train.to_csv('train.csv', index=False)




