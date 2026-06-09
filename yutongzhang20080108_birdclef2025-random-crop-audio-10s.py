import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import random
from tqdm import tqdm


wav_sec = 10
sample_rate = 32000
min_segment = sample_rate*wav_sec


root_path = "../input/birdclef-2025/"
input_path = root_path + '/train_audio/'
out_path = "./train_raw" + str(wav_sec) +"/"
backend='soundfile'

try:
    os.mkdir(out_path)
except FileExistsError:
    pass

train_meta = pd.read_csv(root_path + 'train.csv')


def crop_and_save(index):
    sig, _ = torchaudio.load(input_path + train_meta.iloc[index].filename, backend=backend)
    
    if sig.shape[1] <= min_segment:
        sig = torch.concat([sig, torch.zeros(1, min_segment - sig.shape[1])], dim=1)
    else:
        start_idx = random.randint(0, sig.shape[1] - min_segment)
        sig = sig[:, start_idx:start_idx + min_segment]  # Extract the random chunk
        
    dir_path = out_path + train_meta.iloc[index].filename.split('/')[0] + '/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
    tmp_savename = out_path + train_meta.iloc[index].filename
    torchaudio.save(uri=tmp_savename, src=sig, sample_rate=sample_rate, backend=backend)


for index in tqdm(range(len(train_meta))):
    crop_and_save(index)




