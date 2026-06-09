!pip install -Uqq auditus


import numpy as np
import pandas as pd
from os.path import relpath
from fastcore.all import *
from fastprogress.fastprogress import progress_bar
from fasttransform import Pipeline

from auditus.core import AudioArray
from auditus.transform import AudioLoader, Resampling, TFAudioEmbedding


BASE_PATH = "/kaggle/input/birdclef-2025/"
train_soundscape_paths = np.random.choice(globtastic(f"{BASE_PATH}train_soundscapes", file_glob="*.ogg"), size=250)


train_soundscape_paths[:3]


df = pd.read_csv(f"{BASE_PATH}train.csv")
df.head(2)


train_audio_paths = [f"{BASE_PATH}train_audio/{name}" for name in df['filename']]
print(len(train_audio_paths))
train_audio_paths[:3]


class Truncate(Transform):
    """ Get first 5 seconds for sample rate of 32000 and add padding if necessary. """
    def encodes(self, x:AudioArray): return AudioArray(np.pad(x.a, (0, max(0, 160000-len(x.a))))[:160000], x.sr)


test_path = np.random.choice(train_audio_paths)
print(f"Test file: '{test_path}'")


full_audio = AudioLoader(sr=32000)(test_path)
full_audio.audio()


truncated_audio = Truncate()(full_audio)
truncated_audio.audio()


pipe = Pipeline([AudioLoader(sr=32000), Truncate(), TFAudioEmbedding('/kaggle/input/bird-vocalization-classifier/tensorflow2/bird-vocalization-classifier/8/')])


output = pipe(test_path)
print(f"Embedding Length: '{output.shape}'")
print(f"First 5 elements of embedding: '{output[0][:5]}'")


def get_emb(path): return pipe(path).squeeze(0).tolist()

with ThreadPoolExecutor() as ex:
    train_embs = list(progress_bar(ex.map(get_emb, train_audio_paths), total=len(train_audio_paths)))


df["emb"] = train_embs


df.to_csv("train_with_emb.csv")

