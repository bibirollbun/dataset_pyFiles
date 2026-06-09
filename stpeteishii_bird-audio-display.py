import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from IPython.display import display, Audio


df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
df2=df[['filename','common_name']]
df2['path']=df2['filename'].apply(lambda x:os.path.join('/kaggle/input/birdclef-2025/train_audio',x))
names = df2['common_name'].unique().tolist()
names.sort()


for namei in names[0:20]:
    dfi=df2[df2['common_name']==namei]
    paths=dfi['path'].tolist()
    path=random.choice(paths)
    audio = Audio(path)
    print(namei)
    display(audio)

