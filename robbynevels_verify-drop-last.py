import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class_labels = sorted(os.listdir('../input/birdclef-2025/train_audio/'))
train_meta = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')

class BirdclefDataset(Dataset):
    def __init__(self, df, mode='train'):
        self.df = df
        self.mode = mode

    def __getitem__(self, index):
        target = self.df.iloc[index].primary_label
        y = np.array([1 if item == target else 0 for item in class_labels])
        return y
        
    def __len__(self):
        return len(self.df)


train_df, val_df = train_test_split(train_meta, test_size=0.2, random_state=42)
val_dataset = BirdclefDataset(val_df, mode='val')

val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=1,drop_last=True)

sum(len(batch) for batch in val_loader), len(val_df)




