import pandas as pd
import numpy as np

import os
import random


def extract_article_ids(path):
    """
    Param path:
        path to either test or train folder
    Returns article_id found in the path
    """
    for root, dirs, files in os.walk(path):
        return dirs # Only top level folder names are ids
    
train_root = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train'
test_root = '/kaggle/input/fake-or-real-the-impostor-hunt/data/test'

train_dirs = extract_article_ids(train_root)
test_dirs = extract_article_ids(test_root)

print(f'\nTrain Dataset has {len(train_dirs)} articles')
print(train_dirs)

print(f'\Test Dataset has {len(test_dirs)} articles')
# print(test_dirs)

# Get the last four integers
train_idx = [int(index[-4:]) for index in train_dirs]
test_idx = [int(index[-4:]) for index in test_dirs]

# Sort the lists
train_idx = sorted(train_idx)
test_idx = sorted(test_idx)


train_csv_path = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv'
train_df = pd.read_csv(train_csv_path)
train_df.head()


assert len(train_df['id']) == len(train_idx), "Mismatch in the number of articles and the train.csv"


def read_both_texts(train=True):
    if train: 
        root = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train'
        article_idx = train_df['id'].astype(int)
    else:
        root = '/kaggle/input/fake-or-real-the-impostor-hunt/data/test'
        article_idx = test_idx
        
    """
        Utility Function to load text files. 
        Params: 
            root a dir either to the train folder or test folder
        Returns:
            two lists of texts, the first one file_1.txt and the second one file_2.txt for each article sorted by train_df.
    """
    
    texts_1 = []
    texts_2 = []
    for article_index in article_idx:
        article_index_f = str(article_index).zfill(4)
        
        text1_path = root + f'/article_{article_index_f}' + '/file_1.txt'
        text2_path = root + f'/article_{article_index_f}' + '/file_2.txt'

        with open(text1_path, 'r') as fp:
            texts_1.append(fp.read())
        with open(text2_path, 'r') as fp:
            texts_2.append(fp.read())
    return texts_1, texts_2  


texts_1, texts_2 = read_both_texts(train=True)
train_df['text1'] = texts_1
train_df['text2'] = texts_2

# Get texts for test samples
texts_1, texts_2 = read_both_texts(train=False)

# Create a separate df for test
test_df = pd.DataFrame({
    'id': test_idx,
    'text1': texts_1,
    'text2': texts_2
})


train_df.head()


test_df.head()


train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

