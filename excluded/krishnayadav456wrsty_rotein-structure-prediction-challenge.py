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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


train_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
validation_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")
validation_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
test_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")


print("Train Sequences:", train_sequences.shape)
print("Train Labels:", train_labels.shape)
print("Validation Sequences:", validation_sequences.shape)
print("Test Sequences:", test_sequences.shape)


train_sequences.info()


train_labels.info()
validation_sequences.info()
test_sequences.info()


# Encoding RNA Sequences
# Create a dictionary to map RNA bases to numeric values
rna_vocab = {'A': 1, 'U': 2, 'G': 3, 'C': 4}  

def encode_sequence(seq):
    # Convert each RNA character in the sequence to its corresponding numeric value
    return [rna_vocab[char] for char in seq if char in rna_vocab]  # Ensure only valid characters are processed


# Apply encoding to all sequences
train_sequences['encoded'] = train_sequences['sequence'].apply(encode_sequence)
validation_sequences['encoded'] = validation_sequences['sequence'].apply(encode_sequence)
test_sequences['encoded'] = test_sequences['sequence'].apply(encode_sequence)


# Padding sequences to ensure uniform length across all samples
max_len = max(train_sequences['encoded'].apply(len))  # Determine the max sequence length
train_sequences_padded = pad_sequences(train_sequences['encoded'], maxlen=max_len, padding='post')
validation_sequences_padded = pad_sequences(validation_sequences['encoded'], maxlen=max_len, padding='post')
test_sequences_padded = pad_sequences(test_sequences['encoded'], maxlen=max_len, padding='post')


# Ensure labels are in NumPy array format, excluding the ID column
if isinstance(train_labels, pd.DataFrame):
    train_labels = train_labels.iloc[:, 1:].values  # Convert DataFrame to NumPy array

if isinstance(validation_labels, pd.DataFrame):
    validation_labels = validation_labels.iloc[:, 1:].values  # Convert DataFrame to NumPy array





