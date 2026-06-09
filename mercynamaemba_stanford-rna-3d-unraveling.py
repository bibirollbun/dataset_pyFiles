import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


#Loading the datasets

def load_data():
    train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
    train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
    validation_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")
    validation_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
    test_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
    sample_submission = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")
    return train_sequences, train_labels, validation_sequences, validation_labels, test_sequences, sample_submission

train_sequences, train_labels, validation_sequences, validation_labels, test_sequences, sample_submission = load_data()


import matplotlib.pyplot as plt
import seaborn as sns


# Data Exploration

def explore_data():
    print("Train Sequences Head:")
    print(train_sequences.head())
    print("\nTrain Labels Head:")
    print(train_labels.head())
    print("\nValidation Sequences Head:")
    print(validation_sequences.head())
    print("\nValidation Labels Head:")
    print(validation_labels.head())
    print("\nTest Sequences Head:")
    print(test_sequences.head())
    print("\nTrain Sequences Info:")
    print(train_sequences.info())
    print("\nTrain Labels Info:")
    print(train_labels.info())
    print("\nMissing Values in Train Sequences:")
    print(train_sequences.isnull().sum())
    print("\nMissing Values in Train Labels:")
    print(train_labels.isnull().sum())    
    
    # Sequence Length Distribution
    train_sequences["sequence_length"] = train_sequences["sequence"].apply(len)
    validation_sequences["sequence_length"] = validation_sequences["sequence"].apply(len)
    
    plt.figure(figsize=(10, 5))
    sns.histplot(train_sequences["sequence_length"], bins=50, kde=True, label="Train Sequences", color='blue')
    sns.histplot(validation_sequences["sequence_length"], bins=50, kde=True, label="Validation Sequences", color='orange')
    plt.xlabel("Sequence Length")
    plt.ylabel("Count")
    plt.title("Distribution of RNA Sequence Lengths")
    plt.legend()
    plt.show()

test_sequences["sequence_length"] = test_sequences["sequence"].apply(len)

explore_data()


def encode_sequence(sequence):
    mapping = {"A": 0, "U": 1, "C": 2, "G": 3}
    return [mapping[char] if char in mapping else -1 for char in sequence] 

train_sequences["encoded_sequence"] = train_sequences["sequence"].apply(encode_sequence)
validation_sequences["encoded_sequence"] = validation_sequences["sequence"].apply(encode_sequence)
test_sequences["encoded_sequence"] = test_sequences["sequence"].apply(encode_sequence)
    




