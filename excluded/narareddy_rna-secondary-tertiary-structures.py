# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        break;

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#read train_sequences.csv file
train_sequences = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
print(train_sequences.shape)
train_sequences.head()


#create squence_length column
train_sequences['sequence_length'] = train_sequences['sequence'].str.len()
#print summary of sequence_length and count
print(train_sequences['sequence_length'].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]))
#group by sequence_length and count
sequence_length_count = train_sequences.groupby('sequence_length').size().reset_index(name='count').sort_values(by='count', ascending=False).reset_index(drop=True)
sequence_length_count.head(10)


#load train_labels.csv
train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
print(train_labels.shape)
train_labels.head()


import os
if os.path.exists("/kaggle"):
    !apt-get update && apt-get install -y vienna-rna


import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import re

def run_rnafold(sequence):
    result = subprocess.run(['RNAfold'], input=sequence.encode(), stdout=subprocess.PIPE)
    output = result.stdout.decode().split('\n')
    dot_bracket = output[1].split(' ')[0].strip()
    return dot_bracket


def plot_dot_bracket(dot_bracket, ax, sequence):
    stack = []
    pairs = []
    for i, symbol in enumerate(dot_bracket):
        if symbol == '(':
            stack.append(i)
        elif symbol == ')':
            j = stack.pop()
            pairs.append((j, i))

    # Draw backbone line
    ax.plot(range(len(dot_bracket)), [0]*len(dot_bracket), color='gray', lw=1)

    # Draw base pair arcs
    for i, j in pairs:
        arc = Arc(((i + j) / 2, 0), j - i, 1, theta1=0, theta2=180)
        ax.add_patch(arc)

    # Draw nucleotide letters below the line
    for i, nt in enumerate(sequence):
        ax.text(i, -0.2, nt, ha='center', va='center', fontsize=8, color='black')

    ax.set_ylim(-1, 1.5)
    ax.set_title("RNAfold Dot-Bracket Pairing")
    ax.axis('off')
    
def sequence_2d_3d_visual(tgt_id):
    # Filter data
    seq_row = train_sequences[train_sequences["target_id"] == tgt_id].iloc[0]
    seq = seq_row["sequence"]
    labels = train_labels[train_labels["ID"].str.startswith(tgt_id)]
    # Extract sequence (nt letters) as list
    nt_sequence = list(seq)

    # 3D coordinates
    x = labels["x_1"]
    y = labels["y_1"]
    z = labels["z_1"]
    
    # Run RNAfold
    dot_bracket = run_rnafold(seq)

    # Plot
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    plot_dot_bracket(dot_bracket, ax1, sequence=seq)
    
    # 3D panel
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(x, y, z, marker='o', color='blue')
    ax2.set_title(f"3D Backbone for {tgt_id}")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # Add base letters as annotations
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        if i < len(nt_sequence):
            ax2.text(xi, yi, zi, nt_sequence[i], fontsize=8, color='red')
    
    plt.suptitle(f"RNA Secondary vs Tertiary Structure\nTarget ID: {tgt_id}")
    plt.tight_layout()
    plt.show()


lengths = [10, 20, 30, 40, 50]

for length in lengths:
    match = train_sequences[train_sequences['sequence_length'] == length]
    if not match.empty:
        target_id = match['target_id'].iloc[0]
        print(f"\nLength {length}: {target_id}, Sequence: {match['sequence'].iloc[0]}")
        sequence_2d_3d_visual(target_id)
    else:
        print(f"Length {length}: No match found")

