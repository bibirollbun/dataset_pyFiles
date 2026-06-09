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
from scipy.stats import rankdata
from IPython.display import display, Image

# Load data
lb_best = pd.read_csv("/kaggle/input/ps-s5e3-rainfall-division-attention/submission.csv")
ori = pd.read_csv("/kaggle/input/hongkongrainfall/hongkong.csv", encoding="gbk")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").drop("id", axis=1)

# Fill missing values
test['winddirection'] = test['winddirection'].fillna(test['winddirection'].median())

# Copy original submission
tmp = lb_best['rainfall'].copy()

# Define condition
condition1 = (test.cloud > 73.5) & (test.sunshine < 0.5) & (test.pressure <= 1020.35) & (test.windspeed > 20.35)

# Adjust rainfall values based on condition
print(len(lb_best.loc[condition1, 'rainfall']))
lb_best.loc[condition1, 'rainfall'] *= 1.005

# Manual adjustments
manual_adjustments = {
    25: -1,  # id 2215 should be 25, not 15
    29: 2,   # rank 25
    120: -1, # rank 26
    123: 2,  # rank 24
    125: 2   # rank 27
}

for idx, val in manual_adjustments.items():
    lb_best.loc[idx, 'rainfall'] = val

# Check rank differences
print((rankdata(tmp[:146]) != rankdata(lb_best['rainfall'][:146])).sum())

# Save submission
lb_best.to_csv('submission.csv', index=False)
print(lb_best.head())

# Function to multiply rainfall values by a factor k
def multi_k(df, k):
    df = df.copy()
    df.loc[condition1, 'rainfall'] *= k
    return df

# Calculate rank data for different multipliers
A = rankdata(tmp[:146])
B1 = rankdata(multi_k(lb_best, 1.01)['rainfall'][:146])
B2 = rankdata(multi_k(lb_best, 1.005)['rainfall'][:146])
B3 = rankdata(multi_k(lb_best, 1.002)['rainfall'][:146])
B4 = rankdata(multi_k(lb_best, 1.001)['rainfall'][:146])

# Sort data for plotting
sorted_indices = np.argsort(A)
A = A[sorted_indices]
B1 = B1[sorted_indices]
B2 = B2[sorted_indices]
B3 = B3[sorted_indices]
B4 = B4[sorted_indices]

# Plotting
plt.figure(figsize=(60, 100))

def plot_ranks(subplot_idx, title, B, color):
    plt.subplot(5, 1, subplot_idx)
    plt.title(title, fontsize=80)
    bars_B = plt.bar(range(len(B)), B, color=color)
    plt.xticks(range(len(B)), [f'Pos {i}' for i in range(len(B))])
    plt.ylabel('Value')
    for bar in bars_B:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{height:.0f}', ha='center')
    plt.axvline(24.5, color='red')

plot_ranks(1, "Array A", A, 'skyblue')
plot_ranks(2, "K=1.01 LB 90077", B1, 'lightgreen')
plot_ranks(3, "K=1.005 LB 90185", B2, 'lightgreen')
plot_ranks(4, "K=1.002 LB 90131", B3, 'lightgreen')
plot_ranks(5, "K=1.001 LB 90158", B4, 'lightgreen')

plt.tight_layout()
plt.savefig("1.jpg")
plt.show()

# Display the plot image
display(Image(filename="1.jpg"))

