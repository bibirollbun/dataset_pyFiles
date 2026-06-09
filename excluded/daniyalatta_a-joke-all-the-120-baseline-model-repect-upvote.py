import pandas as pd

train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

submission_df = pd.DataFrame()
submission_df['id'] = test_df['id']

# "My model" prediction
submission_df['BeatsPerMinute'] = 120

submission_df.to_csv('submission.csv', index=False)


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid", palette="deep")
plt.figure(figsize=(12, 7))
sns.histplot(
    train_df['BeatsPerMinute'],
    bins=30,
    kde=True,
    color=sns.color_palette()[0], 
    label='Train Data',
    edgecolor='white',
    linewidth=0.5
)

predicted_value = submission_df['BeatsPerMinute'].iloc[0]
plt.axvline(
    x=predicted_value,
    color='red',
    linestyle='--',
    linewidth=3,
    label='"My Model"'
)

plt.title('Distribution of Train data vs. "My Model"', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('BeatsPerMinute', fontsize=14, labelpad=10)
plt.ylabel('Density', fontsize=14, labelpad=10)
plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
plt.xticks(np.arange(60, 210, 10)) 

plt.tight_layout()
plt.show()




