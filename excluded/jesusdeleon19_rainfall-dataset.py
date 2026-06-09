import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df.head()


df.isna().sum()


df_test.isna().sum()


df.info()


plt.figsize=(14, 10)

rain = df['rainfall'].value_counts()
plt.pie(rain, labels=rain.index, autopct='%1.1f%%')
plt.title('Rainfall Distribution')


plt.tight_layout()
plt.show()


num_cols = len(df.columns)
rows = int(num_cols**0.5)
cols = (num_cols + rows - 1) // rows

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

if num_cols == 1:
      axes = [axes]

axes = axes.flatten()

for i, col in enumerate(df.columns):
        sns.histplot(data=df, x=col, ax=axes[i], kde=True)
        axes[i].set_title(col)

for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


num_cols = len(df.columns)
rows = int(num_cols**0.5)
cols = (num_cols + rows - 1) // rows

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

if num_cols == 1:
      axes = [axes]

axes = axes.flatten()

for i, col in enumerate(df.columns):
        sns.boxplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(col)

for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


num_cols = len(df.columns)
rows = int(num_cols**0.5)
cols = (num_cols + rows - 1) // rows

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

if num_cols == 1:
      axes = [axes]

axes = axes.flatten()

for i, col in enumerate(df.columns):
        sns.scatterplot(data=df, x=col, y='rainfall', ax=axes[i], hue='rainfall')
        axes[i].set_title(col)

for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 9))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm',fmt='.1%')
plt.title('Correlation Heatmap')
plt.show()




