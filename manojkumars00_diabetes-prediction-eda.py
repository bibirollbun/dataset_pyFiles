import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


train_csv_path = "/kaggle/input/playground-series-s5e12/train.csv"


train_df = pd.read_csv(train_csv_path)
train_df.drop(['id'], axis=1, inplace=True)
print(f"Total Columns: {len(train_df.columns)}\n")
print("\n".join(train_df.columns))


train_df.isna().sum()


train_df.dtypes


train_df.describe()


cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
cat_cols += ['family_history_diabetes','hypertension_history', 'cardiovascular_history', 'diagnosed_diabetes']

for col in cat_cols:
    plt.figure(figsize=(10, 4))
    train_df[col].value_counts().plot(kind='bar')
    plt.title(f"Value Counts: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
cat_cols += ['family_history_diabetes','hypertension_history', 'cardiovascular_history']


for col in cat_cols:
    plt.figure(figsize=(10,5))
    prop = (train_df.groupby(col)['diagnosed_diabetes']
                  .mean()
                  .sort_values())
    sns.barplot(x=prop.index, y=prop.values)
    plt.title(f"Diabetes Rate by {col}")
    plt.ylabel("Probability of Diabetes")
    plt.xlabel(col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for col in cat_cols:
    plt.figure(figsize=(10,5))
    sns.countplot(data=train_df, x=col, hue='diagnosed_diabetes')
    plt.title(f"{col} vs Diabetes Status")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols.remove('diagnosed_diabetes')  # remove target

print(f"Numerical Columns ({len(num_cols)}):")

for col in num_cols:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # KDE plot
    sns.kdeplot(data=train_df, x=col, hue="diagnosed_diabetes", ax=axes[0], common_norm=False)
    axes[0].set_title(f"KDE: {col} vs Diabetes")

    # Boxplot
    sns.boxplot(data=train_df, x="diagnosed_diabetes", y=col, ax=axes[1])
    axes[1].set_title(f"Boxplot: {col} by Diabetes Status")
    axes[1].set_xlabel("diagnosed_diabetes")

    plt.tight_layout()
    plt.show()


num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols.remove('diagnosed_diabetes')  # remove target

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

df_sample = train_df.sample(20000, random_state=42)


for num_col1 in num_cols:
    for num_col2 in num_cols:
        for num_col3 in num_cols:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            sc = ax.scatter(
                df_sample[num_col1],
                df_sample[num_col2],
                df_sample[num_col3],
                c=df_sample['diagnosed_diabetes'],
                cmap='coolwarm',
                alpha=0.4
            )
            
            ax.set_xlabel(num_col1)
            ax.set_ylabel(num_col2)
            ax.set_zlabel(num_col3)
            plt.title('3D Scatter colored by diabetes')
            
            plt.colorbar(sc, label='Diabetes')
            plt.show()





