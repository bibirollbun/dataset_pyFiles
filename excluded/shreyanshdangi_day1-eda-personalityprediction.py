import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

%matplotlib inline


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')


df


df.shape


df.columns


df.info()


df.describe()


df.isnull().sum()


df_cleaned = df.dropna()


df_cleaned.shape


print(f"Original number of rows: {len(df)}")
print(f"Number of rows after dropping nulls: {len(df_cleaned)}")



train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

train['dataset'] = 'train'
test['dataset'] = 'test'

df_combined = pd.concat([train, test], axis=0).reset_index(drop=True)

print("Dataset shape:", df.shape)

df_combined.head()


df_combined.shape


df_combined.info()


numerical_cols = df_combined.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df_combined.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)


missing_values = df_combined.isnull().sum()
missing_percent = (missing_values / len(df_combined)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
missing_df = missing_df[missing_df['Missing Values'] > 0]
missing_df


df_combined[numerical_cols].describe()


plt.figure(figsize=(4, 4))
sns.countplot(data=df, x='Personality', palette='pastel', edgecolor='black')

plt.title('Distribution of Personality Types', fontsize=12)
plt.xlabel('Personality Type', fontsize=9)
plt.ylabel('Count', fontsize=9)

plt.tick_params(axis='x', labelsize=9)
plt.tick_params(axis='y', labelsize=9)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 
            'Going_outside', 'Friends_circle_size', 'Post_frequency']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 6))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, color='skyblue', edgecolor='black', ax=axes[i])
    axes[i].set_title(f'Distribution of {col}', fontsize=12)
    axes[i].set_xlabel(col, fontsize=9)
    axes[i].set_ylabel('Count', fontsize=9)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    axes[i].tick_params(axis='x', labelsize=9)
    axes[i].tick_params(axis='y', labelsize=9)

for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 
            'Going_outside', 'Friends_circle_size', 'Post_frequency']

plt.figure(figsize=(14, 6))

for i, col in enumerate(num_cols):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(data=df, y=col, color='#FFA726')

    plt.title(f'Boxplot: {col}', fontsize=12)
    plt.xlabel('')  
    plt.ylabel(col, fontsize=9)

    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 
            'Going_outside', 'Friends_circle_size', 'Post_frequency']

scatter_pairs = [
    ('Time_spent_Alone', 'Social_event_attendance'),
    ('Time_spent_Alone', 'Friends_circle_size'),
    ('Going_outside', 'Post_frequency'),
    ('Friends_circle_size', 'Social_event_attendance'),
    ('Post_frequency', 'Time_spent_Alone'),
]

fig, axes = plt.subplots(nrows=len(scatter_pairs), ncols=1, figsize=(12, 22))  # taller figure
axes = axes.flatten()

for i, (x_col, y_col) in enumerate(scatter_pairs):
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue='Personality',
        palette='Set1',
        alpha=0.7,
        s=60,
        edgecolor='black',
        ax=axes[i]
    )
    axes[i].set_title(f'{y_col} vs {x_col}', fontsize=12)
    axes[i].set_xlabel(x_col, fontsize=9)
    axes[i].set_ylabel(y_col, fontsize=9)
    axes[i].tick_params(axis='x', labelsize=9)
    axes[i].tick_params(axis='y', labelsize=9)
    axes[i].grid(True, linestyle='--', alpha=0.5)

    if i != 0:
        axes[i].get_legend().remove()

plt.subplots_adjust(hspace=0.5)
plt.suptitle('Scatter Plot Relationships Colored by Personality', fontsize=14, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 
            'Going_outside', 'Friends_circle_size', 'Post_frequency']

bubble_combinations = [
    ('Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size'),
    ('Going_outside', 'Post_frequency', 'Social_event_attendance'),
    ('Friends_circle_size', 'Time_spent_Alone', 'Going_outside'),
    ('Post_frequency', 'Friends_circle_size', 'Social_event_attendance'),
    ('Going_outside', 'Social_event_attendance', 'Time_spent_Alone')
]

fig, axes = plt.subplots(nrows=len(bubble_combinations), ncols=1, figsize=(10, 40))
axes = axes.flatten()

for i, (x_col, y_col, size_col) in enumerate(bubble_combinations):
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue='Personality',
        size=size_col,
        sizes=(50, 400),
        alpha=0.6,
        edgecolor='black',
        linewidth=0.5,
        palette='Set2',
        ax=axes[i]
    )

    axes[i].set_title(f'{y_col} vs {x_col} (size = {size_col})', fontsize=11)
    axes[i].set_xlabel(x_col, fontsize=9)
    axes[i].set_ylabel(y_col, fontsize=9)
    axes[i].tick_params(axis='x', labelsize=9)
    axes[i].tick_params(axis='y', labelsize=9)
    axes[i].grid(True, linestyle='--', alpha=0.5)

    leg = axes[i].legend(
        title='Personality',
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title_fontsize=9
    )
    leg.get_frame().set_alpha(0.9)

plt.subplots_adjust(hspace=0.6, right=0.8)
plt.suptitle('Bubble Plots Highlighting Personality Differences', fontsize=14, y=0.995)
plt.tight_layout(rect=[0, 0, 0.85, 0.98])
plt.show()


cat_cols = ['Stage_fear', 'Drained_after_socializing'] 

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
axes = axes.flatten()  

for i, col in enumerate(cat_cols):
    sns.countplot(
        data=df,
        x=col,
        order=df[col].value_counts().index,
        palette='Set2',
        edgecolor='black',
        ax=axes[i]
    )
    axes[i].set_title(f'{col} Distribution', fontsize=12)
    axes[i].set_xlabel(col, fontsize=9)
    axes[i].set_ylabel('Count', fontsize=9)
    axes[i].tick_params(axis='x', labelsize=8, rotation=0)
    axes[i].tick_params(axis='y', labelsize=8)
    axes[i].grid(axis='y', linestyle='--', alpha=0.5)

for j in range(len(cat_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


cols = ['Stage_fear', 'Drained_after_socializing']  
palettes = ['Set1'] * len(cols)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 10))
axes = axes.flatten() 

plt.subplots_adjust(wspace=0.5, hspace=0.5)  

for i, (col, palette) in enumerate(zip(cols, palettes)):
    sns.countplot(
        data=df,
        x=col,
        hue='Personality',
        palette=palette,
        edgecolor='black',
        ax=axes[i]
    )
    axes[i].set_title(f'Distribution of {col} by Personality', fontsize=12)
    axes[i].set_xlabel(f'{col} (0=No, 1=Yes)', fontsize=9)
    axes[i].set_ylabel('Count', fontsize=9)
    axes[i].tick_params(axis='x', labelsize=9, rotation=0)
    axes[i].tick_params(axis='y', labelsize=9)
    axes[i].grid(axis='y', linestyle='--', alpha=0.4)

    axes[i].legend(
        title='Personality',
        labels=['Introvert (0)', 'Extrovert (1)'],
        title_fontsize=9,  
        fontsize=9          
    )

for j in range(len(cols), len(axes)):
    fig.delaxes(axes[j])

plt.show()



plt.figure(figsize=(5, 4))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Numerical Features")
plt.show()

