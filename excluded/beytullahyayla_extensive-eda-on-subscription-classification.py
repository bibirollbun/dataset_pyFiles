import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import express as px
from sklearn.preprocessing import StandardScaler



train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


train_df.head()


print(f"{'*'*50}Train DataFrame Info:{'*'*50}")
print(train_df.info())
print(f"{'*'*50}Test DataFrame Info:{'*'*50}")
print(test_df.info())


print(f"{'*'*50}Train DataFrame Description:{'*'*50}")
print(train_df.describe())
print(f"{'*'*50}Test DataFrame Description:{'*'*50}")
print(test_df.describe())


print(f"{'*'*50}Train DataFrame Null Values:{'*'*50}")
display(train_df.isna().sum())
print(f"{'*'*50}Test DataFrame Null Values:{'*'*50}")
display(test_df.isna().sum())


numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical columns in train_df: {numerical_columns}")


for column in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_df[column], kde=True, bins=30)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    
    stats = train_df[column].describe()
    textstr = '\n'.join((
        f"Mean: {stats['mean']:.2f}",
        f"Median: {train_df[column].median():.2f}",
        f"Std: {stats['std']:.2f}",
        f"Min: {stats['min']:.2f}",
        f"Max: {stats['max']:.2f}"
    ))
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.show()


for column in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=train_df[column])
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)


plt.figure(figsize=(12, 6))
sns.violinplot(data=train_df[numerical_columns].drop(columns=["id"]), inner='quartile')
plt.title('Violin Plot of Numerical Features')
plt.show()
plt.close()


def correlation_heatmap(df, title='Correlation Heatmap'):
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def visualize_correlations_with_target_barplot(df, target_column, title='Correlation with Target'):
    correlations = df.corr()[target_column].drop(target_column).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(correlations))
    sns.barplot(y=correlations.index, x=correlations.values, palette=colors)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Correlation with Target', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
correlation_heatmap(train_df[numerical_columns], title='Train DataFrame Correlation Heatmap')
visualize_correlations_with_target_barplot(train_df[numerical_columns], target_column='y', title='Correlation with Target in Train DataFrame')


categorical_columns = train_df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns in train_df: {categorical_columns}")


for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_df, x=col, order=train_df[col].value_counts().index)
    plt.title(f'Count Plot of {col}')
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

num_plots = len(categorical_columns)
num_cols = 3 
num_rows = (num_plots + num_cols - 1) // num_cols  

fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 4 * num_rows))

axes = axes.flatten()

for i, col in enumerate(categorical_columns):
    wedges, texts, autotexts = axes[i].pie(
        train_df[col].value_counts(),
        labels=train_df[col].value_counts().index,  
        autopct='%1.1f%%', 
        startangle=140
    )
    
    axes[i].legend(wedges, train_df[col].value_counts().index, title=col, loc="center left", bbox_to_anchor=(1, 1), fontsize=10)

    axes[i].set_title(f'Pie Chart of {col}')
    axes[i].set_ylabel('')

for j in range(num_plots, len(axes)):
    axes[j].axis('off')

plt.tight_layout()  
plt.show()



counts = train_df['y'].value_counts()

colors = sns.color_palette("pastel", len(counts))

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    counts,
    labels=counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    textprops={'fontsize': 12, 'color': 'black'},
)

centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

plt.title("Distribution of Subscription Status", fontsize=16, fontweight='bold')

# Autopct yazılarının stilini ayarlama
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.show()



for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_df, x=col, hue='y', order=train_df[col].value_counts().index)
    plt.title(f'{col} vs Target (y)')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title='y', labels=['No', 'Yes'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


train_df_encoded = pd.get_dummies(train_df, columns=categorical_columns, drop_first=True)
test_df_encoded = pd.get_dummies(test_df, columns=categorical_columns, drop_first=True)


train_df_encoded = train_df_encoded.astype(int)
test_df_encoded = test_df_encoded.astype(int)


train_df_encoded.columns


test_df_encoded.columns


train_df_encoded.drop(columns=['id'], inplace=True)
test_ids = test_df_encoded['id']
test_df_encoded.drop(columns=['id'], inplace=True)


numerical_columns = numerical_columns[1:-1]  # Exclude 'id' column if it was included in numerical columns


numerical_columns


standard_scaler = StandardScaler()
standard_scaler.fit(train_df_encoded[numerical_columns])
train_df_encoded[numerical_columns] = standard_scaler.transform(train_df_encoded[numerical_columns])
test_df_encoded[numerical_columns] = standard_scaler.transform(test_df_encoded[numerical_columns])

