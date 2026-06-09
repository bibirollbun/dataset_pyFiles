import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


print(f"Dataset Shape: {train_df.shape}")

print("\nData Info:")
train_df.info()

print("\nNumerical Features Summary:")
display(train_df.describe())

print("\nFirst 10 Rows of the Dataset:")
display(train_df.head(10))


numerical_features = train_df.select_dtypes(include=['number']).columns
categorical_cols = train_df.select_dtypes(exclude=['number']).columns

train_df[numerical_features] = train_df[numerical_features].fillna(train_df[numerical_features].mean())

for col in categorical_cols:
    if train_df[col].isnull().any():
        train_df[col] = train_df[col].fillna(train_df[col].mode()[0])


train_df.isnull().sum()


for feature in numerical_features:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(train_df[feature], kde=True, bins=30)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=train_df[feature])
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()

    print(f"\nStatistics for {feature}:")
    print(f"Skewness: {train_df[feature].skew():.2f}")
    print(f"Number of Missing Values: {train_df[feature].isnull().sum()}")



plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
train_df['Stage_fear'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('Stage Fear')
plt.ylabel('')

plt.subplot(1, 2, 2)
train_df['Drained_after_socializing'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#c2c2f0','#ffb3e6','#c2f0c2','#ffccff'])
plt.title('Drained After Socializing')
plt.ylabel('')

plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))
sns.countplot(x='Personality', data=train_df, palette='Set2')

plt.title('Target Distribution: Personality', fontsize=14)
plt.xlabel('Personality Type')
plt.ylabel('Count')

for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                       textcoords='offset points')

plt.tight_layout()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

colors = sns.color_palette('husl', len(numerical_features))

rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.kdeplot(data=train_df, x=col, fill=True, color=color)
    plt.title(f'KDE Plot of {col}', fontsize=14, color=color)
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()


numeric_df = train_df.select_dtypes(include='number')

sns.pairplot(numeric_df, corner=True, plot_kws={'alpha': 0.5})
plt.suptitle('Pairwise Scatter Plots', y=1.02)
plt.show()


for feature in numerical_features[:-1]:  
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=train_df[feature], y=train_df["Personality"], alpha=0.5
    )
    plt.title(f"{feature} vs. Personality")
    plt.xlabel(feature)
    plt.ylabel("Personality")
    plt.show()

correlation_matrix = train_df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
sns.countplot(data=train_df, x='Stage_fear', hue='Personality', palette='Set2')
plt.title("Stage Fear vs. Personality")
plt.xlabel("Stage Fear")
plt.ylabel("Count")
plt.legend(title='Personality')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=train_df, x='Drained_after_socializing', hue='Personality', palette='Set3')
plt.title("Drained After Socializing vs. Personality")
plt.xlabel("Drained After Socializing")
plt.ylabel("Count")
plt.legend(title='Personality')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



colors = sns.color_palette('husl', len(numerical_features))

rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.lineplot(data=train_df[col], color=color)
    plt.title(f'Trend Plot of {col}', fontsize=14, color=color)
    plt.xlabel('Index')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

colors = sns.color_palette('husl', len(numerical_features))
rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.kdeplot(data=train_df, x=col, fill=True, color=color)
    sns.lineplot(data=train_df[col].sort_values().reset_index(drop=True), color='black', linewidth=1)
    plt.title(f'KDE + Trend of {col}', fontsize=14, color=color)
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()


colors = sns.color_palette('husl', len(numerical_features))
rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.violinplot(data=train_df, y=col, color=color)
    plt.title(f'Violin Plot of {col}', fontsize=14, color=color)
    plt.xlabel('')
    plt.ylabel(col)

plt.tight_layout()
plt.show()

