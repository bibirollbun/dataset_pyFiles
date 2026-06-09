import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')

pd.set_option('display.max_columns', None)
print(df.head())


df.shape


df.describe()


df.info()


df.nunique()


df.isnull().sum()


import warnings
warnings.filterwarnings("ignore")


# Wine-red palette
wine_palette = ['#8B0000', '#B3003C', '#C71585', '#E0115F', '#FF3366']

plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x='gender',
    hue='gender',
    palette=wine_palette
)

plt.legend().remove() 

plt.title('Countplot for gender', color='#4a0000', fontsize=14)
plt.xlabel('column', color='#4a0000')
plt.ylabel('Count', color='#4a0000')

plt.show()




num_cats = df['ethnicity'].nunique()
colors = wine_palette[:num_cats]

plt.figure(figsize=(5,5))

df['ethnicity'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=colors,
    figsize=(5,5),
    textprops={'color': '#4a0000', 'fontsize': 12}
)
plt.ylabel('')
plt.title('Category Proportions', fontsize=14, color='#4a0000')
plt.show()


plt.figure(figsize=(6,4))

sns.barplot(
    data=df,
    x='income_level',
    y='diagnosed_diabetes',
    palette=wine_palette
)

plt.title("Diabetes Rate by Income Level", color="#4a0000", fontsize=14)
plt.xlabel("Income Level", color="#4a0000")
plt.ylabel("Mean Diabetes Rate", color="#4a0000")
plt.xticks(rotation=45, ha="right", color="#4a0000")
plt.yticks(color="#4a0000")

plt.show()



plt.figure(figsize=(6,4))

sns.barplot(
    data=df,
    x='education_level',
    y='diagnosed_diabetes',
    palette=wine_palette
)

plt.title("Diabetes Rate by Education Level", color="#4a0000", fontsize=14)
plt.xlabel("Education Level", color="#4a0000")
plt.ylabel("Mean Diabetes Rate", color="#4a0000")
plt.xticks(rotation=45, ha="right", color="#4a0000")
plt.yticks(color="#4a0000")

plt.show()



plt.figure(figsize=(6,4))

sns.violinplot(
    data=df,
    x='smoking_status',
    y='diagnosed_diabetes',
    palette=wine_palette
)

plt.title("Diabetes Distribution by Smoking Status", color="#4a0000", fontsize=14)
plt.xlabel("Smoking Status", color="#4a0000")
plt.ylabel("Diagnosed Diabetes", color="#4a0000")
plt.xticks(rotation=40, ha="right", color="#4a0000")
plt.yticks(color="#4a0000")

plt.show()



plt.figure(figsize=(6,4))

sns.barplot(
    data=df,
    x='employment_status',
    y='diagnosed_diabetes',
    palette=wine_palette
)

plt.title("Diabetes Rate by Employment Status", color="#4a0000", fontsize=14)
plt.xlabel("Employment Status", color="#4a0000")
plt.ylabel("Mean Diabetes Rate", color="#4a0000")
plt.xticks(rotation=45, ha="right", color="#4a0000")
plt.yticks(color="#4a0000")

plt.show()



numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

fig, axs = plt.subplots(
    nrows=int(np.ceil(len(numeric_cols)/3)),
    ncols=3,
    figsize=(14, 10)
)

axs = axs.flatten()

for i, col in enumerate(numeric_cols):
    axs[i].hist(df[col], bins=10, color=wine_palette[i % len(wine_palette)], edgecolor="#4a0000")
    axs[i].set_title(col, color="#4a0000")
    axs[i].tick_params(axis='both', colors="#4a0000")

# remove empty lines
for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])


plt.tight_layout(pad=3.0) # margin between plots
plt.show()



plt.figure(figsize=(12, 10))

corr = df[numeric_cols].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap=sns.color_palette(wine_palette, as_cmap=True),
    annot_kws={"size": 8, "color": "white"},
    linewidths=0.5,
    linecolor="#4a0000"
)

plt.title("Correlation Matrix", color="#4a0000", fontsize=16, pad=12)
plt.xticks(rotation=45, ha="right", color="#4a0000")
plt.yticks(rotation=0, color="#4a0000")

plt.tight_layout()
plt.show()



