import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import gc, os, random, warnings
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler


plt.style.use("ggplot")
sns.set(font_scale=1.1)
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

print("Train shape :", train.shape)
print("Test  shape :", test.shape)
display(train.head())


display(train.info())
display(train.describe(include="all").T.sort_index())


plt.figure(figsize=(12,6))
sns.heatmap(train.isnull(), cbar=False, cmap='viridis')
plt.title("Missing values heatmap")
plt.show()


missing_pct = train.isnull().mean().sort_values(ascending=False) * 100
missing_pct = missing_pct[missing_pct > 0]

plt.figure(figsize=(10,5))
sns.barplot(x=missing_pct.values, y=missing_pct.index, palette="Blues_d")
plt.title("Percentage of missing values per column")
plt.xlabel("% Missing")
plt.ylabel("Column")
plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(train.isnull().corr(), annot=True, cmap='coolwarm')
plt.title("Correlation between missing values")
plt.show()


target_col = "Personality"
colors = ['skyblue', 'salmon']
vc = train[target_col].value_counts().sort_values(ascending=False)
vc_pct = vc / len(train) * 100

fig, ax = plt.subplots(1,2,figsize=(14,4))
vc.plot.bar(ax=ax[0], color=colors)
ax[0].set_title("Absolute counts"); ax[0].set_ylabel("records")

vc_pct.plot.bar(ax=ax[1], color=colors)
ax[1].set_title("Percentage share"); ax[1].set_ylabel("% of train")
plt.suptitle("Target distribution"); plt.show()


cat_cols = ["Stage_fear"]
colors = ['skyblue', 'salmon']
for col in cat_cols:
    plt.figure(figsize=(8,3))
    sns.countplot(data=train, x=col, palette=colors)
    plt.title(f"{col} distribution"); plt.show()


column = ['Stage_fear', 'Social_event_attendance', 'Drained_after_socializing']

fig, axes = plt.subplots(len(column), 2, figsize=(14, 5 * len(column)))

for i, col in enumerate(column):
    stats = train.groupby(col)['Personality'].value_counts(normalize=True).unstack()

    # We display statistics as text on the left subgraph
    axes[i, 0].axis('off')
    table_text = stats.round(2).to_string()
    axes[i, 0].text(0, 1, table_text, fontsize=12, va='top', family='monospace')
    axes[i, 0].set_title(f"{col} vs Personality (proportions)")

    # Plot countplot
    sns.countplot(data=train, x=col, hue='Personality', ax=axes[i, 1])
    axes[i, 1].set_title(f"{col} vs Personality (counts)")
    axes[i, 1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()



num_cols = train.select_dtypes(include="number").columns.drop("id")
ncols = 3
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows*3))
axes = axes.flatten()

for ax, col in zip(axes, num_cols):
    sns.histplot(train[col], kde=True, ax=ax)
    ax.set_title(col)

plt.tight_layout(); plt.show()


num_cols = train.select_dtypes(include="number").columns.drop("id")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
colors = ['skyblue', 'salmon']

for i, col in enumerate(num_cols[:6]):
    sns.boxplot(
        data=train,
        x=target_col,
        y=col,
        order=vc.index,
        ax=axes[i], palette=colors
    )
    axes[i].set_title(f"{col} by Personality")
    axes[i].tick_params(axis='x')

plt.tight_layout()
plt.show()


num_cols = train.select_dtypes(include="number").columns.drop("id")
n_features = len(num_cols)
ncols = 3
nrows = (n_features + ncols - 1) // ncols 

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
axes = axes.flatten()

for i, feature in enumerate(num_cols):
    sns.scatterplot(
        x=train["Personality"], y=train[feature], alpha=0.5, ax=axes[i]
    )
    axes[i].set_title(f"{feature} vs. Personality")
    axes[i].set_xlabel("Personality")
    axes[i].set_ylabel(feature)

plt.tight_layout()
plt.show()


corr = train[num_cols].corr(method="spearman")
plt.figure(figsize=(9,7))
sns.heatmap(corr, cmap="coolwarm", annot=True, square=True)
plt.title("Spearman correlation")
plt.show()


colors = sns.color_palette('husl', len(num_cols))
rows = -(-len(num_cols) // 2)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(num_cols, colors), 1):
    plt.subplot(rows, 3, i)
    sns.violinplot(data=train, y=col, color=color)
    plt.title(f'Violin Plot of {col}', fontsize=14, color=color)
    plt.xlabel('')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


colors = sns.color_palette('husl', len(num_cols))
rows = -(-len(num_cols) // 2)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(num_cols, colors), 1):
    plt.subplot(rows, 3, i)
    sns.kdeplot(data=train, x=col, fill=True, color=color)
    sns.lineplot(data=train[col].sort_values().reset_index(drop=True), color='black', linewidth=1)
    plt.title(f'KDE + Trend of {col}', fontsize=14, color=color)
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()


def preprocess_data(df):
    df = df.copy()
    
    df.drop(['id'], axis=1, inplace=True)
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        median = df[col].median()
        df[col].fillna(median, inplace=True)

    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col].fillna("Unknown", inplace=True)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


X = preprocess_data(train)

features = X.drop(columns=["Personality"])
target = X["Personality"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=X['Personality'], palette="tab10", s=60)
plt.legend(title="ĞšĞ»Ğ°Ñ�Ñ�", bbox_to_anchor=(1.05, 1), loc='upper left')

    
plt.title("PCA Ğ²Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ�")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()

