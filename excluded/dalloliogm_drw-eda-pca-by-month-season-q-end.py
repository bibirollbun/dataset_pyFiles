import polars as pl
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load and shrink data ---
train_df = pl.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
train_df = train_df.select(pl.all().shrink_dtype()).to_pandas()

# --- 2. Add calendar features ---
train_df['date'] = pd.to_datetime(train_df['timestamp'], unit='s')
train_df['month'] = train_df['date'].dt.month
train_df['dayofweek'] = train_df['date'].dt.dayofweek
train_df['is_tax_season'] = train_df['month'].isin([3, 4]).astype(int)
train_df['is_q_end'] = train_df['date'].dt.is_quarter_end.astype(int)






train_df.head()


train_df.shape


train_df.month.value_counts()


# --- 3. Select numeric features ---
feature_cols = [col for col in train_df.columns if col.startswith('X') or col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']]
X = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

# Optional: Subsample to 20k rows to plot faster
#X = X.sample(n=20000, random_state=42)
meta = train_df.loc[X.index, ['month', 'dayofweek', 'is_tax_season', 'is_q_end']]




X.head()


# --- 4. Dimensionality reduction (PCA + UMAP) ---
pca = PCA(n_components=50).fit_transform(X)
embedding = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(pca)

# --- 5. Plot helper ---
def plot_embedding(embedding, labels, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels, palette='Spectral', s=10, alpha=0.6)
    plt.title(title)
    plt.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# --- 6. Visualizations ---
plot_embedding(embedding, meta['month'], 'PCA+UMAP by Month')


plot_embedding(embedding, meta['dayofweek'], 'PCA+UMAP by Day of Week')



plot_embedding(embedding, meta['is_tax_season'], 'PCA+UMAP by Tax Season')



plot_embedding(embedding, meta['is_q_end'], 'PCA+UMAP by Quarter End')

