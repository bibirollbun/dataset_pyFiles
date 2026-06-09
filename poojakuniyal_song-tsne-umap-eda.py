import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
train_df.head() 


train_df.info()


train_df.describe()


train_df.duplicated().sum(), train_df.isna().sum()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
test_df.head(3)


test_id = test_df['id']


train_df.drop(columns=['id'], axis=1,inplace=True)
test_df.drop(columns=['id'], axis=1, inplace=True)


sns.set_style('whitegrid')


train_df['BeatsPerMinute'].plot(kind='kde')


import math

# Get numeric columns excluding 'BeatsPerMinute'
numeric_cols = train_df.drop(columns='BeatsPerMinute').select_dtypes(include='number').columns

# Determine grid size
n_cols = 2
n_rows = math.ceil(len(numeric_cols) / n_cols)

plt.figure(figsize=(12, 4 * n_rows))  # Adjust height based on rows

# Plot histograms
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(train_df[col], bins=100, kde=True, color='red', element='step')
    plt.title(col)

plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show();


train_df.plot.hexbin(x='Energy', y='BeatsPerMinute', gridsize=50, cmap='coolwarm')


train_df.plot.hexbin(x='MoodScore', y='BeatsPerMinute', gridsize=50, cmap='coolwarm')


column_pairs = [
    ('RhythmScore', 'Energy'),
    ('AudioLoudness', 'InstrumentalScore'),
    ('MoodScore', 'TrackDurationMs'),
    ('AcousticQuality', 'LivePerformanceLikelihood')
]

for x_col, y_col in column_pairs:
    sns.lmplot(
        x=x_col,
        y=y_col,
        data=train_df,
        fit_reg=True,
        height=3,
        scatter_kws={'color': 'skyblue'},
        line_kws={'color': 'blue'}
    )



print(f"The size dataset is {train_df.shape[0]}, rows")


import umap
from sklearn.preprocessing import StandardScaler

df = train_df
df = df.sample(n=200000, random_state=42)
features = df.drop(columns=['BeatsPerMinute'])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Apply UMAP
reducer = umap.UMAP(n_components=2,n_neighbors=10, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Visualize
umap_df = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
umap_df['BPM'] = df['BeatsPerMinute']

plt.figure(figsize=(10, 6))
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='BPM', palette='viridis', alpha=0.7)
plt.title("UMAP Projection Colored by BPM")
plt.show()


df.shape


from sklearn.manifold import TSNE

# Sample 50000 rows
df_sample = train_df.sample(n=50000, random_state=42)

# Drop non-feature columns
features = df_sample.drop(columns=['BeatsPerMinute'])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Run t-SNE in 3D
tsne = TSNE(n_components=3, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Create DataFrame
tsne_df = pd.DataFrame(X_tsne, columns=['X', 'Y', 'Z'])
tsne_df['BPM'] = df_sample['BeatsPerMinute']


print(tsne_df['BPM'].min(), tsne_df['BPM'].max()) # good spread



from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Normalize BPM
norm = plt.Normalize(tsne_df['BPM'].min(), tsne_df['BPM'].max())
colors = plt.cm.plasma(norm(tsne_df['BPM']))
sc = ax.scatter(tsne_df['X'], tsne_df['Y'], tsne_df['Z'],
                c=tsne_df['BPM'], cmap='plasma', alpha=1.0)
fig.colorbar(sc, ax=ax, label='Beats Per Minute')
ax.set_xlabel('TSNE 1')
ax.set_ylabel('TSNE 2')
ax.set_zlabel('TSNE 3')
plt.title("3D t-SNE Visualization Colored by BPM")
plt.show()


target = 'BeatsPerMinute'
for feature in train_df.columns:
    if feature not in [target]:
        sns.lmplot(x=feature, y=target, data=train_df, fit_reg=True, height=3,
                  scatter_kws={'color':'pink'},
                  line_kws= {'color':'red'})
plt.title('Linearity of features with respect to target variable');


from sklearn.model_selection import train_test_split


X = train_df.drop(columns=['BeatsPerMinute'], axis=1)
y = train_df['BeatsPerMinute']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=97)


from sklearn.metrics import mean_squared_error



import lightgbm as lgb

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 64,
    'max_depth': 8,
    'min_data_in_leaf': 700,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_threads': 8
}

model1 = lgb.LGBMRegressor(**params, n_estimators=400)
model1.fit(X_train, y_train)


y_pred1 = model1.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred1))
print(f"Test RMSE: {rmse:.4f}")



test_data = model1.predict(test_df) 


test_data[:10]


submission_df = pd.DataFrame({'id': test_id, 'BeatsPerMinute': test_data})
submission_df.head(10)


submission_df.to_csv('submission.csv', index=False)




