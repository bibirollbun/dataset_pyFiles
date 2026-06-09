import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
pd.option_context('mode.use_inf_as_na', True)


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')


df.info()


df.head()


df.describe()


df.duplicated().sum()


df.isnull().sum()


fig, axes = plt.subplots(4, 3, figsize=(15, 15), sharey=False)
axes = axes.flatten()

for i, col in enumerate(df.columns):
    sns.histplot(data=df, x=col, ax=axes[i])
plt.tight_layout()
plt.show()


df['rainfall'].value_counts()


plt.pie(df['rainfall'].value_counts(), labels=['Rain', 'No Rain'], autopct='%.1f%%', startangle=90, colors=['#577BC1', '#FFFBCA'])
plt.title('Rainfall Distribution')
plt.show()


fig, axes = plt.subplots(4, 3, figsize=(15, 15), sharey=True)
axes = axes.flatten()

# Plot each column
for i, col in enumerate(df.columns):
    if col != 'rainfall':
        sns.scatterplot(data=df, x=col, y='rainfall', alpha=0.05, s=185, ax=axes[i])
        axes[i].set_ylabel('Rainfall')
        axes[i].set_xlabel(col)
        axes[i].set_ylim(-0.3, 1.3)
        axes[i].set_yticks([0, 1])
        axes[i].legend([], [], frameon=False)

# Remove any empty subplots
for i in range(len(df.columns) - 1, 4 * 3):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
correlation_matrix = df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix,  mask=mask, cmap="viridis", annot=True, fmt=".1f", square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


pair_df = df.drop(['day', 'rainfall'], axis=1)
g = sns.PairGrid(pair_df)
g.map_lower(sns.scatterplot)
g.map_diag(sns.histplot, kde=True)

for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    g.axes[i, j].set_visible(False)

plt.show()


from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_vif(df):
  vif = pd.DataFrame()
  vif["feature"] = df.columns
  vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
  return vif


vifs = check_vif(df.drop('rainfall', axis=1))
vifs = vifs.sort_values('VIF', ascending=False)
vifs


plt.figure(figsize=(10, 6))
plt.barh(vifs['feature'], vifs['VIF'], color='#577BC1') 
plt.xlabel("VIF Value")
plt.ylabel("Features")
plt.title("Variance Inflation Factor (VIF) for Each Feature")
plt.gca().invert_yaxis()  
plt.show()


from sklearn.preprocessing import StandardScaler
df_to_scale = df.drop(['rainfall'], axis=1)
scaler = StandardScaler()
scaler.fit(df_to_scale)
df_scaled  = pd.DataFrame(scaler.transform(df_to_scale), columns = df_to_scale.columns)


vifs_on_scaled_data = check_vif(df_scaled)
vifs_on_scaled_data = vifs_on_scaled_data.sort_values('VIF', ascending=False)
vifs_on_scaled_data


plt.figure(figsize=(10, 6))
plt.barh(vifs_on_scaled_data['feature'], vifs_on_scaled_data['VIF'], color='#577BC1') 
plt.xlabel("VIF Value")
plt.ylabel("Features (Scaled)")
plt.title("Variance Inflation Factor (VIF) for Scaled Features")
plt.gca().invert_yaxis()  
plt.show()


from sklearn.feature_selection import mutual_info_classif

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


mi_scores = make_mi_scores(df.drop('rainfall', axis=1),df['rainfall'])
plot_mi_scores(mi_scores)


sns.swarmplot(x='rainfall', y='cloud', data=df)
plt.title('Cloud/Rainfall distribution')
plt.show()


sns.swarmplot(x='rainfall', y='day', data=df)
plt.title('Day/Rainfall distribution')
plt.show()


sns.swarmplot(x='rainfall', y='windspeed', data=df)
plt.title('WindSpeed/Rainfall distribution')
plt.show()

