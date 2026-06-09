import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings("ignore", category=FutureWarning)


train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_data.head()


test_data.head()


train_data.info()


train_data.describe()


test_data.info()


test_data.describe()


print(train_data.isnull().sum())


print(test_data.isnull().sum())


sns.histplot(train_data['rainfall'], bins=20, kde=True)
plt.title("Rainfall Distribution")
plt.show()


sns.countplot(x=train_data['rainfall'])
plt.title("Rainfall Class Distribution")
plt.show()


print(train_data['rainfall']. value_counts())


train_data.hist(figsize=(12, 8), bins=30)
plt.show()



corr = train_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Select relevant numerical columns (all columns except 'id' and 'rainfall')
numerical_features = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

# Subset the train dataset to include only numerical features (excluding 'id' and 'rainfall')
X = train_data[numerical_features]

# Step 1: Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Step 3: Explained Variance Ratio (how much variance is captured by each component)
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# Step 4: Plot Cumulative Explained Variance to decide how many components to keep
cumulative_variance = pca.explained_variance_ratio_.cumsum()

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Step 5: Number of Components to Keep
# Typically, you want to keep enough components to explain 80-90% of the variance
n_components = (cumulative_variance >= 0.90).argmax() + 1
print(f"Number of components to keep: {n_components}")

# Step 6: Apply PCA with the selected number of components
pca = PCA(n_components=n_components)
X_pca_reduced = pca.fit_transform(X_scaled)

# Create a DataFrame of the PCA components for inspection
X_pca_df = pd.DataFrame(X_pca_reduced, columns=[f'PC{i+1}' for i in range(n_components)])

# Preview the PCA DataFrame
print(X_pca_df.head())



# Create 'Month' feature from 'Day'
train_data['month'] = ((train_data['day'] - 1) // 30) + 1  # Approximate months


sns.boxplot(x=train_data['month'], y=train_data['rainfall'])
plt.xticks(rotation=45)
plt.show()


train_data['day_sin'] = np.sin(2 * np.pi * train_data['day'] / 365)
train_data['day_cos'] = np.cos(2 * np.pi * train_data['day'] / 365)


train_data.head()


train_data.info()


# Dataframe
X = train_data.drop(columns=['id', 'rainfall'])  # Exclude 'id' and target column
y = train_data['rainfall']

# Compute mutual information scores
mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)

# Create a DataFrame for better visualization
mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_scores_df = mi_scores_df.sort_values(by='MI Score', ascending=False)

print(mi_scores_df)


def plot_mi_scores(scores_df):
    scores = scores_df.set_index('Feature')['MI Score'].sort_values(ascending=True)  # Convert to Series
    width = np.arange(len(scores))
    ticks = list(scores.index)
    
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.xlabel("Mutual Information Score")
    plt.title("Mutual Information Scores")

plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores_df)  # Now you can pass the whole DataFrame
plt.show()


sns.scatterplot(x=train_data['cloud'], y=train_data['temparature'], hue=train_data['rainfall'])


sns.scatterplot(x=train_data['sunshine'], y=train_data['temparature'], hue=train_data['rainfall'])


sns.scatterplot(x=train_data['pressure'], y=train_data['temparature'], hue=train_data['rainfall'])


sns.scatterplot(x=train_data['humidity'], y=train_data['temparature'], hue=train_data['rainfall'])


sns.scatterplot(x=train_data['windspeed'], y=train_data['winddirection'], hue=train_data['rainfall'])


sns.scatterplot(x=train_data['dewpoint'], y=train_data['temparature'], hue=train_data['rainfall'])


train_data['temp_range'] = train_data['maxtemp'] - train_data['mintemp']


train_data.head()


sns.lmplot(x='temp_range', y='temparature', hue='rainfall', data=train_data)


train_data['dew_point_spread'] = train_data['temparature'] - train_data['dewpoint']


train_data.head()


sns.lmplot(x='dew_point_spread', y='temparature', hue='rainfall', data=train_data)


train_data['relative_humidity'] = (train_data['humidity'] / 100) * train_data['temparature']


train_data.head()


sns.lmplot(x='relative_humidity', y='temparature', hue='rainfall', data=train_data)


train_data['cloud_cover_intensity'] = train_data['cloud'] * train_data['sunshine']


train_data.head()


sns.lmplot(x='cloud_cover_intensity', y='temparature', hue='rainfall', data=train_data)


train_data['wind_u'] = train_data['windspeed'] * np.cos(train_data['winddirection'] * np.pi / 180)


train_data.head()


sns.scatterplot(x=train_data['wind_u'], y=train_data['day'], hue=train_data['rainfall'])


train_data['wind_v'] = train_data['windspeed'] * np.sin(train_data['winddirection'] * np.pi / 180)


train_data.head()


sns.scatterplot(x=train_data['wind_v'], y=train_data['day'], hue=train_data['rainfall'])


train_data['temp_rolling_3'] = train_data['temparature'].rolling(3).mean()
train_data['humidity_rolling_3'] = train_data['humidity'].rolling(3).mean()


train_data.head()


train_data['prev_day_rainfall'] = train_data['rainfall'].shift(1)
train_data['prev_day_humidity'] = train_data['humidity'].shift(1)


train_data.head()




