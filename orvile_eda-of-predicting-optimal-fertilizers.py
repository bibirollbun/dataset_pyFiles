import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats # for Cramér’s V

import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
print("Shape of the traning dataset:",train_df.shape)


train_df.head()


train_df.tail()


train_df.dtypes


print("Traning Dataset Info:\n")
print(train_df.info())


print("Traning Dataset - Numerical Features Summary:\n")
train_df.describe()


train_df['Fertilizer Name'].value_counts()


print("Which columns have missing data?\n")
print(train_df.isnull().sum())


print("Proportional missing data analysis\n")
missing = train_df.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0])


print("How many different values are in each column?\n")
print(train_df.nunique())


print("List of category variables for train:")
categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns
print(categorical_cols)

print("\nList numeric variables for train:")
numeric_cols = train_df.select_dtypes(include=['number']).columns
print(numeric_cols)


print(train_df[train_df == ''].count())

for col in numeric_cols:
    q1 = train_df[col].quantile(0.25)
    q3 = train_df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((train_df[col] < (q1 - 1.5 * iqr)) | (train_df[col] > (q3 + 1.5 * iqr))).sum()
    print(f"{col} number of outliers: {outliers}")


print("Correlation matrix:\n")
corr_matrix = train_df[numeric_cols].corr()
corr_matrix


plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".5f", cmap="coolwarm", square=True, cbar_kws={'label': 'Correlation'})
plt.title("Correlation Matrix between Numerical Variables")
plt.tight_layout()
plt.show()


print("Pairs of variables with very high correlation\n")
threshold = 0.7
high_corr = corr_matrix[(abs(corr_matrix) > threshold) & (abs(corr_matrix) < 1.0)]
print(high_corr.dropna(how='all').dropna(axis=1, how='all'))


categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train_df, x=col, hue='Fertilizer Name') 
    plt.title(f"Scatter between {col} and target")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))  # bias correction
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns
target_col = 'Fertilizer Name'

for col in categorical_cols:
    if col != target_col:
        score = cramers_v(train_df[col], train_df[target_col])
        print(f"Cramér's V between {col} and {target_col}: {score:.3f}")


scores = {}
for col in categorical_cols:
    if col != target_col:
        scores[col] = cramers_v(train_df[col], train_df[target_col])

scores_df = pd.DataFrame(list(scores.items()), columns=["Feature", "CramersV"])
plt.figure(figsize=(8, 5))
sns.barplot(data=scores_df, x="Feature", y="CramersV")
plt.xticks(rotation=45)
plt.title("Cramér's V Score of Categorical Variables with Fertilizer Name")
plt.tight_layout()
plt.show()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
print("Shape of the test dataset:",test_df.shape)


test_df.head()


test_df.tail()


test_df.dtypes


print("Test Dataset Info:\n")
test_df.info()


print("Test Dataset - Numerical Features Summary:\n")
test_df.describe()


print("Which columns have missing data?\n")
print(test_df.isnull().sum())


print("How many different values are in each column?\n")
print(test_df.nunique())


plt.figure(figsize=(12, 6))
sns.countplot(data=train_df, y='Fertilizer Name', order=train_df['Fertilizer Name'].value_counts().index)
plt.title('Fertilizer Distribution')
plt.xlabel('Count')
plt.ylabel('Fertilizer Name')
plt.tight_layout()
plt.show()


numeric_cols = train_df.select_dtypes(include=np.number).columns.drop('id')

train_df[numeric_cols].hist(figsize=(16, 12), bins=30)
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()


grouped_stats = train_df.groupby('Fertilizer Name')[numeric_cols].mean().T
grouped_stats.plot(kind='bar', figsize=(16, 8))
plt.title("Mean Feature Values by Fertilizer")
plt.ylabel("Mean Value")
plt.xlabel("Feature")
plt.tight_layout()
plt.show()


numerical_features = train_df.select_dtypes(include=['number']).columns

for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=train_df, x='Fertilizer Name', y=feature)
    plt.title(f'{feature} by Fertilizer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


cross_tab = pd.crosstab(train_df["Soil Type"], train_df["Fertilizer Name"])

plt.figure(figsize=(12, 6))
sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Soil Type vs. Fertilizer Name (Counts)")
plt.ylabel("Soil Type")
plt.xlabel("Fertilizer Name")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Encode categorical features first
df = train_df.copy()
df['Soil Type'] = LabelEncoder().fit_transform(df['Soil Type'])
df['Crop Type'] = LabelEncoder().fit_transform(df['Crop Type'])
X = df[numerical_features + ['Soil Type', 'Crop Type']]
y = df['Fertilizer Name']

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Fertilizer Name'] = y

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Fertilizer Name', palette='tab10')
plt.title("PCA Projection")
plt.tight_layout()
plt.show()


# Assuming train_df and numerical_features are defined
df = train_df.copy()

# Encode categorical features
df['Soil Type'] = LabelEncoder().fit_transform(df['Soil Type'])
df['Crop Type'] = LabelEncoder().fit_transform(df['Crop Type'])

# Feature and target separation
X = df[numerical_features + ['Soil Type', 'Crop Type']]
y = df['Fertilizer Name']

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA results to dataframe
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Fertilizer Name'] = y

# Plot 1: Original Orientation
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Fertilizer Name', palette='tab10')
plt.title("PCA Projection - Original Orientation")
plt.legend(loc='upper right')  # Set a fixed location for the legend
plt.tight_layout()
plt.show()

# Plot 2: Flipped Axes (PC2 vs PC1)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC2', y='PC1', hue='Fertilizer Name', palette='tab10')
plt.title("PCA Projection - Flipped Axes")
plt.legend(loc='upper right')  # Set a fixed location for the legend
plt.tight_layout()
plt.show()

# Plot 3: Inverted X-axis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Fertilizer Name', palette='tab10')
plt.gca().invert_xaxis()
plt.title("PCA Projection - Inverted X-axis")
plt.legend(loc='upper right')  # Set a fixed location for the legend
plt.tight_layout()
plt.show()

# Plot 4: Inverted Y-axis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Fertilizer Name', palette='tab10')
plt.gca().invert_yaxis()
plt.title("PCA Projection - Inverted Y-axis")
plt.legend(loc='upper right')  # Set a fixed location for the legend
plt.tight_layout()
plt.show()



# Assuming train_df and numerical_features are defined
df = train_df.copy()

# Encode categorical features
df['Soil Type'] = LabelEncoder().fit_transform(df['Soil Type'])
df['Crop Type'] = LabelEncoder().fit_transform(df['Crop Type'])

# Feature and target separation
X = df[numerical_features + ['Soil Type', 'Crop Type']]
y = df['Fertilizer Name']

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA results to dataframe
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Fertilizer Name'] = y

# Define legend locations
legend_locations = ['upper right', 'upper left', 'lower left', 'lower right', 'center']

# Create a plot for each legend location
for loc in legend_locations:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Fertilizer Name', palette='tab10')
    plt.title(f"PCA Projection - Legend at {loc}")
    plt.legend(loc=loc)
    plt.tight_layout()
    plt.show()



import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Assuming train_df is defined and loaded correctly
df = train_df.copy()

# Ensure numerical_features is a list
numerical_features = list(numerical_features)

# Encode categorical features
df['Soil Type'] = LabelEncoder().fit_transform(df['Soil Type'])
df['Crop Type'] = LabelEncoder().fit_transform(df['Crop Type'])

# Verify columns exist in the DataFrame
all_features = numerical_features + ['Soil Type', 'Crop Type']
for feature in all_features:
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame.")

# Feature and target separation
X = df[all_features]
y = df['Fertilizer Name']

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA results to DataFrame
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Fertilizer Name'] = y


# PCA sonuçlarını görselleştirme
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Fertilizer Name', data=pca_df, palette='viridis')
plt.title('PCA of Fertilizer Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='upper right')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Verileri eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Rastgele Orman sınıflandırma modelini oluşturma ve eğitme
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Modeli değerlendirme
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=7, random_state=42)
clusters = kmeans.fit_predict(X_pca)

pca_df['Cluster'] = clusters

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
plt.title('K-means Clustering of PCA Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


from sklearn.metrics import silhouette_score

sil_score = silhouette_score(X_pca, clusters)
print(f"Silhouette Score: {sil_score:.4f}")


from sklearn.metrics import davies_bouldin_score

db_score = davies_bouldin_score(X_pca, clusters)
print(f"Davies-Bouldin Score: {db_score:.4f}")


from sklearn.metrics import calinski_harabasz_score

ch_score = calinski_harabasz_score(X_pca, clusters)
print(f"Calinski-Harabasz Score: {ch_score:.4f}")


from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(y_encoded, clusters)


silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []

k_values = range(2, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)

    #sil = silhouette_score(X_pca, labels)
    #silhouette_scores.append(sil)

    dbs = davies_bouldin_score(X_pca, labels)
    davies_bouldin_scores.append(dbs)

    chs = calinski_harabasz_score(X_pca, labels)
    calinski_harabasz_scores.append(chs)

    #print(f"k={k} | Silhouette: {sil:.3f} | DB: {db:.3f} | CH: {ch:.0f}")
    print(f"k={k} |  DB: {dbs:.3f} | CH: {chs:.0f}")



'''plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Optimal k - Silhouette Method")
plt.grid(True)
plt.show()'''

plt.figure(figsize=(10, 6))
# plt.plot(k_values, silhouette_scores, marker='o', label='Silhouette Score')
plt.plot(k_values, davies_bouldin_scores, marker='s', label='Davies-Bouldin Score')
plt.plot(k_values, calinski_harabasz_scores, marker='^', label='Calinski-Harabasz Score')
plt.xlabel("Number of clusters (k)")
plt.title("Clustering Score Metrics")
plt.grid(True)
plt.legend()
plt.show()


features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous',
            'Crop Type', 'Soil Type']

combined = pd.concat([train_df[features], test_df[features]], axis=0)

combined_encoded = pd.get_dummies(combined)
X_train_enc = combined_encoded.iloc[:len(train_df)]
X_test_enc = combined_encoded.iloc[len(train_df):]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_test_scaled = scaler.transform(X_test_enc)

kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
train_clusters = kmeans.fit_predict(X_train_scaled)
test_clusters = kmeans.predict(X_test_scaled)

cluster_map = pd.DataFrame({
    'Cluster': train_clusters,
    'Fertilizer Name': train_df['Fertilizer Name']
})

cluster_to_fertilizer = cluster_map.groupby('Cluster')['Fertilizer Name'].agg(lambda x: x.mode()[0]).to_dict()

test_predicted_fertilizers = [cluster_to_fertilizer[cluster] for cluster in test_clusters]

submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': test_predicted_fertilizers
})

submission.to_csv('submission_kmeans.csv', index=False)
print("✅ KMeans tabanlı submission oluşturuldu.")


# Hedef değişken (etiket)
y = train_df['Fertilizer Name']

# Özellikler: hedef dışındaki tüm sütunlar
X = train_df.drop(['id','Fertilizer Name'], axis=1)


X = X.copy()  # orijinali korumak için

le_soil = LabelEncoder()
le_crop = LabelEncoder()

X['Soil Type'] = le_soil.fit_transform(X['Soil Type'])
X['Crop Type'] = le_crop.fit_transform(X['Crop Type'])

# y'yi de istersen encode edebilirsin
le_fert = LabelEncoder()
y_encoded = le_fert.fit_transform(y)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Target classes:", le_fert.classes_)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Tahmin
y_pred = clf.predict(X_test)

# Sonuçlar
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


importances = clf.feature_importances_
features = X.columns

fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.show()

