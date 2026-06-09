import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train_df.head()


print("Dataset Shape:", train_df.shape)
print("\nColumn Info:\n", train_df.info())
print("\nMissing Values:\n", train_df.isnull().sum())
print("\nUnique Values per Column:\n", train_df.nunique())


print("\nTarget Value Counts:\n", train_df['Fertilizer Name'].value_counts())
sns.countplot(data=train_df, x='Fertilizer Name')
plt.title("Target Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


train_df.describe().T


train_df['Soil Type'].value_counts()


g = sns.catplot(data=train_df, x="Soil Type", col="Fertilizer Name",
                kind="count", col_wrap=4, height=4, sharey=False)
g.set_titles("{col_name}")
g.set_xticklabels(rotation=45)
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Fertilizer Usage across Soil Types (Faceted)")
plt.show()


nutrients = ['Nitrogen', 'Phosphorous', 'Potassium']
for col in nutrients:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=train_df, x='Soil Type', y=col)
    plt.title(f"{col} levels across Soil Types")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


ct = pd.crosstab(train_df['Soil Type'], train_df['Fertilizer Name'], normalize='index') * 100
ct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20c')
plt.ylabel("Percentage")
plt.title("Fertilizer Composition by Soil Type")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


cat_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']
for col in cat_cols:
    sns.countplot(y=col, data=train_df)
    plt.title(f"Count of {col}")
    plt.show()


# Cross-tab
pivot = pd.crosstab(train_df['Soil Type'], train_df['Crop Type'])

# Plot
plt.figure(figsize=(14, 6))
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu', linewidths=0.5)

plt.title("Soil Type vs Crop Type - Frequency Heatmap", fontsize=16)
plt.xlabel("Crop Type")
plt.ylabel("Soil Type")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen']
sns.heatmap(train_df[cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Environmental Factors and Nitrogen")
plt.show()


cols = ['Temparature', 'Humidity', 'Moisture', 'Potassium']
sns.heatmap(train_df[cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Environmental Factors and Potassium")
plt.show()


cols = ['Temparature', 'Humidity', 'Moisture', 'Phosphorous']
sns.heatmap(train_df[cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Environmental Factors and Phosphorous")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Group and get top fertilizer and its count
def top_fert(x):
    return x.value_counts().idxmax(), x.value_counts().max()

top_fert_data = train_df.groupby('Crop Type')['Fertilizer Name'].apply(top_fert).reset_index()
top_fert_data[['Fertilizer Name', 'Count']] = pd.DataFrame(top_fert_data['Fertilizer Name'].tolist(), index=top_fert_data.index)

# Sort for better visuals
top_fert_data = top_fert_data.sort_values('Count', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=top_fert_data, x='Count', y='Crop Type', hue='Fertilizer Name', dodge=False)

# Annotate bars with count values
for i, row in top_fert_data.iterrows():
    plt.text(row['Count'] + 50, i, f"{row['Count']}", va='center')

plt.title("Most Frequent Fertilizer per Crop (with Count)", fontsize=14)
plt.xlabel("Usage Count")
plt.ylabel("Crop Type")
plt.legend(title='Fertilizer', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = train_df.copy()
for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(['id', 'Fertilizer Name'], axis=1)
y = df['Fertilizer Name']

model = RandomForestClassifier()
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title('Feature Importance for Fertilizer Prediction')
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Select numeric features
features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
X = train_df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Apply PCA
pca = PCA(n_components=2)  # You can change to 3 for 3D
X_pca = pca.fit_transform(X_scaled)

# Put into a DataFrame for plotting
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Crop Type'] = train_df['Crop Type']


plt.figure(figsize=(10, 6))
centroids = pca_df.groupby('Crop Type')[['PC1', 'PC2']].mean().reset_index()
sns.scatterplot(data=centroids, x='PC1', y='PC2', hue='Crop Type', palette='tab10', s=150)

for i, row in centroids.iterrows():
    plt.text(row['PC1'] + 0.001, row['PC2'], row['Crop Type'], fontsize=10)

plt.title('PCA Centroids of Crop Types')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend().remove()
plt.tight_layout()
plt.show()


print("Happy Coding!")




