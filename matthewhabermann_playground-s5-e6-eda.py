# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv").set_index("id")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
train.head()


train.info()
test.info()

# No missing values


train.describe()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# I am making bar plots since all numerical features are integral

for col in train.columns:

    if train[col].dtype == 'object':
        plt.figure(figsize=(8, 6))
        sns.countplot(x=col, data=train, color = "blue")
        plt.title(f'Frequency Distribution of {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        value_counts = train[col].value_counts().sort_index()
        sns.barplot(x=value_counts.index, y=value_counts.values, color='blue', ax=ax1)
        ax1.set_title(f'Frequency Distribution of {col}')
        ax1.set_xlabel(col)
        ax1.set_ylabel('Count')
            
        sns.boxplot(y=train[col], ax=ax2, palette='Set2')
        ax2.set_title(f'{col} Distribution')
        
        plt.tight_layout()
        plt.show()


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

numeric_train = train.select_dtypes(include=['number'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_train)

# Step 2: Apply PCA
pca = PCA()  
X_pca = pca.fit_transform(X_scaled)

# Step 3: Convert back to a DataFrame
pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
train_pca = pd.DataFrame(X_pca, columns=pca_columns)

train_pca.head()



loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=pca_columns,  # so the columns are the principal components
    index=numeric_train.columns,  # and the rows are the original features
)
loadings


# I stole this from the feature engineering course on Kaggle
def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs


plot_variance(pca)

# Interesting that the variance is roughly even. 


sns.catplot(
    y="value",
    col="variable",
    data=train_pca.melt(),
    kind='boxen',
    sharey=False,
    col_wrap=2,
);


correlation_matrix = numeric_train.corr()
plt.figure(figsize=(15, 12))


sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5, square=True)


plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)


plt.title('Correlation Matrix of Features', fontsize=16)
plt.tight_layout()


plt.show()


correlation_matrix = train_pca.corr()
plt.figure(figsize=(15, 12))


sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5, square=True)


plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)


plt.title('Correlation Matrix of Features', fontsize=16)
plt.tight_layout()


plt.show()


categorical_features = train.select_dtypes(include = ["object"]).columns
for col in categorical_features:
    print(train[col].value_counts())


from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

le = LabelEncoder()
y_encoded = le.fit_transform(train["Fertilizer Name"])  # your target column


X = train.drop(columns=["Fertilizer Name"])

# Automatically detect categorical columns (assumes object or category dtype)
cat_cols = X.select_dtypes(include=["object", "category"]).columns
num_cols = X.select_dtypes(include=["number"]).columns

# Encode categorical features
X_encoded = X.copy()
for col in cat_cols:
    X_encoded[col] = LabelEncoder().fit_transform(X[col])



# True/False mask for categorical features
discrete_mask = X_encoded.columns.isin(cat_cols)

mi_scores = mutual_info_classif(X_encoded, y_encoded, discrete_features=discrete_mask)

# Display nicely
mi_series = pd.Series(mi_scores, index=X_encoded.columns).sort_values(ascending=False)
print(mi_series)









