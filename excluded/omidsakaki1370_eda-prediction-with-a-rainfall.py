# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, ttest_ind
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

import warnings
warnings.filterwarnings('ignore')


# Load the data
df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')

# Display the first few rows of the data
df.head()


# Check data types
print(df.info())


# Check for missing values
print(df.isnull().sum())


# Descriptive statistics for numerical columns
print(df.describe())


# Distribution of the target variable (rainfall)
print(df['rainfall'].value_counts())


df.hist(bins=30, figsize=(15, 10))
plt.show()


plt.figure(figsize=(15, 10))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()


sns.countplot(x='rainfall', data=df)
plt.title('Distribution of Rainfall')
plt.show()


# Calculate the correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(x='day', y='pressure', data=df)
plt.title('Pressure Trend Over Time')
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(x='day', y='temparature', data=df)
plt.title('Temperature Trend Over Time')
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(x='day', y='rainfall', data=df)
plt.title('Rainfall Trend Over Time')
plt.show()


def get_season(day):
    if day <= 90:
        return 'Winter'
    elif day <= 180:
        return 'Spring'
    elif day <= 270:
        return 'Summer'
    else:
        return 'Fall'

df['season'] = df['day'].apply(get_season)


def degrees_to_direction(degrees):
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    index = round(degrees % 360 / 45) % 8
    return directions[index]

df['winddirection'] = df['winddirection'].apply(degrees_to_direction)


sns.countplot(x='season', data=df)
plt.title('Distribution of Seasons')
plt.show()


sns.countplot(x='winddirection', data=df)
plt.title('Distribution of winddirection')
plt.show()


#df['rainfall'] = df['rainfall'].apply(lambda x: 1 if x > 0 else 0)


plt.figure(figsize=(10, 6))
sns.boxplot(x='rainfall', y='pressure', data=df)
plt.title('Pressure on Rainy vs Non-Rainy Days')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='rainfall', y='humidity', data=df)
plt.title('Humidity on Rainy vs Non-Rainy Days')
plt.show()


z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))

# Identify outliers (Z-score > 3)
outliers = (z_scores > 3).any(axis=1)
print(f"Number of outliers: {outliers.sum()}")


outlier_rows = df[outliers]
print("Outlier rows:")
print(outlier_rows)


for col in df.select_dtypes(include=[np.number]):
    upper_limit = df[col].quantile(0.99)
    lower_limit = df[col].quantile(0.01)
    df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
    df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])


z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))

# Identify outliers (Z-score > 3)
outliers = (z_scores > 3).any(axis=1)
print(f"Number of outliers: {outliers.sum()}")


df = df[~outliers]
print(f"Data after removing outliers: {df.shape}")


sns.pairplot(df[['pressure', 'temparature', 'humidity', 'rainfall']], hue='rainfall')
plt.show()


columns = ['pressure', 'maxtemp', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

for col in columns:
    plt.figure(figsize=(8, 6)) 
    sns.scatterplot(x='temparature', y=col, data=df, hue='rainfall')
    plt.title(f'Scatter Plot of {col} vs Rainfall') 
    plt.xlabel(col) 
    plt.ylabel('Rainfall') 
    plt.show()  


# T-test for pressure between rainy and non-rainy days
rainy_pressure = df[df['rainfall'] == 1]['pressure']
non_rainy_pressure = df[df['rainfall'] == 0]['pressure']
t_stat, p_value = ttest_ind(rainy_pressure, non_rainy_pressure)
print(f"T-test for pressure: t-statistic = {t_stat}, p-value = {p_value}")


# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Visualize PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['rainfall'])
plt.title('PCA: Dimensionality Reduction to 2 Principal Components')
plt.show()


# Apply K-Means clustering
kmeans = KMeans(n_clusters=3)
df['cluster'] = kmeans.fit_predict(scaled_data)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df['cluster'])
plt.title('Clustering with K-Means')
plt.show()


df.head()


encoder = LabelEncoder()
df['season'] = encoder.fit_transform(df['season'])
df['winddirection'] = encoder.fit_transform(df['winddirection'])


df.head()


X = df.drop(columns=['id', 'day' ,'rainfall'])
y = df['rainfall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print the sizes of the train and test sets
print("The size of the input train data is: {}".format(X_train.shape))
print("The size of the output train data is: {}".format(y_train.shape))
print("The size of the input test data is: {}".format(X_test.shape))
print("The size of the output test data is: {}".format(y_test.shape))


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the sizes of the scaled train and test sets
print("The size of the scaled input train data is: {}".format(X_train_scaled.shape))
print("The size of the scaled input test data is: {}".format(X_test_scaled.shape))


models = {
    'RF': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1],
            'kernel': ['linear', 'rbf']
        }
    },
    'LR': {
        'model': LogisticRegression(random_state=42),
        'params': {
            'C': [0.1, 1],
            'solver': ['liblinear']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5],
            'weights': ['uniform']
        }
    },
    'NB': {
        'model': GaussianNB(),
        'params': {}
    },
    'GB': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [3]
        }
    },
    'XGB': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss'),
        'params': {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [3]
        }
    },
    'LGBM': {
        'model': LGBMClassifier(random_state=42),
        'params': {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [3]
        }
    }
}


results = {}

plt.figure(figsize=(10, 8))

for name, config in models.items():
    print(f"--- Tuning {name} ---")
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=3,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = best_model.predict(X_test_scaled)
    
    # Check if y_test has both classes
    if len(np.unique(y_test)) < 2:
        print(f"Skipping {name} because y_test has only one class.")
        continue
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC': roc_auc,
        'Best Params': grid_search.best_params_
    }

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Rainfall Prediction Models')
plt.legend(loc='lower right')
plt.show()

results_df = pd.DataFrame(results).T
print("\n--- Model Performance Comparison ---")
print(results_df)


# Define the XGBClassifier model
xgb_model = XGBClassifier(
    n_estimators=500,       # Number of trees
    learning_rate=0.05,     # Step size shrinkage
    max_depth=6,            # Maximum depth of trees
    subsample=0.8,          # Randomly sample 80% of the dataset for each tree
    colsample_bytree=0.8,   # Select 80% of features for each tree
    eval_metric="auc",      # Evaluation metric optimized for ROC-AUC
    use_label_encoder=False,
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test_scaled)
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC-AUC:", roc_auc)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'xgb (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for xgb')
plt.legend(loc='lower right')
plt.show()


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
print(test.head())
print(test.isnull().sum())


test['winddirection'] = test['winddirection'].fillna(method = 'pad')
test['winddirection'] = test['winddirection'].apply(degrees_to_direction)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(test.select_dtypes(include=[np.number]))

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

kmeans = KMeans(n_clusters=3)
test['cluster'] = kmeans.fit_predict(scaled_data)

test['season'] = test['day'].apply(get_season)

encoder = LabelEncoder()
test['season'] = encoder.fit_transform(test['season'])
test['winddirection'] = encoder.fit_transform(test['winddirection'])

X_test = test.drop(columns=['id','day'])

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)


target = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
target.head()


y_pred = xgb_model.predict(X_test_scaled)


test_preds_final = y_pred.copy()
submission_file = test.reset_index()[['id']]
submission_file['Predicted rainfall'] = test_preds_final
submission_file = submission_file.set_index("id")
submission_file.head()


submission_file.to_csv("/kaggle/working/submission.csv")

