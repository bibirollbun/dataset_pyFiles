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
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
train.head()


sns.countplot(x='Personality', data=train, palette='pastel')
plt.title("Introvert vs Extrovert Count")
plt.show()

print(train['Personality'].value_counts(normalize=True))


def data_overview(df):
    overview = pd.DataFrame({
        'Missing': df.isnull().sum(),
        'Missing %': df.isnull().mean() * 100,
        'Dtype': df.dtypes,
        'Unique': df.nunique()
    })
    return overview.sort_values(by='Missing', ascending=False)

data_overview(train)


le = LabelEncoder()
train['target'] = le.fit_transform(train['Personality'])  # Extrovert = 1, Introvert = 0


import math

# Number of features
features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']

n_features = len(features)
n_cols = 3
n_rows = math.ceil(n_features / n_cols)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
axs = axs.flatten()  # Flatten the axes array for easy indexing

for i, col in enumerate(features):
    sns.histplot(data=train, x=col, hue='Personality', kde=True, ax=axs[i], palette='Set2')
    axs[i].set_title(f'{col} Distribution by Personality')

# Remove unused subplots if any
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()


# Create a copy of the training data to work with
df = train.copy()

# Now check missing values
missing_info = df.isnull().sum()
print("Missing values:\n", missing_info[missing_info > 0])


# Impute with median or use separate strategy
df['Friends_circle_size'].fillna(df['Friends_circle_size'].median(), inplace=True)


binary_map = {'Yes': 1, 'No': 0}
df['Stage_fear'] = df['Stage_fear'].map(binary_map)
df['Drained_after_socializing'] = df['Drained_after_socializing'].map(binary_map)
df['target'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})

test['Stage_fear'] = test['Stage_fear'].map(binary_map)
test['Drained_after_socializing'] = test['Drained_after_socializing'].map(binary_map)
test['Friends_circle_size'].fillna(df['Friends_circle_size'].median(), inplace=True)


features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']

for col in features:
    plt.figure(figsize=(6,4))
    sns.kdeplot(data=df, x=col, hue='Personality', fill=True, common_norm=False, palette='coolwarm', alpha=0.5)
    plt.title(f'Distribution of {col} by Personality')
    plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df[features + ['target']].corr(), annot=True, cmap='vlag', center=0)
plt.title("Feature Correlation")
plt.show()


# Fill missing values in numerical features
df['Friends_circle_size'].fillna(df['Friends_circle_size'].median(), inplace=True)


# Ensure no missing values in features used
df[features] = df[features].fillna(df[features].median())


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_scaled = StandardScaler().fit_transform(df[features])
pca = PCA(n_components=2)
df[['PCA1', 'PCA2']] = pca.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Personality', palette='Set1')
plt.title("PCA - Personality Clustering")
plt.show()


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(df[features], df['target'])

feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(8,6))
feat_imp.plot(kind='barh', color='teal')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Encode categorical features
df = train.copy()
test_df = test.copy()

binary_map = {'Yes': 1, 'No': 0}
df['Stage_fear'] = df['Stage_fear'].map(binary_map)
df['Drained_after_socializing'] = df['Drained_after_socializing'].map(binary_map)
df['target'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})

test_df['Stage_fear'] = test_df['Stage_fear'].map(binary_map)
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].map(binary_map)

# Fill missing values
df['Friends_circle_size'].fillna(df['Friends_circle_size'].median(), inplace=True)
test_df['Friends_circle_size'].fillna(df['Friends_circle_size'].median(), inplace=True)

# Final feature list
features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
            'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']

X = df[features]
y = df['target']
X_test = test_df[features]


X = X.fillna(X.median())
X_test = X_test.fillna(X.median())  # Use train median to fill test
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=features)
X_test = pd.DataFrame(imputer.transform(X_test), columns=features)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(random_state=42)
rf_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print("Random Forest CV Accuracy:", rf_scores.mean())

# Fit and predict
rf.fit(X, y)
rf_preds = rf.predict(X_test)


from xgboost import XGBClassifier

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_scores = cross_val_score(xgb, X, y, cv=5, scoring='accuracy')
print("XGBoost CV Accuracy:", xgb_scores.mean())

# Fit and predict
xgb.fit(X, y)
xgb_preds = xgb.predict(X_test)


import pandas as pd
import h2o
from h2o.automl import H2OAutoML

#Reading data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

#Droping id column
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

#AutoML Solution
h2o.init()
train_data = h2o.H2OFrame(train)
aml = H2OAutoML(max_runtime_secs=350,seed=5)
aml.train(y='Personality', training_frame=train_data)
best_model = aml.leader
test_data = h2o.H2OFrame(test)
predictions = best_model.predict(test_data)
predictions_df = predictions.as_data_frame()
sub['Personality'] = (predictions_df['predict'].values)
sub.to_csv('submission.csv', index=False)





