import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# !pip install -U kaleido


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from IPython.display import display, HTML
import plotly.express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
import plotly.figure_factory as ff
import plotly.graph_objects as go
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')
import nltk

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

train_df = train.copy()
test_df = test.copy()


# Train Dataset Overview
print('='*40, 'Train Dataset Overview', '='*40)
print(f"Train Dataset Shape: {train_df.shape}")
print('='*86)
print()

print('='*40, 'Train Data Info', '='*40)
train_df.info()
print('='*86)
print()

print('='*40, 'Train Numerical Features Summary', '='*40)
display(train_df.describe())
print('='*86)
print()

print('='*40, 'Train First 10 Rows', '='*40)
display(train_df.head(10))
print('='*86)
print()

print('='*40, 'Train Sample Preview', '='*40)
print(train_df.head())
print('='*86)
print()


# Test Dataset Overview
print('='*40, 'Test Dataset Overview', '='*40)
print(f"Test Dataset Shape: {test_df.shape}")
print('='*86)
print()

print('='*40, 'Test Data Info', '='*40)
test_df.info()
print('='*86)
print()

print('='*40, 'Test Numerical Features Summary', '='*40)
display(test_df.describe())
print('='*86)
print()

print('='*40, 'Test First 10 Rows', '='*40)
display(test_df.head(10))
print('='*86)
print()

print('='*40, 'Test Sample Preview', '='*40)
print(test_df.head())
print('='*86)
print()


# Basic info
print('='*86)
print(train_df['Personality'].value_counts())
print('='*86)

# Missing values
sns.heatmap(train_df.isnull(), cbar=False)
plt.title("Missing Values")
plt.show()

# Correlation
plt.figure(figsize=(14, 8))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# Distribution of features
train_df.drop(columns=['id', 'Personality']).hist(bins=30, figsize=(16, 10))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()


total_nan = train_df.isna().sum().sum()
total_inf = (train_df == np.inf).sum().sum()
total_ninf = (train_df == -np.inf).sum().sum()

print(f"Total NaN values: {total_nan}")
print(f"Total +inf values: {total_inf}")
print(f"Total -inf values: {total_ninf}")


# Fill missing values in specific categorical columns
train_df['Stage_fear'].fillna('Unknown', inplace=True)
train_df['Drained_after_socializing'].fillna('Unknown', inplace=True)
test_df['Stage_fear'].fillna('Unknown', inplace=True)
test_df['Drained_after_socializing'].fillna('Unknown', inplace=True)

# Encode categorical features
cat_cols = ['Stage_fear', 'Drained_after_socializing']
for col in cat_cols:
    le_cat = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    le_cat.fit(combined)
    train_df[col] = le_cat.transform(train_df[col])
    test_df[col] = le_cat.transform(test_df[col])

# Fill other numeric NaNs with column-wise median
train_df.fillna(train_df.median(numeric_only=True), inplace=True)
test_df.fillna(test_df.median(numeric_only=True), inplace=True)

# Encode target
le = LabelEncoder()
train_df['target'] = le.fit_transform(train_df['Personality'])  # Extrovert: 0, Introvert: 1


train_df.head()


# Basic info
print('='*86)
print(train_df['Personality'].value_counts())
print('='*86)

# Missing values
sns.heatmap(train_df.isnull(), cbar=False)
plt.title("Missing Values")
plt.show()

# Correlation
plt.figure(figsize=(14, 8))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# Distribution of features
train_df.drop(columns=['id', 'Personality']).hist(bins=30, figsize=(16, 10))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()


total_nan = train_df.isna().sum().sum()
total_inf = (train_df == np.inf).sum().sum()
total_ninf = (train_df == -np.inf).sum().sum()

print(f"Total NaN values: {total_nan}")
print(f"Total +inf values: {total_inf}")
print(f"Total -inf values: {total_ninf}")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Define numerical and categorical columns
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                  'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']


plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.histplot(train_df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
# plt.savefig('numerical_distributions.png')
plt.show()
plt.close()


plt.figure(figsize=(10, 5))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, 2, i)
    sns.countplot(data=train_df, x=col)
    plt.title(f'Count of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
# plt.savefig('categorical_distributions.png')
plt.show()
plt.close()


plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='Personality')
plt.title('Distribution of Personality')
plt.xlabel('Personality')
plt.ylabel('Count')
# plt.savefig('personality_distribution.png')
plt.show()
plt.close()


plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(data=train_df, x='Personality', y=col)
    plt.title(f'{col} vs Personality')
    plt.xlabel('Personality')
    plt.ylabel(col)
plt.tight_layout()
# plt.savefig('numerical_vs_personality.png')
plt.show()
plt.close()


plt.figure(figsize=(10, 5))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, 2, i)
    sns.countplot(data=train_df, x=col, hue='Personality')
    plt.title(f'{col} vs Personality')
    plt.xlabel(col)
    plt.ylabel('Count')
plt.tight_layout()
# plt.savefig('categorical_vs_personality.png')
plt.show()
plt.close()


plt.figure(figsize=(8, 6))
correlation_matrix = train_df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
# plt.savefig('correlation_matrix.png')
plt.show()
plt.close()


sampled_df = train_df.sample(frac=0.1, random_state=42)
sns.pairplot(sampled_df[numerical_cols + ['Personality']], hue='Personality')
# plt.savefig('pairplot.png')
plt.show()
plt.close()


plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    sns.violinplot(data=train_df, x='Personality', y=col)
    plt.title(f'{col} vs Personality (Violin)')
    plt.xlabel('Personality')
    plt.ylabel(col)
plt.tight_layout()
# plt.savefig('violin_plots.png')
plt.show()
plt.close()

print("EDA and Visualizations completed.")


# Separate features and target
X = train_df.drop(columns=['id', 'Personality', 'target'])
y = train_df['target']
X_test = test_df.drop(columns=['id'])


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test))
val_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(
        n_estimators=10000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.7,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50, verbose=100)
    
    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    val_scores.append(acc)
    print(f"Validation Accuracy: {acc:.4f}")
    
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits  # probability for class '1' (Introvert)


xgb.plot_importance(model, max_num_features=15, importance_type='gain', height=0.6)
plt.title("Top Feature Importances")
plt.show()


# Convert probabilities to final predictions
final_preds = (test_preds > 0.5).astype(int)
submission = test[['id']].copy()
submission['Personality'] = le.inverse_transform(final_preds)
submission.to_csv("submission_basic.csv", index=False)

print("Submission saved as submission_basic.csv")

