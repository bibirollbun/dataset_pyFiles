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


# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Modeling
from xgboost import XGBClassifier

# Warnings
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')



# Check the shapes
print("Train shape:", train.shape)
print("Test shape:", test.shape)


train


# General information about the training data
train.info()


# Check for missing values in the training set
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("Missing values in train set:")
print(missing)


# Fill missing values
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
for col in numerical_cols:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(train[col].median(), inplace=True)

categorical_cols = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(train[col].mode()[0], inplace=True)


# Check target distribution
train['Personality'].value_counts(normalize=True)

sns.countplot(data=train, x='Personality')
plt.title('Distribution of Personality Types')
plt.show

## Note: The target variable 'Personality' is imbalanced. The 'Extrovert' class appears more frequently than the 'Introvert' class.


plt.figure(figsize=(15, 12))

for i, col in enumerate(numerical_cols):
    plt.subplot(3, 2, i+1)
    sns.boxplot(data=train, x='Personality', y=col)
    plt.title(f'{col} by Personality')

plt.tight_layout()
plt.show()


## Introverts tend to spend significantly more time alone.
## Extroverts are more active in social events, go outside more often, have larger friend circles, and post more frequently online.


# Categorical columns to analyze
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

# Set plot layout
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))

for i, col in enumerate(categorical_cols):
    plt.subplot(1, 2, i+1)
    sns.countplot(data=train, x=col, hue='Personality')
    plt.title(f'{col} by Personality')
    plt.xticks(rotation=15)
    plt.legend(title='Personality')

plt.tight_layout()
plt.show


## Individuals who report having stage fear are mostly classified as Introverts.
## Those who feel drained after socializing also tend to be Introverts.
## As expected, Extroverts are less likely to fear public situations and more energized by social interactions.


# Copy the training data to avoid modifying original
df_corr = train.copy()

# Convert target variable to binary (Extrovert: 1, Introvert: 0)
df_corr['Personality'] = df_corr['Personality'].map({'Introvert': 0, 'Extrovert': 1})

# Compute correlation matrix for numerical features + target
corr_matrix = df_corr.corr(numeric_only=True)

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


## `Time_spent_Alone` has a strong negative correlation with `Personality` (-0.75), indicating that introverts tend to spend more time alone.
## Features like `Social_event_attendance`, `Going_outside`, `Post_frequency`, and `Friends_circle_size` show moderate to strong positive correlations (0.63–0.67) with being Extrovert.
## The feature correlations are reasonable and do not indicate problematic multicollinearity.


# Target encoder
target_encoder = LabelEncoder()
train['Personality'] = target_encoder.fit_transform(train['Personality'])  # 0/1 

# Save class order
print(target_encoder.classes_)  # ['Introvert' 'Extrovert']

# Categorical column encoder (separately)
cat_encoder = LabelEncoder()
for col in categorical_cols:
    train[col] = cat_encoder.fit_transform(train[col])
    test[col] = cat_encoder.transform(test[col])


X_train = train.drop(columns=['id', 'Personality'])
y_train = train['Personality']
X_test = test.drop(columns=['id'])

# Handle class imbalance
scale_weight = y_train.value_counts()[1] / y_train.value_counts()[0]

# Train model
model = XGBClassifier(
    max_depth=5,
    learning_rate=0.18252892528897208,
    n_estimators=475,
    subsample=0.6490158672898574,
    colsample_bytree=0.9596111485713041,
    gamma=1.093285113282037,
    reg_alpha=1.9512890981255564,
    reg_lambda=1.2357155385052994,
    scale_pos_weight=scale_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Predict
test_preds = model.predict(X_test)


# Convert predictions back to string labels
submission['Personality'] = target_encoder.inverse_transform(test_preds)
submission.to_csv('submission.csv', index=False)

print(submission.head())

