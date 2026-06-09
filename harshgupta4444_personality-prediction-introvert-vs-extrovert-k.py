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


# Install dependencies
!pip install --upgrade scikit-learn==1.3.0 imbalanced-learn


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


# Feature Engineering
def create_features(df):
    df = df.copy()
    df['Social_Interaction_Ratio'] = df['Social_event_attendance'] / (df['Time_spent_Alone'] + 1e-5)
    df['Energy_Balance'] = df['Drained_after_socializing'].map({'Yes': -1, 'No': 1}) * df['Social_event_attendance']
    df['Social_Activity_Index'] = (df['Social_event_attendance'] + df['Going_outside'] + df['Friends_circle_size'] / 10)
    df['Post_Frequency_Adjusted'] = df['Post_frequency'] / (df['Friends_circle_size'] + 1)
    return df



train_fe = create_features(train)
test_fe = create_features(test)


# Handle Missing Values
def handle_missing(df):
    df = df.copy()
    num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                'Friends_circle_size', 'Post_frequency', 'Social_Interaction_Ratio',
                'Social_Activity_Index', 'Post_Frequency_Adjusted']
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    cat_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df['Energy_Balance'].fillna(df['Energy_Balance'].median(), inplace=True)
    return df


train_clean = handle_missing(train_fe)
test_clean = handle_missing(test_fe)


# Encode Target
train_clean['Personality'] = train_clean['Personality'].map({'Introvert': 0, 'Extrovert': 1})



# Separate Features and Target
X = train_clean.drop(['id', 'Personality'], axis=1)
y = train_clean['Personality']


# Address Class Imbalance
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)



# Define Preprocessing
categorical_features = ['Stage_fear', 'Drained_after_socializing']
numerical_features = [col for col in X_res.columns if col not in categorical_features]



preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])



# Create Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selector', SelectKBest(score_func=f_classif, k=10)),
    ('classifier', SVC(probability=True, random_state=42))
])


# Grid Search Parameters
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'feature_selector__k': [8, 10, 12]
}



# Grid Search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X_res, y_res)


# Best Model
best_model = grid_search.best_estimator_


# Evaluate
X_train, X_val, y_train, y_val = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_val)


print("Classification Report:")
print(classification_report(y_val, y_pred))


cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Introvert', 'Extrovert'], yticklabels=['Introvert', 'Extrovert'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Predict on Test Data
test_final = test_clean.drop('id', axis=1)
test_preds = best_model.predict(test_final)



submission = pd.DataFrame({
    'id': test['id'],
    'Personality': ['Introvert' if x == 0 else 'Extrovert' for x in test_preds]
})




submission.to_csv('submission444.csv', index=False)
print("\nSubmission file created successfully!")
print(submission.head())




