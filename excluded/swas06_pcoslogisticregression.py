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


df_train = pd.read_csv("/kaggle/input/exploring-predictive-health-factors/train.csv")
df_test = pd.read_csv("/kaggle/input/exploring-predictive-health-factors/test.csv")


df_train.head(3)


df_test.head(3)


df_train.shape,df_test.shape


df_train.info()


df_train.columns = df_train.columns.str.replace(" ","_").str.lower()
df_test.columns = df_test.columns.str.replace(" ","_").str.lower()


df_train.columns


df_train.describe()


df_train.isnull().sum(),df_test.isnull().sum()


 df_train['weight_kg'].fillna(df_train['weight_kg'].median(), inplace=True)
 df_test['weight_kg'].fillna(df_test['weight_kg'].median(), inplace=True)


df_train.isnull().sum()


import matplotlib.pyplot as plt
import seaborn as sns

# Create a box plot for the 'weight' column
plt.figure(figsize=(6, 4))
sns.boxplot(y=df_train['weight_kg'])
plt.title('Box Plot of Weight')
plt.ylabel('Weight')
plt.show()


sns.histplot(df_train['weight_kg'], kde=True, bins=30)
plt.title('Histogram of Weight')
plt.xlabel('Weight')
plt.ylabel('Count')
plt.show()


categorical_columns = df_train.select_dtypes(include=['object']).columns
categorical_columns  


for col in categorical_columns:
    print(col, ":", df_train[col].unique())


df_train.isnull().sum()


for col in categorical_columns:    
    print(col, ":", df_train[col].fillna('Unknown', inplace=True))
    print(col, ":", df_train[col].unique())



fig, axes = plt.subplots(4, 3, figsize=(18, 18))  # Adjust figure size as necessary
axes = axes.flatten()  # Flatten the 2D array of axes into 1D for easier iteration

for ax, column in zip(axes, categorical_columns):
    # Calculate the percentage distribution of each category
    category_counts = df_train[column].value_counts(normalize=True) * 100  # normalize=True gives the relative frequencies
    
    # Plotting the distribution using barplot
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
    ax.set_title(f'Percentage Distribution of {column}')
    ax.set_ylabel('Percentage of Policyholders (%)')
    ax.set_xlabel(column)  # Set xlabel to the column name for clarity

plt.tight_layout()  # Adjusts plot parameters for better fit in the figure window
plt.show()


import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score



X = df_train.drop('pcos',axis =1)  # Drop target column
y = df_train['pcos']  # Target variable



y = y.map({'Yes': 1, 'No': 0})  

# Split Data for Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


categorical_columns = ['age', 'hormonal_imbalance', 'hyperandrogenism', 'hirsutism',
       'conception_difficulty', 'insulin_resistance', 'exercise_frequency',
       'exercise_type', 'exercise_duration', 'sleep_hours',
       'exercise_benefit']
numerical_columns =['weight_kg']

# Create transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_columns),
    ('cat', categorical_transformer, categorical_columns)
])

# Classifiers to try
classifiers = {
'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # Add XGBoost
}

# Iterate over classifiers and evaluate using ROC AUC
for name, classifier in classifiers.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Predict probability scores (needed for ROC AUC)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # Only the probability for the positive class
    
    # Compute ROC AUC
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"{name} AUC-ROC: {auc_score:.4f}")


from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# List of classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    'SVM': SVC(probability=True),
    "XGBoost": xgb.XGBClassifier(scale_pos_weight=5)
}

# Iterate through classifiers and apply recall threshold adjustment
for name, model in classifiers.items():
    # Train the model
    pipeline.fit(X_train, y_train)

    # Get predicted probabilities for the positive class (class 1)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Set a new threshold to increase recall (lower than 0.5)
    threshold = 0.4  # Adjust this threshold value to see how recall changes
    y_pred_new = (y_prob >= threshold).astype(int)

    # Evaluate recall with the new threshold
    recall = recall_score(y_test, y_pred_new)
    print(f"{name} - Recall at threshold {threshold}: {recall}")

    # Optionally, plot the ROC curve to visualize the trade-off between recall and precision
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{name} ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.show()


pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", xgb.XGBClassifier())
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)

 # Predict probability scores (needed for ROC AUC)
y_prob = pipeline.predict_proba(X_test)[:, 1]  # Only the probability for the positive class
    
    # Compute ROC AUC
auc_score = roc_auc_score(y_test, y_prob)
print(f"{name} AUC-ROC: {auc_score:.4f}")




y_test_pred = pipeline.predict(df_test)


submission_df = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/sample_submission.csv')
submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

