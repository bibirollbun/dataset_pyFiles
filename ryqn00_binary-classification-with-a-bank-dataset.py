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


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
train.head()


train.info()


import seaborn as sns
import matplotlib.pyplot as plt


# Check for missing values explicitly
print(train.isnull().sum())


# Check the distribution of the target variable 'y'
print(train['y'].value_counts())
print(train['y'].value_counts(normalize=True)) # Shows percentage

# Plot the target variable
sns.countplot(x='y', data=train)
plt.title('Distribution of Target Variable (y)')
plt.show()


train.describe()


train.describe().columns


numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Calculate the grid size dynamically
n_features = len(numerical_features)
n_cols = 3  # You can adjust this
n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division

# Create subplots with the correct grid size
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.ravel()  # Flatten the array

for i, feature in enumerate(numerical_features):
    sns.histplot(train[feature], bins=30, ax=axes[i], kde=True)
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')

# Hide any unused subplots
for i in range(n_features, n_rows * n_cols):
    axes[i].set_visible(False)

plt.tight_layout()
plt.suptitle('Distribution of Numerical Features', y=1.02, fontsize=16)
plt.show()


# Create boxplots to see relationship with target
for feature in numerical_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='y', y=feature, data=train)
    plt.title(f'{feature} vs Subscription (y)')
    plt.show()


categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for feature in categorical_features:
    plt.figure(figsize=(10,5))
    # Plot count of each category, colored by the target 'y'
    sns.countplot(x=feature, hue='y', data=train)
    plt.title(f'Subscription rate by {feature}')
    plt.xticks(rotation=45)
    plt.show()

    # You can also calculate the percentage of subscriptions per category
    print(pd.crosstab(train[feature], train['y'], normalize='index') * 100)


# Compute the correlation matrix
corr_matrix = train.corr(numeric_only=True)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()


def handle_unknown(df = pd.DataFrame):

    # For this strategy, we will only turn unknown values into the most frequent value that appears
    df = df.copy()
    to_change = ['job', 'education', 'contact']
    for col in to_change:
        df[col] = df[col].replace('unknown', df[df[col] != 'unknown'][col].mode()[0])
    return df


from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = train.drop(columns=['id', 'y'], axis=1)
y = train['y']

# Split the data into training and testing sets (e.g., 80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


!pip install --upgrade scikit-learn==1.2.2 imbalanced-learn==0.10.1 --user


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, FunctionTransformer, OrdinalEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

to_ohe = ['job','marital', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
to_ord = ['education']
numerical_features = ['age', 'balance', 'day', 'pdays', 'previous']
# We can probably robust scale everything


handle_unknown_fct = FunctionTransformer(handle_unknown)

preprocess_pipeline = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numerical_features),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), to_ohe),
        ('ord', OrdinalEncoder(), to_ord)
    ],
    remainder='passthrough'
)

# Wrap with full pipeline (including your unknown handler first)
pre_process_pipeline = Pipeline(steps=[
    ('handle_unknown', handle_unknown_fct),   # custom transformer
    ('preprocess', preprocess_pipeline)
])


# Define the models
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=200),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42) # Note: GB doesn't have class_weight, so use SMOTE
}

# Train and evaluate each model
for name, model in models.items():
    # Create a pipeline: Preprocessor -> Model
    pipeline = Pipeline(steps=[('preprocessor', pre_process_pipeline),
                              ('classifier', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    # Print evaluation metrics
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred))


test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
test


pipeline_sub = Pipeline(steps=[('preprocessor', pre_process_pipeline),
                            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=200))])
pipeline.fit(X_train, y_train)
test_ready = test.drop(columns='id')
y_pred_sub = pipeline.predict_proba(test_ready)


y_pred_sub[:,1]


submission = pd.DataFrame({'id': test['id'], 'y': y_pred_sub[:,1]})
submission.to_csv('/kaggle/working/submission.csv', index=False)


# print(classification_report(y_test, y_pred))
# print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba)) # Use predicted probabilities
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
# plt.title('Confusion Matrix')
#plt.show()


#from imblearn.over_sampling import SMOTE
#smote = SMOTE(random_state=42)
#X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
# For Gradient bossting

