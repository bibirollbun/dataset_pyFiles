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


!pip install numpy==1.26.4
!pip install scipy==1.15.3
!pip install scikit-learn==1.7.1
!pip install imbalanced-learn==0.14.0
# Pin other packages at compatible versions manually



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

print("Training data head:")
display(train_df.head())

print("\nTesting data head:")
display(test_df.head())


print("Training data descriptive statistics (numerical):")
display(train_df.describe())

print("\nTraining data descriptive statistics (categorical):")
display(train_df.describe(include='object'))

print("\nTesting data descriptive statistics (numerical):")
display(test_df.describe())

print("\nTesting data descriptive statistics (categorical):")
display(test_df.describe(include='object'))


# Set plot style
sns.set_style("whitegrid")

# Visualize distributions of numerical features in training data
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = numerical_cols.drop('id', errors='ignore') # Exclude id column
if 'y' in numerical_cols:
    numerical_cols = numerical_cols.drop('y') # Exclude target variable for now

print("Visualizing distributions of numerical features in training data:")
train_df[numerical_cols].hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Visualize distributions of categorical features in training data
categorical_cols = train_df.select_dtypes(include='object').columns

print("\nVisualizing distributions of categorical features in training data:")
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols):
    plt.subplot(3, 3, i + 1)
    sns.countplot(data=train_df, x=col, hue=col, palette='viridis') 
    plt.legend([],[], frameon=False) 
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Visualize distributions of numerical features in testing data
numerical_cols_test = test_df.select_dtypes(include=['int64', 'float64']).columns
numerical_cols_test = numerical_cols_test.drop('id', errors='ignore') # Exclude id column

print("Visualizing distributions of numerical features in testing data:")
test_df[numerical_cols_test].hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Visualize distributions of categorical features in testing data
categorical_cols_test = test_df.select_dtypes(include='object').columns

print("\nVisualizing distributions of categorical features in testing data:")
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols_test):
    plt.subplot(3, 3, i + 1)
    sns.countplot(data=test_df, x=col, hue=col, palette='viridis')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45, ha='right')
plt.legend([],[], frameon=False) 
plt.tight_layout()
plt.show()


# Analyze correlations between numerical features in training data
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = numerical_cols.drop('id', errors='ignore') # Exclude id column

print("Correlation matrix of numerical features in training data:")
plt.figure(figsize=(10, 8))
sns.heatmap(train_df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features (Training Data)')
plt.show()

# Analyze relationships between categorical features and the target variable
categorical_cols = train_df.select_dtypes(include='object').columns

print("\nAnalyzing relationships between categorical features and the target variable 'y':")
plt.figure(figsize=(15, 12))
for i, col in enumerate(categorical_cols):
    plt.subplot(3, 3, i + 1)
    sns.countplot(data=train_df, x=col, hue='y', palette='viridis')
    plt.title(f'Relationship between {col} and y')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Subscribed (y)')
plt.tight_layout()
plt.show()


# Identify rows with missing values
missing_rows = train_df[train_df.isnull().any(axis=1)]
print("Rows with missing values:")
display(missing_rows)

# Drop rows with missing values
train_df.dropna(inplace=True)

# Verify that missing values are removed
print("\nMissing values after dropping rows:")
display(train_df.isnull().sum().sum())


# Identify columns with 'unknown' values in train_df
unknown_cols_train = train_df.columns[train_df.isin(['unknown']).any()]
print("Columns with 'unknown' values in training data:")
print(unknown_cols_train)

# Identify columns with 'unknown' values in test_df
unknown_cols_test = test_df.columns[test_df.isin(['unknown']).any()]
print("\nColumns with 'unknown' values in testing data:")
print(unknown_cols_test)

# Examine the distribution of 'unknown' in identified columns of train_df
print("\nValue counts of 'unknown' in identified columns (Training Data):")
for col in unknown_cols_train:
    print(f"\nDistribution of '{col}':")
    display(train_df[col].value_counts())

# Examine the distribution of 'unknown' in identified columns of test_df
print("\nValue counts of 'unknown' in identified columns (Testing Data):")
for col in unknown_cols_test:
    print(f"\nDistribution of '{col}':")
    display(test_df[col].value_counts())


# Analyze the relationship between 'unknown' in identified columns and the target variable 'y'
print("\nAnalyzing relationship between 'unknown' and the target variable 'y' (Training Data):")
plt.figure(figsize=(15, 10))
for i, col in enumerate(unknown_cols_train):
    plt.subplot(2, 2, i + 1)
    sns.countplot(data=train_df, x=col, hue='y', palette='viridis')
    plt.title(f'Relationship between "{col}" and y (including unknown)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Subscribed (y)')
plt.tight_layout()
plt.show()


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# Identify categorical columns
cat_cols = train_df.select_dtypes(include='object').columns
num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
num_cols = num_cols.drop("y")

# get num_cols Index for skewed transformer
num_cols = list(num_cols) 


# Identify the indices of the skewed columns within the numerical features list
skewed_cols_names = ['balance', 'duration', 'campaign', 'previous']
skewed_col_indices = [num_cols.index(col) for col in skewed_cols_names if col in num_cols]
balance_col_index = num_cols.index('balance') if 'balance' in num_cols else None


# Define a custom transformer to apply log1p only to specified columns (by index)
# and handle potential non-positive values in a NumPy array using np.where
class SkewedLogTransformer(FunctionTransformer):
    def __init__(self, skewed_col_indices, balance_col_index):
         super().__init__(func=self.log_transform)
         self.skewed_col_indices = skewed_col_indices
         self.balance_col_index = balance_col_index

    def log_transform(self, X):
        X_transformed = X.copy()
        # Ensure X is a numpy array before proceeding
        if not isinstance(X_transformed, np.ndarray):
            X_transformed = np.asarray(X_transformed)

        # Check if the number of columns matches the expected indices
        if X_transformed.shape[1] != len(num_cols):
             print(f"Warning: Number of columns in SkewedLogTransformer input ({X_transformed.shape[1]}) does not match expected ({len(num_cols)}).")
             pass

        for i in self.skewed_col_indices:
             if i == self.balance_col_index and self.balance_col_index is not None:
                 # Handle negative balance values by mapping to a small positive number before log1p using np.where
                 X_transformed[:, i] = np.log1p(np.where(X_transformed[:, i] >= 0, X_transformed[:, i], 1e-9).astype(float))
             else:
                 # Apply log1p which handles zero values gracefully for other skewed columns
                 X_transformed[:, i] = np.log1p(X_transformed[:, i].astype(float))


        return X_transformed


# Define the numerical pipeline with imputation before scaling and transformation
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Impute missing numerical values
    ('log_transform', SkewedLogTransformer(skewed_col_indices, balance_col_index)), # Apply log transformation to skewed features by index
    ('scaler', StandardScaler()) # Scale numerical features
])

# Define the categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop = "first")) # One-hot encode categorical features, ignoring unknown categories
])

# Instantiate a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='passthrough' # Keep other columns (like 'id' and potentially 'y' if not dropped yet)
)


#check class weights of y
positive_count = (train_df['y'] == 1).sum()
negative_count = (train_df['y'] == 0).sum()

# Calculate ratio
scale_pos_weight = negative_count / positive_count
class_prior = [negative_count / len(train_df), positive_count / len(train_df)]

print(f"Negative / Positive (scale_pos_weight): {scale_pos_weight:.2f}")
print(f"Class Priors: {class_prior}")


# Split the data into training and validation sets
X = train_df.drop("y", axis = 1)
y = train_df["y"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

# Print the shapes of the resulting sets to verify the split
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# define algorithms to experiment
models = {
    "lgbm_model": LGBMClassifier(n_estimators = 340, learning_rate = 0.046,random_state=42, 
                                 objective='binary', metric='binary_logloss', is_unbalance = True, verbose=0),
    "XGBoost": XGBClassifier(n_estimators = 340, eval_metric = "logloss", scale_pos_weight = 7.29),
    "CatBoost": CatBoostClassifier(iterations=1000, scale_pos_weight= 7.29,  random_seed=42, verbose=0)
}


# Create the final pipeline 
# empty list to store the model results
results = []

# train and evaluate each model
for name, model in models.items():
    pipeline = Pipeline(steps = [
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probs >= 0.4).astype(int)

    #model evaluation
    acc = accuracy_score(y_test, predictions)
    pre = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    roc_auc_lr = roc_auc_score(y_test, predictions)

    #appedn results to the empty list
    results.append({
        "name": name,
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1 score": f1,
        "confusion matrix": cm,
        "ROC_AUC": roc_auc_lr
    })

# Identify best model by roc_auc score
best_model_index = np.argmax([r['ROC_AUC'] for r in results])

print(f"Best model is {best_model_index}")


# Create a DataFrame for better display
results_df = pd.DataFrame(results)

# Print the DataFrame without confusion matrix first
print("Model Performance Summary:")
print(results_df.drop(columns=['confusion matrix']).sort_values(by='ROC_AUC', ascending=False).reset_index(drop=True))


X_train = train_df.drop("y", axis = 1)
y_train = train_df["y"]


X_test = test_df

# Print the shapes of the resulting sets to verify the split
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)



model = CatBoostClassifier(iterations=1000, scale_pos_weight= 7.29,  random_seed=42, verbose=0)

pipeline = Pipeline(steps = [
        ("preprocessor", preprocessor),
        ("model", model)
    ])

pipeline.fit(X_train, y_train)
    
test_probs = pipeline.predict_proba(X_test)[:, 1]
submission_df = pd.DataFrame({'id': test_df.id,
                       'y': test_probs})
submission_df.to_csv('submission.csv', index=False)


submission_df.head(10)




