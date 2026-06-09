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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import chi2_contingency
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import time


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


print(train.columns)
print(test.columns)
print(train["y"].unique())


train.info()
test.info()


print(train.isna().sum())
print(train.duplicated().sum())


numerical_columns = ["age", "balance", "duration", "campaign", "pdays", "previous","day"]

for i in numerical_columns:
    plt.figure(figsize=(8, 4))   # Bigger figure for clarity
    # Histogram with kernel density estimate
    sns.histplot(train[i], bins=50, kde=True, color="steelblue", edgecolor="black")
    
    plt.title(f"Distribution of {i}", fontsize=14)
    plt.xlabel(i, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def balance_category(x):
    if x < -500:
        return "high_debt"
    elif x < 0:
        return "low_debt"
    elif x == 0:
        return "no_savings"
    else:
        return "saver"
train["balance_cat"] = train["balance"].apply(balance_category)
print(train["balance_cat"].unique())
train["previous_binary"] = (train["previous"] > 0).astype(int)
train["pdays_binary"] = (train["pdays"] > 0).astype(int)
print(train["pdays_binary"].value_counts())
minmaxscalers_columns=["day"]
standard_scaler_column=["age","balance","duration"]
log_transform_columns=["duration","campaign"]
numerical_to_drop=["previous","pdays"]


y=train["y"]
train=train.drop(["id","pdays","previous","y"],axis=1)


scaler_mm = MinMaxScaler()
train_mm = train.copy()
train_mm[minmaxscalers_columns] = scaler_mm.fit_transform(train[minmaxscalers_columns])
for col in minmaxscalers_columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(train[col], bins=50, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title(f"Original Distribution of {col}")

    sns.histplot(train_mm[col], bins=50, kde=True, ax=axes[1], color="salmon")
    axes[1].set_title(f"MinMax Scaled Distribution of {col}")

    plt.tight_layout()
    plt.show()
print(train_mm.info())


standardscaler_transformer=StandardScaler()
train_std = train.copy()
train_std[standard_scaler_column]=standardscaler_transformer.fit_transform(train[standard_scaler_column])
for col in standard_scaler_column:
    print(train[col].value_counts(sort=True))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(train[col], bins=50, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title(f"Original Distribution of {col}")
    sns.histplot(train_std[col], bins=50, kde=True, ax=axes[1], color="salmon")
    axes[1].set_title(f"Standard Scaled Distribution of {col}")
    plt.tight_layout()
    plt.show()
print(train_std.info())


# --- StandardScaler ---
scaler_std = StandardScaler()
train[standard_scaler_column] = scaler_std.fit_transform(train[standard_scaler_column])

# --- MinMaxScaler ---
scaler_mm = MinMaxScaler()
train[minmaxscalers_columns] = scaler_mm.fit_transform(train[minmaxscalers_columns])


# Choose one or more columns you applied the log transform on
cols = log_transform_columns
train_log=train.copy()
def log_transform(X):
    return np.log1p(X)
log_transformer=FunctionTransformer(log_transform)
train_log[log_transform_columns]=log_transformer.transform(train_log[log_transform_columns])
print(train[log_transform_columns].head())
print(train.info())
for col in cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Original distribution
    sns.histplot(train[col], bins=50, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title(f"Original Distribution of {col}")

    # Log-transformed distribution
    sns.histplot(train_log[col], bins=50, kde=True, ax=axes[1], color="salmon")
    axes[1].set_title(f"Log-Transformed Distribution of {col}")

    plt.tight_layout()
    plt.show()



def log_transform(X):
    return np.log1p(X)
log_transformer=FunctionTransformer(log_transform)
train[log_transform_columns]=log_transformer.transform(train[log_transform_columns])


for col in standard_scaler_column + minmaxscalers_columns + log_transform_columns :
    plt.figure(figsize=(6,4))
    sns.histplot(train[col], bins=50, kde=True, color="salmon")
    plt.title(f"Transformed Distribution of {col}")
    plt.show()


categorical_columns = [
    'job',           # type of job
    'marital',       # marital status
    'education',     # education level
    'default',       # has credit in default?
    'housing',       # has housing loan?
    'loan',          # has personal loan?
    'contact',       # contact communication type
    'month',         # last contact month of year
    'poutcome',      # outcome of previous marketing campaign
]
results=[]
for col in categorical_columns:
    contingency_table = pd.crosstab(train[col],y)
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
    results.append({
            'Feature': col,
            'Chi2 Score': chi2_stat,
            'P Value': p_val
        })
XX=pd.DataFrame(results).sort_values('Chi2 Score', ascending=False)
print()
# Or use a Chi2 score threshold (e.g., > 1000)
significant_features = XX[XX['Chi2 Score'] > 5000]['Feature'].tolist()
print(significant_features)
print(results)


for i in categorical_columns:
    print(train[i].value_counts())


from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Separate the columns
one_hot_columns = ['month', 'marital', 'education',"job",'poutcome','contact']
label_encoder_columns = ['default', 'housing', 'loan','balance_cat']

# Create a copy to preserve original data if needed
train_encoded = train.copy()

# Apply One-Hot Encoding to specified columns
train_encoded = pd.get_dummies(train_encoded, columns=one_hot_columns, prefix=one_hot_columns)

# Apply Label Encoding to the remaining categorical columns
label_encoders = {}  # Store encoders for potential inverse transformation
for col in label_encoder_columns:
    le = LabelEncoder()
    train_encoded[col] = le.fit_transform(train_encoded[col].astype(str))
    label_encoders[col] = le  # Store the encoder

# Display the transformed dataframe info
print("Transformed DataFrame Info:")
print(train_encoded.info())
print(f"\nNew shape: {train_encoded.shape}")


bool_columns = train_encoded.select_dtypes(include=['bool']).columns
train_encoded[bool_columns] = train_encoded[bool_columns].astype(int)

print("Boolean columns converted to integers:")
print(train_encoded.dtypes.value_counts())


x_train,x_test,y_train,y_test=train_test_split(train_encoded,y,test_size=0.2,random_state=42,stratify=y)
print(x_train.shape)
print(y_train.shape)
print(y)


print(x_train.info())
print(y_train.info())
print(x_test.info())
print(y_test.info())


from xgboost import XGBClassifier

param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear']
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'XGBoost': {
        'n_estimators': [30, 50],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
}

base_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, verbosity=0)
}

results = {}
best_models = {}
training_times = {}
search_results = {} 

for name, base_model in base_models.items():
    print(f"\n{'='*60}")
    print(f"Tuning and Training {name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    if name in param_grids:
        search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grids[name],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
        )
        
        # Fit and predict for all models
        search.fit(x_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(x_test)
        
        search_results[name] = search
        print(f"Best parameters for {name}: {search.best_params_}")
        print(f"Best CV score: {search.best_score_:.4f}")
        
    else:
        # For models without hyperparameter tuning
        best_model = base_model
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        print(f"No hyperparameter tuning for {name} (using default parameters)")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    training_time = time.time() - start_time
    training_times[name] = training_time
    
    # Store results
    results[name] = {
        'model': best_model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred,
        'training_time': training_time
    }
    
    best_models[name] = best_model
    
    # Print results
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'Precision': [results[m]['precision'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'F1-Score': [results[m]['f1'] for m in results],
    'Training Time (s)': [results[m]['training_time'] for m in results]
}).sort_values('Accuracy', ascending=False)

print(comparison_df.to_string(index=False))
print(f"\n{'='*80}")
print("BEST PARAMETERS FOR TUNED MODELS")
print(f"{'='*80}")
for name in param_grids.keys():
    if name in search_results:
        print(f"\n{name}:")
        print(f"Best parameters: {search_results[name].best_params_}")
        print(f"Best CV score: {search_results[name].best_score_:.4f}")

tree_model = ['Decision Tree']
for model_name in tree_model:
    if model_name in results and hasattr(results[model_name]['model'], 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        feature_importance = results[model_name]['model'].feature_importances_
        indices = np.argsort(feature_importance)[::-1][:10]
        
        plt.bar(range(len(indices)), feature_importance[indices])
        plt.title(f'Top 10 Features - {model_name}')
        plt.xticks(range(len(indices)), [train_encoded.columns[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Feature Importance')
        plt.tight_layout()
        plt.show()

best_model_name = comparison_df.iloc[0]['Model']
best_model_results = results[best_model_name]

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, best_model_results['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}\nAccuracy: {best_model_results["accuracy"]:.4f}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


print(test.info())


def balance_category(x):
    if x < -500:
        return "high_debt"
    elif x < 0:
        return "low_debt"
    elif x == 0:
        return "no_savings"
    else:
        return "saver"
test["balance_cat"] = test["balance"].apply(balance_category)
print(test["balance_cat"].unique())
test["previous_binary"] = (test["previous"] > 0).astype(int)
test["pdays_binary"] = (test["pdays"] > 0).astype(int)
print(test["pdays_binary"].value_counts())
minmaxscalers_columns=["day"]
standard_scaler_column=["age","balance","duration"]
log_transform_columns=["duration","campaign"]
numerical_to_drop=["previous","pdays"]


scaler_std = StandardScaler()
train[standard_scaler_column] = scaler_std.fit_transform(train[standard_scaler_column])
scaler_mm = MinMaxScaler()
train[minmaxscalers_columns] = scaler_mm.fit_transform(train[minmaxscalers_columns])


def log_transform(X):
    return np.log1p(X)
log_transformer=FunctionTransformer(log_transform)
test[log_transform_columns]=log_transformer.transform(test[log_transform_columns])


for col in standard_scaler_column + minmaxscalers_columns + log_transform_columns :
    plt.figure(figsize=(6,4))
    sns.histplot(test[col], bins=50, kde=True, color="salmon")
    plt.title(f"Transformed Distribution of {col}")
    plt.show()


# Separate the columns
one_hot_columns = ['month', 'marital', 'education',"job",'poutcome','contact']
label_encoder_columns = ['default', 'housing', 'loan', 'balance_cat']

# Create a copy to preserve original data if needed
test_encoded = test.copy()

# Apply One-Hot Encoding to specified columns
test_encoded = pd.get_dummies(test_encoded, columns=one_hot_columns, prefix=one_hot_columns)

# Apply Label Encoding to the remaining categorical columns
label_encoders = {}  # Store encoders for potential inverse transformation
for col in label_encoder_columns:
    le = LabelEncoder()
    test_encoded[col] = le.fit_transform(test_encoded[col].astype(str))
    label_encoders[col] = le  # Store the encoder

# Display the transformed dataframe info
print("Transformed DataFrame Info:")
print(test_encoded.info())
print(f"\nNew shape: {test_encoded.shape}")


bool_columns = test_encoded.select_dtypes(include=['bool']).columns
test_encoded[bool_columns] = test_encoded[bool_columns].astype(int)

print("Boolean columns converted to integers:")
print(test_encoded.dtypes.value_counts())


test_id=test_encoded["id"]
test=test_encoded.drop(["id","pdays","previous"],axis=1)


test.info()


# Get the best XGBoost model
best_xgboost_model = results['XGBoost']['model']

print("Best XGBoost Model Parameters:")
print(best_xgboost_model.get_params())

# Make predictions on the test data
test_predictions = best_xgboost_model.predict(test)

print(f"\nPredictions made on test data with shape: {test.shape}")
print(f"Number of predictions: {len(test_predictions)}")

# Get prediction probabilities (for confidence scores)
test_probabilities = best_xgboost_model.predict_proba(test)
print(f"Prediction probabilities shape: {test_probabilities.shape}")

# Display some sample predictions
print("\nSample predictions:")
for i in range(min(10, len(test_predictions))):
    print(f"Sample {i+1}: Prediction = {test_predictions[i]}, Probabilities = {test_probabilities[i]}")

# Get prediction statistics
unique, counts = np.unique(test_predictions, return_counts=True)
print(f"\nPrediction distribution: {dict(zip(unique, counts))}")

# Create submission DataFrame
test_res = pd.DataFrame({
    "id": test_id,  # Using your test_id variable
    "y": test_probabilities[:, 1]  # Probability of positive class
})

print(f"\nSubmission DataFrame shape: {test_res.shape}")
print(test_res.head())

# Save to CSV
test_res.to_csv('submission.csv', index=False)
print("Predictions saved to 'submission.csv'")

# Show detailed prediction distribution
print(f"\nDetailed prediction distribution:")
print(f"Class 0 predictions: {(test_predictions == 0).sum()}")
print(f"Class 1 predictions: {(test_predictions == 1).sum()}")
print(f"Percentage of positive predictions: {(test_predictions == 1).sum()/len(test_predictions)*100:.2f}%")


print(test_res.shape)


print(y.value_counts())

