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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



# Load the datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Display the first few rows of the data to understand its structure
train_data.head()




test_data.head()


# Check for missing values in both train and test datasets
print(train_data.isnull().sum())

# Check the distribution of the target variable
train_data['rainfall'].value_counts()



print(test_data.isnull().sum())


# Fill missing values with median (or another strategy like mean or mode)
train_data.fillna(train_data.median(), inplace=True)


test_data.fillna(test_data.median(), inplace=True)


# Select features and target variable
X = train_data.drop(['rainfall', 'id'], axis=1)
y = train_data['rainfall']
X_test = test_data.drop(['id'], axis=1)

# Split the train dataset into a training and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features for better performance (especially important for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)



# Initialize and train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict probabilities on validation data
y_val_pred = model.predict_proba(X_val)[:, 1]  # Probability for the positive class

# Evaluate the model performance using ROC-AUC
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f'ROC AUC Score on validation set: {roc_auc}')



# ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()



# Predict the probability of rainfall on the test dataset
y_test_pred = model.predict_proba(X_test_scaled)[:, 1]

# Prepare the submission dataframe
submission = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': y_test_pred
})

# Save the submission file
submission.to_csv('submission.csv', index=False)



# Importing necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

# Create a function to train and evaluate each model
def evaluate_models(models, X_train, y_train, X_val, y_val):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_pred = model.predict_proba(X_val)[:, 1]  # Probability for the positive class
        roc_auc = roc_auc_score(y_val, y_val_pred)
        results[name] = roc_auc
    return results

# Evaluate models
results = evaluate_models(models, X_train, y_train, X_val, y_val)

# Print the ROC-AUC score for each model
for model_name, score in results.items():
    print(f'{model_name}: ROC AUC Score = {score:.4f}')



# Importing necessary libraries for additional models
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import roc_auc_score

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'CatBoost': cb.CatBoostClassifier(iterations=100, depth=10, learning_rate=0.1, random_state=42, verbose=0),
    'Ridge Classifier': RidgeClassifier(),
    'Stochastic Gradient Descent': SGDClassifier(),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'Bagging': BaggingClassifier(n_estimators=100, random_state=42),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    'Perceptron': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    'MLP Classifier': MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42),
}

# Create a function to train and evaluate each model, handling models that don't support predict_proba
def evaluate_models(models, X_train, y_train, X_val, y_val):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):  # Models like RandomForest, XGBoost, etc.
            y_val_pred = model.predict_proba(X_val)[:, 1]  # Probability for the positive class
        elif hasattr(model, "decision_function"):  # For models like RidgeClassifier and SVM with linear loss
            decision_values = model.decision_function(X_val)
            # Apply sigmoid function to decision values to get probabilities
            y_val_pred = 1 / (1 + np.exp(-decision_values))  # Sigmoid function
        else:
            raise AttributeError(f"The model {name} doesn't support probabilistic predictions.")

        roc_auc = roc_auc_score(y_val, y_val_pred)
        results[name] = roc_auc
    return results

# Evaluate models
results = evaluate_models(models, X_train, y_train, X_val, y_val)

# Print the ROC-AUC score for each model
for model_name, score in results.items():
    print(f'{model_name}: ROC AUC Score = {score:.4f}')



# Visualizing the comparison of models using a bar plot
model_names = list(results.keys())
roc_auc_scores = list(results.values())

# Creating a DataFrame for better plotting
model_comparison_df = pd.DataFrame({
    'Model': model_names,
    'ROC AUC Score': roc_auc_scores
})

# Sort by ROC AUC score
model_comparison_df = model_comparison_df.sort_values(by='ROC AUC Score', ascending=False)

# Plotting the ROC AUC scores
plt.figure(figsize=(10, 6))
sns.barplot(x='ROC AUC Score', y='Model', data=model_comparison_df, palette='viridis')
plt.title('Model Comparison - ROC AUC Scores')
plt.xlabel('ROC AUC Score')
plt.ylabel('Model')
plt.show()


# Initialize and train a Random Forest model
model =LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict probabilities on validation data
y_val_pred = model.predict_proba(X_val)[:, 1]  # Probability for the positive class

# Evaluate the model performance using ROC-AUC
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f'ROC AUC Score on validation set: {roc_auc}')



# Predict the probability of rainfall on the test dataset
y_test_pred = model.predict_proba(X_test_scaled)[:, 1]

# Prepare the submission dataframe
submission = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': y_test_pred
})

# Save the submission file
submission.to_csv('submission.csv', index=False)




!pip install pandas-profiling



import pandas as pd
from ydata_profiling import ProfileReport  
 
profile = ProfileReport(train_data, explorative=True)  
profile.to_notebook_iframe() 


# Import necessary libraries
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

# Load the datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Separate features and target variable
X_train = train_data.drop(columns=['id', 'rainfall']) 
y_train = train_data['rainfall']

# Split the dataset into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize LazyClassifier
clf = LazyClassifier()

# Fit and predict on the training data and validate on the validation set
models = clf.fit(X_train_split, X_val_split, y_train_split, y_val_split)

# Display results
print(models[0]) 




