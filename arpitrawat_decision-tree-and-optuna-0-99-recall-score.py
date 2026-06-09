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
import warnings
warnings.filterwarnings("ignore")



train=pd.read_csv('/kaggle/input/product-defect-detection/train_data.csv')
test=pd.read_csv('/kaggle/input/product-defect-detection/test_data.csv')


train.shape, test.shape


train.sample(5)


train.info()


X=train.drop(columns=['ProductID','Defective'])
y=train['Defective']


num_cols=X.select_dtypes(include='float64').columns.tolist()


sns.countplot(x=train['ProductionLine'],hue=train['Defective'])


# Define number of rows and columns for the grid layout
num_plots = len(num_cols)  # Number of numerical columns
rows = (num_plots // 3) + (num_plots % 3 > 0)  # Arrange in a grid (3 columns per row)

# Create subplots with a smaller figure size
fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))  # Adjust figure size for compact layout

# Flatten axes array for easy iteration (if there's only one row, it remains 1D)
axes = axes.flatten()

# Loop through numerical columns and plot histograms
for i, col in enumerate(num_cols):
    sns.histplot(x=train[col], kde=True,hue=train['Defective'], ax=axes[i])
    axes[i].set_title(col, fontsize=10)  # Set title with smaller font size
    axes[i].tick_params(axis='both', labelsize=8)  # Reduce tick label size

# Remove empty subplots (if any)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


sns.scatterplot(x=train['Temperature'], y=train['Pressure'],hue=train['Defective'])


correlation_matrix = X.drop(columns='ProductionLine').corr()  

# heatmap using Seaborn
plt.figure(figsize=(10, 8))  
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Title and other customizations
plt.title('Correlation Heatmap', fontsize=16)
plt.show()


import plotly.express as px

# Create the 3D scatter plot
fig = px.scatter_3d(
    train, 
    x='Sensor1Reading', 
    y='Sensor2Reading', 
    z='Sensor3Reading', 
    color='Defective'
)

# Update layout to add title and labels
fig.update_layout(
    title="3D Scatter Plot of Sensor Readings",  # Title goes here
    scene=dict(
        xaxis_title='Sensor 1 Reading',
        yaxis_title='Sensor 2 Reading',
        zaxis_title='Sensor 3 Reading'
    )
)

# Show the plot
fig.show()



train.info()


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


preprocessing= ColumnTransformer(transformers=[('imputer',SimpleImputer(strategy='mean'),num_cols),('encoder',OneHotEncoder(),['ProductionLine'])]
                                 ,remainder='passthrough')



model=DecisionTreeClassifier()
pipeline=Pipeline(steps=[('preprocessing',preprocessing),('model',model)])
# Define the custom scorers for cross-validation
scorers = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

# Perform cross-validation for each metric
for metric_name, scorer in scorers.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scorer)
    print(f"Cross-validation {metric_name} scores: {scores}")
    print(f"Mean {metric_name} score: {scores.mean():.4f}\n")

# Fit the pipeline and predict on test set
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Compute final test metrics
test_metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}

# Print test set metrics
print("Test Set Performance:")
for metric, value in test_metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")


import optuna

# Custom scorer to maximize recall while considering F1 score
def custom_scorer(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return (recall + f1) / 2  # Balances recall and F1

# Define the Optuna objective function
def objective(trial):
    # Expanded hyperparameter search space
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    splitter = trial.suggest_categorical('splitter', ['best', 'random'])
    max_depth = trial.suggest_int('max_depth', 3, 50, step=3)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 0.1)
    ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.05)
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}])

    # Define model with sampled hyperparameters
    model = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        min_impurity_decrease=min_impurity_decrease,
        ccp_alpha=ccp_alpha,
        class_weight=class_weight,
        random_state=42
    )

    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessing', preprocessing), ('model', model)])
    
    # Perform cross-validation, using recall + F1 score for better performance
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=make_scorer(custom_scorer), n_jobs=-1)
    return scores.mean()  # Optimize for balanced recall + F1

# Run Optuna optimization
study = optuna.create_study(direction='maximize')  # Maximize recall + F1
study.optimize(objective, n_trials=200)  # Increase trials if needed

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)

# Train the best model with optimized parameters
best_params = study.best_params
best_model = DecisionTreeClassifier(**best_params, random_state=42)
pipeline = Pipeline(steps=[('preprocessing', preprocessing), ('model', best_model)])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Compute final test metrics
test_metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1': f1_score(y_test, y_pred)
}

# Print test set metrics
print("\nTest Set Performance (Optimized for Recall & F1):")
for metric, value in test_metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")


test.info()


test_ids=test['ProductID']


test.drop(columns=['ProductID','Defective'],inplace=True)


test_prediction=pipeline.predict(test)


output = pd.DataFrame({
    'ProductID': test_ids,  
    'Defective': test_prediction     
})

# Save to CSV
output.to_csv('submission.csv', index=False)




