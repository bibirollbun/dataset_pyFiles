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


import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical



train_df = pd.read_csv('/kaggle/input/exploring-machine-learning-with-mnist/mnist_train.csv')
test_df = pd.read_csv('/kaggle/input/exploring-machine-learning-with-mnist/mnist_test.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)



X_train = train_df.drop('label', axis=1).values.astype('float32') / 255.0
y_train = train_df['label'].values

X_test = test_df.drop('label', axis=1).values.astype('float32') / 255.0
y_test = test_df['label'].values



def visualize_samples(X, y_true, y_pred=None, title="Sample Images"):
    plt.figure(figsize=(12, 8))
    indices = np.random.choice(len(X), 6, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(2, 3, i + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap='gray')
        label = f"Actual: {y_true[idx]}"
        if y_pred is not None:
            label += f"\nPred: {y_pred[idx]}"
        plt.title(label)
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

def evaluate_model(y_true, y_pred, model_name):
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred, model_name)
    visualize_samples(X_test, y_true, y_pred, title=f"{model_name} Predictions")
    


def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        results[name] = {'accuracy': acc, 'predictions': y_pred}
    return results



models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial', n_jobs=-1),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(n_jobs=-1),
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(),
    'Gaussian NB': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier()
}



results = train_and_evaluate(models, X_train, y_train, X_test, y_test)




model_scores = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results]
}).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

print("\nModel Performance Summary:")
print(model_scores)



PARAMS = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    },
    'SVM': {
        'model': SVC(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    },
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2, 0.3]
        }
    }
}

tuned_results = []

for name, config in PARAMS.items():
    print(f"\nTuning hyperparameters for {name}...")
    grid = GridSearchCV(config['model'], config['params'], cv=3, n_jobs=-1, scoring='accuracy', verbose=1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_accuracy = accuracy_score(y_test, best_model.predict(X_test))
    print(f"Best {name} Accuracy: {best_accuracy:.4f}")
    tuned_results.append({
        'Model': name,
        'Best Params': grid.best_params_,
        'Accuracy': best_accuracy
    })




tuned_df = pd.DataFrame(tuned_results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

print("\nTuned Model Performance Summary:")
print(tuned_df)



# Find the best model from the tuned results
best_model_entry = max(tuned_results, key=lambda x: x['Accuracy'])
best_model_name = best_model_entry['Model']
best_model_params = best_model_entry['Best Params']

# Recreate and train the best model
best_model = PARAMS[best_model_name]['model'].set_params(**best_model_params)
best_model.fit(X_train, y_train)

# Save to disk
joblib.dump(best_model, f"{best_model_name.replace(' ', '_').lower()}_best_model.pkl")
print(f"Saved best model: {best_model_name} with accuracy {best_model_entry['Accuracy']:.4f}")



# Load from disk
loaded_model = joblib.load(f"{best_model_name.replace(' ', '_').lower()}_best_model.pkl")

print(f"Loaded model: {best_model_name}")



selected_model = best_model_name
evaluate_model(y_test, results[selected_model]['predictions'], selected_model)



submission = pd.DataFrame({
    'ID': test_df.index + 1,
    'Label': results[selected_model]['predictions']
})
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission file 'submission.csv' created using model: {selected_model}")


