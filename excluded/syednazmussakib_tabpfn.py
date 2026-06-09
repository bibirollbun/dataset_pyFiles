import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Remove target from features
X = train.drop(['rainfall'], axis=1) 
y = train['rainfall']


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train TabPFN model with multiple evaluation runs
N_EVALUATIONS = 5
test_predictions = []
test_probas = []


for i in range(N_EVALUATIONS):
    # Train model with different seeds
    tabpfn = TabPFNClassifier(device='cpu')
    tabpfn.fit(X_train, y_train)
    
    # Get predictions and probabilities
    test_pred = tabpfn.predict(X_test)
    test_proba = tabpfn.predict_proba(X_test)
    
    test_predictions.append(test_pred)
    test_probas.append(test_proba)

# Ensemble predictions through majority voting
final_predictions = np.array([1 if sum(pred) > N_EVALUATIONS/2 else 0 for pred in zip(*test_predictions)])


# Average probabilities
final_probas = np.mean(test_probas, axis=0)


# Print metrics
print(f"Accuracy: {accuracy_score(y_test, final_predictions):.4f}")
print(f"Precision: {precision_score(y_test, final_predictions):.4f}")
print(f"Recall: {recall_score(y_test, final_predictions):.4f}")
print(f"F1-Score: {f1_score(y_test, final_predictions):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, final_probas[:,1]):.4f}")

