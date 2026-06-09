#Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("Libraries imported successfully.\n")


#Load train.csv and verify
train_df = pd.read_csv("/kaggle/input/playground-series-s4e3/train.csv")
print("train.csv loaded successfully.")
print(f"Train data shape: {train_df.shape}")
print("Sample data:\n", train_df.head(), "\n")



#Define target columns and split features/targets
target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
X = train_df.drop(columns=['id'] + target_cols)
y = train_df[target_cols]
print(f"Features and targets split.")
print(f"Features shape: {X.shape}, Targets shape: {y.shape}\n")


#Split train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Train/validation split done.")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples\n")


#Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
print("Feature scaling completed.\n")



#Define models for multilabel classification
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier()
    }
print("Models defined.\n")


#Train, predict, and evaluate models
results = {}
f1_scores = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    clf = OneVsRestClassifier(model)
    clf.fit(X_train_scaled, y_train)
    
    print(f"Predicting validation set with {model_name}...")
    y_pred = clf.predict(X_val_scaled)
    y_prob = clf.predict_proba(X_val_scaled) if hasattr(clf, "predict_proba") else None
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
    f1_scores[model_name] = f1
    auc = roc_auc_score(y_val, y_prob, average='macro') if y_prob is not None else None
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC': auc
    }
    
    print(f"{model_name} Results:")
    print(f" Accuracy: {accuracy:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall: {recall:.4f}")
    print(f" F1-Score: {f1:.4f}")
    print(f" AUC: {auc if auc else 'N/A'}\n")


#Summary of model performance
results_df = pd.DataFrame(results).T
print("Summary of Model Performance:\n")
print(results_df, "\n")


# Determine best model by highest F1-Score
results_df = pd.DataFrame(results).T
best_model_name = max(f1_scores, key=f1_scores.get)
print("Summary of Model Performance:\n")
print(results_df, "\n")
print(f"✅ Best Model based on F1-Score: {best_model_name} (F1: {f1_scores[best_model_name]:.4f})\n")


# Load test.csv and prepare for prediction
test_df = pd.read_csv("/kaggle/input/playground-series-s4e3/test.csv")
print("test.csv loaded successfully.")
print(f"Test data shape: {test_df.shape}")
print("Sample test data:\n", test_df.head(), "\n")

# Preprocess test features
X_test = test_df.drop(columns=['id'])
X_test_scaled = scaler.transform(X_test)
print("Test data features scaled.\n")

# Predict on test data using the best model (e.g., Logistic Regression)
best_model_name = 'Logistic Regression'  # choose based on performance above
best_clf = OneVsRestClassifier(models[best_model_name])
best_clf.fit(X_train_scaled, y_train)  # retrain on full training set

print(f"Predicting test set with {best_model_name}...")
test_preds = best_clf.predict(X_test_scaled)

# Prepare submission dataframe and save
submission_df = pd.DataFrame(test_preds, columns=target_cols)
submission_df.insert(0, 'id', test_df['id'])
submission_df.to_csv("/kaggle/working/test_predictions.csv", index=False)
print("Test predictions saved to 'test_predictions.csv'.")

