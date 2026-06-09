!pip install -q joblib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import os
os.makedirs("model", exist_ok=True)
os.makedirs("submissions", exist_ok=True)


train = pd.read_csv('/kaggle/input/train-and-test-data/train.csv')
test = pd.read_csv('/kaggle/input/train-and-test-data/test.csv')

print("Train Data:", train.shape)
print("Test Data:", test.shape)
print("\nFirst 5 rows of training data:")
print(train.head())
print("\nMissing values in each column:")
print(train.isnull().sum())









sns.countplot(x='Exited', data=train)
plt.title("Class Balance (0 = Stayed, 1 = Left)")
plt.show()



le = LabelEncoder()
for col in ['Gender', 'Geography']:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])



drop_cols = ['id', 'CustomerId', 'Surname']
X_train_full = train.drop(drop_cols + ['Exited'], axis=1)
y_train_full = train['Exited']
X_test = test.drop(drop_cols, axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'model/scaler.pkl')




X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train_full, test_size=0.2, random_state=42)



def train_and_evaluate(model, model_name):
    print(f"\nTraining model: {model_name}")
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_val_pred

    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_proba)

    print(f"{model_name} - Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | AUC: {auc:.4f}")

    model_path = f"model/{model_name.replace(' ', '_')}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


    test_preds = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test_scaled)

    submission = pd.DataFrame({
        'id': test['id'],
        'Exited': test_preds
    })
    csv_path = f"submissions/{model_name.replace(' ', '_')}_submission.csv"
    submission.to_csv(csv_path, index=False)
    print(f"Submission file saved to {csv_path}")

    return {'model': model_name, 'accuracy': acc, 'f1': f1, 'auc': auc}



models = [
    (RandomForestClassifier(n_estimators=100, random_state=10), "Random Forest"),
    (GradientBoostingClassifier(n_estimators=100, random_state=10), "Gradient Boosting"),
    (SVC(probability=True), "Support Vector Classifier"),
    (LogisticRegression(max_iter=1000), "Logistic Regression")
]

results = []
for model, name in models:
    results.append(train_and_evaluate(model, name))




results_df = pd.DataFrame(results)

print("\n" + "#"*100)
print("\nModel Evaluation Results:")
print(results_df)

best_model = results_df.sort_values(by='auc', ascending=False).iloc[0]
print("\n" + "#"*100)
print("\nBest model based on AUC:")
print(best_model)




model_map = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=10),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=10),
    "Support Vector Classifier": SVC(probability=True)
}

print("\n" + "#"*100)
print(f"\nCross-validating best model: {best_model['model']}")

best_model_instance = model_map[best_model['model']]
cv_scores = cross_val_score(best_model_instance, X_train_scaled, y_train_full, cv=5, scoring='roc_auc')

print("\n" + "#"*100)
print("Cross-validation AUC scores:", cv_scores)
print("\nAverage AUC:", cv_scores.mean())





