# Step 1: Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_squared_log_error, confusion_matrix



train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')



train.head()


test.head()


train.info()


test.info()


train.describe()


test.describe()


plt.plot(train['Personality'])
plt.plot(train['Friends_circle_size'])

plt.show()


plt.hist(train['Personality'])
#plt.plot(train['Going_outside'])

plt.show() 


train.isnull()


le_target = LabelEncoder()
train['Personality'] = le_target.fit_transform(train['Personality'])  # Introvert=0, Extrovert=1

X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']
X_test = test.drop(['id'], axis=1)



cat_cols = X.select_dtypes(include='object').columns

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    le_dict[col] = le



imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)



import missingno as msno


msno.bar(train)


msno.matrix(train)


msno.dendrogram(train)


train.isnull().sum()


X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)



models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Ridge Classifier": RidgeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}



for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)

    # Convert predictions to float for RMSLE (needed for comparison)
    rmsle = np.sqrt(mean_squared_log_error(y_val, np.clip(y_pred.astype(float), 0, 1)))

    print(f"=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"RMSLE: {rmsle:.4f}")
    print(confusion_matrix(y_val, y_pred))
    print("\n")



# Confusion matrix for best model (e.g., XGBoost)
best_model = models["XGBoost"]
y_pred_best = best_model.predict(X_val)

sns.heatmap(confusion_matrix(y_val, y_pred_best), annot=True, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



final_preds = best_model.predict(X_test_scaled)
final_labels = le_target.inverse_transform(final_preds)

submission = pd.DataFrame({
    'id': test['id'],
    'Personality': final_labels
})

submission.to_csv('submission.csv', index=False)
submission.head()



importances = best_model.feature_importances_
features = X.columns

feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(data=feat_df.head(15), x='Importance', y='Feature', palette='viridis')
plt.title("Top Feature Importances - XGBoost")
plt.show()





