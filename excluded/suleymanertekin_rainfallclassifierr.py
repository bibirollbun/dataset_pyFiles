import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer


train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


X = train_data.drop(['id', 'rainfall'], axis=1)
y = train_data['rainfall']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
test_features_imputed = imputer.transform(test_data.drop('id', axis=1))


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
test_features_scaled = scaler.transform(test_features_imputed)


rf_model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)


best_rf_model = grid_search.best_estimator_


y_val_pred = best_rf_model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_val, y_val_pred))


test_predictions = best_rf_model.predict(test_features_scaled)


submission = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': test_predictions
})
submission.to_csv('submission.csv', index=False)

