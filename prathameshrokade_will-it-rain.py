# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


# Load datasets
train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


from ydata_profiling import ProfileReport
ProfileReport(train_df)


# Fill missing values in test set
test_df["winddirection"].fillna(test_df["winddirection"].median(), inplace=True)


# Select features (removing highly correlated ones)
selected_features = ["pressure", "humidity", "cloud", "sunshine", "winddirection", "windspeed"]
X = train_df[selected_features]
y = train_df["rainfall"]


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Train-test split (80-20 stratified)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_df[selected_features])


from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=543, max_depth=3, learning_rate=0.005014920478990856, subsample=0.7255624401364013, colsample_bytree=0.829284796162759, objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

y_val_probs_xgb = xgb_model.predict_proba(X_val_scaled)[:, 1]
auc_score_xgb = roc_auc_score(y_val, y_val_probs_xgb)
print(f"XGBoost Validation AUC-ROC: {auc_score_xgb:.4f}")


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

grid_search = GridSearchCV(XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False, random_state=42), 
                           param_grid, 
                           scoring='roc_auc', 
                           cv=3)

grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
y_val_probs_best = best_model.predict_proba(X_val_scaled)[:, 1]
auc_score_best = roc_auc_score(y_val, y_val_probs_best)
print(f"Best XGBoost Validation AUC-ROC: {auc_score_best:.4f}")


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Define ANN model
ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Train the model
ann_model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=50, batch_size=32)

# Evaluate on validation set
y_val_probs_ann = ann_model.predict(X_val_scaled).flatten()
auc_score_ann = roc_auc_score(y_val, y_val_probs_ann)
print(f"ANN Validation AUC-ROC: {auc_score_ann:.4f}")



from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from kerastuner import HyperModel, RandomSearch

class ANNHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Dense(hp.Int('units_1', min_value=32, max_value=256, step=32), activation='relu', input_shape=(X_train_scaled.shape[1],)))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)))
        
        model.add(Dense(hp.Int('units_2', min_value=32, max_value=256, step=32), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_2', 0.2, 0.5, step=0.1)))
        
        model.add(Dense(1, activation='sigmoid'))
        
        # Use 'AUC' as the metric
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
        return model

tuner = RandomSearch(
    ANNHyperModel(),
    objective='val_AUC',  # Ensure this matches the metric name
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='ann_tuning'
)

# Start the search
tuner.search(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_val_scaled, y_val))

# Get the best model
best_model1 = tuner.get_best_models(num_models=1)[0]
y_val_probs_best = best_model1.predict(X_val_scaled).flatten()
auc_score_best = roc_auc_score(y_val, y_val_probs_best)
print(f"Best ANN Validation AUC-ROC: {auc_score_best:.4f}")


# Get predictions from both models
y_pred_xgb = best_model.predict_proba(X_test_scaled)[:, 1]
y_pred_ann = best_model1.predict(X_test_scaled).flatten()

# Simple averaging ensemble
y_pred_ensemble = (y_pred_xgb + y_pred_ann) / 2


y_pred_ensemble.shape


submission = pd.DataFrame({
    "id": test_df["id"],  
    "rainfall": y_pred_ensemble  
})

submission.to_csv("submission.csv", index=False)
print("Submission file created successfully!")




