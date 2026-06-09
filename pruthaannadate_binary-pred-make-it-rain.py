# Data Handling
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Machine Learning & Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

#Neural Networks
from tensorflow import keras
from tensorflow.keras import layers


#Reading our data here
# Load datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").set_index('id')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')
df_subm = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

# Display basic dataset info
print(df_train.info())
print(df_test.info())

# Check for missing values
print(df_train.isnull().sum())
print(df_test.isnull().sum())



# Handling missing values
imputer = SimpleImputer(strategy='mean')

# Drop 'rainfall' from df_train before imputation
df_train_features = df_train.drop(columns=['rainfall'])  # Features only, no target

# Fit imputer on training data features
df_train_imputed = pd.DataFrame(imputer.fit_transform(df_train_features), columns=df_train_features.columns, index=df_train.index)
df_test_imputed = pd.DataFrame(imputer.transform(df_test), columns=df_test.columns, index=df_test.index)

# Feature selection: Define X (features) and y (target)
X = df_train_imputed  # Use imputed training features
y = df_train['rainfall']  # Target remains unchanged

# Splitting data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalization using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(df_test_imputed)




# Initialize the model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Validate the model
val_preds = rf_model.predict_proba(X_val_scaled)[:, 1]
auc_score = roc_auc_score(y_val, val_preds)
print(f'Validation AUC Score (Random Forest): {auc_score:.4f}')

# Generate test predictions
test_preds = rf_model.predict_proba(X_test_scaled)[:, 1]






# Initialize and train the model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Validate model
val_preds = lr_model.predict_proba(X_val_scaled)[:, 1]
auc_score = roc_auc_score(y_val, val_preds)
print(f'Validation AUC Score (Logistic Regression): {auc_score:.4f}')

# Test predictions
test_preds = lr_model.predict_proba(X_test_scaled)[:, 1]

# Prepare submission
submission_lr = pd.DataFrame({'id': df_test.index, 'rainfall': test_preds})
submission_lr.to_csv('submission_lr.csv', index=False)

print("BEST PERFORMING MODEL_Submission file saved as 'submission_lr.csv'")



# Initialize and train the model
xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Validate model
val_preds = xgb_model.predict_proba(X_val_scaled)[:, 1]
auc_score = roc_auc_score(y_val, val_preds)
print(f'Validation AUC Score (XGBoost): {auc_score:.4f}')

# Test predictions
test_preds = xgb_model.predict_proba(X_test_scaled)[:, 1]




# Define Neural Network Architecture
nn_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary Classification
])

# Compile the model
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Train the model
nn_model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=20, batch_size=32)

# Validate model
val_preds = nn_model.predict(X_val_scaled).flatten()
auc_score = roc_auc_score(y_val, val_preds)
print(f'Validation AUC Score (Neural Network): {auc_score:.4f}')

# Test predictions
test_preds = nn_model.predict(X_test_scaled).flatten()



# Plot ROC Curves
plt.figure(figsize=(10, 6))

# Function to plot ROC Curve for a model
def plot_roc_curve(y_true, y_pred, model_name, color):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')

# Plot ROC Curves for all models
plot_roc_curve(y_val, rf_model.predict_proba(X_val_scaled)[:, 1], "Random Forest", 'green')
plot_roc_curve(y_val, lr_model.predict_proba(X_val_scaled)[:, 1], "Logistic Regression", 'blue')
plot_roc_curve(y_val, xgb_model.predict_proba(X_val_scaled)[:, 1], "XGBoost", 'red')
plot_roc_curve(y_val, nn_model.predict(X_val_scaled).flatten(), "Neural Network", 'purple')

# Plot formatting
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()



# Create a DataFrame with all predictions
predictions_df = pd.DataFrame({
    'Random Forest': rf_model.predict_proba(X_val_scaled)[:, 1],
    'Logistic Regression': lr_model.predict_proba(X_val_scaled)[:, 1],
    'XGBoost': xgb_model.predict_proba(X_val_scaled)[:, 1],
    'Neural Network': nn_model.predict(X_val_scaled).flatten()
})

# Handle infinity and NaN values
predictions_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Convert inf to NaN
predictions_df.fillna(predictions_df.mean(), inplace=True)  # Fill NaN with column mean
predictions_df = predictions_df.clip(0, 1)  # Ensure values stay between 0 and 1

# Plot distributions
plt.figure(figsize=(10, 6))
sns.kdeplot(predictions_df['Random Forest'], label="Random Forest", color='green', fill=True, alpha=0.3)
sns.kdeplot(predictions_df['Logistic Regression'], label="Logistic Regression", color='blue', fill=True, alpha=0.3)
sns.kdeplot(predictions_df['XGBoost'], label="XGBoost", color='red', fill=True, alpha=0.3)
sns.kdeplot(predictions_df['Neural Network'], label="Neural Network", color='purple', fill=True, alpha=0.3)

plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Prediction Probability Distribution')
plt.legend()
plt.show()


