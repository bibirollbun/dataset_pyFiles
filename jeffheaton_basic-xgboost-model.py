import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Load the dataset
train_file_path = "/kaggle/input/app-of-gen-ai-deep-learning-wustl-spring-2025/train.csv"
test_file_path = "/kaggle/input/app-of-gen-ai-deep-learning-wustl-spring-2025/test.csv"

df = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)

# Drop non-numeric columns except for the target
df_numeric = df.select_dtypes(include=['number']).copy()

# Handle categorical columns by encoding them
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col].astype(str))
    
    if col in df_test.columns:
        df_test[col] = df_test[col].astype(str)
        df_test[col] = df_test[col].apply(lambda x: x if x in label_encoders[col].classes_ else 'unknown')
        label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
        df_test[col] = label_encoders[col].transform(df_test[col])

# Define features and target
X = df.drop(columns=['id', 'performance_score'])  # Remove ID and target column
y = df['performance_score']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model with limited complexity for efficiency
model = XGBRegressor(objective='reg:squarederror', n_estimators=25, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Compute R-squared
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.4f}")


# Feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in XGBoost Model")
plt.gca().invert_yaxis()
plt.show()



# Prepare Kaggle submission
test_ids = df_test['id']
df_test = df_test.drop(columns=['id'])
submission_preds = model.predict(df_test)

submission = pd.DataFrame({'id': test_ids, 'performance_score': submission_preds})
submission.to_csv("submission.csv", index=False)
print("Kaggle submission file 'submission.csv' created successfully.")

