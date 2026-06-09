import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay



# read the data by os
train_path = os.path.join(os.getcwd(),".." , "dataset","train.csv")
train_df = pd.read_csv(train_path)


# print the first 5 rows of the data
train_df.head()


# print the last 5 rows of the data
train_df.tail()


# read the test data
test_path = os.path.join(os.getcwd(),".." , "dataset","test.csv")
test_df = pd.read_csv(test_path)


# print the shape of the data
print("Train shape: ", train_df.shape)
print("Test shape: ", test_df.shape)


train_df.describe()


train_df.info()


train_df.hist(bins=50, figsize=(20,15))


# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(train_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df[['maxtemp', 'temparature', 'mintemp', 'dewpoint']], palette="Set2")
plt.title("Boxplot of Temperature-related Features")
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(x=train_df['day'], y=train_df['temparature'], label="Temperature", marker="o")
sns.lineplot(x=train_df['day'], y=train_df['humidity'], label="Humidity", marker="s")
plt.xlabel("Day")
plt.ylabel("Value")
plt.title("Temperature and Humidity Trends Over Time")
plt.legend()
plt.show()


# drop the id column in training and test data
train_df.drop(columns=['id'], inplace=True)
test_df.drop(columns=['id'], inplace=True)


# split the data into features and target
X = train_df.drop(columns=['rainfall'])
y = train_df['rainfall']


# Create a validation set (e.g., 80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)


# define the test data
X_test = test_df
print(X_test.shape)


num_columns = X_train.columns

# Define the numerical pipeline
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Create a ColumnTransformer that only applies the numerical pipeline
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_columns)
])

# Fit the preprocessor on the training data and transform the training, validation, and test sets
X_train_final = preprocessor.fit_transform(X_train)
X_val_final = preprocessor.transform(X_val)
X_test_final = preprocessor.transform(X_test)

print("Transformed training shape:", X_train_final.shape)
print("Transformed validation shape:", X_val_final.shape)
print("Transformed test shape:", X_test_final.shape)


ARTIFACT_FOLDER_PATH = os.path.join(os.getcwd(), "..", "artifacts")
os.makedirs(ARTIFACT_FOLDER_PATH, exist_ok=True)


joblib.dump(preprocessor, os.path.join(ARTIFACT_FOLDER_PATH, "preprocessor.pkl"))


rf_model = RandomForestClassifier(random_state=42,n_jobs=-1)

# Train the model
rf_model.fit(X_train_final, y_train)

# Predict on the validation set
y_val_pred_rf = rf_model.predict(X_val_final)

# Evaluate the model
accuracy_rf = accuracy_score(y_val, y_val_pred_rf)
print("Random Forest Validation Accuracy:", accuracy_rf)
print("Classification Report:")
print(classification_report(y_val, y_val_pred_rf))



cm = confusion_matrix(y_val, y_val_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("XGBoost Confusion Matrix")
plt.show()


# Initialize the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
lr_model.fit(X_train_final, y_train)

# Predict on the validation set
y_val_pred_lr = lr_model.predict(X_val_final)

# Evaluate the model
accuracy_lr = accuracy_score(y_val, y_val_pred_lr)
print("Logistic Regression Validation Accuracy:", accuracy_lr)
print("Classification Report:")
print(classification_report(y_val, y_val_pred_lr))



cm = confusion_matrix(y_val, y_val_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("XGBoost Confusion Matrix")
plt.show()


xgb_model = XGBClassifier(random_state=42, max_depth=3, n_estimators=100, learning_rate=0.1, n_jobs=-1)

# Train the model on the preprocessed training data
xgb_model.fit(X_train_final, y_train)

# Predict on the validation set
y_val_pred_xgb = xgb_model.predict(X_val_final)

# Evaluate the model
accuracy_xgb = accuracy_score(y_val, y_val_pred_xgb)
print("XGBoost Validation Accuracy:", accuracy_xgb)
print("Classification Report:")
print(classification_report(y_val, y_val_pred_xgb))



# Draw the confusion matrix
cm = confusion_matrix(y_val, y_val_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("XGBoost Confusion Matrix")
plt.show()


# Initialize the K-Nearest Neighbors model
knn_model = KNeighborsClassifier()

# Train the model
knn_model.fit(X_train_final, y_train)

# Predict on the validation set
y_val_pred_knn = knn_model.predict(X_val_final)

# Evaluate the model
accuracy_knn = accuracy_score(y_val, y_val_pred_knn)
print("K-Nearest Neighbors Validation Accuracy:", accuracy_knn)
print("Classification Report:")
print(classification_report(y_val, y_val_pred_knn))




# Draw the confusion matrix
cm = confusion_matrix(y_val, y_val_pred_knn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("XGBoost Confusion Matrix")
plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# --- Compute ROC for XGBoost ---
y_val_prob_xgb = xgb_model.predict_proba(X_val_final)[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_val_prob_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# --- Compute ROC for Random Forest ---
y_val_prob_rf = rf_model.predict_proba(X_val_final)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_val, y_val_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# --- Compute ROC for Logistic Regression ---
y_val_prob_lr = lr_model.predict_proba(X_val_final)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_val, y_val_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# --- Compute ROC for K-Nearest Neighbors ---
y_val_prob_knn = knn_model.predict_proba(X_val_final)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_val, y_val_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Plot ROC curves for all models
plt.figure(figsize=(10, 8))
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, 
         label=f'XGBoost ROC (AUC = {roc_auc_xgb:.2f})')
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, 
         label=f'Random Forest ROC (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_lr, tpr_lr, color='red', lw=2, 
         label=f'Logistic Regression ROC (AUC = {roc_auc_lr:.2f})')
plt.plot(fpr_knn, tpr_knn, color='purple', lw=2, 
         label=f'KNN ROC (AUC = {roc_auc_knn:.2f})')

# Diagonal line for random chance
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison for Models')
plt.legend(loc="lower right")
plt.show()



# Ensure the test DataFrame has exactly 730 rows
if test_df.shape[0] == 731:
    test_df = test_df.iloc[:730]
    print("Adjusted test DataFrame to 730 rows.")

# Predict on the preprocessed test data
y_test_pred = xgb_model.predict(X_test_final)

# If the test data has an 'id' column, use it. Otherwise, create an 'id' column starting from 2190.
if "id" in test_df.columns:
    submission = pd.DataFrame({
        "id": test_df["id"],
        "rainfall": y_test_pred
    })
else:
    submission = pd.DataFrame({
        "id": range(2190, 2190 + test_df.shape[0]),
        "rainfall": y_test_pred
    })

# Print the submission DataFrame (first 20 rows)
print("Submission Preview:")
print(submission.head(20))

# Save the submission file to CSV without an index
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")


