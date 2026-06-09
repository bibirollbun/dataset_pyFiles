import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import ( # Changed imports for classification metrics
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, precision_score, recall_score, f1_score
)


df = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/heart_train.csv")


df.info()


df.head()


df.describe()


# target variable
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
# classify the variables between numerical and categorical
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns


numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first') # drop='first' can help reduce multicollinearity

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep any columns not specified (though we specified all here)
)


# Chain the preprocessor and the Logistic Regression model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # --- Changed to Logistic Regression ---
    # Increased max_iter for convergence, added random_state for reproducibility
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')) 
    # Common solvers: 'liblinear' (good for smaller datasets), 'lbfgs', 'saga' (good for larger datasets)
])



y.value_counts(normalize=True)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratify=y is good practice for classification


y_train.value_counts(normalize=True)


y_val.value_counts(normalize=True)


print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

print("\nTraining the Logistic Regression model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")


print("\nEvaluating model on the validation set...")
y_pred_val = model_pipeline.predict(X_val)
y_pred_proba_val = model_pipeline.predict_proba(X_val)[:, 1] # Probabilities for the positive class (class 1)


# Calculate classification metrics
accuracy = accuracy_score(y_val, y_pred_val)
precision = precision_score(y_val, y_pred_val)
recall = recall_score(y_val, y_pred_val)
f1 = f1_score(y_val, y_pred_val)
roc_auc = roc_auc_score(y_val, y_pred_proba_val) 
conf_matrix = confusion_matrix(y_val, y_pred_val)
class_report = classification_report(y_val, y_pred_val)


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)


test_df = pd.read_csv('/kaggle/input/heart-disease-prediction-dataquest/heart_test.csv') # Load the actual Kaggle test data


# Ensure the test set columns match the training set columns (order might not matter for pipeline)
# Important: Use the SAME columns used for training X
test_features = test_df[X.columns] 
# Use the *trained* pipeline (which includes preprocessing)
test_predictions = model_pipeline.predict(test_features)


test_predictions


submission_df = pd.DataFrame({'HeartDisease': test_predictions}) 


sub_sample = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/sample_submission.csv")


sub_sample.head()


sub = pd.DataFrame({'id': submission_df.index.values, 'HeartDisease': test_predictions})


sub.to_csv("submission.csv")

