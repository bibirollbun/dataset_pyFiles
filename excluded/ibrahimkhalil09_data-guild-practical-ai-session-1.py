import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set style
sns.set_style("darkgrid")


# Load the Train Dataset
df = pd.read_csv("/kaggle/input/data-guild-will-you-be-productive-today/train.csv")

# Show first few rows
df.head()


# Basic Data Exploration

print("Dataset Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nClass Distribution:")
print(df['productive_day'].value_counts())


# Visualizing Feature Distributions

plt.figure(figsize=(14, 6))
df.hist(bins=30, figsize=(14, 6), edgecolor='black')
plt.tight_layout()
plt.show()


# Class distribution
sns.countplot(data=df, x="productive_day", palette="coolwarm")
plt.title("Productivity Class Distribution")
plt.show()


# Detecting Outliers
plt.figure(figsize=(12, 5))
sns.boxplot(data=df[['lines_of_code_written', 'browsing_time', 'coffee_intake']])
plt.title("Boxplot to Detect Outliers")
plt.show()


# Handling Missing Data & Outliers

df['hours_of_meetings'].fillna(df['hours_of_meetings'].median(), inplace=True)
df['hours_of_sleep'].fillna(df['hours_of_sleep'].median(), inplace=True)

# Cap extreme outliers
df['lines_of_code_written'] = np.where(df['lines_of_code_written'] > 4000, 4000, df['lines_of_code_written'])

# Handling Duplicates
df.drop_duplicates(inplace=True)


# Dropping Unnecessary Columns: Label shouldn't be include in features
X = df.drop(columns=['productive_day'])
y = df['productive_day']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split Data into Train & Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(f"\nğŸ”¹ Model: {name}")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))




# Plot Model Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.title("Model Performance Comparison")
plt.ylabel("Accuracy Score")
plt.ylim(0, 1)
plt.show()


# Select the best model based on accuracy
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\nğŸ�† The winning model is: {best_model_name} with accuracy: {results[best_model_name]:.4f}")



# Load the test set (provided in the competition)
test_df = pd.read_csv("/kaggle/input/data-guild-will-you-be-productive-today/test.csv")

# Handle missing values
test_df['hours_of_meetings'].fillna(test_df['hours_of_meetings'].median(), inplace=True)
test_df['hours_of_sleep'].fillna(test_df['hours_of_sleep'].median(), inplace=True)

# Scale the test set using the same scaler as training data
test_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

# Make predictions
test_predictions = best_model.predict(test_scaled)


# Create submission DataFrame
submission = pd.read_csv("/kaggle/input/data-guild-will-you-be-productive-today/sample_submission.csv")

submission["productive_day"] = test_predictions

# Save submission file
submission.to_csv("submission.csv", index=False)

print("\nâœ… Submission file 'submission.csv' is ready!")


# ğŸ�¯ Function to Predict Productivity Based on User Input
def predict_productivity(model, scaler):
    print("\nğŸ“� Enter Your Daily Work Stats to Predict Productivity:\n")

    # Get user input
    coffee_intake = float(input("â˜• Coffee intake (cups): "))
    hours_of_meetings = float(input("ğŸ“… Hours spent in meetings: "))
    bugs_assigned = int(input("ğŸ�› Bugs assigned today: "))
    lines_of_code_written = int(input("ğŸ’» Lines of code written: "))
    hours_of_sleep = float(input("ğŸ˜´ Hours of sleep last night: "))
    browsing_time = float(input("ğŸŒ� Hours spent browsing non-work websites: "))

    # Create a dataframe for user input (ensuring it has the correct column names)
    user_data = pd.DataFrame([[
        coffee_intake, hours_of_meetings, bugs_assigned, 
        lines_of_code_written, hours_of_sleep, browsing_time
    ]], columns=['coffee_intake', 'hours_of_meetings', 'bugs_assigned', 
                 'lines_of_code_written', 'hours_of_sleep', 'browsing_time'])

    # Scale the input using the same scaler as training data
    user_data_scaled = pd.DataFrame(scaler.transform(user_data), columns=user_data.columns)

    # Predict productivity
    prediction = model.predict(user_data_scaled)[0]
    result = "âœ… Productive Day!" if prediction == 1 else "â�Œ Not a Productive Day!"

    print("\nğŸ”® Prediction:", result)

# ğŸ�† Use the Best Model for Live Predictions
predict_productivity(best_model, scaler)

