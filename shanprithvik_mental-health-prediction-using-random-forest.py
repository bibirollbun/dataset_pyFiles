# Step 1: Import Libraries & Load Training Data
    #pandas to read and manage data tables (like Excel).
    #matplotlib and seaborn to draw graphs.
    #sklearn is the library that helps us build the Machine Learning model.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

train_data = pd.read_csv("/kaggle/input/depressed-people/train.csv")


# Step 2: Encode Categorical Data

    # Machine learning works with numbers, not words.
    # So we convert text into numbers (like "Male" = 1, "Female" = 0) using LabelEncoder.
    # This is applied to fields like Gender, Sleep Duration, etc.

categorical_fields = [
    'Gender', 'Sleep Duration', 'Dietary Habits',
    'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness',
    'Depression'
]
for feature in categorical_fields:
    encoder = LabelEncoder()
    train_data[feature] = encoder.fit_transform(train_data[feature])
    print(f"Encoded '{feature}' →", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))


# Step 3: Preprocessing (Drop Unneeded Column)
    # We remove the index column because it's just a row number, not useful for training the model.

if 'index' in train_data.columns:
    train_data = train_data.drop(columns=['index'])
train_data.head()


# Step 4: Split Features and Target
    # X contains the input features (like sleep, habits).
    # y is what we want to predict — whether the person has Depression (0 or 1).

X = train_data.drop(columns=['Depression'])
y = train_data['Depression']


# Step 5: Split Into Training and Testing Sets
    # We split the data into 80% for training the model and 20% for testing how well it works.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#  Step 6: Train Random Forest Model
 #We create a Random Forest model, which is made of many decision trees. It’s very good at finding patterns and works better than logistic regression in most cases.

model =RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)


#  Step 7: Predict and Evaluate
    # After training, we test the model and check how accurate it is. Precision, recall, and F1 score give us more detailed information about how good it really is.
y_pred = model.predict(X_test)

print("Model Performance:")
print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall   : {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"F1 Score : {f1_score(y_test, y_pred) * 100:.2f}%")


# Step 8: Confusion Matrix
    # This matrix shows how many predictions were correct and where the model made mistakes — like predicting “No” when it was actually “Yes”

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#  Step 9: Hyperparameter Tuning (Optional, Improves Accuracy)
    # Grid Search tries many combinations to find the best settings for the Random Forest. This helps us improve the model’s performance even more.

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
best_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, best_pred))


#  Step 10: Prepare Test Data
    # We now load the new test data (without depression labels) and prepare it the same way: encode words into numbers, and drop the index column.

test_data = pd.read_csv("/kaggle/input/depressed-people/test.csv")

for feature in categorical_fields[:-1]:  # Skip target column
    test_data[feature] = LabelEncoder().fit_transform(test_data[feature])

test_indices = test_data['index']
test_data = test_data.drop(columns=['index'])
test_data.head()


#  Step 11: Make Final Predictions
    # We use our best model to predict depression on the test data, convert the numbers back into “Yes” and “No”, and save the final result in a CSV file for Kaggle submission.

final_output = best_model.predict(test_data)
decoded_predictions = ['Yes' if label == 1 else 'No' for label in final_output]

submission_df = pd.DataFrame({
    'index': test_indices,
    'Depression': decoded_predictions
})
submission_df.to_csv("submission.csv", index=False)

