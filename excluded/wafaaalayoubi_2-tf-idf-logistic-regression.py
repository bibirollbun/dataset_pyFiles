import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# --- 1. Load and Prepare Data ---
# Based on the findings from our EDA notebook, we know we only need the labeled data.

# Load the full training data
train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')

# Create a new dataframe containing only the rows with a misconception label
labeled_df = train_df.dropna(subset=['Misconception']).copy()

print(f"Using {labeled_df.shape[0]} labeled samples for training and validation.")
print("-" * 30)


# --- 2. Define Features (X) and Target (y) ---
X = labeled_df['StudentExplanation']
y = labeled_df['Misconception']


# --- 3. Create Training and Validation Sets ---
# We use stratify=y to ensure the train and validation sets have the same
# proportion of each misconception label, which is crucial for an imbalanced dataset.
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")


# --- 1. Define the Pipeline ---
# This pipeline will first transform the text data using TF-IDF,
# and then feed the resulting numerical vectors into the Logistic Regression model.

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=3)),
    ('clf', LogisticRegression(random_state=42, max_iter=1000, C=0.5))
])

# Let's break down the parameters:
# TfidfVectorizer:
#   - stop_words='english': Removes common English words like 'the', 'a', etc.
#   - ngram_range=(1, 2): Considers both single words (unigrams) and pairs of words (bigrams). This helps capture more context. e.g., "add numerator"
#   - min_df=3: Ignores words that appear in fewer than 3 documents. This helps remove rare spelling mistakes or noise.
#
# LogisticRegression:
#   - max_iter=1000: Allows the model more iterations to converge to a solution.
#   - C=0.5: A regularization parameter to prevent overfitting.


# --- 2. Train the Pipeline ---
print("Training the TF-IDF + Logistic Regression pipeline...")
pipeline.fit(X_train, y_train)
print("Training complete!")


# --- 3. Evaluate the Model on the Validation Set ---
print("\nEvaluating the model...")

# Get predictions on the validation data
y_pred = pipeline.predict(X_val)

# Calculate simple accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Display a few predictions vs actuals to see how it's doing
results_df = pd.DataFrame({'Text': X_val, 'Actual': y_val, 'Predicted': y_pred})
display(results_df.head(10))


# --- 1. Get the Top 3 Predictions and Probabilities ---

# Get the predicted probabilities for each class for the validation set
# This will be a matrix of size (n_samples, n_classes)
y_pred_probs = pipeline.predict_proba(X_val)

# Get the class labels in the order the model learned them
class_labels = pipeline.classes_

# Get the indices of the top 3 predictions for each sample
# We use np.argsort to sort the probabilities and get their original indices
top3_indices = np.argsort(y_pred_probs, axis=1)[:, ::-1][:, :3]

# Get the actual class names for the top 3 predictions
top3_preds = np.array(class_labels)[top3_indices]


# --- 2. Calculate MAP@3 Score Manually ---
# This logic is the same as the one used in the original notebook we studied

# Convert y_val to a numpy array for easier comparison
y_val_arr = np.array(y_val)
map3_score = 0

for i in range(len(y_val_arr)):
    true_label = y_val_arr[i]
    top3 = top3_preds[i]

    if true_label == top3[0]:
        map3_score += 1.0
    elif true_label == top3[1]:
        map3_score += 1.0 / 2.0
    elif true_label == top3[2]:
        map3_score += 1.0 / 3.0

# The final score is the average over all samples
final_map3 = map3_score / len(y_val_arr)

print(f"Validation MAP@3 Score: {final_map3:.4f}")


# --- 3. Display Results with Top 3 Predictions ---

# Create a DataFrame to show the top 3 predictions alongside the actual label
results_with_top3_df = pd.DataFrame({
    'Text': X_val,
    'Actual': y_val,
    'Pred_1': top3_preds[:, 0],
    'Pred_2': top3_preds[:, 1],
    'Pred_3': top3_preds[:, 2]
})

print("\nSample Predictions (including Top 3):")
display(results_with_top3_df.head(15))


# --- 1. Load the Official Test Data ---
# Note: The test data provided by the competition is just a small sample.
# Our code will be re-run on a hidden, larger test set when we submit.
test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

# The text we need to predict on is in the 'StudentExplanation' column
X_test = test_df['StudentExplanation']

print(f"Loaded {len(X_test)} samples from the test set.")
print("-" * 30)


# --- 2. Make Top 3 Predictions on the Test Set ---

# Use predict_proba to get probabilities for all classes
test_pred_probs = pipeline.predict_proba(X_test)

# Get the indices of the top 3 predictions for each test sample
top3_test_indices = np.argsort(test_pred_probs, axis=1)[:, ::-1][:, :3]

# Get the actual class names for the top 3 predictions
top3_test_preds = np.array(class_labels)[top3_test_indices]


# --- 3. Format for Submission ---

# The submission format requires the top 3 labels to be in a single string,
# separated by spaces.
# e.g., "Incomplete Additive Subtraction"

# We can use a list comprehension to join the top 3 predictions for each row
predictions_for_submission = [' '.join(preds) for preds in top3_test_preds]

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Misconception': predictions_for_submission
})


# --- 4. Save the Submission File ---
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print("First 5 rows of submission.csv:")
display(submission_df.head())




