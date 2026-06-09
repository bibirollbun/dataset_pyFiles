!pip install flaml


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flaml import AutoML
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns


# Data Loading and Preprocessing
# Load the datasets
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
    sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv") # Load Sample Submission
except FileNotFoundError:
    print("Error: Ensure train.csv, test.csv, and sample_submission.csv are in the same directory as this script.")
    exit()

# Combine train and test for preprocessing (handling categorical features consistently)
combined_df = pd.concat([train_df.drop('Fertilizer Name', axis=1), test_df], ignore_index=True)

# Label Encoding for categorical features (consistent across train/test)
categorical_features = [col for col in combined_df.columns if combined_df[col].dtype == 'object']
for col in categorical_features:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])

# Separate train and test
train_processed = combined_df.iloc[:len(train_df)]
test_processed = combined_df.iloc[len(train_df):]

# Target variable encoding
le_target = LabelEncoder()
train_target = le_target.fit_transform(train_df['Fertilizer Name'])
train_df['Fertilizer Name Encoded'] = train_target

# Combine X and y for FLAML
X = train_processed
y = train_target



# FLAML AutoML Configuration and Training
automl = AutoML()

# Use the specified automl.fit call with reduced time_budget
automl.fit(X, y, task="classification", metric='roc_auc_ovo', time_budget=3600, eval_method='cv')



# Model Evaluation and Analysis
# Print the best model found by FLAML
print("Best model:", automl.best_estimator)
print("Best hyperparameters:", automl.best_config)
print("Best ROC AUC on training data: {:.4f}".format(automl.best_loss))

# Predict probabilities on the training set (for evaluation)
y_pred_proba = automl.predict_proba(X)

# Check if y_pred_proba is None before proceeding
if y_pred_proba is None:
    print("Error: y_pred_proba is None.  FLAML likely failed to train a model within the time budget.")
    print("Try increasing the time_budget or simplifying the problem.")
    exit()

# Calculate ROC AUC for each class and overall
roc_auc_scores = []
for i in range(len(le_target.classes_)):
    roc_auc = roc_auc_score(y == i, y_pred_proba[:, i])
    roc_auc_scores.append(roc_auc)
    print(f"ROC AUC for class {le_target.classes_[i]}: {roc_auc:.4f}")

print(f"Mean ROC AUC: {np.mean(roc_auc_scores):.4f}")

# Visualize Predictions and ROC AUC
plt.figure(figsize=(12, 6))

# Plot predicted probabilities for the first few samples
num_samples_to_plot = min(10, len(X))  # Avoid plotting too many samples
plt.subplot(1, 2, 1)
sns.heatmap(y_pred_proba[:num_samples_to_plot], annot=False, cmap="viridis", xticklabels=le_target.classes_)
plt.title("Predicted Probabilities (First {} Samples)".format(num_samples_to_plot))
plt.ylabel("Sample Index")
plt.xlabel("Fertilizer Name")

# Plot ROC AUC scores
plt.subplot(1, 2, 2)
plt.bar(le_target.classes_, roc_auc_scores)
plt.title("ROC AUC per Fertilizer Type")
plt.xlabel("Fertilizer Type")
plt.ylabel("ROC AUC")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()



# Mean Average Precision @ 3 (MAP@3) Calculation
def map_at_k(y_true, y_pred, k=3):
    """
    Calculates Mean Average Precision at k (MAP@k).
    Args:
        y_true (list):  List of true labels.
        y_pred (list of lists): List of lists, where each inner list contains predicted labels.
        k (int): The cutoff rank.

    Returns:
        float: MAP@k score.
    """
    assert len(y_true) == len(y_pred), "Number of true labels must match number of prediction lists."
    return np.mean([apk(y_true[i], y_pred[i], y_true, le_target, k) for i in range(len(y_true))])

def apk(actual, predicted, y_true, le_target, k=3):
    """
    Calculates the average precision at k.
    Args:
        actual (int): The true label (single value).
        predicted (list): A list of predicted labels.
        y_true (list): List of true labels
        le_target (LabelEncoder): the label encoder for the target variable
        k (int): The maximum number of predicted elements.

    Returns:
        float: Average precision at k.
    """
    if isinstance(predicted, np.ndarray):
        if predicted.size == 0:
            return 0.0
    elif not predicted:
        return 0.0


    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if i >= k:
            break
        if p == le_target.inverse_transform([y_true[i]])[0] and p not in predicted[:i]: #Check if hit and not already counted
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not le_target.inverse_transform([y_true[i]])[0] in predicted[:k]:
        return 0.0

    return score / min(1, k) # min(1, k) because actual is always 1

# Get the indices of the top 3 predicted classes for EACH SAMPLE
top_3_indices = np.argsort(y_pred_proba, axis=1)[:, -3:]

#  Apply inverse_transform to each row of top_3_indices
y_pred_top3 = np.array([le_target.inverse_transform(row) for row in top_3_indices])


# Reverse the order of predictions to get the top 3 in descending order of probability
y_pred_top3 = np.fliplr(y_pred_top3)

# Calculate MAP@3
map3_score = map_at_k(y, y_pred_top3)
print(f"MAP@3 on training set: {map3_score:.4f}")


# Prediction on Test Data and Submission File 
# Predict probabilities on the test set
test_pred_proba = automl.predict_proba(test_processed)

# Get the indices of the top 3 predicted classes for EACH SAMPLE
test_top_3_indices = np.argsort(test_pred_proba, axis=1)[:, -3:]

#  Apply inverse_transform to each row of test_top_3_indices
test_pred_top3 = np.array([le_target.inverse_transform(row) for row in test_top_3_indices])

# Reverse the order of predictions to get the top 3 in descending order of probability
test_pred_top3 = np.fliplr(test_pred_top3)

# Create the submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id'],
                              'Fertilizer Name': [' '.join(row) for row in test_pred_top3]})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

# Display the head of the submission file
print("\nSubmission file created successfully!")
submission_df.head()




