# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/x-small-deberta-baseline-a377e3/oof_train_probs.csv')


df


from sklearn.metrics import f1_score

# Assuming your DataFrame is called df
# Get the predicted labels using argmax
val_preds = df[[f'class_{i}' for i in range(8)]].values.argmax(axis=1)

# Get the true labels
val_true = df['true_label'].values

# Compute the F1 score
score = f1_score(val_true, val_preds, average='micro')
print("F1 Score (micro):", score)





logits = df[[f'class_{i}' for i in range(8)]].values
softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
probs = softmax(logits)

# Get max probability and predicted label
max_probs = probs.max(axis=1)
val_preds = probs.argmax(axis=1)
val_true = df['true_label'].values

# Filter: only consider rows where max prob > 0.5
mask = max_probs > 0.3
filtered_preds = val_preds[mask]
filtered_true = val_true[mask]

# Compute F1 score (micro) on filtered set
score = f1_score(filtered_true, filtered_preds, average='micro')
print("F1 Score (micro, confidence > 0.5):", score)


# Get the class columns
class_cols = [col for col in df.columns if col.startswith('class_')]

# Normalize each row of class probabilities so they sum to 1
df[class_cols] = df[class_cols].div(df[class_cols].sum(axis=1), axis=0)



# Get predicted label (most probable class)
df['predicted_label'] = df[class_cols].idxmax(axis=1).apply(lambda x: int(x.split('_')[1]))

# Get confidence (max probability)
df['confidence'] = df[class_cols].max(axis=1)

# Filter where confidence > 0.5
filtered = df[df['confidence'] > 0.95]


# Show how many rows passed the filter
print(f"Rows with confidence > 0.5: {len(filtered)} out of {len(df)}")
score = f1_score(filtered['true_label'], filtered['predicted_label'], average='micro')
print("F1 Score (micro, confidence > 0.5):", score)


from sklearn.metrics import classification_report
import pandas as pd

# Define class names (ordered by class_0 to class_7)
class_names = [
    "Algebra",
    "Geometry and Trigonometry",
    "Calculus and Analysis",
    "Probability and Statistics",
    "Number Theory",
    "Combinatorics and Discrete Math",
    "Linear Algebra",
    "Abstract Algebra and Topology"
]

# Filter high-confidence rows (from previous step)
filtered = df[df['confidence'] > 0.95]

# Get true and predicted labels
y_true = filtered['true_label']
y_pred = filtered['predicted_label']

# Generate classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)


best_test  = pd.read_csv('/kaggle/input/x-small-deberta-baseline-304939/test_probs.csv')


best_test 


# Identify class columns
class_cols = [col for col in best_test .columns if col.startswith('class_')]

# Normalize each row to make them valid probability distributions
best_test [class_cols] = best_test[class_cols].div(best_test [class_cols].sum(axis=1), axis=0)

# Get confidence (max probability per row)
best_test ['confidence'] = best_test [class_cols].max(axis=1)

# Filter rows with confidence > 0.5
filtered_test = best_test [best_test ['confidence'] > 0.98]

# Print how many rows are left
print(f"Rows with confidence > 0.5: {len(filtered_test)} out of {len(best_test )}")


import pandas as pd

# Load both submissions
sub_high = pd.read_csv('/kaggle/input/x-small-deberta-baseline-304939/submission.csv')  # predictions for high confidence
sub_low = pd.read_csv('/kaggle/input/3-subs-ensemble/submission_Qwen2.5_32B_temp00.csv')    # fallback predictions for low confidence

# Load and normalize the test probabilities again
test_df = best_test  # contains class_0 to class_7
class_cols = [col for col in test_df.columns if col.startswith('class_')]

# Normalize to ensure proper probabilities
test_df[class_cols] = test_df[class_cols].div(test_df[class_cols].sum(axis=1), axis=0)

# Calculate confidence
test_df['confidence'] = test_df[class_cols].max(axis=1)

# Identify high-confidence indices
high_conf_indices = test_df[test_df['confidence'] > 0.98].index

# Choose labels based on confidence
final_labels = []
for i in range(len(test_df)):
    if i in high_conf_indices:
        final_labels.append(sub_high.loc[i, 'label'])
    else:
        final_labels.append(sub_low.loc[i, 'label'])

# Create final submission
final_submission = sub_high.copy()
final_submission['label'] = final_labels

# Save
final_submission.to_csv('submission_confidence_based.csv', index=False)
print("Submission saved as submission_confidence_based.csv")



final_submission

