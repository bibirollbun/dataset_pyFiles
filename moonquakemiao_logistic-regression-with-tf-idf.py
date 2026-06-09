
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
# this is for the Kaggle dataset to input the data
# To make the whole competition correct, I input the original competition dataset
train_path = "/kaggle/input/py-sphere-movie-review-sentiment-challenge/train.csv"
test_path = "/kaggle/input/py-sphere-movie-review-sentiment-challenge/test.csv"




# Read the csv
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print("Train dataset shape:", train.shape)
print("Test dataset shape:", test.shape)
display(train.head())
display(test.head())




# TF-IDF Text Feature Extraction
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=10000    
)
X_train = vectorizer.fit_transform(train['review'])
X_test = vectorizer.transform(test['review'])
y_train = train['sentiment']

print("Training Feature Dimension：", X_train.shape)



# Splitting a portion of the training set for simple validation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split original training data into sub-training (80%) and validation (20%) sets.
# `stratify=y_train` ensures class distribution is maintained.
X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Initialize and train a Logistic Regression model.
# `max_iter=1000` for convergence, `random_state=42` for reproducibility.
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_subtrain, y_subtrain)

# Predict on the validation set.
val_pred = clf.predict(X_val)

# Calculate and print validation accuracy.
acc = accuracy_score(y_val, val_pred)
print("Validation Accuracy: %.4f" % acc)


# Initialize a new Logistic Regression model instance.
# max_iter=1000 for convergence, random_state=42 for reproducibility.
clf_all = LogisticRegression(max_iter=1000, random_state=42)

# Train this model on the *entire* training dataset (X_train, y_train).
# This uses all available training data to build the final model.
clf_all.fit(X_train, y_train)

# Make predictions on the unseen test dataset (X_test).
# These predictions will be used for final evaluation or submission.
test_pred = clf_all.predict(X_test)



# Kaggle required format: id, sentiment
import pandas as pd # Assuming pandas is imported earlier

# Create a DataFrame for submission.
# It includes the 'id' from the test set and the predicted 'sentiment'.
sub = pd.DataFrame({
    'id': test['id'],        # Extract 'id' column from the original test data
    'sentiment': test_pred    # Use the predictions made by the model
})

# Save the DataFrame to a CSV file named "submission.csv".
# index=False prevents pandas from writing the DataFrame index as a column in the CSV.
sub.to_csv("submission.csv", index=False)

# Confirm that the submission file has been generated.
print("submission.csv generated")




