import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score as AUC


# Load the training and test datasets from CSV files
df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

# Print the shape of the training dataset (rows, columns)
print(df.shape)

# Display the first 5 rows of the training dataset
df.head()


# The target variable we want to predict
target = "loan_paid_back"

# Get a list of all feature column names except the target
cols = df.drop(columns=target).columns.tolist()

# Identify categorical columns (columns with text/object data type)
cat = [c for c in cols if df[c].dtype == "object"]

# Encoding categorical features into numerical values
# Many machine learning models require numbers instead of text
encode = OrdinalEncoder()

# Fit the encoder on training categorical data and transform it to numbers
df[cat] = encode.fit_transform(df[cat])

# Transform test data using the same encoder (do NOT fit again!)
df_test[cat] = encode.transform(df_test[cat])

df.drop(columns = "id", inplace = True)


def RF(X_train, X_test, y_train):
    """
    Builds a Random Forest model, fits it to the training data, and returns 
    the predicted probabilities for the positive class on the test data.
    """
    
    # Initialize the Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=200,        # Number of trees in the forest; more trees usually give more stable results
        max_depth=10,            # Limits the depth of each tree to prevent overfitting
        min_samples_leaf=50,     # Each leaf must have at least 50 samples; prevents rules from being too specific
        class_weight='balanced', # Adjusts weights to account for imbalanced classes (e.g., 80/20 distribution)
        max_features='sqrt',     # Each split considers only sqrt(total_features) for randomness
        random_state=42,         # Ensures results are reproducible
        n_jobs=-1                # Use all CPU cores to speed up training
    )
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Predict probabilities on the test data
    # [:, 1] selects the probability of the positive class (e.g., loan paid back)
    return model.predict_proba(X_test)[:, 1]



# Separate features (X) and target (y)
X = df.drop(columns=target)
y = df[target]

# StratifiedKFold is a type of cross-validation that ensures each fold has roughly 
# the same proportion of each class as the full dataset
kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

# Array to store out-of-fold (OOF) predictions for all samples
RF_OOF = np.zeros(len(y))

# Loop through each fold
for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f"Fold {i+1}")
    
    # Split data into training and validation sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the Random Forest and get predictions for this fold's validation set
    # Store the predictions in the corresponding indices of RF_OOF
    RF_OOF[test_index] = RF(X_train, X_test, y_train)

# Calculate the overall AUC score using the out-of-fold predictions
RF_AUC = AUC(y, RF_OOF)
print(f"RF AUC: {RF_AUC}")



# Separate features and target for training
X_train = df.drop(columns=target)
y_train = df[target]

# Test features
X_test = df_test

# Predict probabilities on the test set using the trained Random Forest
# Note: We drop 'id' column if it exists in the test data since it is not a feature
RF_pred = RF(X_train, X_test.drop(columns="id"), y_train)

# Create a submission DataFrame
# Copy the 'id' column from the test set
sub = df_test["id"].copy()
sub = pd.DataFrame(sub)

# Add the predictions as a new column with the target name
sub[target] = RF_pred

# Save the submission DataFrame to a CSV file
# index=False prevents pandas from adding an extra index column
sub.to_csv("submission.csv", index=False)




