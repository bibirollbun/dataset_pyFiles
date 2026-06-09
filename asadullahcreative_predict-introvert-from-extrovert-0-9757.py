# =====================================================
# ğŸ§ Import Required Libraries
# =====================================================

# ğŸ“‚ Basic Tools for Data Handling
import pandas as pd              # For loading and manipulating tabular data
import numpy as np               # For numerical operations and array handling

# ğŸ“Š Scikit-learn Tools for Preprocessing & Validation
from sklearn.model_selection import StratifiedKFold        # For stratified cross-validation
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder  # To convert categorical features to numbers
from sklearn.preprocessing import StandardScaler           # To scale features (mean=0, std=1)
from sklearn.metrics import accuracy_score                 # To evaluate model performance (e.g., accuracy)

# ğŸ¤– XGBoost - Gradient Boosting Framework
import xgboost as xgb                # High-performance gradient boosting algorithm for classification/regression

# âœ… All essential libraries successfully imported!



# =====================================================
# ğŸ“¥ Load the Dataset
# =====================================================

# ğŸ”¹ Load the training data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")  # Contains features + target column

# ğŸ”¹ Load the test data (without target labels)
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")    # We will make predictions on this

# ğŸ”¹ Load the sample submission file
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")  # Format for final predictions (for Kaggle submission)

# âœ… Data loaded successfully!



# Let's take a quick look at the data 
train.head()


# Let's check the shape of the data 
print(f"We have {train.shape[0]} rows and {train.shape[1]} columns in the training data")


# Let's check the info of the data
train.info()


# =====================================================
# ğŸ�¯Encode the Target Column
# =====================================================

# ğŸ§  'Personality' is a categorical target (text labels), so we need to convert it to numbers

# ğŸ”¹ Initialize the encoder
le = LabelEncoder()

# ğŸ”¹ Fit the encoder on the 'Personality' column and transform it
train["Personality_encoded"] = le.fit_transform(train["Personality"])

# âœ… Now, we can use 'Personality_encoded' for training our model!



# =====================================================
# Create Features and Target
# =====================================================

# ğŸ”¹ Drop unnecessary columns from training data:
# - 'id' â�¤ just an identifier
# - 'Personality' â�¤ original target (text)
# - 'Personality_encoded' â�¤ used as our target (y), not as input
X = train.drop(columns=["id", "Personality", "Personality_encoded"])

# ğŸ”¹ Set the encoded target variable as y
y = train["Personality_encoded"]

# ğŸ”¹ Prepare test features by dropping the 'id' column (weâ€™ll predict on this)
X_test = test.drop(columns=["id"])

# âœ… Features (X), Target (y), and Test Set (X_test) are ready!



# =====================================================
# ğŸ”¤Encode Categorical Columns (for both train & test)
# =====================================================

# ğŸ§© Combine training and test data to apply consistent encoding
combined = pd.concat([X, X_test], axis=0)

# ğŸ”� Identify all categorical columns (columns with object/string data types)
cat_cols = combined.select_dtypes(include="object").columns.tolist()

# ğŸ�¯ Initialize an ordinal encoder (converts strings to integer levels)
encoder = OrdinalEncoder()

# ğŸ”„ Fit the encoder on the combined data and transform all categorical columns
combined[cat_cols] = encoder.fit_transform(combined[cat_cols])



# =====================================================
# ğŸ”�Split Combined Data Back into Train and Test Sets
# =====================================================

# âœ‚ï¸� Slice back the original training data (same number of rows as X)
X = combined.iloc[:len(X)].reset_index(drop=True)

# âœ‚ï¸� Slice the remaining rows as test data
X_test = combined.iloc[len(X):].reset_index(drop=True)

# ğŸ§¹ reset_index(drop=True) is used to clean up row numbers after slicing

# âœ… Now, X and X_test are fully numeric and ready for modeling!



# =====================================================
# ğŸ“� Scale Features Using StandardScaler
# =====================================================

# ğŸ”¹ Initialize the scaler
scaler = StandardScaler()

# ğŸ”„ Fit the scaler on training data and transform both train and test
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# âœ… Scaling is complete! Now both X_scaled and X_test_scaled are ready for modeling.



# =====================================================
# âš™ï¸� STEP 7: Setup XGBoost Parameters
# =====================================================

# These parameters control how the XGBoost model learns from the data.
params = {
    "objective": "binary:logistic",     # ğŸ�¯ Binary classification (outputs probability between 0 and 1)
    "eval_metric": "logloss",           # ğŸ“‰ Evaluation metric: log loss (lower = better)
    "max_depth": 4,                     # ğŸŒ² Maximum depth of each decision tree (controls complexity)
    "eta": 0.1,                         # ğŸ�¢ Learning rate (smaller = slower but more accurate training)
    "subsample": 0.8,                   # ğŸ”� Use 80% of data for each tree (adds randomness, prevents overfitting)
    "colsample_bytree": 0.8,            # ğŸ“Š Use 80% of features per tree (more randomness)
    "random_state": 42                  # ğŸ”� Reproducibility of results
}

# âœ… Parameters are set! Ready to train the model using Stratified K-Fold



# =====================================================
# ğŸ”� Train XGBoost with Stratified K-Fold Cross-Validation
# =====================================================

# ğŸ§  StratifiedKFold ensures each fold has the same proportion of target classes (balanced labels)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ğŸ“¦ Initialize empty arrays to store prediction results
validation_predictions = np.zeros(len(X))                  # Stores predictions made on validation sets
average_test_predictions = np.zeros(len(X_test))           # Will store averaged predictions across all test folds

# ğŸš€ Start looping through each fold
for fold_number, (train_indices, val_indices) in enumerate(kfold.split(X, y)):
    print(f"ğŸ”„ Training Fold {fold_number + 1}...")  # Track progress

    # ----------------------------------------------
    # ğŸ“� 1. Split the dataset into training and validation parts for this fold
    # ----------------------------------------------
    X_train_fold = X.iloc[train_indices]     # Features for training
    X_val_fold   = X.iloc[val_indices]       # Features for validation
    y_train_fold = y.iloc[train_indices]     # Labels for training
    y_val_fold   = y.iloc[val_indices]       # Labels for validation

    # ----------------------------------------------
    # âš™ï¸� 2. Convert datasets into XGBoost's DMatrix format
    # ----------------------------------------------
    # DMatrix is an optimized data structure used by XGBoost for faster training
    train_data = xgb.DMatrix(X_train_fold, label=y_train_fold)
    val_data   = xgb.DMatrix(X_val_fold, label=y_val_fold)
    test_data  = xgb.DMatrix(X_test)  # Test data stays the same across all folds

    # ----------------------------------------------
    # ğŸ�¯ 3. Train the XGBoost model on training data, validate on validation set
    # ----------------------------------------------
    model = xgb.train(
        params=params,                       # XGBoost hyperparameters (defined earlier)
        dtrain=train_data,                   # Training set
        num_boost_round=100,                 # Max number of boosting rounds
        evals=[(val_data, "validation")],    # Evaluation happens on validation set
        early_stopping_rounds=10,            # Stop training early if no improvement in 10 rounds
        verbose_eval=False                   # Suppress training logs (set to True for debugging)
    )

    # ----------------------------------------------
    # ğŸ“Š 4. Predict on validation set and store results
    # ----------------------------------------------
    # The model returns probabilities, so we apply a threshold (0.5) to convert to class 0 or 1
    val_preds = model.predict(val_data) > 0.5
    validation_predictions[val_indices] = val_preds  # Store predictions in the correct place

    # ----------------------------------------------
    # ğŸ“ˆ 5. Predict on the test set and add to cumulative result
    # ----------------------------------------------
    # Each fold makes predictions on the test set â€” we average them across all folds
    average_test_predictions += model.predict(test_data) / kfold.n_splits

# =====================================================
# âœ… Cross-validation is complete!
# You now have:
# - `validation_predictions`: model outputs for training data (used for evaluation)
# - `average_test_predictions`: final predictions for the test set (to be used in submission)
# =====================================================



# =====================================================
# ğŸ“Š STEP 9: Evaluate Validation Performance
# =====================================================

# ğŸ�¯ Calculate accuracy using true labels vs. validation predictions
cv_accuracy = accuracy_score(y, validation_predictions)

# ğŸ–¨ï¸� Print the Cross-Validation Accuracy
print(f"âœ… Cross-Validation Accuracy: {cv_accuracy:.4f}")



# =====================================================
# ğŸ“¤ STEP 10: Create and Save Submission File
# =====================================================

# ğŸ§  Convert averaged probabilities on test set into binary predictions (0 or 1)
final_test_labels = (average_test_predictions > 0.5).astype(int)

# ğŸ”„ Decode back from label integers to original class names (e.g., 0 â†’ Introvert, 1 â†’ Extrovert)
submission["Personality"] = le.inverse_transform(final_test_labels)

# ğŸ’¾ Save the submission file
submission.to_csv("submission.csv", index=False)

# ğŸ‘€ Show a preview of the first few rows
submission.head()


