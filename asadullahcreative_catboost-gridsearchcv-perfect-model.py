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

# ğŸ¤– CatBoost - Gradient Boosting Framework
from catboost import CatBoostClassifier, Pool                # High-performance gradient boosting algorithm for classification/regression

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



from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

# =====================================================
# ğŸ”� Apply Grid Search CV to find best CatBoost hyperparameters
# =====================================================

# Initialize CatBoostClassifier 
cbc = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Logloss",
    random_seed=42,
    verbose=False
)

# Define the parameter grid to search over
param_grid = {
    'depth': [3, 4, 5, 6],                # analogous to max_depth
    'learning_rate': [0.01, 0.05, 0.1],  # analogous to eta
    'iterations': [50, 100, 150],         # number of boosting rounds
    'subsample': [0.7, 0.8, 0.9],         # row sampling
    'rsm': [0.7, 0.8, 0.9]                # feature subsampling
}

# Set up GridSearchCV for binary classification
grid_search = GridSearchCV(
    estimator=cbc,
    param_grid=param_grid,
    cv=5,
    scoring='neg_log_loss',  
    n_jobs=-1,
    verbose=1
)

# Run Grid Search on your training data (X, y)
grid_search.fit(X, y)

# Access the best CatBoost model from GridSearchCV
best_model = grid_search.best_estimator_

# Print best parameters and best score
print(" Results from Grid Search ")
print("\n The best model across ALL searched params:\n", grid_search.best_estimator_)
print("\n The best score across ALL searched params:\n", grid_search.best_score_)
print("\n The best parameters across ALL searched params:\n", grid_search.best_params_)



# Assuming `model` is your trained CatBoostClassifier

# =====================================================
# ğŸ�¯ Predict on scaled test data using CatBoost model
# =====================================================

# Use predict_proba to get probabilities for the positive class (class 1)
test_probabilities = best_model.predict_proba(X_test_scaled)[:, 1]

# Convert probabilities to binary class predictions using 0.5 threshold
test_predictions = (test_probabilities > 0.5).astype(int)

# âœ… Predictions are ready to be used (e.g. for submission or evaluation)



# =====================================================
# ğŸ“¤Create and Save Submission File
# =====================================================

# ğŸ§  Convert predicted probabilities on scaled test data into binary predictions (0 or 1)
final_test_labels = (test_probabilities > 0.5).astype(int)

# ğŸ”„ Decode integer labels back to original class names using label encoder
submission["Personality"] = le.inverse_transform(final_test_labels)

# ğŸ’¾ Save the submission file to CSV without the index column
submission.to_csv("submission.csv", index=False)

# ğŸ‘€ Display a preview of the first few rows of the submission file
submission.head()


