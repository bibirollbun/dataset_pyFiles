# Cell 1: Load Training Data

# 1.1) Import standard Python libraries for data analysis (pandas, numpy).
import pandas as pd, numpy as np

# 1.2) Import libraries for plotting (matplotlib).
import matplotlib.pyplot as plt

# 1.3) Load the training dataset from Kaggle’s input directory.
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")

# 1.4) Display the shape of the dataset to understand the number of rows and columns.
print("Train shape", train.shape )

# 1.5) Use .head() to preview the first few rows (helpful for sanity checking the data.).
train.head()


# Cell 2: Load Test Data

# 2.1) Load the test dataset.
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# 2.2) Print out the shape and previews the test dataset.
print("Test shape:", test.shape )

# 2.2) Print out the shape and previews the test dataset.
test.head()


# Cell 3: Select Features

# 3.1) Identifies and removes non-feature columns:
# - 'rainfall': This is the target variable we’re trying to predict.
# - 'id': This is a unique identifier and doesn’t carry useful predictive information.
RMV = ['rainfall','id']

# 3.2) This list comprehension filters out rainfall and id from the training columns, keeping only the actual features used for model input.
FEATURES = [c for c in train.columns if not c in RMV]

# 3.3) Print the names of selected features to verify what will be used by the model.
print("Our features are:")
print( FEATURES )


# Cell 4: Import XGBoost + K-Fold

# 4.1) Import K-Fold cross-validation from scikit-learn.
from sklearn.model_selection import KFold

# 4.2) Import XGBoost regression and classification models.
from xgboost import XGBRegressor, XGBClassifier

# 4.3) Also import the main xgboost module to check the version.
import xgboost

# 4.4.) Print the version of XGBoost being used. —-> [Note: This helps ensure compatibility and reproducibility, important because behavior/performance may differ between versions.]
print("Using XGBoost version", xgboost.__version__)


%%time
# Cell 5: Train XGBoost Model with Cross-Validation
# 5.1) %%time is a Jupyter notebook cell magic command that measures and displays the execution time of the entire cell.

# 5.2) Define the number of folds for cross-validation. 5-fold CV splits the training data into 5 parts.
FOLDS = 5

# 5.3) Shuffle ensures randomness, random_state ensures reproducibility.
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# 5.3) Initialize arrays to store predictions:
# 5.3.1) oof_xgb will hold out-of-fold predictions on the training data (used to calculate validation metrics like AUC).
oof_xgb = np.zeros(len(train))
# 5.3.2) pred_xgb will collect predictions on the test set across all folds.
pred_xgb = np.zeros(len(test))

# 5.4) Loop through each fold of the cross-validation.
for i, (train_index, test_index) in enumerate(kf.split(train)):

    # 5.5) Print out the current fold number for tracking progress.
    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)

    # 5.6) Split the data into training and validation sets for this fold.
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"rainfall"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"rainfall"]

    # 5.7) Prepare the test data (same for all folds).
    x_test = test[FEATURES].copy()

    # 5.8) Initialize the XGBoost classifier model.
    model = XGBClassifier(
        device="cuda", # Use GPU (CUDA) for faster training.
        max_depth=6,   # Maximum depth of each tree.
        colsample_bytree=0.9, # Fraction of features used per tree.
        subsample=0.9, # Fraction of training data used per tree.
        n_estimators=10_000,  # Maximum number of trees (with early stopping).
        learning_rate=0.1, # Step size shrinkage for each boosting round.
        eval_metric="auc", # Use AUC as the evaluation metric.
        early_stopping_rounds=100, # Stop training if validation AUC doesn't improve for 100 rounds.
        alpha=1, # L1 regularization term on weights.
    )
    # 5.9) Train the model on the training set and validate on the validation set.
    model.fit(
        x_train, # Training features: the input data (columns) used to learn patterns.
        y_train,  # Training labels: the actual target values (in this case, 'rainfall') the model tries to predict.
        eval_set=[(x_valid, y_valid)],  # Evaluation happens on the validation fold.
        verbose=100 # Print progress every 100 rounds.
    )

    # 5.9) Predict probability for the validation set (OOF prediction).
    # INFER OOF
    oof_xgb[test_index] = model.predict_proba(x_valid)[:,1] # Use probability of class 1 (positive class).

    # 5.10) Predict probability for the test set and accumulate predictions.
    # INFER TEST
    pred_xgb += model.predict_proba(x_test)[:,1] # Use probability of class 1 (positive class).

# 5.11) After all folds are completed, average the predictions across all folds.
# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


# Cell 6: Evaluate XGBoost with ROC AUC

# 6.1) Import AUC metric from scikit-learn.
from sklearn.metrics import roc_auc_score

# 6.2) Get the true labels from the training data.
true = train.rainfall.values

# 6.3) Calculate the ROC AUC score using the true labels and the out-of-fold predictions from XGBoost.
m = roc_auc_score(true, oof_xgb)

# 6.4) Print the cross-validation AUC score, rounded to 3 decimal places.
print(f"XGBoost CV Score AUC = {m:.3f}")


# Cell 7: Visualize Feature Importance

# 7.1) Get feature importance scores from the last trained XGBoost model.
feature_importance = model.feature_importances_

# 7.2) Create a DataFrame to pair feature names with their importance scores.
importance_df = pd.DataFrame({
    "Feature": FEATURES,  
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False) # Sort by most important features.

# 7.3) Set figure size for better visibility.
plt.figure(figsize=(10, 5))

# 7.4) Create a horizontal bar chart showing feature importances.
plt.barh(importance_df["Feature"], importance_df["Importance"])

# 7.5) Add axis labels and chart title.
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")

# 7.6) Invert the Y-axis so the most important features are shown at the top.
plt.gca().invert_yaxis()  

# 7.7) Display the plot.
plt.show()


# Cell 8: Load Best Public Submission

# 8.1) Print "Best Public Notebook achieves LB = 0.915!".
print("Best Public Notebook achieves LB = 0.915!")

# 8.2) Load the submission file.
best_public = pd.read_csv("/kaggle/input/lb-915-public-notebook/submission.csv")

# 8.3) Display the first few rows of the submission file to inspect its structure.
display( best_public.head() )

# 8.4) Extract the 'rainfall' column as a NumPy array to use later for ensembling.
best_public = best_public.rainfall.values


# Cell 9: Create Rank-Based Ensemble Submission

# 9.1) Import function to convert predictions into ranks.
from scipy.stats import rankdata

# 9.2) Print "Ensemble achieves LB = 0.935! Hooray!".
print("Ensemble achieves LB = 0.935! Hooray!")

# 9.3) Load the sample submission file as a template for generating our final predictions.
sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

# 9.4) Create ensemble predictions:
# 9.4.1) Convert both prediction sets (our model and best public model) into ranks.
# 9.4.2) Apply weights: -1 for our XGBoost predictions, +2 for the public notebook's predictions.
# 9.4.3) Combine the ranks — giving more weight to the public model.
sub.rainfall = -1 * rankdata( pred_xgb ) + 2 * rankdata( best_public )

# 9.5) Normalize the resulting ranks to be between 0 and 1 by dividing by the total number of test samples.
sub.rainfall = rankdata( sub.rainfall ) / len(sub)

# 9.6) Print the shape of the submission DataFrame to confirm it matches the expected format.
print( sub.shape )

# 9.7) Save the final ensemble submission to a CSV file.
sub.to_csv(f"submission_ensemble.csv",index=False)

# 9.8) Show the first few rows of the submission.
sub.head()

