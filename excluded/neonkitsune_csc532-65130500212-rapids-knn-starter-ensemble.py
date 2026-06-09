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

# 2.3) Use .head() to preview the first few rows.
test.head()


# Cell 3: Feature Selection

# 3.1) Identifies and removes non-feature columns:
# - 'rainfall': This is the target variable we’re trying to predict.
# - 'id': This is a unique identifier and doesn’t carry useful predictive information.
RMV = ['rainfall','id']

# 3.2) This list comprehension filters out rainfall and id from the training columns, keeping only the actual features used for model input.
FEATURES = [c for c in train.columns if not c in RMV]

# 3.3) Print the names of selected features to verify what will be used by the model.
print("Our features are:")
print( FEATURES )


# Cell 4: Import K-Fold and RAPIDS KNN Classifier

# 4.1) KFold from sklearn is used for cross-validation.
from sklearn.model_selection import KFold

# 4.2) KNeighborsClassifier is imported from RAPIDS cuML, which is GPU-accelerated. —-> [Note: This can significantly speed up KNN training/prediction.]
from cuml.neighbors import KNeighborsClassifier


# Cell 5: Feature Weights Dictionary

# 5.1) Manually assigns weights to each feature.
# 5.2) day is heavily weighted (24x) —-> [Note: Perhaps it’s found to be highly predictive.]
# WEIGHTS TO ADJUST IMPORTANCE OF FEATURES DURING KNN
WGT = {'day': 24, 'pressure': 1, 'maxtemp': 1, 'temparature': 1, 'mintemp': 1, 'dewpoint': 1, 'humidity': 1, 
       'cloud': 1, 'sunshine': 1, 'winddirection': 1, 'windspeed': 1}

# 5.EX) These weights will later be used to scale the features.


%%time
# Cell 6: K-Fold Cross-Validation and KNN Training
# 6.1) %%time is a Jupyter notebook cell magic command that measures and displays the execution time of the entire cell.

# 6.2) Define the number of folds for cross-validation. 5-fold CV splits the training data into 5 parts.
FOLDS = 5

# 6.3) Set up the K-Fold cross-validation object with shuffling and a fixed random state for reproducibility.
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=777)

# 6.4) Initialize arrays to store predictions:
# 6.4.1) oof_knn will hold predictions for the training set during validation.
oof_knn = np.zeros(len(train))
# 6.4.2) pred_knn will accumulate predictions for the test set, averaged across folds.
pred_knn = np.zeros(len(test))

# 6.5) Loop over each fold.
for i, (train_index, test_index) in enumerate(kf.split(train)):

    # 6.6) Print which fold is currently running.
    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)

    # 6.7) Split the training data into training and validation sets for this fold.
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"rainfall"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"rainfall"]

    # 6.8) Prepare the test data (same for all folds).
    x_test = test[FEATURES].copy()

    # 6.9) Standardize each feature using mean and standard deviation from the current training set.
    for c in FEATURES:
        m = x_train[c].mean() # mean of the feature.
        s = x_train[c].std() # standard deviation of the feature.

        # 6.10) Standardize and weight the features.
        x_train[c] = WGT[c] * (x_train[c]-m)/s
        x_valid[c] = WGT[c] * (x_valid[c]-m)/s
        x_test[c] = WGT[c] * (x_test[c]-m)/s

        # 6.11) Handle missing values by replacing them with 0.
        x_test[c] = x_test[c].fillna(0)
        x_train[c] = x_train[c].fillna(0)

    # 6.12) Initialize the K-Nearest Neighbors classifier:
    # 6.12.1) n_neighbors=201 means we look at 201 nearest neighbors for classification.
    # 6.12.2) p=1 means we use the Manhattan distance (L1 norm) instead of Euclidean distance.
    model = KNeighborsClassifier(n_neighbors=201, p=1)

    # 6.13) Train the KNN model on the current training fold.
    model.fit(x_train.values, y_train.values)

    # 6.14) Predict probabilities for the validation set (out-of-fold predictions).
    # INFER OOF
    oof_knn[test_index] = model.predict_proba(x_valid.values)[:,1]

    # 6.15) Predict probabilities for the test set and accumulate them across folds.
    # INFER TEST
    pred_knn += model.predict_proba(x_test.values)[:,1]

# 6.16) After all folds, average the test set predictions.
# COMPUTE AVERAGE TEST PREDS
pred_knn /= FOLDS# 6.1)


# Cell 7: Evaluate KNN Using AUC

# 7.1) Import the function to calculate AUC (Area Under the ROC Curve).
from sklearn.metrics import roc_auc_score

# 7.2) Get the true labels from the training data.
true = train.rainfall.values

# 7.3) Calculate AUC score using the true labels and the out-of-fold predictions from KNN.
m = roc_auc_score(true, oof_knn)

# 7.4) Print the cross-validation AUC score rounded to 3 decimal places.
print(f"KNN CV Score AUC = {m:.3f}")


# Cell 8: Load Best Public Submission for Comparison

# 8.1) Print "Best Public Notebook achieves LB = 0.954!".
print("Best Public Notebook achieves LB = 0.954!")

# 8.2) Load the CSV file that contains predictions.
best_public = pd.read_csv("/kaggle/input/lb-915-public-notebook/submission95427.csv")

# 8.3) Display the first few rows of the file to verify its structure.
display( best_public.head() )

# 8.4) Extract the 'rainfall' column as a NumPy array to use for comparison or ensembling.
best_public = best_public.rainfall.values


# Cell 9: Create Ensemble Submission

# 9.1) Import function to convert predictions into ranks.
from scipy.stats import rankdata

# 9.2) Print "Ensemble achieves LB = 0.961! Hooray!".
print("Ensemble achieves LB = 0.961! Hooray!")

# 9.3) Load the sample submission file to use as a template for our submission.
sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

# 9.4) Create ensemble predictions:
# 9.4.1) Convert both model predictions (KNN and best_public) to ranks.
# 9.4.2) Weight them: best_public gets more influence (1.25x), KNN gets less (-0.25x).
# 9.4.3) Combine them to create a new 'rainfall' prediction.
sub.rainfall = -0.25 * rankdata( pred_knn ) + 1.25 * rankdata( best_public )

# 9.5) Normalize the ranks to be between 0 and 1 by dividing by the total number of samples.
sub.rainfall = -0.25 * rankdata( pred_knn ) + 1.25 * rankdata( best_public )

# 9.6) Print the shape of the submission file to verify size (should match test set).
print( sub.shape )

# 9.7) Save the final submission file for uploading to Kaggle.
sub.to_csv(f"submission_ensemble.csv", index=False)

# 9.8) Display the first few rows of the submission.
sub.head()

