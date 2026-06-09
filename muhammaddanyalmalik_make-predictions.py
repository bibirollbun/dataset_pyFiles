# !pip install rdkit


import pandas as pd

train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

print("Train DataFrame head:")
display(train_df.head())
print("\nTest DataFrame head:")
display(test_df.head())
print("\nSample Submission DataFrame head:")
display(sample_submission_df.head())


print("Train DataFrame head:")
display(train_df.head())
print("\nTrain DataFrame Info:")
train_df.info()
print("\nTrain DataFrame Missing Values:")
display(train_df.isnull().sum())

print("\nTest DataFrame head:")
display(test_df.head())
print("\nTest DataFrame Info:")
test_df.info()
print("\nTest DataFrame Missing Values:")
display(test_df.isnull().sum())

print("\nSample Submission DataFrame head:")
display(sample_submission_df.head())
print("\nSample Submission DataFrame Info:")
sample_submission_df.info()
print("\nSample Submission DataFrame Missing Values:")
display(sample_submission_df.isnull().sum())


missing_values_train = train_df.isnull().sum()
columns_with_missing_train = missing_values_train[missing_values_train > 0].index.tolist()
print("Columns in train_df with missing values:")
print(columns_with_missing_train)


from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def smiles_to_morgan_fingerprint(smiles_string, radius=2, nbits=2048):
    """Converts a SMILES string to a Morgan fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return np.zeros(nbits, dtype=int)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits=nbits)
        return np.array(fingerprint)
    except:
        return np.zeros(nbits, dtype=int)

# Apply featurization to train_df
train_fingerprints = np.vstack(train_df['SMILES'].apply(smiles_to_morgan_fingerprint))
fingerprint_columns = [f'morgan_fp_{i}' for i in range(train_fingerprints.shape[1])]
train_fingerprint_df = pd.DataFrame(train_fingerprints, columns=fingerprint_columns, index=train_df.index)
train_df = pd.concat([train_df.drop('SMILES', axis=1), train_fingerprint_df], axis=1)

# Apply featurization to test_df
test_fingerprints = np.vstack(test_df['SMILES'].apply(smiles_to_morgan_fingerprint))
test_fingerprint_df = pd.DataFrame(test_fingerprints, columns=fingerprint_columns, index=test_df.index)
test_df = pd.concat([test_df.drop('SMILES', axis=1), test_fingerprint_df], axis=1)

print("Train DataFrame after featurization:")
display(train_df.head())
print("\nTest DataFrame after featurization:")
display(test_df.head())


# Identify feature columns (all columns except id and target variables in train_df)
feature_columns = [col for col in train_df.columns if col not in ['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]

# Ensure test_df has the same feature columns as train_df (excluding id)
# We will drop any extra columns in test_df and add any missing columns with 0
test_feature_columns = [col for col in test_df.columns if col != 'id']

# Columns to drop from test_df (if any exist that are not in train_df features)
cols_to_drop_test = [col for col in test_feature_columns if col not in feature_columns]
if cols_to_drop_test:
    test_df = test_df.drop(columns=cols_to_drop_test)

# Columns to add to test_df (if any train_df features are missing in test_df)
cols_to_add_test = [col for col in feature_columns if col not in test_feature_columns]
for col in cols_to_add_test:
    test_df[col] = 0

# Reorder test_df columns to match train_df feature column order (plus 'id')
test_df = test_df[['id'] + feature_columns]

# Display the columns of both dataframes to confirm consistency
print("Train DataFrame columns (excluding targets):")
print(train_df.drop(columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg']).columns.tolist())
print("\nTest DataFrame columns:")
print(test_df.columns.tolist())

# Verify the number of columns match (excluding the 5 target columns in train_df)
print("\nNumber of columns in train_df (excluding targets):", len(train_df.drop(columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg']).columns))
print("Number of columns in test_df:", len(test_df.columns))



# 1. Define a list of target variable column names.
target_columns = ['Tg', 'Rg', 'FFV', 'Density', 'Tc']

# 2. Create a dictionary to store the feature DataFrames (X_train) and target Series (y_train) for each target variable.
train_data = {}

# 3. Iterate through the list of target variables:
for target in target_columns:
    # For each target, create a training feature DataFrame X_train by dropping the 'id' column and all target columns from train_df.
    X_train = train_df.drop(columns=['id'] + target_columns)

    # For each target, create a training target Series y_train containing the values of the current target column from train_df.
    y_train = train_df[target]

    # Store X_train and y_train in the dictionary created in step 2, using the target variable name as the key.
    train_data[target] = {'X_train': X_train, 'y_train': y_train}

# 4. Create the test feature DataFrame X_test by dropping the 'id' column from test_df.
X_test = test_df.drop(columns=['id'])

# 5. Print the shapes of the created training feature and target objects (for the first target variable) and the test feature DataFrame to verify the separation.
first_target = target_columns[0]
print(f"Shape of X_train for {first_target}: {train_data[first_target]['X_train'].shape}")
print(f"Shape of y_train for {first_target}: {train_data[first_target]['y_train'].shape}")
print(f"Shape of X_test: {X_test.shape}")


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Choose three classification models
models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42) # Increased max_iter for convergence
}

# Dictionary to store trained models
trained_models = {}

# Iterate through each target variable
for target, data in train_data.items():
    X_train_target = data['X_train']
    y_train_target = data['y_train']

    # Filter out rows where the target variable is null
    non_null_mask = y_train_target.notnull()
    X_train_filtered = X_train_target[non_null_mask]
    y_train_filtered = y_train_target[non_null_mask]

    # Store models for the current target
    trained_models[target] = {}

    # Iterate through each chosen model
    for model_name, model in models.items():
        print(f"Training {model_name} for target: {target}")
        try:
            # Train the model
            model.fit(X_train_filtered, y_train_filtered)
            # Store the trained model
            trained_models[target][model_name] = model
            print(f"Finished training {model_name} for target: {target}")
        except Exception as e:
            print(f"Error training {model_name} for target {target}: {e}")
            # Optionally store None or the error, depending on how you want to handle failures
            trained_models[target][model_name] = None # Store None if training fails

# Print a summary of the trained models
print("\nTrained Models Summary:")
for target, models_dict in trained_models.items():
    print(f"  Target: {target}")
    for model_name, model in models_dict.items():
        status = "Trained" if model is not None else "Training Failed"
        print(f"    {model_name}: {status}")


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Choose three regression models
models_regression = {
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
    'LinearRegression': LinearRegression()
}

# Dictionary to store trained regression models
trained_models_regression = {}

# Iterate through each target variable
for target, data in train_data.items():
    X_train_target = data['X_train']
    y_train_target = data['y_train']

    # Filter out rows where the target variable is null
    non_null_mask = y_train_target.notnull()
    X_train_filtered = X_train_target[non_null_mask]
    y_train_filtered = y_train_target[non_null_mask]

    # Store models for the current target
    trained_models_regression[target] = {}

    # Iterate through each chosen regression model
    for model_name, model in models_regression.items():
        print(f"Training {model_name} for target: {target}")
        try:
            # Train the model
            model.fit(X_train_filtered, y_train_filtered)
            # Store the trained model
            trained_models_regression[target][model_name] = model
            print(f"Finished training {model_name} for target: {target}")
        except Exception as e:
            print(f"Error training {model_name} for target {target}: {e}")
            # Store None if training fails
            trained_models_regression[target][model_name] = None

# Print a summary of the trained regression models
print("\nTrained Regression Models Summary:")
for target, models_dict in trained_models_regression.items():
    print(f"  Target: {target}")
    for model_name, model in models_dict.items():
        status = "Trained" if model is not None else "Training Failed"
        print(f"    {model_name}: {status}")



from sklearn.model_selection import KFold
import numpy as np

# Dictionary to store out-of-fold predictions for each target and model
oof_preds = {}

# Number of folds for cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Iterate through each target variable
for target, data in train_data.items():
    X_train_target = data['X_train']
    y_train_target = data['y_train']

    # Filter out rows where the target variable is null
    non_null_mask = y_train_target.notnull()
    X_train_filtered = X_train_target[non_null_mask]
    y_train_filtered = y_train_target[non_null_mask]

    # Initialize a dictionary to store OOF predictions for the current target
    oof_preds[target] = {}

    # Iterate through each trained regression model for the current target
    for model_name, model in trained_models_regression[target].items():
        if model is not None: # Only generate predictions if the model was trained successfully
            print(f"Generating OOF predictions for {model_name} on target: {target}")
            # Initialize array to store OOF predictions for the current model and target
            oof_preds[target][model_name] = np.zeros(X_train_filtered.shape[0])

            # Perform cross-validation
            for fold, (train_index, val_index) in enumerate(kf.split(X_train_filtered, y_train_filtered)):
                X_train_fold, X_val_fold = X_train_filtered.iloc[train_index], X_train_filtered.iloc[val_index]
                y_train_fold, y_val_fold = y_train_filtered.iloc[train_index], y_train_filtered.iloc[val_index]

                # Train a fresh instance of the model on the training fold
                fold_model = models_regression[model_name] # Get a fresh instance from the original models_regression dict
                fold_model.fit(X_train_fold, y_train_fold)

                # Predict on the validation fold and store predictions
                oof_preds[target][model_name][val_index] = fold_model.predict(X_val_fold)
            print(f"Finished OOF predictions for {model_name} on target: {target}")
        else:
            print(f"Skipping OOF predictions for {model_name} on target {target} due to previous training failure.")

# Concatenate OOF predictions to create meta-features
meta_X_train = {}
for target in trained_models_regression.keys():
    if target in oof_preds and oof_preds[target]: # Check if OOF predictions exist for the target
        # Stack predictions from all models for the current target
        meta_features = [oof_preds[target][model_name] for model_name in oof_preds[target].keys()]
        meta_X_train[target] = np.vstack(meta_features).T # Transpose to have shape (n_samples, n_models)
        print(f"Created meta-features for target {target} with shape: {meta_X_train[target].shape}")
    else:
        print(f"Could not create meta-features for target {target} as no OOF predictions were generated.")
        meta_X_train[target] = None # Indicate that meta-features could not be created

# Choose a suitable meta-model for regression (e.g., Linear Regression)
from sklearn.linear_model import LinearRegression

# Dictionary to store trained meta-models
trained_meta_models = {}

# Train a meta-model for each target variable
for target, meta_features in meta_X_train.items():
    if meta_features is not None:
        y_train_target = train_data[target]['y_train']
        non_null_mask = y_train_target.notnull()
        y_train_filtered = y_train_target[non_null_mask]

        print(f"Training meta-model for target: {target}")
        meta_model = LinearRegression() # Using Linear Regression as the meta-model
        meta_model.fit(meta_features, y_train_filtered)
        trained_meta_models[target] = meta_model
        print(f"Finished training meta-model for target: {target}")
    else:
        print(f"Skipping meta-model training for target {target} as meta-features were not available.")
        trained_meta_models[target] = None # Indicate that meta-model training failed

# Print a summary of the trained meta-models
print("\nTrained Meta-Models Summary:")
for target, meta_model in trained_meta_models.items():
    status = "Trained" if meta_model is not None else "Training Failed"
    print(f"  Target: {target}: {status}")


import numpy as np

# 1. Create a dictionary test_meta_X to store the predictions of the individual models on the test data for each target variable.
test_meta_X = {}

# 2. Iterate through each target variable in trained_models_regression.
for target, models_dict in trained_models_regression.items():
    print(f"Generating test predictions for individual models for target: {target}")
    target_test_predictions = []

    # 3. For each target, iterate through the trained individual regression models.
    for model_name, model in models_dict.items():
        # 4. If a model was successfully trained (not None), use it to predict on the X_test DataFrame.
        if model is not None:
            try:
                predictions = model.predict(X_test)
                target_test_predictions.append(predictions)
                print(f"  Generated test predictions for {model_name} for target: {target}")
            except Exception as e:
                print(f"  Error generating test predictions for {model_name} for target {target}: {e}")
                # If prediction fails, append a placeholder (e.g., array of NaNs)
                target_test_predictions.append(np.full(X_test.shape[0], np.nan))
        else:
            print(f"  Skipping test predictions for {model_name} for target {target} as model was not trained.")
            # If model was not trained, append a placeholder
            target_test_predictions.append(np.full(X_test.shape[0], np.nan))

    # 5. After iterating through all models for a target, stack the predictions.
    if target_test_predictions:
        # Check if all predictions have the same length before stacking
        if all(len(pred) == X_test.shape[0] for pred in target_test_predictions):
             test_meta_X[target] = np.vstack(target_test_predictions).T # Transpose to have shape (n_samples, n_models)
             print(f"Created test meta-features for target {target} with shape: {test_meta_X[target].shape}")
        else:
            print(f"Could not stack test predictions for target {target} due to inconsistent prediction lengths.")
            test_meta_X[target] = None
    else:
         print(f"No test predictions generated for target {target}.")
         test_meta_X[target] = None


# 6. Create a dictionary final_predictions to store the final predictions from the meta-models on the test meta-features.
final_predictions = {}

# 7. Iterate through each target variable in trained_meta_models.
print("\nGenerating final predictions using meta-models:")
for target, meta_model in trained_meta_models.items():
    # 8. If a meta-model was successfully trained (not None) and meta-features were generated for the test data for that target.
    if meta_model is not None and target in test_meta_X and test_meta_X[target] is not None:
        try:
            # Use the meta-model to predict on the corresponding test meta-features.
            final_preds = meta_model.predict(test_meta_X[target])
            # 9. Store the final predictions for the current target in the final_predictions dictionary.
            final_predictions[target] = final_preds
            print(f"  Generated final predictions for target: {target}")
        except Exception as e:
            print(f"  Error generating final predictions for target {target}: {e}")
            final_predictions[target] = None # Store None if prediction fails
    else:
        # 10. If either the meta-model or test meta-features are not available for a target, indicate that predictions could not be made.
        print(f"  Skipping final predictions for target {target} as meta-model or test meta-features were not available.")
        final_predictions[target] = None

# 11. Print a summary of the targets for which final predictions were generated.
print("\nSummary of Final Predictions:")
predicted_targets = [target for target, preds in final_predictions.items() if preds is not None]
if predicted_targets:
    print("Final predictions generated for the following targets:")
    for target in predicted_targets:
        print(f"  - {target}")
else:
    print("No final predictions were generated for any target.")



# 1. Create a new DataFrame for the submission file, initializing it with the 'id' column from the original test_df.
submission_df = pd.DataFrame({'id': test_df['id']})

# 2. Iterate through the final_predictions dictionary. For each target and its corresponding predictions:
for target, predictions in final_predictions.items():
    # Add the predictions as a new column to the submission DataFrame, using the target name as the column name.
    # Ensure predictions are added only if they exist and have the correct length
    if predictions is not None and len(predictions) == len(test_df):
        submission_df[target] = predictions
    else:
        # Handle cases where predictions were not generated successfully by adding a column of NaNs or a default value if needed
        # For this submission format, we should probably add 0 as per sample_submission_df
        print(f"Predictions for target {target} are missing or have incorrect length. Adding 0s.")
        submission_df[target] = 0

# 3. Ensure the columns in the submission DataFrame are in the same order as the columns in sample_submission_df.
# Get the column order from sample_submission_df
submission_columns_order = sample_submission_df.columns.tolist()

# Reindex the submission_df to match this order
# Use .reindex to handle potential missing columns in either dataframe (though in this case, we expect them to match)
submission_df = submission_df.reindex(columns=submission_columns_order, fill_value=0)


# 4. Save the submission DataFrame to a CSV file named submission.csv, with index=False.
submission_df.to_csv('/kaggle/working/submission.csv', index=False)

# 5. Display the head of the generated submission DataFrame to verify its structure.
print("Generated Submission DataFrame head:")
display(submission_df.head())

