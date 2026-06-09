# --- 2. Environment Setup & Library Imports ---

# This command installs the Optuna library, which we will use for hyperparameter tuning.
# The '-q' flag ensures a quiet installation with less output.
!pip install -q optuna

# Here, we import all the tools (libraries) we'll need for this project.

# NumPy is a fundamental tool for working with numbers and arrays in Python.
import numpy as np
# Pandas is used to work with data in tables (we call them DataFrames). It makes cleaning and organizing data easy.
import pandas as pd

# These libraries are for creating charts and graphs to visualize our data.
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Scikit-learn is a massive library with tons of machine learning tools. We'll use it for:
from sklearn.model_selection import StratifiedKFold, train_test_split # Splitting our data for training and testing.
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder # Preparing our data for the model.
from sklearn.compose import ColumnTransformer # Applying different preprocessing steps to different columns.

# LightGBM is the powerful machine learning model we will use.
import lightgbm as lgb

# These are functions to measure how well our model is doing.
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

# Optuna is the library we'll use for finding the best model settings automatically.
import optuna

# This line just tells the notebook to ignore any harmless warning messages.
import warnings
warnings.filterwarnings('ignore')

# Confirmation message to let us know that all tools have been successfully loaded.
print("Libraries loaded successfully. Let's get started!")


# --- Load Data ---
# We use a try-except block to make our code flexible. It tries to load data from a local path first.
# If that fails (like in a Kaggle environment), it loads from the standard Kaggle input directory.
try:
    # This reads the 'train.csv' file into a pandas DataFrame called 'train_df'.
    train_df = pd.read_csv("train.csv")
    # This does the same for the 'test.csv' file.
    test_df = pd.read_csv("test.csv")
    # This loads the example submission file, so we know what our final output should look like.
    sample_submission = pd.read_csv("sample_submission.csv")
except FileNotFoundError:
    # These are the standard paths for data in Kaggle competitions.
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
    sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

# --- Initial Data Inspection ---
# Print the dimensions (rows, columns) of the training data.
print("Training Data Shape:", train_df.shape)
# Print the dimensions of the test data.
print("Test Data Shape:", test_df.shape)
# Print a header for the next section.
print("\nTraining Data Info:")
# The .info() method gives us a summary of the DataFrame: column names, data types, and non-null counts.
train_df.info()

# Print another title.
print("\nHere's a sneak peek at the first 5 rows of our training data:")
# The .head() method displays the first few rows of the DataFrame.
display(train_df.head())

# --- Target Variable Distribution ---
# Set the size of the plot we are about to create.
plt.figure(figsize=(8, 6))
# Use the seaborn library to create a count plot of our target variable 'y'.
ax = sns.countplot(x='y', data=train_df, palette='viridis')
# Add a title to our plot.
plt.title('Distribution of Term Deposit Subscriptions (Target: y)', fontsize=16, weight='bold')
# Label the x-axis.
plt.xlabel('Subscribed (1) vs. Not Subscribed (0)')
# Label the y-axis.
plt.ylabel('Client Count')
# This loop goes through each bar in the plot and adds a text label showing the count and percentage.
for p in ax.patches:
    # This formats the text label.
    ax.annotate(f'{p.get_height()} ({p.get_height()/len(train_df)*100:.2f}%)', 
                # This sets the position of the label.
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                # These parameters center the text and add a small offset.
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 10), 
                textcoords='offset points')
# This command displays the plot.
plt.show()

# Print out our key finding from this chart.
print("EDA Insight: The dataset is imbalanced. Only a small percentage of clients subscribed. This is important for model evaluation and potentially for training (e.g., using class weights).")

# --- Numerical Feature Analysis ---
# Get a list of all columns that contain numerical data.
numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
# Remove the 'id' and 'y' columns as they are not predictive features we want to analyze this way.
numerical_features.remove('id')
numerical_features.remove('y')

# Create histograms for all numerical features to see their distributions.
train_df[numerical_features].hist(bins=30, figsize=(15, 10), layout=(3, 3))
# Add a main title to the collection of histograms.
plt.suptitle("Distribution of Numerical Features", fontsize=20, weight='bold')
# Adjust the layout to prevent titles from overlapping.
plt.tight_layout(rect=[0, 0, 1, 0.97])
# Show the plots.
plt.show()

# --- Categorical Feature Analysis ---
# Get a list of all columns that contain object (text) data.
categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

# Create an interactive sunburst chart using Plotly Express.
# This chart helps visualize hierarchical relationships between categorical features and the target.
fig = px.sunburst(train_df, path=['job', 'marital', 'education'], 
                  color='y', # Color segments based on the target variable 'y'.
                  color_continuous_scale='ylorrd', # Choose a color scale.
                  title='Sunburst Chart of Subscriptions by Job, Marital Status, and Education')
# Show the interactive chart.
fig.show()

# --- Correlation Analysis ---
# To calculate correlation, all features must be numerical. We create a copy of the DataFrame.
corr_df = train_df.copy()
# We use LabelEncoder to convert each categorical string into a number.
for col in categorical_features:
    corr_df[col] = LabelEncoder().fit_transform(corr_df[col])

# Create a figure for the heatmap.
plt.figure(figsize=(12, 10))
# Use seaborn's heatmap to visualize the correlation matrix.
# 'cmap' sets the color map. 'annot=False' means we won't write the correlation values on the map (too crowded).
sns.heatmap(corr_df.drop('id', axis=1).corr(), cmap='coolwarm', annot=False)
# Set the title for the heatmap.
plt.title('Feature Correlation Matrix', fontsize=16)
# Display the heatmap.
plt.show()

# Print our key finding about the 'duration' feature.
print("EDA Insight: The `duration` of the last contact seems to be highly correlated with the target `y`. As stated in the original dataset's description, this feature should be handled with care, as it is only known *after* the call is made. For a realistic predictive model, it should be dropped.")


# --- Preprocessing and Feature Engineering (CORRECTED) ---

# Step 1: Drop unnecessary columns from the original dataframes
train_proc = train_df.drop(['id', 'duration'], axis=1)
test_proc = test_df.drop(['id', 'duration'], axis=1)

# Step 2: Separate the target variable
y = train_proc.pop('y')

# Step 3: Combine train and test sets for consistent feature engineering
combined_df = pd.concat([train_proc, test_proc], ignore_index=True)

# Step 4: Perform all feature engineering on the combined dataframe
# This ensures new columns exist in both train and test portions
combined_df['contact_ratio'] = combined_df['campaign'] / (combined_df['previous'] + 1)
combined_df['age_balance_interaction'] = combined_df['age'] * combined_df['balance']

# Step 5: NOW, after creating all columns, identify the feature types
# These lists will now correctly include 'contact_ratio' and 'age_balance_interaction'
categorical_features = combined_df.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_features = combined_df.select_dtypes(include=np.number).columns.tolist()

# Step 6: Create the preprocessing pipeline
# The lists passed to the ColumnTransformer now match the columns in the data it will receive
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

# Step 7: Apply the preprocessing pipeline to the combined data
# The preprocessor will be fitted on the entire combined_df and will transform it.
X_full_processed = preprocessor.fit_transform(combined_df)

# Step 8: Split the processed data back into training and test sets
# The first part of the processed data corresponds to the original training data.
X = X_full_processed[:len(train_proc)]
# The second part corresponds to the original test data.
X_test = X_full_processed[len(train_proc):]

# Print a confirmation message and the new shapes of our data.
print("Preprocessing and feature engineering complete.")
print(f"Final training data shape: {X.shape}")
print(f"Final test data shape: {X_test.shape}")


# This is the objective function that Optuna will try to maximize.
def objective(trial):
    """
    This function takes a 'trial' object from Optuna, which suggests hyperparameters.
    It then trains a LightGBM model with these settings and returns its performance score.
    """
    # Here we define the "search space" for each hyperparameter we want to tune.
    # Optuna will intelligently pick values from these ranges.
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 1000, # We'll use a fixed high number and let early stopping find the best number of trees.
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        # trial.suggest_float suggests a floating-point number in a given range. 'log=True' is good for learning rates.
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        # trial.suggest_int suggests an integer in a given range.
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True), # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), # L2 regularization
    }
    
    # We use a simple split of the data for this quick tuning process.
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create the LightGBM model with the suggested parameters.
    model = lgb.LGBMClassifier(**params)
    # Train the model. We use 'early_stopping' so it stops training if performance on the validation set doesn't improve.
    model.fit(X_train_opt, y_train_opt,
              eval_set=[(X_val_opt, y_val_opt)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    # Make predictions on the validation data.
    preds = model.predict_proba(X_val_opt)[:, 1]
    # Calculate the AUC score.
    auc = roc_auc_score(y_val_opt, preds)
    # Return the score to Optuna.
    return auc

# Create a 'study' object to manage the optimization. 'direction='maximize'' tells Optuna our goal is to get the highest score.
study = optuna.create_study(direction='maximize', study_name='LGBM Optimization')
# Start the optimization process. 'n_trials=25' tells Optuna to run the objective function 25 times.
# More trials can lead to better results but will take longer.
study.optimize(objective, n_trials=25)

# After the search is done, retrieve the best set of hyperparameters.
best_params = study.best_params
# Add back some fixed parameters needed for our final model.
best_params['objective'] = 'binary'
best_params['metric'] = 'auc'
best_params['random_state'] = 42
best_params['n_jobs'] = -1
best_params['verbose'] = -1

# Print the winning hyperparameter combination.
print("\nBest Hyperparameters found by Optuna:")
print(best_params)


# --- Cross-Validation Training ---

# We set the number of folds (splits) to 10.
NFOLDS = 10
# StratifiedKFold ensures that each fold maintains the same class balance as the original dataset.
# This is very important for imbalanced datasets like ours.
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Create empty numpy arrays to store our predictions.
# 'oof' (out-of-fold) predictions are the validation predictions for the entire training set.
oof_preds = np.zeros(X.shape[0])
# 'test_preds' will store the sum of predictions for the test set from each of the 10 models.
test_preds = np.zeros(X_test.shape[0])
# This DataFrame will store how important each feature was in each fold.
feature_importances = pd.DataFrame(index=preprocessor.get_feature_names_out())

# This is the main training loop that goes through each of the 10 folds.
# 'enumerate' gives us both the fold number ('fold') and the indices for training/validation sets.
for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    # Print which fold we are currently on.
    print(f"========== FOLD {fold + 1}/{NFOLDS} ==========")
    
    # Split the data into training (X_train, y_train) and validation (X_val, y_val) sets for this specific fold.
    X_train, y_train = X[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X[val_idx], y.iloc[val_idx]
    
    # Initialize the LightGBM model with the best parameters we found using Optuna.
    # We set a high 'n_estimators' because early stopping will find the optimal number of trees automatically.
    model = lgb.LGBMClassifier(**best_params, n_estimators=10000)
    
    # Train the model.
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], # Provide the validation set for monitoring performance.
              eval_metric='auc', # The metric to monitor for early stopping.
              callbacks=[lgb.early_stopping(200, verbose=False)]) # Stop if AUC doesn't improve for 200 rounds.
    
    # --- Make and Store Predictions ---
    # Predict probabilities on the validation set. [:, 1] gets the probability for the positive class.
    val_preds = model.predict_proba(X_val)[:, 1]
    # Store these predictions in the correct slice of our 'oof_preds' array.
    oof_preds[val_idx] = val_preds
    # Predict on the test set and add the predictions to our running total. We'll average them later.
    test_preds += model.predict_proba(X_test)[:, 1] / NFOLDS
    
    # Store the feature importances from this fold's model.
    feature_importances[f'fold_{fold+1}'] = model.feature_importances_

# --- Evaluate our overall performance ---
# Calculate the final AUC score using all the out-of-fold predictions.
oof_auc = roc_auc_score(y, oof_preds)
# Print the final, reliable AUC score.
print(f"\nOverall Out-of-Fold (OOF) ROC AUC Score: {oof_auc:.5f}")

# --- Visualize Feature Importances ---
# Calculate the average importance for each feature across all 10 folds.
feature_importances['mean'] = feature_importances.mean(axis=1)
# Sort the features by their average importance.
feature_importances = feature_importances.sort_values('mean', ascending=False)

# Create a bar plot to show the top 30 most important features.
plt.figure(figsize=(12, 12))
sns.barplot(x='mean', y=feature_importances.index[:30], data=feature_importances.head(30), palette='inferno')
plt.title('Top 30 Feature Importances (Averaged over Folds)', fontsize=16)
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.show()


# --- Create Submission File ---

# Create a new pandas DataFrame for our submission.
# The first column is 'id', which we take from the original test file.
# The second column is 'y', which contains our final, averaged test predictions.
submission = pd.DataFrame({'id': test_df['id'], 'y': test_preds})

# Save the DataFrame to a CSV file named 'submission.csv'.
# 'index=False' is important to prevent pandas from writing the row numbers into the file.
submission.to_csv('submission.csv', index=False)

# Print a confirmation message.
print("Submission file 'submission.csv' created successfully!")
# Display the first few rows of our submission file to make sure it looks correct.
display(submission.head())

# --- Plot the distribution of our final predictions ---
# This is a good sanity check to see if the distribution of our predictions makes sense.
plt.figure(figsize=(10, 6))
# Create a histogram of our predicted probabilities.
sns.histplot(submission['y'], kde=True, bins=50, color='g')
# Add a title.
plt.title('Distribution of Final Test Set Predictions', fontsize=16)
# Add an x-axis label.
plt.xlabel('Predicted Probability of Subscription')
# Add a y-axis label.
plt.ylabel('Count')
# Show the plot.
plt.show()

